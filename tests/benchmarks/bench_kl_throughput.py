from __future__ import annotations

import argparse
import math

from tests.benchmarks.common import (
    CANONICAL_T4_CONFIGS,
    add_output_args,
    bench_stats_fields,
    benchmark_row,
    collect_metadata,
    compile_model,
    compile_pytorch_loss,
    require_cuda_kernel_or_skip,
    validate_positive_timing_args,
    warn_if_noisy,
    write_artifacts,
)
from tests.utils.env import parse_configs, set_seed
from tests.utils.timing import cuda_benchmark, cuda_cleanup, format_result, is_oom_exception, oom_result


def main() -> None:
    parser = argparse.ArgumentParser(description="KL strategy throughput benchmark")
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=parse_configs(CANONICAL_T4_CONFIGS["kl_throughput"]),
    )
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--student-layers", type=int, default=4)
    parser.add_argument("--teacher-layers", type=int, default=12)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=0.4)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--lambda-student", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile on student/teacher models AND on PyTorch loss path.")
    add_output_args(parser)
    args = parser.parse_args()
    validate_positive_timing_args(args)

    runtime = require_cuda_kernel_or_skip("bench_kl_throughput", args)
    torch = runtime.torch
    nn = torch.nn
    F = torch.nn.functional
    from orda_ce_kernel.utils.dispatcher import dynamic_chunk
    from orda_ce_kernel.utils.resolver import resolve_chunk_size

    loss_compile_enabled = not args.no_compile

    class Model(nn.Module):
        def __init__(self, vocab: int, dim: int, layers: int):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=8,
                    dim_feedforward=dim * 4,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ])
            self.head = nn.Linear(dim, vocab, bias=False)

        def forward(self, x):
            h = self.embed(x)
            for layer in self.layers:
                h = layer(h)
            return h

    def build_models():
        student = Model(args.vocab_size, args.hidden_dim, args.student_layers).cuda()
        teacher = Model(args.vocab_size, args.hidden_dim, args.teacher_layers).cuda()
        student.head.weight = teacher.head.weight
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        student.head.weight.requires_grad = True
        opt = torch.optim.AdamW(student.parameters(), lr=3e-4, betas=(0.9, 0.95))
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        head_weight = student.head.weight
        student = compile_model(
            torch, student, "student", enabled=loss_compile_enabled, is_oom_exception=is_oom_exception,
        )
        teacher = compile_model(
            torch, teacher, "teacher", enabled=loss_compile_enabled, is_oom_exception=is_oom_exception,
        )
        return student, teacher, head_weight, opt, scaler

    # PyTorch-eager loss helpers. Each helper is wrapped with torch.compile
    # when compile is enabled — this is the loss-level compile that materially
    # reduces baseline VRAM and matches how a careful PyTorch user would code.
    def _kl_on_slice_eager(hs, ht, w, n_total: int, T: float):
        log_p_s = F.log_softmax(F.linear(hs, w) / T, dim=-1)
        p_t = F.softmax(F.linear(ht.detach(), w.detach()) / T, dim=-1)
        frac = hs.shape[0] / n_total
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T) * frac

    def _kl_sample_eager(hs, ht, w, T: float):
        log_p_s = F.log_softmax(F.linear(hs, w) / T, dim=-1)
        p_t = F.softmax(F.linear(ht.detach(), w.detach()) / T, dim=-1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

    def _kl_fp16_f32_row_sum_eager(hs, ht, w, n_total: int, T: float):
        logits_s = F.linear(hs, w)
        logits_t = F.linear(ht.detach(), w.detach())
        log_p_s = F.log_softmax(logits_s.float() / T, dim=-1)
        with torch.no_grad():
            p_t = F.softmax(logits_t.float() / T, dim=-1)
        kl_rows = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
        return kl_rows.sum() * (T * T) / max(n_total, 1)

    def _kl_chunk_eager(hs_chunk, ht_chunk, w, n_total: int, T: float):
        """Single chunk of chunked_pytorch_kl: fp32 row-sum, no full materialization."""
        logits_s = F.linear(hs_chunk, w)
        logits_t = F.linear(ht_chunk.detach(), w.detach())
        log_p_s = F.log_softmax(logits_s.float() / T, dim=-1)
        with torch.no_grad():
            p_t = F.softmax(logits_t.float() / T, dim=-1)
        kl_rows = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
        return kl_rows.sum() * (T * T) / max(n_total, 1)

    kl_on_slice = compile_pytorch_loss(torch, _kl_on_slice_eager, enabled=loss_compile_enabled)
    kl_sample = compile_pytorch_loss(torch, _kl_sample_eager, enabled=loss_compile_enabled)
    kl_fp16_f32_row_sum = compile_pytorch_loss(torch, _kl_fp16_f32_row_sum_eager, enabled=loss_compile_enabled)
    # `chunked_pytorch_kl` compiles the inner chunk only; the outer Python loop
    # stays eager to avoid graph-break recompiles per chunk_size value.
    kl_chunk_compiled = compile_pytorch_loss(torch, _kl_chunk_eager, enabled=loss_compile_enabled)

    def chunked_pytorch_kl(hs_full, ht_full, w, T: float):
        """fp32 row-sum KL chunked with the same policy as Orda's dynamic_chunk."""
        BT, V = hs_full.shape[0], w.shape[0]
        chunk_size, num_chunks = resolve_chunk_size(BT, "dynamic", V=V)
        total = hs_full.new_zeros((), dtype=torch.float32)
        for i in range(num_chunks):
            lo = i * chunk_size
            hi = min(lo + chunk_size, BT)
            if lo >= hi:
                break
            total = total + kl_chunk_compiled(
                hs_full[lo:hi], ht_full[lo:hi], w, BT, T,
            )
        return total

    print("=" * 116)
    print(
        f"KL throughput benchmark | repeats={args.repeats} steps={args.steps} "
        f"(samples/variant={args.repeats * args.steps}) loss_compile={'on' if loss_compile_enabled else 'off'}"
    )
    print("=" * 116)
    print(
        f"{'Config':<12} | {'Variant':<20} | {'Latency ms':>12} | {'cv %':>6} | "
        f"{'vs no_kl':>10} | {'Peak VRAM MiB':>14} | {'VRAM delta vs no_kl':>20}"
    )
    print("-" * 116)
    rows: list[dict] = []

    # `vs_no_kl` needs the no_kl baseline latency/vram before comparator rows print.
    # kl_triton is now treated as a comparator (not a separate reference axis),
    # so it reports vs_no_kl like every other variant.
    BASELINE_VARIANTS = ("no_kl",)
    variants = [
        "no_kl",
        "kl_triton",
        "fp16_f32_row_sum",
        "kl_full",
        "chunked_pytorch_kl",
        "kl_sample_25",
    ]

    for batch, seq in args.configs:
        set_seed(torch, args.seed)
        x = torch.randint(0, args.vocab_size, (args.grad_accum, batch, seq), device="cuda")
        y = torch.randint(0, args.vocab_size, (args.grad_accum, batch, seq), device="cuda")
        baseline = float("nan")
        baseline_vram = float("nan")

        missing_baselines = [v for v in BASELINE_VARIANTS if v not in variants]
        if missing_baselines:
            raise ValueError(
                f"variants list is missing required baselines {missing_baselines}; "
                "ratio vs_no_kl would be NaN. Re-add them to `variants` before "
                "introducing a --variants CLI."
            )

        # Reorder so no_kl always runs first to populate baseline before comparators.
        ordered_variants = (
            [v for v in BASELINE_VARIANTS if v in variants]
            + [v for v in variants if v not in BASELINE_VARIANTS]
        )

        for variant in ordered_variants:
            cuda_cleanup(torch)
            student = teacher = head_weight = opt = scaler = None

            try:
                student, teacher, head_weight, opt, scaler = build_models()

                def step():
                    opt.zero_grad(set_to_none=True)
                    for i in range(args.grad_accum):
                        with torch.autocast("cuda", dtype=torch.float16):
                            hs = student(x[i])
                            with torch.no_grad():
                                ht = teacher(x[i])
                            hs_f = hs.reshape(-1, hs.shape[-1])
                            ht_f = ht.reshape(-1, ht.shape[-1])
                            target = y[i].reshape(-1)
                            if variant == "kl_triton":
                                loss, *_ = dynamic_chunk(
                                    hs_f,
                                    ht_f,
                                    head_weight,
                                    target,
                                    lambda_student=args.lambda_student,
                                    kl_weight=args.kl_weight,
                                    kl_temperature=args.kl_temperature,
                                )
                            else:
                                ce, *_ = dynamic_chunk(
                                    hs_f,
                                    ht_f,
                                    head_weight,
                                    target,
                                    lambda_student=args.lambda_student,
                                    kl_weight=0.0,
                                    kl_temperature=args.kl_temperature,
                                )
                                if variant == "no_kl":
                                    loss = ce
                                elif variant == "kl_sample_25":
                                    n = hs_f.shape[0]
                                    idx = torch.randperm(n, device="cuda")[: max(n // 4, 1)]
                                    loss = ce + args.kl_weight * kl_sample(
                                        hs_f[idx], ht_f[idx], head_weight, args.kl_temperature,
                                    )
                                elif variant == "fp16_f32_row_sum":
                                    loss = ce + args.kl_weight * kl_fp16_f32_row_sum(
                                        hs_f, ht_f, head_weight, hs_f.shape[0], args.kl_temperature,
                                    )
                                elif variant == "kl_full":
                                    loss = ce + args.kl_weight * kl_on_slice(
                                        hs_f, ht_f, head_weight, hs_f.shape[0], args.kl_temperature,
                                    )
                                elif variant == "chunked_pytorch_kl":
                                    loss = ce + args.kl_weight * chunked_pytorch_kl(
                                        hs_f, ht_f, head_weight, args.kl_temperature,
                                    )
                                else:
                                    raise AssertionError(variant)
                            scaler.scale(loss / args.grad_accum).backward()
                    scaler.step(opt)
                    scaler.update()

                result = cuda_benchmark(
                    torch,
                    step,
                    warmup=args.warmup,
                    steps=args.steps,
                    repeats=args.repeats,
                    cleanup_between_repeats=True,
                    seed=args.seed,
                )
            except Exception as exc:
                if not is_oom_exception(torch, exc):
                    raise
                result = oom_result()

            if variant == "no_kl":
                baseline = result.latency_ms
                baseline_vram = result.peak_vram_mib
            warn_if_noisy(result, variant)
            latency, vram = format_result(result)
            cv_str = "N/A" if math.isnan(result.cv_pct) else f"{result.cv_pct:.1f}"
            vs_no_kl = None if math.isnan(result.latency_ms) or math.isnan(baseline) else result.latency_ms / baseline
            if math.isnan(result.peak_vram_mib) or math.isnan(baseline_vram):
                vram_delta = None
            else:
                vram_delta = result.peak_vram_mib - baseline_vram
            ratio = "N/A" if vs_no_kl is None else f"{vs_no_kl:.2f}x"
            vram_delta_str = "N/A" if vram_delta is None else f"{vram_delta:.1f}"
            print(
                f"{batch}x{seq:<7} | {variant:<20} | {latency:>12} | {cv_str:>6} | "
                f"{ratio:>10} | {vram:>14} | {vram_delta_str:>20}"
            )
            rows.append(
                benchmark_row(
                    config=f"{batch}x{seq}",
                    method=variant,
                    latency_ms=result.latency_ms,
                    peak_vram_mib=result.peak_vram_mib,
                    batch=batch,
                    seq=seq,
                    bt=batch * seq,
                    vocab_size=args.vocab_size,
                    hidden_dim=args.hidden_dim,
                    warmup=args.warmup,
                    steps=args.steps,
                    kl_weight=args.kl_weight,
                    kl_temperature=args.kl_temperature,
                    lambda_student=args.lambda_student,
                    student_layers=args.student_layers,
                    teacher_layers=args.teacher_layers,
                    grad_accum=args.grad_accum,
                    loss_compile=loss_compile_enabled,
                    vs_no_kl=vs_no_kl,
                    peak_vram_delta_vs_no_kl_mib=vram_delta,
                    status="ok" if result.ok else "oom",
                    **bench_stats_fields(result),
                )
            )
            del student, teacher, head_weight, opt, scaler

    print("=" * 116)
    write_artifacts(
        rows,
        collect_metadata(torch, args, "bench_kl_throughput"),
        output_json=args.output_json,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()


