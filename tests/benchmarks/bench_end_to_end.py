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
from tests.utils.env import set_seed
from tests.utils.timing import cuda_benchmark, cuda_cleanup, format_result, is_oom_exception, oom_result


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end student/teacher training benchmark")
    parser.add_argument("--mode", choices=["compare", "orda-large"], default="compare",
                        help=("`compare` runs 8x1024 with PyTorch + Orda variants. "
                              "`orda-large` runs 16x1024 with Orda variants only."))
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override default mode batch size.")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override default mode seq len.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile on student/teacher models AND on PyTorch loss path.")
    add_output_args(parser)
    args = parser.parse_args()

    if args.batch_size is None or args.seq_len is None:
        if args.mode == "compare":
            default_b, default_s = CANONICAL_T4_CONFIGS["end_to_end_compare"]
        else:
            default_b, default_s = CANONICAL_T4_CONFIGS["end_to_end_orda_large"]
        if args.batch_size is None:
            args.batch_size = default_b
        if args.seq_len is None:
            args.seq_len = default_s

    validate_positive_timing_args(args)

    runtime = require_cuda_kernel_or_skip("bench_end_to_end", args)
    torch = runtime.torch
    nn = torch.nn
    F = torch.nn.functional
    from orda_ce_kernel.ops.cross_entropy import distill_cross_entropy

    loss_compile_enabled = not args.no_compile

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class Block(nn.Module):
        def __init__(self, dim: int, heads: int):
            super().__init__()
            self.heads = heads
            self.norm1 = RMSNorm(dim)
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.out = nn.Linear(dim, dim, bias=False)
            self.norm2 = RMSNorm(dim)
            mlp_dim = ((int(8 * dim / 3) + 255) // 256) * 256
            self.w1 = nn.Linear(dim, mlp_dim, bias=False)
            self.w2 = nn.Linear(mlp_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, mlp_dim, bias=False)

        def forward(self, x):
            b, s, d = x.shape
            h = self.norm1(x)
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            head_dim = d // self.heads
            q = q.view(b, s, self.heads, head_dim).transpose(1, 2)
            k = k.view(b, s, self.heads, head_dim).transpose(1, 2)
            v = v.view(b, s, self.heads, head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.out(attn.transpose(1, 2).contiguous().view(b, s, d))
            h = self.norm2(x)
            return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(args.vocab_size, args.hidden_dim)
            self.layers = nn.ModuleList([Block(args.hidden_dim, args.heads) for _ in range(args.layers)])
            self.norm = RMSNorm(args.hidden_dim)
            self.head = nn.Linear(args.hidden_dim, args.vocab_size, bias=False)

        def forward(self, x):
            h = self.embed(x)
            for layer in self.layers:
                h = layer(h)
            return self.norm(h)

    def make_pair():
        student = Model().cuda()
        teacher = Model().cuda()
        student.embed.weight = teacher.embed.weight
        student.head.weight = teacher.head.weight
        teacher.eval()
        for name, p in teacher.named_parameters():
            if name not in {"embed.weight", "head.weight"}:
                p.requires_grad = False
        student.embed.weight.requires_grad = True
        student.head.weight.requires_grad = True
        head_weight = student.head.weight
        opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
        student = compile_model(
            torch, student, "student", enabled=loss_compile_enabled, is_oom_exception=is_oom_exception,
        )
        teacher = compile_model(
            torch, teacher, "teacher", enabled=loss_compile_enabled, is_oom_exception=is_oom_exception,
        )
        return student, teacher, head_weight, opt

    # Loss-level compile for the PyTorch baseline. With both model AND loss
    # compiled, the PyTorch baseline avoids a graph break around the linear+CE+KL
    # path. Triton variants (orda_ce, orda_ce_kl) keep their custom op eager —
    # torch.compile wrapping a Triton autograd.Function regresses on T4.
    def _pytorch_ce_kl_eager(hs, ht, head_weight, target, T: float, kl_w: float, V: int):
        logits_s = F.linear(hs, head_weight)
        logits_t = F.linear(ht, head_weight)
        ce = F.cross_entropy(logits_s.view(-1, V), target)
        ce = ce + F.cross_entropy(logits_t.view(-1, V), target)
        log_p_s = F.log_softmax(logits_s.view(-1, V) / T, dim=-1)
        p_t = F.softmax((logits_t.view(-1, V) / T).detach(), dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        return ce + kl_w * kl

    pytorch_ce_kl = compile_pytorch_loss(torch, _pytorch_ce_kl_eager, enabled=loss_compile_enabled)

    def run_variant(name: str):
        student, teacher, head_weight, opt = make_pair()

        # Track VRAM at two points separately:
        # - loss_peak: max_memory_allocated measured RIGHT AFTER the loss call
        #   (before backward). This captures the loss-kernel working set, which
        #   differs between variants (KL allocates grad_kl_student per chunk).
        # - step_peak: max across the whole step (dominated by student backward).
        #   This is typically the same for all Orda variants because backward
        #   activations dwarf the loss working set.
        loss_peaks: list[float] = []
        _loss_peak_capture = {"v": 0.0}

        def step():
            opt.zero_grad(set_to_none=True)
            hs = student(x)
            with torch.no_grad():
                ht = teacher(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if name == "pytorch":
                loss = pytorch_ce_kl(
                    hs, ht, head_weight, target.view(-1),
                    args.kl_temperature, args.kl_weight, args.vocab_size,
                )
            else:
                loss, *_ = distill_cross_entropy(
                    hs.view(-1, args.hidden_dim),
                    ht.view(-1, args.hidden_dim),
                    head_weight,
                    target.view(-1),
                    kl_weight=args.kl_weight if name == "orda_ce_kl" else 0.0,
                    kl_temperature=args.kl_temperature,
                )
            torch.cuda.synchronize()
            # Capture peak immediately after loss call, before backward frees nothing
            # but adds activation-grad pressure.
            _loss_peak_capture["v"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            loss_peaks.append(_loss_peak_capture["v"])
            loss.backward()
            opt.step()

        result = cuda_benchmark(
            torch,
            step,
            warmup=args.warmup,
            steps=args.steps,
            repeats=args.repeats,
            cleanup_between_repeats=True,
            seed=args.seed,
        )
        tokens_per_step = args.batch_size * args.seq_len
        tps = tokens_per_step / (result.latency_ms / 1000.0) if result.ok else float("nan")
        # Average loss-only peak over timed steps (excluding warmup captures).
        avg_loss_peak = sum(loss_peaks[-args.steps:]) / max(len(loss_peaks[-args.steps:]), 1)
        del student, teacher, head_weight, opt
        return result, tps, avg_loss_peak

    set_seed(torch, args.seed)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device="cuda")
    target = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device="cuda")

    if args.mode == "compare":
        variants = ["pytorch", "orda_ce", "orda_ce_kl"]
    else:
        variants = ["orda_ce", "orda_ce_kl"]

    print("=" * 128)
    print(
        f"End-to-end training benchmark | mode={args.mode} cfg={args.batch_size}x{args.seq_len} "
        f"repeats={args.repeats} steps={args.steps} loss_compile={'on' if loss_compile_enabled else 'off'}"
    )
    print("=" * 128)
    print(
        f"{'Variant':<16} | {'Avg step ms':>12} | {'cv %':>6} | "
        f"{'p95 ms':>10} | {'Throughput tok/s':>18} | {'Step VRAM MiB':>14} | {'Loss VRAM MiB':>14}"
    )
    print(f"{'':16}   {'':12}   {'':6}   {'':10}   {'':18}   {'(full step)':>14}   {'(loss only)':>14}")
    print("-" * 128)
    rows: list[dict] = []
    for variant in variants:
        cuda_cleanup(torch)
        try:
            result, tps, loss_peak = run_variant(variant)
        except Exception as exc:
            if not is_oom_exception(torch, exc):
                raise
            print(f"{variant:<16} | {'OOM':>12} | {'N/A':>6} | {'N/A':>10} | {'N/A':>18} | {'N/A':>14} | {'N/A':>14}")
            rows.append(
                benchmark_row(
                    config=f"{args.batch_size}x{args.seq_len}",
                    method=variant,
                    latency_ms=float("nan"),
                    peak_vram_mib=float("nan"),
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    vocab_size=args.vocab_size,
                    hidden_dim=args.hidden_dim,
                    layers=args.layers,
                    heads=args.heads,
                    mode=args.mode,
                    loss_compile=loss_compile_enabled,
                    status="oom",
                )
            )
            continue

        warn_if_noisy(result, variant)
        latency, vram = format_result(result)
        cv_str = "N/A" if math.isnan(result.cv_pct) else f"{result.cv_pct:.1f}"
        p95_str = "N/A" if math.isnan(result.latency_ms_p95) else f"{result.latency_ms_p95:.2f}"
        loss_peak_str = f"{loss_peak:.1f}"
        print(
            f"{variant:<16} | {latency:>12} | {cv_str:>6} | {p95_str:>10} | "
            f"{tps:>18.2f} | {vram:>14} | {loss_peak_str:>14}"
        )
        rows.append(
            benchmark_row(
                config=f"{args.batch_size}x{args.seq_len}",
                method=variant,
                latency_ms=result.latency_ms,
                peak_vram_mib=result.peak_vram_mib,
                peak_vram_loss_only_mib=loss_peak,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                tokens_per_second=tps,
                vocab_size=args.vocab_size,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                heads=args.heads,
                mode=args.mode,
                loss_compile=loss_compile_enabled,
                status="ok",
                **bench_stats_fields(result),
            )
        )
    print("=" * 128)
    write_artifacts(
        rows,
        collect_metadata(torch, args, "bench_end_to_end"),
        output_json=args.output_json,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()


