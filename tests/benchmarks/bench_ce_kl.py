from __future__ import annotations

import argparse
import math

from tests.benchmarks.common import (
    CANONICAL_T4_CONFIGS,
    add_common_benchmark_args,
    bench_stats_fields,
    benchmark_row,
    collect_metadata,
    compile_pytorch_loss,
    print_table_header,
    print_table_row,
    require_cuda_kernel_or_skip,
    resolve_dtype,
    validate_positive_timing_args,
    warn_if_noisy,
    write_artifacts,
)
from tests.utils.env import set_seed
from tests.utils.timing import cuda_benchmark, cuda_cleanup, format_result, is_oom_exception, oom_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint CE+KL fused kernel benchmark")
    add_common_benchmark_args(parser, default_configs=CANONICAL_T4_CONFIGS["ce_kl"])
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile on the PyTorch baseline loss path.")
    args = parser.parse_args()
    validate_positive_timing_args(args)

    runtime = require_cuda_kernel_or_skip("bench_ce_kl", args)
    torch = runtime.torch
    F = torch.nn.functional
    from orda_ce_kernel.ops.cross_entropy import distill_cross_entropy

    set_seed(torch, args.seed)
    device = torch.device("cuda")
    dtype = resolve_dtype(torch, args)
    loss_compile_enabled = not args.no_compile

    def _pytorch_ce_kl_eager(hs, ht, weight, target, T: float, kl_w: float):
        logits_s = F.linear(hs, weight)
        logits_t = F.linear(ht, weight)
        ce = F.cross_entropy(logits_s, target) + F.cross_entropy(logits_t, target)
        log_p_s = F.log_softmax(logits_s / T, dim=-1)
        p_t = F.softmax((logits_t / T).detach(), dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        return ce + kl_w * kl

    pytorch_ce_kl = compile_pytorch_loss(torch, _pytorch_ce_kl_eager, enabled=loss_compile_enabled)

    print_table_header(
        f"Orda joint CE+KL benchmark (repeats={args.repeats}, steps={args.steps}, "
        f"loss_compile={'on' if loss_compile_enabled else 'off'})",
        torch.cuda.get_device_name(0),
        args.vocab_size,
        args.hidden_dim,
    )
    rows: list[dict] = []

    for batch, seq in args.configs:
        bt = batch * seq
        hs = torch.randn(bt, args.hidden_dim, device=device, dtype=dtype, requires_grad=True)
        ht = torch.randn(bt, args.hidden_dim, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(args.vocab_size, args.hidden_dim, device=device, dtype=dtype, requires_grad=True)
        target = torch.randint(0, args.vocab_size, (bt,), device=device)

        def native_step():
            hs.grad = ht.grad = weight.grad = None
            loss = pytorch_ce_kl(hs, ht, weight, target, args.kl_temperature, args.kl_weight)
            loss.backward()

        def orda_step():
            hs.grad = ht.grad = weight.grad = None
            loss, *_ = distill_cross_entropy(
                hs,
                ht,
                weight,
                target,
                lambda_student=1.0,
                kl_weight=args.kl_weight,
                kl_temperature=args.kl_temperature,
            )
            loss.backward()

        for method, step in [("PyTorch CE+KL separate", native_step), ("Orda fused CE+KL", orda_step)]:
            try:
                cuda_cleanup(torch, device)
                result = cuda_benchmark(
                    torch,
                    step,
                    warmup=args.warmup,
                    steps=args.steps,
                    repeats=args.repeats,
                    device=device,
                    cleanup_between_repeats=False,
                    seed=args.seed,
                )
            except Exception as exc:
                if not is_oom_exception(torch, exc):
                    raise
                result = oom_result()
            warn_if_noisy(result, method)
            latency, vram = format_result(result)
            cv_str = "N/A" if math.isnan(result.cv_pct) else f"{result.cv_pct:.1f}%"
            print_table_row(batch, seq, method, f"{latency} (cv {cv_str})", vram)
            rows.append(
                benchmark_row(
                    config=f"{batch}x{seq}",
                    method=method,
                    latency_ms=result.latency_ms,
                    peak_vram_mib=result.peak_vram_mib,
                    batch=batch,
                    seq=seq,
                    bt=bt,
                    vocab_size=args.vocab_size,
                    hidden_dim=args.hidden_dim,
                    dtype=args.dtype,
                    warmup=args.warmup,
                    steps=args.steps,
                    kl_weight=args.kl_weight,
                    kl_temperature=args.kl_temperature,
                    loss_compile=loss_compile_enabled,
                    status="ok" if result.ok else "oom",
                    **bench_stats_fields(result),
                )
            )

        del hs, ht, weight, target

    print("=" * 88)
    write_artifacts(
        rows,
        collect_metadata(torch, args, "bench_ce_kl"),
        output_json=args.output_json,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()


