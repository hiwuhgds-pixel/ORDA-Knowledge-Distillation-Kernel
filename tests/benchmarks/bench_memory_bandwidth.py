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
    require_cuda_kernel_or_skip,
    resolve_dtype,
    validate_positive_timing_args,
    warn_if_noisy,
    write_artifacts,
)
from tests.utils.env import set_seed
from tests.utils.timing import cuda_benchmark, cuda_cleanup, is_oom_exception, oom_result


def estimate_traffic_gib(bt: int, h: int, v: int, method: str, element_size: int = 2) -> float:
    """Approximate global-memory traffic for relative comparisons.

    This is not a profiler replacement. Use Nsight Compute for authoritative
    DRAM/L2 metrics.
    """
    target_size = 8
    hidden = bt * h * element_size
    weight = v * h * element_size
    logits = bt * v * element_size
    targets = bt * target_size

    if method == "pytorch_ce_kl":
        total = 2 * (hidden + weight + logits) + 8 * logits + targets
        total += 2 * (logits + weight + hidden)
    elif method == "orda_ce":
        total = 2 * hidden + weight + 2 * logits + targets
        total += 2 * logits + 2 * hidden + weight
    elif method == "orda_ce_kl":
        total = 2 * hidden + weight + 2 * logits + targets
        total += 3 * logits + 2 * hidden + weight
    else:
        raise ValueError(method)
    return total / (1024 ** 3)


def verify_current_config(torch, F, distill_cross_entropy, hs, ht, w, target, args) -> None:
    def clone_inputs():
        return (
            hs.detach().clone().requires_grad_(True),
            ht.detach().clone().requires_grad_(True),
            w.detach().clone().requires_grad_(True),
        )

    def require_finite(name: str, value) -> None:
        if value is None:
            raise RuntimeError(f"{name} verification did not produce a gradient")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} verification produced NaN/Inf")

    hs_ref, ht_ref, w_ref = clone_inputs()
    logits_s = F.linear(hs_ref, w_ref)
    logits_t = F.linear(ht_ref, w_ref)
    ce_ref = F.cross_entropy(logits_s, target) + F.cross_entropy(logits_t, target)
    log_p_s = F.log_softmax(logits_s / args.kl_temperature, dim=-1)
    p_t = F.softmax((logits_t / args.kl_temperature).detach(), dim=-1)
    kl_ref = F.kl_div(log_p_s, p_t, reduction="batchmean") * (args.kl_temperature ** 2)
    ce_kl_ref = ce_ref + args.kl_weight * kl_ref
    ce_kl_ref.backward()

    hs_ce, ht_ce, w_ce = clone_inputs()
    ce_orda, *_ = distill_cross_entropy(hs_ce, ht_ce, w_ce, target, kl_weight=0.0)
    ce_orda.backward()

    hs_kl, ht_kl, w_kl = clone_inputs()
    ce_kl_orda, *_ = distill_cross_entropy(
        hs_kl,
        ht_kl,
        w_kl,
        target,
        kl_weight=args.kl_weight,
        kl_temperature=args.kl_temperature,
    )
    ce_kl_orda.backward()

    for name, value in {
        "pytorch_ce_kl": ce_kl_ref,
        "orda_ce": ce_orda,
        "orda_ce_kl": ce_kl_orda,
        "pytorch_hs_grad": hs_ref.grad,
        "pytorch_ht_grad": ht_ref.grad,
        "pytorch_w_grad": w_ref.grad,
        "orda_ce_hs_grad": hs_ce.grad,
        "orda_ce_ht_grad": ht_ce.grad,
        "orda_ce_w_grad": w_ce.grad,
        "orda_ce_kl_hs_grad": hs_kl.grad,
        "orda_ce_kl_ht_grad": ht_kl.grad,
        "orda_ce_kl_w_grad": w_kl.grad,
    }.items():
        require_finite(name, value)

    if not torch.allclose(ce_orda.float(), ce_ref.float(), atol=5e-2, rtol=5e-2):
        raise RuntimeError(
            f"Orda CE verification mismatch: orda={ce_orda.item():.6f}, ref={ce_ref.item():.6f}"
        )
    if not torch.allclose(ce_kl_orda.float(), ce_kl_ref.float(), atol=5e-2, rtol=5e-2):
        raise RuntimeError(
            f"Orda CE+KL verification mismatch: orda={ce_kl_orda.item():.6f}, ref={ce_kl_ref.item():.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimated memory bandwidth benchmark")
    add_common_benchmark_args(parser, default_configs=CANONICAL_T4_CONFIGS["memory_bandwidth"])
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile on the PyTorch baseline loss path.")
    args = parser.parse_args()
    validate_positive_timing_args(args)

    runtime = require_cuda_kernel_or_skip("bench_memory_bandwidth", args)
    torch = runtime.torch
    F = torch.nn.functional
    from orda_ce_kernel.ops.cross_entropy import distill_cross_entropy

    set_seed(torch, args.seed)
    dtype = resolve_dtype(torch, args)
    loss_compile_enabled = not args.no_compile

    print("=" * 100)
    print("Estimated memory bandwidth benchmark")
    print("=" * 100)
    print(f"{'Config':<12} | {'Method':<16} | {'Latency ms':>12} | {'Traffic GiB':>12} | {'Est. GiB/s':>12}")
    print("-" * 100)
    rows: list[dict] = []

    for batch, seq in args.configs:
        bt = batch * seq
        hs = (torch.randn(bt, args.hidden_dim, device="cuda", dtype=dtype) * 0.1).requires_grad_(True)
        ht = (torch.randn(bt, args.hidden_dim, device="cuda", dtype=dtype) * 0.1).requires_grad_(True)
        w = (torch.randn(args.vocab_size, args.hidden_dim, device="cuda", dtype=dtype) * 0.1).requires_grad_(True)
        target = torch.randint(0, args.vocab_size, (bt,), device="cuda")
        verified = False

        if args.verify:
            verify_current_config(torch, F, distill_cross_entropy, hs, ht, w, target, args)
            verified = True
            print(f"[VERIFY] {batch}x{seq}: finite CE/CE+KL loss and gradient sanity checks passed.")

        def _pytorch_ce_kl_eager(hs_, ht_, w_, target_, T, kl_w):
            logits_s = F.linear(hs_, w_)
            logits_t = F.linear(ht_, w_)
            ce = F.cross_entropy(logits_s, target_) + F.cross_entropy(logits_t, target_)
            log_p_s = F.log_softmax(logits_s / T, dim=-1)
            p_t = F.softmax((logits_t / T).detach(), dim=-1)
            kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
            return ce + kl_w * kl

        pytorch_ce_kl_compiled = compile_pytorch_loss(torch, _pytorch_ce_kl_eager, enabled=loss_compile_enabled)

        def pytorch_ce_kl():
            hs.grad = ht.grad = w.grad = None
            loss = pytorch_ce_kl_compiled(hs, ht, w, target, args.kl_temperature, args.kl_weight)
            loss.backward()

        def orda_ce():
            hs.grad = ht.grad = w.grad = None
            loss, *_ = distill_cross_entropy(hs, ht, w, target, kl_weight=0.0)
            loss.backward()

        def orda_ce_kl():
            hs.grad = ht.grad = w.grad = None
            loss, *_ = distill_cross_entropy(
                hs,
                ht,
                w,
                target,
                kl_weight=args.kl_weight,
                kl_temperature=args.kl_temperature,
            )
            loss.backward()

        for name, step in [("pytorch_ce_kl", pytorch_ce_kl), ("orda_ce", orda_ce), ("orda_ce_kl", orda_ce_kl)]:
            try:
                cuda_cleanup(torch)
                result = cuda_benchmark(
                    torch,
                    step,
                    warmup=args.warmup,
                    steps=args.steps,
                    repeats=args.repeats,
                    cleanup_between_repeats=False,
                    seed=args.seed,
                )
            except Exception as exc:
                if not is_oom_exception(torch, exc):
                    raise
                result = oom_result()
            warn_if_noisy(result, name)
            traffic = estimate_traffic_gib(bt, args.hidden_dim, args.vocab_size, name)
            cv_str = "N/A" if math.isnan(result.cv_pct) else f"{result.cv_pct:.1f}"
            if result.ok:
                gib_s = traffic / (result.latency_ms / 1000.0)
                print(f"{batch}x{seq:<7} | {name:<16} | {result.latency_ms:>10.2f} ({cv_str:>4}%) | {traffic:>12.3f} | {gib_s:>12.2f}")
            else:
                gib_s = float("nan")
                print(f"{batch}x{seq:<7} | {name:<16} | {'OOM':>17} | {traffic:>12.3f} | {'N/A':>12}")
            rows.append(
                benchmark_row(
                    config=f"{batch}x{seq}",
                    method=name,
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
                    traffic_gib=traffic,
                    estimated_gib_per_s=gib_s,
                    verified=verified,
                    loss_compile=loss_compile_enabled,
                    status="ok" if result.ok else "oom",
                    **bench_stats_fields(result),
                )
            )

    print("=" * 100)
    if args.verify:
        print("[VERIFY] Use Nsight Compute for authoritative DRAM/L2 traffic; estimates remain relative.")
    write_artifacts(
        rows,
        collect_metadata(torch, args, "bench_memory_bandwidth"),
        output_json=args.output_json,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()


