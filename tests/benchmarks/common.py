from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from tests.utils.env import get_runtime, parse_configs, torch_dtype, validate_cuda_dtype


CANONICAL_T4_CONFIGS = {
    # Values picked to fit Tesla T4 (~15 GiB) at fp16.
    # Small configs (8x1024, 4x2048) show baseline overhead; large configs
    # (16x1024, 8x2048) are where chunked kernels pull ahead.
    "ce_only":               "8x1024,4x2048,16x1024,8x2048",
    "ce_kl":                 "8x1024,4x2048,16x1024,8x2048",
    "kl_throughput":         "8x1024,16x1024",
    "kl_accuracy":           "32x256,16x512,8x1024",
    "memory_bandwidth":      "4x2048,8x2048",
    "end_to_end_compare":    (8, 1024),
    "end_to_end_orda_large": (16, 1024),
}


def add_common_benchmark_args(parser: argparse.ArgumentParser, *, default_configs: str) -> None:
    parser.add_argument("--configs", type=parse_configs, default=parse_configs(default_configs))
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5,
                        help="Outer-loop repeats; total samples = repeats * steps")
    parser.add_argument("--seed", type=int, default=42)
    add_output_args(parser)


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-json", help="Write benchmark metadata and rows to this JSON file")
    parser.add_argument("--output-csv", help="Write benchmark rows plus key metadata to this CSV file")


def validate_positive_timing_args(args) -> None:
    if getattr(args, "warmup", 0) < 0:
        raise ValueError("--warmup must be >= 0")
    if getattr(args, "steps", 1) <= 0:
        raise ValueError("--steps must be > 0")
    if getattr(args, "repeats", 1) <= 0:
        raise ValueError("--repeats must be > 0")


def compile_pytorch_loss(torch, fn, *, enabled: bool = True):
    """Wrap a PyTorch eager loss path with `torch.compile` for fair comparison.

    Triton variants (Orda) must NOT pass through this — Triton custom
    ops trigger graph breaks/recompiles that make the compiled wrapper slower
    than eager. This helper is for PyTorch baselines only.

    The compiled callable is guarded so that the *first* invocation, which is
    where Dynamo/Inductor typically raise, falls back to eager rather than
    crashing the benchmark. Once eager is selected, subsequent calls go
    straight to eager — we never silently mix compiled and eager runs inside
    the same `BenchResult`.
    """
    if not enabled or not hasattr(torch, "compile"):
        return fn
    try:
        compiled = torch.compile(fn, fullgraph=False, dynamic=False)
    except Exception as exc:
        print(f"[compile_pytorch_loss] torch.compile() rejected the function at compile time "
              f"({type(exc).__name__}: {exc}); falling back to eager.")
        return fn

    state = {"fallback": False}

    def guarded(*args, **kwargs):
        if state["fallback"]:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            # First-call Dynamo/Inductor failures land here.
            print(f"[compile_pytorch_loss] compiled callable raised at runtime "
                  f"({type(exc).__name__}: {exc}); switching this variant to eager.")
            state["fallback"] = True
            return fn(*args, **kwargs)

    return guarded


def compile_model(torch, module, name: str, *, enabled: bool = True, is_oom_exception=None):
    """Wrap `torch.compile(module)` with the same eager fallback policy as losses."""
    if not enabled or not hasattr(torch, "compile"):
        return module
    try:
        compiled = torch.compile(module)
    except Exception as exc:
        print(f"[compile_model] torch.compile({name}) rejected the module at compile time "
              f"({type(exc).__name__}: {exc}); falling back to eager.")
        return module

    state = {"fallback": False}

    class GuardedCompiledModel:
        def __call__(self, *args, **kwargs):
            if state["fallback"]:
                return module(*args, **kwargs)
            try:
                return compiled(*args, **kwargs)
            except Exception as exc:
                if is_oom_exception is not None and is_oom_exception(torch, exc):
                    raise
                print(f"[compile_model] compiled {name} raised at runtime "
                      f"({type(exc).__name__}: {exc}); switching this model to eager.")
                state["fallback"] = True
                return module(*args, **kwargs)

        def __getattr__(self, attr):
            return getattr(module, attr)

    return GuardedCompiledModel()


def warn_if_noisy(result, variant_name: str, *, cv_threshold_pct: float = 15.0) -> bool:
    """Print a warning when the coefficient of variation exceeds the threshold.

    Returns True when a warning was emitted. Caller still records the row.
    """
    import math

    cv = getattr(result, "cv_pct", float("nan"))
    if math.isnan(cv) or cv <= cv_threshold_pct:
        return False
    print(
        f"[WARN] {variant_name}: cv={cv:.1f}% exceeds {cv_threshold_pct:.0f}% threshold; "
        "results may be noisy, do not use for headline claims."
    )
    return True


def bench_stats_fields(result) -> dict:
    """Pull the standard stats block out of a `BenchResult` for `benchmark_row(...)`."""
    return {
        "latency_ms_std": getattr(result, "latency_ms_std", None),
        "latency_ms_min": getattr(result, "latency_ms_min", None),
        "latency_ms_max": getattr(result, "latency_ms_max", None),
        "latency_ms_p50": getattr(result, "latency_ms_p50", None),
        "latency_ms_p95": getattr(result, "latency_ms_p95", None),
        "cv_pct": getattr(result, "cv_pct", None),
        "repeats": getattr(result, "repeats", None),
    }


def resolve_dtype(torch, args):
    dtype = torch_dtype(torch, args.dtype)
    validate_cuda_dtype(torch, dtype)
    return dtype


def require_cuda_kernel_or_skip(script_name: str, args):
    try:
        runtime = get_runtime()
    except RuntimeError as exc:
        skip_reason = str(exc)
        print(f"[SKIP] {script_name}: {skip_reason}; no GPU numbers produced.")
        rows = skipped_benchmark_rows(args, skip_reason)
        write_artifacts(
            rows,
            collect_metadata(None, args, script_name),
            output_json=getattr(args, "output_json", None),
            output_csv=getattr(args, "output_csv", None),
        )
        raise SystemExit(0) from exc

    torch = runtime.torch
    orda = runtime.orda
    skip_reason = None
    if not torch.cuda.is_available():
        skip_reason = "CUDA is not available"
    elif not orda.is_available():
        skip_reason = "Triton kernels are not available"

    if skip_reason is None:
        return runtime

    print(f"[SKIP] {script_name}: {skip_reason}; no GPU numbers produced.")
    rows = skipped_benchmark_rows(args, skip_reason)
    write_artifacts(
        rows,
        collect_metadata(torch, args, script_name),
        output_json=getattr(args, "output_json", None),
        output_csv=getattr(args, "output_csv", None),
    )
    raise SystemExit(0)


def skipped_benchmark_rows(args, skip_reason: str) -> list[dict]:
    common = {
        key: getattr(args, key)
        for key in (
            "vocab_size",
            "hidden_dim",
            "dtype",
            "warmup",
            "steps",
            "repeats",
            "kl_weight",
            "kl_temperature",
        )
        if hasattr(args, key)
    }
    base = {
        "status": "skipped",
        "skip_reason": skip_reason,
        **common,
    }
    configs = getattr(args, "configs", None)
    if configs:
        return [
            benchmark_row(
                config=f"{batch}x{seq}",
                method="skip",
                batch=batch,
                seq=seq,
                bt=batch * seq,
                **base,
            )
            for batch, seq in configs
        ]

    row = {
        key: getattr(args, key)
        for key in ("batch_size", "seq_len")
        if hasattr(args, key)
    }
    return [benchmark_row(method="skip", **row, **base)]


def collect_metadata(torch, args, benchmark_name: str) -> dict:
    device = {}
    if torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_mib": int(props.total_memory // (1024 ** 2)),
        }

    return {
        "benchmark": benchmark_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None) if torch is not None else None,
        "cuda": getattr(torch.version, "cuda", None) if torch is not None else None,
        "hip": getattr(torch.version, "hip", None) if torch is not None else None,
        "device": device,
        "args": vars(args),
    }


def benchmark_row(
    *,
    config: str | None = None,
    method: str,
    latency_ms: float | None = None,
    peak_vram_mib: float | None = None,
    **extra,
) -> dict:
    row = {
        "config": config,
        "method": method,
        "latency_ms": latency_ms,
        "peak_vram_mib": peak_vram_mib,
    }
    row.update(extra)
    return row


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


CSV_FIELD_ORDER = [
    "benchmark",
    "timestamp_utc",
    "python",
    "platform",
    "torch",
    "cuda",
    "hip",
    "device",
    "device_capability",
    "config",
    "method",
    "status",
    "skip_reason",
    "latency_ms",
    "latency_ms_std",
    "latency_ms_min",
    "latency_ms_max",
    "latency_ms_p50",
    "latency_ms_p95",
    "cv_pct",
    "repeats",
    "peak_vram_mib",
    "batch",
    "seq",
    "bt",
    "batch_size",
    "seq_len",
    "tokens_per_second",
    "vocab_size",
    "hidden_dim",
    "dtype",
    "warmup",
    "steps",
    "kl_weight",
    "kl_temperature",
    "lambda_student",
    "sample_frac",
    "traffic_gib",
    "estimated_gib_per_s",
    "student_layers",
    "teacher_layers",
    "grad_accum",
    "loss_compile",
    "mode",
    "task",
    "vs_no_kl",
    "peak_vram_delta_vs_no_kl_mib",
    "peak_vram_loss_only_mib",
    "layers",
    "heads",
    "kl_loss",
    "ref_kl_loss",
    "total_loss",
    "kl_rel_err_pct",
    "grad_cosine",
    "mean_abs_diff",
    "max_abs_diff",
    "verified",
]


def write_artifacts(rows: list[dict], metadata: dict, *, output_json: str | None, output_csv: str | None) -> None:
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[ARTIFACT] wrote JSON: {path}")

    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_rows = []
        flat_metadata = {
            "benchmark": metadata.get("benchmark"),
            "timestamp_utc": metadata.get("timestamp_utc"),
            "python": metadata.get("python"),
            "platform": metadata.get("platform"),
            "torch": metadata.get("torch"),
            "cuda": metadata.get("cuda"),
            "hip": metadata.get("hip"),
            "device": metadata.get("device", {}).get("name"),
            "device_capability": metadata.get("device", {}).get("capability"),
        }
        for row in rows:
            csv_rows.append({**flat_metadata, **row})
        row_keys = {key for row in csv_rows for key in row}
        ordered = [key for key in CSV_FIELD_ORDER if key in row_keys]
        extras = sorted(row_keys - set(ordered))
        fieldnames = ordered + extras
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
        print(f"[ARTIFACT] wrote CSV: {path}")


def print_table_header(title: str, device_name: str, vocab: int, hidden: int) -> None:
    print("=" * 88)
    print(title)
    print("=" * 88)
    print(f"Device: {device_name}")
    print(f"Vocab size: {vocab} | Hidden dim: {hidden}")
    print(f"{'Config':<14} | {'Method':<28} | {'Latency ms':>12} | {'Peak VRAM MiB':>14}")
    print("-" * 88)


def print_table_row(batch: int, seq: int, method: str, latency: str, vram: str) -> None:
    print(f"{batch}x{seq:<10} | {method:<28} | {latency:>12} | {vram:>14}")


