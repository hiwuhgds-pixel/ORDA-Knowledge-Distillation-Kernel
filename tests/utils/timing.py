from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class BenchResult:
    """Timing summary across `repeats * steps` CUDA event samples.

    `latency_ms` is the trimmed mean (10%) over all samples and remains the
    primary number for reporting. The extra fields are exposed so dashboards
    can surface variance and tail latency.
    """

    latency_ms: float
    peak_vram_mib: float
    latency_ms_std: float = float("nan")
    latency_ms_min: float = float("nan")
    latency_ms_max: float = float("nan")
    latency_ms_p50: float = float("nan")
    latency_ms_p95: float = float("nan")
    cv_pct: float = float("nan")
    repeats: int = 1
    steps: int = 0
    samples: tuple[float, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not math.isnan(self.latency_ms)


def oom_result() -> BenchResult:
    return BenchResult(latency_ms=float("nan"), peak_vram_mib=float("nan"))


def is_oom_exception(torch, exc: BaseException) -> bool:
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, cuda_oom) or "out of memory" in str(exc).lower()


def cuda_cleanup(torch, device=None) -> None:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gc.collect()


def trimmed_mean(values: list[float], trim_ratio: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    cut = int(len(values) * trim_ratio)
    if cut > 0 and len(values) > 2 * cut:
        values = values[cut:-cut]
    return sum(values) / len(values)


# Backward-compatible private alias.
_trimmed_mean = trimmed_mean


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile q must be in [0, 100]")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def cv_pct(mean: float, std: float) -> float:
    if not math.isfinite(mean) or mean == 0.0 or not math.isfinite(std):
        return float("nan")
    return (std / abs(mean)) * 100.0


def summarize_samples(samples: list[float], *, trim_ratio: float = 0.10) -> dict:
    """Compute the statistics block emitted on every benchmark row."""
    if not samples:
        nan = float("nan")
        return {
            "latency_ms": nan,
            "latency_ms_std": nan,
            "latency_ms_min": nan,
            "latency_ms_max": nan,
            "latency_ms_p50": nan,
            "latency_ms_p95": nan,
            "cv_pct": nan,
        }
    mean = trimmed_mean(samples, trim_ratio)
    std = stdev(samples)
    return {
        "latency_ms": mean,
        "latency_ms_std": std,
        "latency_ms_min": min(samples),
        "latency_ms_max": max(samples),
        "latency_ms_p50": percentile(samples, 50.0),
        "latency_ms_p95": percentile(samples, 95.0),
        "cv_pct": cv_pct(mean, std),
    }


def cuda_benchmark(
    torch,
    step_fn: Callable[[], None],
    *,
    warmup: int,
    steps: int,
    repeats: int = 1,
    device=None,
    trim_ratio: float = 0.10,
    cleanup_between_repeats: bool = False,
    seed: Optional[int] = None,
) -> BenchResult:
    """Run `repeats` outer loops of `steps` timed iterations.

    Each iteration is timed with CUDA events. Stats (mean/std/min/max/p50/p95)
    are computed across `repeats * steps` samples. Peak VRAM is the max across
    every measured iteration.

    `cleanup_between_repeats=True` performs gc + empty_cache + sync between
    outer loops. `seed` (if provided) is re-applied before each outer loop's
    warmup so that input randomness does not contribute to measured variance.
    """
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    samples: list[float] = []
    peak_overall = 0.0

    for repeat_idx in range(repeats):
        if seed is not None:
            torch.manual_seed(seed + repeat_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + repeat_idx)

        for _ in range(warmup):
            step_fn()

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step_fn()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(start.elapsed_time(end))

        peak_overall = max(peak_overall, torch.cuda.max_memory_allocated(device) / (1024 ** 2))

        if cleanup_between_repeats and repeat_idx + 1 < repeats:
            cuda_cleanup(torch, device)

    stats = summarize_samples(samples, trim_ratio=trim_ratio)
    return BenchResult(
        latency_ms=stats["latency_ms"],
        peak_vram_mib=peak_overall,
        latency_ms_std=stats["latency_ms_std"],
        latency_ms_min=stats["latency_ms_min"],
        latency_ms_max=stats["latency_ms_max"],
        latency_ms_p50=stats["latency_ms_p50"],
        latency_ms_p95=stats["latency_ms_p95"],
        cv_pct=stats["cv_pct"],
        repeats=repeats,
        steps=steps,
        samples=tuple(samples),
    )


def format_result(result: BenchResult) -> tuple[str, str]:
    if not result.ok:
        return "OOM", "N/A"
    return f"{result.latency_ms:.2f}", f"{result.peak_vram_mib:.1f}"


