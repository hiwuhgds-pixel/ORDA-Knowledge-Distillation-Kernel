from __future__ import annotations

from collections.abc import Mapping, Sequence

OOM = float("inf")


def trimmed_mean(values: list[float], trim_ratio: float = 0.10) -> float:
    if not values:
        raise ValueError("trimmed_mean requires at least one value")
    values = sorted(values)
    cut = int(len(values) * trim_ratio)
    if cut > 0 and len(values) > 2 * cut:
        values = values[cut:-cut]
    return sum(values) / len(values)


def average_passes(
    samples: Sequence[Mapping[str, float]],
    keys: Sequence[str],
    *,
    guard_key: str,
) -> dict[str, float]:
    """Reject averaging a finite pass with OOM because it would understate cost."""
    if not samples:
        raise ValueError("average_passes requires at least one sample")
    if any(sample[guard_key] == OOM for sample in samples):
        return {key: OOM for key in keys}
    return {key: sum(sample[key] for sample in samples) / len(samples) for key in keys}
