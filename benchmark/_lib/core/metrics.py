"""Keep the OOM sentinel consistent across console, CSV, and worker output."""

from __future__ import annotations

import math
from dataclasses import dataclass

OOM = float("inf")
OOM_TEXT = "OOM"
NA_TEXT = "NA"


def is_oom(value: float) -> bool:
    return value == OOM


# ── Numeric text ─────────────────────────────────────────────────────────────
def fmt_metric(
    value: float, *, precision: int, width: int = 0, suffix: str = ""
) -> str:
    # The unit suffix belongs to the number: an OOM cell reads OOM, not OOM MB.
    text = OOM_TEXT if is_oom(value) else f"{value:.{precision}f}{suffix}"
    return f"{text:>{width}}" if width else text


def fmt_ms(value: float) -> str:
    return fmt_metric(value, precision=2, width=9)


def fmt_mb(value: float) -> str:
    return fmt_metric(value, precision=1, width=12)


def fmt_table_ms(value: float) -> str:
    return fmt_metric(value, precision=2)


def fmt_table_mb(value: float) -> str:
    return fmt_metric(value, precision=1)


def fmt_backend_mb(value: float) -> str:
    return fmt_metric(value, precision=1, suffix=" MB")


def fmt_vocab(vocab: int) -> str:
    return {
        8_192: "8k",
        16_384: "16k",
        32_768: "32k",
        65_536: "64k",
        131_072: "128k",
        262_144: "256k",
    }.get(vocab, str(vocab))


def fmt_precision(dtype, use_amp: bool, use_grad_scaler: bool) -> str:
    suffix = []
    if use_amp:
        suffix.append("amp")
    if use_grad_scaler:
        suffix.append("gradscale")
    if suffix:
        return f"{dtype} ({', '.join(suffix)})"
    return str(dtype)


def csv_cell(value):
    if isinstance(value, float) and is_oom(value):
        return OOM_TEXT
    return value


def pack_metric(value: float):
    # inf is not valid JSON; the worker protocol encodes it as "OOM".
    return OOM_TEXT if is_oom(value) else value


def unpack_metric(value) -> float:
    return OOM if value == OOM_TEXT else float(value)


# ── Metric deltas ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Delta:
    absolute: float | None
    percent: float | None

    @property
    def defined(self) -> bool:
        return self.absolute is not None and self.percent is not None


def delta(baseline: float, candidate: float) -> Delta:
    if is_oom(baseline) or is_oom(candidate):
        return Delta(None, None)
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        return Delta(None, None)
    absolute = candidate - baseline
    if baseline == 0.0:
        return Delta(absolute, None)
    return Delta(absolute, (candidate / baseline - 1.0) * 100.0)


def fmt_delta_absolute(value: Delta, *, precision: int) -> str:
    if value.absolute is None:
        return OOM_TEXT
    return f"{value.absolute:+.{precision}f}"


def fmt_delta_percent(value: Delta) -> str:
    # OOM when the measurement failed, NA when the baseline was zero.
    if value.absolute is None:
        return OOM_TEXT
    if value.percent is None:
        return NA_TEXT
    return f"{value.percent:+.1f}%"


def fmt_delta_percent_na(value: Delta) -> str:
    if value.percent is None:
        return NA_TEXT
    return f"{value.percent:+.1f}%"
