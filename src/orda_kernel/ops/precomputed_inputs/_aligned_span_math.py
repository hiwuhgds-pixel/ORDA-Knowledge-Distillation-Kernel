"""Shared framework-side math for aligned-span execution."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

_LOG_2 = math.log(2.0)
MAX_SPAN_LOG_PROB = -1e-7


# ── Span probability math ────────────────────────────────────────────────────
class SpanObjectiveTerms(NamedTuple):
    student_log_probs: torch.Tensor
    teacher_log_probs: torch.Tensor
    keep: torch.Tensor
    per_span_loss: torch.Tensor
    reduced_loss: torch.Tensor
    gradient: torch.Tensor
    reduction_scale: torch.Tensor


def log1mexp(value: torch.Tensor) -> torch.Tensor:
    """Stable ``log(1 - exp(value))`` for non-positive log-probabilities."""
    lower = value < -_LOG_2
    return torch.where(
        lower,
        torch.log1p(-torch.exp(value.clamp(max=-_LOG_2))),
        torch.log(-torch.expm1(value.clamp(min=-_LOG_2, max=MAX_SPAN_LOG_PROB))),
    )


def zero_preserving_mul(weight: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Multiply while pinning the zero-weight value and gradient to zero."""
    active = weight > 0
    safe_value = torch.where(active, value, torch.zeros_like(value))
    return torch.where(active, weight * safe_value, torch.zeros_like(value))


def span_log_probs_from_nll(
    token_nll: torch.Tensor,
    ranges: torch.Tensor,
) -> torch.Tensor:
    """Reduce row NLL into half-open span log-probabilities in float64."""
    nll64 = token_nll.to(torch.float64)
    ranges64 = ranges.to(device=token_nll.device, dtype=torch.int64)
    positive_inf = torch.isposinf(nll64)
    finite_nll = torch.where(positive_inf, nll64.new_zeros(()), nll64)
    prefix = torch.cat([nll64.new_zeros(1), finite_nll.cumsum(0)])
    inf_prefix = torch.cat(
        [positive_inf.new_zeros(1, dtype=torch.int64), positive_inf.cumsum(0)]
    )
    span_nll = prefix[ranges64[:, 1]] - prefix[ranges64[:, 0]]
    has_zero_support = (inf_prefix[ranges64[:, 1]] - inf_prefix[ranges64[:, 0]]) > 0
    return torch.where(
        has_zero_support, span_nll.new_full((), float("-inf")), -span_nll
    )


def span_valid_counts(mask: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """Count valid rows per half-open span."""
    counts = mask.to(torch.int64)
    prefix = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    ranges64 = ranges.to(device=mask.device, dtype=torch.int64)
    return prefix[ranges64[:, 1]] - prefix[ranges64[:, 0]]


def binary_span_objective(
    student_log_probs_raw: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    keep: torch.Tensor,
    *,
    objective: str,
    reduction: str,
) -> SpanObjectiveTerms:
    """Return stable binary loss and ``dL / d ell_s`` before loss weighting."""
    ell_s_raw = student_log_probs_raw.to(torch.float64)
    ell_t = teacher_log_probs.to(device=ell_s_raw.device, dtype=torch.float64)
    ell_s_capped = ell_s_raw.clamp(max=MAX_SPAN_LOG_PROB)
    ell_s = torch.where(
        ell_s_raw <= MAX_SPAN_LOG_PROB,
        ell_s_raw,
        ell_s_raw + (ell_s_capped - ell_s_raw).detach(),
    )
    ell_t = ell_t.clamp(max=MAX_SPAN_LOG_PROB).detach()
    keep = keep.to(device=ell_s.device, dtype=torch.bool)

    p_s = ell_s.exp()
    p_t = ell_t.exp()
    log_1m_s = log1mexp(ell_s)
    if objective == "binary_kl":
        span_loss = zero_preserving_mul(p_t, ell_t - ell_s) + (1.0 - p_t) * (
            log1mexp(ell_t) - log_1m_s
        )
    elif objective == "binary_ce":
        span_loss = -(zero_preserving_mul(p_t, ell_s) + (1.0 - p_t) * log_1m_s)
    else:
        raise ValueError(f"unknown aligned-span objective {objective!r}")

    span_loss = torch.where(keep, span_loss, span_loss.new_zeros(()))
    retained = keep.sum()
    reduction_scale = (
        ell_s.new_ones(()) / retained.clamp_min(1).to(torch.float64)
        if reduction == "mean"
        else ell_s.new_ones(())
    )
    if reduction not in ("mean", "sum"):
        raise ValueError(f"unknown aligned-span reduction {reduction!r}")

    gradient = (p_s - p_t) / (1.0 - p_s)
    gradient = torch.where(
        keep,
        gradient,
        gradient.new_zeros(()),
    )
    return SpanObjectiveTerms(
        ell_s,
        ell_t,
        keep,
        span_loss,
        span_loss.sum() * reduction_scale,
        gradient,
        reduction_scale,
    )


# ── Span row expansion ───────────────────────────────────────────────────────
def expand_span_values(
    values: torch.Tensor,
    ranges: torch.Tensor,
    n_rows: int,
) -> torch.Tensor:
    """Expand additive span values to rows with an endpoint difference array."""
    if n_rows == 0:
        return values.new_zeros((0,), dtype=torch.float64)
    values64 = values.to(torch.float64)
    ranges64 = ranges.to(device=values.device, dtype=torch.int64)
    delta = values64.new_zeros(n_rows + 1)
    delta.index_add_(0, ranges64[:, 0], values64)
    delta.index_add_(0, ranges64[:, 1], -values64)
    return delta.cumsum(0)[:-1]


def aligned_row_coefficients(
    valid_rows: torch.Tensor,
    ranges: torch.Tensor,
    span_gradient: torch.Tensor,
    *,
    student_ce_weight: float,
    span_kl_weight: float,
    reduction: str,
    span_reduction_scale: torch.Tensor,
) -> torch.Tensor:
    """Assemble the CE-shaped scalar coefficient for each student row."""
    if reduction == "mean":
        ce_scale = span_reduction_scale.new_ones(()) / valid_rows.sum().clamp_min(1).to(
            torch.float64
        )
    elif reduction == "sum":
        ce_scale = span_reduction_scale.new_ones(())
    else:
        raise ValueError(f"unknown aligned-span reduction {reduction!r}")
    row_span = expand_span_values(span_gradient, ranges, valid_rows.shape[0])
    coefficient = (
        valid_rows.to(torch.float64) * float(student_ce_weight) * ce_scale
        - float(span_kl_weight) * span_reduction_scale * row_span
    )
    return coefficient.to(torch.float32)
