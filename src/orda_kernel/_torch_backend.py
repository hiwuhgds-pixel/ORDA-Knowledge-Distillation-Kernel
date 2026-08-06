from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F

from ._api_types import (
    DistillationLossOutput,
    EntropyGate,
    PrecomputedTeacher,
    Reduction,
    SeparateTeacher,
    TiedTeacher,
)
from ._planning.models import (
    TEACHER_GRAD_NONE,
    TEACHER_KL_GRAD_VARIANTS,
    ExecutionPlan,
)
from ._runtime import cast_logits_for_compute
from ._validation import _teacher_mode_and_tensors


# ── Loss and empty-output helpers ────────────────────────────────────────────
def _loss_output_dtype(input_dtype: torch.dtype) -> torch.dtype:
    """Use a float32 floor while preserving the Torch-only float64 path."""
    return torch.promote_types(input_dtype, torch.float32)


def _zeroed_loss_components(
    loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pair a differentiable zero loss with three detached zero components."""
    zero = loss.detach().new_zeros(())
    return loss, zero, zero.clone(), zero.clone()


def _empty_distillation_tensors(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    mode: str,
    teacher_hidden: torch.Tensor | None,
    teacher_weight: torch.Tensor | None,
    student_ce_weight: float,
    kl_weight: float,
    teacher_grad_policy: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gradient_sources: list[torch.Tensor] = []
    student_gradient_live = student_ce_weight != 0.0 or kl_weight != 0.0
    if student_gradient_live:
        gradient_sources.extend((student_hidden, weight))

    teacher_gradient_live = teacher_grad_policy != TEACHER_GRAD_NONE
    if teacher_gradient_live:
        if teacher_hidden is None:
            raise ValueError(f"{mode} empty-batch execution requires teacher_hidden")
        gradient_sources.append(teacher_hidden)
        if mode == "tied":
            if not student_gradient_live:
                gradient_sources.append(weight)
        elif mode == "separate":
            if teacher_weight is None:
                raise ValueError(
                    "separate empty-batch execution requires teacher_weight"
                )
            gradient_sources.append(teacher_weight)
        else:
            raise ValueError(
                f"teacher gradient policy {teacher_grad_policy!r} is invalid for "
                f"mode {mode!r}"
            )

    output_dtype = _loss_output_dtype(student_hidden.dtype)
    if gradient_sources:
        loss = gradient_sources[0].reshape(-1)[:0].sum()
        for source in gradient_sources[1:]:
            loss = loss + source.reshape(-1)[:0].sum()
        loss = loss.to(output_dtype)
    else:
        loss = student_hidden.new_zeros((), dtype=output_dtype)
    return _zeroed_loss_components(loss)


# ── Torch reference backend ──────────────────────────────────────────────────
def _zero_loss_from_sources(*sources: torch.Tensor) -> torch.Tensor:
    """Create a differentiable zero without reading potentially non-finite data."""
    if not sources:
        raise ValueError("at least one source tensor is required")
    loss = sources[0][:0].sum()
    for source in sources[1:]:
        loss = loss + source[:0].sum()
    return loss


def _combine_weighted_losses(
    student_ce: torch.Tensor,
    teacher_ce: torch.Tensor,
    kl: torch.Tensor,
    *,
    student_ce_weight: float,
    teacher_ce_weight: float,
    kl_weight: float,
) -> torch.Tensor:
    """Sum only active loss terms so zero weights cannot propagate NaN or Inf."""
    loss = student_ce.new_zeros(())
    if student_ce_weight != 0.0:
        loss = loss + float(student_ce_weight) * student_ce
    if teacher_ce_weight != 0.0:
        loss = loss + float(teacher_ce_weight) * teacher_ce
    if kl_weight != 0.0:
        loss = loss + float(kl_weight) * kl
    return loss


# ── KL zero-probability limit ────────────────────────────────────────────────
def _zero_preserving_mul(p: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
    """``p * diff`` with the ``p == 0`` limit pinned to zero (value and grad).

    KL terms sum ``p * (log_p - log_q)``; ``p == 0`` (log-prob ``-inf``, e.g.
    masked ``PrecomputedTeacher`` logits) gives ``0 * -inf -> NaN`` whose limit
    is ``0``. ``diff`` is neutralised *before* the multiply because a single
    ``where`` still leaks ``0 * inf -> NaN`` through the gradient when ``p``
    carries grad and ``diff`` is ``±inf``. ``p > 0`` is untouched, so a genuine
    ``+inf`` survives and finite inputs are bit-identical to ``p * diff``.
    """
    positive = p > 0
    safe_diff = torch.where(positive, diff, torch.zeros_like(diff))
    return torch.where(positive, p * safe_diff, torch.zeros_like(diff))


class _FirstOrderProxy(torch.autograd.Function):
    """Attach an explicit first-order gradient and reject higher-order use."""

    @staticmethod
    def forward(ctx, value, logits, gradient, variant):
        ctx.save_for_backward(gradient)
        ctx.variant = variant
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError(
                f"kl_divergence={ctx.variant!r} supports first-order autograd only; "
                "higher-order gradients would omit the stabilized KL curvature"
            )
        (gradient,) = ctx.saved_tensors
        return None, grad_output.unsqueeze(-1) * gradient, None, None


def _first_order_proxy(
    value: torch.Tensor,
    logits: torch.Tensor,
    gradient: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    return _FirstOrderProxy.apply(value.detach(), logits, gradient.detach(), variant)


def _nan_visible_rows(
    rows: torch.Tensor,
    invalid: torch.Tensor,
    *gradient_anchors: torch.Tensor,
) -> torch.Tensor:
    """Make invalid rows and their selected first-order gradients visibly NaN."""
    nan_scale = torch.where(
        invalid,
        torch.full_like(rows, math.nan),
        torch.zeros_like(rows),
    )
    proxy = torch.zeros_like(rows)
    for anchor in gradient_anchors:
        active = torch.where(
            invalid.unsqueeze(-1),
            anchor,
            torch.zeros_like(anchor),
        )
        proxy = proxy + active.sum(dim=-1)
    invalid_rows = proxy * nan_scale
    return torch.where(invalid, invalid_rows, rows)


def _support_safe_renyi_rows(
    logits_s_scaled: torch.Tensor,
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    r: float,
    *,
    reverse: bool,
) -> torch.Tensor:
    """Stable Rényi value and exact logit gradient with support sentinels."""
    log_base, log_other = (log_p_s, log_p_t) if reverse else (log_p_t, log_p_s)
    both_zero = torch.isneginf(log_base) & torch.isneginf(log_other)
    zero = torch.zeros_like(log_base)
    safe_base = torch.where(both_zero, zero, log_base)
    safe_other = torch.where(both_zero, zero, log_other)
    terms = r * safe_base + (1.0 - r) * safe_other
    terms = torch.where(both_zero, torch.full_like(terms, -math.inf), terms)

    numeric_invalid = (
        (~torch.isfinite(log_base) & ~torch.isneginf(log_base))
        | (~torch.isfinite(log_other) & ~torch.isneginf(log_other))
    ).any(dim=-1)
    unsafe = torch.isposinf(terms).any(dim=-1) | (
        ~torch.isfinite(terms).any(dim=-1) & ~numeric_invalid
    )
    valid = ~unsafe & ~numeric_invalid
    safe_terms = torch.where(valid.unsqueeze(-1), terms, zero)
    log_a_lse = torch.logsumexp(safe_terms, dim=-1)

    base_support = torch.isfinite(log_base)
    active = valid.unsqueeze(-1) & base_support
    safe_delta = torch.where(active, log_other - log_base, zero)
    base = torch.where(active, log_base.exp(), zero)
    other_outside_base = valid.unsqueeze(-1) & ~base_support & torch.isfinite(log_other)
    outside_mass = torch.where(
        other_outside_base,
        log_other.exp(),
        zero,
    ).sum(dim=-1)
    # Remove first-order cancellation, then restore the mass outside base support.
    renyi_core = torch.expm1((1.0 - r) * safe_delta) - (
        (1.0 - r) * torch.expm1(safe_delta)
    )
    s = (base * renyi_core).sum(dim=-1) - (1.0 - r) * outside_mass
    use_control_variate = (
        valid & torch.isfinite(s) & (s > -1.0) & (abs(r - 1.0) <= 0.125)
    )
    safe_s = torch.where(use_control_variate, s, torch.zeros_like(s))
    log_a = torch.where(use_control_variate, torch.log1p(safe_s), log_a_lse)
    # Rényi is non-negative; pin residual fp32 sign noise at identity to zero.
    finite_loss = (log_a / (r - 1.0)).clamp_min(0.0)
    value = torch.where(unsafe, torch.full_like(finite_loss, math.inf), finite_loss)
    value = torch.where(numeric_invalid, torch.full_like(value, math.nan), value)

    row_live = valid.unsqueeze(-1)
    student_live = row_live & torch.isfinite(log_p_s)
    safe_log_p_s = torch.where(student_live, log_p_s, zero)
    p_s = torch.where(student_live, safe_log_p_s.exp(), zero)

    joint_live = row_live & torch.isfinite(log_p_s) & torch.isfinite(log_p_t)
    safe_joint_s = torch.where(joint_live, log_p_s, zero)
    safe_joint_t = torch.where(joint_live, log_p_t, zero)
    teacher_power = 1.0 - r if reverse else r
    safe_log_a = torch.where(valid, log_a, torch.zeros_like(log_a)).unsqueeze(-1)
    escort = torch.where(
        joint_live,
        torch.exp(
            safe_joint_s + teacher_power * (safe_joint_t - safe_joint_s) - safe_log_a
        ),
        zero,
    )
    if reverse:
        grad_logits = (r / (r - 1.0)) * (escort - p_s)
    else:
        grad_logits = p_s - escort
    safe_logits = torch.where(torch.isfinite(logits_s_scaled), logits_s_scaled, zero)
    variant = "renyi_reverse" if reverse else "renyi"
    return _first_order_proxy(value, safe_logits, grad_logits, variant)


def _support_safe_reverse_kl_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    *,
    mass_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reverse-KL rows and the zero-gradient missing-support mask."""
    p_s = log_p_s.exp()
    mass = p_s if mass_scale is None else mass_scale.unsqueeze(-1) * p_s
    sampling_support = torch.isfinite(log_p_s)
    if mass_scale is not None:
        sampling_support = sampling_support & (mass_scale.unsqueeze(-1) > 0)
    unsafe = (sampling_support & torch.isneginf(log_p_t)).any(dim=-1)
    active = ~unsafe
    zero = torch.zeros_like(log_p_s)
    safe_mass = torch.where(active.unsqueeze(-1), mass, zero)
    safe_log_p_s = torch.where(active.unsqueeze(-1), log_p_s, zero)
    safe_log_p_t = torch.where(active.unsqueeze(-1), log_p_t, zero)
    finite = _zero_preserving_mul(safe_mass, safe_log_p_s - safe_log_p_t).sum(dim=-1)
    value = torch.where(unsafe, torch.full_like(finite, math.inf), finite)
    return value, unsafe


def _stable_abkd_rows(
    logits_s_scaled: torch.Tensor,
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    a: float,
    b: float,
) -> torch.Tensor:
    """Evaluate ABKD per column and inject its cancellation-free logit gradient."""
    c = a + b
    both_zero = torch.isneginf(log_p_s) & torch.isneginf(log_p_t)
    zero = torch.zeros_like(log_p_s)
    safe_p = torch.where(both_zero, zero, log_p_s)
    safe_q = torch.where(both_zero, zero, log_p_t)
    q_le_p = safe_q <= safe_p
    low_delta = safe_q - safe_p
    high_delta = safe_p - safe_q

    low_taylor = (
        0.5 * low_delta.square()
        + ((a + 2.0 * b) / 6.0) * low_delta.pow(3)
        + ((a * a + 3.0 * a * b + 3.0 * b * b) / 24.0) * low_delta.pow(4)
    )
    high_taylor = (
        0.5 * high_delta.square()
        + ((2.0 * a + b) / 6.0) * high_delta.pow(3)
        + ((3.0 * a * a + 3.0 * a * b + b * b) / 24.0) * high_delta.pow(4)
    )
    low_exact = ((b / c) * torch.expm1(c * low_delta) - torch.expm1(b * low_delta)) / (
        a * b
    )
    high_exact = (
        (a / c) * torch.expm1(c * high_delta) - torch.expm1(a * high_delta)
    ) / (a * b)
    low_core = torch.where(low_delta.abs() <= 1.0e-2, low_taylor, low_exact)
    high_core = torch.where(high_delta.abs() <= 1.0e-2, high_taylor, high_exact)
    p_c = torch.exp(c * safe_p)
    q_c = torch.exp(c * safe_q)
    per_column = torch.where(q_le_p, p_c * low_core, q_c * high_core)
    per_column = torch.where(both_zero, zero, per_column)

    h_low = -p_c * torch.expm1(b * low_delta)
    h_high = q_c * torch.exp(a * high_delta) * torch.expm1(b * high_delta)
    h = torch.where(both_zero, zero, torch.where(q_le_p, h_low, h_high))
    h_sum = h.sum(dim=-1, keepdim=True)
    grad_logits = (h - log_p_s.exp() * h_sum) / b
    value = per_column.sum(dim=-1)

    numeric_invalid = (
        (~torch.isfinite(log_p_s) & ~torch.isneginf(log_p_s))
        | (~torch.isfinite(log_p_t) & ~torch.isneginf(log_p_t))
    ).any(dim=-1)
    grad_logits = torch.where(numeric_invalid.unsqueeze(-1), zero, grad_logits)
    value = torch.where(numeric_invalid, torch.full_like(value, math.nan), value)
    safe_logits = torch.where(torch.isfinite(logits_s_scaled), logits_s_scaled, zero)
    return _first_order_proxy(value, safe_logits, grad_logits, "abkd")


def _support_safe_logsumexp_rows(log_terms: torch.Tensor) -> torch.Tensor:
    """Return ``-inf`` with zero gradient when a row has no finite term."""
    has_finite = torch.isfinite(log_terms).any(dim=-1)
    safe_terms = torch.where(
        has_finite.unsqueeze(-1), log_terms, torch.zeros_like(log_terms)
    )
    finite_lse = torch.logsumexp(safe_terms, dim=-1)
    return torch.where(
        has_finite,
        finite_lse,
        torch.full_like(finite_lse, -math.inf),
    )


def _support_safe_forward_kl_rows(
    log_q_mass: torch.Tensor,
    log_p_mass: torch.Tensor,
) -> torch.Tensor:
    """Normalize log-masses; missing student support returns gradient-free inf."""
    q_support = torch.isfinite(log_q_mass)
    p_support = torch.isfinite(log_p_mass)
    q_has_mass = q_support.any(dim=-1)
    missing_support = (q_support & ~p_support).any(dim=-1)
    active_row = q_has_mass & ~missing_support

    zero = torch.zeros_like(log_q_mass)
    safe_log_q_mass = torch.where(active_row.unsqueeze(-1), log_q_mass, zero)
    safe_log_p_mass = torch.where(active_row.unsqueeze(-1), log_p_mass, zero)
    log_q = safe_log_q_mass - torch.logsumexp(safe_log_q_mass, dim=-1).unsqueeze(-1)
    log_p = safe_log_p_mass - torch.logsumexp(safe_log_p_mass, dim=-1).unsqueeze(-1)
    q_positive = active_row.unsqueeze(-1) & q_support
    q = torch.where(q_positive, log_q.exp(), zero)
    safe_log_q = torch.where(q_positive, log_q, zero)
    safe_log_p = torch.where(q_positive, log_p, zero)
    finite_kl = _zero_preserving_mul(q, safe_log_q - safe_log_p).sum(dim=-1)
    return torch.where(
        missing_support,
        torch.full_like(finite_kl, math.inf),
        finite_kl,
    )


# ── KL family helpers ────────────────────────────────────────────────────────
def _mutual_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    log_p_t_target = log_p_t.detach()
    log_p_s_target = log_p_s.detach()
    return (
        _zero_preserving_mul(log_p_t_target.exp(), log_p_t_target - log_p_s)
        + _zero_preserving_mul(log_p_s_target.exp(), log_p_s_target - log_p_t)
    ).sum(dim=-1) * scale


def _cojeffreys_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return (
        _zero_preserving_mul(log_p_t.exp(), log_p_t - log_p_s)
        + _zero_preserving_mul(log_p_s.exp(), log_p_s - log_p_t)
    ).sum(dim=-1) * scale


def _alpha_divergence_rows(
    logits_s_scaled: torch.Tensor,
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    alpha: float,
    scale: float,
) -> torch.Tensor:
    zero = torch.zeros_like(log_p_s)
    if alpha <= 0.5:
        log_base = log_p_t
        log_other = log_p_s
        exponent = alpha
    else:
        log_base = log_p_s
        log_other = log_p_t
        exponent = 1.0 - alpha

    base_live = torch.isfinite(log_base)
    safe_base = torch.where(base_live, log_base, zero)
    safe_delta = torch.where(base_live, log_other - log_base, zero)
    correction = _zero_preserving_mul(
        torch.where(base_live, safe_base.exp(), zero),
        torch.expm1(exponent * safe_delta),
    ).sum(dim=-1)
    value = (-correction / (alpha * (1.0 - alpha))).clamp_min(0.0) * scale

    student_live = torch.isfinite(log_p_s)
    safe_log_p_s = torch.where(student_live, log_p_s, zero)
    p_s = torch.where(student_live, safe_log_p_s.exp(), zero)
    joint_live = student_live & torch.isfinite(log_p_t)
    safe_joint_s = torch.where(joint_live, log_p_s, zero)
    safe_joint_t = torch.where(joint_live, log_p_t, zero)
    ratio_delta = (1.0 - alpha) * (safe_joint_t - safe_joint_s)
    escort = torch.where(joint_live, (safe_joint_s + ratio_delta).exp(), zero)
    numerator = (1.0 + correction).unsqueeze(-1) * p_s - escort
    # Compensate only small log gaps; tails stay in escort log-space.
    ratio_live = joint_live & (ratio_delta.abs() <= 1.0e-2)
    safe_ratio_delta = torch.where(ratio_live, ratio_delta, zero)
    numerator = torch.where(
        ratio_live,
        p_s * (correction.unsqueeze(-1) - torch.expm1(safe_ratio_delta)),
        numerator,
    )
    grad_logits = torch.where(
        student_live,
        numerator * (scale / (1.0 - alpha)),
        zero,
    )
    safe_logits = torch.where(torch.isfinite(logits_s_scaled), logits_s_scaled, zero)
    return _first_order_proxy(value, safe_logits, grad_logits, "alpha_divergence")


def _skew_forward_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    beta: float,
    scale: float,
) -> torch.Tensor:
    log_m = torch.logaddexp(
        math.log1p(-beta) + log_p_t,
        math.log(beta) + log_p_s,
    )
    return _zero_preserving_mul(log_p_t.exp(), log_p_t - log_m).sum(dim=-1) * scale


def _skew_js_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    beta: float,
    scale: float,
) -> torch.Tensor:
    log_m = torch.logaddexp(
        math.log1p(-beta) + log_p_t,
        math.log(beta) + log_p_s,
    )
    return (
        (1.0 - beta) * _zero_preserving_mul(log_p_t.exp(), log_p_t - log_m).sum(dim=-1)
        + beta * _zero_preserving_mul(log_p_s.exp(), log_p_s - log_m).sum(dim=-1)
    ) * scale


def _skew_reverse_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    beta: float,
    scale: float,
) -> torch.Tensor:
    log_m = torch.logaddexp(
        math.log1p(-beta) + log_p_t,
        math.log(beta) + log_p_s,
    )
    return _zero_preserving_mul(log_p_s.exp(), log_p_s - log_m).sum(dim=-1) * scale


def _dkd_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    lambda_t: float,
    lambda_n: float,
    scale: float,
) -> torch.Tensor:
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
    label_index = safe_labels.unsqueeze(-1)
    log_p_y = log_p_s.gather(dim=-1, index=label_index).squeeze(-1)
    log_q_y = log_p_t.gather(dim=-1, index=label_index).squeeze(-1)

    columns = torch.arange(log_p_s.shape[-1], device=log_p_s.device)
    target_mask = columns.unsqueeze(0) == safe_labels.unsqueeze(-1)
    non_target_p = log_p_s.masked_fill(target_mask, -math.inf)
    non_target_q = log_p_t.masked_fill(target_mask, -math.inf)
    log_p_not_y = _support_safe_logsumexp_rows(non_target_p)
    log_q_not_y = _support_safe_logsumexp_rows(non_target_q)

    tckd = _support_safe_forward_kl_rows(
        torch.stack((log_q_y, log_q_not_y), dim=-1),
        torch.stack((log_p_y, log_p_not_y), dim=-1),
    )
    nckd = _support_safe_forward_kl_rows(non_target_q, non_target_p)
    unsafe = torch.isposinf(tckd) | torch.isposinf(nckd)
    finite = lambda_t * torch.where(unsafe, torch.zeros_like(tckd), tckd)
    finite = finite + lambda_n * torch.where(unsafe, torch.zeros_like(nckd), nckd)
    return torch.where(unsafe, torch.full_like(finite, math.inf), finite) * scale


def _triangular_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    p_s = log_p_s.exp()
    p_t = log_p_t.exp()
    denominator = p_s + p_t
    live = denominator > 0
    delta = torch.where(live, p_s - p_t, torch.zeros_like(p_s))
    safe_denominator = torch.where(live, denominator, torch.ones_like(denominator))
    return (delta.square() / safe_denominator).sum(dim=-1) * scale


def _entropy_adaptive_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    gate: EntropyGate,
    scale: float,
) -> torch.Tensor:
    p_t = log_p_t.exp()
    vocab = log_p_t.shape[-1]
    entropy = -_zero_preserving_mul(p_t, log_p_t).sum(dim=-1)
    normalized = entropy / math.log(vocab) if vocab > 1 else torch.zeros_like(entropy)
    gate_row = gate.alpha_min + (gate.alpha_max - gate.alpha_min) * torch.sigmoid(
        (normalized - gate.tau) / gate.scale
    )
    gate_col = gate_row.unsqueeze(-1)
    fkl = _zero_preserving_mul(gate_col * p_t, log_p_t - log_p_s).sum(dim=-1)
    rkl, unsafe = _support_safe_reverse_kl_rows(
        log_p_s,
        log_p_t,
        mass_scale=1.0 - gate_row,
    )
    finite = torch.where(unsafe, torch.zeros_like(fkl), fkl)
    finite = finite + torch.where(unsafe, torch.zeros_like(rkl), rkl)
    finite = finite * scale
    return torch.where(unsafe, torch.full_like(finite, math.inf), finite)


def _jeffreys_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    fkl = _zero_preserving_mul(log_p_t.exp(), log_p_t - log_p_s).sum(dim=-1)
    rkl, unsafe = _support_safe_reverse_kl_rows(log_p_s, log_p_t)
    finite = torch.where(unsafe, torch.zeros_like(fkl), fkl)
    finite = finite + torch.where(unsafe, torch.zeros_like(rkl), rkl)
    finite = finite * scale
    return torch.where(unsafe, torch.full_like(finite, math.inf), finite)


def _forward_rows(
    log_p_s: torch.Tensor,
    log_p_t: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return _zero_preserving_mul(log_p_t.exp(), log_p_t - log_p_s).sum(dim=-1) * scale


def _torch_reference(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
    student_ce_weight: float,
    teacher_ce_weight: float,
    kl_weight: float,
    kl_temperature: float,
    ignore_index: int,
    reduction: Reduction,
    kl_divergence: str = "forward",
    execution_plan: ExecutionPlan,
) -> DistillationLossOutput:
    # ── Plan and early exits ─────────────────────────────────
    mode, teacher_hidden, teacher_weight, teacher_logits = _teacher_mode_and_tensors(
        teacher
    )
    if execution_plan.backend != "torch" or execution_plan.teacher_mode != mode:
        raise RuntimeError("Torch backend received an incompatible execution plan")
    student_ce_weight = execution_plan.student_ce_weight
    teacher_ce_weight = execution_plan.teacher_ce_weight
    kl_weight = execution_plan.kl_weight
    kl_temperature = execution_plan.kl_temperature
    kl_teacher_temperature = execution_plan.kl_teacher_temperature
    ignore_index = execution_plan.ignore_index
    reduction = cast(Reduction, execution_plan.reduction)
    kl_divergence = execution_plan.specialization.kl_variant
    kl_alpha = execution_plan.kl_alpha
    kl_beta = execution_plan.kl_beta
    entropy_gate = execution_plan.entropy_gate
    if student_hidden.shape[0] == 0:
        return DistillationLossOutput(
            *_empty_distillation_tensors(
                student_hidden=student_hidden,
                weight=weight,
                mode=mode,
                teacher_hidden=teacher_hidden,
                teacher_weight=teacher_weight,
                student_ce_weight=student_ce_weight,
                kl_weight=kl_weight,
                teacher_grad_policy=execution_plan.teacher_grad_policy,
            )
        )
    compute_student_ce = student_ce_weight != 0.0
    compute_teacher_ce = teacher_ce_weight != 0.0
    compute_kl = kl_weight != 0.0

    # ── Chunk-free logits ────────────────────────────────────
    mask = labels != ignore_index
    denominator = torch.clamp(mask.sum(), min=1)

    # Mask before GEMM so ignored non-finite rows cannot poison gradients.
    row_active = mask.unsqueeze(-1)

    def _row_masked(tensor: torch.Tensor) -> torch.Tensor:
        return torch.where(row_active, tensor, tensor.new_zeros(()))

    needs_student_logits = compute_student_ce or compute_kl
    logits_s: torch.Tensor | None = None
    student_row_invalid: torch.Tensor | None = None
    if needs_student_logits:
        logits_s = _row_masked(student_hidden) @ weight.t()
        if logits_s.dtype in (torch.float16, torch.bfloat16):
            logits_s = logits_s.float()
        student_row_invalid = mask & (~torch.isfinite(logits_s)).any(dim=-1)

    needs_teacher_logits = compute_teacher_ce or compute_kl
    logits_t: torch.Tensor | None = None
    teacher_row_dead: torch.Tensor | None = None
    teacher_row_invalid: torch.Tensor | None = None
    if needs_teacher_logits:
        if mode == "tied":
            assert teacher_hidden is not None
            logits_t = _row_masked(teacher_hidden) @ weight.t()
        elif mode == "separate":
            assert teacher_hidden is not None and teacher_weight is not None
            logits_t = _row_masked(teacher_hidden) @ teacher_weight.t()
        elif teacher_logits is not None:
            logits_t = teacher_logits.detach()
            logits_t = cast_logits_for_compute(
                logits_t,
                execution_plan.compute_dtype,
            )
            logits_t = _row_masked(logits_t)
        else:
            assert teacher_hidden is not None and teacher_weight is not None
            logits_t = (_row_masked(teacher_hidden) @ teacher_weight.t()).detach()
        if logits_t.dtype in (torch.float16, torch.bfloat16):
            logits_t = logits_t.float()
        raw_cached_logits = mode == "precomputed" and teacher_logits is not None
        if raw_cached_logits:
            teacher_row_dead = mask & torch.isneginf(logits_t).all(dim=-1)
            invalid_values = torch.isnan(logits_t) | torch.isposinf(logits_t)
        else:
            teacher_row_dead = torch.zeros_like(mask)
            invalid_values = ~torch.isfinite(logits_t)
        teacher_row_invalid = mask & invalid_values.any(dim=-1)
        logits_t = torch.where(
            teacher_row_dead.unsqueeze(-1), logits_t.new_zeros(()), logits_t
        )
    # ── CE terms ─────────────────────────────────────────────
    zero = student_hidden.new_zeros(())
    if compute_student_ce:
        assert logits_s is not None
        ce_s_all = F.cross_entropy(
            logits_s,
            labels,
            ignore_index=ignore_index,
            reduction="none",
        )
        assert student_row_invalid is not None
        ce_s_all = _nan_visible_rows(ce_s_all, student_row_invalid, logits_s)
        student_ce = (
            ce_s_all.sum() / denominator if reduction == "mean" else ce_s_all.sum()
        )
    else:
        student_ce = zero

    if compute_teacher_ce:
        assert logits_t is not None
        ce_t_all = F.cross_entropy(
            logits_t,
            labels,
            ignore_index=ignore_index,
            reduction="none",
        )
        if teacher_row_dead is not None:
            # A neutralised dead row would score CE = log(V); pin it to zero
            # (kernel parity). It still counts in ``denominator``.
            ce_t_all = ce_t_all.masked_fill(teacher_row_dead, 0.0)
        if teacher_row_invalid is not None:
            ce_t_all = _nan_visible_rows(ce_t_all, teacher_row_invalid, logits_t)
        teacher_ce = (
            ce_t_all.sum() / denominator if reduction == "mean" else ce_t_all.sum()
        )
    else:
        teacher_ce = zero

    # ── KL per-variant reference ─────────────────────────────
    if compute_kl:
        assert logits_s is not None and logits_t is not None
        student_temperature = float(kl_temperature)
        teacher_temperature = float(kl_teacher_temperature)
        kl_loss_scale = student_temperature * student_temperature
        logits_s_scaled = logits_s / student_temperature
        log_p_s = F.log_softmax(logits_s_scaled, dim=-1)
        if kl_divergence == "mutual":
            log_p_t = F.log_softmax(logits_t / teacher_temperature, dim=-1)
            kl_all = _mutual_rows(log_p_s, log_p_t, kl_loss_scale)
        elif kl_divergence == "cojeffreys":
            log_p_t = F.log_softmax(logits_t / teacher_temperature, dim=-1)
            kl_all = _cojeffreys_rows(log_p_s, log_p_t, kl_loss_scale)
        elif kl_divergence == "reverse":
            # KL(p_s || p_t): student is the sampling distribution.
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            kl_all, _ = _support_safe_reverse_kl_rows(log_p_s, log_p_t)
            kl_all = kl_all * kl_loss_scale
        elif kl_divergence == "alpha_divergence":
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            kl_all = _alpha_divergence_rows(
                logits_s_scaled,
                log_p_s,
                log_p_t,
                float(kl_alpha),
                kl_loss_scale,
            )
        elif kl_divergence == "skew_forward":
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            kl_all = _skew_forward_rows(
                log_p_s,
                log_p_t,
                float(kl_beta),
                kl_loss_scale,
            )
        elif kl_divergence == "skew_js":
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            kl_all = _skew_js_rows(
                log_p_s,
                log_p_t,
                float(kl_beta),
                kl_loss_scale,
            )
        elif kl_divergence == "skew_reverse":
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            kl_all = _skew_reverse_rows(
                log_p_s,
                log_p_t,
                float(kl_beta),
                kl_loss_scale,
            )
        elif kl_divergence == "renyi":
            # Teacher-first D_r(q || p).
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            r = float(kl_alpha)
            kl_all = (
                _support_safe_renyi_rows(
                    logits_s_scaled,
                    log_p_s,
                    log_p_t,
                    r,
                    reverse=False,
                )
                * kl_loss_scale
            )
        elif kl_divergence == "renyi_reverse":
            # Student-first D_r(p || q).
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            r = float(kl_alpha)
            kl_all = (
                _support_safe_renyi_rows(
                    logits_s_scaled,
                    log_p_s,
                    log_p_t,
                    r,
                    reverse=True,
                )
                * kl_loss_scale
            )
        elif kl_divergence == "dkd":
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            kl_all = _dkd_rows(
                log_p_s,
                log_p_t,
                labels,
                mask,
                float(kl_alpha),
                float(kl_beta),
                kl_loss_scale,
            )
        elif kl_divergence == "abkd":
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            a = float(kl_alpha)
            b = float(kl_beta)
            kl_all = (
                _stable_abkd_rows(logits_s_scaled, log_p_s, log_p_t, a, b)
                * kl_loss_scale
            )
        elif kl_divergence == "triangular":
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            kl_all = _triangular_rows(log_p_s, log_p_t, kl_loss_scale)
        elif kl_divergence == "entropy_adaptive":
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            assert entropy_gate is not None
            kl_all = _entropy_adaptive_rows(
                log_p_s,
                log_p_t,
                entropy_gate,
                kl_loss_scale,
            )
        else:
            log_p_t = F.log_softmax(logits_t.detach() / teacher_temperature, dim=-1)
            if kl_divergence == "jeffreys":
                kl_all = _jeffreys_rows(log_p_s, log_p_t, kl_loss_scale)
            else:
                kl_all = _forward_rows(log_p_s, log_p_t, kl_loss_scale)
        kl_all = kl_all.masked_fill(~mask, 0.0)
        if teacher_row_dead is not None:
            # Pin a dead row's KL to zero (kernel parity).
            kl_all = kl_all.masked_fill(teacher_row_dead, 0.0)
        if student_row_invalid is not None:
            kl_all = _nan_visible_rows(kl_all, student_row_invalid, logits_s)
        if teacher_row_invalid is not None:
            anchors = (
                (logits_s, logits_t)
                if kl_divergence in TEACHER_KL_GRAD_VARIANTS
                else (logits_s,)
            )
            kl_all = _nan_visible_rows(kl_all, teacher_row_invalid, *anchors)
        kl = kl_all.sum() / denominator if reduction == "mean" else kl_all.sum()
    else:
        kl = zero

    loss = _combine_weighted_losses(
        student_ce,
        teacher_ce,
        kl,
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
    )
    output_dtype = _loss_output_dtype(loss.dtype)
    return DistillationLossOutput(
        loss.to(output_dtype),
        student_ce.detach().to(output_dtype),
        teacher_ce.detach().to(output_dtype),
        kl.detach().to(output_dtype),
    )
