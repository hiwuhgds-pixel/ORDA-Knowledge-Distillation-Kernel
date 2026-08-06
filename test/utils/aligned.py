from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ── Span likelihoods ─────────────────────────────────────────────────────────
def _log1mexp(value: torch.Tensor) -> torch.Tensor:
    split = -math.log(2.0)
    return torch.where(
        value < split,
        torch.log1p(-value.exp()),
        torch.log(-torch.expm1(value)),
    )


def span_log_likelihoods(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ranges: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    mask = labels != ignore_index
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
    log_probs = F.log_softmax(hidden @ weight.t(), dim=-1)
    token_ll = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    token_ll = torch.where(mask, token_ll, torch.zeros_like(token_ll))
    prefix = torch.cat((token_ll.new_zeros(1), token_ll.cumsum(dim=0)))
    starts = ranges[:, 0].to(dtype=torch.long)
    ends = ranges[:, 1].to(dtype=torch.long)
    return prefix[ends] - prefix[starts]


def binary_span_rows(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    teacher_probability = teacher_log_prob.exp()
    complement = 1.0 - teacher_probability
    student_complement = _log1mexp(student_log_prob)
    if objective == "binary_kl":
        teacher_complement = _log1mexp(teacher_log_prob)
        first = torch.where(
            teacher_probability > 0,
            teacher_probability * (teacher_log_prob - student_log_prob),
            torch.zeros_like(student_log_prob),
        )
        second = torch.where(
            complement > 0,
            complement * (teacher_complement - student_complement),
            torch.zeros_like(student_log_prob),
        )
        return first + second
    if objective == "binary_ce":
        return -(
            teacher_probability * student_log_prob + complement * student_complement
        )
    raise ValueError(f"unsupported aligned-span objective: {objective!r}")


# ── Aligned loss oracle ──────────────────────────────────────────────────────
def aligned_span_reference_loss(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    labels: torch.Tensor,
    span_ranges: torch.Tensor,
    teacher_span_log_probs: torch.Tensor,
    *,
    objective: str,
    student_ce_weight: float = 1.0,
    span_kl_weight: float = 1.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = labels != ignore_index
    denominator = mask.sum().clamp_min(1)
    student_ce = (
        F.cross_entropy(
            student_hidden @ student_weight.t(),
            labels,
            ignore_index=ignore_index,
            reduction="sum",
        )
        / denominator
    )
    student_span = span_log_likelihoods(
        student_hidden,
        student_weight,
        labels,
        span_ranges,
        ignore_index=ignore_index,
    )
    span_rows = binary_span_rows(student_span, teacher_span_log_probs, objective)
    span_loss = span_rows.mean() if span_rows.numel() else span_rows.sum()
    total = student_ce_weight * student_ce + span_kl_weight * span_loss
    return total, student_ce, span_loss


__all__ = [
    "aligned_span_reference_loss",
    "binary_span_rows",
    "span_log_likelihoods",
]
