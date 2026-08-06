from __future__ import annotations

from numbers import Integral

import torch

from ..._api_types import PrecomputedSpanTeacher
from ..._runtime import resolve_uniform_model_dtype
from ._aligned_span_math import span_log_probs_from_nll, span_valid_counts

_TARGET_LOGIT_ELEMENTS = 1 << 24


def build_precomputed_span_teacher(
    teacher_hidden: torch.Tensor,
    teacher_weight: torch.Tensor,
    teacher_labels: torch.Tensor,
    teacher_ranges: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> PrecomputedSpanTeacher:
    """Materialize detached teacher span log-probabilities in bounded chunks."""
    # ── Shape and dtype checks ───────────────────────────────
    if teacher_hidden.ndim != 2:
        raise ValueError(
            f"teacher_hidden must be 2D, got {tuple(teacher_hidden.shape)}"
        )
    if teacher_weight.ndim != 2 or teacher_weight.shape[1] != teacher_hidden.shape[1]:
        raise ValueError(
            f"teacher_weight must have shape (V, {teacher_hidden.shape[1]}), got "
            f"{tuple(teacher_weight.shape)}"
        )
    if teacher_labels.shape != (teacher_hidden.shape[0],):
        raise ValueError(
            f"teacher_labels must have shape ({teacher_hidden.shape[0]},), got "
            f"{tuple(teacher_labels.shape)}"
        )
    if teacher_labels.dtype != torch.long:
        raise ValueError(
            f"teacher_labels must have dtype torch.long, got {teacher_labels.dtype}"
        )
    if teacher_ranges.ndim != 2 or teacher_ranges.shape[1] != 2:
        raise ValueError(
            f"teacher_ranges must have shape (C, 2), got {tuple(teacher_ranges.shape)}"
        )
    if teacher_ranges.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"teacher_ranges must be int32 or int64, got {teacher_ranges.dtype}"
        )
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        raise ValueError(f"ignore_index must be an integer, got {ignore_index!r}")

    # ── Device and range checks ──────────────────────────────
    device = teacher_hidden.device
    for name, tensor in (
        ("teacher_weight", teacher_weight),
        ("teacher_labels", teacher_labels),
    ):
        if tensor.device != device:
            raise ValueError(
                f"{name} must be on the same device as teacher_hidden, got "
                f"{tensor.device} vs {device}"
            )
    resolve_uniform_model_dtype(
        ("teacher_hidden", teacher_hidden),
        ("teacher_weight", teacher_weight),
    )

    ranges = teacher_ranges.to(device=device, dtype=torch.int64).contiguous()
    rows = teacher_hidden.shape[0]
    vocab_size = teacher_weight.shape[0]
    if vocab_size < 1:
        raise ValueError("teacher_weight must contain at least one vocabulary row")
    if ranges.numel() != 0:
        starts = ranges[:, 0]
        ends = ranges[:, 1]
        if bool((starts < 0).any()) or bool((ends > rows).any()):
            raise ValueError(f"teacher_ranges must lie within [0, {rows}]")
        if bool((starts >= ends).any()):
            raise ValueError("teacher_ranges must satisfy start < end for every span")
    if teacher_labels.numel() != 0:
        valid = (teacher_labels == int(ignore_index)) | (
            (teacher_labels >= 0) & (teacher_labels < vocab_size)
        )
        if not bool(valid.all()):
            bad_values = teacher_labels[~valid][:5].tolist()
            raise ValueError(
                f"teacher_labels contains values outside [0, {vocab_size}) and not "
                f"equal to ignore_index={int(ignore_index)}: {bad_values}"
            )

    span_counts = span_valid_counts(teacher_labels != int(ignore_index), ranges)
    empty_spans = torch.nonzero(span_counts == 0, as_tuple=False).flatten()
    if empty_spans.numel() != 0:
        bad_indices = empty_spans[:5].tolist()
        raise ValueError(
            "teacher_ranges contains spans with no valid teacher tokens at "
            f"indices {bad_indices}"
        )

    # ── Chunked teacher projection ───────────────────────────
    compute_dtype = (
        torch.float64 if teacher_hidden.dtype == torch.float64 else torch.float32
    )
    token_nll = torch.empty(rows, dtype=compute_dtype, device=device)
    invalid_rows = torch.zeros(rows, dtype=torch.bool, device=device)
    chunk_rows = max(1, min(max(1, rows), _TARGET_LOGIT_ELEMENTS // vocab_size))
    weight = teacher_weight.detach().contiguous()
    hidden = teacher_hidden.detach()
    labels = teacher_labels.contiguous()

    with torch.no_grad():
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            labels_chunk = labels[start:end]
            active = labels_chunk != int(ignore_index)
            hidden_chunk = torch.where(
                active.unsqueeze(-1),
                hidden[start:end],
                hidden.new_zeros(()),
            )
            logits = hidden_chunk @ weight.t()
            invalid_rows[start:end] = active & (~torch.isfinite(logits)).any(dim=-1)
            compute_logits = logits.to(compute_dtype)
            safe_labels = torch.where(
                active,
                labels_chunk,
                labels_chunk.new_zeros(()),
            )
            nll = torch.logsumexp(compute_logits, dim=-1) - compute_logits.gather(
                1,
                safe_labels[:, None],
            ).squeeze(1)
            token_nll[start:end] = torch.where(active, nll, nll.new_zeros(()))

        if bool(invalid_rows.any()):
            bad_rows = torch.nonzero(invalid_rows, as_tuple=False).flatten()
            raise ValueError(
                "teacher GEMM produced non-finite logits on active rows "
                f"{bad_rows[:5].tolist()}"
            )
        span_log_probs = span_log_probs_from_nll(token_nll, ranges)
    return PrecomputedSpanTeacher(
        span_log_probs.detach(),
        teacher_ranges=ranges,
    )


__all__ = ["build_precomputed_span_teacher"]
