from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import torch

from .._planning.models import AlignedSpanExecutionPlan, ChunkAttempt
from .._validation import _validate_label_range
from ..ops._kernels.precomputed.aligned_span import (
    PrecomputedAlignedSpanFunction,
)
from .dispatcher import (
    TensorSignature,
    _chunk_attempt_from_num_chunks,
    _execute_with_chunk_retry,
    _merge_chunk_success,
    _raise_auto_chunk_oom,
    _raise_fixed_workspace_oom,
    _recover_and_advance_chunk_attempt,
    _tensor_sig,
)
from .recovery import NonChunkableWorkspaceOOM, is_oom_error


# ── Chunk cache state ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AlignedSpanChunkCacheKey:
    bt: int
    vocab_size: int
    hidden_size: int
    num_spans: int
    kernel_path: str
    compute_student_ce: bool
    compute_span: bool
    objective: str
    reduction: str
    max_fused_size: int
    max_chunks: int | None
    grad_weight_accumulation: str
    compute_dtype: torch.dtype
    student_hidden: TensorSignature
    weight: TensorSignature
    teacher_span_log_probs: TensorSignature
    student_ranges: TensorSignature


_CACHE_LOCK = threading.Lock()
_MAX_CACHE_SIZE = 1024
_cached_num_chunks: OrderedDict[AlignedSpanChunkCacheKey, int] = OrderedDict()


# ── Execution helpers ────────────────────────────────────────────────────────
def clear_aligned_span_chunk_cache() -> None:
    with _CACHE_LOCK:
        _cached_num_chunks.clear()


def get_aligned_span_chunk_cache() -> dict[AlignedSpanChunkCacheKey, int]:
    with _CACHE_LOCK:
        return dict(_cached_num_chunks)


def _cache_key(
    plan: AlignedSpanExecutionPlan,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    teacher_span_log_probs: torch.Tensor,
    student_ranges: torch.Tensor,
) -> AlignedSpanChunkCacheKey:
    hidden_sig = _tensor_sig(student_hidden)
    weight_sig = _tensor_sig(weight)
    teacher_sig = _tensor_sig(teacher_span_log_probs)
    ranges_sig = _tensor_sig(student_ranges)
    assert hidden_sig is not None
    assert weight_sig is not None
    assert teacher_sig is not None
    assert ranges_sig is not None
    return AlignedSpanChunkCacheKey(
        bt=student_hidden.shape[0],
        vocab_size=weight.shape[0],
        hidden_size=student_hidden.shape[1],
        num_spans=student_ranges.shape[0],
        kernel_path=plan.kernel_path,
        compute_student_ce=plan.specialization.compute_student_ce,
        compute_span=plan.specialization.compute_span,
        objective=plan.objective,
        reduction=plan.reduction,
        max_fused_size=plan.max_fused_size,
        max_chunks=plan.chunk_policy.max_chunks,
        grad_weight_accumulation=plan.grad_weight_accumulation,
        compute_dtype=plan.compute_dtype,
        student_hidden=hidden_sig,
        weight=weight_sig,
        teacher_span_log_probs=teacher_sig,
        student_ranges=ranges_sig,
    )


def _apply(
    plan: AlignedSpanExecutionPlan,
    attempt: ChunkAttempt,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher_span_log_probs: torch.Tensor,
    student_ranges: torch.Tensor,
    gradient_materialization_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return PrecomputedAlignedSpanFunction.apply(
        student_hidden,
        weight,
        labels,
        teacher_span_log_probs,
        student_ranges,
        gradient_materialization_scale,
        plan,
        attempt,
    )


@torch._dynamo.disable
def execute_aligned_span_plan(
    plan: AlignedSpanExecutionPlan,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher_span_log_probs: torch.Tensor,
    student_ranges: torch.Tensor,
    *,
    gradient_materialization_scale: torch.Tensor | None = None,
    _labels_prevalidated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # ── Plan and input setup ─────────────────────────────────
    if plan.backend != "triton":
        raise RuntimeError("aligned-span executor requires a Triton plan")

    bt, hidden_size = student_hidden.shape
    vocab_size = weight.shape[0]
    if not _labels_prevalidated:
        _validate_label_range(
            labels,
            vocab_size,
            plan.ignore_index,
            assume_valid_labels=plan.assume_valid_labels,
        )
    policy = plan.chunk_policy
    initial = ChunkAttempt(
        chunk_size=policy.initial_chunk_size,
        num_chunks=policy.initial_num_chunks,
    )
    if bt == 0 or not policy.dynamic:
        try:
            return _apply(
                plan,
                initial,
                student_hidden,
                weight,
                labels,
                teacher_span_log_probs,
                student_ranges,
                gradient_materialization_scale,
            )
        except NonChunkableWorkspaceOOM as exc:
            _raise_fixed_workspace_oom(student_hidden.device, exc)
        except Exception as exc:
            if not (policy.fast and is_oom_error(exc)):
                raise
            _raise_auto_chunk_oom(
                BT=bt,
                V=vocab_size,
                H=hidden_size,
                attempt=initial,
                max_chunks_limit=policy.max_chunks_limit,
                mode="fast",
                device=student_hidden.device,
                cause=exc,
                oom_context="aligned-span chunk-vocab workspace",
            )

    # ── Cached retry plan ────────────────────────────────────
    key = _cache_key(
        plan,
        student_hidden,
        weight,
        teacher_span_log_probs,
        student_ranges,
    )
    with _CACHE_LOCK:
        cached = _cached_num_chunks.get(key)
        if cached is not None:
            _cached_num_chunks.move_to_end(key)
    requested = max(cached or policy.initial_num_chunks, policy.initial_num_chunks)
    requested = min(requested, bt, policy.max_chunks_limit)
    attempt = _chunk_attempt_from_num_chunks(bt, requested)

    def apply_attempt(current_attempt: ChunkAttempt):
        return _apply(
            plan,
            current_attempt,
            student_hidden,
            weight,
            labels,
            teacher_span_log_probs,
            student_ranges,
            gradient_materialization_scale,
        )

    def record_success(current_attempt: ChunkAttempt) -> None:
        with _CACHE_LOCK:
            current = _cached_num_chunks.get(key)
            _cached_num_chunks[key] = _merge_chunk_success(
                cached,
                current,
                current_attempt.num_chunks,
            )
            _cached_num_chunks.move_to_end(key)
            while len(_cached_num_chunks) > _MAX_CACHE_SIZE:
                _cached_num_chunks.popitem(last=False)

    def advance_after_oom(
        current_attempt: ChunkAttempt,
        exc: BaseException,
    ) -> ChunkAttempt:
        return _recover_and_advance_chunk_attempt(
            BT=bt,
            V=vocab_size,
            H=hidden_size,
            attempt=current_attempt,
            max_chunks_limit=policy.max_chunks_limit,
            device=student_hidden.device,
            cause=exc,
            oom_context="aligned-span chunk-vocab workspace",
        )

    # ── Retry execution ──────────────────────────────────────
    try:
        return _execute_with_chunk_retry(
            attempt,
            apply_attempt=apply_attempt,
            record_success=record_success,
            advance_after_oom=advance_after_oom,
            retryable_oom=lambda exc: (
                not isinstance(exc, NonChunkableWorkspaceOOM) and is_oom_error(exc)
            ),
        )
    except NonChunkableWorkspaceOOM as exc:
        _raise_fixed_workspace_oom(student_hidden.device, exc)


__all__ = [
    "clear_aligned_span_chunk_cache",
    "execute_aligned_span_plan",
    "get_aligned_span_chunk_cache",
]
