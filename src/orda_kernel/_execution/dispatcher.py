from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn, TypeVar

import torch

from .._planning.models import (
    ChunkAttempt,
    ExecutionPlan,
    KernelSpecialization,
)
from .._validation import _validate_label_range
from .chunking import _num_nonempty_chunks
from .recovery import (
    NonChunkableWorkspaceOOM,
    recover_after_oom,
)
from .recovery import (
    is_oom_error as _is_oom_error,
)


# ── Chunk helpers ────────────────────────────────────────────────────────────
def _chunk_size_from_num_chunks(BT: int, num_chunks: int) -> int:
    return (BT + num_chunks - 1) // num_chunks


def _chunk_attempt_from_num_chunks(BT: int, requested_num_chunks: int) -> ChunkAttempt:
    """Build a chunk attempt whose count exactly matches non-empty slices."""
    if BT <= 0 or requested_num_chunks <= 0:
        raise ValueError("BT and requested_num_chunks must be > 0")
    chunk_size = _chunk_size_from_num_chunks(BT, requested_num_chunks)
    return ChunkAttempt(
        chunk_size=chunk_size,
        num_chunks=_num_nonempty_chunks(BT, chunk_size),
    )


def _next_chunk_attempt(
    BT: int,
    attempt: ChunkAttempt,
    max_chunks_limit: int,
) -> ChunkAttempt | None:
    """Double the request, stopping when it cannot add a non-empty slice."""
    requested_num_chunks = min(
        attempt.num_chunks * 2,
        BT,
        max_chunks_limit,
    )
    next_attempt = _chunk_attempt_from_num_chunks(BT, requested_num_chunks)
    if next_attempt.num_chunks <= attempt.num_chunks:
        return None
    return next_attempt


def _raise_auto_chunk_oom(
    *,
    BT: int,
    V: int,
    H: int,
    attempt: ChunkAttempt,
    max_chunks_limit: int,
    mode: str,
    device: torch.device,
    cause: BaseException,
    oom_context: str | None = None,
) -> NoReturn:
    """Report an automatic-plan OOM in terms of ``num_chunks``.

    ``max_chunks`` only bounds the automatic modes, so it is deliberately absent
    from the remediation: an explicit ``num_chunks`` ignores the ceiling.
    """
    recover_after_oom(device)
    context = f" in {oom_context}" if oom_context else ""
    header = (
        f"Out of memory{context} with the {mode} chunk plan\n"
        f"BT={BT}, V={V}, H={H}\n"
        f"Used {attempt.num_chunks} chunks of {attempt.chunk_size} rows"
    )
    headroom = _next_chunk_attempt(BT, attempt, max_chunks_limit)
    if headroom is not None:
        raise RuntimeError(
            f"{header}.\n"
            "Options:\n"
            f"  - Set num_chunks={headroom.num_chunks}\n"
            "  - Reduce batch size or sequence length"
        ) from cause

    manual = _chunk_attempt_from_num_chunks(BT, min(attempt.num_chunks * 2, BT))
    if manual.num_chunks <= attempt.num_chunks:
        raise RuntimeError(
            f"{header}.\n"
            "One row per chunk leaves nothing further to split; reduce batch "
            "size or sequence length."
        ) from cause
    reduced = (1.0 - attempt.chunk_size / BT) * 100.0
    marginal = (attempt.chunk_size - manual.chunk_size) / BT * 100.0
    raise RuntimeError(
        f"{header} (chunkable buffer already reduced {reduced:.1f}%).\n"
        f"The {mode} plan will not split further on its own. Passing "
        f"num_chunks={manual.num_chunks} explicitly still works, but reclaims "
        f"only ~{marginal:.1f}% more while raising the launch count from "
        f"{attempt.num_chunks} to {manual.num_chunks}.\n"
        "Reducing batch size or sequence length is usually the better trade."
    ) from cause


def _recover_and_advance_chunk_attempt(
    *,
    BT: int,
    V: int,
    H: int,
    attempt: ChunkAttempt,
    max_chunks_limit: int,
    device: torch.device,
    cause: BaseException,
    oom_context: str | None = None,
) -> ChunkAttempt:
    """Recover an OOM and build the next non-empty chunk partition."""
    next_attempt = _next_chunk_attempt(BT, attempt, max_chunks_limit)
    if next_attempt is None:
        _raise_auto_chunk_oom(
            BT=BT,
            V=V,
            H=H,
            attempt=attempt,
            max_chunks_limit=max_chunks_limit,
            mode="dynamic",
            device=device,
            cause=cause,
            oom_context=oom_context,
        )
    recover_after_oom(device)
    return next_attempt


def _raise_fixed_workspace_oom(
    device: torch.device,
    exc: NonChunkableWorkspaceOOM,
) -> NoReturn:
    recover_after_oom(device)
    raise RuntimeError(
        f"Out of memory in a non-chunkable workspace ({exc}); "
        "splitting into more chunks cannot reduce this allocation. Reduce the "
        "model/token footprint or disable the affected objective."
    ) from exc


_ResultT = TypeVar("_ResultT")


def _execute_with_chunk_retry(
    initial_attempt: ChunkAttempt,
    *,
    apply_attempt: Callable[[ChunkAttempt], _ResultT],
    record_success: Callable[[ChunkAttempt], None],
    advance_after_oom: Callable[[ChunkAttempt, BaseException], ChunkAttempt],
    retryable_oom: Callable[[BaseException], bool] = _is_oom_error,
) -> _ResultT:
    """Run one dynamic-chunk loop with caller-owned execution and cache hooks."""
    attempt = initial_attempt
    while True:
        try:
            result = apply_attempt(attempt)
        except Exception as exc:
            if not retryable_oom(exc):
                raise
            attempt = advance_after_oom(attempt, exc)
            continue
        record_success(attempt)
        return result


def _merge_chunk_success(
    cached_baseline: int | None,
    current_cached: int | None,
    successful_num_chunks: int,
) -> int:
    if current_cached == cached_baseline:
        return successful_num_chunks
    return max(current_cached or 0, successful_num_chunks)


# ── Cache identity ───────────────────────────────────────────────────────────
# Tensor identity is deliberately excluded so equivalent workloads can reuse a
# learned chunk partition across newly allocated tensors.
@dataclass(frozen=True)
class TensorSignature:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    requires_grad: bool
    is_contiguous: bool


@dataclass(frozen=True)
class ChunkCacheKey:
    bt: int
    vocab_size: int
    student_hidden_size: int
    teacher_mode: str
    kernel_path: str
    specialization: KernelSpecialization
    max_fused_size: int
    max_chunks: int | None
    grad_weight_accumulation: str
    compute_dtype: torch.dtype
    student_hidden: TensorSignature
    weight: TensorSignature
    teacher_hidden: TensorSignature | None
    teacher_weight: TensorSignature | None
    teacher_logits: TensorSignature | None


def _tensor_sig(tensor: torch.Tensor | None) -> TensorSignature | None:
    if tensor is None:
        return None
    device = tensor.device
    return TensorSignature(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device_type=device.type,
        device_index=device.index,
        requires_grad=bool(tensor.requires_grad),
        is_contiguous=bool(tensor.is_contiguous()),
    )


def _cache_key_for_plan(
    BT: int,
    V: int,
    H: int,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    execution_plan: ExecutionPlan,
    *,
    teacher_hidden: torch.Tensor | None = None,
    teacher_weight: torch.Tensor | None = None,
    logits_teacher: torch.Tensor | None = None,
) -> ChunkCacheKey:
    student_sig = _tensor_sig(student_hidden)
    weight_sig = _tensor_sig(weight)
    assert student_sig is not None and weight_sig is not None
    teacher_path_active = (
        execution_plan.specialization.compute_teacher_ce
        or execution_plan.specialization.compute_kl
    )
    return ChunkCacheKey(
        bt=int(BT),
        vocab_size=int(V),
        student_hidden_size=int(H),
        teacher_mode=execution_plan.teacher_mode,
        kernel_path=execution_plan.kernel_path,
        specialization=execution_plan.specialization,
        max_fused_size=execution_plan.max_fused_size,
        max_chunks=execution_plan.chunk_policy.max_chunks_limit,
        grad_weight_accumulation=execution_plan.grad_weight_accumulation,
        compute_dtype=execution_plan.compute_dtype,
        student_hidden=student_sig,
        weight=weight_sig,
        teacher_hidden=_tensor_sig(teacher_hidden) if teacher_path_active else None,
        teacher_weight=_tensor_sig(teacher_weight) if teacher_path_active else None,
        teacher_logits=_tensor_sig(logits_teacher) if teacher_path_active else None,
    )


# ── Chunk cache and API ──────────────────────────────────────────────────────
_CACHE_LOCK = threading.Lock()
_MAX_CHUNK_CACHE_SIZE = 1024
_cached_num_chunks: OrderedDict[ChunkCacheKey, int] = OrderedDict()


def clear_chunk_cache():
    with _CACHE_LOCK:
        _cached_num_chunks.clear()
    from .aligned_span import clear_aligned_span_chunk_cache

    clear_aligned_span_chunk_cache()


def get_chunk_cache() -> dict:
    with _CACHE_LOCK:
        tokenwise = dict(_cached_num_chunks)
    from .aligned_span import get_aligned_span_chunk_cache

    return {**tokenwise, **get_aligned_span_chunk_cache()}


# ── Kernel dispatch ──────────────────────────────────────────────────────────
def _resolve_plan_executor(teacher_mode: str):
    if teacher_mode == "tied":
        from ..ops._kernels.tied import _apply_tied_execution_plan

        return _apply_tied_execution_plan
    if teacher_mode == "separate":
        from ..ops._kernels.separate import _apply_separate_execution_plan

        return _apply_separate_execution_plan
    if teacher_mode == "precomputed":
        from ..ops._kernels.precomputed import _apply_precomputed_execution_plan

        return _apply_precomputed_execution_plan
    raise ValueError(
        f"teacher_mode must be 'tied', 'separate', or 'precomputed', got {teacher_mode!r}"
    )


def _call_plan_executor(
    plan_executor,
    execution_plan: ExecutionPlan,
    attempt: ChunkAttempt,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    teacher_weight: torch.Tensor | None,
    logits_teacher: torch.Tensor | None,
    gradient_materialization_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if execution_plan.teacher_mode == "tied":
        if teacher_hidden is None:
            raise RuntimeError("tied execution requires teacher_hidden")
        return plan_executor(
            student_hidden,
            teacher_hidden,
            weight,
            labels,
            gradient_materialization_scale,
            execution_plan,
            attempt,
        )
    if execution_plan.teacher_mode == "separate":
        if teacher_hidden is None or teacher_weight is None:
            raise RuntimeError("separate execution requires teacher tensors")
        return plan_executor(
            student_hidden,
            teacher_hidden,
            weight,
            teacher_weight,
            labels,
            gradient_materialization_scale,
            execution_plan,
            attempt,
        )
    return plan_executor(
        student_hidden,
        weight,
        logits_teacher,
        labels,
        teacher_hidden,
        teacher_weight,
        gradient_materialization_scale,
        execution_plan,
        attempt,
    )


# ── Plan executor ────────────────────────────────────────────────────────────
@torch._dynamo.disable
def execute_plan(
    execution_plan: ExecutionPlan,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    teacher_weight: torch.Tensor | None = None,
    logits_teacher: torch.Tensor | None = None,
    gradient_materialization_scale: torch.Tensor | None = None,
    _labels_prevalidated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute a validated Triton plan without re-resolving policy or config."""
    if execution_plan.backend != "triton":
        raise RuntimeError("execution dispatcher requires a Triton execution plan")

    BT, H = student_hidden.shape
    V = weight.shape[0]
    if not _labels_prevalidated:
        _validate_label_range(
            labels,
            V,
            execution_plan.ignore_index,
            assume_valid_labels=execution_plan.assume_valid_labels,
        )
    plan_executor = _resolve_plan_executor(execution_plan.teacher_mode)
    chunk_policy = execution_plan.chunk_policy

    if not chunk_policy.dynamic:
        try:
            return _call_plan_executor(
                plan_executor,
                execution_plan,
                ChunkAttempt(
                    chunk_size=chunk_policy.initial_chunk_size,
                    num_chunks=chunk_policy.initial_num_chunks,
                ),
                student_hidden,
                teacher_hidden,
                weight,
                labels,
                teacher_weight=teacher_weight,
                logits_teacher=logits_teacher,
                gradient_materialization_scale=gradient_materialization_scale,
            )
        except NonChunkableWorkspaceOOM as exc:
            _raise_fixed_workspace_oom(student_hidden.device, exc)
        except Exception as exc:
            if not (chunk_policy.fast and _is_oom_error(exc)):
                raise
            _raise_auto_chunk_oom(
                BT=BT,
                V=V,
                H=H,
                attempt=ChunkAttempt(
                    chunk_size=chunk_policy.initial_chunk_size,
                    num_chunks=chunk_policy.initial_num_chunks,
                ),
                max_chunks_limit=chunk_policy.max_chunks_limit,
                mode="fast",
                device=student_hidden.device,
                cause=exc,
            )

    # ── Dynamic chunk cache ──────────────────────────────────
    key = _cache_key_for_plan(
        BT,
        V,
        H,
        student_hidden,
        weight,
        execution_plan,
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        logits_teacher=logits_teacher,
    )
    with _CACHE_LOCK:
        cached_num_chunks = _cached_num_chunks.get(key)
        if cached_num_chunks is not None:
            _cached_num_chunks.move_to_end(key)
    requested_num_chunks = max(
        cached_num_chunks or chunk_policy.initial_num_chunks,
        chunk_policy.initial_num_chunks,
    )
    requested_num_chunks = min(
        requested_num_chunks,
        BT,
        chunk_policy.max_chunks_limit,
    )
    attempt = _chunk_attempt_from_num_chunks(BT, requested_num_chunks)

    # ── OOM retry loop ───────────────────────────────────────
    def apply_attempt(current_attempt: ChunkAttempt):
        return _call_plan_executor(
            plan_executor,
            execution_plan,
            current_attempt,
            student_hidden,
            teacher_hidden,
            weight,
            labels,
            teacher_weight=teacher_weight,
            logits_teacher=logits_teacher,
            gradient_materialization_scale=gradient_materialization_scale,
        )

    def record_success(current_attempt: ChunkAttempt) -> None:
        with _CACHE_LOCK:
            current_cached = _cached_num_chunks.get(key)
            _cached_num_chunks[key] = _merge_chunk_success(
                cached_num_chunks,
                current_cached,
                current_attempt.num_chunks,
            )
            _cached_num_chunks.move_to_end(key)
            while len(_cached_num_chunks) > _MAX_CHUNK_CACHE_SIZE:
                _cached_num_chunks.popitem(last=False)

    def advance_after_oom(
        current_attempt: ChunkAttempt,
        exc: BaseException,
    ) -> ChunkAttempt:
        return _recover_and_advance_chunk_attempt(
            BT=BT,
            V=V,
            H=H,
            attempt=current_attempt,
            max_chunks_limit=chunk_policy.max_chunks_limit,
            device=student_hidden.device,
            cause=exc,
        )

    try:
        return _execute_with_chunk_retry(
            attempt,
            apply_attempt=apply_attempt,
            record_success=record_success,
            advance_after_oom=advance_after_oom,
        )
    except NonChunkableWorkspaceOOM as exc:
        _raise_fixed_workspace_oom(student_hidden.device, exc)
