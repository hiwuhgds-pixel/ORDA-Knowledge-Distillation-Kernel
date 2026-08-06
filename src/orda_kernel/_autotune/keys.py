from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ._device import (
    _backend_name,
    _runtime_version,
    _triton_version,
    device_identity,
)
from ._shapes import _max_block_size
from .models import (
    AUTOTUNE_TIMED_CONTRACT_ID,
    CANONICAL_INACTIVE_KL_VARIANT,
    PRECOMPUTED_ALIGNED_SPAN,
    AutotuneRuntimeKey,
    AutotuneWorkload,
    DeviceFingerprint,
)


# ── Device and workload construction ─────────────────────────────────────────
def _device_fingerprint(device: torch.device) -> DeviceFingerprint:
    if device.type != "cuda":
        return DeviceFingerprint(
            device_type=device.type,
            device_index=device.index,
            backend=_backend_name(),
            runtime_version=_runtime_version(),
            torch_version=str(torch.__version__),
            triton_version=_triton_version(),
        )

    index = torch.cuda.current_device() if device.index is None else int(device.index)
    return device_identity(index).fingerprint


def _effective_cap(vocab_size: int, max_fused_size: int) -> int:
    """Return the only max-fused value that can affect a launch candidate."""
    return _max_block_size(vocab_size, max_fused_size)


@dataclass(frozen=True)
class _NormalizedFields:
    """Canonicalized workload fields shared by the runtime key and workload."""

    effective_compute_kl: bool
    effective_t_is_one: bool
    effective_kl_variant: str
    effective_chunk_rows: int
    effective_cap: int
    effective_pipeline_key: tuple[str, ...]


# ── Workload normalization ───────────────────────────────────────────────────
def _normalize_workload_fields(
    *,
    mode: str,
    bt: int,
    vocab_size: int,
    max_fused_size: int,
    compute_kl: bool,
    t_is_one: bool,
    kl_variant: str,
    chunk_rows: int | None,
    pipeline_key: Iterable[str],
) -> _NormalizedFields:
    """Canonicalize the fields that both key builders derive identically."""
    effective_compute_kl = bool(compute_kl)
    effective_t_is_one = True if not effective_compute_kl else bool(t_is_one)
    effective_kl_variant = (
        str(kl_variant) if effective_compute_kl else CANONICAL_INACTIVE_KL_VARIANT
    )
    effective_chunk_rows = int(bt) if chunk_rows is None else int(chunk_rows)
    pipeline_active = effective_compute_kl or mode == PRECOMPUTED_ALIGNED_SPAN
    effective_pipeline_key = (
        tuple(str(item) for item in pipeline_key) if pipeline_active else ()
    )
    return _NormalizedFields(
        effective_compute_kl=effective_compute_kl,
        effective_t_is_one=effective_t_is_one,
        effective_kl_variant=effective_kl_variant,
        effective_chunk_rows=effective_chunk_rows,
        effective_cap=_effective_cap(vocab_size, max_fused_size),
        effective_pipeline_key=effective_pipeline_key,
    )


# ── Key construction ─────────────────────────────────────────────────────────
def _runtime_key(
    *,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    bt: int,
    max_fused_size: int,
    shape_key: Iterable[int],
    compute_kl: bool,
    t_is_one: bool,
    compute_teacher_ce: bool,
    compute_student_ce: bool,
    kl_variant: str,
    teacher_grad_policy: str,
    grad_student_hidden: bool,
    grad_teacher_hidden: bool,
    grad_weight_accum_dtype: str | None,
    grad_teacher_weight_accum_dtype: str | None,
    gradient_materialization_scaled: bool = False,
    chunk_rows: int | None = None,
    chunk_dynamic: bool = False,
    pipeline_key: Iterable[str] = (),
) -> AutotuneRuntimeKey:
    """Build the hot-cache key without CUDA property probes or JSON encoding."""
    normalized = _normalize_workload_fields(
        mode=mode,
        bt=bt,
        vocab_size=vocab_size,
        max_fused_size=max_fused_size,
        compute_kl=compute_kl,
        t_is_one=t_is_one,
        kl_variant=kl_variant,
        chunk_rows=chunk_rows,
        pipeline_key=pipeline_key,
    )
    effective_shape_key = tuple(int(item) for item in shape_key)

    normalized_device = torch.device(device)
    device_index = normalized_device.index
    if normalized_device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return AutotuneRuntimeKey(
        mode=str(mode),
        device_type=normalized_device.type,
        device_index=device_index,
        dtype=str(dtype),
        vocab_size=int(vocab_size),
        bt=int(bt),
        effective_cap=normalized.effective_cap,
        shape_key=effective_shape_key,
        compute_kl=normalized.effective_compute_kl,
        t_is_one=normalized.effective_t_is_one,
        compute_teacher_ce=bool(compute_teacher_ce),
        compute_student_ce=bool(compute_student_ce),
        kl_variant=normalized.effective_kl_variant,
        teacher_grad_policy=str(teacher_grad_policy),
        grad_student_hidden=bool(grad_student_hidden),
        grad_teacher_hidden=bool(grad_teacher_hidden),
        grad_weight_accum_dtype=grad_weight_accum_dtype,
        grad_teacher_weight_accum_dtype=grad_teacher_weight_accum_dtype,
        gradient_materialization_scaled=bool(gradient_materialization_scaled),
        chunk_rows=normalized.effective_chunk_rows,
        chunk_dynamic=bool(chunk_dynamic),
        timed_contract_id=AUTOTUNE_TIMED_CONTRACT_ID,
        pipeline_key=normalized.effective_pipeline_key,
    )


def _workload(
    *,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    bt: int,
    max_fused_size: int,
    shape_key: Iterable[int],
    compute_kl: bool,
    t_is_one: bool,
    compute_teacher_ce: bool,
    compute_student_ce: bool,
    kl_variant: str,
    teacher_grad_policy: str,
    grad_student_hidden: bool,
    grad_teacher_hidden: bool,
    grad_weight_accum_dtype: str | None,
    grad_teacher_weight_accum_dtype: str | None,
    gradient_materialization_scaled: bool = False,
    chunk_rows: int | None = None,
    chunk_dynamic: bool = False,
    pipeline_key: Iterable[str] = (),
) -> AutotuneWorkload:
    normalized = _normalize_workload_fields(
        mode=mode,
        bt=bt,
        vocab_size=vocab_size,
        max_fused_size=max_fused_size,
        compute_kl=compute_kl,
        t_is_one=t_is_one,
        kl_variant=kl_variant,
        chunk_rows=chunk_rows,
        pipeline_key=pipeline_key,
    )
    return AutotuneWorkload(
        mode=str(mode),
        device=_device_fingerprint(device),
        dtype=str(dtype),
        vocab_size=int(vocab_size),
        bt=int(bt),
        effective_cap=normalized.effective_cap,
        shape_key=tuple(int(item) for item in shape_key),
        compute_kl=normalized.effective_compute_kl,
        t_is_one=normalized.effective_t_is_one,
        compute_teacher_ce=bool(compute_teacher_ce),
        compute_student_ce=bool(compute_student_ce),
        kl_variant=normalized.effective_kl_variant,
        teacher_grad_policy=str(teacher_grad_policy),
        grad_student_hidden=bool(grad_student_hidden),
        grad_teacher_hidden=bool(grad_teacher_hidden),
        grad_weight_accum_dtype=grad_weight_accum_dtype,
        grad_teacher_weight_accum_dtype=grad_teacher_weight_accum_dtype,
        gradient_materialization_scaled=bool(gradient_materialization_scaled),
        chunk_rows=normalized.effective_chunk_rows,
        chunk_dynamic=bool(chunk_dynamic),
        pipeline_key=normalized.effective_pipeline_key,
        timed_contract_id=AUTOTUNE_TIMED_CONTRACT_ID,
    )
