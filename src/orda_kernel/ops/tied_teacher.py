"""Public tied-teacher low-level API."""

from __future__ import annotations

import torch

from .._api_types import (
    _DEFAULT_CHUNK_SIZE,
    AutotuneMode,
    ChunkSize,
    EntropyGate,
    GradWeightAccumulation,
)
from .._execution.chunking import (
    DEFAULT_MAX_FUSED_SIZE,
    normalize_chunk_request,
    normalize_max_fused_size,
)
from .._execution.dispatcher import execute_plan
from .._planning.models import VALID_KL_VARIANTS
from .._planning.policy import build_execution_plan
from .._runtime import HAS_TRITON, triton_unavailable_reason
from .._torch_backend import _empty_distillation_tensors
from .._validation import (
    _normalize_bool,
    _normalize_execution_flags,
    _normalize_ignore_index,
    _normalize_loss_scalars,
    _validate_gradient_materialization_scale,
    _validate_label_range,
    _validate_loss_weights_and_reduction,
    _validate_model_dtypes,
    _validate_triton_model_dtype,
    _validate_vocab_size,
)

# ── Public API ───────────────────────────────────────────────────────────────
__all__ = ["tied_distillation_loss"]


@torch._dynamo.disable
def tied_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    weight: torch.Tensor,
    labels: torch.Tensor,
    student_ce_weight: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    chunk_size: ChunkSize = _DEFAULT_CHUNK_SIZE,
    grad_weight_accumulation: GradWeightAccumulation | None = None,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    teacher_ce_weight: float | None = None,
    max_chunks: int | None = None,
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    autotune: AutotuneMode = False,
    *,
    assume_valid_labels: bool = False,
    kl_divergence: str = "forward",
    num_chunks: int | None = None,
    kl_alpha: float | None = None,
    kl_beta: float | None = None,
    entropy_gate: EntropyGate | None = None,
    kl_teacher_temperature: float | None = None,
    gradient_materialization_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # ── Argument normalization ───────────────────────────────
    assume_valid_labels = _normalize_bool("assume_valid_labels", assume_valid_labels)
    (
        student_ce_weight,
        teacher_ce_weight,
        kl_weight,
        kl_temperature,
        kl_teacher_temperature,
    ) = _normalize_loss_scalars(
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
        kl_temperature=kl_temperature,
        kl_teacher_temperature=kl_teacher_temperature,
    )
    ignore_index = _normalize_ignore_index(ignore_index)
    autotune, grad_weight_accumulation = _normalize_execution_flags(
        autotune=autotune,
        grad_weight_accumulation=grad_weight_accumulation,
    )
    effective_teacher_ce_weight = (
        1.0 if teacher_ce_weight is None else teacher_ce_weight
    )
    chunk_size, num_chunks, max_chunks = normalize_chunk_request(
        chunk_size,
        num_chunks=num_chunks,
        max_chunks=max_chunks,
    )
    max_fused_size = normalize_max_fused_size(max_fused_size)
    if kl_divergence not in VALID_KL_VARIANTS:
        raise ValueError(
            f"kl_divergence must be one of {sorted(VALID_KL_VARIANTS)}, "
            f"got {kl_divergence!r}"
        )
    # ── Tensor validation ────────────────────────────────────
    if student_hidden.ndim != 2:
        raise ValueError(
            f"student_hidden must be 2D, got shape={tuple(student_hidden.shape)}"
        )
    BT, H = student_hidden.shape
    device = student_hidden.device
    _validate_gradient_materialization_scale(
        gradient_materialization_scale,
        device,
    )
    if teacher_hidden is None:
        raise ValueError("tied_distillation_loss requires teacher_hidden")
    if teacher_hidden.shape != (BT, H):
        raise ValueError(
            f"teacher_hidden.shape={tuple(teacher_hidden.shape)} does not match "
            f"student_hidden.shape={(BT, H)}"
        )
    if weight.ndim != 2 or weight.shape[1] != H:
        raise ValueError(f"weight must have shape (V, {H}), got {tuple(weight.shape)}")
    V = weight.shape[0]
    _validate_vocab_size(V)
    if labels.shape != (BT,):
        raise ValueError(f"labels must have shape ({BT},), got {tuple(labels.shape)}")
    if labels.dtype != torch.long:
        raise ValueError(f"labels must have dtype torch.long, got {labels.dtype}")
    _validate_loss_weights_and_reduction(
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=effective_teacher_ce_weight,
        kl_weight=kl_weight,
        reduction=reduction,
    )
    teacher_path_active = effective_teacher_ce_weight != 0.0 or kl_weight != 0.0
    model_dtype = _validate_model_dtypes(
        student_hidden=student_hidden,
        weight=weight,
        teacher_hidden=teacher_hidden if teacher_path_active else None,
    )
    # ── Plan and dispatch ────────────────────────────────────
    execution_plan = build_execution_plan(
        backend="triton",
        teacher_mode="tied",
        compute_dtype=model_dtype,
        bt=BT,
        vocab_size=V,
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=effective_teacher_ce_weight,
        kl_weight=kl_weight,
        kl_temperature=kl_temperature,
        kl_teacher_temperature=kl_teacher_temperature,
        kl_variant=kl_divergence,
        ignore_index=ignore_index,
        reduction=reduction,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        max_chunks=max_chunks,
        max_fused_size=max_fused_size,
        autotune=autotune,
        grad_weight_accumulation=grad_weight_accumulation,
        assume_valid_labels=assume_valid_labels,
        kl_alpha=kl_alpha,
        kl_beta=kl_beta,
        entropy_gate=entropy_gate,
    )
    _validate_triton_model_dtype(model_dtype, device)
    _validate_label_range(
        labels,
        V,
        ignore_index,
        assume_valid_labels=assume_valid_labels,
    )
    if not HAS_TRITON:
        raise RuntimeError(
            "tied_distillation_loss requires Triton and CUDA/HIP tensors "
            f"({triton_unavailable_reason()}). "
            "Use distillation_loss(..., backend='auto' or 'torch') for fallback execution."
        )
    if device.type != "cuda":
        raise ValueError(f"tensors must be CUDA/HIP tensors, got device={device}")
    if (
        labels.device != device
        or weight.device != device
        or teacher_hidden.device != device
    ):
        raise ValueError(
            "labels, weight and teacher_hidden must be on the same device as student_hidden"
        )
    if BT == 0:
        return _empty_distillation_tensors(
            student_hidden=student_hidden,
            weight=weight,
            mode="tied",
            teacher_hidden=teacher_hidden,
            teacher_weight=None,
            student_ce_weight=execution_plan.student_ce_weight,
            kl_weight=execution_plan.kl_weight,
            teacher_grad_policy=execution_plan.teacher_grad_policy,
        )

    return execute_plan(
        execution_plan,
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        gradient_materialization_scale=gradient_materialization_scale,
        _labels_prevalidated=True,
    )
