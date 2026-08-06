"""Public precomputed-teacher low-level API."""

from __future__ import annotations

import torch

from .._api_types import (
    _DEFAULT_CHUNK_SIZE,
    AutotuneMode,
    ChunkSize,
    EntropyGate,
    GradWeightAccumulation,
    PrecomputedTeacher,
    _validate_precomputed_dtypes,
    _validate_precomputed_representation,
    _validate_precomputed_values,
)
from .._execution.chunking import (
    DEFAULT_MAX_FUSED_SIZE,
    normalize_chunk_request,
    normalize_max_fused_size,
)
from .._execution.dispatcher import execute_plan
from .._planning.models import STUDENT_GRAD_KL_VARIANTS
from .._planning.policy import build_execution_plan
from .._runtime import HAS_TRITON, triton_unavailable_reason
from .._torch_backend import _empty_distillation_tensors
from .._validation import (
    _normalize_bool,
    _normalize_execution_flags,
    _normalize_ignore_index,
    _normalize_loss_scalars,
    _reconcile_precomputed_logits_temperature,
    _validate_gradient_materialization_scale,
    _validate_label_range,
    _validate_loss_weights_and_reduction,
    _validate_model_dtypes,
    _validate_precomputed_teacher_cache,
    _validate_triton_model_dtype,
    _validate_vocab_size,
)

# ── Public API ───────────────────────────────────────────────────────────────
__all__ = ["precomputed_distillation_loss"]


@torch._dynamo.disable
def precomputed_distillation_loss(
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    logits_teacher: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
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
    teacher_hidden: torch.Tensor | None = None,
    teacher_weight: torch.Tensor | None = None,
    kl_divergence: str = "forward",
    num_chunks: int | None = None,
    kl_alpha: float | None = None,
    kl_beta: float | None = None,
    entropy_gate: EntropyGate | None = None,
    kl_teacher_temperature: float | None = None,
    teacher: PrecomputedTeacher | None = None,
    assume_valid_teacher_cache: bool = False,
    gradient_materialization_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # ── Argument normalization ───────────────────────────────
    assume_valid_labels = _normalize_bool("assume_valid_labels", assume_valid_labels)
    assume_valid_teacher_cache = _normalize_bool(
        "assume_valid_teacher_cache", assume_valid_teacher_cache
    )
    teacher_temperature_explicit = kl_teacher_temperature is not None
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
        0.0 if teacher_ce_weight is None else teacher_ce_weight
    )
    chunk_size, num_chunks, max_chunks = normalize_chunk_request(
        chunk_size,
        num_chunks=num_chunks,
        max_chunks=max_chunks,
    )
    max_fused_size = normalize_max_fused_size(max_fused_size)
    if kl_divergence not in STUDENT_GRAD_KL_VARIANTS:
        raise ValueError(
            "precomputed_distillation_loss supports the teacher-detached "
            f"kl_divergence values {sorted(STUDENT_GRAD_KL_VARIANTS)}, got "
            f"{kl_divergence!r} ('mutual'/'cojeffreys' require SeparateTeacher "
            "or TiedTeacher)"
        )
    descriptor_supplied = teacher is not None
    if teacher is not None:
        if not isinstance(teacher, PrecomputedTeacher):
            raise TypeError(
                "teacher must be a PrecomputedTeacher or None, got "
                f"{type(teacher).__name__}"
            )
        if any(
            tensor is not None
            for tensor in (logits_teacher, teacher_hidden, teacher_weight)
        ):
            raise ValueError(
                "teacher is mutually exclusive with logits_teacher, "
                "teacher_hidden, and teacher_weight"
            )
        if assume_valid_teacher_cache:
            raise ValueError(
                "assume_valid_teacher_cache applies only to raw cache tensors; "
                "PrecomputedTeacher uses its validation stamp"
            )
        _validate_precomputed_teacher_cache(teacher)
        kl_teacher_temperature = _reconcile_precomputed_logits_temperature(
            teacher,
            effective_teacher_ce_weight=effective_teacher_ce_weight,
            kl_weight=kl_weight,
            kl_teacher_temperature=kl_teacher_temperature,
            teacher_temperature_explicit=teacher_temperature_explicit,
        )
        logits_teacher = teacher.logits
        teacher_hidden = teacher.teacher_hidden
        teacher_weight = teacher.teacher_weight

    representation = _validate_precomputed_representation(
        logits=logits_teacher,
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        owner="precomputed_distillation_loss",
        logits_name="logits_teacher",
    )
    if not descriptor_supplied:
        cache_validation = (
            _validate_precomputed_dtypes
            if assume_valid_teacher_cache
            else _validate_precomputed_values
        )
        cache_validation(
            logits=logits_teacher,
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
            owner="precomputed_distillation_loss",
            logits_name="logits_teacher",
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
    if weight.ndim != 2 or weight.shape[1] != H:
        raise ValueError(f"weight must have shape (V, {H}), got {tuple(weight.shape)}")
    V = weight.shape[0]
    _validate_vocab_size(V)
    if representation == "logits":
        assert logits_teacher is not None
        if logits_teacher.shape != (BT, V):
            raise ValueError(
                f"logits_teacher must have shape ({BT}, {V}), got {tuple(logits_teacher.shape)}"
            )
    else:
        assert teacher_hidden is not None and teacher_weight is not None
        if teacher_hidden.ndim != 2 or teacher_hidden.shape[0] != BT:
            raise ValueError(
                f"teacher_hidden must have shape (BT, D_t), got {tuple(teacher_hidden.shape)}"
            )
        if teacher_weight.shape != (V, teacher_hidden.shape[1]):
            raise ValueError(
                f"teacher_weight must have shape ({V}, {teacher_hidden.shape[1]}), "
                f"got {tuple(teacher_weight.shape)}"
            )
    if labels is None:
        raise ValueError("labels is required")
    if labels.shape != (BT,):
        raise ValueError(f"labels must have shape ({BT},), got {tuple(labels.shape)}")
    if labels.dtype != torch.long:
        raise ValueError(f"labels must have dtype torch.long, got {labels.dtype}")
    device_tensors = [labels, weight]
    if representation == "logits":
        assert logits_teacher is not None
        device_tensors.append(logits_teacher)
    else:
        assert teacher_hidden is not None and teacher_weight is not None
        device_tensors.extend([teacher_hidden, teacher_weight])
    if any(t.device != device for t in device_tensors):
        raise ValueError("all tensors must be on the same device as student_hidden")
    _validate_loss_weights_and_reduction(
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=None,
        kl_weight=kl_weight,
        reduction=reduction,
    )
    if effective_teacher_ce_weight != 0.0:
        raise ValueError(
            "Triton precomputed_distillation_loss requires teacher_ce_weight=0.0"
        )
    teacher_path_active = kl_weight != 0.0
    model_dtype = _validate_model_dtypes(
        student_hidden=student_hidden,
        weight=weight,
        teacher_hidden=(
            teacher_hidden
            if teacher_path_active and representation == "hidden_weight"
            else None
        ),
        teacher_weight=(
            teacher_weight
            if teacher_path_active and representation == "hidden_weight"
            else None
        ),
        teacher_logits=logits_teacher if teacher_path_active else None,
    )
    # ── Plan and dispatch ────────────────────────────────────
    execution_plan = build_execution_plan(
        backend="triton",
        teacher_mode="precomputed",
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
            "precomputed_distillation_loss requires Triton and CUDA/HIP tensors "
            f"({triton_unavailable_reason()}). "
            "Use distillation_loss(..., backend='auto' or 'torch') for fallback execution."
        )
    if device.type != "cuda":
        raise ValueError(f"tensors must be CUDA/HIP tensors, got device={device}")
    if BT == 0:
        return _empty_distillation_tensors(
            student_hidden=student_hidden,
            weight=weight,
            mode="precomputed",
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
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
        teacher_weight=teacher_weight,
        logits_teacher=logits_teacher,
        gradient_materialization_scale=gradient_materialization_scale,
        _labels_prevalidated=True,
    )
