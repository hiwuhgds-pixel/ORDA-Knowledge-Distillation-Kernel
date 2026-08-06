from __future__ import annotations

import threading
import warnings

import torch
from torch import nn

from ._api_types import (
    Backend,
    DistillationLossOutput,
    EntropyGate,
    KernelConfig,
    KLVariant,
    PrecomputedTeacher,
    Profile,
    Reduction,
    SeparateTeacher,
    TiedTeacher,
)
from ._api_types import (
    ChunkSize as ChunkSize,
)
from ._config import (
    _resolve_config,
    _resolve_kl_config,
    _resolve_teacher_ce_weight,
)
from ._config import (
    _resolve_profile as _resolve_profile,
)
from ._planning.policy import build_execution_plan
from ._runtime import (
    HAS_TRITON,
    resolve_gradient_materialization_scale,
    triton_dtype_is_supported,
    triton_unavailable_reason,
)
from ._torch_backend import (
    _empty_distillation_tensors,
    _loss_output_dtype,
    _torch_reference,
    _zero_loss_from_sources,
    _zeroed_loss_components,
)
from ._validation import (
    _normalize_ignore_index,
    _normalize_loss_scalars,
    _reconcile_precomputed_logits_temperature,
    _teacher_mode_and_tensors,
    _validate_label_range,
    _validate_model_dtypes,
    _validate_precomputed_teacher_cache,
    _validate_public_args,
    _validate_teacher,
    _validate_tied_teacher_mode,
    _validate_triton_model_dtype,
    _validate_variant,
)

# ── Backend selection ────────────────────────────────────────────────────────
_WARNED_AUTO_TORCH_FALLBACK = False
_AUTO_TORCH_FALLBACK_LOCK = threading.Lock()


def _warn_auto_torch_fallback(
    reason: str,
    *,
    operation: str = "distillation_loss",
) -> None:
    global _WARNED_AUTO_TORCH_FALLBACK
    if _WARNED_AUTO_TORCH_FALLBACK:
        return
    with _AUTO_TORCH_FALLBACK_LOCK:
        if _WARNED_AUTO_TORCH_FALLBACK:
            return
        _WARNED_AUTO_TORCH_FALLBACK = True
    warnings.warn(
        f"ORDA {operation}(backend='auto') fell back to the torch reference "
        f"({reason}). This path materializes the full (BT, vocab) logits and does "
        "not use the fused chunked kernel. Pass backend='torch' to silence this, "
        "or ensure Triton + a supported CUDA/HIP dtype for the fused path.",
        RuntimeWarning,
        stacklevel=3,
    )


def _supports_triton(
    student_hidden: torch.Tensor,
    model_dtype: torch.dtype,
) -> bool:
    return bool(
        HAS_TRITON
        and student_hidden.device.type == "cuda"
        and triton_dtype_is_supported(model_dtype, student_hidden.device)
    )


# ── Functional API ───────────────────────────────────────────────────────────
@torch._dynamo.disable
def distillation_loss(
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
    *,
    student_ce_weight: float = 1.0,
    teacher_ce_weight: float | None = None,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    kl_teacher_temperature: float | None = None,
    kl_divergence: KLVariant = "forward",
    kl_alpha: float | None = None,
    kl_beta: float | None = None,
    entropy_gate: EntropyGate | None = None,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
    profile: Profile = "balanced",
    backend: Backend = "auto",
    config: KernelConfig | None = None,
    gradient_materialization_scale: torch.Tensor | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> DistillationLossOutput:
    # ── Config resolution ────────────────────────────────────
    gradient_materialization_scale = resolve_gradient_materialization_scale(
        student_hidden,
        gradient_materialization_scale,
        grad_scaler,
    )
    resolved = _resolve_config(profile, config)
    teacher_temperature_explicit = (
        kl_teacher_temperature is not None
        or resolved.kl_teacher_temperature is not None
    )
    (
        effective_kl_weight,
        effective_kl_temperature,
        effective_kl_teacher_temperature,
        effective_kl_divergence,
        effective_kl_alpha,
        effective_kl_beta,
        effective_entropy_gate,
    ) = _resolve_kl_config(
        resolved,
        kl_weight,
        kl_temperature,
        kl_divergence,
        kl_alpha,
        kl_beta,
        entropy_gate,
        kl_teacher_temperature=kl_teacher_temperature,
    )
    (
        student_ce_weight,
        teacher_ce_weight,
        effective_kl_weight,
        effective_kl_temperature,
        effective_kl_teacher_temperature,
    ) = _normalize_loss_scalars(
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=effective_kl_weight,
        kl_temperature=effective_kl_temperature,
        kl_teacher_temperature=effective_kl_teacher_temperature,
    )
    ignore_index = _normalize_ignore_index(ignore_index)
    _validate_public_args(
        student_hidden=student_hidden,
        weight=weight,
        labels=labels,
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=effective_kl_weight,
        kl_temperature=effective_kl_temperature,
        kl_teacher_temperature=effective_kl_teacher_temperature,
        reduction=reduction,
        backend=backend,
        config=resolved,
        gradient_materialization_scale=gradient_materialization_scale,
    )

    # ── Teacher validation ───────────────────────────────────
    if isinstance(teacher, PrecomputedTeacher):
        _validate_precomputed_teacher_cache(teacher)
    mode, teacher_hidden, teacher_weight, teacher_logits = _teacher_mode_and_tensors(
        teacher
    )
    _validate_teacher(
        mode=mode,
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        teacher_logits=teacher_logits,
        student_hidden=student_hidden,
        weight=weight,
    )
    effective_teacher_ce_weight = _resolve_teacher_ce_weight(
        mode,
        teacher_ce_weight,
    )
    teacher_path_active = (
        effective_teacher_ce_weight != 0.0 or effective_kl_weight != 0.0
    )
    effective_kl_teacher_temperature = _reconcile_precomputed_logits_temperature(
        teacher,
        effective_teacher_ce_weight=effective_teacher_ce_weight,
        kl_weight=effective_kl_weight,
        kl_teacher_temperature=effective_kl_teacher_temperature,
        teacher_temperature_explicit=teacher_temperature_explicit,
    )
    model_dtype = _validate_model_dtypes(
        student_hidden=student_hidden,
        weight=weight,
        teacher_hidden=teacher_hidden if teacher_path_active else None,
        teacher_weight=teacher_weight if teacher_path_active else None,
        teacher_logits=teacher_logits if teacher_path_active else None,
    )
    _validate_variant(mode, effective_kl_divergence)
    if isinstance(teacher, TiedTeacher):
        _validate_tied_teacher_mode(
            teacher.mode,
            effective_teacher_ce_weight=effective_teacher_ce_weight,
            kl_weight=effective_kl_weight,
            kl_variant=effective_kl_divergence,
        )

    # ── Backend planning ─────────────────────────────────────
    supports_triton = _supports_triton(student_hidden, model_dtype)
    if (
        backend == "triton"
        and mode == "precomputed"
        and effective_teacher_ce_weight != 0.0
    ):
        raise ValueError(
            "backend='triton' with PrecomputedTeacher requires teacher_ce_weight=0.0"
        )

    if (
        resolved.max_fused_size is None
        or resolved.autotune is None
        or resolved.chunk_size is None
    ):
        raise RuntimeError("profile resolution produced an incomplete kernel config")
    # The plan resolves policy support after this layer resolves device support.
    planned_backend = (
        "torch"
        if backend == "torch" or (backend == "auto" and not supports_triton)
        else backend
    )
    execution_plan = build_execution_plan(
        backend=planned_backend,
        teacher_mode=mode,
        compute_dtype=model_dtype,
        bt=student_hidden.shape[0],
        vocab_size=weight.shape[0],
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=effective_teacher_ce_weight,
        kl_weight=effective_kl_weight,
        kl_temperature=effective_kl_temperature,
        kl_teacher_temperature=effective_kl_teacher_temperature,
        kl_variant=effective_kl_divergence,
        ignore_index=ignore_index,
        reduction=reduction,
        chunk_size=resolved.chunk_size,
        num_chunks=resolved.num_chunks,
        max_chunks=resolved.max_chunks,
        max_fused_size=resolved.max_fused_size,
        autotune=resolved.autotune,
        refresh_autotune=bool(resolved.refresh_autotune),
        grad_weight_accumulation=resolved.effective_grad_weight_accumulation,
        assume_valid_labels=resolved.effective_assume_valid_labels,
        kl_alpha=effective_kl_alpha,
        kl_beta=effective_kl_beta,
        entropy_gate=effective_entropy_gate,
    )
    # ── Execution dispatch ───────────────────────────────────
    if backend == "triton":
        _validate_triton_model_dtype(model_dtype, student_hidden.device)
    _validate_label_range(
        labels,
        weight.shape[0],
        ignore_index,
        assume_valid_labels=resolved.effective_assume_valid_labels,
    )
    if backend == "triton" and not supports_triton:
        reason = (
            triton_unavailable_reason()
            if not HAS_TRITON
            else "unsupported dtype or device"
        )
        raise RuntimeError(
            "backend='triton' requires Triton and CUDA/HIP tensors "
            f"({reason}); use backend='auto' or backend='torch' for fallback execution"
        )
    if student_hidden.shape[0] == 0:
        return DistillationLossOutput(
            *_empty_distillation_tensors(
                student_hidden=student_hidden,
                weight=weight,
                mode=execution_plan.teacher_mode,
                teacher_hidden=teacher_hidden,
                teacher_weight=teacher_weight,
                student_ce_weight=execution_plan.student_ce_weight,
                kl_weight=execution_plan.kl_weight,
                teacher_grad_policy=execution_plan.teacher_grad_policy,
            )
        )
    if not (
        execution_plan.student_ce_weight != 0.0
        or execution_plan.specialization.compute_teacher_ce
        or execution_plan.specialization.compute_kl
    ):
        loss = _zero_loss_from_sources(student_hidden, weight).to(
            _loss_output_dtype(student_hidden.dtype)
        )
        return DistillationLossOutput(*_zeroed_loss_components(loss))
    if backend == "auto" and execution_plan.backend == "torch":
        _warn_auto_torch_fallback(
            triton_unavailable_reason()
            if not HAS_TRITON
            else (
                "unsupported dtype/device"
                if not supports_triton
                else "no Triton kernel supports this teacher/KL specialization"
            )
        )
    if execution_plan.backend == "torch":
        return _torch_reference(
            student_hidden=student_hidden,
            weight=weight,
            labels=labels,
            teacher=teacher,
            student_ce_weight=student_ce_weight,
            teacher_ce_weight=effective_teacher_ce_weight,
            kl_weight=effective_kl_weight,
            kl_temperature=effective_kl_temperature,
            ignore_index=ignore_index,
            reduction=reduction,
            kl_divergence=effective_kl_divergence,
            execution_plan=execution_plan,
        )

    from ._execution.dispatcher import execute_plan

    loss, student_ce, teacher_ce, kl = execute_plan(
        execution_plan,
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        teacher_weight=teacher_weight,
        logits_teacher=teacher_logits,
        gradient_materialization_scale=gradient_materialization_scale,
        _labels_prevalidated=True,
    )
    return DistillationLossOutput(loss, student_ce, teacher_ce, kl)


# ── Module API ───────────────────────────────────────────────────────────────
class DistillationLoss(nn.Module):
    def __init__(
        self,
        *,
        student_ce_weight: float = 1.0,
        teacher_ce_weight: float | None = None,
        kl_weight: float = 0.0,
        kl_temperature: float = 1.0,
        kl_teacher_temperature: float | None = None,
        kl_divergence: KLVariant = "forward",
        kl_alpha: float | None = None,
        kl_beta: float | None = None,
        entropy_gate: EntropyGate | None = None,
        ignore_index: int = -100,
        reduction: Reduction = "mean",
        profile: Profile = "balanced",
        backend: Backend = "auto",
        config: KernelConfig | None = None,
    ) -> None:
        super().__init__()
        self.student_ce_weight = student_ce_weight
        self.teacher_ce_weight = teacher_ce_weight
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature
        self.kl_teacher_temperature = kl_teacher_temperature
        self.kl_divergence = kl_divergence
        self.kl_alpha = kl_alpha
        self.kl_beta = kl_beta
        self.entropy_gate = entropy_gate
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.profile = profile
        self.backend = backend
        self.config = config

    def forward(
        self,
        student_hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
        gradient_materialization_scale: torch.Tensor | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> DistillationLossOutput:
        return distillation_loss(
            student_hidden,
            weight,
            labels,
            teacher,
            student_ce_weight=self.student_ce_weight,
            teacher_ce_weight=self.teacher_ce_weight,
            kl_weight=self.kl_weight,
            kl_temperature=self.kl_temperature,
            kl_teacher_temperature=self.kl_teacher_temperature,
            kl_divergence=self.kl_divergence,
            kl_alpha=self.kl_alpha,
            kl_beta=self.kl_beta,
            entropy_gate=self.entropy_gate,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            profile=self.profile,
            backend=self.backend,
            config=self.config,
            gradient_materialization_scale=gradient_materialization_scale,
            grad_scaler=grad_scaler,
        )
