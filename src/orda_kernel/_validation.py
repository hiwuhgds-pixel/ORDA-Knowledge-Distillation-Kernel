from __future__ import annotations

import math
from numbers import Integral, Real

import torch

from ._api_types import (
    AutotuneMode,
    GradWeightAccumulation,
    KernelConfig,
    PrecomputedTeacher,
    SeparateTeacher,
    TiedTeacher,
    _validate_precomputed_representation,
)
from ._planning.models import TEACHER_KL_GRAD_VARIANTS, VALID_KL_VARIANTS
from ._runtime import (
    TRITON_MODEL_DTYPES,
    resolve_uniform_model_dtype,
    triton_dtype_is_supported,
)
from ._scalar_validation import resolve_kl_temperatures


# ── Host label-range validation ──────────────────────────────────────────────
def _normalize_ignore_index(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"ignore_index must be an integer, got {value!r}")
    return int(value)


def _normalize_bool(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool, got {value!r}")
    return value


def _normalize_autotune(value) -> AutotuneMode:
    if isinstance(value, bool) or value == "Max":
        return value
    raise ValueError(f"autotune must be bool or 'Max', got {value!r}")


def _normalize_execution_flags(
    *,
    autotune,
    grad_weight_accumulation,
) -> tuple[AutotuneMode, GradWeightAccumulation]:
    if grad_weight_accumulation is None:
        grad_weight_accumulation = "auto"
    elif grad_weight_accumulation not in ("auto", "compute", "fp32"):
        raise ValueError(
            "grad_weight_accumulation must be 'auto', 'compute', 'fp32', or None, "
            f"got {grad_weight_accumulation!r}"
        )
    return _normalize_autotune(autotune), grad_weight_accumulation


def _validate_label_range(
    labels: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
    *,
    assume_valid_labels: bool = False,
) -> None:
    """Validate label values unless the caller accepts the trusted fast path."""
    if assume_valid_labels:
        return
    try:
        valid_mask = (labels == ignore_index) | ((labels >= 0) & (labels < vocab_size))
    except TypeError:
        if isinstance(labels, torch.Tensor):
            raise
        return
    if not bool(valid_mask.all()):
        bad_values = labels[~valid_mask][:5].tolist()
        raise ValueError(
            f"labels contains values outside [0, {vocab_size}) and not equal "
            f"to ignore_index={ignore_index}: {bad_values}"
        )


# ── Vocabulary and dtype checks ──────────────────────────────────────────────
def _validate_vocab_size(vocab_size: int) -> None:
    if vocab_size <= 0:
        raise ValueError(
            "weight vocabulary dimension must be positive and non-empty, "
            f"got {vocab_size}"
        )


def _validate_model_dtypes(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    teacher_hidden: torch.Tensor | None = None,
    teacher_weight: torch.Tensor | None = None,
    teacher_logits: torch.Tensor | None = None,
) -> torch.dtype:
    named_tensors = [
        ("student_hidden", student_hidden),
        ("weight", weight),
    ]
    for name, tensor in (
        ("teacher_hidden", teacher_hidden),
        ("teacher_weight", teacher_weight),
    ):
        if tensor is not None:
            named_tensors.append((name, tensor))
    for name, tensor in named_tensors:
        if tensor.layout is not torch.strided:
            raise ValueError(
                f"{name} must use torch.strided layout, got {tensor.layout}"
            )
    model_dtype = resolve_uniform_model_dtype(*named_tensors)
    if teacher_logits is not None:
        if teacher_logits.layout is not torch.strided:
            raise ValueError(
                "teacher_logits must use torch.strided layout, "
                f"got {teacher_logits.layout}"
            )
        # Cached logits are copied chunk-wise into the model compute dtype.
        # Validate that they are floating without requiring dtype uniformity.
        resolve_uniform_model_dtype(("teacher_logits", teacher_logits))
    return model_dtype


def _validate_triton_model_dtype(
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if dtype not in TRITON_MODEL_DTYPES:
        raise ValueError(
            "Triton model tensors must use dtype "
            "{torch.float16, torch.bfloat16, torch.float32}; "
            f"got {dtype}"
        )
    if not triton_dtype_is_supported(dtype, device):
        raise ValueError(f"Triton bfloat16 is not supported on device {device}")


# ── Loss scalar normalization ────────────────────────────────────────────────
def _normalize_finite_scalar(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return normalized


def _normalize_loss_scalars(
    *,
    student_ce_weight,
    teacher_ce_weight,
    kl_weight,
    kl_temperature,
    kl_teacher_temperature,
) -> tuple[float, float | None, float, float, float]:
    normalized_student_ce_weight = _normalize_finite_scalar(
        "student_ce_weight",
        student_ce_weight,
    )
    normalized_teacher_ce_weight = (
        None
        if teacher_ce_weight is None
        else _normalize_finite_scalar("teacher_ce_weight", teacher_ce_weight)
    )
    temperatures = resolve_kl_temperatures(
        kl_temperature,
        kl_teacher_temperature,
    )
    return (
        normalized_student_ce_weight,
        normalized_teacher_ce_weight,
        _normalize_finite_scalar("kl_weight", kl_weight),
        temperatures.student,
        temperatures.teacher,
    )


def _reconcile_precomputed_logits_temperature(
    teacher,
    *,
    effective_teacher_ce_weight: float,
    kl_weight: float,
    kl_teacher_temperature: float,
    teacher_temperature_explicit: bool,
) -> float:
    """Return the teacher temperature to apply once a tempered cache is honored.

    A ``PrecomputedTeacher`` built at ``logits_temperature=T`` bakes ``T`` into
    its cached logits, so the loss must apply ``T_t = 1``. Such a cache holds
    only a KL target distribution, so teacher CE is rejected. When the caller
    left ``kl_teacher_temperature`` at its default the cache temperature is
    adopted; an explicitly conflicting value is rejected.
    """
    logits_temperature = getattr(teacher, "logits_temperature", None)
    if logits_temperature is None:
        return kl_teacher_temperature
    if effective_teacher_ce_weight != 0.0:
        raise ValueError(
            "PrecomputedTeacher.logits_temperature requires teacher_ce_weight=0.0; "
            "a tempered cache only holds the KL target distribution"
        )
    if kl_weight == 0.0:
        return kl_teacher_temperature
    if teacher_temperature_explicit and kl_teacher_temperature != logits_temperature:
        raise ValueError(
            "PrecomputedTeacher.logits_temperature="
            f"{logits_temperature} does not match the call's teacher temperature "
            f"{kl_teacher_temperature}; pass kl_teacher_temperature equal to the "
            "cache temperature or rebuild the cache"
        )
    return 1.0


# ── Teacher and KL validation ────────────────────────────────────────────────
def _teacher_mode_and_tensors(
    teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
) -> tuple[str, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(teacher, TiedTeacher):
        return "tied", teacher.hidden, None, None
    if isinstance(teacher, SeparateTeacher):
        return "separate", teacher.hidden, teacher.weight, None
    if isinstance(teacher, PrecomputedTeacher):
        return (
            "precomputed",
            teacher.teacher_hidden,
            teacher.teacher_weight,
            teacher.logits,
        )
    raise TypeError(
        "teacher must be TiedTeacher, SeparateTeacher, or PrecomputedTeacher"
    )


def _validate_precomputed_teacher_cache(teacher: PrecomputedTeacher) -> None:
    """Verify the descriptor's O(1) cache stamp before tensor extraction."""
    teacher._validate_cache_unchanged()


def _validate_variant(mode: str, variant: str) -> None:
    if variant not in VALID_KL_VARIANTS:
        raise ValueError(
            f"kl_divergence must be one of {sorted(VALID_KL_VARIANTS)}, got {variant!r}"
        )
    if variant in TEACHER_KL_GRAD_VARIANTS and mode not in ("separate", "tied"):
        raise ValueError(
            f"kl_divergence={variant!r} requires SeparateTeacher or TiedTeacher: the "
            "teacher must receive gradient from the KL, which the precomputed "
            "mode cannot provide"
        )


def _validate_tied_teacher_mode(
    declared_mode: str | None,
    *,
    effective_teacher_ce_weight: float,
    kl_weight: float,
    kl_variant: str,
) -> None:
    """Check a declared ``TiedTeacher.mode`` against the effective teacher role.

    ``None`` keeps the legacy weight-derived behavior.  An explicit mode never
    changes routing; it asserts that the loss weights and KL variant produce
    the teacher-gradient role the caller intended.
    """
    if declared_mode is None:
        return
    ce_active = float(effective_teacher_ce_weight) != 0.0
    kl_grad_active = float(kl_weight) != 0.0 and kl_variant in TEACHER_KL_GRAD_VARIANTS
    if declared_mode == "frozen" and (ce_active or kl_grad_active):
        raise ValueError(
            "TiedTeacher(mode='frozen') requires teacher_ce_weight=0.0 and no "
            "active teacher-KL-gradient variant; got teacher_ce_weight="
            f"{effective_teacher_ce_weight}, kl_divergence={kl_variant!r} with "
            f"kl_weight={kl_weight}"
        )
    if declared_mode == "trained" and (not ce_active or kl_grad_active):
        raise ValueError(
            "TiedTeacher(mode='trained') requires teacher_ce_weight > 0.0 and no "
            "active teacher-KL-gradient variant; got teacher_ce_weight="
            f"{effective_teacher_ce_weight}, kl_divergence={kl_variant!r} with "
            f"kl_weight={kl_weight}"
        )
    if declared_mode == "cotrained" and not kl_grad_active:
        raise ValueError(
            "TiedTeacher(mode='cotrained') requires kl_weight > 0.0 with "
            "kl_divergence in ('mutual', 'cojeffreys'); got kl_divergence="
            f"{kl_variant!r} with kl_weight={kl_weight}"
        )


# ── Public argument validation ───────────────────────────────────────────────
def _validate_gradient_materialization_scale(
    gradient_materialization_scale,
    device: torch.device,
) -> None:
    if gradient_materialization_scale is None:
        return
    if not isinstance(gradient_materialization_scale, torch.Tensor):
        raise TypeError(
            "gradient_materialization_scale must be a torch.Tensor or None, got "
            f"{type(gradient_materialization_scale).__name__}"
        )
    if gradient_materialization_scale.numel() != 1:
        raise ValueError(
            "gradient_materialization_scale must contain exactly one element, got "
            f"shape={tuple(gradient_materialization_scale.shape)}"
        )
    if gradient_materialization_scale.dtype != torch.float32:
        raise ValueError(
            "gradient_materialization_scale must have dtype torch.float32, got "
            f"{gradient_materialization_scale.dtype}"
        )
    if gradient_materialization_scale.device != device:
        raise ValueError(
            "gradient_materialization_scale must be on the same device as "
            f"student_hidden, got {gradient_materialization_scale.device} vs {device}"
        )
    if gradient_materialization_scale.requires_grad:
        raise ValueError("gradient_materialization_scale must not require gradients")


def _validate_loss_weights_and_reduction(
    *,
    student_ce_weight: float,
    teacher_ce_weight: float | None,
    kl_weight: float,
    reduction: str,
) -> None:
    """Shared loss-weight and reduction checks for all public entry points.

    Temperature normalization and representability are handled by
    ``resolve_kl_temperatures`` before this boundary.
    """
    if student_ce_weight < 0.0:
        raise ValueError(f"student_ce_weight must be >= 0.0, got {student_ce_weight}")
    if teacher_ce_weight is not None and teacher_ce_weight < 0.0:
        raise ValueError(f"teacher_ce_weight must be >= 0.0, got {teacher_ce_weight}")
    if kl_weight < 0.0:
        raise ValueError(f"kl_weight must be >= 0.0, got {kl_weight}")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def _validate_public_args(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    student_ce_weight: float,
    teacher_ce_weight: float | None,
    kl_weight: float,
    kl_temperature: float,
    kl_teacher_temperature: float,
    reduction: str,
    backend: str,
    config: KernelConfig,
    gradient_materialization_scale: torch.Tensor | None,
) -> None:
    # ── Public input contracts ───────────────────────────────
    _validate_loss_weights_and_reduction(
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
        reduction=reduction,
    )
    if backend not in ("auto", "triton", "torch"):
        raise ValueError(
            f"backend must be 'auto', 'triton', or 'torch', got {backend!r}"
        )
    if student_hidden.ndim != 2:
        raise ValueError(
            f"student_hidden must be 2D, got shape={tuple(student_hidden.shape)}"
        )
    if weight.ndim != 2 or weight.shape[1] != student_hidden.shape[1]:
        raise ValueError(
            f"weight must have shape (vocab, {student_hidden.shape[1]}), "
            f"got {tuple(weight.shape)}"
        )
    _validate_vocab_size(weight.shape[0])
    if labels.shape != (student_hidden.shape[0],):
        raise ValueError(
            f"labels must have shape ({student_hidden.shape[0]},), got {tuple(labels.shape)}"
        )
    if labels.dtype != torch.long:
        raise ValueError(f"labels must have dtype torch.long, got {labels.dtype}")
    if weight.device != student_hidden.device:
        raise ValueError(
            "weight must be on the same device as student_hidden, got "
            f"{weight.device} vs {student_hidden.device}"
        )
    if labels.device != student_hidden.device:
        raise ValueError(
            "labels must be on the same device as student_hidden, got "
            f"{labels.device} vs {student_hidden.device}"
        )
    _validate_gradient_materialization_scale(
        gradient_materialization_scale,
        student_hidden.device,
    )
    if config.max_fused_size is None:
        raise RuntimeError("resolved config is missing max_fused_size")
    if config.chunk_size is None:
        raise RuntimeError("resolved config is missing chunk_size")


# ── Teacher tensor validation ────────────────────────────────────────────────
def _validate_teacher(
    *,
    mode: str,
    teacher_hidden: torch.Tensor | None,
    teacher_weight: torch.Tensor | None,
    teacher_logits: torch.Tensor | None,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    # ── Mode-specific shapes ─────────────────────────────────
    bt, hidden = student_hidden.shape
    vocab = weight.shape[0]
    if mode == "tied":
        if teacher_hidden is None:
            raise ValueError("TiedTeacher.hidden must be a torch.Tensor")
        if teacher_hidden.shape != (bt, hidden):
            raise ValueError(
                f"TiedTeacher.hidden must have shape {(bt, hidden)}, "
                f"got {tuple(teacher_hidden.shape)}"
            )
    elif mode == "separate":
        if teacher_hidden is None or teacher_weight is None:
            raise ValueError(
                "SeparateTeacher.hidden and SeparateTeacher.weight must be torch.Tensor values"
            )
        if teacher_hidden.ndim != 2 or teacher_hidden.shape[0] != bt:
            raise ValueError(
                "SeparateTeacher.hidden must have shape (BT, teacher_hidden_dim), "
                f"got {tuple(teacher_hidden.shape)}"
            )
        if teacher_weight.shape != (vocab, teacher_hidden.shape[1]):
            raise ValueError(
                f"SeparateTeacher.weight must have shape "
                f"{(vocab, teacher_hidden.shape[1])}, got {tuple(teacher_weight.shape)}"
            )
    else:
        representation = _validate_precomputed_representation(
            logits=teacher_logits,
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
            owner="PrecomputedTeacher",
            logits_name="logits",
        )
        if representation == "logits":
            if teacher_logits is None:
                raise ValueError("PrecomputedTeacher.logits must be a torch.Tensor")
            if teacher_logits.shape != (bt, vocab):
                raise ValueError(
                    f"PrecomputedTeacher.logits must have shape {(bt, vocab)}, "
                    f"got {tuple(teacher_logits.shape)}"
                )
        else:
            if teacher_hidden is None or teacher_weight is None:
                raise ValueError(
                    "PrecomputedTeacher.teacher_hidden and teacher_weight must be torch.Tensor values"
                )
            if teacher_hidden.ndim != 2 or teacher_hidden.shape[0] != bt:
                raise ValueError(
                    "PrecomputedTeacher.teacher_hidden must have shape (BT, D_t), "
                    f"got {tuple(teacher_hidden.shape)}"
                )
            if teacher_weight.shape != (vocab, teacher_hidden.shape[1]):
                raise ValueError(
                    f"PrecomputedTeacher.teacher_weight must have shape "
                    f"{(vocab, teacher_hidden.shape[1])}, "
                    f"got {tuple(teacher_weight.shape)}"
                )

    # All teacher tensors must follow the student device.
    device = student_hidden.device
    for name, tensor in (
        ("teacher_hidden", teacher_hidden),
        ("teacher_weight", teacher_weight),
        ("teacher_logits", teacher_logits),
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(
                f"{name} must be on the same device as student_hidden, got "
                f"{tensor.device} vs {device}"
            )
