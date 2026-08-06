from __future__ import annotations

import math
from numbers import Integral, Real

import torch

from .._api_types import (
    AutotuneMode,
    ChunkSize,
    EntropyGate,
    GradWeightAccumulation,
)
from .._execution.chunking import (
    _default_max_chunks,
    fast_max_chunks,
    resolve_chunk_size,
)
from .._scalar_validation import ResolvedKLTemperatures, resolve_kl_temperatures
from .capabilities import (
    UnsupportedTritonPolicyError,
    resolve_kernel_path,
    validate_aligned_span_capability,
    validate_mode_variant,
    validate_policy_capability,
)
from .models import (
    CANONICAL_INACTIVE_KL_VARIANT,
    DEFAULT_KL_ALPHA,
    DEFAULT_KL_BETA,
    GATED_KL_VARIANTS,
    KL_HYPERPARAM_SPECS,
    PRECOMPUTED,
    PRECOMPUTED_ALIGNED_SPAN,
    PRECOMPUTED_TOKENWISE,
    SEPARATE_COTRAINED,
    SEPARATE_FROZEN,
    SEPARATE_TRAINED,
    TEACHER_GRAD_CE,
    TEACHER_GRAD_CE_KL,
    TEACHER_GRAD_KL,
    TEACHER_GRAD_NONE,
    TEACHER_KL_GRAD_VARIANTS,
    TIED_COTRAINED,
    TIED_FROZEN,
    TIED_TRAINED,
    VALID_KL_VARIANTS,
    AlignedSpanExecutionPlan,
    AlignedSpanSpecialization,
    ChunkPolicy,
    ExecutionPlan,
    ExecutionPolicy,
    KernelSpecialization,
    KLHyperparamSpec,
)


# ── KL hyperparameter resolution ─────────────────────────────────────────────
def _resolve_kl_hyperparam(
    name: str,
    value: float | None,
    *,
    kl_variant: str,
    spec: KLHyperparamSpec | None,
    canonical_default: float,
) -> float:
    if value is not None and (isinstance(value, bool) or not isinstance(value, Real)):
        raise ValueError(f"{name} must be a real number, got {value!r}")

    if spec is None:
        if value is not None:
            resolved = float(value)
            if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
                raise ValueError(
                    f"{name} must be a finite float in the open interval "
                    f"(0, 1), got {resolved!r}"
                )
        return canonical_default

    if value is None:
        return spec.default
    resolved = float(value)
    lower_valid = resolved > spec.lo if spec.open_lo else resolved >= spec.lo
    upper_valid = resolved < spec.hi if spec.open_hi else resolved <= spec.hi
    if not math.isfinite(resolved) or not lower_valid or not upper_valid:
        left = "(" if spec.open_lo else "["
        right = ")" if spec.open_hi else "]"
        raise ValueError(
            f"{name} for KL variant {kl_variant!r} must be a finite float in "
            f"{left}{spec.lo}, {spec.hi}{right}, got {resolved!r}"
        )
    if spec.exclude is not None and spec.exclude[0] < resolved < spec.exclude[1]:
        raise ValueError(
            f"{name} for KL variant {kl_variant!r} must satisfy "
            "abs(r - 1) >= 0.01; use 'forward' for 'renyi' or 'reverse' for "
            f"'renyi_reverse' at r = 1, got {resolved!r}"
        )
    return resolved


def resolve_kl_hyperparams(
    kl_variant: str,
    *,
    kl_weight: float,
    kl_alpha: float | None,
    kl_beta: float | None,
    entropy_gate: EntropyGate | None,
) -> tuple[float, float, EntropyGate | None]:
    """Validate provided KL hyperparameters and canonicalize unused ones.

    Range checks apply whenever a value is provided (mirroring how
    ``kl_temperature`` is validated even when the KL term is inactive), while
    values that the effective specialization never reads collapse to canonical
    defaults so equivalent plans compare equal.
    """
    if entropy_gate is not None and not isinstance(entropy_gate, EntropyGate):
        raise TypeError(
            f"entropy_gate must be an EntropyGate or None, got {entropy_gate!r}"
        )
    compute_kl = float(kl_weight) != 0.0
    variant_specs = KL_HYPERPARAM_SPECS.get(kl_variant, {})
    effective_alpha = _resolve_kl_hyperparam(
        "kl_alpha",
        kl_alpha,
        kl_variant=kl_variant,
        spec=variant_specs.get("kl_alpha"),
        canonical_default=DEFAULT_KL_ALPHA,
    )
    effective_beta = _resolve_kl_hyperparam(
        "kl_beta",
        kl_beta,
        kl_variant=kl_variant,
        spec=variant_specs.get("kl_beta"),
        canonical_default=DEFAULT_KL_BETA,
    )
    if not compute_kl:
        effective_alpha = DEFAULT_KL_ALPHA
        effective_beta = DEFAULT_KL_BETA
    if compute_kl and kl_variant in GATED_KL_VARIANTS:
        effective_gate = entropy_gate if entropy_gate is not None else EntropyGate()
    else:
        effective_gate = None
    return effective_alpha, effective_beta, effective_gate


# ── Teacher gradients and plans ──────────────────────────────────────────────
def teacher_grad_policy(kernel_path: str, *, teacher_ce_weight: float) -> str:
    if kernel_path in (PRECOMPUTED_TOKENWISE, SEPARATE_FROZEN, TIED_FROZEN):
        return TEACHER_GRAD_NONE
    if kernel_path in (SEPARATE_TRAINED, TIED_TRAINED):
        return TEACHER_GRAD_CE
    if kernel_path in (SEPARATE_COTRAINED, TIED_COTRAINED):
        return TEACHER_GRAD_CE_KL if teacher_ce_weight != 0.0 else TEACHER_GRAD_KL
    raise ValueError(f"unknown kernel path {kernel_path!r}")


def _validate_grad_weight_accumulation(value: object) -> None:
    if value not in ("auto", "compute", "fp32"):
        raise ValueError(
            "grad_weight_accumulation must be 'auto', 'compute', or 'fp32', got "
            f"{value!r}"
        )


def _build_chunk_policy(
    *,
    bt: int,
    vocab_size: int,
    chunk_size: ChunkSize,
    num_chunks: int | None,
    max_chunks: int | None,
) -> ChunkPolicy:
    dynamic = num_chunks is None and chunk_size.is_dynamic
    fast = num_chunks is None and chunk_size.is_fast
    max_chunks_limit = (
        fast_max_chunks(max_chunks) if fast else _default_max_chunks(bt, max_chunks)
    )
    if bt == 0:
        return ChunkPolicy(
            initial_chunk_size=0,
            initial_num_chunks=0,
            max_chunks=max_chunks,
            max_chunks_limit=max_chunks_limit,
            dynamic=dynamic,
            fast=fast,
        )
    if dynamic:
        initial_chunk_size, initial_num_chunks = resolve_chunk_size(
            bt,
            chunk_size,
            V=vocab_size,
            max_chunks=max_chunks_limit,
        )
    else:
        initial_chunk_size, initial_num_chunks = resolve_chunk_size(
            bt,
            chunk_size,
            max_chunks=max_chunks,
            num_chunks=num_chunks,
        )
    return ChunkPolicy(
        initial_chunk_size=int(initial_chunk_size),
        initial_num_chunks=int(initial_num_chunks),
        max_chunks=max_chunks,
        max_chunks_limit=max_chunks_limit,
        dynamic=dynamic,
        fast=fast,
    )


def build_execution_plan(
    *,
    backend: str,
    teacher_mode: str,
    compute_dtype: torch.dtype,
    bt: int,
    vocab_size: int,
    student_ce_weight: float,
    teacher_ce_weight: float,
    kl_weight: float,
    kl_temperature: float,
    kl_variant: str,
    ignore_index: int,
    reduction: str,
    chunk_size: ChunkSize,
    num_chunks: int | None,
    max_chunks: int | None,
    max_fused_size: int,
    autotune: AutotuneMode,
    grad_weight_accumulation: GradWeightAccumulation,
    refresh_autotune: bool = False,
    assume_valid_labels: bool = False,
    kl_alpha: float | None = None,
    kl_beta: float | None = None,
    entropy_gate: EntropyGate | None = None,
    kl_teacher_temperature: float | None = None,
) -> ExecutionPlan:
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"unknown execution backend {backend!r}")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        raise ValueError(f"ignore_index must be an integer, got {ignore_index!r}")
    if not (isinstance(autotune, bool) or autotune == "Max"):
        raise ValueError(f"autotune must be bool or 'Max', got {autotune!r}")
    for name, value in (
        ("refresh_autotune", refresh_autotune),
        ("assume_valid_labels", assume_valid_labels),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be bool, got {value!r}")
    _validate_grad_weight_accumulation(grad_weight_accumulation)
    if autotune is False and refresh_autotune:
        raise ValueError("refresh_autotune=True requires autotune=True or 'Max'")

    # ── Policy resolution ────────────────────────────────────
    temperatures = resolve_kl_temperatures(
        kl_temperature,
        kl_teacher_temperature,
    )
    try:
        policy = _resolve_execution_policy_resolved(
            teacher_mode,
            teacher_ce_weight=teacher_ce_weight,
            kl_weight=kl_weight,
            temperatures=temperatures,
            kl_variant=kl_variant,
            _enforce_kernel_capabilities=backend != "torch",
        )
    except UnsupportedTritonPolicyError:
        if backend != "auto":
            raise
        # The policy is valid for the reference backend; only the selected
        # Triton implementation lacks this particular specialization.
        policy = _resolve_execution_policy_resolved(
            teacher_mode,
            teacher_ce_weight=teacher_ce_weight,
            kl_weight=kl_weight,
            temperatures=temperatures,
            kl_variant=kl_variant,
            _enforce_kernel_capabilities=False,
        )
        effective_backend = "torch"
    else:
        effective_backend = "torch" if backend == "torch" else "triton"
    effective_kl_alpha, effective_kl_beta, effective_entropy_gate = (
        resolve_kl_hyperparams(
            str(kl_variant),
            kl_weight=float(kl_weight),
            kl_alpha=kl_alpha,
            kl_beta=kl_beta,
            entropy_gate=entropy_gate,
        )
    )
    # ── Plan assembly ────────────────────────────────────────
    if not policy.specialization.compute_kl:
        temperatures = ResolvedKLTemperatures(student=1.0, teacher=1.0)
    chunk_policy = _build_chunk_policy(
        bt=bt,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        max_chunks=max_chunks,
    )
    return ExecutionPlan(
        backend=effective_backend,
        teacher_mode=policy.teacher_mode,
        kernel_path=policy.kernel_path,
        specialization=policy.specialization,
        compute_dtype=compute_dtype,
        chunk_policy=chunk_policy,
        max_fused_size=int(max_fused_size),
        autotune=autotune,
        refresh_autotune=refresh_autotune,
        grad_weight_accumulation=grad_weight_accumulation,
        teacher_grad_policy=policy.specialization.teacher_grad_policy,
        student_ce_weight=float(student_ce_weight),
        teacher_ce_weight=float(teacher_ce_weight),
        kl_weight=float(kl_weight),
        kl_temperature=temperatures.student,
        kl_teacher_temperature=temperatures.teacher,
        ignore_index=int(ignore_index),
        reduction=reduction,
        assume_valid_labels=assume_valid_labels,
        kl_alpha=effective_kl_alpha,
        kl_beta=effective_kl_beta,
        entropy_gate=effective_entropy_gate,
    )


# ── Aligned-span execution plans ─────────────────────────────────────────────
def build_aligned_span_execution_plan(
    *,
    backend: str,
    teacher_type: str,
    compute_dtype: torch.dtype,
    bt: int,
    vocab_size: int,
    num_spans: int,
    student_ce_weight: float,
    span_kl_weight: float,
    ignore_index: int,
    reduction: str,
    objective: str,
    chunk_size: ChunkSize,
    num_chunks: int | None,
    max_chunks: int | None,
    max_fused_size: int,
    autotune: AutotuneMode,
    grad_weight_accumulation: GradWeightAccumulation,
    refresh_autotune: bool = False,
    assume_valid_labels: bool = False,
) -> AlignedSpanExecutionPlan:
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"unknown execution backend {backend!r}")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        raise ValueError(f"ignore_index must be an integer, got {ignore_index!r}")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    if objective not in ("binary_kl", "binary_ce"):
        raise ValueError(
            f"objective must be 'binary_kl' or 'binary_ce', got {objective!r}"
        )
    if not (isinstance(autotune, bool) or autotune == "Max"):
        raise ValueError(f"autotune must be bool or 'Max', got {autotune!r}")
    for name, value in (
        ("refresh_autotune", refresh_autotune),
        ("assume_valid_labels", assume_valid_labels),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be bool, got {value!r}")
    _validate_grad_weight_accumulation(grad_weight_accumulation)
    if autotune is False and refresh_autotune:
        raise ValueError("refresh_autotune=True requires autotune=True or 'Max'")
    for name, value in (
        ("student_ce_weight", student_ce_weight),
        ("span_kl_weight", span_kl_weight),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")

    specialization = AlignedSpanSpecialization(
        compute_student_ce=float(student_ce_weight) != 0.0,
        compute_span=float(span_kl_weight) != 0.0 and num_spans > 0,
    )
    effective_backend = "torch" if backend == "torch" else "triton"
    if backend != "torch":
        try:
            validate_aligned_span_capability(
                kernel_path=PRECOMPUTED_ALIGNED_SPAN,
                teacher_mode=PRECOMPUTED,
                teacher_type=teacher_type,
                objective=objective,
                compute_student_ce=specialization.compute_student_ce,
            )
        except UnsupportedTritonPolicyError:
            if backend != "auto":
                raise
            effective_backend = "torch"

    return AlignedSpanExecutionPlan(
        backend=effective_backend,
        teacher_mode=PRECOMPUTED,
        kernel_path=PRECOMPUTED_ALIGNED_SPAN,
        specialization=specialization,
        compute_dtype=compute_dtype,
        chunk_policy=_build_chunk_policy(
            bt=bt,
            vocab_size=vocab_size,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            max_chunks=max_chunks,
        ),
        max_fused_size=int(max_fused_size),
        autotune=autotune,
        refresh_autotune=refresh_autotune,
        grad_weight_accumulation=grad_weight_accumulation,
        student_ce_weight=float(student_ce_weight),
        span_kl_weight=float(span_kl_weight),
        ignore_index=int(ignore_index),
        reduction=reduction,
        objective=objective,
        assume_valid_labels=assume_valid_labels,
    )


# ── Execution-policy resolution ──────────────────────────────────────────────
def resolve_execution_policy(
    teacher_mode: str,
    *,
    teacher_ce_weight: float,
    kl_weight: float,
    kl_temperature: float,
    kl_variant: str,
    kl_teacher_temperature: float | None = None,
    _enforce_kernel_capabilities: bool = True,
) -> ExecutionPolicy:
    temperatures = resolve_kl_temperatures(
        kl_temperature,
        kl_teacher_temperature,
    )
    return _resolve_execution_policy_resolved(
        teacher_mode,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
        temperatures=temperatures,
        kl_variant=kl_variant,
        _enforce_kernel_capabilities=_enforce_kernel_capabilities,
    )


def _resolve_execution_policy_resolved(
    teacher_mode: str,
    *,
    teacher_ce_weight: float,
    kl_weight: float,
    temperatures: ResolvedKLTemperatures,
    kl_variant: str,
    _enforce_kernel_capabilities: bool,
) -> ExecutionPolicy:
    if teacher_ce_weight < 0.0:
        raise ValueError(f"teacher_ce_weight must be >= 0.0, got {teacher_ce_weight}")
    if kl_weight < 0.0:
        raise ValueError(f"kl_weight must be >= 0.0, got {kl_weight}")
    kl_temperature = temperatures.student
    kl_teacher_temperature = temperatures.teacher
    if kl_variant not in VALID_KL_VARIANTS:
        raise ValueError(f"unknown KL variant {kl_variant!r}")

    compute_kl = float(kl_weight) != 0.0
    compute_teacher_ce = float(teacher_ce_weight) != 0.0
    requires_teacher_kl_grad = compute_kl and kl_variant in TEACHER_KL_GRAD_VARIANTS
    if requires_teacher_kl_grad and kl_temperature != kl_teacher_temperature:
        raise ValueError(
            f"kl_divergence={kl_variant!r} requires kl_teacher_temperature "
            "to equal kl_temperature; asymmetric temperatures are supported "
            "only for teacher-detached KL variants"
        )
    validate_mode_variant(teacher_mode, kl_variant)
    kernel_path = resolve_kernel_path(
        teacher_mode,
        teacher_ce_weight=float(teacher_ce_weight),
        kl_weight=float(kl_weight),
        kl_variant=kl_variant,
    )
    if _enforce_kernel_capabilities:
        validate_policy_capability(
            teacher_mode=teacher_mode,
            kernel_path=kernel_path,
            kl_variant=kl_variant,
            compute_kl=compute_kl,
            compute_teacher_ce=compute_teacher_ce,
            requires_teacher_kl_grad=requires_teacher_kl_grad,
        )
    # ``t_is_one`` means both resolved KL temperatures use the unit fast path.
    effective_t_is_one = not compute_kl or (
        kl_temperature == 1.0 and kl_teacher_temperature == 1.0
    )
    effective_kl_variant = kl_variant if compute_kl else CANONICAL_INACTIVE_KL_VARIANT
    specialization = KernelSpecialization(
        compute_kl=compute_kl,
        t_is_one=effective_t_is_one,
        compute_teacher_ce=compute_teacher_ce,
        kl_variant=effective_kl_variant,
        teacher_grad_policy=teacher_grad_policy(
            kernel_path,
            teacher_ce_weight=float(teacher_ce_weight),
        ),
    )
    return ExecutionPolicy(
        teacher_mode=teacher_mode,
        kernel_path=kernel_path,
        specialization=specialization,
    )
