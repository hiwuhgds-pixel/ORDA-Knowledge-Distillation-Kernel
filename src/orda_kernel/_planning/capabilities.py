from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .models import (
    PRECOMPUTED,
    PRECOMPUTED_ALIGNED_SPAN,
    PRECOMPUTED_TOKENWISE,
    SEPARATE,
    SEPARATE_COTRAINED,
    SEPARATE_FROZEN,
    SEPARATE_TRAINED,
    STUDENT_GRAD_KL_VARIANTS,
    TEACHER_KL_GRAD_VARIANTS,
    TIED,
    TIED_COTRAINED,
    TIED_FROZEN,
    TIED_TRAINED,
    VALID_KL_VARIANTS,
)


# ── Capability model and registry ────────────────────────────────────────────
class UnsupportedTritonPolicyError(ValueError):
    """A semantically valid policy that no current Triton path can execute."""


@dataclass(frozen=True)
class KernelCapabilities:
    teacher_modes: frozenset[str]
    kl_variants: frozenset[str]
    inactive_kl_variants: frozenset[str]
    supports_teacher_ce: bool
    supports_teacher_kl_grad: bool


@dataclass(frozen=True)
class AlignedSpanCapabilities:
    teacher_modes: frozenset[str]
    teacher_types: frozenset[str]
    objectives: frozenset[str]
    supports_student_ce: bool
    supports_teacher_grad: bool


KERNEL_CAPABILITIES: Mapping[str, KernelCapabilities] = MappingProxyType(
    {
        TIED_FROZEN: KernelCapabilities(
            teacher_modes=frozenset({TIED}),
            kl_variants=STUDENT_GRAD_KL_VARIANTS,
            inactive_kl_variants=VALID_KL_VARIANTS,
            supports_teacher_ce=False,
            supports_teacher_kl_grad=False,
        ),
        TIED_TRAINED: KernelCapabilities(
            teacher_modes=frozenset({TIED}),
            kl_variants=STUDENT_GRAD_KL_VARIANTS,
            inactive_kl_variants=VALID_KL_VARIANTS,
            supports_teacher_ce=True,
            supports_teacher_kl_grad=False,
        ),
        TIED_COTRAINED: KernelCapabilities(
            teacher_modes=frozenset({TIED}),
            kl_variants=TEACHER_KL_GRAD_VARIANTS,
            inactive_kl_variants=frozenset(),
            supports_teacher_ce=True,
            supports_teacher_kl_grad=True,
        ),
        SEPARATE_TRAINED: KernelCapabilities(
            teacher_modes=frozenset({SEPARATE}),
            kl_variants=STUDENT_GRAD_KL_VARIANTS,
            inactive_kl_variants=VALID_KL_VARIANTS,
            supports_teacher_ce=True,
            supports_teacher_kl_grad=False,
        ),
        SEPARATE_FROZEN: KernelCapabilities(
            teacher_modes=frozenset({SEPARATE}),
            kl_variants=STUDENT_GRAD_KL_VARIANTS,
            inactive_kl_variants=VALID_KL_VARIANTS,
            supports_teacher_ce=False,
            supports_teacher_kl_grad=False,
        ),
        SEPARATE_COTRAINED: KernelCapabilities(
            teacher_modes=frozenset({SEPARATE}),
            kl_variants=TEACHER_KL_GRAD_VARIANTS,
            inactive_kl_variants=frozenset(),
            supports_teacher_ce=True,
            supports_teacher_kl_grad=True,
        ),
        PRECOMPUTED_TOKENWISE: KernelCapabilities(
            teacher_modes=frozenset({PRECOMPUTED}),
            kl_variants=STUDENT_GRAD_KL_VARIANTS,
            inactive_kl_variants=STUDENT_GRAD_KL_VARIANTS,
            supports_teacher_ce=False,
            supports_teacher_kl_grad=False,
        ),
    }
)

ALIGNED_SPAN_CAPABILITIES: Mapping[str, AlignedSpanCapabilities] = MappingProxyType(
    {
        PRECOMPUTED_ALIGNED_SPAN: AlignedSpanCapabilities(
            teacher_modes=frozenset({PRECOMPUTED}),
            teacher_types=frozenset({"precomputed_span"}),
            objectives=frozenset({"binary_kl", "binary_ce"}),
            supports_student_ce=True,
            supports_teacher_grad=False,
        )
    }
)


# ── Capability lookup ────────────────────────────────────────────────────────
def kernel_capabilities(kernel_path: str) -> KernelCapabilities:
    try:
        return KERNEL_CAPABILITIES[kernel_path]
    except KeyError as exc:
        raise ValueError(f"unknown kernel path {kernel_path!r}") from exc


def aligned_span_capabilities(kernel_path: str) -> AlignedSpanCapabilities:
    try:
        return ALIGNED_SPAN_CAPABILITIES[kernel_path]
    except KeyError as exc:
        raise ValueError(f"unknown aligned-span kernel path {kernel_path!r}") from exc


def validate_aligned_span_capability(
    *,
    kernel_path: str,
    teacher_mode: str,
    teacher_type: str,
    objective: str,
    compute_student_ce: bool,
) -> None:
    capability = aligned_span_capabilities(kernel_path)
    if teacher_mode not in capability.teacher_modes:
        raise UnsupportedTritonPolicyError(
            f"kernel path {kernel_path!r} does not support teacher mode "
            f"{teacher_mode!r}"
        )
    if teacher_type not in capability.teacher_types:
        raise UnsupportedTritonPolicyError(
            "backend='triton' aligned-span execution requires "
            "PrecomputedSpanTeacher; materialize the detached teacher span "
            "cache first"
        )
    if objective not in capability.objectives:
        raise UnsupportedTritonPolicyError(
            f"aligned-span objective {objective!r} is not supported by "
            f"kernel path {kernel_path!r}"
        )
    if compute_student_ce and not capability.supports_student_ce:
        raise UnsupportedTritonPolicyError(
            f"kernel path {kernel_path!r} does not support student CE"
        )


def resolve_kernel_path(
    teacher_mode: str,
    *,
    teacher_ce_weight: float,
    kl_weight: float,
    kl_variant: str,
) -> str:
    if teacher_mode == TIED:
        if kl_variant in TEACHER_KL_GRAD_VARIANTS and kl_weight != 0.0:
            return TIED_COTRAINED
        if teacher_ce_weight > 0.0:
            return TIED_TRAINED
        return TIED_FROZEN
    if teacher_mode == PRECOMPUTED:
        return PRECOMPUTED_TOKENWISE
    if teacher_mode != SEPARATE:
        raise ValueError(f"unknown teacher mode {teacher_mode!r}")
    if kl_variant in TEACHER_KL_GRAD_VARIANTS and kl_weight != 0.0:
        return SEPARATE_COTRAINED
    if teacher_ce_weight > 0.0:
        return SEPARATE_TRAINED
    return SEPARATE_FROZEN


# ── Capability validation ────────────────────────────────────────────────────
def validate_mode_variant(teacher_mode: str, kl_variant: str) -> None:
    if kl_variant in TEACHER_KL_GRAD_VARIANTS and teacher_mode not in (
        SEPARATE,
        TIED,
    ):
        raise ValueError(
            f"KL variant {kl_variant!r} requires the separate or tied teacher mode"
        )


def validate_policy_capability(
    *,
    teacher_mode: str,
    kernel_path: str,
    kl_variant: str,
    compute_kl: bool,
    compute_teacher_ce: bool,
    requires_teacher_kl_grad: bool,
) -> None:
    capability = kernel_capabilities(kernel_path)
    if teacher_mode not in capability.teacher_modes:
        raise UnsupportedTritonPolicyError(
            f"kernel path {kernel_path!r} does not support teacher mode {teacher_mode!r}"
        )
    supported_variants = (
        capability.kl_variants if compute_kl else capability.inactive_kl_variants
    )
    if kl_variant not in supported_variants:
        raise UnsupportedTritonPolicyError(
            f"KL variant {kl_variant!r} is not supported by kernel path {kernel_path!r}"
        )
    if compute_teacher_ce and not capability.supports_teacher_ce:
        raise UnsupportedTritonPolicyError(
            f"kernel path {kernel_path!r} does not support teacher CE"
        )
    if requires_teacher_kl_grad and not capability.supports_teacher_kl_grad:
        raise UnsupportedTritonPolicyError(
            f"kernel path {kernel_path!r} does not support teacher KL gradients"
        )
