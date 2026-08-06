"""Precomputed-teacher Triton execution path."""

from __future__ import annotations

import torch

from ...._planning.models import (
    PRECOMPUTED_TOKENWISE,
    ChunkAttempt,
    ExecutionPlan,
)
from .aligned_span import PrecomputedAlignedSpanFunction
from .tokenwise import PrecomputedCEFunction

_FUNCTION_BY_KERNEL_PATH = {
    PRECOMPUTED_TOKENWISE: PrecomputedCEFunction,
}

__all__ = ["PrecomputedAlignedSpanFunction", "PrecomputedCEFunction"]


def _apply_precomputed_execution_plan(
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    logits_teacher: torch.Tensor | None,
    labels: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    teacher_weight: torch.Tensor | None,
    gradient_materialization_scale: torch.Tensor | None,
    execution_plan: ExecutionPlan,
    attempt: ChunkAttempt,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        execution_plan.backend != "triton"
        or execution_plan.teacher_mode != "precomputed"
    ):
        raise RuntimeError("precomputed execution received an incompatible plan")
    try:
        function = _FUNCTION_BY_KERNEL_PATH[execution_plan.kernel_path]
    except KeyError as exc:
        raise RuntimeError(
            f"unexpected precomputed kernel path {execution_plan.kernel_path!r}"
        ) from exc
    return function.apply(
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
