"""Separate-teacher Triton execution paths."""

from __future__ import annotations

import torch

from ...._planning.models import (
    SEPARATE_COTRAINED,
    SEPARATE_FROZEN,
    SEPARATE_TRAINED,
    ChunkAttempt,
    ExecutionPlan,
)
from .cotrained import SeparateCotrainedCEFunction
from .frozen import SeparateFrozenCEFunction
from .trained import SeparateTrainedCEFunction

# ── Kernel routing and execution ─────────────────────────────────────────────
_FUNCTION_BY_KERNEL_PATH = {
    SEPARATE_FROZEN: SeparateFrozenCEFunction,
    SEPARATE_TRAINED: SeparateTrainedCEFunction,
    SEPARATE_COTRAINED: SeparateCotrainedCEFunction,
}


def _apply_separate_execution_plan(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    gradient_materialization_scale: torch.Tensor | None,
    execution_plan: ExecutionPlan,
    attempt: ChunkAttempt,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if execution_plan.backend != "triton" or execution_plan.teacher_mode != "separate":
        raise RuntimeError("separate execution received an incompatible execution plan")
    try:
        function = _FUNCTION_BY_KERNEL_PATH[execution_plan.kernel_path]
    except KeyError as exc:
        raise RuntimeError(
            f"unexpected separate-teacher kernel path {execution_plan.kernel_path!r}"
        ) from exc
    if execution_plan.kernel_path == SEPARATE_FROZEN:
        teacher_hidden = teacher_hidden.detach()
        teacher_weight = teacher_weight.detach()
    return function.apply(
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        gradient_materialization_scale,
        execution_plan,
        attempt,
    )
