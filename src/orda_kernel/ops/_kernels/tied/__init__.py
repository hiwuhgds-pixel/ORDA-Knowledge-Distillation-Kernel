"""Tied-teacher Triton execution paths."""

from __future__ import annotations

import torch

from ...._planning.models import (
    TIED_COTRAINED,
    TIED_FROZEN,
    TIED_TRAINED,
    ChunkAttempt,
    ExecutionPlan,
)
from .cotrained import TiedCotrainedCEFunction
from .frozen import TiedFrozenCEFunction
from .trained import TiedTrainedCEFunction

# ── Kernel routing and execution ─────────────────────────────────────────────
_FUNCTION_BY_KERNEL_PATH = {
    TIED_FROZEN: TiedFrozenCEFunction,
    TIED_TRAINED: TiedTrainedCEFunction,
    TIED_COTRAINED: TiedCotrainedCEFunction,
}

__all__ = [
    "TiedCotrainedCEFunction",
    "TiedFrozenCEFunction",
    "TiedTrainedCEFunction",
]


def _apply_tied_execution_plan(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    gradient_materialization_scale: torch.Tensor | None,
    execution_plan: ExecutionPlan,
    attempt: ChunkAttempt,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if execution_plan.backend != "triton" or execution_plan.teacher_mode != "tied":
        raise RuntimeError("tied execution received an incompatible execution plan")
    try:
        function = _FUNCTION_BY_KERNEL_PATH[execution_plan.kernel_path]
    except KeyError as exc:
        raise RuntimeError(
            f"unexpected tied-teacher kernel path {execution_plan.kernel_path!r}"
        ) from exc
    if execution_plan.kernel_path == TIED_FROZEN:
        teacher_hidden = teacher_hidden.detach()
    return function.apply(
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        gradient_materialization_scale,
        execution_plan,
        attempt,
    )
