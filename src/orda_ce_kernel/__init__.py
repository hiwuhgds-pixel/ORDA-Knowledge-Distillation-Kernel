import torch

from ._runtime import HAS_TRITON
from .api import (
    DistillationLoss,
    DistillationLossOutput,
    KernelConfig,
    PrecomputedTeacher,
    SeparateTeacher,
    TiedTeacher,
    distillation_loss,
)


def is_available() -> bool:
    """Return True when the Triton CUDA/HIP kernels can be selected."""
    return bool(HAS_TRITON and torch.cuda.is_available())


__all__ = [
    "DistillationLoss",
    "DistillationLossOutput",
    "KernelConfig",
    "PrecomputedTeacher",
    "SeparateTeacher",
    "TiedTeacher",
    "distillation_loss",
    "is_available",
]
