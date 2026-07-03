from .tied_teacher import tied_distillation_loss
from .separate_teacher import (
    separate_distillation_loss,
)
from .precomputed_teacher import precomputed_distillation_loss

__all__ = [
    "precomputed_distillation_loss",
    "separate_distillation_loss",
    "tied_distillation_loss",
]
