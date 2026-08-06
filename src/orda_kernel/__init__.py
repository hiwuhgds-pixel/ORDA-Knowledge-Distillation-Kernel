from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

import torch

from ._api_types import (
    AlignedLikelihoodConfig,
    AlignedSpanPairs,
    CrossVocabTeacher,
    PrecomputedSpanTeacher,
)
from ._runtime import HAS_TRITON
from .api import (
    ChunkSize,
    DistillationLoss,
    DistillationLossOutput,
    EntropyGate,
    KernelConfig,
    PrecomputedTeacher,
    SeparateTeacher,
    TiedTeacher,
    distillation_loss,
)
from .ops.precomputed_inputs import (
    aligned_span_distillation_loss,
    build_exact_token_map,
    build_precomputed_span_teacher,
    generalized_jensen_shannon,
    mix_teacher_logits,
    project_teacher_logits,
)

try:
    __version__ = _distribution_version("orda-kernel")
except PackageNotFoundError:
    __version__ = "0+unknown"


# ── Runtime controls ─────────────────────────────────────────────────────────
def is_available() -> bool:
    """Return True when the Triton CUDA/HIP kernels can be selected."""
    return bool(HAS_TRITON and torch.cuda.is_available())


def reset_autotune_cache() -> None:
    """Clear in-memory and disk autotune state."""
    from ._autotune import reset_autotune_cache as _reset_autotune_cache

    _reset_autotune_cache()


def begin_autotune_run() -> None:
    """Start a new process-local autotune run without clearing winners."""
    from ._autotune import begin_autotune_run as _begin_autotune_run

    _begin_autotune_run()


# ── Public surface ───────────────────────────────────────────────────────────
__all__ = [
    "AlignedLikelihoodConfig",
    "AlignedSpanPairs",
    "ChunkSize",
    "CrossVocabTeacher",
    "DistillationLoss",
    "DistillationLossOutput",
    "EntropyGate",
    "KernelConfig",
    "PrecomputedSpanTeacher",
    "PrecomputedTeacher",
    "SeparateTeacher",
    "TiedTeacher",
    "aligned_span_distillation_loss",
    "begin_autotune_run",
    "build_exact_token_map",
    "build_precomputed_span_teacher",
    "distillation_loss",
    "generalized_jensen_shannon",
    "is_available",
    "mix_teacher_logits",
    "project_teacher_logits",
    "reset_autotune_cache",
]
