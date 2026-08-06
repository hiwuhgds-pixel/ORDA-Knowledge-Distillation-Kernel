"""Precomputed-input builders and the aligned-span loss."""

from __future__ import annotations

from .aligned_span import aligned_span_distillation_loss
from .cross_vocab import build_exact_token_map, project_teacher_logits
from .multi_teacher import generalized_jensen_shannon, mix_teacher_logits
from .span_teacher import build_precomputed_span_teacher

__all__ = [
    "aligned_span_distillation_loss",
    "build_exact_token_map",
    "build_precomputed_span_teacher",
    "generalized_jensen_shannon",
    "mix_teacher_logits",
    "project_teacher_logits",
]
