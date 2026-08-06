from __future__ import annotations

import torch
import torch.nn.functional as F

from .spec import LossSpec, torch_loss_terms


def _maybe_compile(loss_fn, *, compile: bool):
    return torch.compile(loss_fn, mode="default") if compile else loss_fn


def separate_torch_loss_fn(spec: LossSpec, *, compile: bool):
    def loss_fn(h_s, h_t, student_weight, teacher_weight, labels):
        logits_s = F.linear(h_s, student_weight)
        logits_t = F.linear(h_t, teacher_weight)
        return torch_loss_terms(logits_s, logits_t, labels, spec)

    return _maybe_compile(loss_fn, compile=compile)


def precomputed_torch_loss_fn(spec: LossSpec, source: str, *, compile: bool):
    def loss_fn(h_s, student_weight, teacher_cache, teacher_weight, labels):
        logits_s = F.linear(h_s, student_weight)
        if source == "logits":
            logits_t = teacher_cache
        else:
            logits_t = F.linear(teacher_cache, teacher_weight)
        return torch_loss_terms(logits_s, logits_t, labels, spec)

    return _maybe_compile(loss_fn, compile=compile)
