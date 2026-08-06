from __future__ import annotations

import torch

from .spec import LossSpec


# ── Teacher adapters ─────────────────────────────────────────────────────────
def kernel_config(*, num_chunks: int | None = None, autotune: bool | str = False):
    from orda_kernel import KernelConfig

    return KernelConfig(num_chunks=num_chunks, autotune=autotune)


def separate_teacher(h_t, teacher_weight):
    from orda_kernel import SeparateTeacher

    return SeparateTeacher(h_t, teacher_weight)


def precomputed_teacher(cache, teacher_weight, source: str):
    from orda_kernel import PrecomputedTeacher

    if source == "logits":
        return PrecomputedTeacher(logits=cache)
    return PrecomputedTeacher(teacher_hidden=cache, teacher_weight=teacher_weight)


def tied_teacher(teacher_hidden, tied_mode: str):
    from orda_kernel import TiedTeacher

    return TiedTeacher(teacher_hidden, mode=tied_mode)


def orda_loss(
    hidden,
    weight,
    labels,
    teacher,
    spec: LossSpec,
    *,
    profile: str = "balanced",
    config=None,
    grad_scaler: torch.amp.GradScaler | None = None,
):
    from orda_kernel import distillation_loss

    return distillation_loss(
        hidden,
        weight,
        labels,
        teacher,
        student_ce_weight=spec.student_ce_weight,
        teacher_ce_weight=spec.teacher_ce_weight,
        kl_weight=spec.kl_weight,
        kl_temperature=spec.temperature,
        kl_divergence=spec.kl_divergence,
        profile=profile,
        backend="triton",
        config=config,
        grad_scaler=grad_scaler,
    ).loss


# ── Loss factories ───────────────────────────────────────────────────────────
def separate_orda_loss_fn(spec: LossSpec, *, num_chunks: int | None = None):
    config = kernel_config(num_chunks=num_chunks)

    def loss_fn(h_s, h_t, student_weight, teacher_weight, labels):
        return orda_loss(
            h_s,
            student_weight,
            labels,
            separate_teacher(h_t, teacher_weight),
            spec,
            config=config,
        )

    return loss_fn


def precomputed_orda_loss_fn(
    spec: LossSpec, source: str, *, num_chunks: int | None = None
):
    config = kernel_config(num_chunks=num_chunks)

    def loss_fn(h_s, student_weight, teacher_cache, teacher_weight, labels):
        return orda_loss(
            h_s,
            student_weight,
            labels,
            precomputed_teacher(teacher_cache, teacher_weight, source),
            spec,
            config=config,
        )

    return loss_fn
