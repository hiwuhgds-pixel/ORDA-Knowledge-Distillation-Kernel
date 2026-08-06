"""CPU routing and gradient-ownership contracts for split tied execution."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import orda_kernel as ock
from orda_kernel._planning.models import (
    TEACHER_GRAD_CE,
    TEACHER_GRAD_CE_KL,
    TEACHER_GRAD_KL,
    TEACHER_GRAD_NONE,
    TIED_COTRAINED,
    TIED_FROZEN,
    TIED_TRAINED,
)
from orda_kernel._planning.policy import resolve_execution_policy


def _tied_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(421)
    student_hidden = torch.randn(
        4, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    teacher_hidden = torch.randn(
        4, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    weight = torch.randn(
        5, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    labels = torch.tensor([0, 1, 3, 4], dtype=torch.long)
    return student_hidden, teacher_hidden, weight, labels


# ── Policy routing ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("variant", "kl_weight", "teacher_ce_weight", "kernel_path", "teacher_policy"),
    [
        ("forward", 0.0, 0.0, TIED_FROZEN, TEACHER_GRAD_NONE),
        ("forward", 0.4, 0.0, TIED_FROZEN, TEACHER_GRAD_NONE),
        ("cojeffreys", 0.0, 0.5, TIED_TRAINED, TEACHER_GRAD_CE),
        ("forward", 0.4, 0.5, TIED_TRAINED, TEACHER_GRAD_CE),
        ("mutual", 0.4, 0.0, TIED_COTRAINED, TEACHER_GRAD_KL),
        ("cojeffreys", 0.4, 0.5, TIED_COTRAINED, TEACHER_GRAD_CE_KL),
    ],
)
def test_tied_policy_routes_by_gradient_topology(
    variant: str,
    kl_weight: float,
    teacher_ce_weight: float,
    kernel_path: str,
    teacher_policy: str,
):
    policy = resolve_execution_policy(
        "tied",
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
        kl_temperature=1.0,
        kl_variant=variant,
    )

    assert policy.kernel_path == kernel_path
    assert policy.specialization.teacher_grad_policy == teacher_policy


# ── Gradient ownership ───────────────────────────────────────────────────────
def test_tied_frozen_detaches_teacher_hidden_without_freezing_shared_weight():
    student_hidden, teacher_hidden, weight, labels = _tied_inputs()
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.TiedTeacher(teacher_hidden, mode="frozen"),
        teacher_ce_weight=0.0,
        kl_weight=0.4,
        backend="torch",
    )
    out.loss.backward()

    assert teacher_hidden.grad is None
    assert student_hidden.grad is not None
    assert weight.grad is not None
    assert bool(torch.count_nonzero(weight.grad))

    hs, ht, shared_weight, labels = _tied_inputs()
    frozen_ce = ock.distillation_loss(
        hs,
        shared_weight,
        labels,
        ock.TiedTeacher(ht, mode="frozen"),
        teacher_ce_weight=0.0,
        kl_weight=0.0,
        backend="torch",
    )
    frozen_ce.loss.backward()

    hs_ref, _ht_ref, weight_ref, labels_ref = _tied_inputs()
    F.cross_entropy(hs_ref @ weight_ref.T, labels_ref).backward()
    torch.testing.assert_close(hs.grad, hs_ref.grad)
    torch.testing.assert_close(shared_weight.grad, weight_ref.grad)
    assert ht.grad is None


def test_tied_trained_accumulates_student_and_teacher_ce_into_shared_weight():
    student_hidden, teacher_hidden, weight, labels = _tied_inputs()
    teacher_ce_weight = 0.6
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.TiedTeacher(teacher_hidden, mode="trained"),
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.0,
        backend="torch",
    )
    out.loss.backward()

    hs_ref, ht_ref, weight_ref, labels_ref = _tied_inputs()
    expected = F.cross_entropy(hs_ref @ weight_ref.T, labels_ref)
    expected = expected + teacher_ce_weight * F.cross_entropy(
        ht_ref @ weight_ref.T, labels_ref
    )
    expected.backward()

    assert teacher_hidden.grad is not None
    torch.testing.assert_close(student_hidden.grad, hs_ref.grad)
    torch.testing.assert_close(teacher_hidden.grad, ht_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)


def _cotrained_grads(
    teacher_ce_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_hidden, teacher_hidden, weight, labels = _tied_inputs()
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.TiedTeacher(teacher_hidden, mode="cotrained"),
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.4,
        kl_divergence="mutual",
        backend="torch",
    )
    out.loss.backward()
    assert student_hidden.grad is not None
    assert teacher_hidden.grad is not None
    assert weight.grad is not None
    return student_hidden.grad, teacher_hidden.grad, weight.grad


def test_tied_cotrained_keeps_teacher_kl_and_adds_teacher_ce_when_active():
    student_kl, teacher_kl, weight_kl = _cotrained_grads(teacher_ce_weight=0.0)
    student_both, teacher_both, weight_both = _cotrained_grads(teacher_ce_weight=0.6)

    assert bool(torch.count_nonzero(teacher_kl))
    torch.testing.assert_close(student_both, student_kl)

    _hs_ref, ht_ref, weight_ref, labels_ref = _tied_inputs()
    (0.6 * F.cross_entropy(ht_ref @ weight_ref.T, labels_ref)).backward()
    torch.testing.assert_close(teacher_both - teacher_kl, ht_ref.grad)
    torch.testing.assert_close(weight_both - weight_kl, weight_ref.grad)
