from __future__ import annotations

import pytest
import torch

from orda_ce_kernel import PrecomputedTeacher, SeparateTeacher, TiedTeacher, distillation_loss
from utils.assertions import assert_grad_close, assert_loss_components_match, assert_zero_scalar
from utils.factories import (
    make_precomputed_hidden_inputs,
    make_precomputed_logits_inputs,
    make_separate_inputs,
    make_tied_inputs,
)
from utils.reference import reference_distillation_loss


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_tied_teacher_torch_backend_matches_reference(reduction: str):
    hs, ht, weight, labels = make_tied_inputs(
        torch,
        BT=9,
        H=7,
        V=17,
        dtype=torch.float32,
        seed=101,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        TiedTeacher(ht),
        student_ce_weight=0.8,
        teacher_ce_weight=0.25,
        kl_weight=0.35,
        kl_temperature=1.4,
        reduction=reduction,
        backend="torch",
    )
    out.loss.backward()

    ref = reference_distillation_loss(
        torch,
        hs,
        weight,
        labels,
        teacher_mode="tied",
        teacher_hidden=ht,
        student_ce_weight=0.8,
        teacher_ce_weight=0.25,
        kl_weight=0.35,
        kl_temperature=1.4,
        reduction=reduction,
    )
    ref[0].backward()

    assert_loss_components_match(torch, out, ref, atol=1e-6, rtol=1e-5)
    assert_grad_close(torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden")
    assert_grad_close(torch, ht.grad, ref[5].grad, atol=1e-6, rtol=1e-5, name="teacher_hidden")
    assert_grad_close(torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight")


def test_separate_teacher_pure_kd_torch_backend_matches_reference():
    hs, ht, weight, teacher_weight, labels = make_separate_inputs(
        torch,
        BT=10,
        Hs=6,
        Ht=8,
        V=19,
        dtype=torch.float32,
        seed=102,
        teacher_requires_grad=True,
        teacher_weight_requires_grad=True,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        SeparateTeacher(ht, teacher_weight),
        kl_weight=0.4,
        kl_temperature=1.6,
        backend="torch",
    )
    out.loss.backward()

    ref = reference_distillation_loss(
        torch,
        hs,
        weight,
        labels,
        teacher_mode="separate",
        teacher_hidden=ht,
        teacher_weight=teacher_weight,
        kl_weight=0.4,
        kl_temperature=1.6,
    )
    ref[0].backward()

    assert_zero_scalar(torch, out.teacher_ce, name="teacher_ce")
    assert_loss_components_match(torch, out, ref, atol=1e-6, rtol=1e-5)
    assert_grad_close(torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden")
    assert_grad_close(torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight")
    assert ht.grad is None
    assert teacher_weight.grad is None


def test_separate_teacher_full_torch_backend_matches_reference():
    hs, ht, weight, teacher_weight, labels = make_separate_inputs(
        torch,
        BT=10,
        Hs=6,
        Ht=8,
        V=19,
        dtype=torch.float32,
        seed=103,
        teacher_requires_grad=True,
        teacher_weight_requires_grad=True,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        SeparateTeacher(ht, teacher_weight),
        teacher_ce_weight=1.0,
        kl_weight=0.4,
        kl_temperature=1.6,
        backend="torch",
    )
    out.loss.backward()

    ref = reference_distillation_loss(
        torch,
        hs,
        weight,
        labels,
        teacher_mode="separate",
        teacher_hidden=ht,
        teacher_weight=teacher_weight,
        teacher_ce_weight=1.0,
        kl_weight=0.4,
        kl_temperature=1.6,
    )
    ref[0].backward()

    assert_loss_components_match(torch, out, ref, atol=1e-6, rtol=1e-5)
    assert_grad_close(torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden")
    assert_grad_close(torch, ht.grad, ref[5].grad, atol=1e-6, rtol=1e-5, name="teacher_hidden")
    assert_grad_close(torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight")
    assert_grad_close(torch, teacher_weight.grad, ref[7].grad, atol=1e-6, rtol=1e-5, name="teacher_weight")


def test_precomputed_logits_torch_backend_matches_reference():
    hs, weight, teacher_logits, labels = make_precomputed_logits_inputs(
        torch,
        BT=11,
        H=7,
        V=23,
        dtype=torch.float32,
        seed=104,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        PrecomputedTeacher(logits=teacher_logits),
        kl_weight=0.3,
        kl_temperature=1.5,
        backend="torch",
    )
    out.loss.backward()

    ref = reference_distillation_loss(
        torch,
        hs,
        weight,
        labels,
        teacher_mode="precomputed",
        teacher_logits=teacher_logits,
        kl_weight=0.3,
        kl_temperature=1.5,
    )
    ref[0].backward()

    assert_zero_scalar(torch, out.teacher_ce, name="teacher_ce")
    assert_loss_components_match(torch, out, ref, atol=1e-6, rtol=1e-5)
    assert_grad_close(torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden")
    assert_grad_close(torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight")


def test_precomputed_hidden_weight_torch_backend_matches_logits_reference():
    hs, weight, teacher_hidden, teacher_weight, labels = make_precomputed_hidden_inputs(
        torch,
        BT=11,
        Hs=7,
        Ht=9,
        V=23,
        dtype=torch.float32,
        seed=105,
    )

    teacher = PrecomputedTeacher(
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
    )
    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        kl_weight=0.3,
        kl_temperature=1.5,
        backend="torch",
    )
    out.loss.backward()

    ref = reference_distillation_loss(
        torch,
        hs,
        weight,
        labels,
        teacher_mode="precomputed",
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        kl_weight=0.3,
        kl_temperature=1.5,
    )
    ref[0].backward()

    assert_zero_scalar(torch, out.teacher_ce, name="teacher_ce")
    assert_loss_components_match(torch, out, ref, atol=1e-6, rtol=1e-5)
    assert_grad_close(torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden")
    assert_grad_close(torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight")


def test_all_ignored_targets_produce_zero_loss_and_grads_on_torch_backend():
    hs, ht, weight, _labels = make_tied_inputs(
        torch,
        BT=8,
        H=6,
        V=13,
        dtype=torch.float32,
        seed=106,
        ignore_index=None,
    )
    labels = torch.full((8,), -100, dtype=torch.long)

    out = distillation_loss(
        hs,
        weight,
        labels,
        TiedTeacher(ht),
        kl_weight=0.5,
        kl_temperature=1.7,
        backend="torch",
    )
    out.loss.backward()

    for tensor in out:
        assert torch.equal(tensor, torch.zeros_like(tensor))
    assert torch.all(hs.grad == 0)
    assert torch.all(ht.grad == 0)
    assert torch.all(weight.grad == 0)
