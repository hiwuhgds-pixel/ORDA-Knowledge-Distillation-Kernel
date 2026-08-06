from __future__ import annotations

import pytest
import torch
from utils.assertions import (
    assert_grad_close,
    assert_loss_components_match,
    assert_zero_scalar,
)
from utils.factories import (
    make_precomputed_hidden_inputs,
    make_precomputed_logits_inputs,
    make_separate_inputs,
    make_tied_inputs,
)
from utils.reference import reference_distillation_loss

from orda_kernel import (
    PrecomputedTeacher,
    SeparateTeacher,
    TiedTeacher,
    distillation_loss,
)


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

    # The independent FP64 oracle uses a different reduction order, so only
    # this deterministic FP32 sum path needs extra slack.
    component_atol = 2e-6 if reduction == "sum" and hs.dtype == torch.float32 else 1e-6
    assert_loss_components_match(torch, out, ref, atol=component_atol, rtol=1e-5)
    assert_grad_close(
        torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden"
    )
    assert_grad_close(
        torch, ht.grad, ref[5].grad, atol=1e-6, rtol=1e-5, name="teacher_hidden"
    )
    assert_grad_close(
        torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight"
    )


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
    assert_grad_close(
        torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden"
    )
    assert_grad_close(
        torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight"
    )
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
    assert_grad_close(
        torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden"
    )
    assert_grad_close(
        torch, ht.grad, ref[5].grad, atol=1e-6, rtol=1e-5, name="teacher_hidden"
    )
    assert_grad_close(
        torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight"
    )
    assert_grad_close(
        torch,
        teacher_weight.grad,
        ref[7].grad,
        atol=1e-6,
        rtol=1e-5,
        name="teacher_weight",
    )


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
    assert_grad_close(
        torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden"
    )
    assert_grad_close(
        torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight"
    )


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
    assert_grad_close(
        torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden"
    )
    assert_grad_close(
        torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight"
    )


@pytest.mark.parametrize("kl_divergence", ["mutual", "cojeffreys"])
def test_separate_cotrained_torch_backend_matches_reference(kl_divergence: str):
    hs, ht, weight, teacher_weight, labels = make_separate_inputs(
        torch,
        BT=10,
        Hs=6,
        Ht=8,
        V=19,
        dtype=torch.float32,
        seed=107,
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
        kl_divergence=kl_divergence,
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
        kl_divergence=kl_divergence,
    )
    ref[0].backward()

    assert_loss_components_match(torch, out, ref, atol=1e-6, rtol=1e-5)
    assert_grad_close(
        torch, hs.grad, ref[4].grad, atol=1e-6, rtol=1e-5, name="student_hidden"
    )
    assert_grad_close(
        torch, ht.grad, ref[5].grad, atol=1e-6, rtol=1e-5, name="teacher_hidden"
    )
    assert_grad_close(
        torch, weight.grad, ref[6].grad, atol=1e-6, rtol=1e-5, name="weight"
    )
    assert_grad_close(
        torch,
        teacher_weight.grad,
        ref[7].grad,
        atol=1e-6,
        rtol=1e-5,
        name="teacher_weight",
    )


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


# ── Low-precision accumulation ───────────────────────────────────────────────
def test_float16_cross_entropy_reduces_in_float32():
    rows = 1_000
    student_hidden = torch.ones(
        rows,
        2,
        dtype=torch.float16,
        requires_grad=True,
    )
    teacher_hidden = torch.zeros(rows, 2, dtype=torch.float16)
    weight = torch.tensor(
        [[25.0, 25.0], [-25.0, -25.0]],
        dtype=torch.float16,
        requires_grad=True,
    )
    labels = torch.ones(rows, dtype=torch.long)

    out = distillation_loss(
        student_hidden,
        weight,
        labels,
        TiedTeacher(teacher_hidden),
        student_ce_weight=1.0,
        teacher_ce_weight=0.0,
        kl_weight=0.0,
        reduction="mean",
        backend="torch",
    )
    out.loss.backward()

    assert out.loss.dtype is torch.float32
    assert torch.isfinite(out.loss)
    assert torch.allclose(out.loss, torch.tensor(100.0), atol=1e-4, rtol=0.0)
    assert student_hidden.grad is not None
    assert torch.isfinite(student_hidden.grad).all()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()


# ── Inactive weighted branches ───────────────────────────────────────────────
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_zero_weight_tied_student_branch_does_not_propagate_nonfinite_inputs(
    nonfinite: float,
):
    student_hidden = torch.full((4, 3), nonfinite, requires_grad=True)
    teacher_hidden = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(7, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])

    out = distillation_loss(
        student_hidden,
        weight,
        labels,
        TiedTeacher(teacher_hidden),
        student_ce_weight=0.0,
        teacher_ce_weight=1.0,
        kl_weight=0.0,
        backend="torch",
    )
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert torch.equal(out.student_ce, torch.zeros_like(out.student_ce))
    assert torch.equal(out.kl, torch.zeros_like(out.kl))
    assert student_hidden.grad is None
    assert teacher_hidden.grad is not None and torch.isfinite(teacher_hidden.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_zero_weight_frozen_teacher_branch_does_not_read_nonfinite_teacher(
    nonfinite: float,
):
    student_hidden = torch.randn(4, 3, requires_grad=True)
    teacher_hidden = torch.full((4, 5), nonfinite, requires_grad=True)
    weight = torch.randn(7, 3, requires_grad=True)
    teacher_weight = torch.full((7, 5), nonfinite, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])

    out = distillation_loss(
        student_hidden,
        weight,
        labels,
        SeparateTeacher(teacher_hidden, teacher_weight),
        teacher_ce_weight=0.0,
        kl_weight=0.0,
        backend="torch",
    )
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert torch.equal(out.teacher_ce, torch.zeros_like(out.teacher_ce))
    assert torch.equal(out.kl, torch.zeros_like(out.kl))
    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()
    assert teacher_hidden.grad is None
    assert teacher_weight.grad is None


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_all_zero_weights_return_finite_zero_without_reading_nonfinite_models(
    nonfinite: float,
):
    student_hidden = torch.full((4, 3), nonfinite, requires_grad=True)
    teacher_hidden = torch.full((4, 3), nonfinite, requires_grad=True)
    weight = torch.full((7, 3), nonfinite, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])

    out = distillation_loss(
        student_hidden,
        weight,
        labels,
        TiedTeacher(teacher_hidden),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=0.0,
        backend="torch",
    )
    out.loss.backward()

    for component in out:
        assert torch.equal(component, torch.zeros_like(component))
    assert torch.all(student_hidden.grad == 0)
    assert torch.all(weight.grad == 0)
    assert teacher_hidden.grad is None


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_precomputed_descriptor_rejects_nonfinite_logits_even_when_kl_is_zero(
    nonfinite: float,
):
    student_hidden = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(7, 3, requires_grad=True)
    teacher_logits = torch.full((4, 7), nonfinite)
    labels = torch.tensor([0, 1, 2, 3])

    with pytest.raises(ValueError, match="finite values or -inf"):
        distillation_loss(
            student_hidden,
            weight,
            labels,
            PrecomputedTeacher(logits=teacher_logits),
            student_ce_weight=1.0,
            kl_weight=0.0,
            backend="torch",
        )


# ── Ignored-row safety ───────────────────────────────────────────────────────
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_ignored_rows_with_nonfinite_hidden_keep_finite_gradients(nonfinite: float):
    # A subset of rows is ignored and carries a non-finite hidden value. The
    # torch backend must route zero gradient through those rows (like the Triton
    # kernel's tl.where select) instead of ``0 * NaN = NaN`` corrupting the
    # shared weight grad and the ignored rows' hidden grad.
    student_hidden = torch.randn(6, 4, requires_grad=True)
    teacher_hidden = torch.randn(6, 4, requires_grad=True)
    weight = torch.randn(9, 4, requires_grad=True)
    with torch.no_grad():
        student_hidden[1] = nonfinite
        student_hidden[4] = nonfinite
        teacher_hidden[4] = nonfinite
    labels = torch.tensor([0, -100, 2, 3, -100, 5])

    out = distillation_loss(
        student_hidden,
        weight,
        labels,
        TiedTeacher(teacher_hidden),
        student_ce_weight=0.8,
        teacher_ce_weight=0.25,
        kl_weight=0.35,
        kl_temperature=1.4,
        backend="torch",
    )
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.student_ce)
    assert torch.isfinite(out.teacher_ce)
    assert torch.isfinite(out.kl)
    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(teacher_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()
    # Ignored rows receive exactly zero gradient.
    assert torch.all(student_hidden.grad[[1, 4]] == 0)
    assert torch.all(teacher_hidden.grad[[1, 4]] == 0)


def test_ignored_nonfinite_rows_do_not_change_valid_row_result():
    # Injecting non-finite values only into ignored rows must leave the loss and
    # every gradient on the valid rows bit-for-bit unchanged.
    def run(taint: bool):
        hs = torch.randn(6, 4, generator=torch.Generator().manual_seed(3))
        ht = torch.randn(6, 4, generator=torch.Generator().manual_seed(4))
        w = torch.randn(9, 4, generator=torch.Generator().manual_seed(5))
        hs = hs.clone().requires_grad_(True)
        ht = ht.clone().requires_grad_(True)
        w = w.clone().requires_grad_(True)
        if taint:
            with torch.no_grad():
                hs[1] = float("nan")
                ht[1] = float("inf")
        labels = torch.tensor([0, -100, 2, 3, 4, 5])
        out = distillation_loss(
            hs,
            w,
            labels,
            TiedTeacher(ht),
            student_ce_weight=0.8,
            teacher_ce_weight=0.25,
            kl_weight=0.35,
            kl_temperature=1.4,
            backend="torch",
        )
        out.loss.backward()
        return out.loss.detach().clone(), hs.grad.clone(), w.grad.clone()

    clean_loss, clean_hs_grad, clean_w_grad = run(taint=False)
    taint_loss, taint_hs_grad, taint_w_grad = run(taint=True)

    assert torch.equal(clean_loss, taint_loss)
    valid = [0, 2, 3, 4, 5]
    assert torch.equal(clean_hs_grad[valid], taint_hs_grad[valid])
    assert torch.equal(clean_w_grad, taint_w_grad)


# ── Loss dtype ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_torch_backend_promotes_loss_to_at_least_float32(dtype):
    student_hidden = (torch.randn(4, 8, dtype=dtype) * 0.1).requires_grad_(True)
    teacher_hidden = (torch.randn(4, 8, dtype=dtype) * 0.1).requires_grad_(True)
    weight = (torch.randn(16, 8, dtype=dtype) * 0.1).requires_grad_(True)
    labels = torch.randint(0, 16, (4,))

    out = distillation_loss(
        student_hidden,
        weight,
        labels,
        TiedTeacher(teacher_hidden),
        student_ce_weight=0.7,
        teacher_ce_weight=0.3,
        kl_weight=0.2,
        kl_temperature=1.4,
        backend="torch",
    )

    # The Triton kernel accumulates loss in float32 for every dtype it supports
    # (fp16/bf16/fp32), so the torch backend lifts those to float32 to agree.
    # float64 is a torch-only path the kernel never runs, so its extra precision
    # is preserved rather than discarded.
    expected_dtype = torch.promote_types(dtype, torch.float32)
    assert out.loss.dtype == expected_dtype
    assert out.student_ce.dtype == expected_dtype
    assert out.teacher_ce.dtype == expected_dtype
    assert out.kl.dtype == expected_dtype
