from __future__ import annotations

import pytest

from tests.utils.assertions import assert_grad_cosine, assert_loss_components_match
from tests.utils.env import pytest_skip_if_no_cuda_kernel
from tests.utils.factories import make_precomputed_inputs, make_separate_inputs, make_tied_inputs
from tests.utils.reference import reference_distill_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel import (
    KernelConfig,
    PrecomputedTeacher,
    SeparateTeacher,
    TiedTeacher,
    distillation_loss,
)


@pytest.mark.gpu
def test_public_tied_teacher_matches_fp64_reference():
    hs, ht, weight, labels = make_tied_inputs(
        torch, BT=18, H=32, V=257, dtype=torch.float16, seed=7101,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        TiedTeacher(ht),
        student_ce_weight=0.8,
        teacher_ce_weight=0.25,
        kd_weight=0.35,
        temperature=1.4,
        label_smoothing=0.05,
        config=KernelConfig(chunk_size=7, quantize_grad_weight=False),
        backend="triton",
    )
    out.loss.backward()

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, w_ref, _ = reference_distill_loss(
        torch,
        hs.cpu(),
        ht.cpu(),
        weight.cpu(),
        labels.cpu(),
        lambda_student=0.8,
        teacher_loss_weight=0.25,
        kl_weight=0.35,
        kl_temperature=1.4,
        label_smoothing=0.05,
        teacher_mode="tied",
    )
    ref_loss.backward()

    assert_loss_components_match(
        torch,
        out,
        (ref_loss, ref_s, ref_t, ref_kl),
        atol=4e-3,
        rtol=4e-3,
    )
    assert_grad_cosine(torch, hs.grad, hs_ref.grad, min_cos=0.995, name="student_hidden")
    assert_grad_cosine(torch, ht.grad, ht_ref.grad, min_cos=0.995, name="teacher_hidden")
    assert_grad_cosine(torch, weight.grad, w_ref.grad, min_cos=0.995, name="weight")


@pytest.mark.gpu
def test_public_separate_teacher_defaults_to_pure_kd():
    hs, ht, weight, teacher_weight, labels = make_separate_inputs(
        torch,
        BT=16,
        Hs=48,
        Ht=64,
        V=769,
        dtype=torch.float16,
        seed=7102,
        weight_teacher_requires_grad=True,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        SeparateTeacher(ht, teacher_weight),
        kd_weight=0.3,
        temperature=1.7,
        backend="triton",
    )
    out.loss.backward()

    assert torch.equal(out.teacher_ce, torch.zeros_like(out.teacher_ce))
    assert teacher_weight.grad is None

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, _ht_ref, w_ref, _wt_ref = reference_distill_loss(
        torch,
        hs.cpu(),
        ht.cpu(),
        weight.cpu(),
        labels.cpu(),
        kl_weight=0.3,
        kl_temperature=1.7,
        teacher_mode="separate",
        weight_teacher=teacher_weight.cpu(),
    )
    ref_loss.backward()

    assert_loss_components_match(
        torch,
        out,
        (ref_loss, ref_s, torch.zeros_like(ref_t), ref_kl),
        atol=4e-3,
        rtol=4e-3,
    )
    assert_grad_cosine(torch, hs.grad, hs_ref.grad, min_cos=0.995, name="student_hidden")
    assert_grad_cosine(torch, weight.grad, w_ref.grad, min_cos=0.995, name="weight")


@pytest.mark.gpu
def test_public_precomputed_teacher_uses_supplied_logits_without_teacher_grad_path():
    hs, weight, teacher_logits, labels = make_precomputed_inputs(
        torch,
        BT=16,
        H=48,
        V=769,
        dtype=torch.float16,
        seed=7103,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        PrecomputedTeacher(teacher_logits),
        kd_weight=0.25,
        temperature=1.3,
        backend="triton",
    )
    out.loss.backward()

    assert torch.equal(out.teacher_ce, torch.zeros_like(out.teacher_ce))

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, _ht_ref, w_ref, _wt_ref = reference_distill_loss(
        torch,
        hs.cpu(),
        None,
        weight.cpu(),
        labels.cpu(),
        kl_weight=0.25,
        kl_temperature=1.3,
        teacher_mode="precomputed",
        logits_teacher=teacher_logits.cpu(),
    )
    ref_loss.backward()

    assert_loss_components_match(
        torch,
        out,
        (ref_loss, ref_s, torch.zeros_like(ref_t), ref_kl),
        atol=4e-3,
        rtol=4e-3,
    )
    assert_grad_cosine(torch, hs.grad, hs_ref.grad, min_cos=0.995, name="student_hidden")
    assert_grad_cosine(torch, weight.grad, w_ref.grad, min_cos=0.995, name="weight")


@pytest.mark.gpu
@pytest.mark.parametrize("teacher_mode", ["tied", "separate", "precomputed"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_public_reduction_modes_match_fp64_reference(teacher_mode, reduction):
    if teacher_mode == "tied":
        hs, ht, weight, labels = make_tied_inputs(
            torch, BT=13, H=24, V=193, dtype=torch.float16, seed=7201,
        )
        teacher = TiedTeacher(ht)
        ref_kwargs = dict(ht=ht.cpu(), weight_teacher=None, logits_teacher=None)
    elif teacher_mode == "separate":
        hs, ht, weight, teacher_weight, labels = make_separate_inputs(
            torch, BT=13, Hs=24, Ht=32, V=193, dtype=torch.float16, seed=7202,
        )
        teacher = SeparateTeacher(ht, teacher_weight)
        ref_kwargs = dict(ht=ht.cpu(), weight_teacher=teacher_weight.cpu(), logits_teacher=None)
    else:
        hs, weight, teacher_logits, labels = make_precomputed_inputs(
            torch, BT=13, H=24, V=193, dtype=torch.float16, seed=7203,
        )
        teacher = PrecomputedTeacher(teacher_logits)
        ref_kwargs = dict(ht=None, weight_teacher=None, logits_teacher=teacher_logits.cpu())

    labels = labels.clone()
    labels[2] = -100
    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        kd_weight=0.4,
        temperature=1.6,
        reduction=reduction,
        config=KernelConfig(chunk_size=5, quantize_grad_weight=False),
        backend="triton",
    )

    ref_loss, ref_s, ref_t, ref_kl, *_ = reference_distill_loss(
        torch,
        hs.cpu(),
        ref_kwargs["ht"],
        weight.cpu(),
        labels.cpu(),
        reduction=reduction,
        kl_weight=0.4,
        kl_temperature=1.6,
        teacher_mode=teacher_mode,
        weight_teacher=ref_kwargs["weight_teacher"],
        logits_teacher=ref_kwargs["logits_teacher"],
    )

    expected_teacher_ce = ref_t if teacher_mode == "tied" else torch.zeros_like(ref_t)
    assert_loss_components_match(
        torch,
        out,
        (ref_loss, ref_s, expected_teacher_ce, ref_kl),
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("profile", ["fast", "debug"])
def test_public_profiles_run_finite_smoke(profile):
    hs, ht, weight, labels = make_tied_inputs(
        torch, BT=8, H=16, V=129, dtype=torch.float16, seed=7204,
    )
    out = distillation_loss(
        hs,
        weight,
        labels,
        TiedTeacher(ht),
        kd_weight=0.2,
        temperature=1.2,
        profile=profile,
        backend="triton",
    )
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.student_ce)
    assert torch.isfinite(out.teacher_ce)
    assert torch.isfinite(out.kl)
