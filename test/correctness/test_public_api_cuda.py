from __future__ import annotations

import pytest

from utils.assertions import assert_grad_close, assert_loss_components_match, assert_scalar_close
from utils.env import pytest_skip_if_no_cuda_kernel
from utils.factories import (
    make_precomputed_hidden_inputs,
    make_precomputed_logits_inputs,
    make_separate_inputs,
    make_tied_inputs,
)
from utils.reference import reference_distillation_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel import KernelConfig, PrecomputedTeacher, SeparateTeacher, TiedTeacher, distillation_loss


def _noncontiguous_randn(shape, *, dtype, device, requires_grad=False, seed=0):
    torch.manual_seed(seed)
    base = torch.randn(shape[1], shape[0], dtype=dtype, device=device)
    tensor = base.t()
    assert tensor.shape == shape
    assert not tensor.is_contiguous()
    return tensor.requires_grad_(requires_grad)


def _run_reference(hs, weight, labels, ref_kwargs, *, reduction="mean"):
    ref = reference_distillation_loss(
        torch,
        hs.cpu(),
        weight.cpu(),
        labels.cpu(),
        kl_weight=0.35,
        kl_temperature=1.4,
        reduction=reduction,
        **{
            key: value.cpu() if hasattr(value, "cpu") else value
            for key, value in ref_kwargs.items()
        },
    )
    ref[0].backward()
    return ref


def _run_reference_with_params(
    hs,
    weight,
    labels,
    ref_kwargs,
    *,
    kl_weight,
    kl_temperature,
    reduction="mean",
):
    ref = reference_distillation_loss(
        torch,
        hs.cpu(),
        weight.cpu(),
        labels.cpu(),
        kl_weight=kl_weight,
        kl_temperature=kl_temperature,
        reduction=reduction,
        **{
            key: value.cpu() if hasattr(value, "cpu") else value
            for key, value in ref_kwargs.items()
        },
    )
    ref[0].backward()
    return ref


def _assert_common_grads(hs, weight, ref):
    assert hs.grad is not None
    assert weight.grad is not None
    assert_grad_close(torch, hs.grad, ref[4].grad, atol=5e-2, rtol=5e-2, name="student_hidden")
    assert_grad_close(torch, weight.grad, ref[6].grad, atol=5e-2, rtol=5e-2, name="weight")


def _make_mode_inputs(teacher_mode: str, *, dtype, seed, V=257):
    teacher_ce_weight = None
    teacher_weight_for_grad = None
    if teacher_mode == "tied":
        hs, ht, weight, labels = make_tied_inputs(
            torch,
            BT=16,
            H=32,
            V=V,
            dtype=dtype,
            seed=seed,
            device="cuda",
        )
        teacher = TiedTeacher(ht)
        ref_kwargs = dict(teacher_mode="tied", teacher_hidden=ht)
    elif teacher_mode in ("separate", "separate-full"):
        hs, ht, weight, teacher_weight, labels = make_separate_inputs(
            torch,
            BT=16,
            Hs=32,
            Ht=48,
            V=V,
            dtype=dtype,
            seed=seed,
            device="cuda",
            teacher_requires_grad=teacher_mode == "separate-full",
            teacher_weight_requires_grad=teacher_mode == "separate-full",
        )
        if teacher_mode == "separate-full":
            teacher_ce_weight = 1.0
            teacher_weight_for_grad = teacher_weight
        teacher = SeparateTeacher(ht, teacher_weight)
        ref_kwargs = dict(
            teacher_mode="separate",
            teacher_hidden=ht,
            teacher_weight=teacher_weight,
            teacher_ce_weight=teacher_ce_weight,
        )
    elif teacher_mode == "precomputed-logits":
        hs, weight, teacher_logits, labels = make_precomputed_logits_inputs(
            torch,
            BT=16,
            H=32,
            V=V,
            dtype=dtype,
            seed=seed,
            device="cuda",
        )
        teacher_logits = teacher_logits.float()
        teacher = PrecomputedTeacher(logits=teacher_logits)
        ref_kwargs = dict(teacher_mode="precomputed", teacher_logits=teacher_logits)
    else:
        hs, weight, teacher_hidden, teacher_weight, labels = make_precomputed_hidden_inputs(
            torch,
            BT=16,
            Hs=32,
            Ht=48,
            V=V,
            dtype=dtype,
            seed=seed,
            device="cuda",
        )
        teacher = PrecomputedTeacher(teacher_hidden=teacher_hidden, teacher_weight=teacher_weight)
        ref_kwargs = dict(
            teacher_mode="precomputed",
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        )
    return hs, weight, labels, teacher, teacher_ce_weight, teacher_weight_for_grad, ref_kwargs


@pytest.mark.gpu
@pytest.mark.parametrize(
    "teacher_mode",
    ["tied", "separate", "separate-full", "precomputed-logits", "precomputed-hidden"],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_public_triton_backend_matches_fp64_reference(teacher_mode: str, reduction: str):
    teacher_ce_weight = None
    teacher_weight_for_grad = None
    if teacher_mode == "tied":
        hs, ht, weight, labels = make_tied_inputs(
            torch,
            BT=16,
            H=32,
            V=257,
            dtype=torch.float16,
            seed=201,
            device="cuda",
        )
        teacher = TiedTeacher(ht)
        ref_kwargs = dict(teacher_mode="tied", teacher_hidden=ht)
    elif teacher_mode in ("separate", "separate-full"):
        hs, ht, weight, teacher_weight, labels = make_separate_inputs(
            torch,
            BT=16,
            Hs=32,
            Ht=48,
            V=257,
            dtype=torch.float16,
            seed=202,
            device="cuda",
            teacher_requires_grad=teacher_mode == "separate-full",
            teacher_weight_requires_grad=teacher_mode == "separate-full",
        )
        if teacher_mode == "separate-full":
            teacher_ce_weight = 1.0
            teacher_weight_for_grad = teacher_weight
        teacher = SeparateTeacher(ht, teacher_weight)
        ref_kwargs = dict(
            teacher_mode="separate",
            teacher_hidden=ht,
            teacher_weight=teacher_weight,
            teacher_ce_weight=teacher_ce_weight,
        )
    elif teacher_mode == "precomputed-logits":
        hs, weight, teacher_logits, labels = make_precomputed_logits_inputs(
            torch,
            BT=16,
            H=32,
            V=257,
            dtype=torch.float16,
            seed=203,
            device="cuda",
        )
        teacher = PrecomputedTeacher(logits=teacher_logits)
        ref_kwargs = dict(teacher_mode="precomputed", teacher_logits=teacher_logits)
    else:
        hs, weight, teacher_hidden, teacher_weight, labels = make_precomputed_hidden_inputs(
            torch,
            BT=16,
            Hs=32,
            Ht=48,
            V=257,
            dtype=torch.float16,
            seed=204,
            device="cuda",
        )
        teacher = PrecomputedTeacher(teacher_hidden=teacher_hidden, teacher_weight=teacher_weight)
        ref_kwargs = dict(
            teacher_mode="precomputed",
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        )

    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=1.4,
        reduction=reduction,
        backend="triton",
        config=KernelConfig(chunk_size=7),
    )
    out.loss.backward()

    ref = _run_reference(hs, weight, labels, ref_kwargs, reduction=reduction)

    assert_loss_components_match(torch, out, ref, atol=5e-3, rtol=5e-3)
    assert_scalar_close(torch, out.loss, ref[0], atol=5e-3, rtol=5e-3, name="loss")
    _assert_common_grads(hs, weight, ref)
    if teacher_mode == "tied":
        assert_grad_close(torch, ht.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden")
    if teacher_mode == "separate-full":
        assert_grad_close(torch, ht.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden")
        assert teacher_weight_for_grad is not None
        assert_grad_close(
            torch,
            teacher_weight_for_grad.grad,
            ref[7].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_weight",
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "teacher_mode",
    ["tied", "separate", "separate-full", "precomputed-logits", "precomputed-hidden"],
)
def test_public_triton_backend_skips_kl_when_weight_is_zero(teacher_mode: str):
    hs, weight, labels, teacher, teacher_ce_weight, teacher_weight_for_grad, ref_kwargs = _make_mode_inputs(
        teacher_mode,
        dtype=torch.float16,
        seed=401,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.0,
        kl_temperature=1.4,
        backend="triton",
        config=KernelConfig(chunk_size=7),
    )
    out.loss.backward()

    ref = _run_reference_with_params(
        hs,
        weight,
        labels,
        ref_kwargs,
        kl_weight=0.0,
        kl_temperature=1.4,
    )

    assert_scalar_close(torch, out.loss, ref[0], atol=5e-3, rtol=5e-3, name="loss")
    assert_scalar_close(torch, out.student_ce, ref[1], atol=5e-3, rtol=5e-3, name="student_ce")
    assert_scalar_close(torch, out.teacher_ce, ref[2], atol=5e-3, rtol=5e-3, name="teacher_ce")
    assert_scalar_close(torch, out.kl, torch.zeros_like(out.kl), atol=0.0, rtol=0.0, name="kl")
    _assert_common_grads(hs, weight, ref)
    if teacher_mode == "tied":
        assert_grad_close(torch, teacher.hidden.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden")
    if teacher_mode == "separate-full":
        assert_grad_close(torch, teacher.hidden.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden")
        assert teacher_weight_for_grad is not None
        assert_grad_close(
            torch,
            teacher_weight_for_grad.grad,
            ref[7].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_weight",
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "teacher_mode",
    ["tied", "separate", "separate-full", "precomputed-logits", "precomputed-hidden"],
)
@pytest.mark.parametrize("max_fused_size", [128, 512])
def test_public_triton_backend_temperature_one_matches_reference(teacher_mode: str, max_fused_size: int):
    hs, weight, labels, teacher, teacher_ce_weight, teacher_weight_for_grad, ref_kwargs = _make_mode_inputs(
        teacher_mode,
        dtype=torch.float16,
        seed=501 + max_fused_size,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=1.0,
        backend="triton",
        config=KernelConfig(chunk_size=7, max_fused_size=max_fused_size),
    )
    out.loss.backward()

    ref = _run_reference_with_params(
        hs,
        weight,
        labels,
        ref_kwargs,
        kl_weight=0.35,
        kl_temperature=1.0,
    )

    assert_loss_components_match(torch, out, ref, atol=5e-3, rtol=5e-3)
    assert_scalar_close(torch, out.loss, ref[0], atol=5e-3, rtol=5e-3, name="loss")
    _assert_common_grads(hs, weight, ref)
    if teacher_mode == "tied":
        assert_grad_close(torch, teacher.hidden.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden")
    if teacher_mode == "separate-full":
        assert_grad_close(torch, teacher.hidden.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden")
        assert teacher_weight_for_grad is not None
        assert_grad_close(
            torch,
            teacher_weight_for_grad.grad,
            ref[7].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_weight",
        )


@pytest.mark.gpu
def test_tied_triton_backend_teacher_ce_zero_keeps_teacher_grad_zero():
    hs, ht, weight, labels = make_tied_inputs(
        torch,
        BT=16,
        H=32,
        V=257,
        dtype=torch.float16,
        seed=601,
        device="cuda",
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        TiedTeacher(ht),
        teacher_ce_weight=0.0,
        kl_weight=0.35,
        kl_temperature=1.4,
        backend="triton",
        config=KernelConfig(chunk_size=7),
    )
    out.loss.backward()

    assert ht.grad is not None
    assert_grad_close(torch, ht.grad, torch.zeros_like(ht.grad), atol=0.0, rtol=0.0, name="teacher_hidden")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "teacher_mode",
    ["tied", "separate", "separate-full", "precomputed-logits", "precomputed-hidden"],
)
def test_public_triton_backend_preserves_noncontiguous_input_gradients(teacher_mode: str):
    BT, Hs, Ht, V = 16, 32, 48, 257
    labels = torch.randint(0, V, (BT,), device="cuda")
    labels[::4] = -100
    teacher_ce_weight = None
    teacher_weight_for_grad = None

    hs = _noncontiguous_randn((BT, Hs), dtype=torch.float16, device="cuda", requires_grad=True, seed=301)
    weight = _noncontiguous_randn((V, Hs), dtype=torch.float16, device="cuda", requires_grad=True, seed=302)

    if teacher_mode == "tied":
        ht = _noncontiguous_randn((BT, Hs), dtype=torch.float16, device="cuda", requires_grad=True, seed=303)
        teacher = TiedTeacher(ht)
        ref_kwargs = dict(teacher_mode="tied", teacher_hidden=ht)
    elif teacher_mode in ("separate", "separate-full"):
        ht = _noncontiguous_randn(
            (BT, Ht),
            dtype=torch.float16,
            device="cuda",
            requires_grad=teacher_mode == "separate-full",
            seed=304,
        )
        teacher_weight = _noncontiguous_randn(
            (V, Ht),
            dtype=torch.float16,
            device="cuda",
            requires_grad=teacher_mode == "separate-full",
            seed=305,
        )
        if teacher_mode == "separate-full":
            teacher_ce_weight = 1.0
            teacher_weight_for_grad = teacher_weight
        teacher = SeparateTeacher(ht, teacher_weight)
        ref_kwargs = dict(
            teacher_mode="separate",
            teacher_hidden=ht,
            teacher_weight=teacher_weight,
            teacher_ce_weight=teacher_ce_weight,
        )
    elif teacher_mode == "precomputed-logits":
        teacher_logits = _noncontiguous_randn(
            (BT, V),
            dtype=torch.float16,
            device="cuda",
            requires_grad=False,
            seed=306,
        )
        teacher = PrecomputedTeacher(logits=teacher_logits)
        ref_kwargs = dict(teacher_mode="precomputed", teacher_logits=teacher_logits)
    else:
        teacher_hidden = _noncontiguous_randn(
            (BT, Ht),
            dtype=torch.float16,
            device="cuda",
            requires_grad=False,
            seed=307,
        )
        teacher_weight = _noncontiguous_randn(
            (V, Ht),
            dtype=torch.float16,
            device="cuda",
            requires_grad=False,
            seed=308,
        )
        teacher = PrecomputedTeacher(teacher_hidden=teacher_hidden, teacher_weight=teacher_weight)
        ref_kwargs = dict(
            teacher_mode="precomputed",
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        )

    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=1.4,
        backend="triton",
        config=KernelConfig(chunk_size=7),
    )
    out.loss.backward()

    ref = _run_reference(hs, weight, labels, ref_kwargs)

    _assert_common_grads(hs, weight, ref)
    if teacher_mode == "tied":
        assert ht.grad is not None
        assert_grad_close(torch, ht.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden")
    if teacher_mode == "separate-full":
        assert ht.grad is not None
        assert teacher_weight_for_grad is not None
        assert teacher_weight_for_grad.grad is not None
        assert_grad_close(torch, ht.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden")
        assert_grad_close(
            torch,
            teacher_weight_for_grad.grad,
            ref[7].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_weight",
        )
