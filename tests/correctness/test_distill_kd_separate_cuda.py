"""Mode 'separate' (A) correctness — student and teacher use independent weights."""
from __future__ import annotations

import pytest

from tests.utils.assertions import assert_grad_cosine, assert_no_grad, assert_zero_scalar
from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed
from tests.utils.factories import make_separate_inputs
from tests.utils.reference import cosine_sim, reference_distill_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.ops.cross_entropy import distill_cross_entropy


def _make_inputs(BT, H_s, H_t, V, seed, w_teacher_requires_grad=False):
    set_seed(torch, seed)
    hs = (torch.randn(BT, H_s, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H_t, device="cuda", dtype=torch.float16) * 0.1)
    # Teacher hidden states typically don't require grad in KD.
    w_s = (torch.randn(V, H_s, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    w_t = (torch.randn(V, H_t, device="cuda", dtype=torch.float16) * 0.1)
    if w_teacher_requires_grad:
        w_t = w_t.requires_grad_(True)
    target = torch.randint(0, V, (BT,), device="cuda")
    target[::6] = -100
    return hs, ht, w_s, w_t, target


@pytest.mark.gpu
def test_separate_pure_kd_zero_teacher_loss():
    """teacher_loss_weight=0 → loss_t == 0, grad_W only from student branch."""
    BT, H_s, H_t, V = 16, 64, 64, 1024
    hs, ht, w_s, w_t, target = _make_inputs(BT, H_s, H_t, V, seed=42)

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs, ht, w_s, target,
        lambda_student=1.0,
        kl_weight=0.4,
        kl_temperature=1.5,
        teacher_mode="separate",
        weight_teacher=w_t,
        teacher_loss_weight=0.0,
    )
    loss.backward()

    assert torch.equal(loss_t, torch.zeros_like(loss_t))

    # Reference: separate mode, teacher_loss_weight=0.
    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, w_s_ref, w_t_ref = reference_distill_loss(
        torch,
        hs.cpu(), ht.cpu(), w_s.cpu(), target.cpu(),
        lambda_student=1.0,
        kl_weight=0.4,
        kl_temperature=1.5,
        teacher_mode="separate",
        weight_teacher=w_t.cpu(),
        teacher_loss_weight=0.0,
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=2e-3, rtol=2e-3)
    assert cosine_sim(torch, hs.grad.cpu(), hs_ref.grad) > 0.995
    assert cosine_sim(torch, w_s.grad.cpu(), w_s_ref.grad) > 0.995

    # Teacher weight has no grad (not in autograd graph for student-only path).
    assert w_t.grad is None


@pytest.mark.gpu
def test_separate_default_teacher_loss_weight_is_pure_kd():
    hs, ht, w_s, w_t, target = make_separate_inputs(
        torch,
        BT=16,
        Hs=48,
        Ht=64,
        V=769,
        dtype=torch.float16,
        seed=46,
        weight_teacher_requires_grad=True,
    )

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs,
        ht,
        w_s,
        target,
        lambda_student=1.0,
        kl_weight=0.35,
        kl_temperature=1.7,
        teacher_mode="separate",
        weight_teacher=w_t,
    )
    loss.backward()

    assert_zero_scalar(torch, loss_t, name="loss_t")
    assert_no_grad(w_t, name="weight_teacher")

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, _ht_ref, w_s_ref, _w_t_ref = reference_distill_loss(
        torch,
        hs.cpu(),
        ht.cpu(),
        w_s.cpu(),
        target.cpu(),
        lambda_student=1.0,
        kl_weight=0.35,
        kl_temperature=1.7,
        teacher_mode="separate",
        weight_teacher=w_t.cpu(),
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=3e-3, rtol=3e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=3e-3, rtol=3e-3)
    assert_grad_cosine(torch, hs.grad, hs_ref.grad, min_cos=0.995, name="h_student")
    assert_grad_cosine(torch, w_s.grad, w_s_ref.grad, min_cos=0.995, name="weight")


@pytest.mark.gpu
def test_separate_co_distillation_teacher_loss_weight():
    """teacher_loss_weight=0.1 with W_teacher.requires_grad=True → both grads flow."""
    BT, H_s, H_t, V = 16, 64, 64, 1024
    hs, ht, w_s, w_t, target = _make_inputs(BT, H_s, H_t, V, seed=43, w_teacher_requires_grad=True)

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs, ht, w_s, target,
        lambda_student=1.0,
        kl_weight=0.4,
        kl_temperature=1.5,
        teacher_mode="separate",
        weight_teacher=w_t,
        teacher_loss_weight=0.1,
    )
    loss.backward()

    # Teacher CE term is non-zero (scaled by 0.1).
    assert loss_t.item() != 0.0

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, w_s_ref, w_t_ref = reference_distill_loss(
        torch,
        hs.cpu(), ht.cpu(), w_s.cpu(), target.cpu(),
        lambda_student=1.0,
        kl_weight=0.4,
        kl_temperature=1.5,
        teacher_mode="separate",
        weight_teacher=w_t.cpu(),
        teacher_loss_weight=0.1,
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(loss_t.cpu(), ref_t.float(), atol=2e-3, rtol=2e-3)
    assert cosine_sim(torch, hs.grad.cpu(), hs_ref.grad) > 0.995
    assert cosine_sim(torch, w_s.grad.cpu(), w_s_ref.grad) > 0.995
    assert w_t.grad is not None
    assert cosine_sim(torch, w_t.grad.cpu(), w_t_ref.grad) > 0.995


@pytest.mark.gpu
def test_separate_w_teacher_requires_grad_but_loss_weight_zero():
    """teacher_loss_weight=0 + W_teacher.requires_grad=True → W_teacher.grad stays None."""
    BT, H_s, H_t, V = 12, 32, 32, 512
    hs, ht, w_s, w_t, target = _make_inputs(BT, H_s, H_t, V, seed=44, w_teacher_requires_grad=True)

    loss, *_ = distill_cross_entropy(
        hs, ht, w_s, target,
        lambda_student=1.0,
        kl_weight=0.3,
        teacher_mode="separate",
        weight_teacher=w_t,
        teacher_loss_weight=0.0,
    )
    loss.backward()
    # No teacher CE → no grad for w_t even though it requires grad.
    assert w_t.grad is None


@pytest.mark.gpu
def test_separate_different_hidden_dims():
    """Student H_s != teacher H_t (typical when distilling smaller model)."""
    BT, H_s, H_t, V = 16, 32, 96, 1024
    hs, ht, w_s, w_t, target = _make_inputs(BT, H_s, H_t, V, seed=45)

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs, ht, w_s, target,
        lambda_student=1.0,
        kl_weight=0.4,
        kl_temperature=1.5,
        teacher_mode="separate",
        weight_teacher=w_t,
        teacher_loss_weight=0.0,
    )
    loss.backward()

    for tensor in [loss, loss_s, kl, hs.grad, w_s.grad]:
        assert torch.isfinite(tensor).all()


