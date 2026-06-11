"""Mode 'precomputed' (B) correctness — teacher logits given as cached tensor."""
from __future__ import annotations

import pytest

from tests.utils.assertions import assert_grad_cosine, assert_zero_scalar
from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed
from tests.utils.factories import make_precomputed_inputs
from tests.utils.reference import cosine_sim, reference_distill_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.ops.cross_entropy import distill_cross_entropy


def _make_inputs(BT, H_s, V, seed):
    set_seed(torch, seed)
    hs = (torch.randn(BT, H_s, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    w_s = (torch.randn(V, H_s, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    # Teacher logits: pre-computed, NO grad path.
    logits_t = (torch.randn(BT, V, device="cuda", dtype=torch.float16) * 0.1)
    target = torch.randint(0, V, (BT,), device="cuda")
    target[::6] = -100
    return hs, w_s, logits_t, target


@pytest.mark.gpu
def test_precomputed_pure_kd_zero_teacher_loss():
    """teacher_loss_weight=0 → loss_t == 0, student-only path matches reference."""
    BT, H_s, V = 16, 64, 1024
    hs, w_s, logits_t, target = _make_inputs(BT, H_s, V, seed=51)

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs, None, w_s, target,
        lambda_student=1.0,
        kl_weight=0.4,
        kl_temperature=1.5,
        teacher_mode="precomputed",
        logits_teacher=logits_t,
        teacher_loss_weight=0.0,
    )
    loss.backward()

    assert torch.equal(loss_t, torch.zeros_like(loss_t))

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, _, w_s_ref, _ = reference_distill_loss(
        torch,
        hs.cpu(), None, w_s.cpu(), target.cpu(),
        lambda_student=1.0,
        kl_weight=0.4,
        kl_temperature=1.5,
        teacher_mode="precomputed",
        logits_teacher=logits_t.cpu(),
        teacher_loss_weight=0.0,
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=2e-3, rtol=2e-3)
    assert cosine_sim(torch, hs.grad.cpu(), hs_ref.grad) > 0.995
    assert cosine_sim(torch, w_s.grad.cpu(), w_s_ref.grad) > 0.995


@pytest.mark.gpu
def test_precomputed_default_teacher_loss_weight_is_pure_kd():
    hs, w_s, logits_t, target = make_precomputed_inputs(
        torch,
        BT=16,
        H=48,
        V=769,
        dtype=torch.float16,
        seed=55,
    )

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs,
        None,
        w_s,
        target,
        lambda_student=1.0,
        kl_weight=0.35,
        kl_temperature=1.7,
        teacher_mode="precomputed",
        logits_teacher=logits_t,
    )
    loss.backward()

    assert_zero_scalar(torch, loss_t, name="loss_t")

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, _, w_s_ref, _ = reference_distill_loss(
        torch,
        hs.cpu(),
        None,
        w_s.cpu(),
        target.cpu(),
        lambda_student=1.0,
        kl_weight=0.35,
        kl_temperature=1.7,
        teacher_mode="precomputed",
        logits_teacher=logits_t.cpu(),
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=3e-3, rtol=3e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=3e-3, rtol=3e-3)
    assert_grad_cosine(torch, hs.grad, hs_ref.grad, min_cos=0.995, name="h_student")
    assert_grad_cosine(torch, w_s.grad, w_s_ref.grad, min_cos=0.995, name="weight")


@pytest.mark.gpu
def test_precomputed_teacher_loss_monitoring():
    """teacher_loss_weight=0.5 → loss_t computed but no grad path through logits_t."""
    BT, H_s, V = 16, 64, 1024
    hs, w_s, logits_t, target = _make_inputs(BT, H_s, V, seed=52)

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs, None, w_s, target,
        lambda_student=1.0,
        kl_weight=0.0,  # disable KL for clear monitoring test
        teacher_mode="precomputed",
        logits_teacher=logits_t,
        teacher_loss_weight=0.5,
    )
    loss.backward()

    # loss_t is the teacher CE (unscaled). loss = lambda_s * loss_s + 0.5 * loss_t.
    assert loss_t.item() > 0.0

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, _, w_s_ref, _ = reference_distill_loss(
        torch,
        hs.cpu(), None, w_s.cpu(), target.cpu(),
        lambda_student=1.0,
        kl_weight=0.0,
        teacher_mode="precomputed",
        logits_teacher=logits_t.cpu(),
        teacher_loss_weight=0.5,
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(loss_t.cpu(), ref_t.float(), atol=2e-3, rtol=2e-3)
    # Student grad still correct (only path that flows).
    assert cosine_sim(torch, hs.grad.cpu(), hs_ref.grad) > 0.995


@pytest.mark.gpu
def test_precomputed_wrong_logits_shape_raises():
    BT, H_s, V = 8, 32, 256
    hs, w_s, _, target = _make_inputs(BT, H_s, V, seed=53)

    # Wrong V dimension
    bad_logits = torch.randn(BT, V + 1, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="logits_teacher"):
        distill_cross_entropy(
            hs, None, w_s, target,
            teacher_mode="precomputed",
            logits_teacher=bad_logits,
        )

    # Wrong BT dimension
    bad_logits = torch.randn(BT + 2, V, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="logits_teacher"):
        distill_cross_entropy(
            hs, None, w_s, target,
            teacher_mode="precomputed",
            logits_teacher=bad_logits,
        )


@pytest.mark.gpu
def test_precomputed_logits_teacher_requires_grad_raises():
    BT, H_s, V = 8, 32, 256
    hs, w_s, _, target = _make_inputs(BT, H_s, V, seed=54)
    bad_logits = torch.randn(BT, V, device="cuda", dtype=torch.float16, requires_grad=True)
    with pytest.raises(ValueError, match="requires_grad|grad"):
        distill_cross_entropy(
            hs, None, w_s, target,
            teacher_mode="precomputed",
            logits_teacher=bad_logits,
        )


