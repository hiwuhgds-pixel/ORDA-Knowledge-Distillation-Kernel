from __future__ import annotations

import pytest

from tests.utils.assertions import assert_grad_cosine, assert_loss_components_match, assert_scalar_close
from tests.utils.env import pytest_skip_if_no_cuda_kernel
from tests.utils.factories import make_tied_inputs
from tests.utils.reference import reference_distill_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.ops.cross_entropy import DistillCrossEntropyLoss
from orda_ce_kernel.utils.dispatcher import dynamic_chunk


@pytest.mark.gpu
def test_total_loss_uses_raw_components_and_explicit_weights():
    hs, ht, weight, target = make_tied_inputs(
        torch, BT=18, H=32, V=257, dtype=torch.float16, seed=6101,
    )
    lambda_student = 0.7
    teacher_loss_weight = 0.25
    kl_weight = 0.4

    loss, loss_s, loss_t, kl = dynamic_chunk(
        hs,
        ht,
        weight,
        target,
        lambda_student=lambda_student,
        teacher_loss_weight=teacher_loss_weight,
        kl_weight=kl_weight,
        kl_temperature=1.6,
    )

    expected = lambda_student * loss_s + teacher_loss_weight * loss_t + kl_weight * kl
    assert_scalar_close(torch, loss, expected, atol=2e-3, rtol=2e-3, name="weighted total loss")


@pytest.mark.gpu
def test_backward_respects_upstream_grad_scale():
    hs0, ht0, weight0, target = make_tied_inputs(
        torch, BT=20, H=32, V=503, dtype=torch.float16, seed=6102,
    )

    def run(scale: float):
        hs = hs0.detach().clone().requires_grad_(True)
        ht = ht0.detach().clone().requires_grad_(True)
        weight = weight0.detach().clone().requires_grad_(True)
        loss, *_ = dynamic_chunk(hs, ht, weight, target, kl_weight=0.35, kl_temperature=1.3)
        (loss * scale).backward()
        return hs.grad.detach(), ht.grad.detach(), weight.grad.detach()

    hs_grad, ht_grad, weight_grad = run(1.0)
    hs_scaled, ht_scaled, weight_scaled = run(3.0)

    assert torch.allclose(hs_scaled.float(), hs_grad.float() * 3.0, atol=2e-3, rtol=2e-3)
    assert torch.allclose(ht_scaled.float(), ht_grad.float() * 3.0, atol=2e-3, rtol=2e-3)
    assert torch.allclose(weight_scaled.float(), weight_grad.float() * 3.0, atol=3e-3, rtol=3e-3)


@pytest.mark.gpu
def test_distill_cross_entropy_loss_module_matches_functional_api():
    hs, ht, weight, target = make_tied_inputs(
        torch, BT=16, H=32, V=257, dtype=torch.float16, seed=6103,
    )
    module = DistillCrossEntropyLoss(
        lambda_student=1.2,
        label_smoothing=0.05,
        chunk_size=7,
        kl_weight=0.3,
        kl_temperature=1.4,
        teacher_mode="tied",
    )

    loss, loss_s, loss_t, kl = module(hs, ht, weight, target)
    loss.backward()

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, w_ref, _ = reference_distill_loss(
        torch,
        hs.cpu(),
        ht.cpu(),
        weight.cpu(),
        target.cpu(),
        lambda_student=1.2,
        label_smoothing=0.05,
        kl_weight=0.3,
        kl_temperature=1.4,
        teacher_mode="tied",
    )
    ref_loss.backward()

    assert_scalar_close(torch, loss, ref_loss, atol=3e-3, rtol=3e-3, name="loss")
    assert_loss_components_match(torch, (loss, loss_s, loss_t, kl), (ref_loss, ref_s, ref_t, ref_kl), atol=3e-3, rtol=3e-3)
    assert_grad_cosine(torch, hs.grad, hs_ref.grad, min_cos=0.995, name="h_student")
    assert_grad_cosine(torch, ht.grad, ht_ref.grad, min_cos=0.995, name="h_teacher")
    assert_grad_cosine(torch, weight.grad, w_ref.grad, min_cos=0.995, name="weight")


