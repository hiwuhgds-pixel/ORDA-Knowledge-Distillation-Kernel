from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed
from tests.utils.reference import cosine_sim, reference_distill_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.utils.dispatcher import dynamic_chunk


def _build(BT, H, V, seed):
    set_seed(torch, seed)
    hs = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    w = (torch.randn(V, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    return hs, ht, w


@pytest.mark.gpu
def test_all_ignored_targets_produce_zero_loss_and_grads():
    BT, H, V = 16, 32, 257
    hs, ht, w = _build(BT, H, V, seed=1)
    target = torch.full((BT,), -100, device="cuda", dtype=torch.long)

    loss, loss_s, loss_t, kl = dynamic_chunk(
        hs, ht, w, target,
        lambda_student=1.0, kl_weight=0.4, kl_temperature=1.5,
    )
    loss.backward()

    for tensor in [loss, loss_s, loss_t, kl]:
        assert torch.equal(tensor, torch.zeros_like(tensor))
    assert torch.all(hs.grad == 0)
    assert torch.all(ht.grad == 0)
    assert torch.all(w.grad == 0)


@pytest.mark.gpu
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_sparse_valid_tokens_match_reference_and_zero_grad_on_ignored(reduction: str):
    BT, H, V = 32, 32, 503
    hs, ht, w = _build(BT, H, V, seed=2)
    target = torch.full((BT,), -100, device="cuda", dtype=torch.long)
    valid_idx = torch.tensor([3, 17], device="cuda")
    target[valid_idx] = torch.randint(0, V, (valid_idx.numel(),), device="cuda")

    loss, loss_s, loss_t, kl = dynamic_chunk(
        hs, ht, w, target,
        lambda_student=1.0,
        reduction=reduction,
        kl_weight=0.4,
        kl_temperature=1.5,
    )
    loss.backward()

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, w_ref, _ = reference_distill_loss(
        torch, hs.cpu(), ht.cpu(), w.cpu(), target.cpu(),
        lambda_student=1.0, reduction=reduction,
        kl_weight=0.4, kl_temperature=1.5,
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(loss_t.cpu(), ref_t.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=3e-3, rtol=3e-3)

    ignored_mask = target == -100
    assert torch.all(hs.grad[ignored_mask] == 0)
    assert torch.all(ht.grad[ignored_mask] == 0)

    # Valid-row gradients should at least point the same direction as reference.
    ignored_mask_cpu = ignored_mask.cpu()
    hs_grad_valid_orda = hs.grad[~ignored_mask].cpu()
    hs_grad_valid_ref = hs_ref.grad[~ignored_mask_cpu]
    assert cosine_sim(torch, hs_grad_valid_orda, hs_grad_valid_ref) > 0.99


@pytest.mark.gpu
def test_tail_row_ignored_does_not_corrupt_dynamic_chunk_tail():
    # Force a configuration where the dynamic chunker may split the tail off.
    BT, H, V = 65, 32, 1009  # BT odd to ensure chunking does not align perfectly.
    hs, ht, w = _build(BT, H, V, seed=3)
    target = torch.randint(0, V, (BT,), device="cuda")
    target[-1] = -100  # only tail row ignored

    loss, loss_s, loss_t, kl = dynamic_chunk(
        hs, ht, w, target,
        lambda_student=1.0, kl_weight=0.3, kl_temperature=1.4,
    )
    loss.backward()

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, w_ref, _ = reference_distill_loss(
        torch, hs.cpu(), ht.cpu(), w.cpu(), target.cpu(),
        lambda_student=1.0, kl_weight=0.3, kl_temperature=1.4,
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=3e-3, rtol=3e-3)
    assert torch.all(hs.grad[-1] == 0)
    assert torch.all(ht.grad[-1] == 0)
    assert cosine_sim(torch, w.grad.cpu(), w_ref.grad) > 0.995


