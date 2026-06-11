from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed
from tests.utils.reference import cosine_sim, reference_distill_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.utils.dispatcher import dynamic_chunk


def _tied_weight_setup(BT, H, V, seed):
    set_seed(torch, seed)
    hs = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    # Single weight tensor shared by student and teacher (tied head pattern).
    weight = (torch.randn(V, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    target = torch.randint(0, V, (BT,), device="cuda")
    target[::6] = -100
    return hs, ht, weight, target


@pytest.mark.gpu
@pytest.mark.parametrize("kl_weight", [0.0, 0.4])
def test_tied_head_weight_grad_matches_reference(kl_weight: float):
    """Orda CE+KL with shared weight: grad_weight must match the fp64 reference.

    The kernel computes CE for both student and teacher through the same weight
    tensor, so grad_weight accumulates contributions from both paths. The
    reference uses `reference_distill_loss` which also computes both CE terms
    through a shared fp64 weight clone — the comparison is apples-to-apples.
    The magnitude check guards against double-counting (which would give ~2x
    the expected norm).
    """
    BT, H, V = 32, 48, 1009
    hs, ht, weight, target = _tied_weight_setup(BT, H, V, seed=5)

    loss, *_ = dynamic_chunk(
        hs, ht, weight, target,
        lambda_student=1.0,
        kl_weight=kl_weight,
        kl_temperature=1.5,
        teacher_mode="tied",  # explicit: magnitude ratio assertion below is tied-specific
    )
    loss.backward()
    grad_orda = weight.grad.detach().clone()

    # reference_distill_loss computes both student and teacher CE through w_ref
    # (same structure as the kernel), with detach_teacher_kl=True.
    ref_loss, _, _, _, hs_ref, ht_ref, w_ref, _ = reference_distill_loss(
        torch,
        hs.cpu(), ht.cpu(), weight.cpu(), target.cpu(),
        lambda_student=1.0,
        kl_weight=kl_weight,
        kl_temperature=1.5,
    )
    ref_loss.backward()
    grad_ref = w_ref.grad.detach()  # already on CPU

    assert torch.isfinite(grad_orda).all()
    assert cosine_sim(torch, grad_orda.cpu(), grad_ref) > 0.995

    # Magnitude: tied weight accumulates both CE branches, so norm should be
    # consistent with the reference — not ~2x (double-count) or ~0 (dropped).
    orda_norm = grad_orda.float().norm().item()
    ref_norm = grad_ref.float().norm().item()
    ratio = orda_norm / max(ref_norm, 1e-12)
    assert 0.9 <= ratio <= 1.1, (
        f"Tied weight grad magnitude ratio {ratio:.3f} unexpected "
        f"(expected ~1.0, got orda={orda_norm:.2e} ref={ref_norm:.2e})."
    )



