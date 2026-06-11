from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed
from tests.utils.reference import cosine_sim, reference_distill_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.utils.dispatcher import dynamic_chunk


def _make_inputs(BT, H, V, seed):
    set_seed(torch, seed)
    hs = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    weight = (torch.randn(V, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    target = torch.randint(0, V, (BT,), device="cuda")
    return hs, ht, weight, target


@pytest.mark.gpu
@pytest.mark.parametrize("vocab_size", [503, 1024, 5003])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5, 2.0, 4.0])
def test_orda_kl_temperature_sweep_matches_fp64_reference(vocab_size: int, temperature: float):
    """KL path stability across temperature extremes vs fp64 golden reference.

    Replaces an earlier draft that tried to backward through ``kl`` directly —
    the kernel returns ``kl_loss.detach()`` so only ``loss`` carries the graph.
    We backward through ``loss`` and compare against the fp64 reference, which
    is the same path ``test_cuda_kernel_matches_fp64_reference`` validates but
    with the temperature sweep added.
    """
    BT, H = 32, 64
    kl_weight = 0.4
    hs, ht, weight, target = _make_inputs(BT, H, vocab_size, seed=2026)

    loss, loss_s, loss_t, kl = dynamic_chunk(
        hs, ht, weight, target,
        lambda_student=1.0,
        kl_weight=kl_weight,
        kl_temperature=temperature,
    )
    loss.backward()

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, w_ref, _ = reference_distill_loss(
        torch, hs.cpu(), ht.cpu(), weight.cpu(), target.cpu(),
        lambda_student=1.0,
        kl_weight=kl_weight,
        kl_temperature=temperature,
    )
    ref_loss.backward()

    for tensor in [loss, kl, hs.grad, ht.grad, weight.grad]:
        assert torch.isfinite(tensor).all()

    # Loss tolerance loosened to 5e-3 — extreme temperatures (0.5, 4.0) shift
    # the row-sum magnitude enough that 2e-3 is too tight on fp16.
    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=5e-3, rtol=5e-3)
    assert torch.allclose(loss_t.cpu(), ref_t.float(), atol=5e-3, rtol=5e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=5e-3, rtol=5e-3)
    assert cosine_sim(torch, hs.grad.cpu(), hs_ref.grad) > 0.99
    assert cosine_sim(torch, ht.grad.cpu(), ht_ref.grad) > 0.99
    assert cosine_sim(torch, weight.grad.cpu(), w_ref.grad) > 0.99


@pytest.mark.gpu
def test_use_kl_in_kernel_false_zeroes_kl_term():
    """`use_kl_in_kernel=False` is NOT a Python KL fallback — it disables KL entirely.

    Runtime behavior is now controlled by explicit function args or module
    options, not global config setters.
    """
    BT, H, V = 24, 48, 1024
    kl_weight, T = 0.4, 1.5
    hs0, ht0, w0, target = _make_inputs(BT, H, V, seed=7)

    def run(enable_kl: bool):
        hs = hs0.detach().clone().requires_grad_(True)
        ht = ht0.detach().clone().requires_grad_(True)
        w = w0.detach().clone().requires_grad_(True)
        loss, loss_s, loss_t, kl = dynamic_chunk(
            hs, ht, w, target,
            lambda_student=1.0,
            kl_weight=kl_weight,
            kl_temperature=T,
            use_kl_in_kernel=enable_kl,
        )
        loss.backward()
        return (loss.detach(), loss_s.detach(), loss_t.detach(), kl.detach(),
                hs.grad.detach(), w.grad.detach())

    loss_on, ls_on, lt_on, kl_on, ghs_on, gw_on = run(True)
    loss_off, ls_off, lt_off, kl_off, ghs_off, gw_off = run(False)

    # CE terms identical regardless of KL flag.
    assert torch.allclose(ls_on.float(), ls_off.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(lt_on.float(), lt_off.float(), atol=2e-3, rtol=2e-3)
    # KL term is non-trivial when on, exactly zero when off.
    assert kl_on.float().abs().item() > 0.0
    assert torch.equal(kl_off, torch.zeros_like(kl_off))
    # Loss off == loss_s + loss_t (no KL contribution).
    assert torch.allclose(loss_off.float(), (ls_off + lt_off).float(), atol=2e-3, rtol=2e-3)
    # Gradients differ when KL adds a non-zero contribution. The delta is
    # small (kl_weight=0.4 * tiny kl scalar) so we compare relative L2 norm
    # rather than allclose with a fixed atol.
    delta = (ghs_on.float() - ghs_off.float()).norm().item()
    base = max(ghs_off.float().norm().item(), 1e-12)
    assert delta / base > 1e-4, (
        f"hs grad with KL enabled is indistinguishable from disabled "
        f"(rel delta {delta / base:.2e}); KL term may not be wired."
    )


