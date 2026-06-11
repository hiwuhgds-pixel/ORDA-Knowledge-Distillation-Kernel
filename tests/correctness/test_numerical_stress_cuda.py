from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.ops.cross_entropy import distill_cross_entropy


@pytest.mark.gpu
def test_extreme_logits_do_not_create_nan_or_inf():
    set_seed(torch, 2024)
    BT, H, V = 8, 32, 64
    hs_raw = torch.randn(BT, H, device="cuda", dtype=torch.float16)
    ht_raw = torch.randn(BT, H, device="cuda", dtype=torch.float16)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float16, requires_grad=True)

    with torch.no_grad():
        logits = hs_raw @ weight.detach().t()
        target = logits.argmax(dim=-1)
        scale = min(256.0, 10000.0 / max(float(logits.abs().max().item()), 1.0))

    hs = (hs_raw * scale).detach().requires_grad_(True)
    ht = (ht_raw * scale).detach().requires_grad_(True)

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs,
        ht,
        weight,
        target,
        lambda_student=1.0,
        kl_weight=0.5,
        kl_temperature=1.5,
        use_int8_quant=False,
    )
    loss.backward()

    for name, tensor in {
        "loss": loss,
        "loss_s": loss_s,
        "loss_t": loss_t,
        "kl": kl,
        "hs.grad": hs.grad,
        "ht.grad": ht.grad,
        "weight.grad": weight.grad,
    }.items():
        assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"


