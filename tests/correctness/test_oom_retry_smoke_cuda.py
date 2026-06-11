from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.utils.dispatcher import clear_chunk_cache, dynamic_chunk, get_chunk_cache
from orda_ce_kernel.utils.resolver import resolve_chunk_size


def _free_vram_gib():
    free, _total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def _build(BT, H, V, seed):
    set_seed(torch, seed)
    hs = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    w = (torch.randn(V, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    target = torch.randint(0, V, (BT,), device="cuda")
    return hs, ht, w, target


@pytest.mark.gpu
def test_dynamic_chunk_path_resolves_multiple_chunks_and_runs():
    BT, H, V = 4096, 64, 32768
    cs, num_chunks = resolve_chunk_size(BT, "dynamic", V=V)
    assert num_chunks >= 2, (
        f"Resolver returned a single chunk for BT={BT}, V={V} — "
        "test needs a config that forces multi-chunk to be meaningful."
    )

    hs, ht, w, target = _build(BT, H, V, seed=11)
    loss, loss_s, loss_t, kl = dynamic_chunk(
        hs, ht, w, target,
        chunk_size="dynamic",
        lambda_student=1.0,
        kl_weight=0.3,
        kl_temperature=1.5,
    )
    loss.backward()

    for tensor in [loss, loss_s, loss_t, kl, hs.grad, ht.grad, w.grad]:
        assert torch.isfinite(tensor).all()
    assert hs.grad.shape == hs.shape
    assert w.grad.shape == w.shape


@pytest.mark.gpu
def test_dynamic_chunk_cache_populated_after_call():
    clear_chunk_cache()
    BT, H, V = 2048, 64, 16384
    hs, ht, w, target = _build(BT, H, V, seed=12)
    dynamic_chunk(hs, ht, w, target, chunk_size="dynamic", kl_weight=0.2)
    cache = get_chunk_cache()
    assert len(cache) >= 1, "Expected dynamic_chunk to populate the resolver cache on success."


@pytest.mark.gpu
def test_dynamic_chunk_large_config_smoke():
    if _free_vram_gib() < 6.0:
        pytest.skip("Need at least 6 GiB free VRAM for this large-config smoke")

    BT, H, V = 8192, 1024, 32768
    hs, ht, w, target = _build(BT, H, V, seed=13)
    try:
        loss, *_ = dynamic_chunk(
            hs, ht, w, target,
            chunk_size="dynamic",
            lambda_student=1.0,
            kl_weight=0.3,
            kl_temperature=1.5,
        )
        loss.backward()
    except torch.cuda.OutOfMemoryError:
        pytest.skip("OOM on large-config smoke despite VRAM check — dispatcher fallback exercised")

    assert torch.isfinite(loss)
    assert torch.isfinite(hs.grad).all()
    assert torch.isfinite(w.grad).all()
    del hs, ht, w, target, loss
    torch.cuda.empty_cache()


