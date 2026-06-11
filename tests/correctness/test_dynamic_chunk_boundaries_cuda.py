from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed
from tests.utils.reference import cosine_sim, reference_distill_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.utils.dispatcher import dynamic_chunk
from orda_ce_kernel.ops.cross_entropy import distill_cross_entropy
from orda_ce_kernel.utils.resolver import resolve_chunk_size


def _make(BT, H, V, seed):
    set_seed(torch, seed)
    hs = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    w = (torch.randn(V, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    target = torch.randint(0, V, (BT,), device="cuda")
    target[::5] = -100  # ensure ignored rows including potential tail
    return hs, ht, w, target


def _check_against_reference(hs, ht, w, target, **kw):
    loss, loss_s, loss_t, kl = dynamic_chunk(hs, ht, w, target, **kw)
    loss.backward()

    reference_keys = {
        "lambda_student",
        "ignore_index",
        "reduction",
        "label_smoothing",
        "kl_weight",
        "kl_temperature",
        "teacher_mode",
        "weight_teacher",
        "logits_teacher",
        "teacher_loss_weight",
    }
    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, w_ref, _ = reference_distill_loss(
        torch, hs.cpu(), ht.cpu(), w.cpu(), target.cpu(),
        **{k: v for k, v in kw.items() if k in reference_keys},
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(loss_t.cpu(), ref_t.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=3e-3, rtol=3e-3)
    assert cosine_sim(torch, hs.grad.cpu(), hs_ref.grad) > 0.995
    assert cosine_sim(torch, ht.grad.cpu(), ht_ref.grad) > 0.995
    # weight grad cosine loosened to 0.99 — fp16 grad_W accumulation order
    # depends on chunk count, so multi-chunk configs at large V noise more.
    assert cosine_sim(torch, w.grad.cpu(), w_ref.grad) > 0.99


@pytest.mark.gpu
@pytest.mark.parametrize("vocab_size", [127, 1024, 5003])
@pytest.mark.parametrize("offset", [-1, 0, 1, "double-minus-one"])
def test_dynamic_chunk_boundary_around_resolved_size(vocab_size: int, offset):
    H = 64
    # Probe resolver to find a representative dynamic chunk size for this V.
    # Use a moderately large BT so resolver returns >1 chunk; then derive
    # boundary BT values around the resolved chunk size.
    probe_BT = 4096
    cs_dyn, _ = resolve_chunk_size(probe_BT, "dynamic", V=vocab_size)
    if offset == "double-minus-one":
        BT = max(2, 2 * cs_dyn - 1)
    else:
        BT = max(1, cs_dyn + offset)

    hs, ht, w, target = _make(BT, H, vocab_size, seed=BT + vocab_size)
    _check_against_reference(
        hs, ht, w, target,
        chunk_size="dynamic",
        lambda_student=1.0,
        kl_weight=0.3,
        kl_temperature=1.5,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_explicit_chunk_size_matches_reference(chunk_size: int):
    BT, H, V = 96, 32, 1009
    hs, ht, w, target = _make(BT, H, V, seed=chunk_size + 1)
    _check_against_reference(
        hs, ht, w, target,
        chunk_size=chunk_size,
        lambda_student=1.0,
        kl_weight=0.25,
        kl_temperature=1.2,
    )


@pytest.mark.gpu
def test_explicit_chunk_size_equal_to_BT_matches_reference():
    BT, H, V = 48, 32, 503
    hs, ht, w, target = _make(BT, H, V, seed=48)
    _check_against_reference(
        hs, ht, w, target,
        chunk_size=BT,
        lambda_student=1.0,
        kl_weight=0.25,
        kl_temperature=1.2,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("hidden_dim", [63, 128, 257])
def test_non_power_of_two_hidden_dim(hidden_dim: int):
    BT, V = 24, 1024
    hs, ht, w, target = _make(BT, hidden_dim, V, seed=hidden_dim)
    _check_against_reference(
        hs, ht, w, target,
        chunk_size="dynamic",
        lambda_student=1.0,
        kl_weight=0.3,
        kl_temperature=1.5,
    )


@pytest.mark.gpu
def test_small_max_fused_size_still_matches_reference():
    BT, H, V = 24, 32, 257
    hs, ht, w, target = _make(BT, H, V, seed=257)
    _check_against_reference(
        hs,
        ht,
        w,
        target,
        chunk_size=7,
        max_fused_size=64,
        lambda_student=1.0,
        kl_weight=0.3,
        kl_temperature=1.5,
    )


