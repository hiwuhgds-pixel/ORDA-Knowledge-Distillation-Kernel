from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed
from tests.utils.reference import cosine_sim, reference_distill_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.ops.cross_entropy import distill_cross_entropy
from orda_ce_kernel.utils.dispatcher import dynamic_chunk


def _skip_if_dtype_unsupported(dtype):
    if dtype is torch.bfloat16 and hasattr(torch.cuda, "is_bf16_supported"):
        if not torch.cuda.is_bf16_supported():
            pytest.skip("bf16 is not supported on this CUDA device")


@pytest.mark.gpu
@pytest.mark.parametrize("vocab_size", [127, 503, 1024, 5003])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_cuda_kernel_matches_fp64_reference(vocab_size: int, label_smoothing: float):
    set_seed(torch, 42)
    BT, H = 16, 64
    hs = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    weight = (torch.randn(vocab_size, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    target = torch.randint(0, vocab_size, (BT,), device="cuda")
    target[::7] = -100

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs,
        ht,
        weight,
        target,
        lambda_student=1.3,
        label_smoothing=label_smoothing,
        kl_weight=0.4,
        kl_temperature=1.5,
    )
    loss.backward()

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, weight_ref, _ = reference_distill_loss(
        torch,
        hs.cpu(),
        ht.cpu(),
        weight.cpu(),
        target.cpu(),
        lambda_student=1.3,
        label_smoothing=label_smoothing,
        kl_weight=0.4,
        kl_temperature=1.5,
    )
    ref_loss.backward()

    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(loss_t.cpu(), ref_t.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=2e-3, rtol=2e-3)
    assert cosine_sim(torch, hs.grad.cpu(), hs_ref.grad) > 0.995
    assert cosine_sim(torch, ht.grad.cpu(), ht_ref.grad) > 0.995
    assert cosine_sim(torch, weight.grad.cpu(), weight_ref.grad) > 0.995


@pytest.mark.gpu
def test_ignore_index_zeroes_hidden_gradients():
    set_seed(torch, 123)
    BT, H, V = 10, 32, 128
    hs = torch.randn(BT, H, device="cuda", dtype=torch.float16, requires_grad=True)
    ht = torch.randn(BT, H, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.randint(0, V, (BT,), device="cuda")
    ignored = torch.tensor([1, 4, 9], device="cuda")
    target[ignored] = -100

    loss, *_ = distill_cross_entropy(hs, ht, weight, target, kl_weight=0.5, kl_temperature=1.2)
    loss.backward()

    assert torch.all(hs.grad[ignored] == 0)
    assert torch.all(ht.grad[ignored] == 0)


@pytest.mark.gpu
def test_cuda_kernel_rejects_out_of_range_targets():
    hs = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    ht = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(32, 16, device="cuda", dtype=torch.float16)
    target = torch.tensor([0, 1, 32, -100], device="cuda")

    with pytest.raises(ValueError, match="target"):
        distill_cross_entropy(hs, ht, weight, target)


@pytest.mark.gpu
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_ce_only_reduction_matches_fp64_reference(reduction: str):
    set_seed(torch, 314)
    BT, H, V = 12, 32, 257
    hs = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    weight = (torch.randn(V, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    target = torch.randint(0, V, (BT,), device="cuda")
    target[::4] = -100

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs,
        ht,
        weight,
        target,
        lambda_student=0.7,
        reduction=reduction,
        kl_weight=0.0,
        use_int8_quant=False,
    )
    loss.backward()

    ref_loss, ref_s, ref_t, ref_kl, hs_ref, ht_ref, weight_ref, _ = reference_distill_loss(
        torch,
        hs.cpu(),
        ht.cpu(),
        weight.cpu(),
        target.cpu(),
        lambda_student=0.7,
        reduction=reduction,
        kl_weight=0.0,
    )
    ref_loss.backward()

    assert torch.allclose(loss.cpu(), ref_loss.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(loss_s.cpu(), ref_s.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(loss_t.cpu(), ref_t.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(kl.cpu(), ref_kl.float(), atol=2e-6, rtol=0)
    assert cosine_sim(torch, hs.grad.cpu(), hs_ref.grad) > 0.995
    assert cosine_sim(torch, ht.grad.cpu(), ht_ref.grad) > 0.995
    assert cosine_sim(torch, weight.grad.cpu(), weight_ref.grad) > 0.995


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_kernel_dtype_and_chunk_flags_are_finite(dtype):
    _skip_if_dtype_unsupported(dtype)
    set_seed(torch, 2718)
    BT, H, V = 9, 24, 509
    hs = (torch.randn(BT, H, device="cuda", dtype=dtype) * 0.05).requires_grad_(True)
    ht = (torch.randn(BT, H, device="cuda", dtype=dtype) * 0.05).requires_grad_(True)
    weight = (torch.randn(V, H, device="cuda", dtype=dtype) * 0.05).requires_grad_(True)
    target = torch.randint(0, V, (BT,), device="cuda")

    loss, loss_s, loss_t, kl = distill_cross_entropy(
        hs,
        ht,
        weight,
        target,
        chunk_size=4,
        lambda_student=1.1,
        label_smoothing=0.05,
        kl_weight=0.25,
        kl_temperature=2.0,
        use_fast_math_exp=False,
        use_fast_math_log=False,
        use_fast_math_mul=True,
        use_online_softmax=True,
        use_fp32_accum=True,
    )
    loss.backward()

    for tensor in [loss, loss_s, loss_t, kl, hs.grad, ht.grad, weight.grad]:
        assert torch.isfinite(tensor).all()


@pytest.mark.gpu
def test_dynamic_chunk_matches_direct_kernel_for_same_inputs():
    set_seed(torch, 99)
    BT, H, V = 20, 32, 1009
    hs = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    weight = (torch.randn(V, H, device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
    target = torch.randint(0, V, (BT,), device="cuda")

    direct = distill_cross_entropy(hs, ht, weight, target, chunk_size=BT, kl_weight=0.3)
    dynamic = dynamic_chunk(hs, ht, weight, target, chunk_size="dynamic", kl_weight=0.3)

    for lhs, rhs in zip(direct, dynamic):
        assert torch.allclose(lhs, rhs, atol=2e-3, rtol=2e-3)


@pytest.mark.gpu
def test_cuda_kernel_is_reproducible_with_same_inputs():
    set_seed(torch, 12345)
    BT, H, V = 10, 16, 193
    hs = torch.randn(BT, H, device="cuda", dtype=torch.float16)
    ht = torch.randn(BT, H, device="cuda", dtype=torch.float16)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float16)
    target = torch.randint(0, V, (BT,), device="cuda")

    out1 = distill_cross_entropy(hs, ht, weight, target, kl_weight=0.2, use_int8_quant=False)
    out2 = distill_cross_entropy(hs, ht, weight, target, kl_weight=0.2, use_int8_quant=False)

    for lhs, rhs in zip(out1, out2):
        assert torch.equal(lhs, rhs)


