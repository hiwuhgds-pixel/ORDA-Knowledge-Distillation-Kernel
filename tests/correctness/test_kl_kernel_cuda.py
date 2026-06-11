from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.ops.kl_kernel import kl_from_logits_chunk


@pytest.mark.gpu
@pytest.mark.parametrize("vocab_size", [127, 1024, 5003])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("temperature", [0.7, 1.5])
def test_kl_kernel_matches_pytorch_reference(vocab_size: int, online: bool, temperature: float):
    set_seed(torch, 11)
    n_rows = 8
    dtype = torch.float16
    T = temperature
    kl_weight = 0.4
    logits_s = (torch.randn(n_rows, vocab_size, device="cuda", dtype=dtype) * 0.2)
    logits_t = (torch.randn(n_rows, vocab_size, device="cuda", dtype=dtype) * 0.2)
    logits_chunk = torch.cat([logits_s, logits_t], dim=0).contiguous()
    target = torch.randint(0, vocab_size, (n_rows,), device="cuda")
    target[::5] = -100
    n_non_ignore = int((target != -100).sum().item())

    kl, grad = kl_from_logits_chunk(
        logits_chunk,
        target,
        n_rows=n_rows,
        kl_weight=kl_weight,
        kl_temperature=T,
        n_non_ignore=n_non_ignore,
        ignore_index=-100,
        use_online_softmax=online,
    )

    logits_s_ref = (logits_s.float() / T)
    logits_t_ref = (logits_t.float() / T)
    log_p_s = torch.nn.functional.log_softmax(logits_s_ref, dim=-1)
    log_p_t = torch.nn.functional.log_softmax(logits_t_ref, dim=-1)
    p_s = log_p_s.exp()
    p_t = log_p_t.exp()
    mask = target != -100
    kl_rows = (p_t * (log_p_t - log_p_s)).sum(dim=-1).masked_fill(~mask, 0.0)
    kl_ref = kl_rows.sum() * (T * T) / max(n_non_ignore, 1)
    grad_ref = (p_s - p_t) * (kl_weight * T / max(n_non_ignore, 1))
    grad_ref = grad_ref.masked_fill(~mask[:, None], 0.0)

    assert torch.allclose(kl.float(), kl_ref.float(), atol=1e-3, rtol=1e-3)
    assert torch.allclose(grad.float(), grad_ref.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("fast_exp", "fast_log", "fast_mul", "online"),
    [
        (False, False, False, False),
        (False, False, True, True),
        (True, True, True, True),
    ],
)
def test_kl_kernel_flag_variants_are_finite(fast_exp: bool, fast_log: bool, fast_mul: bool, online: bool):
    set_seed(torch, 17)
    n_rows, vocab_size = 6, 257
    logits_chunk = (torch.randn(2 * n_rows, vocab_size, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    target = torch.randint(0, vocab_size, (n_rows,), device="cuda")
    target[-1] = -100

    kl, grad = kl_from_logits_chunk(
        logits_chunk,
        target,
        n_rows=n_rows,
        kl_weight=0.2,
        kl_temperature=2.0,
        n_non_ignore=int((target != -100).sum().item()),
        use_fast_math_exp=fast_exp,
        use_fast_math_log=fast_log,
        use_fast_math_mul=fast_mul,
        use_online_softmax=online,
    )

    assert torch.isfinite(kl)
    assert torch.isfinite(grad).all()
    assert torch.all(grad[target == -100] == 0)


@pytest.mark.gpu
def test_kl_kernel_all_ignored_targets_returns_zero_loss_and_grad():
    n_rows, vocab_size = 4, 131
    logits_chunk = torch.randn(2 * n_rows, vocab_size, device="cuda", dtype=torch.float16)
    target = torch.full((n_rows,), -100, device="cuda", dtype=torch.long)

    kl, grad = kl_from_logits_chunk(
        logits_chunk,
        target,
        n_rows=n_rows,
        kl_weight=0.4,
        kl_temperature=1.5,
        n_non_ignore=0,
        ignore_index=-100,
    )

    assert torch.equal(kl, torch.zeros_like(kl))
    assert torch.equal(grad, torch.zeros_like(grad))


