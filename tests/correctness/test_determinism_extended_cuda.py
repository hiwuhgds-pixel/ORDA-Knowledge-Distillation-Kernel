from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel, set_seed

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.utils.dispatcher import dynamic_chunk


def _inputs(seed):
    set_seed(torch, seed)
    BT, H, V = 12, 32, 257
    hs = torch.randn(BT, H, device="cuda", dtype=torch.float16)
    ht = torch.randn(BT, H, device="cuda", dtype=torch.float16)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float16)
    target = torch.randint(0, V, (BT,), device="cuda")
    return hs, ht, weight, target


def _run_twice(hs, ht, weight, target, **kwargs):
    out1 = dynamic_chunk(
        hs, ht, weight, target,
        kl_weight=0.3,
        kl_temperature=1.4,
        **kwargs,
    )
    out2 = dynamic_chunk(
        hs, ht, weight, target,
        kl_weight=0.3,
        kl_temperature=1.4,
        **kwargs,
    )
    for lhs, rhs in zip(out1, out2):
        assert torch.equal(lhs, rhs)


@pytest.mark.gpu
def test_determinism_with_fast_math_enabled():
    hs, ht, w, target = _inputs(seed=101)
    _run_twice(
        hs, ht, w, target,
        use_fast_math_exp=True,
        use_fast_math_log=True,
        use_fast_math_mul=True,
    )


@pytest.mark.gpu
def test_determinism_with_online_softmax_enabled():
    hs, ht, w, target = _inputs(seed=102)
    _run_twice(hs, ht, w, target, use_online_softmax=True)


@pytest.mark.gpu
def test_determinism_with_all_flags_enabled():
    hs, ht, w, target = _inputs(seed=103)
    _run_twice(
        hs, ht, w, target,
        use_fast_math_exp=True,
        use_fast_math_log=True,
        use_fast_math_mul=True,
        use_online_softmax=True,
        use_kl_in_kernel=True,
    )


@pytest.mark.gpu
def test_determinism_with_kl_in_kernel_disabled():
    hs, ht, w, target = _inputs(seed=104)
    _run_twice(hs, ht, w, target, use_kl_in_kernel=False)


