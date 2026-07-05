from __future__ import annotations

import torch

from orda_ce_kernel.utils import _autotune


def test_valid_candidates_never_exceed_effective_block_size():
    candidates = _autotune.valid_candidates(vocab_size=8192, max_fused_size=8192)
    assert candidates
    assert all(config.block_size <= 8192 for config in candidates)


def test_default_config_uses_t4_profile_when_device_is_t4(monkeypatch):
    monkeypatch.setattr(_autotune, "_is_t4_device", lambda: True)
    assert _autotune.default_config(_autotune.PRECOMPUTED, 32768, 32768) == _autotune.LaunchConfig(16384, 8)
    assert _autotune.default_config(_autotune.SEPARATE_STUDENT, 32768, 32768) == _autotune.LaunchConfig(16384, 8)
    assert _autotune.default_config(_autotune.TIED, 32768, 32768) == _autotune.LaunchConfig(8192, 16)


def test_default_config_uses_large_profile_for_non_t4(monkeypatch):
    monkeypatch.setattr(_autotune, "_is_t4_device", lambda: False)
    assert _autotune.default_config(_autotune.PRECOMPUTED, 32768, 32768) == _autotune.LaunchConfig(8192, 8)
    assert _autotune.default_config(_autotune.SEPARATE_STUDENT, 32768, 32768) == _autotune.LaunchConfig(8192, 8)
    assert _autotune.default_config(_autotune.SEPARATE_FULL, 32768, 32768) == _autotune.LaunchConfig(16384, 32)
    assert _autotune.default_config(_autotune.TIED, 32768, 32768) == _autotune.LaunchConfig(16384, 32)


def test_default_config_is_clamped_to_vocab_or_max_fused_size(monkeypatch):
    monkeypatch.setattr(_autotune, "_is_t4_device", lambda: False)
    config = _autotune.default_config(_autotune.TIED, vocab_size=4096, max_fused_size=8192)
    assert config.block_size <= 4096
    assert config.num_warps == 32


def test_is_oom_error_accepts_cuda_oom_only():
    assert _autotune._is_oom_error(torch.cuda.OutOfMemoryError("boom"))
    assert _autotune._is_oom_error(RuntimeError("CUDA out of memory"))
    assert not _autotune._is_oom_error(RuntimeError("CPU out of memory"))
    assert not _autotune._is_oom_error(RuntimeError("shape mismatch"))


def test_select_config_returns_default_without_autotune(monkeypatch):
    monkeypatch.setattr(_autotune, "_is_t4_device", lambda: False)
    selected = _autotune.select_config(
        mode=_autotune.PRECOMPUTED,
        device=torch.device("cpu"),
        dtype=torch.float16,
        vocab_size=32768,
        n_rows=1024,
        max_fused_size=32768,
        shape_key=(1024,),
        autotune=False,
        bench_fn=lambda *_args: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    assert selected == _autotune.LaunchConfig(8192, 8)


def test_cache_key_separates_specialized_workloads():
    base = _autotune._cache_key(
        mode=_autotune.TIED,
        device=torch.device("cpu"),
        dtype=torch.float16,
        vocab_size=32768,
        n_rows=1024,
        max_fused_size=32768,
        shape_key=(1024,),
        compute_kl=True,
        t_is_one=False,
        compute_teacher_ce=True,
    )
    kl_off = _autotune._cache_key(
        mode=_autotune.TIED,
        device=torch.device("cpu"),
        dtype=torch.float16,
        vocab_size=32768,
        n_rows=1024,
        max_fused_size=32768,
        shape_key=(1024,),
        compute_kl=False,
        t_is_one=False,
        compute_teacher_ce=True,
    )
    t_one = _autotune._cache_key(
        mode=_autotune.TIED,
        device=torch.device("cpu"),
        dtype=torch.float16,
        vocab_size=32768,
        n_rows=1024,
        max_fused_size=32768,
        shape_key=(1024,),
        compute_kl=True,
        t_is_one=True,
        compute_teacher_ce=True,
    )
    teacher_ce_off = _autotune._cache_key(
        mode=_autotune.TIED,
        device=torch.device("cpu"),
        dtype=torch.float16,
        vocab_size=32768,
        n_rows=1024,
        max_fused_size=32768,
        shape_key=(1024,),
        compute_kl=True,
        t_is_one=False,
        compute_teacher_ce=False,
    )

    assert base != kl_off
    assert base != t_one
    assert base != teacher_ce_off
