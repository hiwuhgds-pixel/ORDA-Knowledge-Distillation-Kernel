from __future__ import annotations

import torch

from orda_ce_kernel.utils import dispatcher


def test_chunk_size_from_num_chunks_rounds_up():
    assert dispatcher._chunk_size_from_num_chunks(10, 3) == 4
    assert dispatcher._chunk_size_from_num_chunks(10, 5) == 2


def test_chunk_cache_is_copy_and_can_be_cleared():
    dispatcher.clear_chunk_cache()
    cache = dispatcher.get_chunk_cache()
    cache[("fake",)] = 1
    assert dispatcher.get_chunk_cache() == {}


def test_is_oom_error_accepts_cuda_exception_and_message():
    assert dispatcher._is_oom_error(torch.cuda.OutOfMemoryError("boom"))
    assert dispatcher._is_oom_error(RuntimeError("CUDA out of memory"))
    assert not dispatcher._is_oom_error(RuntimeError("CPU out of memory"))
    assert not dispatcher._is_oom_error(RuntimeError("shape mismatch"))


def test_chunk_cache_key_separates_teacher_mode_and_teacher_tensors():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    h_teacher = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)
    teacher_weight = torch.randn(4, 2, dtype=torch.float16)
    logits_teacher = torch.randn(8, 4, dtype=torch.float16)

    tied = dispatcher._cache_key(8, 4, 2, h_student, weight, "tied", teacher_hidden=h_teacher)
    separate = dispatcher._cache_key(
        8,
        4,
        2,
        h_student,
        weight,
        "separate",
        teacher_hidden=h_teacher,
        teacher_weight=teacher_weight,
    )
    precomputed_logits = dispatcher._cache_key(
        8,
        4,
        2,
        h_student,
        weight,
        "precomputed",
        logits_teacher=logits_teacher,
    )

    assert tied != separate
    assert separate != precomputed_logits
    assert tied != precomputed_logits


def test_chunk_cache_key_separates_runtime_contract_fields():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)
    logits_teacher = torch.randn(8, 4, dtype=torch.float32)

    base = dispatcher._cache_key(
        8,
        4,
        2,
        h_student,
        weight,
        "precomputed",
        max_fused_size=64,
        max_chunks=8,
        logits_teacher=logits_teacher,
    )
    different_fused = dispatcher._cache_key(
        8,
        4,
        2,
        h_student,
        weight,
        "precomputed",
        max_fused_size=128,
        max_chunks=8,
        logits_teacher=logits_teacher,
    )
    different_teacher_dtype = dispatcher._cache_key(
        8,
        4,
        2,
        h_student,
        weight,
        "precomputed",
        max_fused_size=64,
        max_chunks=8,
        logits_teacher=logits_teacher.to(torch.float16),
    )

    assert base != different_fused
    assert base != different_teacher_dtype


def test_chunk_cache_key_separates_kl_enabled_state():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)
    h_teacher = torch.randn(8, 2, dtype=torch.float16)
    teacher_weight = torch.randn(4, 2, dtype=torch.float16)

    kl_on = dispatcher._cache_key(
        8,
        4,
        2,
        h_student,
        weight,
        "separate",
        teacher_hidden=h_teacher,
        teacher_weight=teacher_weight,
        compute_kl=True,
    )
    kl_off = dispatcher._cache_key(
        8,
        4,
        2,
        h_student,
        weight,
        "separate",
        teacher_hidden=h_teacher,
        teacher_weight=teacher_weight,
        compute_kl=False,
    )

    assert kl_on != kl_off


def test_dynamic_chunk_retries_after_oom_and_caches_success(monkeypatch):
    dispatcher.clear_chunk_cache()
    calls: list[int] = []

    def fake_call_kernel_fn(*args, chunk_size=None, **kwargs):
        calls.append(chunk_size)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic out of memory")
        one = torch.tensor(1.0)
        return one, one, one, one

    monkeypatch.setattr(dispatcher, "_resolve_kernel_fn", lambda _mode: object())
    monkeypatch.setattr(dispatcher, "_call_kernel_fn", fake_call_kernel_fn)

    h_student = torch.randn(4, 2)
    h_teacher = torch.randn(4, 2)
    weight = torch.randn(4, 2)
    target = torch.randint(0, 4, (4,))

    result = dispatcher.dynamic_chunk(
        h_student,
        h_teacher,
        weight,
        target,
        chunk_size="dynamic",
    )
    assert len(result) == 4
    assert calls == [4, 2]
    assert list(dispatcher.get_chunk_cache().values()) == [2]
    dispatcher.clear_chunk_cache()


def test_dynamic_chunk_num_chunks_uses_fixed_path(monkeypatch):
    calls: list[int] = []

    def fake_call_kernel_fn(*args, chunk_size=None, **kwargs):
        calls.append(chunk_size)
        one = torch.tensor(1.0)
        return one, one, one, one

    monkeypatch.setattr(dispatcher, "_resolve_kernel_fn", lambda _mode: object())
    monkeypatch.setattr(dispatcher, "_call_kernel_fn", fake_call_kernel_fn)

    h_student = torch.randn(10, 2)
    h_teacher = torch.randn(10, 2)
    weight = torch.randn(4, 2)
    target = torch.randint(0, 4, (10,))

    result = dispatcher.dynamic_chunk(
        h_student,
        h_teacher,
        weight,
        target,
        num_chunks=3,
    )
    assert len(result) == 4
    assert calls == [4]
