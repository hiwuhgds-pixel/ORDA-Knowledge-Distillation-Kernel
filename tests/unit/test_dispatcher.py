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
    assert not dispatcher._is_oom_error(RuntimeError("shape mismatch"))


def test_chunk_cache_key_separates_teacher_mode_and_student_only():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    h_teacher = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)

    tied = dispatcher._cache_key(8, 4, 2, h_student, weight, "tied", False, h_teacher=h_teacher)
    separate = dispatcher._cache_key(8, 4, 2, h_student, weight, "separate", False, h_teacher=h_teacher)
    student_only = dispatcher._cache_key(8, 4, 2, h_student, weight, "separate", True, h_teacher=h_teacher)

    assert tied != separate
    assert separate != student_only


def test_chunk_cache_key_separates_runtime_contract_fields():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    h_teacher = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)
    logits_teacher = torch.randn(8, 4, dtype=torch.float32)

    base = dispatcher._cache_key(
        8, 4, 2, h_student, weight, "precomputed", True,
        max_fused_size=64, max_chunks=8, logits_teacher=logits_teacher,
    )
    different_fused = dispatcher._cache_key(
        8, 4, 2, h_student, weight, "precomputed", True,
        max_fused_size=128, max_chunks=8, logits_teacher=logits_teacher,
    )
    different_teacher_dtype = dispatcher._cache_key(
        8, 4, 2, h_student, weight, "precomputed", True,
        max_fused_size=64, max_chunks=8, logits_teacher=logits_teacher.to(torch.float16),
    )
    different_teacher_hidden = dispatcher._cache_key(
        8, 4, 2, h_student, weight, "tied", False,
        max_fused_size=64, max_chunks=8, h_teacher=h_teacher,
    )

    assert base != different_fused
    assert base != different_teacher_dtype
    assert base != different_teacher_hidden


def test_dynamic_chunk_retries_after_oom_and_caches_success(monkeypatch):
    dispatcher.clear_chunk_cache()
    calls: list[int] = []

    def fake_distill_cross_entropy(*args, chunk_size=None, **kwargs):
        calls.append(chunk_size)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic out of memory")
        one = torch.tensor(1.0)
        return one, one, one, one

    import orda_ce_kernel.ops.cross_entropy as cross_entropy

    monkeypatch.setattr(cross_entropy, "distill_cross_entropy", fake_distill_cross_entropy)
    h_student = torch.randn(4, 2)
    h_teacher = torch.randn(4, 2)
    weight = torch.randn(4, 2)
    target = torch.randint(0, 4, (4,))

    result = dispatcher.dynamic_chunk(
        h_student, h_teacher, weight, target,
        chunk_size="dynamic", max_chunks=8,
    )
    assert len(result) == 4
    assert calls == [4, 2]
    assert list(dispatcher.get_chunk_cache().values()) == [2]
    dispatcher.clear_chunk_cache()


