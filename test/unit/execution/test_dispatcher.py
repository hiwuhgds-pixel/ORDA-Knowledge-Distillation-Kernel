from __future__ import annotations

import inspect

import pytest
import torch

from orda_kernel import ChunkSize
from orda_kernel._execution import dispatcher
from orda_kernel._planning.policy import build_execution_plan


# ── Plan factory ─────────────────────────────────────────────────────────────
def _plan(
    *,
    teacher_mode: str = "tied",
    bt: int,
    vocab_size: int = 4,
    chunk_size: ChunkSize,
    num_chunks: int | None = None,
    max_chunks: int | None = None,
    teacher_ce_weight: float = 1.0,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    kl_variant: str = "forward",
    autotune=False,
    grad_weight_accumulation: str = "auto",
    compute_dtype: torch.dtype = torch.float16,
):
    """Build a Triton ExecutionPlan the same way the public API does."""
    return build_execution_plan(
        backend="triton",
        teacher_mode=teacher_mode,
        compute_dtype=compute_dtype,
        bt=bt,
        vocab_size=vocab_size,
        student_ce_weight=1.0,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
        kl_temperature=kl_temperature,
        kl_variant=kl_variant,
        ignore_index=-100,
        reduction="mean",
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        max_chunks=max_chunks,
        max_fused_size=16384,
        autotune=autotune,
        grad_weight_accumulation=grad_weight_accumulation,
    )


# ── Chunk arithmetic ─────────────────────────────────────────────────────────
def test_chunk_size_from_num_chunks_rounds_up():
    assert dispatcher._chunk_size_from_num_chunks(10, 3) == 4
    assert dispatcher._chunk_size_from_num_chunks(10, 5) == 2


def test_chunk_attempt_count_matches_nonempty_slices():
    attempt = dispatcher._chunk_attempt_from_num_chunks(5, 4)

    assert attempt.chunk_size == 2
    assert attempt.num_chunks == 3


def test_chunk_cache_is_copy_and_can_be_cleared():
    dispatcher.clear_chunk_cache()
    cache = dispatcher.get_chunk_cache()
    cache[("fake",)] = 1
    assert dispatcher.get_chunk_cache() == {}


@pytest.mark.parametrize(
    ("baseline", "current", "successful", "expected"),
    [(None, None, 2, 2), (2, 2, 4, 4), (2, 8, 4, 8), (None, 8, 4, 8)],
)
def test_merge_chunk_success_preserves_concurrent_increases(
    baseline,
    current,
    successful,
    expected,
):
    assert dispatcher._merge_chunk_success(baseline, current, successful) == expected


def test_is_oom_error_accepts_cuda_exception_and_message():
    assert dispatcher._is_oom_error(torch.cuda.OutOfMemoryError("boom"))
    for message in (
        "CUDA out of memory",
        "CUDA error: out of memory",
        "HIP out of memory",
        "HIP error: out of memory",
    ):
        assert dispatcher._is_oom_error(RuntimeError(message))
    assert not dispatcher._is_oom_error(RuntimeError("CPU out of memory"))
    assert not dispatcher._is_oom_error(RuntimeError("shape mismatch"))


# ── Cache-key identity (plan-derived) ────────────────────────────────────────
def test_chunk_cache_key_separates_teacher_mode_and_teacher_tensors():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    h_teacher = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)
    teacher_weight = torch.randn(4, 2, dtype=torch.float16)
    logits_teacher = torch.randn(8, 4, dtype=torch.float16)

    tied = dispatcher._cache_key_for_plan(
        8,
        4,
        2,
        h_student,
        weight,
        _plan(teacher_mode="tied", bt=8, chunk_size=ChunkSize.dynamic()),
        teacher_hidden=h_teacher,
    )
    separate = dispatcher._cache_key_for_plan(
        8,
        4,
        2,
        h_student,
        weight,
        _plan(
            teacher_mode="separate",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            teacher_ce_weight=1.0,
        ),
        teacher_hidden=h_teacher,
        teacher_weight=teacher_weight,
    )
    precomputed_logits = dispatcher._cache_key_for_plan(
        8,
        4,
        2,
        h_student,
        weight,
        _plan(
            teacher_mode="precomputed",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            teacher_ce_weight=0.0,
        ),
        logits_teacher=logits_teacher,
    )

    assert tied != separate
    assert separate != precomputed_logits
    assert tied != precomputed_logits


def test_chunk_cache_key_separates_runtime_contract_fields():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)
    logits_teacher = torch.randn(8, 4, dtype=torch.float32)

    base = dispatcher._cache_key_for_plan(
        8,
        4,
        2,
        h_student,
        weight,
        _plan(
            teacher_mode="precomputed",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            max_chunks=8,
            teacher_ce_weight=0.0,
            kl_weight=1.0,
        ),
        logits_teacher=logits_teacher,
    )
    different_teacher_dtype = dispatcher._cache_key_for_plan(
        8,
        4,
        2,
        h_student,
        weight,
        _plan(
            teacher_mode="precomputed",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            max_chunks=8,
            teacher_ce_weight=0.0,
            kl_weight=1.0,
        ),
        logits_teacher=logits_teacher.to(torch.float16),
    )

    assert base != different_teacher_dtype


def test_chunk_cache_key_canonicalizes_inactive_teacher_tensors():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)
    logits_fp16 = torch.randn(8, 4, dtype=torch.float16)
    logits_fp32 = logits_fp16.float()

    plan = _plan(
        teacher_mode="precomputed",
        bt=8,
        chunk_size=ChunkSize.dynamic(),
        teacher_ce_weight=0.0,
        kl_weight=0.0,
    )
    fp16_key = dispatcher._cache_key_for_plan(
        8, 4, 2, h_student, weight, plan, logits_teacher=logits_fp16
    )
    fp32_key = dispatcher._cache_key_for_plan(
        8, 4, 2, h_student, weight, plan, logits_teacher=logits_fp32
    )

    assert fp16_key == fp32_key


def test_chunk_cache_key_separates_effective_compute_dtype():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)

    native = dispatcher._cache_key_for_plan(
        8,
        4,
        2,
        h_student,
        weight,
        _plan(bt=8, chunk_size=ChunkSize.dynamic(), compute_dtype=torch.float16),
    )
    autocast = dispatcher._cache_key_for_plan(
        8,
        4,
        2,
        h_student,
        weight,
        _plan(bt=8, chunk_size=ChunkSize.dynamic(), compute_dtype=torch.float32),
    )

    assert native.bt == autocast.bt == 8
    assert native != autocast


def test_chunk_cache_key_separates_kl_enabled_state():
    h_student = torch.randn(8, 2, dtype=torch.float16)
    weight = torch.randn(4, 2, dtype=torch.float16)
    h_teacher = torch.randn(8, 2, dtype=torch.float16)
    teacher_weight = torch.randn(4, 2, dtype=torch.float16)

    kl_on = dispatcher._cache_key_for_plan(
        8,
        4,
        2,
        h_student,
        weight,
        _plan(
            teacher_mode="separate",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            teacher_ce_weight=1.0,
            kl_weight=1.0,
        ),
        teacher_hidden=h_teacher,
        teacher_weight=teacher_weight,
    )
    kl_off = dispatcher._cache_key_for_plan(
        8,
        4,
        2,
        h_student,
        weight,
        _plan(
            teacher_mode="separate",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            teacher_ce_weight=1.0,
            kl_weight=0.0,
        ),
        teacher_hidden=h_teacher,
        teacher_weight=teacher_weight,
    )

    assert kl_on != kl_off


def test_chunk_cache_key_separates_specialization_and_memory_dimensions():
    h_student = torch.randn(8, 2, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(4, 2, dtype=torch.float16, requires_grad=True)
    h_teacher = torch.randn(8, 3, dtype=torch.float16, requires_grad=True)
    teacher_weight = torch.randn(4, 3, dtype=torch.float16, requires_grad=True)

    tensors = dict(
        teacher_hidden=h_teacher,
        teacher_weight=teacher_weight,
    )

    def key(plan):
        return dispatcher._cache_key_for_plan(
            8, 4, 2, h_student, weight, plan, **tensors
        )

    base_plan = _plan(
        teacher_mode="separate",
        bt=8,
        chunk_size=ChunkSize.dynamic(),
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_temperature=2.0,
        kl_variant="forward",
    )
    base = key(base_plan)
    variant_plans = [
        _plan(
            teacher_mode="separate",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            teacher_ce_weight=0.0,
            kl_weight=1.0,
            kl_temperature=1.0,
            kl_variant="forward",
        ),
        _plan(
            teacher_mode="separate",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            teacher_ce_weight=0.0,
            kl_weight=1.0,
            kl_temperature=2.0,
            kl_variant="jeffreys",
        ),
        _plan(
            teacher_mode="separate",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            teacher_ce_weight=1.0,
            kl_weight=1.0,
            kl_temperature=2.0,
            kl_variant="forward",
        ),
        _plan(
            teacher_mode="separate",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            teacher_ce_weight=0.0,
            kl_weight=1.0,
            kl_temperature=2.0,
            kl_variant="mutual",
        ),
        _plan(
            teacher_mode="separate",
            bt=8,
            chunk_size=ChunkSize.dynamic(),
            teacher_ce_weight=0.0,
            kl_weight=1.0,
            kl_temperature=2.0,
            kl_variant="forward",
            grad_weight_accumulation="fp32",
        ),
    ]
    assert all(key(plan) != base for plan in variant_plans)
    # A detached student tensor changes the memory signature but not the plan.
    detached = dispatcher._cache_key_for_plan(
        8, 4, 2, h_student.detach(), weight, base_plan, **tensors
    )
    assert detached != base


# ── Chunk retry recovery ─────────────────────────────────────────────────────
def test_execute_plan_retries_after_oom_and_caches_success(monkeypatch):
    dispatcher.clear_chunk_cache()
    calls: list[int] = []

    def fake_call(kernel_fn, plan, attempt, *args, **kwargs):
        calls.append(attempt.chunk_size)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic out of memory")
        one = torch.tensor(1.0)
        return one, one, one, one

    monkeypatch.setattr(dispatcher, "_resolve_plan_executor", lambda _mode: object())
    monkeypatch.setattr(dispatcher, "_call_plan_executor", fake_call)

    h_student = torch.randn(4, 2)
    h_teacher = torch.randn(4, 2)
    weight = torch.randn(4, 2)
    labels = torch.randint(0, 4, (4,))

    plan = _plan(bt=4, chunk_size=ChunkSize.dynamic())
    for _ in range(2):
        result = dispatcher.execute_plan(
            plan,
            h_student,
            h_teacher,
            weight,
            labels,
            _labels_prevalidated=True,
        )
        assert len(result) == 4

    assert calls == [4, 2, 2]
    assert list(dispatcher.get_chunk_cache().values()) == [2]
    dispatcher.clear_chunk_cache()


def test_execute_plan_does_not_retry_non_chunkable_workspace_oom(monkeypatch):
    dispatcher.clear_chunk_cache()
    calls: list[int] = []

    def fake_call(kernel_fn, plan, attempt, *args, **kwargs):
        calls.append(attempt.num_chunks)
        raise dispatcher.NonChunkableWorkspaceOOM("fixed workspace")

    monkeypatch.setattr(dispatcher, "_resolve_plan_executor", lambda _mode: object())
    monkeypatch.setattr(dispatcher, "_call_plan_executor", fake_call)
    monkeypatch.setattr(dispatcher, "recover_after_oom", lambda _device: None)

    with pytest.raises(RuntimeError, match="non-chunkable workspace"):
        dispatcher.execute_plan(
            _plan(bt=4, chunk_size=ChunkSize.dynamic()),
            torch.randn(4, 2),
            torch.randn(4, 2),
            torch.randn(4, 2),
            torch.randint(0, 4, (4,)),
            _labels_prevalidated=True,
        )

    assert calls == [1]


def test_execute_plan_normalizes_stale_cache_to_nonempty_count(monkeypatch):
    dispatcher.clear_chunk_cache()
    cache_key = object()
    calls: list[int] = []

    def fake_call(kernel_fn, plan, attempt, *args, **kwargs):
        calls.append(attempt.chunk_size)
        one = torch.tensor(1.0)
        return one, one, one, one

    monkeypatch.setattr(dispatcher, "_resolve_plan_executor", lambda _mode: object())
    monkeypatch.setattr(
        dispatcher, "_cache_key_for_plan", lambda *_args, **_kwargs: cache_key
    )
    monkeypatch.setattr(dispatcher, "_call_plan_executor", fake_call)
    dispatcher._cached_num_chunks[cache_key] = 4

    dispatcher.execute_plan(
        _plan(bt=5, chunk_size=ChunkSize.dynamic(), max_chunks=4),
        torch.randn(5, 2),
        torch.randn(5, 2),
        torch.randn(4, 2),
        torch.randint(0, 4, (5,)),
        _labels_prevalidated=True,
    )

    assert calls == [2]
    assert list(dispatcher.get_chunk_cache().values()) == [3]
    dispatcher.clear_chunk_cache()


def test_execute_plan_does_not_regress_a_concurrent_safer_cache_entry(monkeypatch):
    dispatcher.clear_chunk_cache()
    cache_key = object()
    one = torch.tensor(1.0)

    monkeypatch.setattr(dispatcher, "_resolve_plan_executor", lambda _mode: object())
    monkeypatch.setattr(
        dispatcher, "_cache_key_for_plan", lambda *_args, **_kwargs: cache_key
    )

    def fake_call(_executor, _plan, attempt, *_args, **_kwargs):
        assert attempt.num_chunks == 1
        with dispatcher._CACHE_LOCK:
            dispatcher._cached_num_chunks[cache_key] = 4
        return one, one, one, one

    monkeypatch.setattr(dispatcher, "_call_plan_executor", fake_call)
    dispatcher.execute_plan(
        _plan(bt=8, chunk_size=ChunkSize.dynamic()),
        torch.randn(8, 2),
        torch.randn(8, 2),
        torch.randn(4, 2),
        torch.randint(0, 4, (8,)),
        _labels_prevalidated=True,
    )

    assert dispatcher.get_chunk_cache()[cache_key] == 4
    dispatcher.clear_chunk_cache()


def test_execute_plan_stops_when_max_chunks_cannot_add_slice(monkeypatch):
    dispatcher.clear_chunk_cache()
    calls: list[int] = []

    def fake_call(kernel_fn, plan, attempt, *args, **kwargs):
        calls.append(attempt.chunk_size)
        raise torch.cuda.OutOfMemoryError("synthetic out of memory")

    monkeypatch.setattr(dispatcher, "_resolve_plan_executor", lambda _mode: object())
    monkeypatch.setattr(dispatcher, "_call_plan_executor", fake_call)
    monkeypatch.setattr(dispatcher, "recover_after_oom", lambda _device: None)

    h_student = torch.randn(5, 2)
    h_teacher = torch.randn(5, 2)
    weight = torch.randn(4, 2)
    labels = torch.randint(0, 4, (5,))

    with pytest.raises(RuntimeError) as excinfo:
        dispatcher.execute_plan(
            _plan(bt=5, chunk_size=ChunkSize.dynamic(), max_chunks=4),
            h_student,
            h_teacher,
            weight,
            labels,
            _labels_prevalidated=True,
        )

    message = str(excinfo.value)
    # Rounding, not the ceiling, ends the loop at 3 chunks under max_chunks=4.
    assert "Used 3 chunks of 2 rows" in message
    assert "num_chunks=5" in message
    assert "max_chunks" not in message

    assert calls == [5, 3, 2]
    dispatcher.clear_chunk_cache()


def test_execute_plan_fixed_chunk_path_runs_once(monkeypatch):
    calls: list[int] = []

    def fake_call(kernel_fn, plan, attempt, *args, **kwargs):
        calls.append(attempt.chunk_size)
        one = torch.tensor(1.0)
        return one, one, one, one

    monkeypatch.setattr(dispatcher, "_resolve_plan_executor", lambda _mode: object())
    monkeypatch.setattr(dispatcher, "_call_plan_executor", fake_call)

    h_student = torch.randn(10, 2)
    h_teacher = torch.randn(10, 2)
    weight = torch.randn(4, 2)
    labels = torch.randint(0, 4, (10,))

    result = dispatcher.execute_plan(
        _plan(bt=10, chunk_size=ChunkSize.dynamic(), num_chunks=3),
        h_student,
        h_teacher,
        weight,
        labels,
        _labels_prevalidated=True,
    )
    assert len(result) == 4
    assert calls == [4]


def test_execute_plan_preserves_max_autotune_mode(monkeypatch):
    captured = []

    def fake_call(kernel_fn, plan, attempt, *args, **kwargs):
        captured.append(plan.autotune)
        one = torch.tensor(1.0)
        return one, one, one, one

    monkeypatch.setattr(dispatcher, "_resolve_plan_executor", lambda _mode: object())
    monkeypatch.setattr(dispatcher, "_call_plan_executor", fake_call)

    dispatcher.execute_plan(
        _plan(bt=4, chunk_size=ChunkSize.nrows(2), autotune="Max"),
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randint(0, 4, (4,)),
        _labels_prevalidated=True,
    )

    assert captured == ["Max"]


def test_execute_plan_requires_triton_backend():
    torch_plan = build_execution_plan(
        backend="torch",
        teacher_mode="tied",
        compute_dtype=torch.float16,
        bt=4,
        vocab_size=4,
        student_ce_weight=1.0,
        teacher_ce_weight=1.0,
        kl_weight=0.0,
        kl_temperature=1.0,
        kl_variant="forward",
        ignore_index=-100,
        reduction="mean",
        chunk_size=ChunkSize.dynamic(),
        num_chunks=None,
        max_chunks=None,
        max_fused_size=16384,
        autotune=False,
        grad_weight_accumulation="auto",
    )
    with pytest.raises(RuntimeError, match="Triton execution plan"):
        dispatcher.execute_plan(
            torch_plan,
            torch.randn(4, 2),
            torch.randn(4, 2),
            torch.randn(4, 2),
            torch.randint(0, 4, (4,)),
            _labels_prevalidated=True,
        )


def test_execute_plan_host_validates_labels_at_its_boundary(monkeypatch):
    monkeypatch.setattr(dispatcher, "_resolve_plan_executor", lambda _mode: object())
    monkeypatch.setattr(
        dispatcher,
        "_call_plan_executor",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not dispatch on invalid labels")
        ),
    )
    h_student = torch.randn(4, 2)
    h_teacher = torch.randn(4, 2)
    weight = torch.randn(4, 2)
    labels = torch.tensor([0, 1, -1, 3])

    with pytest.raises(ValueError, match="labels.*outside"):
        dispatcher.execute_plan(
            _plan(bt=4, chunk_size=ChunkSize.dynamic()),
            h_student,
            h_teacher,
            weight,
            labels,
        )


def test_execute_plan_prevalidated_labels_skip_host_check(monkeypatch):
    monkeypatch.setattr(
        dispatcher,
        "_validate_label_range",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("prevalidated labels must not be host-validated again")
        ),
    )
    monkeypatch.setattr(dispatcher, "_resolve_plan_executor", lambda _mode: object())
    one = torch.tensor(1.0)
    monkeypatch.setattr(
        dispatcher,
        "_call_plan_executor",
        lambda *a, **k: (one, one, one, one),
    )

    result = dispatcher.execute_plan(
        _plan(bt=4, chunk_size=ChunkSize.nrows(2)),
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.tensor([0, 1, 999, 3]),  # out of range, but prevalidated=True
        _labels_prevalidated=True,
    )
    assert result == (one, one, one, one)


def test_execute_plan_public_signature_has_prevalidation_flag():
    assert (
        "_labels_prevalidated" in inspect.signature(dispatcher.execute_plan).parameters
    )


# ── Fast chunk plans ─────────────────────────────────────────────────────────
def test_fast_plan_is_static_with_a_sixteen_chunk_ceiling():
    policy = _plan(bt=65536, chunk_size=ChunkSize.fast()).chunk_policy
    assert policy.fast is True
    assert policy.dynamic is False
    assert policy.initial_num_chunks == 16
    assert policy.initial_chunk_size == 4096
    assert policy.max_chunks_limit == 16


def test_explicit_num_chunks_clears_the_fast_flag():
    policy = _plan(bt=4096, chunk_size=ChunkSize.fast(), num_chunks=8).chunk_policy
    assert policy.fast is False
    assert policy.dynamic is False
    assert policy.initial_num_chunks == 8


def _raise_oom(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory")


def test_fast_plan_oom_below_the_ceiling_suggests_the_next_partition(monkeypatch):
    monkeypatch.setattr(dispatcher, "_call_plan_executor", _raise_oom)
    monkeypatch.setattr(dispatcher, "recover_after_oom", lambda device: None)

    with pytest.raises(RuntimeError) as excinfo:
        dispatcher.execute_plan(
            _plan(bt=4096, chunk_size=ChunkSize.fast()),
            torch.randn(4096, 2),
            torch.randn(4096, 2),
            torch.randn(4, 2),
            torch.zeros(4096, dtype=torch.long),
            _labels_prevalidated=True,
        )

    message = str(excinfo.value)
    assert "Used 4 chunks of 1024 rows" in message
    assert "num_chunks=8" in message
    assert "max_chunks" not in message


def test_fast_plan_oom_at_the_ceiling_reports_the_marginal_gain(monkeypatch):
    monkeypatch.setattr(dispatcher, "_call_plan_executor", _raise_oom)
    monkeypatch.setattr(dispatcher, "recover_after_oom", lambda device: None)

    with pytest.raises(RuntimeError) as excinfo:
        dispatcher.execute_plan(
            _plan(bt=65536, chunk_size=ChunkSize.fast()),
            torch.randn(65536, 2),
            torch.randn(65536, 2),
            torch.randn(4, 2),
            torch.zeros(65536, dtype=torch.long),
            _labels_prevalidated=True,
        )

    message = str(excinfo.value)
    # The 16 chunks that actually ran, not 65536/1024 = 64.
    assert "Used 16 chunks of 4096 rows" in message
    assert "reduced 93.8%" in message
    assert "~3.1% more" in message
    assert "num_chunks=32" in message
    assert "launch count from 16 to 32" in message
    assert "max_chunks" not in message


def test_dynamic_plan_oom_at_the_ceiling_reports_the_marginal_gain(monkeypatch):
    dispatcher.clear_chunk_cache()
    monkeypatch.setattr(dispatcher, "_call_plan_executor", _raise_oom)
    monkeypatch.setattr(dispatcher, "recover_after_oom", lambda device: None)

    with pytest.raises(RuntimeError) as excinfo:
        dispatcher.execute_plan(
            _plan(bt=65536, chunk_size=ChunkSize.dynamic()),
            torch.randn(65536, 2),
            torch.randn(65536, 2),
            torch.randn(4, 2),
            torch.zeros(65536, dtype=torch.long),
            _labels_prevalidated=True,
        )

    message = str(excinfo.value)
    # Dynamic starts at its 32-chunk ceiling here, so it never gets a retry.
    assert "Used 32 chunks of 2048 rows" in message
    assert "reduced 96.9%" in message
    assert "~1.6% more" in message
    assert "num_chunks=64" in message
    assert "max_chunks" not in message
    dispatcher.clear_chunk_cache()


def test_chunk_oom_stops_suggesting_at_one_row_per_chunk(monkeypatch):
    dispatcher.clear_chunk_cache()
    monkeypatch.setattr(dispatcher, "_call_plan_executor", _raise_oom)
    monkeypatch.setattr(dispatcher, "recover_after_oom", lambda device: None)

    with pytest.raises(RuntimeError) as excinfo:
        dispatcher.execute_plan(
            _plan(bt=4, chunk_size=ChunkSize.dynamic(), max_chunks=8),
            torch.randn(4, 2),
            torch.randn(4, 2),
            torch.randn(4, 2),
            torch.zeros(4, dtype=torch.long),
            _labels_prevalidated=True,
        )

    message = str(excinfo.value)
    assert "Used 4 chunks of 1 rows" in message
    assert "nothing further to split" in message
    assert "num_chunks=" not in message
    assert "max_chunks" not in message
    dispatcher.clear_chunk_cache()


def test_fast_plan_oom_stays_silent_when_num_chunks_was_explicit(monkeypatch):
    monkeypatch.setattr(dispatcher, "_call_plan_executor", _raise_oom)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        dispatcher.execute_plan(
            _plan(bt=4096, chunk_size=ChunkSize.fast(), num_chunks=8),
            torch.randn(4096, 2),
            torch.randn(4096, 2),
            torch.randn(4, 2),
            torch.zeros(4096, dtype=torch.long),
            _labels_prevalidated=True,
        )


def test_nrows_plan_oom_is_not_rewritten(monkeypatch):
    monkeypatch.setattr(dispatcher, "_call_plan_executor", _raise_oom)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        dispatcher.execute_plan(
            _plan(bt=4096, chunk_size=ChunkSize.nrows(1024)),
            torch.randn(4096, 2),
            torch.randn(4096, 2),
            torch.randn(4, 2),
            torch.zeros(4096, dtype=torch.long),
            _labels_prevalidated=True,
        )
