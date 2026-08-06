from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext

import pytest
import torch
from utils.autotune import (
    runtime_key_from_selection,
    runtime_values,
    selection_kwargs,
)

from orda_kernel._autotune import benchmark as autotune_benchmark
from orda_kernel._autotune import cache as autotune_cache
from orda_kernel._autotune import keys as autotune_keys
from orda_kernel._autotune import tuner as autotune_tuner
from orda_kernel._autotune._env import _bounded_env_int
from orda_kernel._autotune.benchmark import (
    BenchmarkTask,
    FullBatchLaunchRequest,
    FullBatchLaunchShape,
    _weighted_median,
    adaptive_robust_repetitions,
    bench_config,
    staged_invocation_bound,
)
from orda_kernel._autotune.models import (
    AUTOTUNE_SCHEMA_VERSION,
    PRECOMPUTED_ALIGNED_SPAN,
    TEACHER_GRAD_CE,
    TEACHER_GRAD_NONE,
    TIED_TRAINED,
    AutotuneRuntimeKey,
    AutotuneWorkload,
    LaunchConfig,
)


@pytest.fixture(autouse=True)
def _isolated_autotune_state(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDA_AUTOTUNE_CACHE_DIR", str(tmp_path / "autotune"))
    autotune_tuner.reset_autotune_cache()
    yield
    autotune_tuner.reset_autotune_cache()


def _key(**overrides) -> str:
    values = {
        "vocab_size": 32768,
        "bt": 2048,
        "max_fused_size": 32768,
        "shape_key": (1024,),
        "chunk_rows": 1024,
        "chunk_dynamic": False,
    }
    values.update(overrides)
    values = runtime_values(torch, **values)
    return autotune_keys._workload(**values).disk_key()


# ── Candidate and key contracts ──────────────────────────────────────────────
def test_bounded_env_int_reports_malformed_and_clamped_values(monkeypatch):
    monkeypatch.setenv("ORDA_TEST_BOUNDED_INT", "bad")
    with pytest.warns(RuntimeWarning, match="not an integer"):
        assert _bounded_env_int("ORDA_TEST_BOUNDED_INT", 3, minimum=1, maximum=5) == 3

    monkeypatch.setenv("ORDA_TEST_BOUNDED_INT", "0")
    with pytest.warns(RuntimeWarning, match=r"outside \[1, 5\].*using 1"):
        assert _bounded_env_int("ORDA_TEST_BOUNDED_INT", 3, minimum=1, maximum=5) == 1


def test_valid_candidates_never_exceed_effective_block_size():
    candidates = autotune_tuner.valid_candidates(8192, 8192)
    assert candidates
    assert all(config.block_size <= 8192 for config in candidates)


@pytest.mark.parametrize(
    ("max_fused_size", "expected_blocks"),
    [
        (16384, {8192, 16384}),
        (32768, {8192, 16384, 32768}),
    ],
)
def test_candidate_grid_preserves_only_supported_large_blocks(
    max_fused_size, expected_blocks
):
    candidates = autotune_tuner.valid_candidates(65536, max_fused_size)
    assert {candidate.block_size for candidate in candidates} == expected_blocks
    assert {candidate.num_warps for candidate in candidates} == {8, 16, 32}


def test_candidate_grid_caps_warps_for_wave64_device(monkeypatch):
    monkeypatch.setattr(autotune_tuner, "_max_num_warps_for_device", lambda _: 16)
    device = torch.device("cuda")

    candidates = autotune_tuner.valid_candidates(65536, 32768, device)

    assert {candidate.num_warps for candidate in candidates} == {8, 16}
    fallback = autotune_tuner.default_config(
        TIED_TRAINED,
        65536,
        32768,
        device=device,
        _is_t4_device_fn=lambda _: False,
    )
    assert fallback == LaunchConfig(16384, 16)


def test_sub_8192_effective_cap_uses_default_without_benchmark_candidates():
    assert autotune_tuner.valid_candidates(4096, 4096) == []
    assert autotune_tuner.default_config(TIED_TRAINED, 4096, 4096).block_size == 4096


def test_cache_key_contains_schema_bt_and_canonical_chunk_policy():
    payload = json.loads(_key())
    assert payload["schema_version"] == AUTOTUNE_SCHEMA_VERSION
    assert payload["bt"] == 2048
    assert payload["chunk_rows"] == 1024
    assert payload["chunk_dynamic"] is False
    assert not {
        "full_chunk_count",
        "tail_rows",
        "num_chunks",
    }.intersection(payload)
    assert "assume_valid_labels" not in payload


def test_trusted_label_setting_is_not_an_autotune_workload_dimension():
    assert "assume_valid_labels" not in AutotuneRuntimeKey._fields
    assert "assume_valid_labels" not in AutotuneWorkload.__dataclass_fields__


def test_disk_identity_keeps_bt_without_derived_topology():
    payload = json.loads(_key(bt=4096))
    assert payload["bt"] == 4096
    assert all(
        field not in payload
        for field in ("full_chunk_count", "tail_rows", "num_chunks")
    )


def test_cache_key_separates_specialization_and_timed_dimensions():
    base = _key()
    variants = [
        _key(bt=4096),
        _key(chunk_rows=512),
        _key(compute_kl=False),
        _key(t_is_one=True),
        _key(compute_teacher_ce=False),
        _key(kl_variant="jeffreys"),
        _key(teacher_grad_policy=TEACHER_GRAD_NONE),
        _key(grad_student_hidden=False),
        _key(grad_teacher_hidden=False),
        _key(grad_weight_accum_dtype=None),
        _key(grad_teacher_weight_accum_dtype="torch.float16"),
        _key(grad_weight_accum_dtype="torch.float32"),
        _key(
            pipeline_key=(
                "logits",
                "allow_teacher_neg_inf_support",
                "torch.float32",
                "contiguous",
            )
        ),
    ]
    assert all(candidate != base for candidate in variants)
    assert len(set(variants)) == len(variants)


@pytest.mark.parametrize("compute_kl", [False, True])
def test_gradient_materialization_scaling_separates_hot_and_disk_identity(
    compute_kl,
):
    unscaled = runtime_values(
        torch,
        compute_kl=compute_kl,
        gradient_materialization_scaled=False,
    )
    scaled = {
        **unscaled,
        "gradient_materialization_scaled": True,
    }

    unscaled_runtime = autotune_keys._runtime_key(**unscaled)
    scaled_runtime = autotune_keys._runtime_key(**scaled)
    unscaled_workload = autotune_keys._workload(**unscaled)
    scaled_workload = autotune_keys._workload(**scaled)
    scaled_payload = json.loads(scaled_workload.disk_key())

    assert unscaled_runtime != scaled_runtime
    assert unscaled_workload != scaled_workload
    assert unscaled_workload.disk_key() != scaled_workload.disk_key()
    assert scaled_runtime.gradient_materialization_scaled is True
    assert scaled_payload["gradient_materialization_scaled"] is True


def test_inactive_kl_identity_is_canonical_in_hot_and_disk_keys():
    first = _key(
        compute_kl=False,
        t_is_one=False,
        kl_variant="cojeffreys",
        pipeline_key=("hidden_weight", "1024"),
    )
    second = _key(
        compute_kl=False,
        t_is_one=True,
        kl_variant="forward",
    )
    assert first == second

    first_runtime = runtime_key_from_selection(
        selection_kwargs(
            torch,
            compute_kl=False,
            t_is_one=False,
            kl_variant="cojeffreys",
            pipeline_key=("hidden_weight",),
        )
    )
    second_runtime = runtime_key_from_selection(
        selection_kwargs(
            torch,
            compute_kl=False,
            t_is_one=True,
            kl_variant="forward",
        )
    )
    assert first_runtime == second_runtime


def test_aligned_span_keeps_pipeline_identity_without_tokenwise_kl():
    base = {
        "mode": PRECOMPUTED_ALIGNED_SPAN,
        "compute_kl": False,
        "t_is_one": False,
        "kl_variant": "cojeffreys",
        "teacher_grad_policy": TEACHER_GRAD_NONE,
    }
    span_active = _key(**base, pipeline_key=("aligned_span", "span_active"))
    span_inactive = _key(**base, pipeline_key=("aligned_span", "span_inactive"))
    no_pipeline = _key(**base)

    assert span_active != span_inactive
    assert span_active != no_pipeline
    payload = json.loads(span_active)
    assert payload["compute_kl"] is False
    assert payload["kl_variant"] == "forward"
    assert payload["pipeline_key"] == ["aligned_span", "span_active"]


def test_models_have_no_independent_tail_tuning_identity():
    workload_fields = AutotuneWorkload.__dataclass_fields__
    runtime_fields = AutotuneRuntimeKey._fields
    for fields in (workload_fields, runtime_fields):
        assert "launch_role" not in fields
        assert "tail_tuned" not in fields
        assert "fixed_full_config" not in fields
        assert "full_chunk_count" not in fields
        assert "tail_rows" not in fields
        assert "num_chunks" not in fields


# ── Hot path semantics ───────────────────────────────────────────────────────
def test_hot_winner_skips_workload_fingerprint_and_disk_construction(monkeypatch):
    values = selection_kwargs(
        torch,
    )
    winner = LaunchConfig(8192, 8)
    key = runtime_key_from_selection(values)
    assert autotune_cache.remember_local_winner(
        key,
        winner,
        generation=autotune_cache.local_generation(),
    )
    monkeypatch.setattr(
        autotune_tuner,
        "_workload",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("hot winner must not construct AutotuneWorkload")
        ),
    )
    monkeypatch.setattr(
        autotune_tuner,
        "default_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hot winner must not resolve a default profile")
        ),
    )

    assert autotune_tuner.select_config(**values) == winner


def test_begin_run_preserves_hot_winner_but_clears_run_state():
    values = selection_kwargs(
        torch,
    )
    winner = LaunchConfig(8192, 8)
    key = runtime_key_from_selection(values)
    assert autotune_cache.remember_local_winner(
        key,
        winner,
        generation=autotune_cache.local_generation(),
    )
    autotune_tuner._mark_tuned(key, 2)
    autotune_tuner._mark_failed(key)

    autotune_tuner.begin_autotune_run()

    assert autotune_cache.get_hot_winner(key) == winner
    assert not autotune_tuner._tuned_to_depth(key, 1)
    assert not autotune_tuner._failed_this_run(key)


def test_hot_winner_is_shared_between_true_and_max(monkeypatch):
    candidate = LaunchConfig(8192, 8)
    calls = []
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [candidate])
    monkeypatch.setattr(
        autotune_tuner,
        "bench_config",
        lambda *_args, **_kwargs: calls.append(candidate) or 1.0,
    )

    values = selection_kwargs(
        torch,
    )
    assert autotune_tuner.select_config(**values) == candidate
    assert autotune_tuner.select_config(**values) == candidate
    assert len(calls) == 2

    max_values = selection_kwargs(torch, autotune="Max")
    assert autotune_tuner.select_config(**max_values) == candidate
    assert autotune_tuner.select_config(**max_values) == candidate
    assert len(calls) == 4


def test_true_refresh_retunes_once_per_runtime_key_and_run(monkeypatch):
    winner = LaunchConfig(8192, 8)
    values = selection_kwargs(torch, refresh_autotune=True)
    key = runtime_key_from_selection(values)
    assert autotune_cache.remember_local_winner(
        key,
        winner,
        generation=autotune_cache.local_generation(),
    )
    calls = []
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [winner])
    monkeypatch.setattr(
        autotune_tuner,
        "bench_config",
        lambda *_args, **_kwargs: calls.append(True) or 1.0,
    )

    assert autotune_tuner.select_config(**values) == winner
    assert autotune_tuner.select_config(**values) == winner
    assert len(calls) == 2

    autotune_tuner.begin_autotune_run()
    assert autotune_tuner.select_config(**values) == winner
    assert len(calls) == 4


def test_failed_regular_selection_retries_after_backoff(monkeypatch):
    candidate = LaunchConfig(8192, 8)
    calls = []
    now = [100.0]
    monkeypatch.setattr(autotune_tuner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [candidate])
    monkeypatch.setattr(
        autotune_tuner,
        "bench_config",
        lambda *_args, **_kwargs: (
            calls.append(candidate)
            or (_ for _ in ()).throw(RuntimeError("temporary resource failure"))
        ),
    )
    monkeypatch.setattr(autotune_tuner, "is_resource_error", lambda _: True)
    fallback = autotune_tuner.default_config(
        TIED_TRAINED,
        8192,
        8192,
        device=torch.device("cpu"),
    )

    values = selection_kwargs(
        torch,
    )
    expected_warning = (
        "ORDA autotune found no viable config; using default "
        f"{fallback.block_size}/{fallback.num_warps}."
    )
    with pytest.warns(RuntimeWarning) as caught:
        assert autotune_tuner.select_config(**values) == fallback
    assert [str(warning.message) for warning in caught] == [expected_warning]
    assert autotune_tuner.select_config(**values) == fallback
    assert len(calls) == 1

    now[0] += 1.1
    with pytest.warns(RuntimeWarning) as caught:
        assert autotune_tuner.select_config(**values) == fallback
    assert [str(warning.message) for warning in caught] == [expected_warning]
    assert len(calls) == 2


def test_failed_key_still_reads_a_peer_winner(monkeypatch):
    winner = LaunchConfig(8192, 8)
    fallback = LaunchConfig(8192, 32)
    values = selection_kwargs(
        torch,
    )
    monkeypatch.setattr(
        autotune_tuner, "default_config", lambda *_args, **_kw: fallback
    )
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [winner])
    monkeypatch.setattr(
        autotune_tuner,
        "bench_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("temporary resource failure")
        ),
    )
    monkeypatch.setattr(autotune_tuner, "is_resource_error", lambda _: True)

    with pytest.warns(RuntimeWarning):
        assert autotune_tuner.select_config(**values) == fallback

    disk_reads = []
    monkeypatch.setattr(
        autotune_tuner,
        "read_disk_winner",
        lambda *_args, **_kwargs: disk_reads.append(True) or winner,
    )
    values["benchmark_factory"] = lambda: (_ for _ in ()).throw(
        AssertionError("peer winner must suppress benchmarking")
    )
    assert autotune_tuner.select_config(**values) == winner
    assert disk_reads == [True]


def test_settled_key_still_reads_a_recovered_store(monkeypatch):
    winner = LaunchConfig(8192, 8)
    values = selection_kwargs(
        torch,
    )
    key = runtime_key_from_selection(values)
    autotune_tuner._mark_settled(key)
    monkeypatch.setattr(autotune_tuner, "read_disk_winner", lambda *_a, **_k: winner)
    values["benchmark_factory"] = lambda: (_ for _ in ()).throw(
        AssertionError("disk winner must suppress benchmarking")
    )

    assert autotune_tuner.select_config(**values) == winner


def test_no_robust_warning_text_is_exact(monkeypatch):
    candidate = LaunchConfig(8192, 8)
    calls = []
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [candidate])

    def screen_then_fail(*_args, **_kwargs):
        calls.append(candidate)
        if len(calls) == 1:
            return 1.0
        raise RuntimeError("temporary resource failure")

    monkeypatch.setattr(autotune_tuner, "bench_config", screen_then_fail)
    monkeypatch.setattr(autotune_tuner, "is_resource_error", lambda _: True)
    fallback = autotune_tuner.default_config(
        TIED_TRAINED,
        8192,
        8192,
        device=torch.device("cpu"),
    )
    expected_warning = (
        "ORDA autotune found no robust config; using default "
        f"{fallback.block_size}/{fallback.num_warps}."
    )

    with pytest.warns(RuntimeWarning) as caught:
        assert (
            autotune_tuner.select_config(
                **selection_kwargs(
                    torch,
                )
            )
            == fallback
        )
    assert [str(warning.message) for warning in caught] == [expected_warning]
    assert len(calls) == 2


def test_failed_max_refresh_preserves_winner_and_retries_after_backoff(monkeypatch):
    winner = LaunchConfig(8192, 8)
    values = selection_kwargs(torch, autotune="Max")
    key = runtime_key_from_selection(values)
    calls = []
    now = [100.0]
    monkeypatch.setattr(autotune_tuner.time, "monotonic", lambda: now[0])
    assert autotune_cache.remember_local_winner(
        key,
        winner,
        generation=autotune_cache.local_generation(),
    )
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [winner])
    monkeypatch.setattr(
        autotune_tuner,
        "bench_config",
        lambda *_args, **_kwargs: (
            calls.append(True)
            or (_ for _ in ()).throw(RuntimeError("temporary resource failure"))
        ),
    )
    monkeypatch.setattr(autotune_tuner, "is_resource_error", lambda _: True)

    with pytest.warns(RuntimeWarning, match="retune found no viable config"):
        assert autotune_tuner.select_config(**values) == winner
    assert autotune_tuner.select_config(**values) == winner
    assert not autotune_tuner._tuned_to_depth(key, 2)
    assert len(calls) == 1

    now[0] += 1.1
    with pytest.warns(RuntimeWarning, match="retune found no viable config"):
        assert autotune_tuner.select_config(**values) == winner
    assert len(calls) == 2


def test_max_reset_during_timing_does_not_revive_the_pre_reset_winner(monkeypatch):
    old_winner = LaunchConfig(8192, 8)
    fallback = LaunchConfig(8192, 32)
    values = selection_kwargs(torch, autotune="Max")
    key = runtime_key_from_selection(values)
    assert autotune_cache.remember_local_winner(
        key,
        old_winner,
        generation=autotune_cache.local_generation(),
    )
    calls = []
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [old_winner])
    monkeypatch.setattr(
        autotune_tuner,
        "default_config",
        lambda *_args, **_kwargs: fallback,
    )

    def reset_during_first_timing(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 1:
            autotune_cache.reset_cache()
        return 1.0

    monkeypatch.setattr(autotune_tuner, "bench_config", reset_during_first_timing)

    assert autotune_tuner.select_config(**values) == fallback
    assert autotune_cache.get_hot_winner(key) is None


def test_device_lock_rechecks_winner_before_constructing_a_benchmark(monkeypatch):
    values = selection_kwargs(
        torch,
    )
    key = runtime_key_from_selection(values)
    winner = LaunchConfig(8192, 8)

    @contextmanager
    def winner_after_lock(_device):
        assert autotune_cache.remember_local_winner(
            key,
            winner,
            generation=autotune_cache.local_generation(),
        )
        yield None

    monkeypatch.setattr(autotune_tuner, "device_benchmark_lock", winner_after_lock)
    monkeypatch.setattr(autotune_tuner, "read_disk_winner", lambda *_, **__: None)
    monkeypatch.setattr(
        autotune_tuner,
        "valid_candidates",
        lambda *_: [winner],
    )
    values["benchmark_factory"] = lambda: (_ for _ in ()).throw(
        AssertionError("post-lock winner must suppress benchmark construction")
    )

    assert autotune_tuner.select_config(**values) == winner


def test_store_unavailable_selection_degrades_to_a_local_hot_winner(monkeypatch):
    values = selection_kwargs(
        torch,
    )
    candidate = LaunchConfig(8192, 8)

    @contextmanager
    def unavailable_store(_device):
        yield None

    monkeypatch.setattr(autotune_tuner, "device_benchmark_lock", unavailable_store)
    monkeypatch.setattr(autotune_tuner, "read_disk_winner", lambda *_, **__: None)
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [candidate])
    monkeypatch.setattr(autotune_tuner, "bench_config", lambda *_, **__: 1.0)

    assert autotune_tuner.select_config(**values) == candidate
    assert (
        autotune_cache.get_hot_winner(runtime_key_from_selection(values)) == candidate
    )


def test_busy_peer_returns_without_starting_an_overlapping_benchmark(monkeypatch):
    values = selection_kwargs(
        torch,
    )
    fallback = autotune_tuner.default_config(
        TIED_TRAINED,
        8192,
        8192,
        device=torch.device("cpu"),
    )

    @contextmanager
    def busy_peer(_device):
        raise autotune_cache._DeviceBenchmarkBusy("cpu:None")
        yield

    monkeypatch.setattr(autotune_tuner, "device_benchmark_lock", busy_peer)
    monkeypatch.setattr(autotune_tuner, "read_disk_winner", lambda *_, **__: None)
    values["benchmark_factory"] = lambda: (_ for _ in ()).throw(
        AssertionError("busy peer must suppress local benchmarking")
    )

    assert autotune_tuner.select_config(**values) == fallback


def test_tuner_passes_lease_heartbeat_to_each_benchmark(monkeypatch):
    candidate = LaunchConfig(8192, 8)
    heartbeat_callbacks = []
    renewal_leases = []
    lease = autotune_cache.DeviceBenchmarkLease(
        device_key="cpu:None",
        owner_id="unit-test",
        path=autotune_cache.cache_path(),
        generation=0,
    )

    @contextmanager
    def owned_device_lock(_device):
        yield lease

    def bench_with_heartbeat(*_args, **kwargs):
        heartbeat = kwargs.get("heartbeat")
        assert heartbeat is not None
        heartbeat_callbacks.append(heartbeat)
        heartbeat()
        return 1.0

    monkeypatch.setattr(autotune_tuner, "device_benchmark_lock", owned_device_lock)
    monkeypatch.setattr(autotune_tuner, "read_disk_winner", lambda *_, **__: None)
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [candidate])
    monkeypatch.setattr(autotune_tuner, "bench_config", bench_with_heartbeat)
    monkeypatch.setattr(
        autotune_tuner,
        "renew_device_benchmark_lease",
        lambda current: renewal_leases.append(current) or True,
    )
    monkeypatch.setattr(autotune_tuner, "persist_winner", lambda *_, **__: True)

    assert (
        autotune_tuner.select_config(
            **selection_kwargs(
                torch,
            )
        )
        == candidate
    )
    assert len(heartbeat_callbacks) == 2
    assert renewal_leases
    assert all(current == lease for current in renewal_leases)


# ── Full-batch timing controls ───────────────────────────────────────────────
def test_full_batch_shape_tracks_full_chunks_and_tail():
    assert tuple(FullBatchLaunchShape.__dataclass_fields__) == (
        "bt",
        "chunk_rows",
        "chunk_dynamic",
    )
    shape = FullBatchLaunchShape.from_attempt(
        bt=10,
        chunk_rows=4,
        chunk_dynamic=True,
    )
    assert shape.full_chunk_count == 2
    assert shape.tail_rows == 2
    assert shape.num_chunks == 3
    assert shape.iter_chunk_rows() == (4, 4, 2)


def test_full_batch_shape_derives_a_tail_only_execution():
    shape = FullBatchLaunchShape.from_attempt(
        bt=5,
        chunk_rows=8,
        chunk_dynamic=False,
    )
    assert shape.full_chunk_count == 0
    assert shape.tail_rows == 5
    assert shape.num_chunks == 1
    assert shape.iter_chunk_rows() == (5,)


def test_autotune_disabled_full_plan_does_not_build_a_benchmark_task(monkeypatch):
    default = LaunchConfig(8192, 8)
    monkeypatch.setattr(autotune_tuner, "default_config", lambda *_, **__: default)
    monkeypatch.setattr(
        autotune_tuner,
        "select_config",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("autotune-disabled path must not select")
        ),
    )

    request = FullBatchLaunchRequest(
        mode=TIED_TRAINED,
        device=torch.device("cpu"),
        dtype=torch.float32,
        vocab_size=32768,
        bt=10,
        chunk_rows=4,
        chunk_dynamic=False,
        max_fused_size=32768,
        shape_key=(16,),
        autotune=False,
        refresh_autotune=False,
        compute_kl=False,
        t_is_one=True,
        compute_teacher_ce=False,
        compute_student_ce=True,
        kl_variant="forward",
        teacher_grad_policy=TEACHER_GRAD_NONE,
        grad_student_hidden=True,
        grad_teacher_hidden=False,
        grad_weight_accum_dtype="torch.float16",
        grad_teacher_weight_accum_dtype=None,
    )
    plan = autotune_tuner.select_full_batch_launch_plan(
        request,
        benchmark_factory=lambda: (_ for _ in ()).throw(AssertionError("no task")),
    )
    assert plan.config == default


def test_repeated_full_plan_calls_use_hot_config_without_workload_rebuild(monkeypatch):
    candidate = LaunchConfig(8192, 8)
    workload_calls = []
    original_workload = autotune_tuner._workload
    monkeypatch.setattr(autotune_tuner, "valid_candidates", lambda *_: [candidate])
    monkeypatch.setattr(autotune_tuner, "bench_config", lambda *_, **__: 1.0)

    def counting_workload(**kwargs):
        workload_calls.append(kwargs)
        return original_workload(**kwargs)

    monkeypatch.setattr(autotune_tuner, "_workload", counting_workload)
    request = FullBatchLaunchRequest(
        mode=TIED_TRAINED,
        device=torch.device("cpu"),
        dtype=torch.float16,
        vocab_size=8192,
        bt=5,
        chunk_rows=2,
        chunk_dynamic=False,
        max_fused_size=8192,
        shape_key=(8,),
        autotune=True,
        refresh_autotune=False,
        compute_kl=True,
        t_is_one=False,
        compute_teacher_ce=True,
        compute_student_ce=True,
        kl_variant="renyi",
        teacher_grad_policy=TEACHER_GRAD_CE,
        grad_student_hidden=True,
        grad_teacher_hidden=True,
        grad_weight_accum_dtype="torch.float16",
        grad_teacher_weight_accum_dtype=None,
        renyi_near_one=True,
        gradient_materialization_scaled=True,
    )

    first = autotune_tuner.select_full_batch_launch_plan(
        request, benchmark_factory=lambda: object()
    )
    second = autotune_tuner.select_full_batch_launch_plan(
        request, benchmark_factory=lambda: object()
    )
    assert first.config == second.config == candidate
    assert first is not second
    assert len(workload_calls) == 1
    assert workload_calls[0]["pipeline_key"] == ("renyi-near-one",)
    assert workload_calls[0]["gradient_materialization_scaled"] is True


@pytest.mark.parametrize(
    ("autotune", "expected_controls", "expected_bound"),
    [
        (True, (2, 3, 3, 2, 4, 12), 87),
        ("Max", (2, 3, 3, 3, 6, 18), 108),
    ],
)
def test_staged_controls_and_full_plan_bound_are_exact(
    autotune, expected_controls, expected_bound
):
    controls = autotune_benchmark.autotune_controls(autotune)
    assert (
        controls.screen_warmup_reps,
        controls.screen_reps,
        controls.finalists,
        controls.robust_warmup_reps,
        controls.robust_min_reps,
        controls.robust_max_reps,
    ) == expected_controls
    assert (
        staged_invocation_bound(candidate_count=9, autotune=autotune) == expected_bound
    )
    assert (
        staged_invocation_bound(
            candidate_count=9,
            tuned_roles=1,
            autotune=autotune,
        )
        == expected_bound
    )


def test_adaptive_robust_repetitions_stays_within_validated_bounds():
    for timing in (0.0, 0.01, 1.0, 1000.0):
        repetitions = adaptive_robust_repetitions(timing, minimum=2, maximum=9)
        assert 2 <= repetitions <= 9


def test_weighted_median_matches_the_odd_expanded_sample_median():
    assert _weighted_median([(1.0, 1), (5.0, 3), (9.0, 1)]) == 5.0


def test_weighted_median_averages_the_two_even_expanded_middle_samples():
    assert _weighted_median([(1.0, 2), (5.0, 2)]) == 3.0


def test_screen_defaults_measure_independent_intervals(monkeypatch):
    runs = []
    intervals = []

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            pass

        def synchronize(self):
            intervals.append(True)

        def elapsed_time(self, _other):
            return 1.0

    monkeypatch.setattr(
        autotune_benchmark.torch.cuda,
        "device",
        lambda _: nullcontext(),
    )
    monkeypatch.setattr(autotune_benchmark.torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(autotune_benchmark.torch.cuda, "Event", FakeEvent)

    bench_config(
        BenchmarkTask(run=lambda *_: runs.append(True)),
        LaunchConfig(8192, 8),
        device=torch.device("cuda", 0),
        warmup_reps=0,
        repetitions=autotune_benchmark.AUTOTUNE_SCREEN_REPS,
    )
    assert len(runs) == len(intervals) == 3


def test_bench_config_weights_a_short_final_batch_by_its_repetition_count(
    monkeypatch,
):
    calls = []
    preparations = []
    heartbeats = []

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.call_count = None

        def record(self):
            self.call_count = len(calls)

        def synchronize(self):
            assert self.call_count is not None

        def elapsed_time(self, other):
            assert self.call_count is not None
            assert other.call_count is not None
            return 20.0 if self.call_count == 1 else 100.0

    def run(_block_size, _num_warps):
        calls.append(True)

    monkeypatch.setattr(autotune_benchmark, "AUTOTUNE_TIMING_BATCH", 2)
    monkeypatch.setattr(
        autotune_benchmark.torch.cuda,
        "device",
        lambda _: nullcontext(),
    )
    monkeypatch.setattr(autotune_benchmark.torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(autotune_benchmark.torch.cuda, "Event", FakeEvent)

    score = bench_config(
        BenchmarkTask(run=run, prepare_batch=lambda reps: preparations.append(reps)),
        LaunchConfig(8192, 8),
        device=torch.device("cuda", 0),
        warmup_reps=1,
        repetitions=3,
        heartbeat=lambda: heartbeats.append(len(calls)),
    )
    assert calls == [True] * 4
    assert preparations == [1, 2, 1]
    assert heartbeats == [1, 3, 4]
    assert score == 10.0
