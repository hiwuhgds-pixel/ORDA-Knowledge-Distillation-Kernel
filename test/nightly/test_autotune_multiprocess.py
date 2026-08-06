from __future__ import annotations

import multiprocessing
import os
import queue
import traceback
import warnings
from pathlib import Path

import pytest
import torch
from utils.autotune import identity, selection_kwargs

from orda_kernel._autotune import cache
from orda_kernel._autotune import tuner as autotune_tuner
from orda_kernel._autotune.models import AUTOTUNE_SCHEMA_VERSION, LaunchConfig

_DEFAULT_CHILD_TIMEOUT_SECONDS = "120" if os.name == "nt" else "30"
_CHILD_TIMEOUT_SECONDS = float(
    os.environ.get("ORDA_TEST_CHILD_TIMEOUT_SECONDS", _DEFAULT_CHILD_TIMEOUT_SECONDS)
)


def _child_result(result_queue, processes, label: str):
    try:
        result = result_queue.get(timeout=_CHILD_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        status = [
            (process.pid, process.exitcode, process.is_alive()) for process in processes
        ]
        raise AssertionError(
            f"{label} produced no result; process state={status}"
        ) from exc
    if result[0] == "error":
        raise AssertionError(f"{label} child failed:\n{result[1]}")
    return result[1:]


def _join_or_terminate(process):
    if process is None:
        return
    process.join(timeout=_CHILD_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join(timeout=_CHILD_TIMEOUT_SECONDS)


def _close_queue(result_queue):
    result_queue.close()
    result_queue.join_thread()


def _cold_init_worker(cache_file_text: str, barrier, result_queue) -> None:
    try:
        barrier.wait(timeout=_CHILD_TIMEOUT_SECONDS)
        connection = cache._connect(Path(cache_file_text))
        if connection is None:
            raise RuntimeError("child could not open the SQLite cache")
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            connection.close()
        result_queue.put(("ok", version, tables))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))


def _selection_worker(
    cache_dir_text: str,
    role: str,
    owner_started,
    release_owner,
    result_queue,
) -> None:
    original_cache_dir = os.environ.get("ORDA_AUTOTUNE_CACHE_DIR")
    original_candidates = autotune_tuner.valid_candidates
    original_bench = autotune_tuner.bench_config
    try:
        os.environ["ORDA_AUTOTUNE_CACHE_DIR"] = cache_dir_text
        candidate = LaunchConfig(8192, 8)
        calls = []
        autotune_tuner.valid_candidates = lambda *_: [candidate]

        def bench_once(*_args, **_kwargs):
            calls.append(True)
            if role == "owner" and len(calls) == 1:
                owner_started.set()
                if not release_owner.wait(timeout=_CHILD_TIMEOUT_SECONDS):
                    raise RuntimeError("parent did not release benchmark owner")
            return 1.0

        autotune_tuner.bench_config = bench_once
        selected = autotune_tuner.select_config(**selection_kwargs(torch))
        result_queue.put(("ok", role, tuple(selected), len(calls)))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
    finally:
        autotune_tuner.valid_candidates = original_candidates
        autotune_tuner.bench_config = original_bench
        if original_cache_dir is None:
            os.environ.pop("ORDA_AUTOTUNE_CACHE_DIR", None)
        else:
            os.environ["ORDA_AUTOTUNE_CACHE_DIR"] = original_cache_dir


def _stale_owner_worker(cache_dir_text: str, commit_event, release_event, result_queue):
    original_cache_dir = os.environ.get("ORDA_AUTOTUNE_CACHE_DIR")
    try:
        os.environ["ORDA_AUTOTUNE_CACHE_DIR"] = cache_dir_text
        key, workload = identity(torch)
        lease = cache.try_acquire_device_benchmark_lease("cpu:None")
        if lease is None:
            raise RuntimeError("stale owner did not acquire the device lease")
        try:
            result_queue.put(("ok", "lease", lease.generation))
            if not commit_event.wait(timeout=_CHILD_TIMEOUT_SECONDS):
                raise RuntimeError("parent did not request stale commit")
            persisted = cache.persist_winner(
                key,
                workload,
                LaunchConfig(8192, 8),
                lease=lease,
            )
            result_queue.put(("ok", "persisted", persisted))
            if not release_event.wait(timeout=_CHILD_TIMEOUT_SECONDS):
                raise RuntimeError("parent did not release stale owner")
        finally:
            cache.release_device_benchmark_lease(lease)
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
    finally:
        if original_cache_dir is None:
            os.environ.pop("ORDA_AUTOTUNE_CACHE_DIR", None)
        else:
            os.environ["ORDA_AUTOTUNE_CACHE_DIR"] = original_cache_dir


def _unavailable_store_worker(cache_dir_text: str, result_queue) -> None:
    original_cache_dir = os.environ.get("ORDA_AUTOTUNE_CACHE_DIR")
    original_candidates = autotune_tuner.valid_candidates
    original_bench = autotune_tuner.bench_config
    try:
        os.environ["ORDA_AUTOTUNE_CACHE_DIR"] = cache_dir_text
        candidate = LaunchConfig(8192, 8)
        autotune_tuner.valid_candidates = lambda *_: [candidate]
        calls = []
        autotune_tuner.bench_config = lambda *_args, **_kwargs: (
            calls.append(True) or 1.0
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selected = autotune_tuner.select_config(**selection_kwargs(torch))
        result_queue.put(
            (
                "ok",
                tuple(selected),
                len(calls),
                [str(warning.message) for warning in caught],
            )
        )
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
    finally:
        autotune_tuner.valid_candidates = original_candidates
        autotune_tuner.bench_config = original_bench
        if original_cache_dir is None:
            os.environ.pop("ORDA_AUTOTUNE_CACHE_DIR", None)
        else:
            os.environ["ORDA_AUTOTUNE_CACHE_DIR"] = original_cache_dir


# ── Spawned process contracts ────────────────────────────────────────────────
@pytest.mark.multiprocess
def test_cold_schema_initializes_across_spawned_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    cache_file = tmp_path / "cold" / "autotune.sqlite3"
    result_queue = context.Queue()
    worker_count = 4
    barrier = context.Barrier(worker_count)
    workers = [
        context.Process(
            target=_cold_init_worker,
            args=(str(cache_file), barrier, result_queue),
        )
        for _ in range(worker_count)
    ]
    try:
        for worker in workers:
            worker.start()
        results = [
            _child_result(result_queue, workers, "cold schema initialization")
            for _ in workers
        ]
        for worker in workers:
            _join_or_terminate(worker)
        assert all(worker.exitcode == 0 for worker in workers)
        assert all(
            version == AUTOTUNE_SCHEMA_VERSION
            and {"cache_state", "entries", "device_leases"}.issubset(tables)
            for version, tables in results
        )
    finally:
        for worker in workers:
            _join_or_terminate(worker)
        _close_queue(result_queue)


@pytest.mark.multiprocess
def test_corrupt_schema_recovers_once_across_spawned_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    cache_file = tmp_path / "corrupt" / "autotune.sqlite3"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"not a database")
    result_queue = context.Queue()
    worker_count = 4
    barrier = context.Barrier(worker_count)
    workers = [
        context.Process(
            target=_cold_init_worker,
            args=(str(cache_file), barrier, result_queue),
        )
        for _ in range(worker_count)
    ]
    try:
        for worker in workers:
            worker.start()
        results = [
            _child_result(result_queue, workers, "corrupt schema recovery")
            for _ in workers
        ]
        for worker in workers:
            _join_or_terminate(worker)
        assert all(worker.exitcode == 0 for worker in workers)
        assert all(
            version == AUTOTUNE_SCHEMA_VERSION
            and {"cache_state", "entries", "device_leases"}.issubset(tables)
            for version, tables in results
        )
        assert cache._recovery_lock_path(cache_file).exists()
    finally:
        for worker in workers:
            _join_or_terminate(worker)
        _close_queue(result_queue)


@pytest.mark.multiprocess
def test_same_device_true_rechecks_winner_after_serialized_benchmark(tmp_path):
    context = multiprocessing.get_context("spawn")
    cache_dir = tmp_path / "shared"
    result_queue = context.Queue()
    owner_started = context.Event()
    release_owner = context.Event()
    owner = context.Process(
        target=_selection_worker,
        args=(str(cache_dir), "owner", owner_started, release_owner, result_queue),
    )
    waiter = None
    try:
        owner.start()
        assert owner_started.wait(timeout=_CHILD_TIMEOUT_SECONDS)
        waiter = context.Process(
            target=_selection_worker,
            args=(str(cache_dir), "waiter", owner_started, release_owner, result_queue),
        )
        waiter.start()
        release_owner.set()
        first = _child_result(result_queue, (owner, waiter), "selection owner")
        second = _child_result(result_queue, (owner, waiter), "selection waiter")
        results = {item[0]: item[1:] for item in (first, second)}
        _join_or_terminate(owner)
        _join_or_terminate(waiter)
        assert owner.exitcode == 0
        assert waiter.exitcode == 0
        assert results["owner"] == ((8192, 8), 2)
        assert results["waiter"] == ((8192, 8), 0)
    finally:
        release_owner.set()
        _join_or_terminate(owner)
        _join_or_terminate(waiter)
        _close_queue(result_queue)


@pytest.mark.multiprocess
def test_reset_fences_spawned_device_owner_and_preserves_lease_serialization(
    tmp_path,
    monkeypatch,
):
    context = multiprocessing.get_context("spawn")
    cache_dir = tmp_path / "reset"
    monkeypatch.setenv("ORDA_AUTOTUNE_CACHE_DIR", str(cache_dir))
    cache.reset_cache()
    result_queue = context.Queue()
    commit_event = context.Event()
    release_event = context.Event()
    owner = context.Process(
        target=_stale_owner_worker,
        args=(str(cache_dir), commit_event, release_event, result_queue),
    )
    fresh = None
    try:
        owner.start()
        label, generation = _child_result(result_queue, (owner,), "stale owner lease")
        assert label == "lease"
        assert generation == 0

        cache.reset_cache()
        assert cache.try_acquire_device_benchmark_lease("cpu:None") is None
        commit_event.set()
        label, persisted = _child_result(result_queue, (owner,), "stale owner commit")
        assert (label, persisted) == ("persisted", False)

        release_event.set()
        _join_or_terminate(owner)
        assert owner.exitcode == 0

        key, workload = identity(torch)
        fresh = cache.try_acquire_device_benchmark_lease("cpu:None")
        assert fresh is not None
        assert cache.persist_winner(
            key,
            workload,
            LaunchConfig(16384, 16),
            lease=fresh,
        )
    finally:
        commit_event.set()
        release_event.set()
        if fresh is not None:
            cache.release_device_benchmark_lease(fresh)
        _join_or_terminate(owner)
        _close_queue(result_queue)


@pytest.mark.multiprocess
def test_spawned_store_unavailable_selection_degrades_to_local_tuning(tmp_path):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    worker = context.Process(
        target=_unavailable_store_worker,
        args=(str(blocked_parent), result_queue),
    )
    try:
        worker.start()
        selected, benchmark_calls, messages = _child_result(
            result_queue,
            (worker,),
            "store unavailable selection",
        )
        _join_or_terminate(worker)
        assert worker.exitcode == 0
        assert selected == (8192, 8)
        assert benchmark_calls == 2
        assert sum("process-local autotuning" in message for message in messages) == 1
    finally:
        _join_or_terminate(worker)
        _close_queue(result_queue)
