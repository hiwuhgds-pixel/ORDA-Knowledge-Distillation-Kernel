from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import replace

import pytest
import torch
from utils.autotune import indexed_identity

from orda_kernel._autotune import cache
from orda_kernel._autotune._contract import contract_id_for_source_digest
from orda_kernel._autotune.models import (
    AUTOTUNE_SCHEMA_VERSION,
    LaunchConfig,
)


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDA_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    cache.reset_cache()
    yield cache.cache_path()
    cache.reset_cache()


def _persist(path, key, workload, config, *, owner_id: str | None = None):
    lease = cache.try_acquire_device_benchmark_lease(
        "cpu:None",
        owner_id=owner_id,
        path=path,
    )
    assert lease is not None
    try:
        assert cache.persist_winner(key, workload, config, lease=lease)
    finally:
        cache.release_device_benchmark_lease(lease)


def _read_disk_winners(path) -> dict[str, LaunchConfig]:
    """Snapshot every persisted winner for a store; test-only observation helper."""
    connection = cache._connect(path)
    if connection is None:
        return {}
    try:
        return {
            str(cache_key): LaunchConfig(int(block_size), int(num_warps))
            for cache_key, block_size, num_warps in connection.execute(
                "SELECT cache_key, block_size, num_warps FROM entries"
            )
        }
    finally:
        connection.close()


# ── Positive winner store ────────────────────────────────────────────────────
def test_positive_disk_roundtrip_promotes_the_single_hot_winner(isolated_cache):
    key, workload = indexed_identity(torch, 0)
    config = LaunchConfig(8192, 8)
    _persist(isolated_cache, key, workload, config)
    cache.invalidate_hot_winner(key)

    loaded = cache.read_disk_winner(
        key,
        workload,
        valid_configs={config},
        fallback=LaunchConfig(16384, 32),
    )
    assert loaded == config
    assert cache.get_hot_winner(key) == config


def test_positive_disk_winner_does_not_cross_timed_contract_identity(isolated_cache):
    key, workload = indexed_identity(torch, 0)
    config = LaunchConfig(8192, 8)
    _persist(isolated_cache, key, workload, config)

    changed_contract_id = contract_id_for_source_digest("f" * 64)
    changed_key = key._replace(timed_contract_id=changed_contract_id)
    changed_workload = replace(workload, timed_contract_id=changed_contract_id)

    assert changed_key != key
    assert changed_workload.disk_key() != workload.disk_key()
    assert (
        cache.read_disk_winner(
            changed_key,
            changed_workload,
            valid_configs={config},
            fallback=LaunchConfig(16384, 32),
        )
        is None
    )
    assert cache.get_hot_winner(changed_key) is None


def test_positive_store_prunes_by_deterministic_lru_order(isolated_cache, monkeypatch):
    monkeypatch.setattr(cache, "_MAX_DISK_ENTRIES", 2)
    first_key, first_workload = indexed_identity(torch, 0)
    second_key, second_workload = indexed_identity(torch, 1)
    third_key, third_workload = indexed_identity(torch, 2)
    first = LaunchConfig(8192, 8)
    second = LaunchConfig(8192, 16)
    third = LaunchConfig(8192, 32)

    _persist(isolated_cache, first_key, first_workload, first)
    _persist(isolated_cache, second_key, second_workload, second)
    cache.invalidate_hot_winner(first_key)
    assert (
        cache.read_disk_winner(
            first_key,
            first_workload,
            valid_configs={first},
            fallback=first,
        )
        == first
    )
    _persist(isolated_cache, third_key, third_workload, third)

    winners = _read_disk_winners(isolated_cache)
    assert set(winners) == {first_workload.disk_key(), third_workload.disk_key()}


def test_compact_hot_winner_cache_keeps_lru_behavior(isolated_cache, monkeypatch):
    monkeypatch.setattr(cache, "_MAX_HOT_ENTRIES", 2)
    first_key, _ = indexed_identity(torch, 0)
    second_key, _ = indexed_identity(torch, 1)
    third_key, _ = indexed_identity(torch, 2)
    generation = cache.local_generation()
    assert cache.remember_local_winner(
        first_key,
        LaunchConfig(8192, 8),
        generation=generation,
    )
    assert cache.remember_local_winner(
        second_key,
        LaunchConfig(8192, 16),
        generation=generation,
    )
    assert cache.get_hot_winner(first_key) == LaunchConfig(8192, 8)
    assert cache.remember_local_winner(
        third_key,
        LaunchConfig(8192, 32),
        generation=generation,
    )

    assert cache.get_hot_winner(first_key) == LaunchConfig(8192, 8)
    assert cache.get_hot_winner(second_key) is None
    assert cache.get_hot_winner(third_key) == LaunchConfig(8192, 32)


def test_store_has_no_durable_negative_table(isolated_cache):
    connection = cache._connect(isolated_cache)
    assert connection is not None
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()
    assert {"cache_state", "entries", "device_leases"}.issubset(tables)
    assert "negative_entries" not in tables


# ── Disposable schema lifecycle ──────────────────────────────────────────────
@pytest.mark.parametrize("payload", [b"not a database", b"SQLite format 3\x00bad"])
def test_corrupt_owned_store_recovers_once(isolated_cache, payload):
    isolated_cache.parent.mkdir(parents=True, exist_ok=True)
    isolated_cache.write_bytes(payload)

    connection = cache._connect(isolated_cache)
    assert connection is not None
    try:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == AUTOTUNE_SCHEMA_VERSION
        )
        generation = connection.execute(
            "SELECT generation FROM cache_state WHERE id = 1"
        ).fetchone()
        assert generation == (0,)
    finally:
        connection.close()


def test_recovery_waits_for_open_package_connections(isolated_cache):
    connection = cache._connect(isolated_cache)
    assert connection is not None
    started = threading.Event()
    acquired = threading.Event()

    def recover():
        started.set()
        with cache._database_recovery_lock(isolated_cache) as owns_lock:
            if owns_lock:
                acquired.set()

    worker = threading.Thread(target=recover)
    worker.start()
    assert started.wait(timeout=5)
    assert not acquired.wait(timeout=0.05)
    connection.close()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert acquired.is_set()


def test_recovery_lock_file_is_persistent_not_stale_owned(isolated_cache):
    lock_path = cache._recovery_lock_path(isolated_cache)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"prior owner")

    with cache._database_recovery_lock(isolated_cache) as acquired:
        assert acquired
    assert lock_path.exists()


def test_same_version_missing_table_recreates_disposable_schema(isolated_cache):
    isolated_cache.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(isolated_cache)
    try:
        connection.execute(f"PRAGMA user_version={AUTOTUNE_SCHEMA_VERSION}")
        connection.execute(
            "CREATE TABLE cache_state "
            "(id INTEGER PRIMARY KEY, generation INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO cache_state(id, generation) VALUES(1, 7)")
        connection.commit()
    finally:
        connection.close()

    repaired = cache._connect(isolated_cache)
    assert repaired is not None
    try:
        assert repaired.execute("SELECT COUNT(*) FROM entries").fetchone() == (0,)
        assert repaired.execute("SELECT COUNT(*) FROM device_leases").fetchone() == (0,)
        generation = repaired.execute(
            "SELECT generation FROM cache_state WHERE id = 1"
        ).fetchone()
        assert generation == (0,)
    finally:
        repaired.close()


def test_invalid_generation_value_recreates_disposable_schema(isolated_cache):
    connection = cache._connect(isolated_cache)
    assert connection is not None
    try:
        connection.execute("UPDATE cache_state SET generation = 'invalid'")
    finally:
        connection.close()
    cache._invalidate_initialized_path(isolated_cache)

    repaired = cache._connect(isolated_cache)
    assert repaired is not None
    try:
        generation = repaired.execute(
            "SELECT generation FROM cache_state WHERE id = 1"
        ).fetchone()
        assert generation == (0,)
    finally:
        repaired.close()


def test_same_version_wrong_conflict_target_recreates_disposable_schema(
    isolated_cache,
):
    isolated_cache.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(isolated_cache)
    try:
        connection.execute(f"PRAGMA user_version={AUTOTUNE_SCHEMA_VERSION}")
        connection.execute(
            "CREATE TABLE cache_state "
            "(id INTEGER PRIMARY KEY, generation INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO cache_state(id, generation) VALUES(1, 0)")
        connection.execute(
            """
            CREATE TABLE entries (
                cache_key TEXT,
                block_size INTEGER NOT NULL,
                num_warps INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL,
                last_access_ns INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE device_leases (
                device_key TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                expires_ns INTEGER NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    repaired = cache._connect(isolated_cache)
    assert repaired is not None
    try:
        repaired.execute(
            """
            INSERT INTO entries(
                cache_key, block_size, num_warps, updated_ns, last_access_ns
            )
            VALUES('winner', 8192, 8, 1, 1)
            ON CONFLICT(cache_key) DO UPDATE SET block_size = excluded.block_size
            """
        )
    finally:
        repaired.close()


def test_late_schema_recreate_preserves_an_existing_winner_and_lease(
    isolated_cache,
    monkeypatch,
):
    key, workload = indexed_identity(torch, 0)
    winner = LaunchConfig(8192, 8)
    _persist(isolated_cache, key, workload, winner)
    lease = cache.try_acquire_device_benchmark_lease(
        "cpu:None",
        owner_id="late-recreate-owner",
        path=isolated_cache,
    )
    assert lease is not None
    connection = cache._connect(isolated_cache)
    assert connection is not None
    original_is_current_valid = cache._schema_is_current_valid
    checks = 0

    def stale_then_current(current):
        nonlocal checks
        checks += 1
        if checks == 1:
            return False
        return original_is_current_valid(current)

    monkeypatch.setattr(cache, "_schema_is_current_valid", stale_then_current)
    cache._invalidate_initialized_path(isolated_cache)
    try:
        cache._initialize_schema(connection, isolated_cache)
        assert checks == 2
        assert connection.execute(
            "SELECT block_size, num_warps FROM entries WHERE cache_key = ?",
            (workload.disk_key(),),
        ).fetchone() == (winner.block_size, winner.num_warps)
        assert connection.execute(
            "SELECT owner_id, generation FROM device_leases WHERE device_key = ?",
            (lease.device_key,),
        ).fetchone() == (lease.owner_id, lease.generation)
    finally:
        connection.close()
        cache.release_device_benchmark_lease(lease)


def test_initialized_path_skips_repeated_schema_verification(
    isolated_cache,
    monkeypatch,
):
    connection = cache._connect(isolated_cache)
    assert connection is not None
    connection.close()
    cache._invalidate_initialized_path(isolated_cache)
    checks = []
    original_verify_schema = cache._verify_schema

    def count_verification(current):
        checks.append(current)
        return original_verify_schema(current)

    monkeypatch.setattr(cache, "_verify_schema", count_verification)
    first = cache._connect(isolated_cache)
    assert first is not None
    first.close()
    second = cache._connect(isolated_cache)
    assert second is not None
    second.close()

    assert len(checks) == 1


@pytest.mark.parametrize("replace", [False, True], ids=["deleted", "replaced"])
def test_externally_changed_store_reinitializes_schema(isolated_cache, replace):
    connection = cache._connect(isolated_cache)
    assert connection is not None
    connection.close()
    if replace:
        replacement = isolated_cache.with_name("replacement.sqlite3")
        sqlite3.connect(replacement).close()
        replacement.replace(isolated_cache)
    else:
        isolated_cache.unlink()

    recreated = cache._connect(isolated_cache)
    assert recreated is not None
    try:
        assert cache._schema_is_current_valid(recreated)
    finally:
        recreated.close()


def test_reused_file_identity_reinitializes_deleted_store(
    isolated_cache,
    monkeypatch,
):
    reused_identity = (1, 1)
    monkeypatch.setattr(
        cache,
        "_database_file_identity",
        lambda path: reused_identity if path.exists() else None,
    )
    cache._invalidate_initialized_path(isolated_cache)

    connection = cache._connect(isolated_cache)
    assert connection is not None
    connection.close()
    isolated_cache.unlink()

    recreated = cache._connect(isolated_cache)
    assert recreated is not None
    try:
        assert cache._schema_is_current_valid(recreated)
    finally:
        recreated.close()


@pytest.mark.parametrize(
    ("index_name", "replacement"),
    [
        (
            "entries_lru_idx",
            "CREATE INDEX entries_lru_idx ON entries(updated_ns)",
        ),
        ("device_leases_expiry_idx", None),
    ],
)
def test_required_index_contract_recreates_disposable_schema(
    isolated_cache,
    index_name,
    replacement,
):
    connection = cache._connect(isolated_cache)
    assert connection is not None
    try:
        assert (
            cache._required_index_columns(connection) == cache._REQUIRED_INDEX_COLUMNS
        )
        connection.execute(f"DROP INDEX {index_name}")
        if replacement is not None:
            connection.execute(replacement)
    finally:
        connection.close()
    cache._invalidate_initialized_path(isolated_cache)

    repaired = cache._connect(isolated_cache)
    assert repaired is not None
    try:
        assert cache._required_index_columns(repaired) == cache._REQUIRED_INDEX_COLUMNS
    finally:
        repaired.close()


# ── Generation fence and reset ───────────────────────────────────────────────
def test_reset_fences_stale_device_owner_without_opening_a_second_owner(
    isolated_cache,
):
    key, workload = indexed_identity(torch, 0)
    stale = cache.try_acquire_device_benchmark_lease(
        "cpu:None",
        owner_id="stale-owner",
        path=isolated_cache,
    )
    assert stale is not None
    try:
        assert cache.persist_winner(
            key,
            workload,
            LaunchConfig(8192, 8),
            lease=stale,
        )
        cache.reset_cache()

        assert cache.get_hot_winner(key) is None
        assert not cache.persist_winner(
            key,
            workload,
            LaunchConfig(16384, 16),
            lease=stale,
        )
        assert (
            cache.try_acquire_device_benchmark_lease(
                "cpu:None",
                owner_id="racing-owner",
                path=isolated_cache,
            )
            is None
        )
    finally:
        cache.release_device_benchmark_lease(stale)

    fresh = cache.try_acquire_device_benchmark_lease(
        "cpu:None",
        owner_id="fresh-owner",
        path=isolated_cache,
    )
    assert fresh is not None
    try:
        assert cache.persist_winner(
            key,
            workload,
            LaunchConfig(16384, 16),
            lease=fresh,
        )
    finally:
        cache.release_device_benchmark_lease(fresh)


def test_concurrent_resets_are_idempotent_and_increment_generation(isolated_cache):
    connection = cache._connect(isolated_cache)
    assert connection is not None
    connection.close()
    barrier = threading.Barrier(2)
    failures = []

    def resetter():
        try:
            barrier.wait()
            cache.reset_cache()
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    threads = [threading.Thread(target=resetter) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    connection = cache._connect(isolated_cache)
    assert connection is not None
    try:
        generation = connection.execute(
            "SELECT generation FROM cache_state WHERE id = 1"
        ).fetchone()
        assert generation == (2,)
        assert connection.execute("SELECT COUNT(*) FROM entries").fetchone() == (0,)
    finally:
        connection.close()


def test_reset_clears_every_cache_path_used_by_this_process(tmp_path, monkeypatch):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    monkeypatch.setenv("ORDA_AUTOTUNE_CACHE_DIR", str(first_dir))
    cache.reset_cache()
    first_path = cache.cache_path()
    first_key, first_workload = indexed_identity(torch, 0)
    _persist(first_path, first_key, first_workload, LaunchConfig(8192, 8))

    monkeypatch.setenv("ORDA_AUTOTUNE_CACHE_DIR", str(second_dir))
    second_path = cache.cache_path()
    second_key, second_workload = indexed_identity(torch, 1)
    _persist(second_path, second_key, second_workload, LaunchConfig(8192, 16))

    cache.reset_cache()
    assert _read_disk_winners(first_path) == {}
    assert _read_disk_winners(second_path) == {}


# ── Device serialization fallback ────────────────────────────────────────────
@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork is unavailable")
def test_forked_child_reinitializes_process_locks():
    read_fd, write_fd = os.pipe()
    cache._DEVICE_LOCK_GUARD.acquire()
    try:
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            acquired = cache._DEVICE_LOCK_GUARD.acquire(timeout=0.5)
            fresh_owner = cache._LEASE_OWNER_PREFIX.startswith(f"{os.getpid()}-")
            os.write(write_fd, b"ok" if acquired and fresh_owner else b"failed")
            os._exit(0)
    finally:
        cache._DEVICE_LOCK_GUARD.release()
        os.close(write_fd)
    try:
        assert os.read(read_fd, 16) == b"ok"
    finally:
        os.close(read_fd)
        os.waitpid(child_pid, 0)


def test_local_device_lock_serializes_benchmark_sections(isolated_cache):
    owner_entered = threading.Event()
    release_owner = threading.Event()
    waiter_started = threading.Event()
    waiter_entered = threading.Event()

    def owner():
        with cache.device_benchmark_lock(torch.device("cpu")):
            owner_entered.set()
            assert release_owner.wait(timeout=5)

    def waiter():
        assert owner_entered.wait(timeout=5)
        waiter_started.set()
        with cache.device_benchmark_lock(torch.device("cpu")):
            waiter_entered.set()

    owner_thread = threading.Thread(target=owner)
    waiter_thread = threading.Thread(target=waiter)
    owner_thread.start()
    assert owner_entered.wait(timeout=5)
    waiter_thread.start()
    assert waiter_started.wait(timeout=5)
    assert not waiter_entered.wait(timeout=0.05)
    release_owner.set()
    owner_thread.join()
    waiter_thread.join()
    assert waiter_entered.is_set()


def test_unavailable_sqlite_store_yields_a_local_only_device_lock(
    tmp_path,
    monkeypatch,
):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("ORDA_AUTOTUNE_CACHE_DIR", str(blocked_parent))

    with pytest.warns(RuntimeWarning, match="process-local autotuning"):
        with cache.device_benchmark_lock(torch.device("cpu")) as lease:
            assert lease is None


def test_device_lease_timeout_does_not_start_an_overlapping_benchmark(monkeypatch):
    clock = iter([0.0, 0.0, 0.25, 0.5, 0.75, 1.0])
    entered = False
    monkeypatch.setattr(cache, "_LEASE_WAIT_SECONDS", 1)
    monkeypatch.setattr(cache, "_LEASE_POLL_MS", 250)
    monkeypatch.setattr(
        cache,
        "try_acquire_device_benchmark_lease",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(cache.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(cache.time, "sleep", lambda _seconds: None)

    with pytest.warns(RuntimeWarning, match="already benchmarking"):
        with pytest.raises(cache._DeviceBenchmarkBusy):
            with cache.device_benchmark_lock(torch.device("cpu")):
                entered = True
    assert not entered


def test_device_lock_renews_lease_while_body_is_blocked(
    isolated_cache,
    monkeypatch,
):
    original_renew = cache.renew_device_benchmark_lease
    renewed_past_initial_expiry = threading.Event()
    renewed_leases = []
    monkeypatch.setattr(cache, "_LEASE_SECONDS", 0.08)

    def observe_renewal(lease):
        renewed_leases.append(lease)
        renewed = original_renew(lease)
        if len(renewed_leases) >= 5:
            renewed_past_initial_expiry.set()
        return renewed

    monkeypatch.setattr(cache, "renew_device_benchmark_lease", observe_renewal)

    with cache.device_benchmark_lock(torch.device("cpu")) as lease:
        assert lease is not None
        assert renewed_past_initial_expiry.wait(timeout=2)
        assert (
            cache.try_acquire_device_benchmark_lease(
                lease.device_key,
                owner_id="competing-owner",
                path=isolated_cache,
            )
            is None
        )

    assert renewed_leases
    assert all(renewed_lease == lease for renewed_lease in renewed_leases)
