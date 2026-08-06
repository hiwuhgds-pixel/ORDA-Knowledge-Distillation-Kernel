from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
    _LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.LockFileEx.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    )
    _KERNEL32.LockFileEx.restype = wintypes.BOOL
    _KERNEL32.UnlockFileEx.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    )
    _KERNEL32.UnlockFileEx.restype = wintypes.BOOL
else:
    import fcntl

import torch

from ._device import device_identity, reset_device_identity_cache
from ._env import _bounded_env_int
from .models import (
    AUTOTUNE_SCHEMA_VERSION,
    AutotuneRuntimeKey,
    AutotuneWorkload,
    LaunchConfig,
)

# ── Environment controls ─────────────────────────────────────────────────────
_MAX_HOT_ENTRIES = _bounded_env_int(
    "ORDA_AUTOTUNE_MEMORY_ENTRIES", 4096, minimum=1, maximum=65_536
)
_MAX_DISK_ENTRIES = _bounded_env_int(
    "ORDA_AUTOTUNE_DISK_ENTRIES", 4096, minimum=1, maximum=65_536
)
_LEASE_SECONDS = _bounded_env_int(
    "ORDA_AUTOTUNE_LEASE_SECONDS", 60, minimum=10, maximum=3_600
)
_LEASE_POLL_MS = _bounded_env_int(
    "ORDA_AUTOTUNE_LEASE_POLL_MS", 100, minimum=25, maximum=5_000
)
_LEASE_WAIT_SECONDS = _bounded_env_int(
    "ORDA_AUTOTUNE_LEASE_WAIT_SECONDS", 2, minimum=1, maximum=3_600
)
_SQLITE_TIMEOUT_SECONDS = _bounded_env_int(
    "ORDA_AUTOTUNE_SQLITE_TIMEOUT_SECONDS", 15, minimum=1, maximum=120
)
_SQLITE_INIT_RETRIES = _bounded_env_int(
    "ORDA_AUTOTUNE_SQLITE_INIT_RETRIES", 3, minimum=1, maximum=10
)
_SQLITE_INIT_RETRY_MS = _bounded_env_int(
    "ORDA_AUTOTUNE_SQLITE_INIT_RETRY_MS", 50, minimum=1, maximum=1_000
)
_RECOVERY_LOCK_TIMEOUT_SECONDS = float(_SQLITE_TIMEOUT_SECONDS)


# ── Cache state and paths ────────────────────────────────────────────────────
_CACHE_LOCK = threading.RLock()
_HOT_WINNERS: OrderedDict[AutotuneRuntimeKey, LaunchConfig] = OrderedDict()
_LOCAL_GENERATION = 0
_USED_CACHE_PATHS: set[Path] = set()
_WARNED_STORE_PATHS: set[Path] = set()
_WARNED_LEASE_KEYS: set[str] = set()

_DEVICE_LOCK_GUARD = threading.Lock()
_DEVICE_LOCKS: dict[tuple[str, int | None], threading.Lock] = {}
_SCHEMA_INIT_LOCK = threading.RLock()
_INITIALIZED_PATHS: dict[Path, tuple[int, int]] = {}
_RECOVERY_LOCK = threading.Lock()
_LEASE_OWNER_PREFIX = f"{os.getpid()}-{uuid.uuid4().hex}"
_REQUIRED_INDEX_COLUMNS: dict[str, tuple[str, ...]] = {
    "entries_lru_idx": ("last_access_ns", "cache_key"),
    "device_leases_expiry_idx": ("expires_ns",),
}


class _StoreUnavailable(RuntimeError):
    pass


class _DeviceBenchmarkBusy(RuntimeError):
    pass


class _CacheSchemaError(RuntimeError):
    pass


def _reset_process_locks_after_fork() -> None:
    """Replace locks and lease ownership inherited by a forked child."""
    global _CACHE_LOCK, _DEVICE_LOCK_GUARD, _DEVICE_LOCKS
    global _SCHEMA_INIT_LOCK, _RECOVERY_LOCK, _LEASE_OWNER_PREFIX

    _CACHE_LOCK = threading.RLock()
    _DEVICE_LOCK_GUARD = threading.Lock()
    _DEVICE_LOCKS = {}
    _SCHEMA_INIT_LOCK = threading.RLock()
    _RECOVERY_LOCK = threading.Lock()
    _LEASE_OWNER_PREFIX = f"{os.getpid()}-{uuid.uuid4().hex}"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_process_locks_after_fork)


def _now_ns() -> int:
    return time.time_ns()


def _invalidate_initialized_path(path: Path) -> None:
    """Force this process to verify a path again after destructive recovery."""
    with _SCHEMA_INIT_LOCK:
        _INITIALIZED_PATHS.pop(_canonical_cache_path(path), None)


def _cache_root() -> Path:
    override = os.environ.get("ORDA_AUTOTUNE_CACHE_DIR")
    if override:
        return Path(override)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "orda_kernel"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if os.name != "nt" and xdg_cache:
        return Path(xdg_cache) / "orda_kernel"
    return Path.home() / ".cache" / "orda_kernel"


def _canonical_cache_path(path: Path | str) -> Path:
    """Resolve one stable identity for path aliases and relative inputs."""
    return Path(path).resolve()


def _database_file_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino


def cache_path() -> Path:
    return _canonical_cache_path(
        _cache_root() / f"autotune_v{AUTOTUNE_SCHEMA_VERSION}.sqlite3"
    )


def _owned_database_paths(path: Path) -> tuple[Path, Path, Path]:
    return (
        path,
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
    )


def _recovery_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.recovery.lock")


# ── Store diagnostics ────────────────────────────────────────────────────────
def _warn_store_unavailable(path: Path, exc: BaseException) -> None:
    with _CACHE_LOCK:
        if path in _WARNED_STORE_PATHS:
            return
        _WARNED_STORE_PATHS.add(path)
    warnings.warn(
        f"ORDA autotune SQLite store is unavailable at {path}: {exc}; "
        "using process-local autotuning only.",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_device_benchmark_busy(device_key: str) -> None:
    with _CACHE_LOCK:
        if device_key in _WARNED_LEASE_KEYS:
            return
        _WARNED_LEASE_KEYS.add(device_key)
    warnings.warn(
        "ORDA autotune is already benchmarking this device in another process; "
        "using the latest cached or default config for this call.",
        RuntimeWarning,
        stacklevel=3,
    )


def _is_locked_store_error(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _is_recoverable_store_error(exc: BaseException) -> bool:
    if isinstance(exc, _CacheSchemaError):
        return True
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "malformed",
            "not a database",
            "file is encrypted",
        )
    )


# ── File locking ─────────────────────────────────────────────────────────────
class _StoreFileLock:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle.closed:
            return
        try:
            if sys.platform == "win32":
                os_handle = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
                _KERNEL32.UnlockFileEx(
                    os_handle,
                    0,
                    0xFFFFFFFF,
                    0xFFFFFFFF,
                    ctypes.byref(_Overlapped()),
                )
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            handle.close()

    def __del__(self) -> None:
        try:
            self.release()
        except OSError:
            pass


def _try_lock_store_file(handle: BinaryIO, *, exclusive: bool) -> bool:
    try:
        if sys.platform == "win32":
            flags = _LOCKFILE_FAIL_IMMEDIATELY
            if exclusive:
                flags |= _LOCKFILE_EXCLUSIVE_LOCK
            os_handle = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
            return bool(
                _KERNEL32.LockFileEx(
                    os_handle,
                    flags,
                    0,
                    0xFFFFFFFF,
                    0xFFFFFFFF,
                    ctypes.byref(_Overlapped()),
                )
            )
        else:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _acquire_store_file_lock(
    path: Path,
    *,
    exclusive: bool,
    timeout_seconds: float,
) -> _StoreFileLock | None:
    lock_path = _recovery_lock_path(path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b", buffering=0)
    except OSError:
        return None

    deadline = time.monotonic() + timeout_seconds
    while not _try_lock_store_file(handle, exclusive=exclusive):
        if time.monotonic() >= deadline:
            handle.close()
            return None
        time.sleep(_SQLITE_INIT_RETRY_MS / 1_000.0)
    return _StoreFileLock(handle)


@contextmanager
def _database_recovery_lock(path: Path):
    lock = _acquire_store_file_lock(
        path,
        exclusive=True,
        timeout_seconds=_RECOVERY_LOCK_TIMEOUT_SECONDS,
    )
    try:
        yield lock is not None
    finally:
        if lock is not None:
            lock.release()


class _CoordinatedConnection(sqlite3.Connection):
    _store_lock: _StoreFileLock | None = None

    def close(self) -> None:
        lock = self._store_lock
        self._store_lock = None
        try:
            super().close()
        finally:
            if lock is not None:
                lock.release()


def _rollback_quietly(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("ROLLBACK")
    except sqlite3.DatabaseError:
        pass


# ── Schema management ────────────────────────────────────────────────────────
def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE cache_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            generation INTEGER NOT NULL
        )
        """
    )
    connection.execute("INSERT INTO cache_state(id, generation) VALUES(1, 0)")
    connection.execute(
        """
        CREATE TABLE entries (
            cache_key TEXT PRIMARY KEY,
            block_size INTEGER NOT NULL,
            num_warps INTEGER NOT NULL,
            updated_ns INTEGER NOT NULL,
            last_access_ns INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX entries_lru_idx ON entries(last_access_ns, cache_key)"
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
    connection.execute(
        "CREATE INDEX device_leases_expiry_idx ON device_leases(expires_ns)"
    )
    connection.execute(f"PRAGMA user_version={AUTOTUNE_SCHEMA_VERSION}")


def _schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return 0 if row is None else int(row[0])


def _schema_is_current_valid(connection: sqlite3.Connection) -> bool:
    try:
        if _schema_version(connection) != AUTOTUNE_SCHEMA_VERSION:
            return False
        _verify_schema(connection)
    except (sqlite3.DatabaseError, _CacheSchemaError):
        return False
    return True


def _recreate_schema(connection: sqlite3.Connection, path: Path) -> bool:
    """Recreate only when the schema is still invalid under the write lock."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        # A competing process can repair the store while this caller waits for
        # BEGIN IMMEDIATE. Re-read inside the transaction before dropping it.
        if _schema_is_current_valid(connection):
            connection.execute("COMMIT")
            return False
        _invalidate_initialized_path(path)
        connection.execute("DROP TABLE IF EXISTS device_leases")
        connection.execute("DROP TABLE IF EXISTS entries")
        connection.execute("DROP TABLE IF EXISTS cache_state")
        _create_schema(connection)
        connection.execute("COMMIT")
        return True
    except BaseException:
        _rollback_quietly(connection)
        _invalidate_initialized_path(path)
        raise


def _required_index_columns(
    connection: sqlite3.Connection,
) -> dict[str, tuple[str, ...]]:
    """Read the two performance-critical index definitions, and nothing else."""
    return {
        index_name: tuple(
            str(row[2])
            for row in connection.execute(f"PRAGMA index_info({index_name})")
        )
        for index_name in _REQUIRED_INDEX_COLUMNS
    }


def _verify_required_indexes(connection: sqlite3.Connection) -> None:
    actual = _required_index_columns(connection)
    if actual != _REQUIRED_INDEX_COLUMNS:
        raise _CacheSchemaError(
            "autotune cache required indexes do not match the current schema"
        )


def _verify_schema(connection: sqlite3.Connection) -> None:
    _read_generation(connection)
    connection.execute(
        "SELECT cache_key, block_size, num_warps FROM entries LIMIT 1"
    ).fetchone()
    connection.execute(
        "SELECT device_key, owner_id, generation FROM device_leases LIMIT 1"
    ).fetchone()
    _verify_required_indexes(connection)
    # These no-op query plans validate every column and ON CONFLICT target the
    # store relies on without reintroducing per-column/index introspection.
    connection.execute(
        """
        EXPLAIN INSERT INTO cache_state(id, generation) VALUES(1, 0)
        ON CONFLICT(id) DO NOTHING
        """
    ).fetchone()
    connection.execute(
        """
        EXPLAIN INSERT INTO entries(
            cache_key, block_size, num_warps, updated_ns, last_access_ns
        )
        VALUES('verify', 1, 1, 1, 1)
        ON CONFLICT(cache_key) DO UPDATE SET
            block_size = excluded.block_size,
            num_warps = excluded.num_warps,
            updated_ns = excluded.updated_ns,
            last_access_ns = excluded.last_access_ns
        """
    ).fetchone()
    connection.execute(
        """
        EXPLAIN INSERT INTO device_leases(device_key, owner_id, generation, expires_ns)
        VALUES('verify', 'verify', 0, 1)
        ON CONFLICT(device_key) DO NOTHING
        """
    ).fetchone()


def _initialize_schema(connection: sqlite3.Connection, path: Path) -> None:
    with _SCHEMA_INIT_LOCK:
        normalized_path = _canonical_cache_path(path)
        identity_before = _database_file_identity(normalized_path)
        if (
            identity_before is not None
            and _INITIALIZED_PATHS.get(normalized_path) == identity_before
        ):
            # POSIX filesystems may immediately reuse an inode after unlink.
            # Reject a new empty database even when its file identity collides.
            if _schema_version(connection) == AUTOTUNE_SCHEMA_VERSION:
                return
            _INITIALIZED_PATHS.pop(normalized_path, None)
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            if not _schema_is_current_valid(connection):
                _recreate_schema(connection, normalized_path)
        except BaseException:
            _INITIALIZED_PATHS.pop(normalized_path, None)
            raise
        identity_after = _database_file_identity(normalized_path)
        if identity_before is not None and identity_after == identity_before:
            _INITIALIZED_PATHS[normalized_path] = identity_before
        else:
            _INITIALIZED_PATHS.pop(normalized_path, None)


# ── Database recovery ────────────────────────────────────────────────────────
def _revalidate_recovery_candidate(path: Path) -> bool | None:
    """Return whether the current database is usable without deleting it.

    None represents a transient filesystem or SQLite lock failure. It must not
    be treated as corruption because another process may still own the store.
    """
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            path,
            timeout=float(_SQLITE_TIMEOUT_SECONDS),
            isolation_level=None,
        )
        connection.execute(f"PRAGMA busy_timeout={_SQLITE_TIMEOUT_SECONDS * 1000}")
        _initialize_schema(connection, path)
        connection.execute("PRAGMA synchronous=NORMAL")
        return True
    except OSError:
        return None
    except (sqlite3.DatabaseError, _CacheSchemaError) as exc:
        return False if _is_recoverable_store_error(exc) else None
    finally:
        if connection is not None:
            connection.close()


def _revalidate_recovery_candidate_with_retries(path: Path) -> bool | None:
    """Absorb transient states while another process closes or rebuilds SQLite."""
    result: bool | None = None
    for attempt in range(_SQLITE_INIT_RETRIES):
        result = _revalidate_recovery_candidate(path)
        if result is not None:
            return result
        if attempt + 1 < _SQLITE_INIT_RETRIES:
            time.sleep(_SQLITE_INIT_RETRY_MS * (attempt + 1) / 1_000.0)
    return result


def _recover_database(path: Path) -> bool:
    """Best-effort recovery for a corrupt owned SQLite database."""
    _invalidate_initialized_path(path)
    with _RECOVERY_LOCK:
        with _database_recovery_lock(path) as acquired:
            if not acquired:
                if _revalidate_recovery_candidate_with_retries(path) is True:
                    return True
                _warn_store_unavailable(
                    path,
                    _StoreUnavailable(
                        "timed out acquiring the per-database recovery lock"
                    ),
                )
                return False

            current_database = _revalidate_recovery_candidate_with_retries(path)
            if current_database is True:
                return True
            if current_database is None:
                return False

            try:
                for owned_path in _owned_database_paths(path):
                    owned_path.unlink(missing_ok=True)
            except OSError as exc:
                _warn_store_unavailable(path, exc)
                return False
    return True


# ── Cache connections ────────────────────────────────────────────────────────
def _connect(
    path: Path | None = None,
    *,
    recover: bool = True,
) -> sqlite3.Connection | None:
    # ── Connection setup and recovery ────────────────────────
    resolved_path = cache_path() if path is None else _canonical_cache_path(path)
    with _CACHE_LOCK:
        _USED_CACHE_PATHS.add(resolved_path)
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _invalidate_initialized_path(resolved_path)
        _warn_store_unavailable(resolved_path, exc)
        return None

    attempt = 0
    while True:
        store_lock = _acquire_store_file_lock(
            resolved_path,
            exclusive=False,
            timeout_seconds=float(_SQLITE_TIMEOUT_SECONDS),
        )
        if store_lock is None:
            _warn_store_unavailable(
                resolved_path,
                _StoreUnavailable("timed out waiting for cache recovery"),
            )
            return None
        connection: _CoordinatedConnection | None = None
        try:
            connection = cast(
                _CoordinatedConnection,
                sqlite3.connect(
                    resolved_path,
                    timeout=float(_SQLITE_TIMEOUT_SECONDS),
                    isolation_level=None,
                    factory=_CoordinatedConnection,
                ),
            )
            connection._store_lock = store_lock
            connection.execute(f"PRAGMA busy_timeout={_SQLITE_TIMEOUT_SECONDS * 1000}")
            _initialize_schema(connection, resolved_path)
            connection.execute("PRAGMA synchronous=NORMAL")
            return connection
        except (OSError, sqlite3.DatabaseError, _CacheSchemaError) as exc:
            if connection is not None:
                connection.close()
            else:
                store_lock.release()
            _invalidate_initialized_path(resolved_path)
            if _is_locked_store_error(exc) and attempt + 1 < _SQLITE_INIT_RETRIES:
                attempt += 1
                time.sleep(_SQLITE_INIT_RETRY_MS * attempt / 1_000.0)
                continue
            if (
                recover
                and _is_recoverable_store_error(exc)
                and _recover_database(resolved_path)
            ):
                return _connect(resolved_path, recover=False)
            _warn_store_unavailable(resolved_path, exc)
            return None


def _read_generation(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT generation FROM cache_state WHERE id = 1"
    ).fetchone()
    if row is None:
        raise _CacheSchemaError("autotune cache has no generation row")
    try:
        return int(row[0])
    except (TypeError, ValueError) as exc:
        raise _CacheSchemaError(
            f"autotune cache generation is not an integer: {row[0]!r}"
        ) from exc


# ── Compact process-local hot cache ──────────────────────────────────────────
def _remember_hot_locked(key: AutotuneRuntimeKey, config: LaunchConfig) -> None:
    _HOT_WINNERS[key] = config
    _HOT_WINNERS.move_to_end(key)
    while len(_HOT_WINNERS) > _MAX_HOT_ENTRIES:
        _HOT_WINNERS.popitem(last=False)


def get_hot_winner(key: AutotuneRuntimeKey) -> LaunchConfig | None:
    """Return a process-local winner without steady-state disk I/O.

    Peer resets cannot evict this RAM entry; generation fencing still protects
    benchmark writes, so any stale entry remains a valid config for this key.
    """
    with _CACHE_LOCK:
        config = _HOT_WINNERS.get(key)
        if config is not None:
            _HOT_WINNERS.move_to_end(key)
        return config


def invalidate_hot_winner(key: AutotuneRuntimeKey) -> None:
    with _CACHE_LOCK:
        _HOT_WINNERS.pop(key, None)


def local_generation() -> int:
    with _CACHE_LOCK:
        return _LOCAL_GENERATION


def remember_local_winner(
    key: AutotuneRuntimeKey,
    config: LaunchConfig,
    *,
    generation: int,
) -> bool:
    """Remember a local-only result unless reset raced this selection."""
    with _CACHE_LOCK:
        if generation != _LOCAL_GENERATION:
            return False
        _remember_hot_locked(key, config)
        return True


# ── Winner lookup and persistence ────────────────────────────────────────────
def _upsert_entry(
    connection: sqlite3.Connection,
    cache_key: str,
    config: LaunchConfig,
    now: int,
) -> None:
    connection.execute(
        """
        INSERT INTO entries(
            cache_key, block_size, num_warps, updated_ns, last_access_ns
        )
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            block_size = excluded.block_size,
            num_warps = excluded.num_warps,
            updated_ns = excluded.updated_ns,
            last_access_ns = excluded.last_access_ns
        """,
        (cache_key, config.block_size, config.num_warps, now, now),
    )
    overflow = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    overflow -= _MAX_DISK_ENTRIES
    if overflow > 0:
        connection.execute(
            """
            DELETE FROM entries
            WHERE cache_key IN (
                SELECT cache_key FROM entries
                ORDER BY last_access_ns ASC, cache_key ASC
                LIMIT ?
            )
            """,
            (overflow,),
        )


def read_disk_winner(
    key: AutotuneRuntimeKey,
    workload: AutotuneWorkload,
    *,
    valid_configs: set[LaunchConfig],
    fallback: LaunchConfig,
) -> LaunchConfig | None:
    """Load a durable winner and promote it to the sole process-local LRU."""
    # ── Winner lookup ────────────────────────────────────────
    with _CACHE_LOCK:
        hot = _HOT_WINNERS.get(key)
        if hot is not None:
            _HOT_WINNERS.move_to_end(key)
            return hot
        generation = _LOCAL_GENERATION

    path = cache_path()
    connection = _connect(path)
    if connection is None:
        return None
    try:
        cache_key = workload.disk_key()
        row = connection.execute(
            "SELECT block_size, num_warps FROM entries WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        config = LaunchConfig(int(row[0]), int(row[1]))
        if config not in valid_configs and config != fallback:
            return None
        connection.execute(
            "UPDATE entries SET last_access_ns = ? WHERE cache_key = ?",
            (_now_ns(), cache_key),
        )
        return (
            config
            if remember_local_winner(key, config, generation=generation)
            else None
        )
    except (ValueError, sqlite3.DatabaseError) as exc:
        _warn_store_unavailable(path, exc)
        return None
    finally:
        connection.close()


def persist_winner(
    key: AutotuneRuntimeKey,
    workload: AutotuneWorkload,
    config: LaunchConfig,
    *,
    lease: DeviceBenchmarkLease,
) -> bool:
    """Commit only while the active device lease still owns this generation."""
    local_generation_snapshot = local_generation()
    connection = _connect(lease.path)
    if connection is None:
        raise _StoreUnavailable(f"ORDA autotune store is unavailable at {lease.path}")
    try:
        now = _now_ns()
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT owner_id, generation, expires_ns FROM device_leases
            WHERE device_key = ?
            """,
            (lease.device_key,),
        ).fetchone()
        if (
            _read_generation(connection) != lease.generation
            or row is None
            or str(row[0]) != lease.owner_id
            or int(row[1]) != lease.generation
            or int(row[2]) <= now
        ):
            connection.execute("COMMIT")
            return False
        _upsert_entry(connection, workload.disk_key(), config, now)
        connection.execute("COMMIT")
        return remember_local_winner(key, config, generation=local_generation_snapshot)
    except (sqlite3.DatabaseError, _CacheSchemaError) as exc:
        _rollback_quietly(connection)
        _warn_store_unavailable(lease.path, exc)
        raise _StoreUnavailable("ORDA autotune winner transaction failed") from exc
    finally:
        connection.close()


# ── Device benchmark leases ──────────────────────────────────────────────────
@dataclass(frozen=True)
class DeviceBenchmarkLease:
    device_key: str
    owner_id: str
    path: Path
    generation: int


def _device_lock_key(device: torch.device) -> tuple[tuple[str, int | None], str]:
    normalized = torch.device(device)
    index = normalized.index
    if normalized.type == "cuda":
        index = torch.cuda.current_device() if index is None else int(index)
        return (
            (normalized.type, index),
            device_identity(index).physical_key,
        )
    return (normalized.type, index), f"{normalized.type}:{index}"


def try_acquire_device_benchmark_lease(
    device_key: str,
    *,
    owner_id: str | None = None,
    path: Path | None = None,
) -> DeviceBenchmarkLease | None:
    resolved_path = cache_path() if path is None else _canonical_cache_path(path)
    connection = _connect(resolved_path)
    if connection is None:
        raise _StoreUnavailable(
            f"ORDA autotune device lease store is unavailable at {resolved_path}"
        )
    candidate_owner = owner_id or f"{_LEASE_OWNER_PREFIX}-{uuid.uuid4().hex}"
    try:
        now = _now_ns()
        connection.execute("BEGIN IMMEDIATE")
        generation = _read_generation(connection)
        connection.execute("DELETE FROM device_leases WHERE expires_ns <= ?", (now,))
        row = connection.execute(
            "SELECT owner_id FROM device_leases WHERE device_key = ?",
            (device_key,),
        ).fetchone()
        if row is not None:
            connection.execute("COMMIT")
            return None
        connection.execute(
            """
            INSERT INTO device_leases(device_key, owner_id, generation, expires_ns)
            VALUES(?, ?, ?, ?)
            """,
            (
                device_key,
                candidate_owner,
                generation,
                now + _LEASE_SECONDS * 1_000_000_000,
            ),
        )
        connection.execute("COMMIT")
        return DeviceBenchmarkLease(
            device_key=device_key,
            owner_id=candidate_owner,
            path=resolved_path,
            generation=generation,
        )
    except (sqlite3.DatabaseError, _CacheSchemaError) as exc:
        _rollback_quietly(connection)
        _warn_store_unavailable(resolved_path, exc)
        raise _StoreUnavailable(
            "ORDA autotune device lease transaction failed"
        ) from exc
    finally:
        connection.close()


def renew_device_benchmark_lease(lease: DeviceBenchmarkLease) -> bool:
    connection = _connect(lease.path)
    if connection is None:
        raise _StoreUnavailable(
            f"ORDA autotune device lease store is unavailable at {lease.path}"
        )
    try:
        now = _now_ns()
        connection.execute("BEGIN IMMEDIATE")
        if _read_generation(connection) != lease.generation:
            connection.execute("COMMIT")
            return False
        cursor = connection.execute(
            """
            UPDATE device_leases SET expires_ns = ?
            WHERE device_key = ? AND owner_id = ? AND generation = ? AND expires_ns > ?
            """,
            (
                now + _LEASE_SECONDS * 1_000_000_000,
                lease.device_key,
                lease.owner_id,
                lease.generation,
                now,
            ),
        )
        connection.execute("COMMIT")
        return cursor.rowcount == 1
    except (sqlite3.DatabaseError, _CacheSchemaError) as exc:
        _rollback_quietly(connection)
        _warn_store_unavailable(lease.path, exc)
        raise _StoreUnavailable("ORDA autotune device lease renewal failed") from exc
    finally:
        connection.close()


def release_device_benchmark_lease(lease: DeviceBenchmarkLease) -> None:
    connection = _connect(lease.path)
    if connection is None:
        return
    try:
        connection.execute(
            """
            DELETE FROM device_leases
            WHERE device_key = ? AND owner_id = ? AND generation = ?
            """,
            (lease.device_key, lease.owner_id, lease.generation),
        )
    except sqlite3.DatabaseError as exc:
        _warn_store_unavailable(lease.path, exc)
    finally:
        connection.close()


def _renew_device_benchmark_lease_until_stopped(
    lease: DeviceBenchmarkLease,
    stop: threading.Event,
) -> None:
    interval_seconds = float(_LEASE_SECONDS) / 4.0
    while not stop.wait(interval_seconds):
        try:
            if not renew_device_benchmark_lease(lease):
                return
        except _StoreUnavailable:
            # A transient store failure may clear before the current lease
            # expires, so keep trying while the local benchmark lock is held.
            continue


@contextmanager
def device_benchmark_lock(device: torch.device):
    """Serialize local and cross-process timing for one physical device."""
    local_key, physical_key = _device_lock_key(torch.device(device))
    with _DEVICE_LOCK_GUARD:
        lock = _DEVICE_LOCKS.setdefault(local_key, threading.Lock())
    with lock:
        lease: DeviceBenchmarkLease | None = None
        deadline = time.monotonic() + _LEASE_WAIT_SECONDS
        while lease is None:
            try:
                lease = try_acquire_device_benchmark_lease(physical_key)
            except _StoreUnavailable:
                # SQLite is optional. The local lock still prevents local overlap.
                yield None
                return
            if lease is None:
                if time.monotonic() >= deadline:
                    _warn_device_benchmark_busy(physical_key)
                    raise _DeviceBenchmarkBusy(physical_key)
                time.sleep(_LEASE_POLL_MS / 1_000.0)
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_renew_device_benchmark_lease_until_stopped,
            args=(lease, heartbeat_stop),
            name="orda-autotune-device-lease",
            daemon=True,
        )
        heartbeat_started = False
        try:
            heartbeat_thread.start()
            heartbeat_started = True
            yield lease
        finally:
            heartbeat_stop.set()
            if heartbeat_started:
                heartbeat_thread.join()
            release_device_benchmark_lease(lease)


# ── Cache reset ──────────────────────────────────────────────────────────────
def _reset_database(path: Path) -> None:
    if not path.exists():
        return
    connection = _connect(path, recover=False)
    if connection is None:
        _warn_store_unavailable(
            path, _StoreUnavailable("could not open store to reset winners")
        )
        return
    try:
        connection.execute("BEGIN IMMEDIATE")
        generation = _read_generation(connection)
        connection.execute(
            "UPDATE cache_state SET generation = ? WHERE id = 1",
            (generation + 1,),
        )
        connection.execute("DELETE FROM entries")
        # Keep live leases until release or expiry.
        connection.execute(
            "DELETE FROM device_leases WHERE expires_ns <= ?",
            (_now_ns(),),
        )
        connection.execute("COMMIT")
    except (sqlite3.DatabaseError, _CacheSchemaError) as exc:
        _rollback_quietly(connection)
        _invalidate_initialized_path(path)
        _warn_store_unavailable(path, exc)
    finally:
        connection.close()


def reset_cache() -> None:
    """Fence active device owners and clear durable plus local winners."""
    global _LOCAL_GENERATION
    with _CACHE_LOCK:
        paths = set(_USED_CACHE_PATHS)
        paths.add(cache_path())
    try:
        for path in paths:
            _reset_database(path)
    finally:
        # Publish the new generation only after durable winners are gone, then
        # evict any stale winner a concurrent reader loaded during the reset.
        with _CACHE_LOCK:
            _USED_CACHE_PATHS.clear()
            _LOCAL_GENERATION += 1
            _HOT_WINNERS.clear()
            _WARNED_STORE_PATHS.clear()
            _WARNED_LEASE_KEYS.clear()
    reset_device_identity_cache()
