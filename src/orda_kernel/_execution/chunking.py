from __future__ import annotations

from numbers import Integral

from .._api_types import ChunkSize
from .._constants import DEFAULT_MAX_FUSED_SIZE as DEFAULT_MAX_FUSED_SIZE


# ── Chunk request validation ─────────────────────────────────────────────────
def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _normalize_positive_integer(
    name: str,
    value: int | None,
    *,
    allow_none: bool,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return normalized


def normalize_chunk_request(
    chunk_size: ChunkSize,
    num_chunks: int | None = None,
    max_chunks: int | None = None,
) -> tuple[ChunkSize, int | None, int | None]:
    if not isinstance(chunk_size, ChunkSize):
        raise TypeError(
            "chunk_size must be created with ChunkSize.nrows(...), "
            f"ChunkSize.fast(), or ChunkSize.dynamic(), got {chunk_size!r}"
        )
    normalized_num_chunks = _normalize_positive_integer(
        "num_chunks",
        num_chunks,
        allow_none=True,
    )
    normalized_max_chunks = _normalize_positive_integer(
        "max_chunks",
        max_chunks,
        allow_none=True,
    )
    if normalized_num_chunks is not None and not (
        chunk_size.is_dynamic or chunk_size.is_fast
    ):
        raise ValueError("chunk_size and num_chunks are mutually exclusive")
    return chunk_size, normalized_num_chunks, normalized_max_chunks


def normalize_max_fused_size(max_fused_size: int) -> int:
    normalized = _normalize_positive_integer(
        "max_fused_size",
        max_fused_size,
        allow_none=False,
    )
    if normalized is None:
        raise AssertionError("max_fused_size normalization returned None")
    if not is_power_of_two(normalized):
        raise ValueError(
            f"max_fused_size must be a power of two, got {max_fused_size!r}"
        )
    if normalized > DEFAULT_MAX_FUSED_SIZE:
        raise ValueError(
            "max_fused_size must be <= "
            f"{DEFAULT_MAX_FUSED_SIZE}, got {max_fused_size!r}"
        )
    return normalized


def _max_useful_chunks(bt: int) -> int:
    return max(1, int(bt) // 512)


def _default_max_chunks(bt: int, max_chunks: int | None) -> int:
    """Resolve the dynamic-chunk ceiling without overriding explicit limits."""
    if max_chunks is not None:
        return max_chunks
    return min(32, 2 * _max_useful_chunks(bt))


# ── Fast plan ────────────────────────────────────────────────────────────────
# Throughput-oriented: a small, vocabulary-independent partition. Past
# FAST_MAX_CHUNKS * FAST_CHUNK_ROWS rows the chunks widen instead of
# multiplying, trading memory for fewer launches.
FAST_CHUNK_ROWS = 1024
FAST_MIN_CHUNKS = 2
FAST_MAX_CHUNKS = 16


def fast_max_chunks(max_chunks: int | None) -> int:
    """Resolve the fast-plan ceiling without overriding an explicit limit."""
    if max_chunks is not None:
        return max_chunks
    return FAST_MAX_CHUNKS


# ── Chunk sizing ─────────────────────────────────────────────────────────────
def _chunks_from_raw(raw: float) -> int:
    if raw < 1.5:
        return 1
    import math

    return 1 << (math.floor(math.log2(raw / 1.5)) + 1)


def _num_nonempty_chunks(bt: int, chunk_size: int) -> int:
    """Return the number of non-empty slices produced by ``chunk_size``."""
    return (bt + chunk_size - 1) // chunk_size


def resolve_chunk_size(
    BT: int,
    chunk_size_arg: ChunkSize,
    V: int | None = None,
    max_chunks: int | None = None,
    num_chunks: int | None = None,
) -> tuple[int, int]:
    """Compute chunk_size and num_chunks from either chunk_size or num_chunks."""
    bt = int(BT)
    if bt <= 0:
        raise ValueError("BT (batch x sequence length) must be > 0.")

    chunk_size_arg, num_chunks, max_chunks = normalize_chunk_request(
        chunk_size_arg,
        num_chunks,
        max_chunks,
    )

    if num_chunks is not None:
        effective_chunks = min(num_chunks, bt)
        chunk_size = (bt + effective_chunks - 1) // effective_chunks
        realized_chunks = _num_nonempty_chunks(bt, chunk_size)
        return chunk_size, realized_chunks

    if chunk_size_arg.is_fast:
        fast_chunks = min(
            max(FAST_MIN_CHUNKS, _num_nonempty_chunks(bt, FAST_CHUNK_ROWS)),
            fast_max_chunks(max_chunks),
            bt,
        )
        chunk_size = (bt + fast_chunks - 1) // fast_chunks
        return chunk_size, _num_nonempty_chunks(bt, chunk_size)

    if chunk_size_arg.is_dynamic:
        if V is None:
            return bt, 1

        max_useful_chunks = _max_useful_chunks(bt)
        max_chunks = _default_max_chunks(bt, max_chunks)

        raw_pressure = (float(bt) / 1024.0) * (float(V) / 32768.0)
        raw_bt_floor = float(bt) / 4096.0
        num_chunks = _chunks_from_raw(max(raw_pressure, raw_bt_floor))

        num_chunks = min(max_chunks, max_useful_chunks, num_chunks, bt)
        chunk_size = (bt + num_chunks - 1) // num_chunks
        return chunk_size, _num_nonempty_chunks(bt, chunk_size)

    chunk_size = min(chunk_size_arg.value, bt)
    return chunk_size, _num_nonempty_chunks(bt, chunk_size)
