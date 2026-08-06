from __future__ import annotations

import pytest

from orda_kernel import ChunkSize
from orda_kernel._execution.chunking import resolve_chunk_size


# ── Chunk strategy cases ─────────────────────────────────────────────────────
@pytest.mark.parametrize("bt", [0, -1])
def test_resolve_chunk_size_rejects_nonpositive_batch(bt):
    with pytest.raises(ValueError, match="BT"):
        resolve_chunk_size(bt, ChunkSize.dynamic(), V=128)


def test_resolve_chunk_size_explicit_value_is_clamped_to_batch():
    assert resolve_chunk_size(128, ChunkSize.nrows(512), V=32768) == (128, 1)
    assert resolve_chunk_size(128, ChunkSize.nrows(64), V=32768) == (64, 2)


def test_resolve_chunk_size_explicit_num_chunks_rounds_up():
    dynamic = ChunkSize.dynamic()
    assert resolve_chunk_size(10, dynamic, V=32768, num_chunks=3) == (4, 3)
    # ceil(BT / requested_chunks) can leave fewer non-empty slices than the
    # request; the returned count must match the loop's actual slices.
    assert resolve_chunk_size(5, dynamic, V=32768, num_chunks=4) == (2, 3)
    assert resolve_chunk_size(10, dynamic, V=32768, num_chunks=20) == (1, 10)


def test_resolve_chunk_size_rejects_ambiguous_chunk_controls():
    import pytest

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_chunk_size(10, ChunkSize.nrows(4), V=32768, num_chunks=3)


def test_explicit_chunk_requests_override_max_chunks():
    assert resolve_chunk_size(
        8,
        ChunkSize.dynamic(),
        V=32768,
        num_chunks=8,
        max_chunks=1,
    ) == (1, 8)
    assert resolve_chunk_size(10, ChunkSize.nrows(1), V=32768, max_chunks=1) == (1, 10)


def test_resolve_chunk_size_dynamic_returns_positive_partition():
    chunk_size, num_chunks = resolve_chunk_size(
        8192,
        ChunkSize.dynamic(),
        V=32768,
    )
    assert chunk_size > 0
    assert num_chunks > 0
    assert chunk_size * num_chunks >= 8192


# ── Dynamic and fast cases ───────────────────────────────────────────────────
def test_resolve_chunk_size_dynamic_without_vocab_uses_single_chunk():
    assert resolve_chunk_size(128, ChunkSize.dynamic(), V=None) == (128, 1)


@pytest.mark.parametrize("raw_value", [None, "auto", "dynamic", -2, -1, 0, 512])
def test_resolve_chunk_size_rejects_raw_legacy_values(raw_value):
    with pytest.raises(TypeError, match="ChunkSize"):
        resolve_chunk_size(128, raw_value, V=None)  # type: ignore[arg-type]


def test_resolve_chunk_size_dynamic_respects_max_chunks():
    _, num_chunks = resolve_chunk_size(
        32768,
        ChunkSize.dynamic(),
        V=131072,
        max_chunks=2,
    )
    assert num_chunks <= 2


def test_resolve_chunk_size_defaults_to_max_32_chunks():
    _, num_chunks = resolve_chunk_size(32768, ChunkSize.dynamic(), V=131072)
    assert num_chunks == 32


@pytest.mark.parametrize(
    ("bt", "expected"),
    [
        (1, (1, 1)),
        (512, (256, 2)),
        (1500, (750, 2)),
        (2048, (1024, 2)),
        (8192, (1024, 8)),
        (16384, (1024, 16)),
        (65536, (4096, 16)),
    ],
)
def test_resolve_chunk_size_fast_partition(bt, expected):
    assert resolve_chunk_size(bt, ChunkSize.fast(), V=32768) == expected


def test_resolve_chunk_size_fast_ignores_vocab_size():
    assert resolve_chunk_size(8192, ChunkSize.fast(), V=None) == (1024, 8)
    assert resolve_chunk_size(8192, ChunkSize.fast(), V=262144) == (1024, 8)


def test_resolve_chunk_size_fast_explicit_max_chunks_overrides_ceiling():
    assert resolve_chunk_size(
        65536,
        ChunkSize.fast(),
        V=32768,
        max_chunks=8,
    ) == (8192, 8)
    assert resolve_chunk_size(
        65536,
        ChunkSize.fast(),
        V=32768,
        max_chunks=32,
    ) == (2048, 32)


def test_resolve_chunk_size_num_chunks_overrides_fast():
    assert resolve_chunk_size(8192, ChunkSize.fast(), V=32768, num_chunks=4) == (
        2048,
        4,
    )
    assert resolve_chunk_size(65536, ChunkSize.fast(), V=32768, num_chunks=32) == (
        2048,
        32,
    )


# ── ChunkSize contracts ──────────────────────────────────────────────────────
def test_resolve_chunk_size_nrows_still_rejects_num_chunks():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_chunk_size(8192, ChunkSize.nrows(1024), V=32768, num_chunks=4)


def test_chunk_size_fast_repr_and_value():
    fast = ChunkSize.fast()
    assert fast.is_fast is True
    assert fast.is_dynamic is False
    assert repr(fast) == "ChunkSize.fast()"
    assert fast == ChunkSize.fast()
    with pytest.raises(TypeError, match="fast ChunkSize"):
        _ = fast.value


def test_chunk_size_rejects_unknown_string_sentinel():
    with pytest.raises(ValueError, match="ChunkSize"):
        ChunkSize(_nrows="bogus")  # type: ignore[arg-type]
