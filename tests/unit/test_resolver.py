from __future__ import annotations

import pytest

from orda_ce_kernel.utils.resolver import resolve_chunk_size


def test_resolve_chunk_size_rejects_empty_batch():
    with pytest.raises(ValueError, match="BT"):
        resolve_chunk_size(0, None, V=128)


def test_resolve_chunk_size_explicit_value_is_clamped_to_batch():
    assert resolve_chunk_size(128, 512, V=32768) == (128, 1)
    assert resolve_chunk_size(128, 64, V=32768) == (64, 2)


def test_resolve_chunk_size_dynamic_returns_positive_partition():
    chunk_size, num_chunks = resolve_chunk_size(8192, "dynamic", V=32768)
    assert chunk_size > 0
    assert num_chunks > 0
    assert chunk_size * num_chunks >= 8192


@pytest.mark.parametrize("auto_value", [None, "dynamic", "auto", -2, 0, -1.0])
def test_resolve_chunk_size_auto_aliases_without_vocab_use_single_chunk(auto_value):
    assert resolve_chunk_size(128, auto_value, V=None) == (128, 1)


def test_resolve_chunk_size_dynamic_respects_max_chunks():
    _, num_chunks = resolve_chunk_size(32768, "dynamic", V=131072, max_chunks=2)
    assert num_chunks <= 2


def test_resolve_chunk_size_defaults_to_double_max_useful_chunks():
    # With BT=32768, max_useful_chunks = 64.
    # Max chunks defaults to 2 * 64 = 128.
    # num_chunks raw is 512.
    # final_chunks = min(128, 64, 512, 32768) = 64.
    _, num_chunks = resolve_chunk_size(32768, "dynamic", V=131072)
    assert num_chunks == 64


