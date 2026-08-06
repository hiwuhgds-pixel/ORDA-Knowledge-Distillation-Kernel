"""Shared deterministic sharding helpers for direct GPU suites."""

from __future__ import annotations

import hashlib

# ── Direct-GPU suite identity ────────────────────────────────────────────────
DIRECT_GPU_TESTS = {
    "dlogits": "test/correctness/test_dlogits_contract_cuda.py",
    "kl-variants": "test/correctness/test_kl_variants_cuda.py",
    "low-level": "test/correctness/test_low_level_ops_cuda.py",
    "public-api": "test/correctness/test_public_api_cuda.py",
}
DIRECT_GPU_TEST_PATHS = tuple(DIRECT_GPU_TESTS.values())


def is_direct_gpu_nodeid(nodeid: str) -> bool:
    return any(
        nodeid == path or nodeid.startswith(f"{path}::")
        for path in DIRECT_GPU_TEST_PATHS
    )


def direct_gpu_shard_index(nodeid: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count
