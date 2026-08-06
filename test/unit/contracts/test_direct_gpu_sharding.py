"""Static integrity checks for the direct-GPU CI shard partition."""

from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import conftest
from utils.sharding import (
    DIRECT_GPU_TEST_PATHS,
    direct_gpu_shard_index,
    is_direct_gpu_nodeid,
)


def test_direct_gpu_nodeid_uses_exact_path_boundary():
    path = DIRECT_GPU_TEST_PATHS[0]
    assert is_direct_gpu_nodeid(path)
    assert is_direct_gpu_nodeid(f"{path}::test_contract[0]")
    assert not is_direct_gpu_nodeid(f"{path}.bak::test_contract[0]")
    assert not is_direct_gpu_nodeid(
        "test/correctness/test_low_level_ops_cuda.pyx::test_contract[0]"
    )


# ── Eight-way SHA-256 partition ──────────────────────────────────────────────
def test_all_eight_direct_gpu_shards_are_complete_and_disjoint():
    nodeids = {
        f"{path}::test_contract[{ordinal}]"
        for path in DIRECT_GPU_TEST_PATHS
        for ordinal in range(1024)
    }
    shards = [
        {
            nodeid
            for nodeid in nodeids
            if direct_gpu_shard_index(nodeid, shard_count=8) == shard_index
        }
        for shard_index in range(8)
    ]

    assert all(shard for shard in shards)
    assert set().union(*shards) == nodeids
    assert sum(len(shard) for shard in shards) == len(nodeids)
    assert all(left.isdisjoint(right) for left, right in combinations(shards, 2))


def test_collection_hook_partitions_actual_nodeids_and_preserves_nonmatching_paths(
    monkeypatch,
):
    actual_nodeids = [
        f"{path}::test_public_contract[{ordinal}]"
        for path in DIRECT_GPU_TEST_PATHS
        for ordinal in range(32)
    ]
    nonmatching_nodeids = (
        "test/correctness/test_cartesian_cuda.py"
        "::test_full_cartesian_group[tied-forward-float32-high_level]",
        "test/correctness/test_public_api_cuda.py.bak::test_not_a_test[0]",
    )
    all_nodeids = tuple(actual_nodeids) + nonmatching_nodeids
    selected_by_shard = []

    for shard_index in range(8):
        monkeypatch.setenv("ORDA_GPU_DIRECT_SHARD_COUNT", "8")
        monkeypatch.setenv("ORDA_GPU_DIRECT_SHARD_INDEX", str(shard_index))
        selected = [SimpleNamespace(nodeid=nodeid) for nodeid in all_nodeids]
        deselected = []

        def record_deselected(items, target=deselected):
            target.extend(items)

        config = SimpleNamespace(
            hook=SimpleNamespace(pytest_deselected=record_deselected)
        )

        conftest.pytest_collection_modifyitems(config, selected)
        selected_ids = {item.nodeid for item in selected}
        selected_by_shard.append(selected_ids & set(actual_nodeids))

        assert set(nonmatching_nodeids).issubset(selected_ids)
        assert {item.nodeid for item in deselected} == (
            set(actual_nodeids) - selected_by_shard[-1]
        )

    assert set().union(*selected_by_shard) == set(actual_nodeids)
    assert sum(len(shard) for shard in selected_by_shard) == len(actual_nodeids)
    assert all(
        left.isdisjoint(right) for left, right in combinations(selected_by_shard, 2)
    )


def test_cartesian_collection_disables_verbose_pass_logging(monkeypatch):
    monkeypatch.delenv("ORDA_GPU_DIRECT_SHARD_COUNT", raising=False)
    monkeypatch.delenv("ORDA_GPU_DIRECT_SHARD_INDEX", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(verbose=2))
    items = [
        SimpleNamespace(
            nodeid=(
                "test/correctness/test_cartesian_cuda.py::"
                "test_full_cartesian_group[tied-forward-float32-high_level]"
            )
        )
    ]

    conftest.pytest_collection_modifyitems(config, items)

    assert config.option.verbose == 0
