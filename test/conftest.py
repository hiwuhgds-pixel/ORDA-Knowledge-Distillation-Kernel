from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import monotonic

import pytest
from utils.sharding import (
    DIRECT_GPU_TEST_PATHS,
    direct_gpu_shard_index,
    is_direct_gpu_nodeid,
)

_CARTESIAN_PATH = "test/correctness/test_cartesian_cuda.py"
_CARTESIAN_PROGRESS_STEP = 20


@dataclass
class _ProgressState:
    total: int
    file_totals: dict[str, int]
    completed: int = 0
    file_completed: dict[str, int] = field(default_factory=dict)
    cartesian_cases_total: int = 0
    cartesian_cases_by_nodeid: dict[str, int] = field(default_factory=dict)
    cartesian_cases_done: int = 0
    cartesian_started_at: float | None = None
    next_cartesian_percent: int = _CARTESIAN_PROGRESS_STEP


_PROGRESS_KEY = pytest.StashKey[_ProgressState]()


def _test_path(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_percent(percent: float) -> str:
    if 0.0 < percent < 0.01:
        return f"{percent:.4f}%"
    if percent < 1.0:
        return f"{percent:.2f}%"
    return f"{percent:.1f}%"


def _write_progress(config: pytest.Config, message: str) -> None:
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    reporter.ensure_newline()
    reporter.write_line(f"[progress] {message}", cyan=True)


def _record_completed_report(report, config: pytest.Config) -> None:
    completed = report.when == "call" or (
        report.when == "setup" and (report.failed or report.skipped)
    )
    if not completed or getattr(report, "_orda_progress_recorded", False):
        return
    report._orda_progress_recorded = True

    state = config.stash.get(_PROGRESS_KEY, None)
    if state is None:
        return
    state.completed += 1
    path = _test_path(report.nodeid)
    path_completed = state.file_completed.get(path, 0) + 1
    state.file_completed[path] = path_completed

    # Passing statuses are intentionally silent. Keep pytest's native failure
    # percentage correct by recording those hidden node IDs ourselves.
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    reported_nodeids = getattr(reporter, "_progress_nodeids_reported", None)
    if reported_nodeids is not None:
        reported_nodeids.add(report.nodeid)

    overall_percent = (
        100.0 if not state.total else 100.0 * state.completed / state.total
    )
    if path != _CARTESIAN_PATH:
        path_total = state.file_totals.get(path, 0)
        if path_completed == path_total:
            _write_progress(
                config,
                f"{path} complete ({path_completed}/{path_total}); "
                f"overall {state.completed}/{state.total} "
                f"({_format_percent(overall_percent)})",
            )
        return

    state.cartesian_cases_done += state.cartesian_cases_by_nodeid.get(report.nodeid, 1)
    cartesian_total = state.cartesian_cases_total
    now = monotonic()
    if state.cartesian_started_at is None:
        state.cartesian_started_at = now
        _write_progress(
            config,
            f"{path} started: {cartesian_total} cases",
        )
        return

    cartesian_percent = (
        100.0
        if not cartesian_total
        else 100.0 * state.cartesian_cases_done / cartesian_total
    )
    if (
        state.cartesian_cases_done != cartesian_total
        and cartesian_percent < state.next_cartesian_percent
    ):
        return
    while state.next_cartesian_percent <= cartesian_percent:
        state.next_cartesian_percent += _CARTESIAN_PROGRESS_STEP

    elapsed = now - state.cartesian_started_at
    rate = state.cartesian_cases_done / elapsed if elapsed > 0.0 else 0.0
    remaining = max(cartesian_total - state.cartesian_cases_done, 0)
    eta = remaining / rate if rate > 0.0 else 0.0
    _write_progress(
        config,
        f"{path}: {state.cartesian_cases_done}/{cartesian_total} "
        f"({cartesian_percent:.1f}%), elapsed {_format_duration(elapsed)}, "
        f"ETA {_format_duration(eta)}",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires CUDA/HIP and Triton kernels")


def pytest_report_teststatus(report, config):
    """Keep large Cartesian runs quiet while preserving non-pass diagnostics."""
    _record_completed_report(report, config)
    if report.when == "call" and report.passed and not hasattr(report, "wasxfail"):
        return "passed", "", ""
    return None


def _cartesian_group_case_count(item) -> int:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return 1
    members = callspec.params.get("group_cases")
    if members is None:
        return 1
    return len(members)


def pytest_collection_finish(session: pytest.Session) -> None:
    file_totals: dict[str, int] = {}
    cartesian_cases_total = 0
    cartesian_cases_by_nodeid: dict[str, int] = {}
    for item in session.items:
        path = _test_path(item.nodeid)
        file_totals[path] = file_totals.get(path, 0) + 1
        if path == _CARTESIAN_PATH:
            case_count = _cartesian_group_case_count(item)
            cartesian_cases_by_nodeid[item.nodeid] = case_count
            cartesian_cases_total += case_count
    session.config.stash[_PROGRESS_KEY] = _ProgressState(
        total=len(session.items),
        file_totals=file_totals,
        cartesian_cases_total=cartesian_cases_total,
        cartesian_cases_by_nodeid=cartesian_cases_by_nodeid,
    )


# ── Deterministic direct-GPU sharding ────────────────────────────────────────
def _verify_direct_gpu_partition(items, shard_count: int) -> None:
    nodeids = [item.nodeid for item in items if is_direct_gpu_nodeid(item.nodeid)]
    missing_paths = [
        path
        for path in DIRECT_GPU_TEST_PATHS
        if not any(
            nodeid == path or nodeid.startswith(f"{path}::") for nodeid in nodeids
        )
    ]
    if missing_paths:
        raise pytest.UsageError(
            f"direct GPU collection is missing test paths: {missing_paths!r}"
        )
    if len(nodeids) != len(set(nodeids)):
        raise pytest.UsageError("direct GPU collection contains duplicate node IDs")

    shard_sizes = [0] * shard_count
    for nodeid in nodeids:
        shard_sizes[direct_gpu_shard_index(nodeid, shard_count)] += 1
    if any(size == 0 for size in shard_sizes):
        raise pytest.UsageError(
            f"direct GPU collection leaves an empty shard: sizes={shard_sizes!r}"
        )
    if sum(shard_sizes) != len(nodeids):
        raise pytest.UsageError("direct GPU shard partition does not cover collection")


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    raw_verify_count = os.environ.get("ORDA_VERIFY_DIRECT_SHARD_COUNT")
    if raw_verify_count is not None:
        try:
            verify_count = int(raw_verify_count)
        except ValueError as exc:
            raise pytest.UsageError(
                "ORDA_VERIFY_DIRECT_SHARD_COUNT must be an integer"
            ) from exc
        if verify_count < 1:
            raise pytest.UsageError("ORDA_VERIFY_DIRECT_SHARD_COUNT must be >= 1")
        _verify_direct_gpu_partition(items, verify_count)

    # A verbose Cartesian run would print hundreds of thousands of passing
    # node IDs. Keep failures visible through normal report handling instead.
    if any(item.nodeid.startswith(f"{_CARTESIAN_PATH}::") for item in items):
        option = getattr(config, "option", None)
        if option is not None:
            option.verbose = 0

    raw_count = os.environ.get("ORDA_GPU_DIRECT_SHARD_COUNT")
    raw_index = os.environ.get("ORDA_GPU_DIRECT_SHARD_INDEX")
    if raw_count is None and raw_index is None:
        return
    if raw_count is None or raw_index is None:
        raise pytest.UsageError(
            "ORDA_GPU_DIRECT_SHARD_COUNT and ORDA_GPU_DIRECT_SHARD_INDEX "
            "must be set together"
        )
    try:
        shard_count = int(raw_count)
        shard_index = int(raw_index)
    except ValueError as exc:
        raise pytest.UsageError("direct GPU shard values must be integers") from exc
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise pytest.UsageError(
            "direct GPU shard requires count >= 1 and 0 <= index < count"
        )

    selected = []
    deselected = []
    for item in items:
        if not is_direct_gpu_nodeid(item.nodeid):
            selected.append(item)
            continue
        bucket = direct_gpu_shard_index(item.nodeid, shard_count)
        (selected if bucket == shard_index else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(autouse=True)
def clear_cuda_cache_after_test(request):
    # Flushing after every generic GPU case destroys allocator reuse across the
    # 345k-case Cartesian matrix. Only dedicated autotune tests need isolation.
    if request.node.get_closest_marker("autotune") is None:
        yield
        return
    yield
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
