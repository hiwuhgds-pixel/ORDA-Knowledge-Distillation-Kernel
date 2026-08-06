"""Canonical autotune models, benchmark controls, and selection APIs."""

from __future__ import annotations

from ._contract import AUTOTUNE_TIMED_CONTRACT_ID
from .benchmark import (
    BenchmarkTask,
    FullBatchLaunchPlan,
    FullBatchLaunchRequest,
    FullBatchLaunchShape,
    bench_config,
    full_batch_benchmark_task,
    staged_invocation_bound,
)
from .cache import cache_path, reset_cache
from .models import (
    AUTOTUNE_SCHEMA_VERSION,
    PRECOMPUTED_ALIGNED_SPAN,
    PRECOMPUTED_TOKENWISE,
    SEPARATE_COTRAINED,
    SEPARATE_FROZEN,
    SEPARATE_TRAINED,
    TIED_COTRAINED,
    TIED_FROZEN,
    TIED_TRAINED,
    AutotuneRuntimeKey,
    AutotuneWorkload,
    DeviceFingerprint,
    LaunchConfig,
)
from .tuner import (
    begin_autotune_run,
    default_config,
    reset_autotune_cache,
    select_config,
    select_full_batch_launch_plan,
    valid_candidates,
)

__all__ = [
    "AUTOTUNE_SCHEMA_VERSION",
    "AUTOTUNE_TIMED_CONTRACT_ID",
    "AutotuneRuntimeKey",
    "AutotuneWorkload",
    "BenchmarkTask",
    "DeviceFingerprint",
    "FullBatchLaunchPlan",
    "FullBatchLaunchRequest",
    "FullBatchLaunchShape",
    "LaunchConfig",
    "PRECOMPUTED_ALIGNED_SPAN",
    "PRECOMPUTED_TOKENWISE",
    "SEPARATE_COTRAINED",
    "SEPARATE_FROZEN",
    "SEPARATE_TRAINED",
    "TIED_COTRAINED",
    "TIED_FROZEN",
    "TIED_TRAINED",
    "begin_autotune_run",
    "bench_config",
    "cache_path",
    "default_config",
    "full_batch_benchmark_task",
    "reset_autotune_cache",
    "reset_cache",
    "select_config",
    "select_full_batch_launch_plan",
    "staged_invocation_bound",
    "valid_candidates",
]
