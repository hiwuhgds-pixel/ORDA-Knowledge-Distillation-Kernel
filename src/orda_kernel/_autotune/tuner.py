from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable
from threading import RLock

import torch

from .._api_types import AutotuneMode
from .._execution.recovery import is_oom_error, recover_after_oom
from ._device import device_identity
from ._env import _bounded_env_int
from ._shapes import _max_block_size
from .benchmark import (
    BenchmarkTask,
    FullBatchLaunchPlan,
    FullBatchLaunchRequest,
    FullBatchLaunchShape,
    adaptive_robust_repetitions,
    autotune_controls,
    bench_config,
    is_resource_error,
)
from .cache import (
    _DeviceBenchmarkBusy,
    _StoreUnavailable,
    device_benchmark_lock,
    get_hot_winner,
    invalidate_hot_winner,
    local_generation,
    persist_winner,
    read_disk_winner,
    remember_local_winner,
    renew_device_benchmark_lease,
    reset_cache,
)
from .keys import _runtime_key, _workload
from .models import (
    PRECOMPUTED_ALIGNED_SPAN,
    PRECOMPUTED_TOKENWISE,
    SEPARATE_COTRAINED,
    SEPARATE_FROZEN,
    SEPARATE_TRAINED,
    TIED_COTRAINED,
    TIED_FROZEN,
    TIED_TRAINED,
    AutotuneRuntimeKey,
    LaunchConfig,
)

# ── Tuning defaults and state ────────────────────────────────────────────────
# T4 fallbacks are empirically pinned; other devices use the general table.
T4_DEFAULT_CONFIGS: dict[str, LaunchConfig] = {
    PRECOMPUTED_ALIGNED_SPAN: LaunchConfig(16384, 8),
    PRECOMPUTED_TOKENWISE: LaunchConfig(16384, 8),
    SEPARATE_TRAINED: LaunchConfig(8192, 8),
    SEPARATE_COTRAINED: LaunchConfig(8192, 8),
    SEPARATE_FROZEN: LaunchConfig(16384, 8),
    TIED_FROZEN: LaunchConfig(8192, 16),
    TIED_TRAINED: LaunchConfig(8192, 16),
    TIED_COTRAINED: LaunchConfig(8192, 16),
}

DEFAULT_CONFIGS: dict[str, LaunchConfig] = {
    PRECOMPUTED_ALIGNED_SPAN: LaunchConfig(8192, 8),
    PRECOMPUTED_TOKENWISE: LaunchConfig(8192, 8),
    SEPARATE_TRAINED: LaunchConfig(16384, 32),
    SEPARATE_COTRAINED: LaunchConfig(16384, 32),
    SEPARATE_FROZEN: LaunchConfig(8192, 8),
    TIED_FROZEN: LaunchConfig(16384, 32),
    TIED_TRAINED: LaunchConfig(16384, 32),
    TIED_COTRAINED: LaunchConfig(16384, 32),
}


# Trials need fresh logits because every kernel writes dlogits in place.
_CANDIDATES: tuple[LaunchConfig, ...] = tuple(
    LaunchConfig(block_size, num_warps)
    for block_size in (8192, 16384, 32768)
    for num_warps in (8, 16, 32)
)


_QUICK_TUNE = 1
_DEEP_TUNE = 2
_FAILURE_BACKOFF_BASE_MS = _bounded_env_int(
    "ORDA_AUTOTUNE_FAILURE_BACKOFF_MS", 1_000, minimum=10, maximum=60_000
)
_FAILURE_BACKOFF_MAX_MS = max(
    _FAILURE_BACKOFF_BASE_MS,
    _bounded_env_int(
        "ORDA_AUTOTUNE_FAILURE_BACKOFF_MAX_MS",
        30_000,
        minimum=10,
        maximum=600_000,
    ),
)
_RUN_STATE_LOCK = RLock()
_TUNED_THIS_RUN: dict[AutotuneRuntimeKey, int] = {}
_FAILED_THIS_RUN: dict[AutotuneRuntimeKey, tuple[int, float]] = {}
# Store-degraded winners may disappear from the bounded hot LRU.
_SETTLED_THIS_RUN: set[AutotuneRuntimeKey] = set()


def begin_autotune_run() -> None:
    """Start a local run without evicting the compact winner cache."""
    with _RUN_STATE_LOCK:
        _TUNED_THIS_RUN.clear()
        _FAILED_THIS_RUN.clear()
        _SETTLED_THIS_RUN.clear()


def _requested_tune_depth(
    autotune: AutotuneMode,
    refresh_autotune: bool,
) -> int:
    if autotune == "Max":
        return _DEEP_TUNE
    if autotune is True and refresh_autotune:
        return _QUICK_TUNE
    return 0


def _tuned_to_depth(key: AutotuneRuntimeKey, requested_depth: int) -> bool:
    if requested_depth == 0:
        return False
    with _RUN_STATE_LOCK:
        return _TUNED_THIS_RUN.get(key, 0) >= requested_depth


def _mark_tuned(key: AutotuneRuntimeKey, depth: int) -> None:
    if depth == 0:
        return
    with _RUN_STATE_LOCK:
        _TUNED_THIS_RUN[key] = max(depth, _TUNED_THIS_RUN.get(key, 0))


def _failed_this_run(key: AutotuneRuntimeKey) -> bool:
    now = time.monotonic()
    with _RUN_STATE_LOCK:
        state = _FAILED_THIS_RUN.get(key)
        return state is not None and now < state[1]


def _mark_failed(key: AutotuneRuntimeKey) -> None:
    now = time.monotonic()
    with _RUN_STATE_LOCK:
        previous_count = _FAILED_THIS_RUN.get(key, (0, 0.0))[0]
        failure_count = previous_count + 1
        delay_ms = min(
            _FAILURE_BACKOFF_MAX_MS,
            _FAILURE_BACKOFF_BASE_MS * (1 << min(previous_count, 20)),
        )
        _FAILED_THIS_RUN[key] = (failure_count, now + delay_ms / 1_000.0)


def _clear_failed(key: AutotuneRuntimeKey) -> None:
    with _RUN_STATE_LOCK:
        _FAILED_THIS_RUN.pop(key, None)


def _settled_this_run(key: AutotuneRuntimeKey) -> bool:
    with _RUN_STATE_LOCK:
        return key in _SETTLED_THIS_RUN


def _mark_settled(key: AutotuneRuntimeKey) -> None:
    with _RUN_STATE_LOCK:
        _SETTLED_THIS_RUN.add(key)


# ── Block-size helpers ───────────────────────────────────────────────────────
def _max_num_warps_for_device(device: torch.device | None) -> int:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return 32
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    return device_identity(index).max_num_warps


def valid_candidates(
    vocab_size: int,
    max_fused_size: int,
    device: torch.device | None = None,
) -> list[LaunchConfig]:
    max_block = _max_block_size(vocab_size, max_fused_size)
    max_num_warps = _max_num_warps_for_device(device)
    return [
        config
        for config in _CANDIDATES
        if config.block_size <= max_block and config.num_warps <= max_num_warps
    ]


def _timing_sort_key(item: tuple[float, LaunchConfig]) -> tuple[float, int, int]:
    timing, config = item
    return float(timing), config.block_size, config.num_warps


class _LostDeviceLease(RuntimeError):
    pass


# ── Device profile selection ─────────────────────────────────────────────────
def _is_t4_device(device: torch.device | None = None) -> bool:
    if not torch.cuda.is_available() or (device is not None and device.type != "cuda"):
        return False
    index = (
        torch.cuda.current_device()
        if device is None or device.index is None
        else int(device.index)
    )
    return device_identity(index).is_t4


def _default_configs_for_device(
    device: torch.device | None = None,
    *,
    _is_t4_device_fn: Callable[..., bool] | None = None,
) -> dict[str, LaunchConfig]:
    is_t4_device = _is_t4_device if _is_t4_device_fn is None else _is_t4_device_fn
    if device is None:
        is_t4 = is_t4_device()
    elif device.type != "cuda":
        is_t4 = False
    else:
        is_t4 = is_t4_device(device)
    return T4_DEFAULT_CONFIGS if is_t4 else DEFAULT_CONFIGS


def default_config(
    mode: str,
    vocab_size: int,
    max_fused_size: int,
    *,
    device: torch.device | None = None,
    _is_t4_device_fn: Callable[[torch.device | None], bool] | None = None,
) -> LaunchConfig:
    config = _default_configs_for_device(
        device,
        _is_t4_device_fn=_is_t4_device_fn,
    )[mode]
    config = LaunchConfig(
        config.block_size,
        min(config.num_warps, _max_num_warps_for_device(device)),
    )
    max_block = _max_block_size(vocab_size, max_fused_size)
    if config.block_size <= max_block:
        return config

    candidates = valid_candidates(vocab_size, max_fused_size, device)
    same_warps = [
        candidate for candidate in candidates if candidate.num_warps == config.num_warps
    ]
    if same_warps:
        return same_warps[-1]
    if candidates:
        return candidates[-1]
    return LaunchConfig(max_block, config.num_warps)


# ── Config selection ─────────────────────────────────────────────────────────
def _latest_winner_or_fallback(
    key: AutotuneRuntimeKey,
    workload,
    *,
    valid_configs: set[LaunchConfig],
    fallback: LaunchConfig,
) -> LaunchConfig:
    winner = get_hot_winner(key)
    if winner is not None:
        return winner
    winner = read_disk_winner(
        key,
        workload,
        valid_configs=valid_configs,
        fallback=fallback,
    )
    return fallback if winner is None else winner


def select_config(
    *,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    bt: int,
    max_fused_size: int,
    shape_key: Iterable[int],
    autotune: AutotuneMode,
    benchmark_factory: Callable[[], BenchmarkTask] | None,
    compute_kl: bool,
    t_is_one: bool,
    compute_teacher_ce: bool,
    compute_student_ce: bool,
    kl_variant: str,
    teacher_grad_policy: str,
    grad_student_hidden: bool,
    grad_teacher_hidden: bool,
    grad_weight_accum_dtype: str | None,
    grad_teacher_weight_accum_dtype: str | None,
    gradient_materialization_scaled: bool = False,
    chunk_rows: int | None = None,
    chunk_dynamic: bool = False,
    pipeline_key: Iterable[str] = (),
    refresh_autotune: bool = False,
    _is_t4_device_fn: Callable[[torch.device | None], bool] | None = None,
) -> LaunchConfig:
    if not isinstance(refresh_autotune, bool):
        raise ValueError(f"refresh_autotune must be bool, got {refresh_autotune!r}")
    if autotune is False:
        if refresh_autotune:
            raise ValueError("refresh_autotune=True requires autotune=True or 'Max'")
        return default_config(
            mode,
            vocab_size,
            max_fused_size,
            device=device,
            _is_t4_device_fn=_is_t4_device_fn,
        )
    controls = autotune_controls(autotune)
    if benchmark_factory is None:
        return default_config(
            mode,
            vocab_size,
            max_fused_size,
            device=device,
            _is_t4_device_fn=_is_t4_device_fn,
        )

    # ── Cached-winner lookup ─────────────────────────────────
    shape_key = tuple(shape_key)
    pipeline_key = tuple(pipeline_key)
    key = _runtime_key(
        mode=mode,
        device=device,
        dtype=dtype,
        vocab_size=vocab_size,
        bt=bt,
        max_fused_size=max_fused_size,
        shape_key=shape_key,
        compute_kl=compute_kl,
        t_is_one=t_is_one,
        compute_teacher_ce=compute_teacher_ce,
        compute_student_ce=compute_student_ce,
        kl_variant=kl_variant,
        teacher_grad_policy=teacher_grad_policy,
        grad_student_hidden=grad_student_hidden,
        grad_teacher_hidden=grad_teacher_hidden,
        grad_weight_accum_dtype=grad_weight_accum_dtype,
        grad_teacher_weight_accum_dtype=grad_teacher_weight_accum_dtype,
        gradient_materialization_scaled=gradient_materialization_scaled,
        chunk_rows=chunk_rows,
        chunk_dynamic=chunk_dynamic,
        pipeline_key=pipeline_key,
    )
    requested_depth = _requested_tune_depth(autotune, refresh_autotune)
    retune_requested = requested_depth != 0 and not _tuned_to_depth(
        key, requested_depth
    )
    old_winner = get_hot_winner(key)
    if old_winner is not None and not retune_requested:
        _clear_failed(key)
        return old_winner

    fallback = default_config(
        mode,
        vocab_size,
        max_fused_size,
        device=device,
        _is_t4_device_fn=_is_t4_device_fn,
    )

    candidates = valid_candidates(vocab_size, max_fused_size, device)
    if not candidates:
        return fallback
    valid_set = set(candidates)
    workload = _workload(
        mode=mode,
        device=device,
        dtype=dtype,
        vocab_size=vocab_size,
        bt=bt,
        max_fused_size=max_fused_size,
        shape_key=shape_key,
        compute_kl=compute_kl,
        t_is_one=t_is_one,
        compute_teacher_ce=compute_teacher_ce,
        compute_student_ce=compute_student_ce,
        kl_variant=kl_variant,
        teacher_grad_policy=teacher_grad_policy,
        grad_student_hidden=grad_student_hidden,
        grad_teacher_hidden=grad_teacher_hidden,
        grad_weight_accum_dtype=grad_weight_accum_dtype,
        grad_teacher_weight_accum_dtype=grad_teacher_weight_accum_dtype,
        gradient_materialization_scaled=gradient_materialization_scaled,
        chunk_rows=chunk_rows,
        chunk_dynamic=chunk_dynamic,
        pipeline_key=pipeline_key,
    )
    disk_winner = read_disk_winner(
        key,
        workload,
        valid_configs=valid_set,
        fallback=fallback,
    )
    if disk_winner is not None:
        old_winner = disk_winner
    if old_winner is not None and not retune_requested:
        _clear_failed(key)
        return old_winner
    if not retune_requested and _settled_this_run(key):
        return fallback
    if _failed_this_run(key):
        return old_winner if old_winner is not None else fallback

    # ── Benchmark under device lease ─────────────────────────
    try:
        with device_benchmark_lock(device) as device_lease:
            selection_generation = local_generation()
            retune_requested = requested_depth != 0 and not _tuned_to_depth(
                key, requested_depth
            )
            winner = get_hot_winner(key)
            if winner is None:
                winner = read_disk_winner(
                    key,
                    workload,
                    valid_configs=valid_set,
                    fallback=fallback,
                )
            if winner is not None:
                old_winner = winner
            if winner is not None and not retune_requested:
                _clear_failed(key)
                return winner
            if not retune_requested and _settled_this_run(key):
                return fallback
            if _failed_this_run(key):
                return old_winner if old_winner is not None else fallback

            def renew_device_lease_or_degrade() -> None:
                nonlocal device_lease
                if device_lease is None:
                    return
                try:
                    if not renew_device_benchmark_lease(device_lease):
                        raise _LostDeviceLease(
                            "ORDA autotune lost its device benchmark lease"
                        )
                except _StoreUnavailable:
                    # The local lock remains valid; do not attempt an unguarded
                    # disk write when the store returns.
                    device_lease = None

            # Screening pass.
            screened: list[tuple[float, LaunchConfig]] = []
            renew_device_lease_or_degrade()
            try:
                task = benchmark_factory()
            except Exception as exc:
                if is_oom_error(exc):
                    recover_after_oom(device)
                    task = None
                elif is_resource_error(exc):
                    task = None
                else:
                    raise
            if task is not None:
                for config in candidates:
                    renew_device_lease_or_degrade()
                    try:
                        timing = bench_config(
                            task,
                            config,
                            device=device,
                            warmup_reps=controls.screen_warmup_reps,
                            repetitions=controls.screen_reps,
                            heartbeat=renew_device_lease_or_degrade,
                        )
                    except Exception as exc:
                        if is_oom_error(exc):
                            recover_after_oom(device)
                            continue
                        if is_resource_error(exc):
                            continue
                        raise
                    screened.append((timing, config))

            if not screened:
                renew_device_lease_or_degrade()
                return _finish_failed_selection(
                    key,
                    workload,
                    retune_requested=retune_requested,
                    valid_configs=valid_set,
                    fallback=fallback,
                    outcome="viable",
                )

            # Robust pass.
            assert task is not None
            finalists = sorted(screened, key=_timing_sort_key)[: controls.finalists]
            robust: list[tuple[float, LaunchConfig]] = []
            for screen_ms, config in finalists:
                renew_device_lease_or_degrade()
                try:
                    timing = bench_config(
                        task,
                        config,
                        device=device,
                        warmup_reps=controls.robust_warmup_reps,
                        repetitions=adaptive_robust_repetitions(
                            screen_ms,
                            minimum=controls.robust_min_reps,
                            maximum=controls.robust_max_reps,
                        ),
                        heartbeat=renew_device_lease_or_degrade,
                    )
                except Exception as exc:
                    if is_oom_error(exc):
                        recover_after_oom(device)
                        continue
                    if is_resource_error(exc):
                        continue
                    raise
                robust.append((timing, config))

            if not robust:
                renew_device_lease_or_degrade()
                return _finish_failed_selection(
                    key,
                    workload,
                    retune_requested=retune_requested,
                    valid_configs=valid_set,
                    fallback=fallback,
                    outcome="robust",
                )

            # Persist the winner.
            selected = min(robust, key=_timing_sort_key)[1]
            renew_device_lease_or_degrade()
            store_degraded = device_lease is None
            if device_lease is None:
                persisted = remember_local_winner(
                    key,
                    selected,
                    generation=selection_generation,
                )
            else:
                try:
                    persisted = persist_winner(
                        key,
                        workload,
                        selected,
                        lease=device_lease,
                    )
                except _StoreUnavailable:
                    store_degraded = True
                    persisted = remember_local_winner(
                        key,
                        selected,
                        generation=selection_generation,
                    )
            if not persisted:
                invalidate_hot_winner(key)
                return _latest_winner_or_fallback(
                    key,
                    workload,
                    valid_configs=valid_set,
                    fallback=fallback,
                )
            if store_degraded:
                _mark_settled(key)
            _clear_failed(key)
            _mark_tuned(key, requested_depth)
            return selected
    except _DeviceBenchmarkBusy:
        _mark_failed(key)
        return _latest_winner_or_fallback(
            key,
            workload,
            valid_configs=valid_set,
            fallback=fallback,
        )
    except _LostDeviceLease:
        # Re-read durable state after a reset or owner replacement.
        invalidate_hot_winner(key)
        return _latest_winner_or_fallback(
            key,
            workload,
            valid_configs=valid_set,
            fallback=fallback,
        )


def _finish_failed_selection(
    key: AutotuneRuntimeKey,
    workload,
    *,
    retune_requested: bool,
    valid_configs: set[LaunchConfig],
    fallback: LaunchConfig,
    outcome: str,
) -> LaunchConfig:
    _mark_failed(key)
    if retune_requested:
        current = _latest_winner_or_fallback(
            key,
            workload,
            valid_configs=valid_configs,
            fallback=fallback,
        )
        warnings.warn(
            f"ORDA autotune retune found no {outcome} config; retaining "
            f"{current.block_size}/{current.num_warps} until the retry backoff expires.",
            RuntimeWarning,
            stacklevel=3,
        )
        return current
    warnings.warn(
        f"ORDA autotune found no {outcome} config; using default "
        f"{fallback.block_size}/{fallback.num_warps}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return fallback


# ── Full-batch launch selection ──────────────────────────────────────────────
def select_full_batch_launch_plan(
    request: FullBatchLaunchRequest,
    *,
    benchmark_factory: Callable[[], BenchmarkTask] | None,
) -> FullBatchLaunchPlan:
    """Select one launch config before executing the real full-BT chunk loop."""
    shape = FullBatchLaunchShape.from_attempt(
        bt=request.bt,
        chunk_rows=request.chunk_rows,
        chunk_dynamic=request.chunk_dynamic,
    )
    if not isinstance(request.refresh_autotune, bool):
        raise ValueError(
            f"refresh_autotune must be bool, got {request.refresh_autotune!r}"
        )
    if request.autotune is False and request.refresh_autotune:
        raise ValueError("refresh_autotune=True requires autotune=True or 'Max'")
    if request.autotune is not False:
        autotune_controls(request.autotune)

    # Autotune-off skips benchmark, workload, cache-key, and device discovery.
    if request.autotune is False or benchmark_factory is None:
        return FullBatchLaunchPlan(
            shape,
            default_config(
                request.mode,
                request.vocab_size,
                request.max_fused_size,
                device=request.device,
            ),
        )

    config = select_config(
        mode=request.mode,
        device=request.device,
        dtype=request.dtype,
        vocab_size=request.vocab_size,
        bt=request.bt,
        max_fused_size=request.max_fused_size,
        shape_key=request.shape_key,
        autotune=request.autotune,
        refresh_autotune=request.refresh_autotune,
        benchmark_factory=benchmark_factory,
        compute_kl=request.compute_kl,
        t_is_one=request.t_is_one,
        compute_teacher_ce=request.compute_teacher_ce,
        compute_student_ce=request.compute_student_ce,
        kl_variant=request.kl_variant,
        teacher_grad_policy=request.teacher_grad_policy,
        grad_student_hidden=request.grad_student_hidden,
        grad_teacher_hidden=request.grad_teacher_hidden,
        grad_weight_accum_dtype=request.grad_weight_accum_dtype,
        grad_teacher_weight_accum_dtype=(request.grad_teacher_weight_accum_dtype),
        gradient_materialization_scaled=request.gradient_materialization_scaled,
        chunk_rows=shape.chunk_rows,
        chunk_dynamic=shape.chunk_dynamic,
        pipeline_key=(
            (*request.pipeline_key, "renyi-near-one")
            if request.renyi_near_one
            else request.pipeline_key
        ),
    )
    return FullBatchLaunchPlan(shape, config)


# ── Public cache reset ───────────────────────────────────────────────────────
def reset_autotune_cache() -> None:
    """Clear winners while fencing active device benchmark owners."""
    reset_cache()
    begin_autotune_run()
