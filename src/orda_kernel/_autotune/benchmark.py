from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .._api_types import AutotuneMode
from ._env import _bounded_env_int
from .models import LaunchConfig

# ── Environment controls ─────────────────────────────────────────────────────
AUTOTUNE_SCREEN_WARMUP_REPS = _bounded_env_int(
    "ORDA_AUTOTUNE_SCREEN_WARMUP_REPS", 2, minimum=0, maximum=2
)
AUTOTUNE_SCREEN_REPS = _bounded_env_int(
    "ORDA_AUTOTUNE_SCREEN_REPS", 3, minimum=3, maximum=5
)
AUTOTUNE_FINALISTS = _bounded_env_int(
    "ORDA_AUTOTUNE_FINALISTS", 3, minimum=1, maximum=3
)
AUTOTUNE_ROBUST_MIN_REPS = _bounded_env_int(
    "ORDA_AUTOTUNE_ROBUST_MIN_REPS", 4, minimum=1, maximum=8
)
AUTOTUNE_ROBUST_MAX_REPS = max(
    AUTOTUNE_ROBUST_MIN_REPS,
    _bounded_env_int("ORDA_AUTOTUNE_ROBUST_MAX_REPS", 12, minimum=1, maximum=12),
)
AUTOTUNE_ROBUST_TARGET_MS = _bounded_env_int(
    "ORDA_AUTOTUNE_ROBUST_TARGET_MS", 80, minimum=1, maximum=500
)
AUTOTUNE_ROBUST_WARMUP_REPS = 2
AUTOTUNE_TIMING_BATCH = _bounded_env_int(
    "ORDA_AUTOTUNE_TIMING_BATCH", 1, minimum=1, maximum=16
)


@dataclass(frozen=True)
class AutotuneControls:
    screen_warmup_reps: int
    screen_reps: int
    finalists: int
    robust_warmup_reps: int
    robust_min_reps: int
    robust_max_reps: int


def autotune_controls(autotune: AutotuneMode) -> AutotuneControls:
    if autotune is True:
        return AutotuneControls(
            screen_warmup_reps=AUTOTUNE_SCREEN_WARMUP_REPS,
            screen_reps=AUTOTUNE_SCREEN_REPS,
            finalists=AUTOTUNE_FINALISTS,
            robust_warmup_reps=AUTOTUNE_ROBUST_WARMUP_REPS,
            robust_min_reps=AUTOTUNE_ROBUST_MIN_REPS,
            robust_max_reps=AUTOTUNE_ROBUST_MAX_REPS,
        )
    if autotune == "Max":
        return AutotuneControls(
            screen_warmup_reps=2,
            screen_reps=3,
            finalists=3,
            robust_warmup_reps=3,
            robust_min_reps=6,
            robust_max_reps=18,
        )
    raise ValueError(f"autotune must be True or 'Max', got {autotune!r}")


# ── Benchmark tasks ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BenchmarkTask:
    """Run a full-BT pipeline with scratch reset before each timed batch."""

    run: Callable[[int, int], object]
    prepare_batch: Callable[[int], None] | None = None


@dataclass(frozen=True)
class FullBatchLaunchShape:
    bt: int
    chunk_rows: int
    chunk_dynamic: bool

    @classmethod
    def from_attempt(
        cls,
        *,
        bt: int,
        chunk_rows: int,
        chunk_dynamic: bool,
    ) -> FullBatchLaunchShape:
        return cls(
            bt=int(bt),
            chunk_rows=int(chunk_rows),
            chunk_dynamic=bool(chunk_dynamic),
        )

    def __post_init__(self) -> None:
        if self.bt < 1:
            raise ValueError(f"bt must be >= 1, got {self.bt}")
        if self.chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {self.chunk_rows}")

    @property
    def full_chunk_count(self) -> int:
        return self.bt // self.chunk_rows

    @property
    def tail_rows(self) -> int:
        return self.bt % self.chunk_rows

    @property
    def num_chunks(self) -> int:
        return self.full_chunk_count + int(self.tail_rows != 0)

    def iter_chunk_rows(self) -> tuple[int, ...]:
        rows = (self.chunk_rows,) * self.full_chunk_count
        return rows if self.tail_rows == 0 else (*rows, self.tail_rows)


@dataclass(frozen=True)
class FullBatchLaunchPlan:
    shape: FullBatchLaunchShape
    config: LaunchConfig


@dataclass(frozen=True, slots=True)
class FullBatchLaunchRequest:
    mode: str
    device: torch.device
    dtype: torch.dtype
    vocab_size: int
    bt: int
    chunk_rows: int
    chunk_dynamic: bool
    max_fused_size: int
    shape_key: tuple[int, ...]
    autotune: AutotuneMode
    refresh_autotune: bool
    compute_kl: bool
    t_is_one: bool
    compute_teacher_ce: bool
    compute_student_ce: bool
    kl_variant: str
    teacher_grad_policy: str
    grad_student_hidden: bool
    grad_teacher_hidden: bool
    grad_weight_accum_dtype: str | None
    grad_teacher_weight_accum_dtype: str | None
    pipeline_key: tuple[str, ...] = ()
    renyi_near_one: bool = False
    gradient_materialization_scaled: bool = False


def full_batch_benchmark_task(
    *,
    run_config: Callable[[LaunchConfig], object],
    prepare_batch: Callable[[int], None] | None = None,
) -> BenchmarkTask:
    def run(block_size: int, num_warps: int) -> object:
        return run_config(LaunchConfig(int(block_size), int(num_warps)))

    return BenchmarkTask(run=run, prepare_batch=prepare_batch)


# ── Failure and repetition policy ────────────────────────────────────────────
def is_resource_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of resource" in msg
        or "too many resources requested" in msg
        or "register spill" in msg
        or "shared memory" in msg
        and "exceed" in msg
    )


def adaptive_robust_repetitions(
    screen_ms: float, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    """Choose bounded robust samples from the observed exact full-plan latency."""
    minimum = AUTOTUNE_ROBUST_MIN_REPS if minimum is None else minimum
    maximum = AUTOTUNE_ROBUST_MAX_REPS if maximum is None else maximum
    if not math.isfinite(screen_ms) or screen_ms <= 0.0:
        return maximum
    target_reps = math.ceil(AUTOTUNE_ROBUST_TARGET_MS / screen_ms)
    return min(
        maximum,
        max(minimum, target_reps),
    )


def staged_invocation_bound(
    *, candidate_count: int, tuned_roles: int = 1, autotune: AutotuneMode = True
) -> int:
    """Hard upper bound for full-plan calls under validated staged controls."""
    if candidate_count < 0 or tuned_roles < 0:
        raise ValueError("candidate_count and tuned_roles must be >= 0")
    if autotune is False:
        return 0
    controls = autotune_controls(autotune)
    finalists = min(int(candidate_count), controls.finalists)
    per_role = int(candidate_count) * (
        controls.screen_warmup_reps + controls.screen_reps
    ) + finalists * (controls.robust_warmup_reps + controls.robust_max_reps)
    return int(tuned_roles) * per_role


def _weighted_median(samples: list[tuple[float, int]]) -> float:
    """Return the median batch time weighted by its exact repetition count."""
    if not samples:
        raise ValueError("weighted median requires at least one sample")
    if any(weight < 0 for _, weight in samples):
        raise ValueError("weighted median does not accept negative weights")
    total_weight = sum(weight for _, weight in samples)
    if total_weight < 1:
        raise ValueError("weighted median requires positive total weight")

    lower_rank = (total_weight - 1) // 2
    upper_rank = total_weight // 2
    accumulated = 0
    lower_value: float | None = None
    upper_value: float | None = None
    for value, weight in sorted(samples):
        if weight == 0:
            continue
        next_accumulated = accumulated + weight
        if lower_value is None and lower_rank < next_accumulated:
            lower_value = float(value)
        if upper_rank < next_accumulated:
            upper_value = float(value)
            break
        accumulated = next_accumulated
    if lower_value is None or upper_value is None:
        raise ValueError("weighted median ranks were not covered by the samples")
    return (lower_value + upper_value) / 2.0


# ── Timing loop ──────────────────────────────────────────────────────────────
def bench_config(
    task: BenchmarkTask,
    config: LaunchConfig,
    *,
    device: torch.device | None = None,
    warmup_reps: int = AUTOTUNE_SCREEN_WARMUP_REPS,
    repetitions: int = AUTOTUNE_ROBUST_MAX_REPS,
    heartbeat: Callable[[], None] | None = None,
) -> float:
    if warmup_reps < 0:
        raise ValueError(f"warmup_reps must be >= 0, got {warmup_reps}")
    if repetitions < 1:
        raise ValueError(f"repetitions must be >= 1, got {repetitions}")
    benchmark_device = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )

    def run_once() -> None:
        if task.prepare_batch is not None:
            task.prepare_batch(1)
        outputs = task.run(config.block_size, config.num_warps)
        del outputs

    # ── Warmup and timed samples ─────────────────────────────
    with torch.cuda.device(benchmark_device):
        for _ in range(warmup_reps):
            run_once()
            if heartbeat is not None:
                heartbeat()
        torch.cuda.synchronize(benchmark_device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: list[tuple[float, int]] = []
        requested_repetitions = int(repetitions)
        completed_repetitions = 0
        while completed_repetitions < requested_repetitions:
            batch_reps = min(
                AUTOTUNE_TIMING_BATCH,
                requested_repetitions - completed_repetitions,
            )
            if task.prepare_batch is not None:
                # Scratch resets are enqueued before the start event, so the
                # measured interval contains only the steady-state full-BT plan.
                task.prepare_batch(batch_reps)
            start.record()
            for _ in range(batch_reps):
                outputs = task.run(config.block_size, config.num_warps)
                del outputs
            end.record()
            end.synchronize()
            duration_ms = float(start.elapsed_time(end)) / batch_reps
            times.append((duration_ms, batch_reps))
            completed_repetitions += batch_reps
            # Lease maintenance is deliberately outside the event interval.
            if heartbeat is not None:
                heartbeat()
    return _weighted_median(times)
