from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from ..core.device import (
    full_cleanup,
    peak_mb,
    reset_peak_memory,
    seed_all,
    sync,
)
from ..core.metrics import OOM
from ..core.oom import is_oom_error
from ..core.stats import trimmed_mean
from ..core.validation import validate_benchmark_args


# ── Training context ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrainMeasurement:
    ms: float
    work_vram_mb: float
    teacher_cache_mb: float = 0.0
    total_vram_mb: float = 0.0

    @classmethod
    def oom(cls) -> TrainMeasurement:
        return cls(OOM, OOM, OOM, OOM)

    @property
    def is_oom(self) -> bool:
        return self.ms == OOM


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    forward_loss: Callable[[], torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> Callable[[], torch.Tensor]:
    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            loss = forward_loss()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    return step


@dataclass
class BuiltStep:
    step: Callable[[], torch.Tensor]
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    teacher_cache_bytes: int = 0


@dataclass(frozen=True)
class TrainContext:
    vocab: int
    seq: int
    batch: int
    warmup: int
    update_steps: int
    dtype: torch.dtype
    device: torch.device
    use_amp: bool
    use_grad_scaler: bool
    mode: str

    def grad_scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and self.use_grad_scaler),
        )

    def sgd(self, parameters) -> torch.optim.SGD:
        return torch.optim.SGD(parameters, lr=1e-3)


class TrainWorkload(Protocol):
    def build(self, ctx: TrainContext) -> BuiltStep: ...


# ── Step measurement ─────────────────────────────────────────────────────────
def measure_steps(
    step,
    *,
    optimizer,
    scaler,
    warmup: int,
    update_steps: int,
    device: torch.device,
) -> tuple[float, float]:
    """Reject non-finite, gradient-free, or skipped steps before timing."""
    validation_steps = max(warmup, 1)
    scale_before = None
    for index in range(validation_steps):
        if index == validation_steps - 1 and scaler.is_enabled():
            scale_before = scaler.get_scale()
        loss = step()

    finite_checks = [torch.isfinite(loss).all()]
    active_gradients = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                active_gradients += 1
                finite_checks.append(torch.isfinite(parameter.grad).all())
    if active_gradients == 0:
        raise RuntimeError("training preflight produced no gradients")
    if not bool(torch.stack(finite_checks).all().item()):
        raise FloatingPointError(
            "training preflight produced non-finite loss or gradients"
        )
    if scale_before is not None and scaler.get_scale() < scale_before:
        raise FloatingPointError("GradScaler skipped the final training preflight step")

    sync(device)
    reset_peak_memory(device)
    sync(device)

    measured_scale = scaler.get_scale() if scaler.is_enabled() else None
    samples = []
    if device.type == "cuda":
        event_pairs = []
        for _ in range(update_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            event_pairs.append((start, end))
        sync(device)
        samples = [start.elapsed_time(end) for start, end in event_pairs]
    else:
        for _ in range(update_steps):
            started = time.perf_counter()
            step()
            samples.append((time.perf_counter() - started) * 1_000)
    if measured_scale is not None and scaler.get_scale() < measured_scale:
        raise FloatingPointError(
            "GradScaler skipped at least one measured training step"
        )
    return trimmed_mean(samples, 0.10), peak_mb(device)


# ── Isolated execution ───────────────────────────────────────────────────────
def run_isolated(
    workload: TrainWorkload,
    ctx: TrainContext,
    *,
    seed: int | None = None,
) -> TrainMeasurement:
    validate_benchmark_args(
        vocab=ctx.vocab,
        seq=ctx.seq,
        batch=ctx.batch,
        warmup=ctx.warmup,
        update_steps=ctx.update_steps,
    )
    full_cleanup(ctx.device)
    if seed is not None:
        seed_all(ctx.device, seed)
    built = None
    try:
        built = workload.build(ctx)
        ms, total_vram_mb = measure_steps(
            built.step,
            optimizer=built.optimizer,
            scaler=built.scaler,
            warmup=ctx.warmup,
            update_steps=ctx.update_steps,
            device=ctx.device,
        )
        teacher_cache_mb = built.teacher_cache_bytes / 1024**2
        return TrainMeasurement(
            ms=ms,
            work_vram_mb=max(0.0, total_vram_mb - teacher_cache_mb),
            teacher_cache_mb=teacher_cache_mb,
            total_vram_mb=total_vram_mb,
        )
    except Exception as exc:
        if not is_oom_error(exc):
            raise
        return TrainMeasurement.oom()
    finally:
        built = None
        full_cleanup(ctx.device)


# ── Pass aggregation ─────────────────────────────────────────────────────────
def run_passes(
    modes: Sequence[str],
    passes: int,
    measure: Callable[[str, int], TrainMeasurement],
    *,
    keys: Sequence[str] = ("ms", "work_vram_mb"),
    on_pass_start: Callable[[int, str], None] | None = None,
    on_result: Callable[[str, TrainMeasurement], None] | None = None,
    on_averages_start: Callable[[], None] | None = None,
    on_average: Callable[[str, dict[str, float]], None] | None = None,
) -> dict[str, dict[str, float]]:
    """Emit progress during long sweeps while alternating backend order."""
    from ..core.stats import average_passes
    from ..suites.specs import alternating_passes

    samples: dict[str, list[dict[str, float]]] = {mode: [] for mode in modes}
    for pass_index, direction, order in alternating_passes(modes, passes):
        if on_pass_start is not None:
            on_pass_start(pass_index, direction)
        for mode in order:
            result = measure(mode, pass_index)
            samples[mode].append({key: getattr(result, key) for key in keys})
            if on_result is not None:
                on_result(mode, result)

    if on_averages_start is not None:
        on_averages_start()
    averages = {}
    for mode in modes:
        averages[mode] = average_passes(samples[mode], keys, guard_key="ms")
        if on_average is not None:
            on_average(mode, averages[mode])
    return averages
