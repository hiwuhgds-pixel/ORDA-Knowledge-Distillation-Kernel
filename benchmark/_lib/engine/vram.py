from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch

from ..core.device import (
    full_cleanup,
    memory_allocated,
    peak_extra_mb,
    reset_peak_memory,
    sync,
    tensor_bytes,
)
from ..core.metrics import OOM
from ..core.oom import is_oom_error
from ..core.validation import validate_benchmark_args


# ── Measurement result ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class VramMeasurement:
    status: str
    backend_extra_mb: float
    cache_teacher_mb: float | None = None
    total_extra_mb: float | None = None
    oom_stage: str | None = None

    @classmethod
    def oom(cls, stage: str, *, with_cache: bool) -> VramMeasurement:
        return cls(
            status="oom",
            backend_extra_mb=OOM,
            cache_teacher_mb=OOM if with_cache else None,
            total_extra_mb=OOM if with_cache else None,
            oom_stage=stage,
        )


# ── Workload protocol ────────────────────────────────────────────────────────
class VramWorkload(Protocol):
    """Keep precomputed cache bytes separate from the backend working set."""

    cache_indices: tuple[int, ...]

    def loss_fn(self): ...

    def inputs(self) -> tuple: ...

    def run(self, loss_fn, inputs: tuple) -> None: ...


def backward_inputs(
    loss_fn,
    inputs: tuple,
    *,
    differentiable: tuple[tuple[int, bool], ...],
    clear_grads: tuple[int, ...],
    device: torch.device,
) -> None:
    args = list(inputs)
    for index, requires_grad in differentiable:
        args[index] = args[index].detach().requires_grad_(requires_grad)
    for index in clear_grads:
        args[index].grad = None
    loss = loss_fn(*args)
    loss.backward()
    sync(device)


# ── VRAM measurement ─────────────────────────────────────────────────────────
def measure_once(
    workload: VramWorkload,
    *,
    warmup: int,
    measure_iters: int,
    device: torch.device,
) -> VramMeasurement:
    """Guard each stage separately so OOM is attributed to its source."""
    cache_indices = tuple(getattr(workload, "cache_indices", ()))
    with_cache = bool(cache_indices)
    full_cleanup(device)
    loss_fn = inputs = None

    def oom_result(stage: str) -> VramMeasurement:
        print(f"[WARN] VRAM measurement OOM during {stage}")
        return VramMeasurement.oom(stage, with_cache=with_cache)

    @contextmanager
    def guard(stage: str):
        try:
            yield
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            raise _StageOOM(stage) from exc

    try:
        try:
            with guard("loss setup"):
                loss_fn = workload.loss_fn()

            for _ in range(warmup):
                try:
                    with guard("input/cache setup"):
                        inputs = workload.inputs()
                    with guard("warmup"):
                        workload.run(loss_fn, inputs)
                finally:
                    inputs = None

            with guard("input/cache setup"):
                inputs = workload.inputs()
            cache_mb = (
                tensor_bytes(*(inputs[index] for index in cache_indices)) / 1024**2
            )
            sync(device)
            reset_peak_memory(device)
            baseline = memory_allocated(device)
            with guard("measured backend"):
                for _ in range(measure_iters):
                    workload.run(loss_fn, inputs)
        except _StageOOM as exc:
            return oom_result(exc.stage)

        backend_mb = peak_extra_mb(device, baseline)
        if not with_cache:
            return VramMeasurement(status="ok", backend_extra_mb=backend_mb)
        return VramMeasurement(
            status="ok",
            backend_extra_mb=backend_mb,
            cache_teacher_mb=cache_mb,
            total_extra_mb=backend_mb + cache_mb,
        )
    finally:
        loss_fn = inputs = None
        full_cleanup(device)


class _StageOOM(Exception):
    def __init__(self, stage: str):
        super().__init__(stage)
        self.stage = stage


def validate_shape(vocab: int, seq: int, batch: int, warmup: int, measure_iters: int):
    validate_benchmark_args(
        vocab=vocab,
        seq=seq,
        batch=batch,
        warmup=warmup,
        measure_iters=measure_iters,
    )


@contextmanager
def relaxed_recompile_limit(limit: int = 64):
    """Prevent eager fallback while one case compiles many shapes."""
    compiler = getattr(torch, "compiler", None)
    config = getattr(compiler, "config", None)
    if config is None or not hasattr(config, "recompile_limit"):
        yield
        return
    previous = config.recompile_limit
    config.recompile_limit = limit
    try:
        yield
    finally:
        config.recompile_limit = previous
