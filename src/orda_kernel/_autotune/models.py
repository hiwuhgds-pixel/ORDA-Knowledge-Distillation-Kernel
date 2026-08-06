from __future__ import annotations

import json
from dataclasses import dataclass
from typing import NamedTuple

from .._planning.models import (
    CANONICAL_INACTIVE_KL_VARIANT as CANONICAL_INACTIVE_KL_VARIANT,
)
from .._planning.models import (
    PRECOMPUTED_ALIGNED_SPAN,
    PRECOMPUTED_TOKENWISE,
    SEPARATE_COTRAINED,
    SEPARATE_FROZEN,
    SEPARATE_TRAINED,
    TEACHER_GRAD_CE,
    TEACHER_GRAD_CE_KL,
    TEACHER_GRAD_KL,
    TEACHER_GRAD_NONE,
    TIED_COTRAINED,
    TIED_FROZEN,
    TIED_TRAINED,
    VALID_KL_VARIANTS,
)
from ._contract import AUTOTUNE_TIMED_CONTRACT_ID

# ── Cache identity ───────────────────────────────────────────────────────────
# Schema versions the store; the timed contract versions benchmarked code.
AUTOTUNE_SCHEMA_VERSION = 1

_VALID_MODES = frozenset(
    {
        TIED_FROZEN,
        TIED_TRAINED,
        TIED_COTRAINED,
        SEPARATE_TRAINED,
        SEPARATE_FROZEN,
        SEPARATE_COTRAINED,
        PRECOMPUTED_TOKENWISE,
        PRECOMPUTED_ALIGNED_SPAN,
    }
)
_VALID_TEACHER_GRAD_POLICIES = frozenset(
    {
        TEACHER_GRAD_NONE,
        TEACHER_GRAD_CE,
        TEACHER_GRAD_KL,
        TEACHER_GRAD_CE_KL,
    }
)


# ── Launch and process-local identity ────────────────────────────────────────
class LaunchConfig(NamedTuple):
    block_size: int
    num_warps: int


class AutotuneRuntimeKey(NamedTuple):
    """Compact in-process identity that avoids device fingerprinting on hot hits."""

    mode: str
    device_type: str
    device_index: int | None
    dtype: str
    vocab_size: int
    bt: int
    effective_cap: int
    shape_key: tuple[int, ...]
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
    gradient_materialization_scaled: bool
    chunk_rows: int
    chunk_dynamic: bool
    timed_contract_id: str
    pipeline_key: tuple[str, ...] = ()


# ── Device identity ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DeviceFingerprint:
    device_type: str
    device_index: int | None
    backend: str
    runtime_version: str | None
    torch_version: str
    triton_version: str | None
    name: str | None = None
    major: int | None = None
    minor: int | None = None
    multi_processor_count: int | None = None
    total_memory: int | None = None
    physical_id: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "type": self.device_type,
            "index": self.device_index,
            "backend": self.backend,
            "runtime_version": self.runtime_version,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
            "name": self.name,
            "major": self.major,
            "minor": self.minor,
            "multi_processor_count": self.multi_processor_count,
            "total_memory": self.total_memory,
            "physical_id": self.physical_id,
        }


# ── Workload identity ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AutotuneWorkload:
    mode: str
    device: DeviceFingerprint
    dtype: str
    vocab_size: int
    bt: int
    effective_cap: int
    shape_key: tuple[int, ...]
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
    gradient_materialization_scaled: bool
    chunk_rows: int
    chunk_dynamic: bool
    pipeline_key: tuple[str, ...] = ()
    schema_version: int = AUTOTUNE_SCHEMA_VERSION
    timed_contract_id: str = AUTOTUNE_TIMED_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"unknown autotune mode {self.mode!r}")
        if self.teacher_grad_policy not in _VALID_TEACHER_GRAD_POLICIES:
            raise ValueError(
                f"unknown teacher gradient policy {self.teacher_grad_policy!r}"
            )
        if self.kl_variant not in VALID_KL_VARIANTS:
            raise ValueError(f"unknown KL variant {self.kl_variant!r}")
        if self.bt < 1:
            raise ValueError(f"bt must be >= 1, got {self.bt}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.effective_cap < 1:
            raise ValueError(f"effective_cap must be >= 1, got {self.effective_cap}")
        if any(item < 1 for item in self.shape_key):
            raise ValueError(f"shape_key values must be >= 1, got {self.shape_key}")
        if any(not isinstance(item, str) or not item for item in self.pipeline_key):
            raise ValueError(
                "pipeline_key values must be non-empty strings, got "
                f"{self.pipeline_key}"
            )
        if self.chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {self.chunk_rows}")
        if not self.timed_contract_id:
            raise ValueError("timed_contract_id must be non-empty")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "timed_contract_id": self.timed_contract_id,
            "mode": self.mode,
            "device": self.device.to_payload(),
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "bt": self.bt,
            "effective_cap": self.effective_cap,
            "shape_key": list(self.shape_key),
            "pipeline_key": list(self.pipeline_key),
            "compute_kl": self.compute_kl,
            "t_is_one": self.t_is_one,
            "compute_teacher_ce": self.compute_teacher_ce,
            "compute_student_ce": self.compute_student_ce,
            "kl_variant": self.kl_variant,
            "teacher_grad_policy": self.teacher_grad_policy,
            "grad_student_hidden": self.grad_student_hidden,
            "grad_teacher_hidden": self.grad_teacher_hidden,
            "grad_weight_accum_dtype": self.grad_weight_accum_dtype,
            "grad_teacher_weight_accum_dtype": (self.grad_teacher_weight_accum_dtype),
            "gradient_materialization_scaled": self.gradient_materialization_scaled,
            "chunk_rows": self.chunk_rows,
            "chunk_dynamic": self.chunk_dynamic,
        }

    def disk_key(self) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
