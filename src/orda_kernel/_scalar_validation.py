from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch


@dataclass(frozen=True)
class ResolvedKLTemperatures:
    student: float
    teacher: float


def _normalize_temperature(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.0, got {value!r}")
    return normalized


def _is_normal_float32(value: float) -> bool:
    fp32 = torch.finfo(torch.float32)
    return math.isfinite(value) and fp32.tiny <= abs(value) <= fp32.max


def resolve_kl_temperatures(
    kl_temperature,
    kl_teacher_temperature,
) -> ResolvedKLTemperatures:
    student = _normalize_temperature("kl_temperature", kl_temperature)
    teacher = _normalize_temperature(
        "kl_teacher_temperature",
        student if kl_teacher_temperature is None else kl_teacher_temperature,
    )
    try:
        student_inverse = 1.0 / student
        student_squared = student * student
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "kl_temperature must keep 1 / T and T**2 representable as finite "
            f"normal float32 values, got {student}"
        ) from exc
    if not all(
        _is_normal_float32(value) for value in (student_inverse, student_squared)
    ):
        raise ValueError(
            "kl_temperature must keep 1 / T and T**2 representable as finite "
            f"normal float32 values, got {student}"
        )
    try:
        teacher_inverse = 1.0 / teacher
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "kl_teacher_temperature must keep 1 / T_t representable as a "
            f"finite normal float32 value, got {teacher}"
        ) from exc
    if not _is_normal_float32(teacher_inverse):
        raise ValueError(
            "kl_teacher_temperature must keep 1 / T_t representable as a "
            f"finite normal float32 value, got {teacher}"
        )
    return ResolvedKLTemperatures(student=student, teacher=teacher)
