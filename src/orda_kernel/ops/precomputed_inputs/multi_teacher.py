"""Detached multi-teacher distribution utilities for cached teachers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F


# ── Input validation ─────────────────────────────────────────────────────────
def _row_indices(mask: torch.Tensor) -> list[int]:
    return torch.nonzero(mask, as_tuple=False).flatten()[:5].tolist()


def _validate_teacher_logits(
    logits_list: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], torch.dtype, torch.device]:
    if not isinstance(logits_list, Sequence) or len(logits_list) == 0:
        raise ValueError("logits_list must contain at least one teacher tensor")

    first = logits_list[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 2:
        raise ValueError("each teacher logits tensor must be a 2D torch.Tensor")
    if not first.is_floating_point():
        raise ValueError(f"teacher logits must use a floating dtype, got {first.dtype}")

    shape = first.shape
    dtype = first.dtype
    device = first.device
    teachers: list[torch.Tensor] = []
    for index, logits in enumerate(logits_list):
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError(
                f"logits_list[{index}] must be a 2D torch.Tensor, got "
                f"{getattr(logits, 'shape', logits)!r}"
            )
        if logits.shape != shape:
            raise ValueError(
                f"all teacher logits must have the same shape; logits_list[0] "
                f"has {tuple(shape)}, logits_list[{index}] has {tuple(logits.shape)}"
            )
        if logits.device != device:
            raise ValueError(
                "all teacher logits must be on the same device; "
                f"logits_list[0]={device}, logits_list[{index}]={logits.device}"
            )
        if logits.dtype != dtype:
            raise ValueError(
                "all teacher logits must have the same dtype; "
                f"logits_list[0]={dtype}, logits_list[{index}]={logits.dtype}"
            )
        if bool(torch.isnan(logits).any()):
            raise ValueError(f"logits_list[{index}] contains NaN")
        if bool(torch.isposinf(logits).any()):
            raise ValueError(f"logits_list[{index}] contains +inf")

        dead_rows = ~torch.isfinite(logits).any(dim=-1)
        if bool(dead_rows.any()):
            raise ValueError(
                f"logits_list[{index}] has all-dead rows with no finite logit: "
                f"{_row_indices(dead_rows)}"
            )
        teachers.append(logits.detach())
    return tuple(teachers), dtype, device


def _normalized_weights(
    weights: Sequence[float] | torch.Tensor,
    *,
    teacher_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        raw_weights = (
            weights.to(device=device)
            if isinstance(weights, torch.Tensor)
            else torch.as_tensor(weights, device=device, dtype=torch.float64)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("weights must be a 1D numeric tensor or sequence") from exc
    if raw_weights.ndim != 1 or raw_weights.numel() != teacher_count:
        raise ValueError(
            f"weights must have exactly {teacher_count} entries, got "
            f"shape={tuple(raw_weights.shape)}"
        )
    if torch.is_complex(raw_weights) or raw_weights.dtype == torch.bool:
        raise ValueError("weights must be real numeric values")

    # Validate in fp64 so extreme finite Python weights survive parsing.
    raw_weights = raw_weights.detach().to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(raw_weights).all()):
        raise ValueError("weights must be finite")
    if not bool((raw_weights > 0).all()):
        raise ValueError("weights must be strictly > 0")

    # Log-space normalization avoids overflow for imbalanced weights.
    log_weights = raw_weights.log()
    normalized_log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
    return normalized_log_weights.to(dtype), normalized_log_weights.exp().to(dtype)


def _require_representable_normalized_weights(
    normalized_weights: torch.Tensor,
) -> None:
    if bool((normalized_weights == 0).any()):
        raise ValueError(
            "normalized weights must remain representable in the teacher compute dtype"
        )


def _prepare_teacher_log_probs(
    logits_list: Sequence[torch.Tensor],
    weights: Sequence[float] | torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    teachers, input_dtype, device = _validate_teacher_logits(logits_list)
    compute_dtype = torch.float64 if input_dtype == torch.float64 else torch.float32
    with torch.no_grad():
        log_probs = tuple(
            F.log_softmax(logits, dim=-1, dtype=compute_dtype) for logits in teachers
        )
        log_weights, normalized_weights = _normalized_weights(
            weights,
            teacher_count=len(teachers),
            device=device,
            dtype=compute_dtype,
        )
    return log_probs, log_weights, normalized_weights


# ── Teacher mixing ───────────────────────────────────────────────────────────
def _arithmetic_log_probs(
    log_probs: tuple[torch.Tensor, ...], log_weights: torch.Tensor
) -> torch.Tensor:
    stacked = torch.stack(log_probs, dim=0)
    return torch.logsumexp(stacked + log_weights[:, None, None], dim=0)


def _geometric_log_probs(
    log_probs: tuple[torch.Tensor, ...], normalized_weights: torch.Tensor
) -> torch.Tensor:
    _require_representable_normalized_weights(normalized_weights)
    stacked = torch.stack(log_probs, dim=0)
    zero_support = torch.isneginf(stacked).any(dim=0)
    safe_log_probs = torch.where(
        torch.isneginf(stacked), torch.zeros_like(stacked), stacked
    )
    mixed = (normalized_weights[:, None, None] * safe_log_probs).sum(dim=0)
    mixed = torch.where(zero_support, torch.full_like(mixed, float("-inf")), mixed)

    dead_rows = ~torch.isfinite(mixed).any(dim=-1)
    if bool(dead_rows.any()):
        raise ValueError(
            "geometric mixing has no common teacher support for rows: "
            f"{_row_indices(dead_rows)}"
        )
    return mixed


def _mix_log_probs(
    log_probs: tuple[torch.Tensor, ...],
    log_weights: torch.Tensor,
    normalized_weights: torch.Tensor,
    mode: Literal["arithmetic", "geometric"],
) -> torch.Tensor:
    if mode == "arithmetic":
        return _arithmetic_log_probs(log_probs, log_weights)
    if mode == "geometric":
        return _geometric_log_probs(log_probs, normalized_weights)
    raise ValueError(f"mode must be 'arithmetic' or 'geometric', got {mode!r}")


# ── Public utilities ─────────────────────────────────────────────────────────
def mix_teacher_logits(
    logits_list: Sequence[torch.Tensor],
    weights: Sequence[float] | torch.Tensor,
    *,
    mode: Literal["arithmetic", "geometric"] = "arithmetic",
) -> torch.Tensor:
    """Mix teacher distributions into detached cached ``[N, V]`` logits.

    Arithmetic mixing returns normalized log-probabilities. Geometric mixing is
    a product-of-experts log-score; any zero-probability teacher support stays
    zero, so every row must retain at least one common supported token.
    """
    log_probs, log_weights, normalized_weights = _prepare_teacher_log_probs(
        logits_list, weights
    )
    return _mix_log_probs(log_probs, log_weights, normalized_weights, mode).detach()


def generalized_jensen_shannon(
    logits_list: Sequence[torch.Tensor],
    weights: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Return the detached generalized Jensen-Shannon diagnostic for each row."""
    log_probs, log_weights, normalized_weights = _prepare_teacher_log_probs(
        logits_list, weights
    )
    _require_representable_normalized_weights(normalized_weights)
    mixture = _arithmetic_log_probs(log_probs, log_weights)

    kl_rows = []
    for log_prob in log_probs:
        probs = log_prob.exp()
        supported = probs > 0
        log_ratio = torch.where(
            supported, log_prob - mixture, torch.zeros_like(log_prob)
        )
        kl_terms = torch.where(supported, probs * log_ratio, torch.zeros_like(log_prob))
        kl_rows.append(kl_terms.sum(dim=-1))
    diagnostic = (normalized_weights[:, None] * torch.stack(kl_rows, dim=0)).sum(dim=0)
    return diagnostic.clamp_min(0.0).detach()
