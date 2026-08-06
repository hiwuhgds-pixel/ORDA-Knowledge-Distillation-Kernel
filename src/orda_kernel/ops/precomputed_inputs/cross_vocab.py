"""Cross-vocabulary token mapping onto the precomputed-teacher path.

These helpers project a teacher distribution over its own vocabulary onto the
student vocabulary through an exact string match, producing logits that feed
directly into :class:`PrecomputedTeacher`. Tokens without an exact match keep
zero probability on the student side; the kernel's own softmax renormalizes
whatever mass survives the projection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import torch
import torch.nn.functional as F

_ZERO_SUPPORT_LOGIT = float("-inf")


# ── Vocabulary mapping ───────────────────────────────────────────────────────
def build_exact_token_map(
    student_vocab: Sequence[str],
    teacher_vocab: Sequence[str],
) -> torch.Tensor:
    """Map each teacher token index to its exact-match student index.

    Returns an int64 tensor of shape ``[len(teacher_vocab)]`` holding the
    student index of the identical token string, or ``-1`` when the student
    vocabulary has no such token. Duplicate strings resolve to the first
    occurrence on the student side.
    """
    student_index: dict[str, int] = {}
    for index, token in enumerate(student_vocab):
        if not isinstance(token, str):
            raise TypeError(
                f"student_vocab[{index}] must be a str, got {type(token).__name__}"
            )
        student_index.setdefault(token, index)
    mapped = []
    for index, token in enumerate(teacher_vocab):
        if not isinstance(token, str):
            raise TypeError(
                f"teacher_vocab[{index}] must be a str, got {type(token).__name__}"
            )
        mapped.append(student_index.get(token, -1))
    return torch.tensor(mapped, dtype=torch.int64)


# ── Distribution projection ──────────────────────────────────────────────────
def project_teacher_logits(
    teacher_logits: torch.Tensor,
    token_map: torch.Tensor,
    student_vocab_size: int,
    *,
    unmatched: Literal["renormalize", "mask"] = "renormalize",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Project ``[N, V_t]`` teacher logits onto ``[N, V_s]`` student columns.

    Matched teacher probability mass lands on the mapped student columns
    (mass from duplicate teacher tokens is summed); unmatched student columns
    receive ``-inf``, preserving exact zero support under every temperature.

    ``temperature`` divides the raw teacher logits before any mass is
    aggregated, so duplicate-token mass is summed at that temperature.
    Temperature does not commute with many-to-one aggregation, so the output
    is bound to this temperature; declare it via
    ``PrecomputedTeacher(logits=..., logits_temperature=temperature)`` so the
    consuming loss does not apply a teacher temperature a second time.

    ``unmatched="renormalize"`` returns proper log-probabilities (matched mass
    rescaled to one); ``unmatched="mask"`` keeps the absolute matched log-mass.
    Both feed :class:`PrecomputedTeacher` identically because the kernel
    softmaxes its cached logits.

    Always grad-free: the input is detached so the output feeds
    :class:`PrecomputedTeacher` (which rejects grad tensors).
    """
    # ── Input validation ─────────────────────────────────────
    if not isinstance(teacher_logits, torch.Tensor) or teacher_logits.ndim != 2:
        raise ValueError(
            "teacher_logits must be a 2D torch.Tensor, got "
            f"{getattr(teacher_logits, 'shape', teacher_logits)!r}"
        )
    if not teacher_logits.is_floating_point():
        raise ValueError(f"teacher_logits must be floating, got {teacher_logits.dtype}")
    # Detach so a requires_grad input still yields a grad-free output.
    teacher_logits = teacher_logits.detach()
    if not isinstance(token_map, torch.Tensor) or token_map.ndim != 1:
        raise ValueError("token_map must be a 1D torch.Tensor")
    if token_map.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"token_map must be int32 or int64, got {token_map.dtype}")
    if token_map.shape[0] != teacher_logits.shape[1]:
        raise ValueError(
            f"token_map must have {teacher_logits.shape[1]} entries "
            f"(one per teacher token), got {token_map.shape[0]}"
        )
    if unmatched not in ("renormalize", "mask"):
        raise ValueError(
            f"unmatched must be 'renormalize' or 'mask', got {unmatched!r}"
        )
    if isinstance(student_vocab_size, bool) or not isinstance(
        student_vocab_size, Integral
    ):
        raise TypeError(
            f"student_vocab_size must be an integer >= 1, got {student_vocab_size!r}"
        )
    if student_vocab_size < 1:
        raise ValueError(f"student_vocab_size must be >= 1, got {student_vocab_size}")
    student_vocab_size = int(student_vocab_size)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError(f"temperature must be a float, got {temperature!r}")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")
    token_map = token_map.to(device=teacher_logits.device, dtype=torch.int64)
    below_sentinel = token_map < -1
    if bool(below_sentinel.any()):
        bad_indices = torch.nonzero(below_sentinel, as_tuple=False).flatten()
        raise ValueError(
            "token_map may only use -1 as the unmatched sentinel; got values "
            f"below -1 at indices {bad_indices[:5].tolist()}: "
            f"{token_map[below_sentinel][:5].tolist()}"
        )
    matched = token_map >= 0
    if not bool(matched.any()):
        raise ValueError("token_map matches no student token")
    matched_targets = token_map[matched]
    if int(matched_targets.max()) >= student_vocab_size:
        raise ValueError(
            "token_map contains a student index >= "
            f"student_vocab_size={student_vocab_size}"
        )
    # NaN/+inf are invalid; -inf denotes zero support. Each row still needs
    # finite mass on at least one matched column.
    if bool(torch.isnan(teacher_logits).any()):
        raise ValueError("teacher_logits contains NaN")
    if bool((teacher_logits == float("inf")).any()):
        raise ValueError("teacher_logits contains +inf (-inf is allowed)")
    matched_row_max = teacher_logits[:, matched].amax(dim=-1)
    empty_rows = matched_row_max == float("-inf")
    if bool(empty_rows.any()):
        bad_rows = torch.nonzero(empty_rows, as_tuple=False).flatten()
        raise ValueError(
            "every row of teacher_logits needs at least one finite logit on a "
            "matched teacher column; rows with zero matched mass: "
            f"{bad_rows[:5].tolist()}"
        )

    # ── Projection semantics ─────────────────────────────────
    compute_dtype = (
        torch.float32
        if teacher_logits.dtype in (torch.float16, torch.bfloat16)
        else teacher_logits.dtype
    )
    projection_dtype = compute_dtype
    if (
        temperature != 1.0
        and float(torch.tensor(temperature, dtype=compute_dtype)) == 0.0
    ):
        # Keep a positive Python scalar from becoming a zero tensor divisor.
        projection_dtype = torch.float64
    logits = teacher_logits.to(projection_dtype)
    source_finite = torch.isfinite(logits)
    row_max = logits.amax(dim=-1, keepdim=True)
    center = torch.where(torch.isneginf(row_max), torch.zeros_like(row_max), row_max)
    logits = logits - center
    if temperature != 1.0:
        logits = logits / temperature
    logits = torch.where(
        source_finite,
        logits.clamp_min(torch.finfo(projection_dtype).min),
        logits,
    )
    if unmatched == "renormalize":
        log_p_matched = F.log_softmax(logits[:, matched], dim=-1)
    else:
        log_p_matched = F.log_softmax(logits, dim=-1)[:, matched]

    # Group duplicate teacher tokens without leaving log space.
    n_rows = teacher_logits.shape[0]
    idx = matched_targets.unsqueeze(0).expand(n_rows, -1)  # [N, M]
    # A duplicate group with only -inf remains exact zero support.
    group_amax = torch.full(
        (n_rows, student_vocab_size),
        float("-inf"),
        dtype=projection_dtype,
        device=teacher_logits.device,
    ).scatter_reduce(1, idx, log_p_matched, reduce="amax", include_self=True)
    # A finite shift keeps all-(-inf) groups at zero instead of NaN.
    amax_per_col = group_amax.gather(1, idx)
    safe_amax = torch.where(
        torch.isfinite(amax_per_col), amax_per_col, torch.zeros_like(amax_per_col)
    )
    sum_exp = torch.zeros(
        (n_rows, student_vocab_size),
        dtype=projection_dtype,
        device=teacher_logits.device,
    )
    sum_exp.index_add_(1, matched_targets, (log_p_matched - safe_amax).exp())
    valid = torch.isfinite(group_amax)  # reached columns with finite matched mass
    projected_log = torch.where(valid, sum_exp, torch.ones_like(sum_exp)).log()
    projected_log = projected_log + torch.where(
        valid, group_amax, torch.zeros_like(group_amax)
    )
    result = torch.where(
        valid,
        projected_log,
        torch.full_like(projected_log, _ZERO_SUPPORT_LOGIT),
    )
    return result.to(compute_dtype)
