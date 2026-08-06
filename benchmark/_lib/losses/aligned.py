from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..core.device import sync
from ..core.validation import validate_positive_int

# log1p(-exp(0)) is -inf, which would poison the binary terms.
MAX_SPAN_LOG_PROB = -1.0e-7
_LOG_2 = math.log(2.0)


# ── Span setup ───────────────────────────────────────────────────────────────
def cross_vocab_teacher_vocab_size(student_vocab_size: int) -> int:
    return student_vocab_size + 1


def aligned_span_ranges(
    n_rows: int, device: torch.device, span_length: int = 4
) -> torch.Tensor:
    validate_positive_int("n_rows", n_rows)
    validate_positive_int("span_length", span_length)
    starts = torch.arange(0, n_rows, span_length, device=device, dtype=torch.int64)
    ends = (starts + span_length).clamp_max(n_rows)
    return torch.stack((starts, ends), dim=-1)


def build_cross_vocab_span_cache(
    n_student_rows: int,
    student_vocab_size: int,
    teacher_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    student_span_length: int = 4,
    teacher_span_lengths: tuple[int, ...] = (3, 4, 5),
):
    if not teacher_span_lengths:
        raise ValueError("teacher_span_lengths must not be empty")
    for length in teacher_span_lengths:
        validate_positive_int("teacher_span_length", length)
    from orda_kernel import build_precomputed_span_teacher

    student_ranges = aligned_span_ranges(n_student_rows, device, student_span_length)
    span_count = student_ranges.shape[0]
    span_index = torch.arange(span_count, device=device, dtype=torch.int64)
    teacher_length_options = torch.tensor(
        teacher_span_lengths, device=device, dtype=torch.int64
    )
    teacher_lengths = teacher_length_options[
        span_index.remainder(len(teacher_span_lengths))
    ]
    teacher_ends = teacher_lengths.cumsum(0)
    teacher_starts = torch.cat((teacher_ends.new_zeros(1), teacher_ends[:-1]))
    teacher_ranges = torch.stack((teacher_starts, teacher_ends), dim=-1)
    full_cycles, remainder = divmod(span_count, len(teacher_span_lengths))
    n_teacher_rows = full_cycles * sum(teacher_span_lengths) + sum(
        teacher_span_lengths[:remainder]
    )
    teacher_vocab_size = cross_vocab_teacher_vocab_size(student_vocab_size)
    teacher_hidden = (
        torch.randn(
            n_teacher_rows,
            teacher_dim,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    ).to(dtype)
    teacher_weight = (
        torch.randn(
            teacher_vocab_size,
            teacher_dim,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    ).to(dtype)
    teacher_labels = torch.randint(
        0,
        teacher_vocab_size,
        (n_teacher_rows,),
        device=device,
    )
    teacher = build_precomputed_span_teacher(
        teacher_hidden,
        teacher_weight,
        teacher_labels,
        teacher_ranges,
    )
    return student_ranges, teacher_ranges, teacher


def _log1mexp(value: torch.Tensor) -> torch.Tensor:
    lower = value < -_LOG_2
    return torch.where(
        lower,
        torch.log1p(-torch.exp(value.clamp(max=-_LOG_2))),
        torch.log(-torch.expm1(value.clamp(min=-_LOG_2, max=MAX_SPAN_LOG_PROB))),
    )


# ── Span validation ──────────────────────────────────────────────────────────
def validate_aligned_component_inputs(
    n_rows: int,
    labels: torch.Tensor,
    ranges: torch.Tensor,
    teacher_span_log_probs: torch.Tensor,
):
    if labels.ndim != 1 or labels.shape[0] != n_rows:
        raise ValueError(
            f"labels must have shape [{n_rows}], got {tuple(labels.shape)}"
        )
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(
            f"ranges must have shape [spans, 2], got {tuple(ranges.shape)}"
        )
    if teacher_span_log_probs.numel() != ranges.shape[0]:
        raise ValueError(
            "teacher span count must match student ranges: "
            f"{teacher_span_log_probs.numel()} != {ranges.shape[0]}"
        )
    ranges64 = ranges.to(dtype=torch.int64)
    if bool(
        (
            (ranges64[:, 0] < 0)
            | (ranges64[:, 0] >= ranges64[:, 1])
            | (ranges64[:, 1] > n_rows)
        )
        .any()
        .item()
    ):
        raise ValueError("ranges must satisfy 0 <= start < end <= number of rows")


# ── Span objective ───────────────────────────────────────────────────────────
def aligned_components(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ranges: torch.Tensor,
    teacher_span_log_probs: torch.Tensor,
    objective: str,
    *,
    include_ce: bool = True,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [rows, vocab], got {tuple(logits.shape)}"
        )
    if validate:
        validate_aligned_component_inputs(
            logits.shape[0], labels, ranges, teacher_span_log_probs
        )
    ranges64 = ranges.to(device=logits.device, dtype=torch.int64)
    logits_f = (
        logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    )
    ce = F.cross_entropy(logits_f, labels) if include_ce else logits_f.new_zeros(())
    log_probs = F.log_softmax(logits_f, dim=-1)
    token_nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    token_nll = token_nll.to(torch.float64)
    prefix = torch.cat((token_nll.new_zeros(1), token_nll.cumsum(0)))
    ell_s = -(prefix[ranges64[:, 1]] - prefix[ranges64[:, 0]])
    ell_s = ell_s.clamp(max=MAX_SPAN_LOG_PROB)
    ell_t = teacher_span_log_probs.to(device=ell_s.device, dtype=torch.float64)
    ell_t = ell_t.clamp(max=MAX_SPAN_LOG_PROB).detach()
    p_t = ell_t.exp()
    if objective == "binary_kl":
        span_rows = p_t * (ell_t - ell_s) + (1.0 - p_t) * (
            _log1mexp(ell_t) - _log1mexp(ell_s)
        )
    elif objective == "binary_ce":
        span_rows = -(p_t * ell_s + (1.0 - p_t) * _log1mexp(ell_s))
    else:
        raise ValueError(f"Unknown aligned-span objective: {objective}")
    span = span_rows.mean()
    return ce + span, ce, span


def aligned_inputs(
    n_rows: int,
    vocab: int,
    dim: int,
    *,
    teacher_dim: int | None = None,
    dtype: torch.dtype,
    device: torch.device,
    hidden_requires_grad: bool = False,
    hidden_scale: float = 1.0,
    weight_scale: float = 1.0,
    student_span_length: int = 4,
    teacher_span_lengths: tuple[int, ...] = (3, 4, 5),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Match scale and differentiability to the accuracy versus train workload."""
    hidden = torch.randn(n_rows, dim, device=device, dtype=dtype) * hidden_scale
    weight = torch.randn(vocab, dim, device=device, dtype=dtype) * weight_scale
    if hidden_requires_grad:
        hidden.requires_grad_(True)
    weight.requires_grad_(True)
    labels = torch.randint(0, vocab, (n_rows,), device=device)
    student_ranges, teacher_ranges, teacher = build_cross_vocab_span_cache(
        n_rows,
        vocab,
        dim if teacher_dim is None else teacher_dim,
        dtype=dtype,
        device=device,
        student_span_length=student_span_length,
        teacher_span_lengths=teacher_span_lengths,
    )
    return (
        hidden,
        weight,
        labels,
        student_ranges,
        teacher_ranges,
        teacher.span_log_probs,
    )


# ── Loss construction ────────────────────────────────────────────────────────
def make_aligned_loss_fn(
    objective: str,
    mode: str,
    *,
    include_ce: bool = True,
    num_chunks: int | None = None,
    autotune: bool | str | None = None,
    gradient_materialization_scale: torch.Tensor | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
):
    if mode in ("torch", "torch-compile"):

        def raw_loss_fn(
            hidden,
            weight,
            labels,
            student_ranges,
            teacher_ranges,
            teacher_span_log_probs,
        ):
            logits = F.linear(hidden, weight)
            return aligned_components(
                logits,
                labels,
                student_ranges,
                teacher_span_log_probs,
                objective,
                include_ce=include_ce,
                validate=False,
            )

        implementation = (
            torch.compile(raw_loss_fn, mode="default")
            if mode == "torch-compile"
            else raw_loss_fn
        )
        validated = False

        def loss_fn(
            hidden,
            weight,
            labels,
            student_ranges,
            teacher_ranges,
            teacher_span_log_probs,
        ):
            nonlocal validated
            if not validated:
                validate_aligned_component_inputs(
                    hidden.shape[0],
                    labels,
                    student_ranges,
                    teacher_span_log_probs,
                )
                validated = True
            return implementation(
                hidden,
                weight,
                labels,
                student_ranges,
                teacher_ranges,
                teacher_span_log_probs,
            )

        return loss_fn

    from orda_kernel import (
        AlignedLikelihoodConfig,
        AlignedSpanPairs,
        KernelConfig,
        PrecomputedSpanTeacher,
        aligned_span_distillation_loss,
    )

    config = AlignedLikelihoodConfig(
        student_ce_weight=1.0 if include_ce else 0.0,
        span_kl_weight=1.0,
        objective=objective,
    )
    kernel_config = (
        KernelConfig(num_chunks=num_chunks, autotune=autotune)
        if num_chunks is not None or autotune is not None
        else None
    )

    def loss_fn(
        hidden,
        weight,
        labels,
        student_ranges,
        teacher_ranges,
        teacher_span_log_probs,
    ):
        out = aligned_span_distillation_loss(
            hidden,
            weight,
            labels,
            PrecomputedSpanTeacher(
                teacher_span_log_probs,
                teacher_ranges=teacher_ranges,
            ),
            AlignedSpanPairs(student_ranges, teacher_ranges),
            config,
            backend="triton",
            kernel_config=kernel_config,
            gradient_materialization_scale=gradient_materialization_scale,
            grad_scaler=grad_scaler,
        )
        return out.loss, out.student_ce, out.kl

    return loss_fn


# ── Benchmark runner ─────────────────────────────────────────────────────────
def run_aligned_loss_only(
    loss_fn,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    student_ranges: torch.Tensor,
    teacher_ranges: torch.Tensor,
    teacher_span_log_probs: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> None:
    hidden_loss = hidden.detach().requires_grad_(True)
    weight.grad = None
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        loss, _, _ = loss_fn(
            hidden_loss,
            weight,
            labels,
            student_ranges,
            teacher_ranges,
            teacher_span_log_probs,
        )
    loss.backward()
    sync(device)
