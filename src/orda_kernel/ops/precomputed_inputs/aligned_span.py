"""Span-aligned cross-vocabulary distillation (ALM-style).

Student and teacher tokenize the same text differently, so per-column KL is
meaningless across vocabularies. Instead, each aligned span compares the
likelihood the two models assign to the observed continuation:

    l_s(c) = sum_{i in c_s} log p_s(y_i)      l_t(c) = sum_{j in c_t} log p_t(y_j)

The per-span event space is binary — the observed span versus any other
continuation — and the objective is binary forward KL (or binary
cross-entropy) between those two likelihoods. The teacher side is always
detached; the student gradient per token row is ``g_c * (onehot(y_i) - p_i)``,
which is why this objective can later reuse the fused CE/GEMM pipeline.

``CrossVocabTeacher`` uses the torch reference. ``PrecomputedSpanTeacher`` can
use a chunked two-stage Triton pipeline without changing the objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..._api_types import (
    AlignedLikelihoodConfig,
    AlignedSpanPairs,
    Backend,
    CrossVocabTeacher,
    DistillationLossOutput,
    KernelConfig,
    PrecomputedSpanTeacher,
    Profile,
)
from ..._config import _resolve_config
from ..._planning.policy import build_aligned_span_execution_plan
from ..._runtime import (
    HAS_TRITON,
    resolve_gradient_materialization_scale,
    resolve_uniform_model_dtype,
    triton_dtype_is_supported,
    triton_unavailable_reason,
)
from ..._torch_backend import (
    _combine_weighted_losses,
    _loss_output_dtype,
)
from ..._validation import _validate_gradient_materialization_scale
from ._aligned_span_math import (
    binary_span_objective,
    span_log_probs_from_nll,
    span_valid_counts,
)


def _span_log_likelihoods(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ranges: torch.Tensor,
    *,
    ignore_index: int,
) -> torch.Tensor:
    """Segment-reduce per-token label log-probabilities into span sums."""
    mask = labels != ignore_index
    # Mask ignored rows before the GEMM so a non-finite hidden row cannot poison
    # log_softmax backward (runs under no_grad, kept for path consistency).
    logits = torch.where(mask.unsqueeze(-1), hidden, hidden.new_zeros(())) @ weight.t()
    invalid_rows = mask & (~torch.isfinite(logits)).any(dim=-1)
    if bool(invalid_rows.any()):
        bad_rows = torch.nonzero(invalid_rows, as_tuple=False).flatten()[:5].tolist()
        raise ValueError(
            "CrossVocabTeacher GEMM produced non-finite logits on active rows "
            f"{bad_rows}"
        )
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    # Ignored rows gather class 0 and are zeroed below; a positive out-of-vocab
    # ignore_index is safe because it never reaches the gather.
    safe_labels = torch.where(mask, labels, labels.new_zeros(()))
    token_ll = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    token_ll = torch.where(mask, token_ll, token_ll.new_zeros(()))
    return span_log_probs_from_nll(-token_ll, ranges)


# ── Validation helpers ───────────────────────────────────────────────────────
def _validate_ranges(name: str, ranges: torch.Tensor, n_rows: int) -> None:
    if ranges.numel() == 0:
        return
    start = ranges[:, 0]
    end = ranges[:, 1]
    if bool((start < 0).any()) or bool((end > n_rows).any()):
        raise ValueError(
            f"{name} must lie within [0, {n_rows}], got bounds "
            f"[{int(start.min())}, {int(end.max())}]"
        )
    if bool((start >= end).any()):
        raise ValueError(f"{name} must satisfy start < end for every span")


def _validate_span_labels(
    name: str,
    labels: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
    *,
    assume_valid_labels: bool = False,
) -> None:
    """Validate label dtype and, unless trusted, the value range."""
    if labels.dtype != torch.long:
        raise ValueError(f"{name} must have dtype torch.long, got {labels.dtype}")
    if assume_valid_labels or labels.numel() == 0:
        return
    valid = (labels == ignore_index) | ((labels >= 0) & (labels < vocab_size))
    if not bool(valid.all()):
        bad_values = labels[~valid][:5].tolist()
        raise ValueError(
            f"{name} contains values outside [0, {vocab_size}) and not equal to "
            f"ignore_index={ignore_index}: {bad_values}"
        )


def _validate_span_teacher(
    teacher: CrossVocabTeacher | PrecomputedSpanTeacher,
    span_pairs: AlignedSpanPairs,
    num_spans: int,
    ignore_index: int,
    *,
    assume_valid_labels: bool = False,
) -> None:
    """Validate structural teacher contracts even when the span term is disabled."""
    if isinstance(teacher, CrossVocabTeacher):
        if teacher.hidden.ndim != 2 or teacher.weight.ndim != 2:
            raise ValueError("CrossVocabTeacher.hidden and .weight must be 2D tensors")
        if teacher.weight.shape[1] != teacher.hidden.shape[1]:
            raise ValueError(
                "CrossVocabTeacher.weight must have shape "
                f"(V_t, {teacher.hidden.shape[1]}), got "
                f"{tuple(teacher.weight.shape)}"
            )
        if teacher.weight.shape[0] < 1:
            raise ValueError(
                "CrossVocabTeacher.weight must contain at least one vocabulary row"
            )
        if (
            teacher.labels.ndim != 1
            or teacher.labels.shape[0] != teacher.hidden.shape[0]
        ):
            raise ValueError(
                "CrossVocabTeacher.labels must have shape "
                f"({teacher.hidden.shape[0]},), got {tuple(teacher.labels.shape)}"
            )
        _validate_ranges(
            "AlignedSpanPairs.teacher_ranges",
            span_pairs.teacher_ranges,
            teacher.hidden.shape[0],
        )
        _validate_span_labels(
            "CrossVocabTeacher.labels",
            teacher.labels,
            teacher.weight.shape[0],
            ignore_index,
            assume_valid_labels=assume_valid_labels,
        )
    else:
        teacher._validate_cache_unchanged()
        if teacher.span_log_probs.shape[0] != num_spans:
            raise ValueError(
                "PrecomputedSpanTeacher.span_log_probs must have "
                f"{num_spans} entries (one per span), got "
                f"{teacher.span_log_probs.shape[0]}"
            )
        if teacher.teacher_ranges is not None:
            cached = teacher.teacher_ranges
            called = span_pairs.teacher_ranges
            if cached.device.type != "meta" and called.device.type != "meta":
                if not torch.equal(
                    cached.to(device=called.device, dtype=torch.int64),
                    called.to(dtype=torch.int64),
                ):
                    raise ValueError(
                        "AlignedSpanPairs.teacher_ranges does not match the "
                        "ranges PrecomputedSpanTeacher was built from; "
                        "rebuild the cache for these spans"
                    )


def _validate_span_model_tensors(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    student_labels: torch.Tensor,
    teacher: CrossVocabTeacher | PrecomputedSpanTeacher,
) -> torch.dtype:
    """Apply the main API's model dtype/device policy before span computation."""
    if student_weight.device != student_hidden.device:
        raise ValueError(
            "student_weight must be on the same device as student_hidden, got "
            f"{student_weight.device} vs {student_hidden.device}"
        )
    if student_labels.device != student_hidden.device:
        raise ValueError(
            "student_labels must be on the same device as student_hidden, got "
            f"{student_labels.device} vs {student_hidden.device}"
        )

    named_tensors: list[tuple[str, torch.Tensor]] = [
        ("student_hidden", student_hidden),
        ("student_weight", student_weight),
    ]
    if isinstance(teacher, CrossVocabTeacher):
        for name, tensor in (
            ("CrossVocabTeacher.hidden", teacher.hidden),
            ("CrossVocabTeacher.weight", teacher.weight),
            ("CrossVocabTeacher.labels", teacher.labels),
        ):
            if tensor.device != student_hidden.device:
                raise ValueError(
                    f"{name} must be on the same device as student_hidden, got "
                    f"{tensor.device} vs {student_hidden.device}"
                )
        named_tensors.extend(
            [
                ("CrossVocabTeacher.hidden", teacher.hidden),
                ("CrossVocabTeacher.weight", teacher.weight),
            ]
        )
    return resolve_uniform_model_dtype(*named_tensors)


def _validate_aligned_span_call(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    student_labels: torch.Tensor,
    teacher: CrossVocabTeacher | PrecomputedSpanTeacher,
    span_pairs: AlignedSpanPairs,
    config: AlignedLikelihoodConfig | None = None,
    *,
    assume_valid_labels: bool = False,
) -> tuple[AlignedLikelihoodConfig, torch.dtype]:
    if config is None:
        config = AlignedLikelihoodConfig()
    if not isinstance(span_pairs, AlignedSpanPairs):
        raise TypeError(
            f"span_pairs must be an AlignedSpanPairs, got {type(span_pairs).__name__}"
        )
    if not isinstance(config, AlignedLikelihoodConfig):
        raise TypeError(
            f"config must be an AlignedLikelihoodConfig, got {type(config).__name__}"
        )
    if not isinstance(teacher, (CrossVocabTeacher, PrecomputedSpanTeacher)):
        raise TypeError(
            "teacher must be CrossVocabTeacher or PrecomputedSpanTeacher, got "
            f"{type(teacher).__name__}"
        )
    if student_hidden.ndim != 2:
        raise ValueError(
            f"student_hidden must be 2D, got {tuple(student_hidden.shape)}"
        )
    if student_weight.ndim != 2 or student_weight.shape[1] != student_hidden.shape[1]:
        raise ValueError(
            f"student_weight must have shape (V_s, {student_hidden.shape[1]}), "
            f"got {tuple(student_weight.shape)}"
        )
    if student_weight.shape[0] < 1:
        raise ValueError("student_weight must contain at least one vocabulary row")
    if student_labels.ndim != 1 or student_labels.shape[0] != student_hidden.shape[0]:
        raise ValueError(
            f"student_labels must have shape ({student_hidden.shape[0]},), "
            f"got {tuple(student_labels.shape)}"
        )
    model_dtype = _validate_span_model_tensors(
        student_hidden,
        student_weight,
        student_labels,
        teacher,
    )
    _validate_span_labels(
        "student_labels",
        student_labels,
        student_weight.shape[0],
        config.ignore_index,
        assume_valid_labels=assume_valid_labels,
    )
    num_spans = span_pairs.student_ranges.shape[0]
    _validate_ranges(
        "AlignedSpanPairs.student_ranges",
        span_pairs.student_ranges,
        student_hidden.shape[0],
    )
    _validate_span_teacher(
        teacher,
        span_pairs,
        num_spans,
        config.ignore_index,
        assume_valid_labels=assume_valid_labels,
    )
    return config, model_dtype


# ── Torch reference ──────────────────────────────────────────────────────────
def _aligned_span_torch_reference(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    student_labels: torch.Tensor,
    teacher: CrossVocabTeacher | PrecomputedSpanTeacher,
    span_pairs: AlignedSpanPairs,
    config: AlignedLikelihoodConfig | None = None,
    *,
    _validated: bool = False,
) -> DistillationLossOutput:
    if _validated:
        assert isinstance(config, AlignedLikelihoodConfig)
    else:
        config, _ = _validate_aligned_span_call(
            student_hidden,
            student_weight,
            student_labels,
            teacher,
            span_pairs,
            config,
        )
    num_spans = span_pairs.student_ranges.shape[0]

    output_dtype = _loss_output_dtype(student_hidden.dtype)
    zero = student_hidden.new_zeros((), dtype=output_dtype)

    # ── Student spans and CE ─────────────────────────────────
    compute_ce = config.student_ce_weight != 0.0
    compute_span = config.span_kl_weight != 0.0 and num_spans > 0
    if student_hidden.shape[0] == 0 or not (compute_ce or compute_span):
        loss = (
            student_hidden.reshape(-1)[:0].sum() + student_weight.reshape(-1)[:0].sum()
        ).to(output_dtype)
        return DistillationLossOutput(loss, zero, zero.clone(), zero.clone())

    mask = student_labels != config.ignore_index
    denominator = torch.clamp(mask.sum(), min=1)

    # Mask ignored rows before the GEMM so a non-finite hidden row cannot poison
    # gradients (log_softmax backward emits NaN even with zero upstream grad).
    logits_s = (
        torch.where(mask.unsqueeze(-1), student_hidden, student_hidden.new_zeros(()))
        @ student_weight.t()
    )
    if logits_s.dtype in (torch.float16, torch.bfloat16):
        logits_s = logits_s.float()

    if compute_ce:
        ce_all = F.cross_entropy(
            logits_s,
            student_labels,
            ignore_index=config.ignore_index,
            reduction="none",
        )
        student_ce = (
            ce_all.sum() / denominator if config.reduction == "mean" else ce_all.sum()
        )
    else:
        student_ce = zero

    if compute_span:
        log_probs_s = F.log_softmax(logits_s, dim=-1)
        # Ignored rows gather class 0 and are zeroed below; a positive
        # out-of-vocab ignore_index is safe (it never reaches gather).
        safe_labels = torch.where(mask, student_labels, student_labels.new_zeros(()))
        token_ll = log_probs_s.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
        token_ll = torch.where(mask, token_ll, token_ll.new_zeros(()))
        student_ranges = span_pairs.student_ranges.to(
            device=token_ll.device, dtype=torch.int64
        )
        ell_s_raw = span_log_probs_from_nll(-token_ll, student_ranges)
        # A span with zero valid student tokens carries no evidence (ell_s -> 0
        # -> clamp -> p_s ~ 1, a spurious "probability one"); drop it from the
        # sum and the mean denominator.
        span_keep = span_valid_counts(mask, student_ranges) > 0

        # ── Detached teacher spans ───────────────────────────
        # Teacher contract already validated upfront by _validate_span_teacher.
        if isinstance(teacher, CrossVocabTeacher):
            with torch.no_grad():
                ell_t = _span_log_likelihoods(
                    teacher.hidden.detach(),
                    teacher.weight.detach(),
                    teacher.labels,
                    span_pairs.teacher_ranges,
                    ignore_index=config.ignore_index,
                )
            # An all-ignored teacher span is likewise evidence-free; drop it.
            # (A precomputed teacher supplies each span's log-prob directly, so
            # -inf is legal there and only the student side gates the span.)
            teacher_mask = teacher.labels != config.ignore_index
            span_keep = span_keep & (
                span_valid_counts(teacher_mask, span_pairs.teacher_ranges) > 0
            )
        else:
            ell_t = teacher.span_log_probs.to(device=ell_s_raw.device)

        # ── Binary span divergence ───────────────────────────
        span_terms = binary_span_objective(
            ell_s_raw,
            ell_t,
            span_keep,
            objective=config.objective,
            reduction=config.reduction,
        )
        span_div = span_terms.reduced_loss
    else:
        span_div = zero

    loss = _combine_weighted_losses(
        student_ce,
        zero,
        span_div,
        student_ce_weight=config.student_ce_weight,
        teacher_ce_weight=0.0,
        kl_weight=config.span_kl_weight,
    ).to(output_dtype)

    return DistillationLossOutput(
        loss,
        student_ce.detach().to(output_dtype),
        zero.clone(),
        span_div.detach().to(output_dtype),
    )


# ── Public entry point ───────────────────────────────────────────────────────
_ALIGNED_FORBIDDEN_KERNEL_FIELDS = (
    "kl_weight",
    "kl_temperature",
    "kl_teacher_temperature",
    "kl_divergence",
    "kl_alpha",
    "kl_beta",
    "entropy_gate",
)


def _validate_aligned_kernel_config(config: KernelConfig | None) -> None:
    if config is None:
        return
    if not isinstance(config, KernelConfig):
        raise TypeError(
            f"kernel_config must be a KernelConfig or None, got {type(config).__name__}"
        )
    incompatible = [
        name
        for name in _ALIGNED_FORBIDDEN_KERNEL_FIELDS
        if getattr(config, name) is not None
    ]
    if incompatible:
        raise ValueError(
            "aligned-span execution does not accept KL-specific KernelConfig "
            f"fields: {incompatible}"
        )


@torch._dynamo.disable
def aligned_span_distillation_loss(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    student_labels: torch.Tensor,
    teacher: CrossVocabTeacher | PrecomputedSpanTeacher,
    span_pairs: AlignedSpanPairs,
    config: AlignedLikelihoodConfig | None = None,
    *,
    profile: Profile = "balanced",
    backend: Backend = "auto",
    kernel_config: KernelConfig | None = None,
    gradient_materialization_scale: torch.Tensor | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> DistillationLossOutput:
    """Distill across vocabularies by matching aligned span likelihoods."""
    gradient_materialization_scale = resolve_gradient_materialization_scale(
        student_hidden,
        gradient_materialization_scale,
        grad_scaler,
    )
    if backend not in ("auto", "triton", "torch"):
        raise ValueError(
            f"backend must be 'auto', 'triton', or 'torch', got {backend!r}"
        )
    if backend == "triton" and isinstance(teacher, CrossVocabTeacher):
        raise ValueError(
            "backend='triton' aligned-span execution requires "
            "PrecomputedSpanTeacher; materialize the detached teacher span "
            "cache first"
        )
    _validate_aligned_kernel_config(kernel_config)
    resolved = _resolve_config(profile, kernel_config)
    config, model_dtype = _validate_aligned_span_call(
        student_hidden,
        student_weight,
        student_labels,
        teacher,
        span_pairs,
        config,
        assume_valid_labels=resolved.effective_assume_valid_labels,
    )
    _validate_gradient_materialization_scale(
        gradient_materialization_scale,
        student_hidden.device,
    )

    supports_triton = bool(
        HAS_TRITON
        and student_hidden.device.type == "cuda"
        and triton_dtype_is_supported(model_dtype, student_hidden.device)
    )
    use_triton = (
        backend != "torch"
        and isinstance(teacher, PrecomputedSpanTeacher)
        and supports_triton
    )
    if backend == "triton" and not use_triton:
        reason = (
            triton_unavailable_reason()
            if not HAS_TRITON
            else "unsupported dtype or device"
        )
        raise RuntimeError(
            "backend='triton' aligned-span execution requires Triton and "
            f"CUDA/HIP tensors with a supported compute dtype ({reason})"
        )
    if (
        backend == "auto"
        and isinstance(teacher, PrecomputedSpanTeacher)
        and not supports_triton
        and student_hidden.shape[0] != 0
    ):
        from ...api import _warn_auto_torch_fallback

        _warn_auto_torch_fallback(
            triton_unavailable_reason()
            if not HAS_TRITON
            else "aligned-span Triton unsupported dtype/device",
            operation="aligned_span_distillation_loss",
        )
    if not use_triton:
        return _aligned_span_torch_reference(
            student_hidden,
            student_weight,
            student_labels,
            teacher,
            span_pairs,
            config,
            _validated=True,
        )

    if (
        resolved.max_fused_size is None
        or resolved.autotune is None
        or resolved.chunk_size is None
    ):
        raise RuntimeError("profile resolution produced an incomplete kernel config")
    plan = build_aligned_span_execution_plan(
        backend="triton",
        teacher_type="precomputed_span",
        compute_dtype=model_dtype,
        bt=student_hidden.shape[0],
        vocab_size=student_weight.shape[0],
        num_spans=span_pairs.student_ranges.shape[0],
        student_ce_weight=config.student_ce_weight,
        span_kl_weight=config.span_kl_weight,
        ignore_index=config.ignore_index,
        reduction=config.reduction,
        objective=config.objective,
        chunk_size=resolved.chunk_size,
        num_chunks=resolved.num_chunks,
        max_chunks=resolved.max_chunks,
        max_fused_size=resolved.max_fused_size,
        autotune=resolved.autotune,
        refresh_autotune=bool(resolved.refresh_autotune),
        grad_weight_accumulation=resolved.effective_grad_weight_accumulation,
        assume_valid_labels=resolved.effective_assume_valid_labels,
    )
    from ..._execution.aligned_span import execute_aligned_span_plan

    assert isinstance(teacher, PrecomputedSpanTeacher)
    loss, student_ce, teacher_ce, span_div = execute_aligned_span_plan(
        plan,
        student_hidden,
        student_weight,
        student_labels,
        teacher.span_log_probs,
        span_pairs.student_ranges,
        gradient_materialization_scale=gradient_materialization_scale,
        _labels_prevalidated=True,
    )
    return DistillationLossOutput(loss, student_ce, teacher_ce, span_div)
