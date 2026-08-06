from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.autograd.function import once_differentiable

from ...._autotune import (
    BenchmarkTask,
    FullBatchLaunchRequest,
    LaunchConfig,
    default_config,
    full_batch_benchmark_task,
    select_full_batch_launch_plan,
)
from ...._constants import DEFAULT_MAX_FUSED_SIZE
from ...._execution.recovery import NonChunkableWorkspaceOOM, is_oom_error
from ...._planning.models import (
    PRECOMPUTED,
    PRECOMPUTED_ALIGNED_SPAN,
    TEACHER_GRAD_NONE,
    AlignedSpanExecutionPlan,
)
from ...._runtime import HAS_TRITON, resolve_grad_weight_accum_dtype
from ...._torch_backend import _loss_output_dtype
from ...._validation import _validate_gradient_materialization_scale
from ..._common import (
    accumulate_weight_gradient_,
    cast_for_compute,
    claim_cached_gradients,
    grad_accum_dtype_key,
    input_gradient_tuple,
    kernel_launch_guard,
    prepare_gradient_materialization_factor,
    scale_cached_gradient_,
)
from ...precomputed_inputs._aligned_span_math import (
    aligned_row_coefficients,
    binary_span_objective,
    span_log_probs_from_nll,
    span_valid_counts,
)
from ..ce_kl_fwdbwd import (
    KERNEL_STAGE_ALIGNED_GRAD,
    KERNEL_STAGE_ALIGNED_STATS,
    KL_FORWARD,
    ce_kl_fwdbwd_kernel,
    compute_autocast_context,
)


@contextmanager
def _fixed_workspace_guard(name: str):
    try:
        yield
    except Exception as exc:
        if is_oom_error(exc):
            raise NonChunkableWorkspaceOOM(
                f"aligned-span non-chunkable workspace {name!r} allocation failed"
            ) from exc
        raise


# ── Stage launches ───────────────────────────────────────────────────────────
def _aligned_span_stats_fwdbwd(
    logits_chunk: torch.Tensor,
    labels_chunk: torch.Tensor,
    nll_out: torch.Tensor,
    lse_out: torch.Tensor,
    *,
    n_rows: int | None = None,
    ignore_index: int = -100,
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    block_size: int | None = None,
    num_warps: int | None = None,
    assume_valid: bool = False,
) -> None:
    if not HAS_TRITON:
        raise RuntimeError(
            "_aligned_span_stats_fwdbwd requires Triton and CUDA/HIP tensors"
        )
    if not assume_valid:
        _validate_stage_inputs(
            logits_chunk,
            labels_chunk,
            nll_out,
            lse_out,
            n_rows=n_rows,
        )
    rows, vocab_size = logits_chunk.shape
    n_rows = rows if n_rows is None else int(n_rows)
    if block_size is None or num_warps is None:
        config = default_config(
            PRECOMPUTED_ALIGNED_SPAN,
            vocab_size,
            int(max_fused_size),
            device=logits_chunk.device,
        )
        block_size = config.block_size
        num_warps = config.num_warps

    with kernel_launch_guard(logits_chunk.device):
        ce_kl_fwdbwd_kernel[(n_rows,)](
            logits_chunk,
            logits_chunk.stride(0),
            logits_chunk,
            logits_chunk.stride(0),
            labels_chunk,
            labels_chunk.stride(0),
            nll_out,
            lse_out,
            nll_out,
            nll_out,
            n_rows,
            vocab_size,
            ignore_index,
            student_scale=0.0,
            teacher_scale=0.0,
            T_inv_s=1.0,
            T_inv_t=1.0,
            kl_grad_scale_s=0.0,
            kl_loss_scale=0.0,
            kl_alpha=0.5,
            kl_beta=0.5,
            gate_alpha_min=0.0,
            gate_alpha_span=0.0,
            gate_tau=0.0,
            gate_inv_scale=0.0,
            inv_log_vocab=0.0,
            TIED_LAYOUT=False,
            COMPUTE_TEACHER_CE=False,
            COMPUTE_KL=False,
            TEACHER_KL_GRAD=False,
            BOTH_T_ONE=True,
            KL_STYLE=KL_FORWARD,
            KERNEL_STAGE=KERNEL_STAGE_ALIGNED_STATS,
            ALLOW_TEACHER_NEG_INF_SUPPORT=False,
            RENYI_NEAR_ONE=False,
            SCALE_GRADIENT_MATERIALIZATION=False,
            BLOCK_SIZE=int(block_size),
            num_warps=int(num_warps),
        )


def _aligned_span_grad_fwdbwd(
    logits_chunk: torch.Tensor,
    labels_chunk: torch.Tensor,
    lse: torch.Tensor,
    row_coefficients: torch.Tensor,
    *,
    n_rows: int | None = None,
    ignore_index: int = -100,
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    block_size: int | None = None,
    num_warps: int | None = None,
    assume_valid: bool = False,
    gradient_materialization_scale: torch.Tensor | None = None,
) -> None:
    if not HAS_TRITON:
        raise RuntimeError(
            "_aligned_span_grad_fwdbwd requires Triton and CUDA/HIP tensors"
        )
    if not assume_valid:
        _validate_stage_inputs(
            logits_chunk,
            labels_chunk,
            lse,
            row_coefficients,
            n_rows=n_rows,
        )
        _validate_gradient_materialization_scale(
            gradient_materialization_scale,
            logits_chunk.device,
        )
    rows, vocab_size = logits_chunk.shape
    n_rows = rows if n_rows is None else int(n_rows)
    if block_size is None or num_warps is None:
        config = default_config(
            PRECOMPUTED_ALIGNED_SPAN,
            vocab_size,
            int(max_fused_size),
            device=logits_chunk.device,
        )
        block_size = config.block_size
        num_warps = config.num_warps

    with kernel_launch_guard(logits_chunk.device):
        ce_kl_fwdbwd_kernel[(n_rows,)](
            logits_chunk,
            logits_chunk.stride(0),
            logits_chunk,
            logits_chunk.stride(0),
            labels_chunk,
            labels_chunk.stride(0),
            row_coefficients,
            lse,
            row_coefficients,
            (
                gradient_materialization_scale
                if gradient_materialization_scale is not None
                else row_coefficients
            ),
            n_rows,
            vocab_size,
            ignore_index,
            student_scale=0.0,
            teacher_scale=0.0,
            T_inv_s=1.0,
            T_inv_t=1.0,
            kl_grad_scale_s=0.0,
            kl_loss_scale=0.0,
            kl_alpha=0.5,
            kl_beta=0.5,
            gate_alpha_min=0.0,
            gate_alpha_span=0.0,
            gate_tau=0.0,
            gate_inv_scale=0.0,
            inv_log_vocab=0.0,
            TIED_LAYOUT=False,
            COMPUTE_TEACHER_CE=False,
            COMPUTE_KL=False,
            TEACHER_KL_GRAD=False,
            BOTH_T_ONE=True,
            KL_STYLE=KL_FORWARD,
            KERNEL_STAGE=KERNEL_STAGE_ALIGNED_GRAD,
            ALLOW_TEACHER_NEG_INF_SUPPORT=False,
            RENYI_NEAR_ONE=False,
            SCALE_GRADIENT_MATERIALIZATION=(gradient_materialization_scale is not None),
            BLOCK_SIZE=int(block_size),
            num_warps=int(num_warps),
        )


def _validate_stage_inputs(
    logits_chunk: torch.Tensor,
    labels_chunk: torch.Tensor,
    first_buffer: torch.Tensor,
    second_buffer: torch.Tensor,
    *,
    n_rows: int | None,
) -> None:
    if logits_chunk.ndim != 2:
        raise ValueError(f"logits_chunk must be 2D, got {tuple(logits_chunk.shape)}")
    rows = logits_chunk.shape[0]
    expected_rows = rows if n_rows is None else int(n_rows)
    if rows != expected_rows:
        raise ValueError(f"logits_chunk must have {expected_rows} rows, got {rows}")
    if labels_chunk.shape != (expected_rows,):
        raise ValueError(
            f"labels_chunk must have shape ({expected_rows},), got "
            f"{tuple(labels_chunk.shape)}"
        )
    for name, buffer in (
        ("first_buffer", first_buffer),
        ("second_buffer", second_buffer),
    ):
        if buffer.shape != (expected_rows,):
            raise ValueError(
                f"{name} must have shape ({expected_rows},), got {tuple(buffer.shape)}"
            )
    tensors = (labels_chunk, first_buffer, second_buffer)
    if any(tensor.device != logits_chunk.device for tensor in tensors):
        raise ValueError("aligned stage tensors must share one device")
    if not logits_chunk.is_cuda:
        raise ValueError("aligned stages require CUDA/HIP tensors")
    if not logits_chunk.is_contiguous():
        raise ValueError("logits_chunk must be contiguous")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("aligned stage labels and row buffers must be contiguous")


# ── Two-pass pipeline ────────────────────────────────────────────────────────
def _run_aligned_pipeline(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher_span_log_probs: torch.Tensor,
    student_ranges: torch.Tensor,
    plan: AlignedSpanExecutionPlan,
    chunk_size: int,
    num_chunks: int,
    launch_config: LaunchConfig,
    need_grad_student_hidden: bool,
    need_grad_weight: bool,
    gradient_materialization_scale: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    bt = student_hidden.shape[0]
    device = student_hidden.device
    compute_dtype = plan.compute_dtype
    with _fixed_workspace_guard("student/weight compute copies"):
        student_hidden_c = cast_for_compute(student_hidden.contiguous(), compute_dtype)
        weight_cast = weight.contiguous().to(compute_dtype)
    labels_c = labels.contiguous()
    with _fixed_workspace_guard("range and teacher metadata"):
        ranges_c = student_ranges.to(device=device, dtype=torch.int64).contiguous()
        teacher_c = teacher_span_log_probs.to(device=device, dtype=torch.float64)
        valid_rows = labels_c != plan.ignore_index
    output_dtype = _loss_output_dtype(student_hidden.dtype)
    zero = torch.zeros((), dtype=output_dtype, device=device)
    active = plan.specialization.compute_student_ce or plan.specialization.compute_span
    grad_weight: torch.Tensor | None
    grad_weight_accum_dtype = resolve_grad_weight_accum_dtype(
        weight.dtype,
        compute_dtype,
        plan.grad_weight_accumulation,
    )

    if bt == 0 or not active:
        with _fixed_workspace_guard("empty-result gradients"):
            grad_hidden = (
                torch.zeros_like(student_hidden) if need_grad_student_hidden else None
            )
            grad_weight = (
                torch.zeros(weight.shape, dtype=grad_weight_accum_dtype, device=device)
                if need_grad_weight
                else None
            )
        return zero, zero.clone(), zero.clone(), grad_hidden, grad_weight

    with _fixed_workspace_guard("token statistics"):
        token_nll = torch.empty(bt, dtype=torch.float32, device=device)
        row_lse = torch.empty(bt, dtype=torch.float32, device=device)
    single_chunk_logits: torch.Tensor | None = None

    with torch.no_grad(), compute_autocast_context(device, compute_dtype):
        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, bt)
            hidden_chunk = student_hidden_c[start:end]
            logits_chunk = hidden_chunk @ weight_cast.t()
            _aligned_span_stats_fwdbwd(
                logits_chunk,
                labels_c[start:end],
                token_nll[start:end],
                row_lse[start:end],
                n_rows=end - start,
                ignore_index=plan.ignore_index,
                max_fused_size=plan.max_fused_size,
                block_size=launch_config.block_size,
                num_warps=launch_config.num_warps,
                assume_valid=True,
            )
            if num_chunks == 1:
                single_chunk_logits = logits_chunk

        if num_chunks > 1:
            # The loop variable otherwise keeps the final statistics chunk alive
            # while the gradient pass materializes a second logits chunk.
            del logits_chunk

        with _fixed_workspace_guard("span reductions"):
            if plan.specialization.compute_student_ce:
                student_ce = token_nll.sum()
                if plan.reduction == "mean":
                    student_ce = student_ce / valid_rows.sum().clamp_min(1)
            else:
                student_ce = token_nll.new_zeros(())

            if plan.specialization.compute_span:
                ell_s_raw = span_log_probs_from_nll(token_nll, ranges_c)
                span_keep = span_valid_counts(valid_rows, ranges_c) > 0
                span_terms = binary_span_objective(
                    ell_s_raw,
                    teacher_c,
                    span_keep,
                    objective=plan.objective,
                    reduction=plan.reduction,
                )
                span_div = span_terms.reduced_loss
                span_gradient = span_terms.gradient
                span_reduction_scale = span_terms.reduction_scale
                row_coefficients = aligned_row_coefficients(
                    valid_rows,
                    ranges_c,
                    span_gradient,
                    student_ce_weight=plan.student_ce_weight,
                    span_kl_weight=plan.span_kl_weight,
                    reduction=plan.reduction,
                    span_reduction_scale=span_reduction_scale,
                )
            else:
                span_div = token_nll.new_zeros(())
                ce_scale = (
                    token_nll.new_ones(())
                    / valid_rows.sum().clamp_min(1).to(torch.float32)
                    if plan.reduction == "mean"
                    else token_nll.new_ones(())
                )
                row_coefficients = (
                    valid_rows.to(torch.float32)
                    * float(plan.student_ce_weight)
                    * ce_scale
                )

            loss = (
                float(plan.student_ce_weight) * student_ce
                + float(plan.span_kl_weight) * span_div
            )

        with _fixed_workspace_guard("output gradients"):
            grad_student_hidden = (
                torch.empty_like(student_hidden) if need_grad_student_hidden else None
            )
            grad_weight = (
                torch.empty(weight.shape, dtype=grad_weight_accum_dtype, device=device)
                if need_grad_weight
                else None
            )
        if need_grad_student_hidden or need_grad_weight:
            for chunk_id in range(num_chunks):
                start = chunk_id * chunk_size
                end = min(start + chunk_size, bt)
                hidden_chunk = student_hidden_c[start:end]
                grad_logits_chunk = (
                    single_chunk_logits
                    if num_chunks == 1
                    else hidden_chunk @ weight_cast.t()
                )
                assert grad_logits_chunk is not None
                _aligned_span_grad_fwdbwd(
                    grad_logits_chunk,
                    labels_c[start:end],
                    row_lse[start:end],
                    row_coefficients[start:end],
                    n_rows=end - start,
                    ignore_index=plan.ignore_index,
                    max_fused_size=plan.max_fused_size,
                    block_size=launch_config.block_size,
                    num_warps=launch_config.num_warps,
                    assume_valid=True,
                    gradient_materialization_scale=gradient_materialization_scale,
                )
                if grad_student_hidden is not None:
                    grad_student_hidden[start:end] = grad_logits_chunk @ weight_cast
                if need_grad_weight:
                    assert grad_weight is not None
                    accumulate_weight_gradient_(
                        grad_weight,
                        grad_logits_chunk,
                        hidden_chunk,
                        row_active=valid_rows[start:end],
                        overwrite=chunk_id == 0,
                        dlogits_zero_for_inactive=True,
                    )

    return (
        loss.to(output_dtype),
        student_ce.detach().to(output_dtype),
        span_div.detach().to(output_dtype),
        grad_student_hidden,
        grad_weight,
    )


def _make_aligned_benchmark_task(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher_span_log_probs: torch.Tensor,
    student_ranges: torch.Tensor,
    plan: AlignedSpanExecutionPlan,
    chunk_size: int,
    num_chunks: int,
    need_grad_student_hidden: bool,
    need_grad_weight: bool,
    gradient_materialization_scale: torch.Tensor | None = None,
) -> BenchmarkTask:
    benchmark_gradient_materialization_scale = (
        gradient_materialization_scale.new_ones(())
        if gradient_materialization_scale is not None
        else None
    )

    def run_config(config: LaunchConfig) -> torch.Tensor:
        loss, _, _, _, _ = _run_aligned_pipeline(
            student_hidden=student_hidden,
            weight=weight,
            labels=labels,
            teacher_span_log_probs=teacher_span_log_probs,
            student_ranges=student_ranges,
            plan=plan,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            launch_config=config,
            need_grad_student_hidden=need_grad_student_hidden,
            need_grad_weight=need_grad_weight,
            gradient_materialization_scale=(benchmark_gradient_materialization_scale),
        )
        return loss

    return full_batch_benchmark_task(run_config=run_config)


def _aligned_pipeline_key(plan: AlignedSpanExecutionPlan) -> tuple[str, str]:
    span_state = "span_active" if plan.specialization.compute_span else "span_inactive"
    return ("aligned_span", span_state)


# ── Autograd function ────────────────────────────────────────────────────────
class PrecomputedAlignedSpanFunction(torch.autograd.Function):
    INPUT_COUNT = 8

    @staticmethod
    def forward(
        ctx,
        student_hidden,
        weight,
        labels,
        teacher_span_log_probs,
        student_ranges,
        gradient_materialization_scale,
        execution_plan,
        attempt,
    ):
        if (
            execution_plan.backend != "triton"
            or execution_plan.teacher_mode != PRECOMPUTED
            or execution_plan.kernel_path != PRECOMPUTED_ALIGNED_SPAN
        ):
            raise RuntimeError(
                "PrecomputedAlignedSpanFunction received an incompatible plan"
            )

        bt, hidden_size = student_hidden.shape
        vocab_size = weight.shape[0]
        need_grad_student_hidden = student_hidden.requires_grad
        need_grad_weight = weight.requires_grad
        compute_grad = need_grad_student_hidden or need_grad_weight
        _, _, gradient_materialization_scale = prepare_gradient_materialization_factor(
            labels,
            execution_plan.ignore_index,
            execution_plan.reduction,
            gradient_materialization_scale,
            execution_plan.compute_dtype,
        )
        grad_weight_accum_dtype = resolve_grad_weight_accum_dtype(
            weight.dtype,
            execution_plan.compute_dtype,
            execution_plan.grad_weight_accumulation,
        )

        if bt == 0 or not (
            execution_plan.specialization.compute_student_ce
            or execution_plan.specialization.compute_span
        ):
            config = default_config(
                PRECOMPUTED_ALIGNED_SPAN,
                vocab_size,
                execution_plan.max_fused_size,
                device=student_hidden.device,
            )
        else:
            benchmark_factory = None
            if execution_plan.autotune is not False:

                def benchmark_factory() -> BenchmarkTask:
                    return _make_aligned_benchmark_task(
                        student_hidden=student_hidden,
                        weight=weight,
                        labels=labels,
                        teacher_span_log_probs=teacher_span_log_probs,
                        student_ranges=student_ranges,
                        plan=execution_plan,
                        chunk_size=attempt.chunk_size,
                        num_chunks=attempt.num_chunks,
                        need_grad_student_hidden=need_grad_student_hidden,
                        need_grad_weight=need_grad_weight,
                        gradient_materialization_scale=(gradient_materialization_scale),
                    )

            request = FullBatchLaunchRequest(
                mode=PRECOMPUTED_ALIGNED_SPAN,
                device=student_hidden.device,
                dtype=execution_plan.compute_dtype,
                vocab_size=vocab_size,
                bt=bt,
                chunk_rows=attempt.chunk_size,
                chunk_dynamic=execution_plan.chunk_policy.dynamic,
                max_fused_size=execution_plan.max_fused_size,
                shape_key=(hidden_size, max(1, student_ranges.shape[0])),
                autotune=execution_plan.autotune,
                refresh_autotune=execution_plan.refresh_autotune,
                compute_kl=False,
                t_is_one=True,
                compute_teacher_ce=False,
                compute_student_ce=execution_plan.specialization.compute_student_ce,
                kl_variant="forward",
                teacher_grad_policy=TEACHER_GRAD_NONE,
                grad_student_hidden=need_grad_student_hidden,
                grad_teacher_hidden=False,
                grad_weight_accum_dtype=(
                    grad_accum_dtype_key(grad_weight_accum_dtype)
                    if need_grad_weight
                    else None
                ),
                grad_teacher_weight_accum_dtype=None,
                gradient_materialization_scaled=(
                    gradient_materialization_scale is not None
                ),
                pipeline_key=_aligned_pipeline_key(execution_plan),
            )
            config = select_full_batch_launch_plan(
                request,
                benchmark_factory=benchmark_factory,
            ).config

        loss, student_ce, span_div, grad_student_hidden, grad_weight = (
            _run_aligned_pipeline(
                student_hidden=student_hidden,
                weight=weight,
                labels=labels,
                teacher_span_log_probs=teacher_span_log_probs,
                student_ranges=student_ranges,
                plan=execution_plan,
                chunk_size=attempt.chunk_size,
                num_chunks=attempt.num_chunks,
                launch_config=config,
                need_grad_student_hidden=need_grad_student_hidden,
                need_grad_weight=need_grad_weight,
                gradient_materialization_scale=gradient_materialization_scale,
            )
        )

        teacher_ce = student_ce.new_zeros(())
        if gradient_materialization_scale is None:
            ctx.save_for_backward(grad_student_hidden, grad_weight)
        else:
            ctx.save_for_backward(
                grad_student_hidden,
                grad_weight,
                gradient_materialization_scale,
            )
        ctx.compute_grad = compute_grad
        ctx.gradient_materialization_scaled = gradient_materialization_scale is not None
        ctx.weight_dtype = weight.dtype
        ctx.mark_non_differentiable(student_ce, teacher_ce, span_div)
        return loss, student_ce, teacher_ce, span_div

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _gs=None, _gt=None, _gspan=None):
        if grad_output is None or not ctx.compute_grad:
            return (None,) * PrecomputedAlignedSpanFunction.INPUT_COUNT

        claim_cached_gradients(ctx, owner="PrecomputedAlignedSpanFunction")
        if ctx.gradient_materialization_scaled:
            grad_student_hidden, grad_weight, materialization_scale = ctx.saved_tensors
            effective_grad = grad_output / materialization_scale
        else:
            grad_student_hidden, grad_weight = ctx.saved_tensors
            effective_grad = grad_output
        grad_student_hidden = scale_cached_gradient_(
            grad_student_hidden,
            effective_grad,
        )
        grad_weight = scale_cached_gradient_(
            grad_weight,
            effective_grad,
            output_dtype=ctx.weight_dtype,
        )
        return input_gradient_tuple(
            PrecomputedAlignedSpanFunction.INPUT_COUNT,
            (0, grad_student_hidden),
            (1, grad_weight),
        )


__all__ = [
    "PrecomputedAlignedSpanFunction",
    "_aligned_span_grad_fwdbwd",
    "_aligned_span_stats_fwdbwd",
]
