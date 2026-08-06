"""Frozen SeparateTeacher path: teacher supplies KL targets without gradients."""

from __future__ import annotations

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
from ...._planning.models import SEPARATE_FROZEN
from ...._runtime import HAS_TRITON, resolve_grad_weight_accum_dtype
from ...._torch_backend import _combine_weighted_losses
from ...._validation import _validate_gradient_materialization_scale
from ..._common import (
    accumulate_weight_gradient_,
    cast_for_compute,
    claim_cached_gradients,
    grad_accum_dtype_key,
    input_gradient_tuple,
    kernel_launch_guard,
    prepare_gradient_materialization_factor,
    reduce_loss_buffers,
    scale_cached_gradient_,
    validate_dual_logit_inputs,
)
from .._tokenwise_launch import is_renyi_near_one, resolve_tokenwise_kl_launch
from ..ce_kl_fwdbwd import (
    KERNEL_STAGE_TOKENWISE,
    ce_kl_fwdbwd_kernel,
    compute_autocast_context,
)


# ── Python wrapper ───────────────────────────────────────────────────────────
def _separate_frozen_fwdbwd(
    logits_student_chunk: torch.Tensor,
    logits_teacher_chunk: torch.Tensor,
    labels_chunk: torch.Tensor,
    student_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    *,
    kl_weight: float,
    kl_temperature: float,
    n_rows: int | None = None,
    ignore_index: int = -100,
    student_scale: float = 1.0,
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    block_size: int | None = None,
    num_warps: int | None = None,
    kl_divergence: str = "forward",
    assume_valid: bool = False,
    kl_alpha: float = 0.5,
    kl_beta: float = 0.5,
    entropy_gate=None,
    kl_teacher_temperature: float | None = None,
    gradient_materialization_factor: torch.Tensor | None = None,
) -> None:
    """Forward + in-place student backward for a frozen separate teacher."""
    # ── Validation and launch specialization ─────────────────
    if not HAS_TRITON:
        raise RuntimeError(
            "_separate_frozen_fwdbwd requires Triton and CUDA/HIP tensors."
        )
    if not assume_valid and logits_student_chunk.ndim != 2:
        raise ValueError(
            f"logits_student_chunk must be 2D, got {tuple(logits_student_chunk.shape)}"
        )
    student_rows, vocab_size = logits_student_chunk.shape
    if n_rows is None:
        n_rows = student_rows
    if not assume_valid:
        labels_chunk = validate_dual_logit_inputs(
            logits_student_chunk,
            logits_teacher_chunk,
            labels_chunk,
            (student_loss, kl_loss),
            n_rows=n_rows,
        )
        _validate_gradient_materialization_scale(
            gradient_materialization_factor,
            logits_student_chunk.device,
        )
    if block_size is None or num_warps is None:
        launch_config = default_config(
            SEPARATE_FROZEN,
            vocab_size,
            int(max_fused_size),
            device=logits_student_chunk.device,
        )
        block_size = launch_config.block_size
        num_warps = launch_config.num_warps
    compute_kl = float(kl_weight) != 0.0
    launch = resolve_tokenwise_kl_launch(
        compute_kl=compute_kl,
        kl_temperature=kl_temperature,
        kl_teacher_temperature=kl_teacher_temperature,
        kl_divergence=kl_divergence,
        kl_alpha=kl_alpha,
        vocab_size=vocab_size,
        entropy_gate=entropy_gate,
    )
    kl_teacher_temperature = launch.teacher_temperature
    both_t_one = launch.both_t_one

    # ── Fused kernel launch ──────────────────────────────────
    with kernel_launch_guard(logits_student_chunk.device):
        ce_kl_fwdbwd_kernel[(n_rows,)](
            logits_student_chunk,
            logits_student_chunk.stride(0),
            logits_teacher_chunk,
            logits_teacher_chunk.stride(0),
            labels_chunk,
            labels_chunk.stride(0),
            student_loss,
            student_loss,
            kl_loss,
            (
                gradient_materialization_factor
                if gradient_materialization_factor is not None
                else student_loss
            ),
            n_rows,
            vocab_size,
            ignore_index,
            student_scale=float(student_scale),
            teacher_scale=0.0,
            T_inv_s=1.0 / float(kl_temperature),
            T_inv_t=1.0 / float(kl_teacher_temperature),
            kl_grad_scale_s=float(kl_weight) * float(kl_temperature),
            kl_loss_scale=float(kl_temperature) * float(kl_temperature),
            kl_alpha=float(kl_alpha),
            kl_beta=float(kl_beta),
            gate_alpha_min=launch.gate_alpha_min,
            gate_alpha_span=launch.gate_alpha_span,
            gate_tau=launch.gate_tau,
            gate_inv_scale=launch.gate_inv_scale,
            inv_log_vocab=launch.inv_log_vocab,
            TIED_LAYOUT=False,
            COMPUTE_TEACHER_CE=False,
            COMPUTE_KL=compute_kl,
            TEACHER_KL_GRAD=False,
            BOTH_T_ONE=both_t_one,
            KL_STYLE=launch.kl_style,
            KERNEL_STAGE=KERNEL_STAGE_TOKENWISE,
            ALLOW_TEACHER_NEG_INF_SUPPORT=False,
            RENYI_NEAR_ONE=launch.renyi_near_one,
            SCALE_GRADIENT_MATERIALIZATION=(
                gradient_materialization_factor is not None
            ),
            BLOCK_SIZE=int(block_size),
            num_warps=int(num_warps),
        )


# ── Full-BT benchmark executor ───────────────────────────────────────────────
def _make_separate_frozen_benchmark_task(
    *,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    compute_dtype: torch.dtype,
    compute_kl: bool,
    chunk_size: int,
    num_chunks: int,
    need_grad_student_hidden: bool,
    need_grad_weight: bool,
    grad_weight_accum_dtype: torch.dtype,
    student_ce_weight: float,
    kl_weight: float,
    kl_temperature: float,
    ignore_index: int,
    reduction: str,
    max_fused_size: int,
    kl_divergence: str,
    kl_alpha: float = 0.5,
    kl_beta: float = 0.5,
    entropy_gate=None,
    kl_teacher_temperature: float | None = None,
    gradient_materialization_scale: torch.Tensor | None = None,
) -> BenchmarkTask:
    """Build the frozen-teacher full-BT pipeline without Function.apply."""
    device = student_hidden.device
    bt = student_hidden.shape[0]
    with torch.no_grad():
        student_hidden_c = cast_for_compute(student_hidden.contiguous(), compute_dtype)
        weight_cast = weight.contiguous().to(compute_dtype)
        labels_c = labels.contiguous()
        teacher_hidden_c = (
            cast_for_compute(teacher_hidden.contiguous(), compute_dtype)
            if compute_kl
            else None
        )
        teacher_weight_c = (
            teacher_weight.contiguous().to(compute_dtype) if compute_kl else None
        )

    gradient_materialization_factor = (
        gradient_materialization_scale.new_ones(())
        if gradient_materialization_scale is not None
        else None
    )

    # ── Benchmark workspaces ─────────────────────────────────
    grad_student_hidden_scratch = (
        torch.empty_like(student_hidden) if need_grad_student_hidden else None
    )
    grad_weight_scratch = (
        torch.empty(weight_cast.shape, dtype=grad_weight_accum_dtype, device=device)
        if need_grad_weight
        else None
    )
    loss_workspace = {
        "student_loss": torch.empty(bt, dtype=torch.float32, device=device),
        "kl_loss": torch.empty(bt, dtype=torch.float32, device=device),
    }
    student_path_active = student_ce_weight != 0.0 or kl_weight != 0.0

    def prepare_batch(_repetitions: int) -> None:
        for buffer in loss_workspace.values():
            buffer.zero_()

    def run_config(config: LaunchConfig) -> torch.Tensor:
        student_loss = loss_workspace["student_loss"]
        kl_loss = loss_workspace["kl_loss"]
        grad_student_hidden = grad_student_hidden_scratch
        grad_weight = grad_weight_scratch
        weight_started = False

        with torch.no_grad(), compute_autocast_context(device, compute_dtype):
            # Process each chunk.
            for chunk_id in range(num_chunks):
                start = chunk_id * chunk_size
                end = min(start + chunk_size, bt)
                n_rows = end - start
                student_hidden_chunk = student_hidden_c[start:end]
                labels_chunk = labels_c[start:end]
                logits_s_chunk = student_hidden_chunk @ weight_cast.t()
                if compute_kl:
                    assert teacher_hidden_c is not None and teacher_weight_c is not None
                    logits_t_chunk = teacher_hidden_c[start:end] @ teacher_weight_c.t()
                else:
                    logits_t_chunk = logits_s_chunk

                _separate_frozen_fwdbwd(
                    logits_s_chunk,
                    logits_t_chunk,
                    labels_chunk,
                    student_loss[start:end],
                    kl_loss[start:end],
                    n_rows=n_rows,
                    kl_weight=kl_weight,
                    kl_temperature=kl_temperature,
                    kl_teacher_temperature=kl_teacher_temperature,
                    ignore_index=ignore_index,
                    student_scale=float(student_ce_weight),
                    max_fused_size=max_fused_size,
                    block_size=config.block_size,
                    num_warps=config.num_warps,
                    kl_divergence=kl_divergence,
                    assume_valid=True,
                    kl_alpha=kl_alpha,
                    kl_beta=kl_beta,
                    entropy_gate=entropy_gate,
                    gradient_materialization_factor=gradient_materialization_factor,
                )

                if grad_student_hidden is not None:
                    if student_path_active:
                        grad_student_hidden[start:end] = logits_s_chunk @ weight_cast
                    else:
                        grad_student_hidden[start:end].zero_()
                if grad_weight is not None and student_path_active:
                    accumulate_weight_gradient_(
                        grad_weight,
                        logits_s_chunk,
                        student_hidden_chunk,
                        row_active=labels_chunk != ignore_index,
                        overwrite=not weight_started,
                        dlogits_zero_for_inactive=True,
                    )
                    weight_started = True

        # ── Reduce and combine ───────────────────────────────
        (loss_s, kl_loss_value), _ = reduce_loss_buffers(
            labels_c,
            ignore_index,
            reduction,
            student_loss,
            kl_loss,
        )
        return _combine_weighted_losses(
            loss_s,
            loss_s.new_zeros(()),
            kl_loss_value,
            student_ce_weight=student_ce_weight,
            teacher_ce_weight=0.0,
            kl_weight=kl_weight,
        )

    return full_batch_benchmark_task(
        run_config=run_config,
        prepare_batch=prepare_batch,
    )


class SeparateFrozenCEFunction(torch.autograd.Function):
    INPUT_COUNT = 8

    @staticmethod
    def forward(
        ctx,
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        gradient_materialization_scale,
        execution_plan,
        attempt,
    ):
        # ── Execution-plan state ─────────────────────────────
        if (
            execution_plan.backend != "triton"
            or execution_plan.kernel_path != SEPARATE_FROZEN
        ):
            raise RuntimeError(
                "SeparateFrozenCEFunction received an incompatible execution plan"
            )
        if execution_plan.teacher_ce_weight != 0.0:
            raise ValueError("SeparateFrozenCEFunction requires teacher_ce_weight=0.0")

        BT, _ = student_hidden.shape
        V = weight.shape[0]
        device = student_hidden.device
        specialization = execution_plan.specialization
        student_ce_weight = execution_plan.student_ce_weight
        kl_weight = execution_plan.kl_weight
        kl_temperature = execution_plan.kl_temperature
        kl_teacher_temperature = execution_plan.kl_teacher_temperature
        ignore_index = execution_plan.ignore_index
        reduction = execution_plan.reduction
        max_fused_size = execution_plan.max_fused_size
        autotune = execution_plan.autotune
        kl_divergence = specialization.kl_variant
        kl_alpha = execution_plan.kl_alpha
        kl_beta = execution_plan.kl_beta
        entropy_gate = execution_plan.entropy_gate
        compute_kl = specialization.compute_kl
        compute_dtype = execution_plan.compute_dtype
        chunk_size_actual = attempt.chunk_size
        num_chunks = attempt.num_chunks
        need_grad_student_hidden = student_hidden.requires_grad
        need_grad_weight = weight.requires_grad
        grad_weight_accum_dtype = resolve_grad_weight_accum_dtype(
            weight.dtype,
            compute_dtype,
            execution_plan.grad_weight_accumulation,
        )
        grad_teacher_weight_accum_dtype = None
        # ── Autotune launch plan ─────────────────────────────
        benchmark_factory = None
        if autotune is not False:

            def benchmark_factory() -> BenchmarkTask:
                return _make_separate_frozen_benchmark_task(
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden,
                    weight=weight,
                    teacher_weight=teacher_weight,
                    labels=labels,
                    compute_dtype=compute_dtype,
                    compute_kl=compute_kl,
                    chunk_size=chunk_size_actual,
                    num_chunks=num_chunks,
                    need_grad_student_hidden=need_grad_student_hidden,
                    need_grad_weight=need_grad_weight,
                    grad_weight_accum_dtype=grad_weight_accum_dtype,
                    student_ce_weight=float(student_ce_weight),
                    kl_weight=float(kl_weight),
                    kl_temperature=float(kl_temperature),
                    kl_teacher_temperature=float(kl_teacher_temperature),
                    ignore_index=ignore_index,
                    reduction=reduction,
                    max_fused_size=max_fused_size,
                    kl_divergence=kl_divergence,
                    kl_alpha=kl_alpha,
                    kl_beta=kl_beta,
                    entropy_gate=entropy_gate,
                    gradient_materialization_scale=gradient_materialization_scale,
                )

        launch_request = FullBatchLaunchRequest(
            mode=SEPARATE_FROZEN,
            device=device,
            dtype=compute_dtype,
            vocab_size=V,
            bt=BT,
            chunk_rows=chunk_size_actual,
            chunk_dynamic=execution_plan.chunk_policy.dynamic,
            max_fused_size=max_fused_size,
            shape_key=(
                (student_hidden.shape[1], teacher_hidden.shape[1])
                if compute_kl
                else (student_hidden.shape[1],)
            ),
            autotune=autotune,
            refresh_autotune=execution_plan.refresh_autotune,
            compute_kl=compute_kl,
            t_is_one=specialization.t_is_one,
            compute_teacher_ce=False,
            compute_student_ce=float(student_ce_weight) != 0.0,
            kl_variant=kl_divergence,
            teacher_grad_policy=specialization.teacher_grad_policy,
            grad_student_hidden=need_grad_student_hidden,
            grad_teacher_hidden=False,
            grad_weight_accum_dtype=(
                grad_accum_dtype_key(grad_weight_accum_dtype)
                if need_grad_weight
                else None
            ),
            grad_teacher_weight_accum_dtype=grad_accum_dtype_key(
                grad_teacher_weight_accum_dtype
            ),
            gradient_materialization_scaled=(
                gradient_materialization_scale is not None
            ),
            renyi_near_one=is_renyi_near_one(
                compute_kl=compute_kl,
                kl_divergence=kl_divergence,
                kl_alpha=kl_alpha,
            ),
        )
        launch_plan = select_full_batch_launch_plan(
            launch_request,
            benchmark_factory=benchmark_factory,
        )

        # ── Inputs and gradient workspaces ───────────────────
        student_hidden = student_hidden.contiguous()
        weight = weight.contiguous()
        labels = labels.contiguous()
        if compute_kl:
            teacher_hidden = teacher_hidden.contiguous()
            teacher_weight = teacher_weight.contiguous()

        weight_cast = weight.to(compute_dtype)
        student_hidden_c = cast_for_compute(student_hidden, compute_dtype)
        if compute_kl:
            teacher_weight_cast = teacher_weight.to(compute_dtype)
            teacher_hidden_c = cast_for_compute(teacher_hidden, compute_dtype)
        else:
            teacher_weight_cast = None
            teacher_hidden_c = None

        compute_grad = need_grad_student_hidden or need_grad_weight
        grad_student_hidden = (
            torch.empty_like(student_hidden) if need_grad_student_hidden else None
        )
        grad_weight = (
            torch.empty(
                weight.shape,
                dtype=grad_weight_accum_dtype,
                device=device,
            )
            if need_grad_weight
            else None
        )
        weight_started = False
        loss_student_buf = torch.empty(BT, dtype=torch.float32, device=device)
        kl_buf = torch.empty(BT, dtype=torch.float32, device=device)
        (
            gradient_materialization_factor,
            mean_denominator,
            gradient_materialization_scale,
        ) = prepare_gradient_materialization_factor(
            labels,
            ignore_index,
            reduction,
            gradient_materialization_scale,
            compute_dtype,
        )

        student_scale = float(student_ce_weight)
        student_path_active = student_scale != 0.0 or compute_kl
        if need_grad_weight and not student_path_active:
            grad_weight.zero_()
            weight_started = True
        with torch.no_grad(), compute_autocast_context(device, compute_dtype):
            for chunk_id in range(num_chunks):
                # Slice the current chunk.
                start = chunk_id * chunk_size_actual
                end = min(start + chunk_size_actual, BT)
                n_rows = end - start

                # Materialize logits and run the row kernel.
                student_hidden_chunk = student_hidden_c[start:end]
                teacher_hidden_chunk = (
                    teacher_hidden_c[start:end] if compute_kl else None
                )
                labels_chunk = labels[start:end]
                launch_config = launch_plan.config
                logits_s_chunk = student_hidden_chunk @ weight_cast.t()
                if compute_kl:
                    logits_t_chunk = teacher_hidden_chunk @ teacher_weight_cast.t()
                else:
                    logits_t_chunk = logits_s_chunk

                student_loss_chunk = loss_student_buf[start:end]
                kl_chunk = kl_buf[start:end]
                _separate_frozen_fwdbwd(
                    logits_s_chunk,
                    logits_t_chunk,
                    labels_chunk,
                    student_loss_chunk,
                    kl_chunk,
                    n_rows=n_rows,
                    kl_weight=kl_weight,
                    kl_temperature=kl_temperature,
                    kl_teacher_temperature=kl_teacher_temperature,
                    ignore_index=ignore_index,
                    student_scale=student_scale,
                    max_fused_size=max_fused_size,
                    block_size=launch_config.block_size,
                    num_warps=launch_config.num_warps,
                    kl_divergence=kl_divergence,
                    assume_valid=True,
                    kl_alpha=kl_alpha,
                    kl_beta=kl_beta,
                    entropy_gate=entropy_gate,
                    gradient_materialization_factor=gradient_materialization_factor,
                )

                del logits_t_chunk

                # Materialize gradients.
                if compute_grad:
                    student_logits_grad = logits_s_chunk
                    if grad_student_hidden is not None:
                        if student_path_active:
                            grad_student_hidden[start:end] = (
                                student_logits_grad @ weight_cast
                            )
                        else:
                            grad_student_hidden[start:end].zero_()

                    if need_grad_weight and student_path_active:
                        accumulate_weight_gradient_(
                            grad_weight,
                            student_logits_grad,
                            student_hidden_chunk,
                            row_active=labels_chunk != ignore_index,
                            overwrite=not weight_started,
                            dlogits_zero_for_inactive=True,
                        )
                        weight_started = True
                    del student_logits_grad

                del logits_s_chunk

        # ── Reduction and backward ───────────────────────────
        (loss_s, kl_loss), mean_scale = reduce_loss_buffers(
            labels,
            ignore_index,
            reduction,
            loss_student_buf,
            kl_buf,
            mean_denominator=mean_denominator,
        )

        loss_t = loss_s.new_zeros(())
        loss = _combine_weighted_losses(
            loss_s,
            loss_t,
            kl_loss,
            student_ce_weight=student_ce_weight,
            teacher_ce_weight=0.0,
            kl_weight=kl_weight,
        )

        backward_scale = (
            gradient_materialization_scale
            if gradient_materialization_scale is not None
            else mean_scale
        )
        ctx.save_for_backward(grad_student_hidden, grad_weight, backward_scale)
        ctx.compute_grad = compute_grad
        ctx.gradient_materialization_scaled = gradient_materialization_scale is not None
        ctx.weight_dtype = weight.dtype

        loss_s_out = loss_s.detach()
        loss_t_out = loss_t.detach()
        kl_loss_out = kl_loss.detach()
        ctx.mark_non_differentiable(loss_s_out, loss_t_out, kl_loss_out)
        return loss, loss_s_out, loss_t_out, kl_loss_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _gs=None, _gt=None, _gkl=None):
        if grad_output is None or not ctx.compute_grad:
            return (None,) * SeparateFrozenCEFunction.INPUT_COUNT

        claim_cached_gradients(ctx, owner="SeparateFrozenCEFunction")
        grad_student_hidden, grad_weight, backward_scale = ctx.saved_tensors
        effective_grad = (
            grad_output / backward_scale
            if ctx.gradient_materialization_scaled
            else grad_output * backward_scale
        )

        grad_student_hidden = scale_cached_gradient_(
            grad_student_hidden, effective_grad
        )
        grad_weight = scale_cached_gradient_(
            grad_weight,
            effective_grad,
            output_dtype=ctx.weight_dtype,
        )

        return input_gradient_tuple(
            SeparateFrozenCEFunction.INPUT_COUNT,
            (0, grad_student_hidden),
            (2, grad_weight),
        )
