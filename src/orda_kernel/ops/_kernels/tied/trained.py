"""Trained TiedTeacher path: teacher CE trains the shared projection."""

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
from ...._planning.models import TIED_TRAINED
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
)
from .._tokenwise_launch import is_renyi_near_one, resolve_tokenwise_kl_launch
from ..ce_kl_fwdbwd import (
    KERNEL_STAGE_TOKENWISE,
    ce_kl_fwdbwd_kernel,
    compute_autocast_context,
)


# ── Python wrapper ───────────────────────────────────────────────────────────
def _tied_trained_fwdbwd(
    logits_chunk: torch.Tensor,
    labels_chunk: torch.Tensor,
    student_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    *,
    teacher_loss: torch.Tensor,
    kl_weight: float,
    kl_temperature: float,
    n_rows: int | None = None,
    ignore_index: int = -100,
    student_scale: float = 1.0,
    teacher_scale: float = 1.0,
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
    """Forward + in-place backward for a trained tied teacher."""
    # ── Validation and launch specialization ─────────────────
    if not HAS_TRITON:
        raise RuntimeError("_tied_trained_fwdbwd requires Triton and CUDA/HIP tensors.")

    if not assume_valid:
        if logits_chunk.ndim != 2:
            raise ValueError(
                f"logits_chunk must be 2D, got {tuple(logits_chunk.shape)}"
            )
        if labels_chunk.ndim != 1:
            raise ValueError(
                f"labels_chunk must be 1D, got {tuple(labels_chunk.shape)}"
            )
        if student_loss.ndim != 1 or teacher_loss.ndim != 1 or kl_loss.ndim != 1:
            raise ValueError(
                "student_loss, teacher_loss and kl_loss must be 1D tensors"
            )
        if not logits_chunk.is_cuda:
            raise ValueError(
                f"logits_chunk must be a CUDA/HIP tensor, got {logits_chunk.device}"
            )
        if labels_chunk.device != logits_chunk.device:
            raise ValueError("labels_chunk and logits_chunk must be on the same device")
        if (
            student_loss.device != logits_chunk.device
            or teacher_loss.device != logits_chunk.device
            or kl_loss.device != logits_chunk.device
        ):
            raise ValueError("loss buffers must be on the same device as logits_chunk")
        if not logits_chunk.is_contiguous():
            raise ValueError(
                "logits_chunk must be contiguous because gradients are written in-place"
            )
        if (
            not student_loss.is_contiguous()
            or not teacher_loss.is_contiguous()
            or not kl_loss.is_contiguous()
        ):
            raise ValueError("loss buffers must be contiguous")

    if not assume_valid:
        _validate_gradient_materialization_scale(
            gradient_materialization_factor,
            logits_chunk.device,
        )
    total_rows, vocab_size = logits_chunk.shape
    compute_teacher_ce = float(teacher_scale) != 0.0
    compute_kl = float(kl_weight) != 0.0
    if not compute_teacher_ce:
        raise ValueError("trained tied execution requires teacher_scale != 0.0")
    if n_rows is None:
        if total_rows % 2 != 0:
            raise ValueError(
                "n_rows is required when logits_chunk has an odd row count"
            )
        n_rows = total_rows // 2
    if not assume_valid:
        expected_rows = 2 * n_rows
        if total_rows != expected_rows:
            raise ValueError(
                "logits_chunk rows must equal "
                f"{expected_rows}, got {total_rows} and {n_rows}"
            )
        if labels_chunk.shape[0] != n_rows:
            raise ValueError(
                f"labels_chunk must have {n_rows} rows, got {labels_chunk.shape[0]}"
            )
        if (
            student_loss.shape[0] != n_rows
            or teacher_loss.shape[0] != n_rows
            or kl_loss.shape[0] != n_rows
        ):
            raise ValueError("loss buffers must have n_rows elements")
        labels_chunk = labels_chunk.contiguous()
    if block_size is None or num_warps is None:
        launch_config = default_config(
            TIED_TRAINED,
            vocab_size,
            int(max_fused_size),
            device=logits_chunk.device,
        )
        block_size = launch_config.block_size
        num_warps = launch_config.num_warps
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
    with kernel_launch_guard(logits_chunk.device):
        ce_kl_fwdbwd_kernel[(n_rows,)](
            logits_chunk,
            logits_chunk.stride(0),
            logits_chunk,
            logits_chunk.stride(0),
            labels_chunk,
            labels_chunk.stride(0),
            student_loss,
            teacher_loss,
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
            teacher_scale=float(teacher_scale),
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
            TIED_LAYOUT=True,
            COMPUTE_TEACHER_CE=True,
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
def _make_tied_trained_benchmark_task(
    *,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    compute_dtype: torch.dtype,
    chunk_size: int,
    num_chunks: int,
    need_grad_student_hidden: bool,
    need_grad_teacher_hidden: bool,
    need_grad_weight: bool,
    grad_weight_accum_dtype: torch.dtype,
    student_ce_weight: float,
    teacher_ce_weight: float,
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
    """Build a reusable full-BT executor without entering autograd Function.apply."""
    device = student_hidden.device
    bt = student_hidden.shape[0]
    with torch.no_grad():
        student_hidden_c = cast_for_compute(student_hidden.contiguous(), compute_dtype)
        teacher_hidden_c = cast_for_compute(teacher_hidden.contiguous(), compute_dtype)
        weight_cast = weight.contiguous().to(compute_dtype)
        labels_c = labels.contiguous()

    gradient_materialization_factor = (
        gradient_materialization_scale.new_ones(())
        if gradient_materialization_scale is not None
        else None
    )

    # ── Benchmark workspaces ─────────────────────────────────
    grad_student_hidden_scratch = (
        torch.empty_like(student_hidden) if need_grad_student_hidden else None
    )
    grad_teacher_hidden_scratch = (
        torch.empty_like(teacher_hidden) if need_grad_teacher_hidden else None
    )
    grad_weight_scratch = (
        torch.empty(weight_cast.shape, dtype=grad_weight_accum_dtype, device=device)
        if need_grad_weight
        else None
    )
    loss_workspace = {
        "student_loss": torch.empty(bt, dtype=torch.float32, device=device),
        "teacher_loss": torch.empty(bt, dtype=torch.float32, device=device),
        "kl_loss": torch.empty(bt, dtype=torch.float32, device=device),
    }
    student_path_active = student_ce_weight != 0.0 or kl_weight != 0.0

    def prepare_batch(_repetitions: int) -> None:
        for buffer in loss_workspace.values():
            buffer.zero_()

    def run_config(config: LaunchConfig) -> torch.Tensor:
        student_loss = loss_workspace["student_loss"]
        teacher_loss = loss_workspace["teacher_loss"]
        kl_loss = loss_workspace["kl_loss"]
        grad_student_hidden = grad_student_hidden_scratch
        grad_teacher_hidden = grad_teacher_hidden_scratch
        grad_weight = grad_weight_scratch
        weight_started = False

        with torch.no_grad(), compute_autocast_context(device, compute_dtype):
            # Process each chunk.
            for chunk_id in range(num_chunks):
                start = chunk_id * chunk_size
                end = min(start + chunk_size, bt)
                n_rows = end - start
                student_hidden_chunk = student_hidden_c[start:end]
                teacher_hidden_chunk = teacher_hidden_c[start:end]
                labels_chunk = labels_c[start:end]
                hidden_chunk_concat = torch.cat(
                    [student_hidden_chunk, teacher_hidden_chunk],
                    dim=0,
                )
                logits_chunk = hidden_chunk_concat @ weight_cast.t()

                _tied_trained_fwdbwd(
                    logits_chunk,
                    labels_chunk,
                    student_loss[start:end],
                    kl_loss[start:end],
                    teacher_loss=teacher_loss[start:end],
                    n_rows=n_rows,
                    kl_weight=kl_weight,
                    kl_temperature=kl_temperature,
                    kl_teacher_temperature=kl_teacher_temperature,
                    ignore_index=ignore_index,
                    student_scale=float(student_ce_weight),
                    teacher_scale=float(teacher_ce_weight),
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
                        grad_student_hidden[start:end] = (
                            logits_chunk[:n_rows] @ weight_cast
                        )
                    else:
                        grad_student_hidden[start:end].zero_()
                if grad_teacher_hidden is not None:
                    grad_teacher_hidden[start:end] = logits_chunk[n_rows:] @ weight_cast
                if grad_weight is not None:
                    if student_path_active:
                        grad_src = logits_chunk
                        hidden_src = hidden_chunk_concat
                        row_active = (labels_chunk != ignore_index).repeat(2)
                    else:
                        grad_src = logits_chunk[n_rows:]
                        hidden_src = teacher_hidden_chunk
                        row_active = labels_chunk != ignore_index
                    accumulate_weight_gradient_(
                        grad_weight,
                        grad_src,
                        hidden_src,
                        row_active=row_active,
                        overwrite=not weight_started,
                        dlogits_zero_for_inactive=True,
                    )
                    weight_started = True

        # ── Reduce and combine ───────────────────────────────
        (loss_s, loss_t, kl_loss_value), _ = reduce_loss_buffers(
            labels_c,
            ignore_index,
            reduction,
            student_loss,
            teacher_loss,
            kl_loss,
        )
        return _combine_weighted_losses(
            loss_s,
            loss_t,
            kl_loss_value,
            student_ce_weight=student_ce_weight,
            teacher_ce_weight=teacher_ce_weight,
            kl_weight=kl_weight,
        )

    return full_batch_benchmark_task(
        run_config=run_config,
        prepare_batch=prepare_batch,
    )


# ── Autograd Function ────────────────────────────────────────────────────────
class TiedTrainedCEFunction(torch.autograd.Function):
    INPUT_COUNT = 7

    @staticmethod
    def forward(
        ctx,
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        gradient_materialization_scale,
        execution_plan,
        attempt,
    ):
        # ── Execution-plan state ─────────────────────────────
        if (
            execution_plan.backend != "triton"
            or execution_plan.kernel_path != TIED_TRAINED
        ):
            raise RuntimeError(
                "TiedTrainedCEFunction received an incompatible execution plan"
            )
        if execution_plan.teacher_ce_weight <= 0.0:
            raise ValueError("TiedTrainedCEFunction requires teacher_ce_weight > 0.0")

        BT, _ = student_hidden.shape
        V = weight.shape[0]
        device = student_hidden.device
        specialization = execution_plan.specialization
        student_ce_weight = execution_plan.student_ce_weight
        teacher_ce_weight = execution_plan.teacher_ce_weight
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
        compute_teacher_ce = specialization.compute_teacher_ce
        if not compute_teacher_ce:
            raise RuntimeError("trained tied plan must compute teacher CE")
        teacher_grad_policy = specialization.teacher_grad_policy
        student_scale = float(student_ce_weight)
        teacher_scale = float(teacher_ce_weight)
        student_path_active = student_scale != 0.0 or compute_kl
        need_grad_student_hidden = student_hidden.requires_grad and student_path_active
        need_grad_teacher_hidden = teacher_hidden.requires_grad
        need_grad_weight = weight.requires_grad
        compute_dtype = execution_plan.compute_dtype
        grad_weight_accum_dtype = resolve_grad_weight_accum_dtype(
            weight.dtype,
            compute_dtype,
            execution_plan.grad_weight_accumulation,
        )
        grad_teacher_weight_accum_dtype = None
        chunk_size_actual = attempt.chunk_size
        num_chunks = attempt.num_chunks

        # ── Autotune launch plan ─────────────────────────────
        benchmark_factory = None
        if autotune is not False:

            def benchmark_factory() -> BenchmarkTask:
                return _make_tied_trained_benchmark_task(
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden,
                    weight=weight,
                    labels=labels,
                    compute_dtype=compute_dtype,
                    chunk_size=chunk_size_actual,
                    num_chunks=num_chunks,
                    need_grad_student_hidden=need_grad_student_hidden,
                    need_grad_teacher_hidden=need_grad_teacher_hidden,
                    need_grad_weight=need_grad_weight,
                    grad_weight_accum_dtype=grad_weight_accum_dtype,
                    student_ce_weight=float(student_ce_weight),
                    teacher_ce_weight=teacher_scale,
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
            mode=TIED_TRAINED,
            device=device,
            dtype=compute_dtype,
            vocab_size=V,
            bt=BT,
            chunk_rows=chunk_size_actual,
            chunk_dynamic=execution_plan.chunk_policy.dynamic,
            max_fused_size=max_fused_size,
            shape_key=(student_hidden.shape[1],),
            autotune=autotune,
            refresh_autotune=execution_plan.refresh_autotune,
            compute_kl=compute_kl,
            t_is_one=specialization.t_is_one,
            compute_teacher_ce=compute_teacher_ce,
            compute_student_ce=student_scale != 0.0,
            kl_variant=kl_divergence,
            teacher_grad_policy=teacher_grad_policy,
            grad_student_hidden=need_grad_student_hidden,
            grad_teacher_hidden=need_grad_teacher_hidden,
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
        teacher_hidden = teacher_hidden.contiguous()
        weight = weight.contiguous()
        labels = labels.contiguous()
        weight_cast = weight.to(compute_dtype)
        student_hidden_c = cast_for_compute(student_hidden, compute_dtype)
        teacher_hidden_c = cast_for_compute(teacher_hidden, compute_dtype)
        compute_grad = (
            need_grad_student_hidden or need_grad_teacher_hidden or need_grad_weight
        )
        grad_student_hidden = (
            torch.empty_like(student_hidden) if need_grad_student_hidden else None
        )
        grad_teacher_hidden = (
            torch.empty_like(teacher_hidden) if need_grad_teacher_hidden else None
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
        loss_teacher_buf = torch.empty(BT, dtype=torch.float32, device=device)
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

        with torch.no_grad(), compute_autocast_context(device, compute_dtype):
            for chunk_id in range(num_chunks):
                # Slice the current chunk.
                start = chunk_id * chunk_size_actual
                end = min(start + chunk_size_actual, BT)
                n_rows = end - start

                # Materialize logits and run the row kernel.
                student_hidden_chunk = student_hidden_c[start:end]
                teacher_hidden_chunk = teacher_hidden_c[start:end]
                labels_chunk = labels[start:end]
                launch_config = launch_plan.config
                hidden_chunk_concat = torch.cat(
                    [student_hidden_chunk, teacher_hidden_chunk],
                    dim=0,
                )
                logits_chunk = hidden_chunk_concat @ weight_cast.t()

                student_loss_chunk = loss_student_buf[start:end]
                teacher_loss_chunk = loss_teacher_buf[start:end]
                kl_chunk = kl_buf[start:end]
                _tied_trained_fwdbwd(
                    logits_chunk,
                    labels_chunk,
                    student_loss_chunk,
                    kl_chunk,
                    teacher_loss=teacher_loss_chunk,
                    n_rows=n_rows,
                    kl_weight=kl_weight,
                    kl_temperature=kl_temperature,
                    kl_teacher_temperature=kl_teacher_temperature,
                    ignore_index=ignore_index,
                    student_scale=student_scale,
                    teacher_scale=teacher_scale,
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

                # Materialize gradients.
                if compute_grad:
                    student_logits_grad = logits_chunk[:n_rows]
                    if (
                        grad_student_hidden is not None
                        or grad_teacher_hidden is not None
                    ):
                        if grad_student_hidden is not None:
                            if student_path_active:
                                grad_student_hidden[start:end] = (
                                    student_logits_grad @ weight_cast
                                )
                            else:
                                grad_student_hidden[start:end].zero_()
                        if grad_teacher_hidden is not None:
                            teacher_logits_grad = logits_chunk[n_rows:]
                            grad_teacher_hidden[start:end] = (
                                teacher_logits_grad @ weight_cast
                            )

                    if need_grad_weight:
                        row_active = labels_chunk != ignore_index
                        if student_path_active:
                            grad_src = logits_chunk
                            hidden_src = hidden_chunk_concat
                            # ``hidden_chunk_concat`` stacks student then teacher
                            # rows, so the row-mask is repeated to cover both.
                            row_active = row_active.repeat(2)
                        else:
                            grad_src = logits_chunk[n_rows:]
                            hidden_src = teacher_hidden_chunk
                        accumulate_weight_gradient_(
                            grad_weight,
                            grad_src,
                            hidden_src,
                            row_active=row_active,
                            overwrite=not weight_started,
                            dlogits_zero_for_inactive=True,
                        )
                        weight_started = True

                del logits_chunk, hidden_chunk_concat

        # ── Reduction and backward ───────────────────────────
        (loss_s, loss_t_raw, kl_loss), mean_scale = reduce_loss_buffers(
            labels,
            ignore_index,
            reduction,
            loss_student_buf,
            loss_teacher_buf,
            kl_buf,
            mean_denominator=mean_denominator,
        )

        loss_t = loss_t_raw
        loss = _combine_weighted_losses(
            loss_s,
            loss_t,
            kl_loss,
            student_ce_weight=student_ce_weight,
            teacher_ce_weight=teacher_scale,
            kl_weight=kl_weight,
        )

        backward_scale = (
            gradient_materialization_scale
            if gradient_materialization_scale is not None
            else mean_scale
        )
        ctx.save_for_backward(
            grad_student_hidden, grad_teacher_hidden, grad_weight, backward_scale
        )
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
            return (None,) * TiedTrainedCEFunction.INPUT_COUNT

        claim_cached_gradients(ctx, owner="TiedTrainedCEFunction")
        grad_student_hidden, grad_teacher_hidden, grad_weight, backward_scale = (
            ctx.saved_tensors
        )
        effective_grad = (
            grad_output / backward_scale
            if ctx.gradient_materialization_scaled
            else grad_output * backward_scale
        )

        grad_student_hidden = scale_cached_gradient_(
            grad_student_hidden, effective_grad
        )
        grad_teacher_hidden = scale_cached_gradient_(
            grad_teacher_hidden, effective_grad
        )
        grad_weight = scale_cached_gradient_(
            grad_weight,
            effective_grad,
            output_dtype=ctx.weight_dtype,
        )

        return input_gradient_tuple(
            TiedTrainedCEFunction.INPUT_COUNT,
            (0, grad_student_hidden),
            (1, grad_teacher_hidden),
            (2, grad_weight),
        )
