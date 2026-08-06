from __future__ import annotations

import inspect
from functools import cache

import torch

from .._execution.recovery import NonChunkableWorkspaceOOM, is_oom_error


# ── Autograd and compute helpers ─────────────────────────────────────────────
def forward_input_count(function: type[torch.autograd.Function]) -> int:
    """Return the positional ``forward`` input count after ``ctx``."""
    return len(inspect.signature(function.forward).parameters) - 1


def input_gradient_tuple(
    input_count: int,
    *placed: tuple[int, torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    """Place live gradients in the fixed positional autograd ABI."""
    grads: list[torch.Tensor | None] = [None] * input_count
    for index, grad in placed:
        grads[index] = grad
    return tuple(grads)


def cast_for_compute(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def kernel_launch_guard(device: torch.device):
    """Pin a Triton launch to the operand's CUDA/HIP device."""
    return torch.cuda.device(device)


# ── Gradient accumulation and caching ────────────────────────────────────────
def grad_accum_dtype_key(dtype: torch.dtype | None) -> str | None:
    """Render an accumulator dtype for the autotune workload key."""
    return None if dtype is None else str(dtype)


def accumulate_weight_gradient_(
    destination: torch.Tensor,
    dlogits: torch.Tensor,
    hidden: torch.Tensor,
    *,
    row_active: torch.Tensor | None,
    overwrite: bool,
    dlogits_zero_for_inactive: bool = False,
) -> torch.Tensor:
    """Accumulate ``dlogits.T @ hidden`` into ``destination`` in place.

    ``destination.dtype`` is the resolved accumulator policy; ``overwrite`` is
    required because the buffer is uninitialized before the first write.

    ``dlogits_zero_for_inactive`` is an internal trust boundary: enable it only
    for kernel output that guarantees every inactive ``dlogits`` row is exact
    zero. It avoids materializing another ``[N, V]`` tensor, but ``hidden`` must
    still be masked because IEEE multiplication makes ``0 * NaN`` a NaN.
    """
    if dlogits.dtype != hidden.dtype:
        raise ValueError(
            "weight-gradient GEMM operands must share a dtype, got "
            f"dlogits={dlogits.dtype} and hidden={hidden.dtype}"
        )
    if destination.device != dlogits.device or hidden.device != dlogits.device:
        raise ValueError(
            "weight-gradient accumulation requires one device, got "
            f"destination={destination.device}, dlogits={dlogits.device}, "
            f"hidden={hidden.device}"
        )
    expected_shape = (dlogits.shape[1], hidden.shape[1])
    if tuple(destination.shape) != expected_shape:
        raise ValueError(
            f"weight-gradient accumulator must have shape {expected_shape}, got "
            f"{tuple(destination.shape)}"
        )

    if row_active is not None:
        if not dlogits_zero_for_inactive:
            dlogits = torch.where(
                row_active.unsqueeze(-1),
                dlogits,
                dlogits.new_zeros(()),
            )
        hidden = torch.where(row_active.unsqueeze(-1), hidden, hidden.new_zeros(()))

    # Operands are already cast for compute; autocast would re-cast the product.
    with torch.autocast(destination.device.type, enabled=False):
        if destination.dtype == dlogits.dtype:
            if overwrite:
                return torch.mm(dlogits.t(), hidden, out=destination)
            destination.addmm_(dlogits.t(), hidden)
            return destination

        # Some supported PyTorch versions mis-decompose aten.addmm.dtype_out under
        # torch.compile (pytorch/pytorch#190936). ORDA's kernel execution is already
        # intentionally Dynamo-disabled, so supported paths never trace this operation;
        # the upstream compiler bug therefore does not warrant a more complex design.
        if _addmm_supports_out_dtype(destination, dlogits, hidden):
            torch.addmm(
                destination,
                dlogits.t(),
                hidden,
                beta=0 if overwrite else 1,
                out_dtype=destination.dtype,
                out=destination,
            )
            return destination

        try:
            contribution = torch.mm(dlogits.t(), hidden)
        except Exception as exc:
            if is_oom_error(exc):
                raise NonChunkableWorkspaceOOM(
                    "weight-gradient mixed-dtype fallback allocates a fixed "
                    f"{tuple(destination.shape)} workspace"
                ) from exc
            raise
    if overwrite:
        return destination.copy_(contribution)
    return destination.add_(contribution)


@cache
def _addmm_out_dtype_is_usable(
    device_type: str,
    device_index: int | None,
    operand_dtype: torch.dtype,
    destination_dtype: torch.dtype,
) -> bool:
    """Trial-launch the mixed-dtype overload once per device and dtype pair."""
    device = torch.device(
        device_type, *(() if device_index is None else (device_index,))
    )
    probe = torch.zeros((1, 1), dtype=destination_dtype, device=device)
    operand = torch.zeros((1, 1), dtype=operand_dtype, device=device)
    try:
        torch.addmm(
            probe,
            operand,
            operand,
            beta=0,
            out_dtype=destination_dtype,
            out=probe,
        )
    except NotImplementedError:
        return False
    except RuntimeError as exc:
        if "could not run" in str(exc).lower():
            return False
        raise
    return True


def _addmm_supports_out_dtype(
    destination: torch.Tensor,
    dlogits: torch.Tensor,
    hidden: torch.Tensor,
) -> bool:
    """False selects the ``mm`` fallback; the overload exists but may be unbuilt."""
    if destination.dtype != torch.float32 or dlogits.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        return False
    if destination.device.type != "cuda" or torch.version.hip is not None:
        return False
    if not hasattr(torch.ops.aten.addmm, "dtype_out"):
        return False
    return _addmm_out_dtype_is_usable(
        destination.device.type,
        destination.device.index,
        dlogits.dtype,
        destination.dtype,
    )


def claim_cached_gradients(context: object, *, owner: str) -> None:
    """Claim one-shot buffers before their in-place upstream scaling."""
    attribute = "_orda_cached_gradients_consumed"
    if getattr(context, attribute, False):
        raise RuntimeError(
            f"{owner} supports one backward pass per forward because cached "
            "gradients are consumed in place. Recompute the loss instead of "
            "using retain_graph=True."
        )
    setattr(context, attribute, True)


def scale_cached_gradient_(
    gradient: torch.Tensor | None,
    scale: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Scale a saved gradient in place, casting only when required."""
    if gradient is None:
        return None
    gradient.mul_(scale)
    if output_dtype is not None and gradient.dtype != output_dtype:
        return gradient.to(output_dtype)
    return gradient


# ── Loss reduction ───────────────────────────────────────────────────────────
def normalize_gradient_materialization_scale(
    gradient_materialization_scale: torch.Tensor | None,
    storage_dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    if gradient_materialization_scale is None:
        return None
    fp32_tiny = torch.finfo(torch.float32).tiny
    storage_limits = torch.finfo(storage_dtype)
    safe = (
        torch.isfinite(gradient_materialization_scale)
        & (gradient_materialization_scale >= max(fp32_tiny, storage_limits.tiny))
        & (gradient_materialization_scale <= min(1.0 / fp32_tiny, storage_limits.max))
    )
    return torch.where(
        safe,
        gradient_materialization_scale,
        gradient_materialization_scale.new_ones(()),
    )


def prepare_gradient_materialization_factor(
    labels: torch.Tensor,
    ignore_index: int,
    reduction: str,
    gradient_materialization_scale: torch.Tensor | None,
    storage_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Prepare the optional factor applied when dlogits are materialized.

    Mean reduction follows the gradient GEMMs so FP16/BF16 dlogits retain
    their unscaled signal. A requested ``scale / N`` is materialized only
    when safe; otherwise the neutral factor defers reduction to backward.
    """
    if gradient_materialization_scale is None:
        return None, None, None
    requested_scale = gradient_materialization_scale
    gradient_materialization_scale = normalize_gradient_materialization_scale(
        gradient_materialization_scale
    )
    assert gradient_materialization_scale is not None
    scale_safe = gradient_materialization_scale == requested_scale
    if reduction == "mean":
        denominator = torch.clamp((labels != ignore_index).sum(), min=1)
        reduction_scale = denominator.to(torch.float32).reciprocal()
        factor = gradient_materialization_scale * reduction_scale
        neutral_scale = denominator.to(torch.float32)
    else:
        denominator = None
        factor = gradient_materialization_scale
        neutral_scale = gradient_materialization_scale.new_ones(())

    storage_limits = torch.finfo(storage_dtype)
    factor_safe = (
        scale_safe
        & torch.isfinite(factor)
        & (factor >= storage_limits.tiny)
        & (factor <= storage_limits.max)
    )
    effective_scale = torch.where(
        factor_safe,
        gradient_materialization_scale,
        neutral_scale,
    )
    effective_factor = torch.where(
        factor_safe,
        factor,
        factor.new_ones(()),
    )
    return effective_factor, denominator, effective_scale


def reduce_loss_buffers(
    labels: torch.Tensor,
    ignore_index: int,
    reduction: str,
    *buffers: torch.Tensor,
    mean_denominator: torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    if not buffers:
        raise ValueError("at least one loss buffer is required")
    totals = tuple(buffer.sum() for buffer in buffers)
    if reduction == "mean":
        denominator = (
            torch.clamp((labels != ignore_index).sum(), min=1)
            if mean_denominator is None
            else mean_denominator
        )
        mean_scale = denominator.to(torch.float32).reciprocal()
        return tuple(total / denominator for total in totals), mean_scale
    return totals, totals[0].new_ones(())


# ── Low-level input validation ───────────────────────────────────────────────
def validate_dual_logit_inputs(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    labels: torch.Tensor,
    loss_buffers: tuple[torch.Tensor, ...],
    *,
    n_rows: int,
) -> torch.Tensor:
    if logits_student.ndim != 2:
        raise ValueError(
            f"logits_student_chunk must be 2D, got {tuple(logits_student.shape)}"
        )
    if logits_teacher.ndim != 2:
        raise ValueError(
            f"logits_teacher_chunk must be 2D, got {tuple(logits_teacher.shape)}"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels_chunk must be 1D, got {tuple(labels.shape)}")
    if any(buffer.ndim != 1 for buffer in loss_buffers):
        raise ValueError("loss buffers must be 1D tensors")
    if not logits_student.is_cuda:
        raise ValueError(
            f"logits_student_chunk must be a CUDA/HIP tensor, got {logits_student.device}"
        )
    if logits_teacher.device != logits_student.device:
        raise ValueError(
            "logits_teacher_chunk and logits_student_chunk must be on the same device"
        )
    if labels.device != logits_student.device:
        raise ValueError("labels_chunk and logits chunks must be on the same device")
    if any(buffer.device != logits_student.device for buffer in loss_buffers):
        raise ValueError("loss buffers must be on the same device as logits chunks")
    if not logits_student.is_contiguous():
        raise ValueError(
            "logits_student_chunk must be contiguous because gradients are written in-place"
        )
    if not logits_teacher.is_contiguous():
        raise ValueError("logits_teacher_chunk must be contiguous")
    if any(not buffer.is_contiguous() for buffer in loss_buffers):
        raise ValueError("loss buffers must be contiguous")

    student_rows, vocab_size = logits_student.shape
    if student_rows != n_rows:
        raise ValueError(
            f"logits_student_chunk rows must equal n_rows, got {student_rows} and {n_rows}"
        )
    if logits_teacher.shape != (n_rows, vocab_size):
        raise ValueError(
            f"logits_teacher_chunk must have shape ({n_rows}, {vocab_size}), "
            f"got {tuple(logits_teacher.shape)}"
        )
    if labels.shape[0] != n_rows:
        raise ValueError(f"labels_chunk must have {n_rows} rows, got {labels.shape[0]}")
    if any(buffer.shape[0] != n_rows for buffer in loss_buffers):
        raise ValueError("loss buffers must have n_rows elements")
    return labels.contiguous()
