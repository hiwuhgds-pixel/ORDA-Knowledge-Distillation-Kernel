import torch
from torch import nn

try:
    import triton
    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    triton = None  # type: ignore[assignment]
    _HAS_TRITON = False

from .._runtime import is_hip
from ..utils.resolver import DEFAULT_MAX_FUSED_SIZE, is_power_of_two, resolve_chunk_size
from .kernels import _exact_ce_fwdbwd_kernel_merged
from .kl_kernel import kl_from_logits_chunk as _kl_from_logits_chunk_triton

from .quant import (
    quantize_grad_w as _quantize_grad_w,
    dequantize_grad_w as _dequantize_grad_w,
    quantize_rowwise_int8 as _quantize_rowwise_int8,
    quantize_rowwise_int8_stochastic as _quantize_rowwise_int8_stochastic
)

_VALID_TEACHER_MODES = ("tied", "separate", "precomputed")


def _resolve_teacher_loss_weight(mode: str, override: float | None) -> float:
    """Resolve effective teacher_loss_weight per mode."""
    if override is not None:
        return float(override)
    return 1.0 if mode == "tied" else 0.0


# ── Custom Autograd Function ──────────────────────────────────────────────────

class DistillCEFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        h_student,
        h_teacher,
        weight,
        target,
        lambda_student       = 1.0,
        ignore_index         = -100,
        reduction            = "mean",
        label_smoothing      = 0.0,
        chunk_size           = None,
        use_int8_quant       = None,
        use_stochastic_quant = None,
        use_fast_math_exp    = None,
        use_fast_math_log    = None,
        use_fast_math_mul    = None,
        use_online_softmax   = None,
        use_fp32_accum       = None,
        use_kl_in_kernel     = None,
        kl_weight            = 0.0,
        kl_temperature       = 1.0,
        teacher_mode         = "tied",
        weight_teacher       = None,
        logits_teacher       = None,
        teacher_loss_weight  = 1.0,
        max_chunks           = None,
        max_fused_size       = DEFAULT_MAX_FUSED_SIZE,
        stochastic_seed      = None,
    ):
        BT, H  = h_student.shape
        V      = weight.shape[0]
        device = h_student.device

        if not _HAS_TRITON:
            raise RuntimeError(
                "distill_cross_entropy requires Triton and CUDA/HIP tensors. "
                "Use distillation_loss(..., backend='auto' or 'torch') for fallback execution."
            )

        student_only = (teacher_loss_weight == 0.0)
        mode_precomputed = (teacher_mode == "precomputed")
        mode_separate    = (teacher_mode == "separate")

        h_student = h_student.contiguous()
        weight    = weight.contiguous()
        target    = target.contiguous()
        if not mode_precomputed:
            h_teacher = h_teacher.contiguous()
        if mode_separate:
            weight_teacher = weight_teacher.contiguous()
        if mode_precomputed:
            logits_teacher = logits_teacher.contiguous()

        compute_dtype = torch.bfloat16 if is_hip() else torch.float16
        weight_cast   = weight.to(compute_dtype)
        h_student_c   = h_student if h_student.dtype == compute_dtype else h_student.to(compute_dtype)
        if mode_separate:
            weight_teacher_cast = weight_teacher.to(compute_dtype)
            h_teacher_c = h_teacher if h_teacher.dtype == compute_dtype else h_teacher.to(compute_dtype)
        elif mode_precomputed:
            weight_teacher_cast = None
            logits_teacher_cast = logits_teacher if logits_teacher.dtype == compute_dtype else logits_teacher.to(compute_dtype)
            h_teacher_c = None
        else:  # tied
            weight_teacher_cast = None
            h_teacher_c = h_teacher if h_teacher.dtype == compute_dtype else h_teacher.to(compute_dtype)

        # ── Determine which grad buffers to allocate ─────────────────────────
        # In tied mode: weight provides both student and teacher logits.
        # In separate mode: weight is student-only; weight_teacher may also need grad if teacher_loss_weight > 0.
        # In precomputed mode: no teacher hidden states → grad_h_teacher always None.
        compute_grad = (
            h_student.requires_grad
            or weight.requires_grad
            or (mode_separate and (h_teacher.requires_grad or weight_teacher.requires_grad))
            or (teacher_mode == "tied" and h_teacher.requires_grad)
        )
        grad_h_student = torch.zeros_like(h_student) if h_student.requires_grad else None
        if teacher_mode == "tied":
            grad_h_teacher = torch.zeros_like(h_teacher) if h_teacher.requires_grad else None
        elif mode_separate:
            grad_h_teacher = torch.zeros_like(h_teacher) if h_teacher.requires_grad else None
        else:  # precomputed
            grad_h_teacher = None

        grad_W           = None
        need_grad_W      = weight.requires_grad
        grad_W_teacher   = None
        need_grad_W_teacher = (
            mode_separate
            and weight_teacher is not None
            and weight_teacher.requires_grad
            and not student_only
        )

        BLOCK_SIZE = min(int(max_fused_size), triton.next_power_of_2(V))
        num_warps  = 32 if not is_hip() else 16

        n_non_ignore = (target != ignore_index).sum().item()
        # 1/n_non_ignore is NOT embedded here to avoid fp16 underflow in the kernel; mean
        # normalisation is applied in fp32 inside backward() via ctx.mean_scale instead.
        student_scale = float(lambda_student)
        # In tied/separate with teacher_loss_weight > 0, teacher CE contributes scaled by teacher_loss_weight.
        # The kernel multiplies logit gradients by teacher_scale; we fold teacher_loss_weight in here so
        # the GEMM backward correctly scales W_teacher / W_tied teacher contribution.
        teacher_scale = float(teacher_loss_weight)

        chunk_size_actual, num_chunks = resolve_chunk_size(BT, chunk_size, V=V, max_chunks=max_chunks)
        actual_use_exp    = False if use_fast_math_exp  is None else use_fast_math_exp
        actual_use_log    = False if use_fast_math_log  is None else use_fast_math_log
        actual_use_mul    = False if use_fast_math_mul  is None else use_fast_math_mul
        actual_use_online = False if use_online_softmax is None else use_online_softmax
        actual_use_fp32_accum = False if use_fp32_accum is None else use_fp32_accum
        actual_use_kl_in_kernel = True if use_kl_in_kernel is None else use_kl_in_kernel

        loss_student_1d = torch.zeros(BT, dtype=torch.float32, device=device)
        loss_teacher_1d = torch.zeros(BT, dtype=torch.float32, device=device)
        kl_accum        = torch.zeros(1,  dtype=torch.float32, device=device)
        compute_kl      = kl_weight > 0.0 and actual_use_kl_in_kernel

        with torch.no_grad():
            for chunk_id in range(num_chunks):
                start  = chunk_id * chunk_size_actual
                end    = min(start + chunk_size_actual, BT)
                n_rows = end - start

                h_chunk_student = h_student_c[start:end]
                t_c             = target[start:end]

                # ── Build logits_chunk per mode ──────────────────────────────
                if teacher_mode == "tied":
                    h_chunk_teacher = h_teacher_c[start:end]
                    h_chunk_concat = torch.cat([h_chunk_student, h_chunk_teacher], dim=0)
                    logits_chunk   = h_chunk_concat @ weight_cast.t()
                elif mode_separate:
                    h_chunk_teacher = h_teacher_c[start:end]
                    logits_s_chunk = h_chunk_student @ weight_cast.t()
                    logits_t_chunk = h_chunk_teacher @ weight_teacher_cast.t()
                    logits_chunk = torch.cat([logits_s_chunk, logits_t_chunk], dim=0)
                    del logits_s_chunk, logits_t_chunk
                    h_chunk_concat = None  # not used for GEMM bwd in separate mode
                else:  # precomputed
                    logits_s_chunk = h_chunk_student @ weight_cast.t()
                    logits_t_chunk = logits_teacher_cast[start:end]
                    logits_chunk = torch.cat([logits_s_chunk, logits_t_chunk], dim=0)
                    del logits_s_chunk
                    h_chunk_concat = None
                    h_chunk_teacher = None

                # ── KL distillation (Triton kernel — reuse logits_chunk) ─────
                grad_kl_student = None
                if compute_kl:
                    kl_chunk, grad_kl_student = _kl_from_logits_chunk_triton(
                        logits_chunk,
                        t_c,
                        n_rows=n_rows,
                        kl_weight=kl_weight,
                        kl_temperature=kl_temperature,
                        n_non_ignore=n_non_ignore,
                        ignore_index=ignore_index,
                        use_fast_math_exp=actual_use_exp,
                        use_fast_math_log=actual_use_log,
                        use_fast_math_mul=actual_use_mul,
                        use_online_softmax=actual_use_online,
                        max_fused_size=max_fused_size,
                        reduction="sum",  # always sum; mean normalisation applied in backward at fp32
                    )
                    kl_accum += kl_chunk
                    if not compute_grad:
                        grad_kl_student = None

                # ── CE Kernel ────────────────────────────────────────────────
                # student_only → grid [n_rows], else [2*n_rows].
                grid_rows = n_rows if student_only else 2 * n_rows
                loss_buf  = torch.empty(grid_rows, dtype=torch.float32, device=device)

                _exact_ce_fwdbwd_kernel_merged[(grid_rows,)](
                    X_ptr=logits_chunk,   X_stride=logits_chunk.stride(0),
                    Y_ptr=t_c,            Y_stride=t_c.stride(0),
                    L_ptr=loss_buf,
                    n_cols=V,
                    n_rows=n_rows,
                    ignore_index=ignore_index,
                    label_smoothing=label_smoothing,
                    student_scale=student_scale,
                    teacher_scale=teacher_scale,
                    ONLINE_SOFTMAX=actual_use_online,
                    FAST_MATH_EXP=actual_use_exp,
                    FAST_MATH_LOG=actual_use_log,
                    FAST_MATH_MUL=actual_use_mul,
                    STUDENT_ONLY=student_only,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=num_warps,
                )

                loss_student_1d[start:end] = loss_buf[:n_rows]
                if not student_only:
                    loss_teacher_1d[start:end] = loss_buf[n_rows:]

                # ── Gradient computation ─────────────────────────────────────
                if compute_grad:
                    # Merge KL student grad into student logits.
                    if grad_kl_student is not None:
                        logits_chunk[:n_rows].add_(grad_kl_student.to(logits_chunk.dtype))

                    if teacher_mode == "tied":
                        # Tied path: existing behavior. GEMM bwd over full logits_chunk × h_concat.
                        if grad_h_student is not None or grad_h_teacher is not None:
                            grad_h_cat = logits_chunk @ weight_cast
                            if grad_h_student is not None:
                                grad_h_student[start:end] = grad_h_cat[:n_rows]
                            if grad_h_teacher is not None:
                                grad_h_teacher[start:end] = grad_h_cat[n_rows:]
                            del grad_h_cat

                        if need_grad_W:
                            grad_src = logits_chunk if not student_only else logits_chunk[:n_rows]
                            h_src    = h_chunk_concat if not student_only else h_chunk_student
                            if actual_use_fp32_accum:
                                chunk_contrib = torch.mm(grad_src.t(), h_src).float()
                                if grad_W is None:
                                    grad_W = chunk_contrib
                                else:
                                    grad_W.add_(chunk_contrib)
                            else:
                                if grad_W is None:
                                    grad_W = grad_src.t() @ h_src
                                else:
                                    grad_W.addmm_(grad_src.t(), h_src)

                    elif mode_separate:
                        # Student-side: always from student grad slice × student hidden.
                        if grad_h_student is not None:
                            grad_h_student[start:end] = logits_chunk[:n_rows] @ weight_cast

                        if need_grad_W:
                            if actual_use_fp32_accum:
                                chunk_contrib = torch.mm(logits_chunk[:n_rows].t(), h_chunk_student).float()
                                if grad_W is None:
                                    grad_W = chunk_contrib
                                else:
                                    grad_W.add_(chunk_contrib)
                            else:
                                if grad_W is None:
                                    grad_W = logits_chunk[:n_rows].t() @ h_chunk_student
                                else:
                                    grad_W.addmm_(logits_chunk[:n_rows].t(), h_chunk_student)

                        # Teacher-side: only when teacher_loss_weight > 0.
                        if not student_only:
                            if grad_h_teacher is not None:
                                grad_h_teacher[start:end] = logits_chunk[n_rows:] @ weight_teacher_cast

                            if need_grad_W_teacher:
                                if actual_use_fp32_accum:
                                    chunk_contrib = torch.mm(logits_chunk[n_rows:].t(), h_chunk_teacher).float()
                                    if grad_W_teacher is None:
                                        grad_W_teacher = chunk_contrib
                                    else:
                                        grad_W_teacher.add_(chunk_contrib)
                                else:
                                    if grad_W_teacher is None:
                                        grad_W_teacher = logits_chunk[n_rows:].t() @ h_chunk_teacher
                                    else:
                                        grad_W_teacher.addmm_(logits_chunk[n_rows:].t(), h_chunk_teacher)

                    else:  # precomputed
                        # Only student backprop. logits_teacher has no upstream tensor.
                        if grad_h_student is not None:
                            grad_h_student[start:end] = logits_chunk[:n_rows] @ weight_cast

                        if need_grad_W:
                            if actual_use_fp32_accum:
                                chunk_contrib = torch.mm(logits_chunk[:n_rows].t(), h_chunk_student).float()
                                if grad_W is None:
                                    grad_W = chunk_contrib
                                else:
                                    grad_W.add_(chunk_contrib)
                            else:
                                if grad_W is None:
                                    grad_W = logits_chunk[:n_rows].t() @ h_chunk_student
                                else:
                                    grad_W.addmm_(logits_chunk[:n_rows].t(), h_chunk_student)

                del logits_chunk, loss_buf
                if h_chunk_concat is not None:
                    del h_chunk_concat

        grad_W_a       = None
        grad_W_scale   = None
        grad_W_target  = None
        unique_targets = None

        actual_use_int8 = False if use_int8_quant is None else use_int8_quant
        actual_use_stoc = False if use_stochastic_quant is None else use_stochastic_quant

        if grad_W is not None:
            if actual_use_int8:
                if actual_use_stoc and stochastic_seed is not None:
                    generator = torch.Generator(device=grad_W.device)
                    generator.manual_seed(int(stochastic_seed))

                    def _qfn(tensor):
                        return _quantize_rowwise_int8_stochastic(tensor, generator=generator)
                else:
                    _qfn = _quantize_rowwise_int8_stochastic if actual_use_stoc else _quantize_rowwise_int8
                grad_W_a, grad_W_scale, grad_W_target, unique_targets = \
                    _quantize_grad_w(grad_W, target, ignore_index, quantize_fn=_qfn)
            else:
                grad_W_a = grad_W

        # Teacher grad_W (separate mode): not quantized.

        if reduction == "mean":
            denom  = max(n_non_ignore, 1)
            loss_s = loss_student_1d.sum() / denom
            loss_t = loss_teacher_1d.sum() / denom
        else:
            denom  = 1
            loss_s = loss_student_1d.sum()
            loss_t = loss_teacher_1d.sum()

        kl_loss_raw = kl_accum[0] if compute_kl else loss_s.new_zeros(1).squeeze()
        kl_loss = kl_loss_raw / denom if reduction == "mean" else kl_loss_raw

        loss = loss_s * float(lambda_student) + loss_t * float(teacher_loss_weight) + kl_weight * kl_loss

        ctx.save_for_backward(
            grad_h_student, grad_h_teacher,
            grad_W_a, grad_W_scale, grad_W_target, unique_targets,
            grad_W_teacher,
        )
        ctx.compute_grad   = compute_grad
        ctx.use_int8_quant = actual_use_int8
        ctx.teacher_mode   = teacher_mode
        ctx.mean_scale     = 1.0 / denom  # fp32 mean normalisation; not embedded in kernel

        return loss, loss_s.detach(), loss_t.detach(), kl_loss.detach()

    @staticmethod
    def backward(ctx, grad_output, _gs=None, _gt=None, _gkl=None):
        (grad_h_student, grad_h_teacher,
         grad_W_a, grad_W_scale, grad_W_target, unique_targets,
         grad_W_teacher) = ctx.saved_tensors

        if grad_output is None or not ctx.compute_grad:
            return (None,) * 26

        effective_grad = grad_output * ctx.mean_scale

        if grad_h_student is not None:
            grad_h_student = grad_h_student * effective_grad
        if grad_h_teacher is not None:
            grad_h_teacher = grad_h_teacher * effective_grad

        grad_W = None
        if grad_W_a is not None:
            if ctx.use_int8_quant:
                grad_W = _dequantize_grad_w(
                    grad_W_a, grad_W_scale, grad_W_target, unique_targets, effective_grad)
            else:
                grad_W = grad_W_a * effective_grad

        if grad_W_teacher is not None:
            grad_W_teacher = grad_W_teacher * effective_grad

        # Arg slot order in forward (after ctx):
        # 0: h_student, 1: h_teacher, 2: weight, 3: target,
        # 4: lambda_student, 5: ignore_index, 6: reduction, 7: label_smoothing, 8: chunk_size,
        # 9-15: flag args, 16: kl_weight, 17: kl_temperature,
        # 18: teacher_mode, 19: weight_teacher, 20: logits_teacher, 21: teacher_loss_weight.
        # 22: max_chunks, 23: max_fused_size, 24: stochastic_seed.
        return (
            grad_h_student, grad_h_teacher, grad_W, None,
            None, None, None, None, None,
            None, None, None, None, None, None, None,
            None, None, None,
            None, grad_W_teacher, None, None, None, None, None,
        )



# ── Public API ────────────────────────────────────────────────────────────────

@torch._dynamo.disable
def distill_cross_entropy(
    h_student:            torch.Tensor,
    h_teacher:            torch.Tensor | None,
    weight:               torch.Tensor,
    target:               torch.Tensor,
    lambda_student:       float            = 1.0,
    ignore_index:         int              = -100,
    reduction:            str              = "mean",
    label_smoothing:      float            = 0.0,
    chunk_size:           str | int | None = None,
    use_int8_quant:       bool | None      = None,
    use_stochastic_quant: bool | None      = None,
    use_fast_math_exp:    bool | None      = None,
    use_fast_math_log:    bool | None      = None,
    use_fast_math_mul:    bool | None      = None,
    use_online_softmax:   bool | None      = None,
    use_fp32_accum:       bool | None      = None,
    use_kl_in_kernel:     bool | None      = None,
    kl_weight:            float            = 0.0,
    kl_temperature:       float            = 1.0,
    teacher_mode:         str | None       = None,
    weight_teacher:       torch.Tensor | None = None,
    logits_teacher:       torch.Tensor | None = None,
    teacher_loss_weight:  float | None     = None,
    max_chunks:           int | None       = None,
    max_fused_size:       int              = DEFAULT_MAX_FUSED_SIZE,
    stochastic_seed:      int | None       = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # ── Resolve defaults ────────────────────────────────────────────────────────
    _ignore_index    = ignore_index
    _label_smoothing = label_smoothing
    _teacher_mode    = "tied" if teacher_mode is None else teacher_mode
    if _teacher_mode not in _VALID_TEACHER_MODES:
        raise ValueError(
            f"teacher_mode must be one of {_VALID_TEACHER_MODES}, got {_teacher_mode!r}"
        )
    _teacher_loss_weight = _resolve_teacher_loss_weight(_teacher_mode, teacher_loss_weight)

    # ── Input Validation ─────────────────────────────────────────────────────────
    if not (0.0 <= _label_smoothing <= 1.0):
        raise ValueError(
            f"label_smoothing must be in [0.0, 1.0], got {_label_smoothing}"
        )
    if reduction not in ("mean", "sum"):
        raise ValueError(
            f"reduction must be 'mean' or 'sum', got {reduction!r}"
        )
    if kl_weight < 0.0:
        raise ValueError(f"kl_weight must be >= 0.0, got {kl_weight}")
    if kl_temperature <= 0.0:
        raise ValueError(f"kl_temperature must be > 0.0, got {kl_temperature}")
    if _teacher_loss_weight < 0.0:
        raise ValueError(f"teacher_loss_weight must be >= 0.0, got {_teacher_loss_weight}")
    if max_chunks is not None and max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")
    if max_fused_size < 1:
        raise ValueError(f"max_fused_size must be >= 1, got {max_fused_size}")
    if not is_power_of_two(int(max_fused_size)):
        raise ValueError(f"max_fused_size must be a power of two, got {max_fused_size}")
    if stochastic_seed is not None and int(stochastic_seed) < 0:
        raise ValueError(f"stochastic_seed must be >= 0, got {stochastic_seed}")

    if h_student.ndim != 2:
        raise ValueError(f"h_student must be 2D, got shape={tuple(h_student.shape)}")
    BT, H = h_student.shape
    device = h_student.device
    if weight.ndim != 2 or weight.shape[1] != H:
        raise ValueError(f"weight must have shape (V, {H}), got {tuple(weight.shape)}")
    if target.shape != (BT,):
        raise ValueError(f"target must have shape ({BT},), got {tuple(target.shape)}")
    if target.dtype == torch.bool or target.is_floating_point():
        raise ValueError(f"target must be an integer class-index tensor, got dtype={target.dtype}")

    V = weight.shape[0]
    valid_mask = (target == _ignore_index) | ((target >= 0) & (target < V))
    if not valid_mask.all():
        bad_vals = target[~valid_mask].tolist()[:5]
        raise ValueError(
            f"target contains values outside [0, {V}) "
            f"and not equal to ignore_index={_ignore_index}: {bad_vals}"
        )

    # ── Mode-specific validation ────────────────────────────────────────────────
    if _teacher_mode == "tied":
        if weight_teacher is not None:
            raise ValueError(
                "teacher_mode='tied' does not accept weight_teacher"
            )
        if logits_teacher is not None:
            raise ValueError(
                "teacher_mode='tied' does not accept logits_teacher"
            )
        if h_teacher is None:
            raise ValueError("teacher_mode='tied' requires h_teacher")
        if h_teacher.shape != (BT, H):
            raise ValueError(f"h_teacher.shape={tuple(h_teacher.shape)} does not match h_student.shape={(BT, H)}")
        if h_teacher.device != device:
            raise ValueError(f"h_teacher must be on the same device as h_student ({device}), got {h_teacher.device}")

    elif _teacher_mode == "separate":
        if weight_teacher is None:
            raise ValueError("teacher_mode='separate' requires weight_teacher")
        if logits_teacher is not None:
            raise ValueError(
                "teacher_mode='separate' does not accept logits_teacher"
            )
        if h_teacher is None:
            raise ValueError("teacher_mode='separate' requires h_teacher")
        H_t = h_teacher.shape[1] if h_teacher.ndim == 2 else None
        if h_teacher.shape[0] != BT or H_t is None:
            raise ValueError(f"h_teacher must have shape (BT, H_teacher), got {tuple(h_teacher.shape)}")
        if weight_teacher.ndim != 2 or weight_teacher.shape != (V, H_t):
            raise ValueError(
                f"weight_teacher must have shape ({V}, {H_t}), got {tuple(weight_teacher.shape)}"
            )
        if h_teacher.device != device or weight_teacher.device != device:
            raise ValueError("h_teacher and weight_teacher must be on the same device as h_student")

    else:  # precomputed
        if logits_teacher is None:
            raise ValueError("teacher_mode='precomputed' requires logits_teacher [BT, V]")
        if weight_teacher is not None:
            raise ValueError(
                "teacher_mode='precomputed' does not accept weight_teacher"
            )
        if logits_teacher.shape != (BT, V):
            raise ValueError(
                f"logits_teacher must have shape ({BT}, {V}), got {tuple(logits_teacher.shape)}"
            )
        if logits_teacher.device != device:
            raise ValueError(
                f"logits_teacher must be on the same device as h_student ({device}), got {logits_teacher.device}"
            )
        if logits_teacher.requires_grad:
            raise ValueError(
                "teacher_mode='precomputed' requires logits_teacher without gradients; "
                "use teacher_mode='separate' when the teacher path needs gradients"
            )

    if device.type != "cuda":
        raise ValueError(f"tensors must be CUDA/HIP tensors, got device={device}")
    if target.device != device:
        raise ValueError(f"target must be on the same device as h_student ({device}), got {target.device}")
    if weight.device != device:
        raise ValueError(f"weight must be on the same device as h_student ({device}), got {weight.device}")
    if _teacher_mode in ("tied", "separate") and h_teacher.device != device:
        raise ValueError(f"h_teacher must be on the same device as h_student ({device}), got {h_teacher.device}")
    if _teacher_mode == "separate" and weight_teacher.device != device:
        raise ValueError(f"weight_teacher must be on the same device as h_student ({device}), got {weight_teacher.device}")
    if _teacher_mode == "precomputed" and logits_teacher.device != device:
        raise ValueError(
            f"logits_teacher must be on the same device as h_student ({device}), got {logits_teacher.device}"
        )

    # h_teacher must be a placeholder (autograd Function requires Tensor, not None).
    if h_teacher is None:
        h_teacher = h_student.new_empty(0)

    return DistillCEFunction.apply(
        h_student, h_teacher, weight, target,
        lambda_student, _ignore_index, reduction,
        _label_smoothing, chunk_size,
        use_int8_quant, use_stochastic_quant,
        use_fast_math_exp, use_fast_math_log, use_fast_math_mul,
        use_online_softmax, use_fp32_accum, use_kl_in_kernel,
        kl_weight, kl_temperature,
        _teacher_mode, weight_teacher, logits_teacher, _teacher_loss_weight,
        max_chunks, max_fused_size, stochastic_seed,
    )


class DistillCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        lambda_student:       float            = 1.0,
        ignore_index:         int              = -100,
        reduction:            str              = "mean",
        label_smoothing:      float            = 0.0,
        chunk_size:           str | int | None = None,
        use_int8_quant:       bool | None      = None,
        use_stochastic_quant: bool | None      = None,
        use_fast_math_exp:    bool | None      = None,
        use_fast_math_log:    bool | None      = None,
        use_fast_math_mul:    bool | None      = None,
        use_online_softmax:   bool | None      = None,
        use_fp32_accum:       bool | None      = None,
        use_kl_in_kernel:     bool | None      = None,
        kl_weight:            float            = 0.0,
        kl_temperature:       float            = 1.0,
        teacher_mode:         str              = "tied",
        teacher_loss_weight:  float | None     = None,
        max_chunks:           int | None       = None,
        max_fused_size:       int              = DEFAULT_MAX_FUSED_SIZE,
        stochastic_seed:      int | None       = None,
    ):
        super().__init__()
        self.options = dict(
            lambda_student=lambda_student,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            chunk_size=chunk_size,
            use_int8_quant=use_int8_quant,
            use_stochastic_quant=use_stochastic_quant,
            use_fast_math_exp=use_fast_math_exp,
            use_fast_math_log=use_fast_math_log,
            use_fast_math_mul=use_fast_math_mul,
            use_online_softmax=use_online_softmax,
            use_fp32_accum=use_fp32_accum,
            use_kl_in_kernel=use_kl_in_kernel,
            kl_weight=kl_weight,
            kl_temperature=kl_temperature,
            teacher_mode=teacher_mode,
            teacher_loss_weight=teacher_loss_weight,
            max_chunks=max_chunks,
            max_fused_size=max_fused_size,
            stochastic_seed=stochastic_seed,
        )

    def forward(
        self,
        h_student:      torch.Tensor,
        h_teacher:      torch.Tensor | None,
        weight:         torch.Tensor,
        target:         torch.Tensor,
        *,
        weight_teacher: torch.Tensor | None = None,
        logits_teacher: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return distill_cross_entropy(
            h_student,
            h_teacher,
            weight,
            target,
            weight_teacher=weight_teacher,
            logits_teacher=logits_teacher,
            **self.options,
        )
