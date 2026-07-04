import torch

from .._runtime import HAS_TRITON, default_compute_dtype, tl, tl_highprec_exp, tl_highprec_log, triton
from ..utils._autotune import SEPARATE_FULL, SEPARATE_STUDENT, default_config, select_config
from ..utils.resolver import DEFAULT_MAX_FUSED_SIZE, is_power_of_two, resolve_chunk_size


# ── Kernel: Student CE + KL Fwd/Bwd ──────────────────────────────────────────
if HAS_TRITON:
    @triton.jit
    def _separate_student_ce_kl_fwdbwd_kernel(
        S_ptr, S_stride,
        T_ptr, T_stride,
        Y_ptr, Y_stride,
        L_ptr,
        K_ptr,
        n_rows,
        n_cols,
        ignore_index,
        student_scale,
        T_inv,
        kl_grad_scale,
        T_sq,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Forward losses and in-place student dlogits for separate teacher."""
        i = tl.program_id(0).to(tl.int64)
        y = tl.load(Y_ptr + i * Y_stride)

        tl.device_assert(
            ((y >= 0) & (y < n_cols)) | (y == ignore_index),
            "labels out of range",
        )
        ignored = y == ignore_index
        y_safe = tl.where(ignored, 0, y)

        row_s = S_ptr + i * S_stride
        row_t = T_ptr + i * T_stride
        offs = tl.arange(0, BLOCK_SIZE)

        m_s = float("-inf")
        d_s_ce = 0.0
        d_s_kl = 0.0
        m_t_kl = float("-inf")
        d_t_kl = 0.0

        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            x_s = tl.load(row_s + cols, mask=mask, other=float("-inf"), eviction_policy="evict_last").to(tl.float32)
            x_t = tl.load(row_t + cols, mask=mask, other=float("-inf"), eviction_policy="evict_last").to(tl.float32)
            x_t_kl = x_t * T_inv

            m_s_new = tl.maximum(m_s, tl.max(x_s))
            m_t_kl_new = tl.maximum(m_t_kl, tl.max(x_t_kl))

            d_s_ce = (
                d_s_ce * tl_highprec_exp(m_s - m_s_new)
                + tl.sum(tl_highprec_exp(x_s - m_s_new))
            )
            d_s_kl = (
                d_s_kl * tl_highprec_exp(T_inv * (m_s - m_s_new))
                + tl.sum(tl_highprec_exp((x_s - m_s_new) * T_inv))
            )
            d_t_kl = (
                d_t_kl * tl_highprec_exp(m_t_kl - m_t_kl_new)
                + tl.sum(tl_highprec_exp(x_t_kl - m_t_kl_new))
            )

            m_s = m_s_new
            m_t_kl = m_t_kl_new

        m_s_kl = m_s * T_inv
        log_d_s_ce = tl_highprec_log(d_s_ce)
        log_d_s_kl = tl_highprec_log(d_s_kl)
        log_d_t_kl = tl_highprec_log(d_t_kl)
        lse_s_ce = m_s + log_d_s_ce
        logit_label_s = tl.load(row_s + y_safe).to(tl.float32)
        tl.store(L_ptr + i, tl.where(ignored, 0.0, lse_s_ce - logit_label_s))

        kl_row = 0.0
        student_row_scale = tl.where(ignored, 0.0, student_scale)

        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            is_label = cols == y_safe

            x_s = tl.load(row_s + cols, mask=mask, other=float("-inf"), eviction_policy="evict_first").to(tl.float32)
            x_t = tl.load(row_t + cols, mask=mask, other=float("-inf"), eviction_policy="evict_first").to(tl.float32)

            v_t_kl = x_t * T_inv - m_t_kl
            log_p_t = v_t_kl - log_d_t_kl
            p_t_kl = tl_highprec_exp(log_p_t)

            v_s_kl = x_s * T_inv - m_s_kl
            log_p_s = v_s_kl - log_d_s_kl
            p_s_kl = tl_highprec_exp(log_p_s)
            kl_row += tl.sum(tl.where(mask, p_t_kl * (log_p_t - log_p_s), 0.0))

            student_grad = tl_highprec_exp(x_s - m_s) / d_s_ce
            student_grad = tl.where(is_label, student_grad - 1.0, student_grad)
            student_grad = student_grad * student_row_scale

            grad_s = student_grad + kl_grad_scale * (p_s_kl - p_t_kl)
            grad_s = tl.where(ignored, 0.0, grad_s)
            tl.store(row_s + cols, grad_s, mask=mask)

        kl_row = tl.where(ignored, 0.0, kl_row)
        tl.store(K_ptr + i, kl_row * T_sq)

else:
    def _separate_student_ce_kl_fwdbwd_kernel(*args, **kwargs):
        raise RuntimeError("Triton kernels are unavailable; install Triton and use CUDA/HIP tensors.")


# ── Kernel: Full CE + KL Fwd/Bwd ─────────────────────────────────────────────
if HAS_TRITON:
    @triton.jit
    def _separate_full_ce_kl_fwdbwd_kernel(
        S_ptr, S_stride,
        T_ptr, T_stride,
        Y_ptr, Y_stride,
        L_ptr,
        T_L_ptr,
        K_ptr,
        n_rows,
        n_cols,
        ignore_index,
        student_scale,
        teacher_scale,
        T_inv,
        kl_grad_scale,
        T_sq,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Forward losses and in-place dlogits for separate full distillation."""
        i = tl.program_id(0).to(tl.int64)
        y = tl.load(Y_ptr + i * Y_stride)

        tl.device_assert(
            ((y >= 0) & (y < n_cols)) | (y == ignore_index),
            "labels out of range",
        )
        ignored = y == ignore_index
        y_safe = tl.where(ignored, 0, y)

        row_s = S_ptr + i * S_stride
        row_t = T_ptr + i * T_stride
        offs = tl.arange(0, BLOCK_SIZE)

        m_s = float("-inf")
        d_s_ce = 0.0
        d_s_kl = 0.0
        m_t = float("-inf")
        d_t_ce = 0.0
        d_t_kl = 0.0

        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            x_s = tl.load(row_s + cols, mask=mask, other=float("-inf"), eviction_policy="evict_last").to(tl.float32)
            x_t = tl.load(row_t + cols, mask=mask, other=float("-inf"), eviction_policy="evict_last").to(tl.float32)

            m_s_new = tl.maximum(m_s, tl.max(x_s))
            m_t_new = tl.maximum(m_t, tl.max(x_t))

            d_s_ce = (
                d_s_ce * tl_highprec_exp(m_s - m_s_new)
                + tl.sum(tl_highprec_exp(x_s - m_s_new))
            )
            d_s_kl = (
                d_s_kl * tl_highprec_exp(T_inv * (m_s - m_s_new))
                + tl.sum(tl_highprec_exp((x_s - m_s_new) * T_inv))
            )
            d_t_ce = (
                d_t_ce * tl_highprec_exp(m_t - m_t_new)
                + tl.sum(tl_highprec_exp(x_t - m_t_new))
            )
            d_t_kl = (
                d_t_kl * tl_highprec_exp(T_inv * (m_t - m_t_new))
                + tl.sum(tl_highprec_exp((x_t - m_t_new) * T_inv))
            )

            m_s = m_s_new
            m_t = m_t_new

        m_s_kl = m_s * T_inv
        m_t_kl = m_t * T_inv
        log_d_s_ce = tl_highprec_log(d_s_ce)
        log_d_s_kl = tl_highprec_log(d_s_kl)
        log_d_t_kl = tl_highprec_log(d_t_kl)
        lse_s_ce = m_s + log_d_s_ce
        logit_label_s = tl.load(row_s + y_safe).to(tl.float32)
        tl.store(L_ptr + i, tl.where(ignored, 0.0, lse_s_ce - logit_label_s))

        lse_t_ce = m_t + tl_highprec_log(d_t_ce)
        logit_label_t = tl.load(row_t + y_safe).to(tl.float32)
        tl.store(T_L_ptr + i, tl.where(ignored, 0.0, lse_t_ce - logit_label_t))

        kl_row = 0.0
        student_row_scale = tl.where(ignored, 0.0, student_scale)
        teacher_row_scale = tl.where(ignored, 0.0, teacher_scale)

        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            is_label = cols == y_safe

            x_s = tl.load(row_s + cols, mask=mask, other=float("-inf"), eviction_policy="evict_first").to(tl.float32)
            x_t = tl.load(row_t + cols, mask=mask, other=float("-inf"), eviction_policy="evict_first").to(tl.float32)

            teacher_grad = tl_highprec_exp(x_t - m_t) / d_t_ce
            teacher_grad = tl.where(is_label, teacher_grad - 1.0, teacher_grad)
            teacher_grad = teacher_grad * teacher_row_scale
            tl.store(row_t + cols, teacher_grad, mask=mask)

            v_t_kl = x_t * T_inv - m_t_kl
            log_p_t = v_t_kl - log_d_t_kl
            p_t_kl = tl_highprec_exp(log_p_t)

            v_s_kl = x_s * T_inv - m_s_kl
            log_p_s = v_s_kl - log_d_s_kl
            p_s_kl = tl_highprec_exp(log_p_s)
            kl_row += tl.sum(tl.where(mask, p_t_kl * (log_p_t - log_p_s), 0.0))

            student_grad = tl_highprec_exp(x_s - m_s) / d_s_ce
            student_grad = tl.where(is_label, student_grad - 1.0, student_grad)
            student_grad = student_grad * student_row_scale

            grad_s = student_grad + kl_grad_scale * (p_s_kl - p_t_kl)
            grad_s = tl.where(ignored, 0.0, grad_s)
            tl.store(row_s + cols, grad_s, mask=mask)

        kl_row = tl.where(ignored, 0.0, kl_row)
        tl.store(K_ptr + i, kl_row * T_sq)

else:
    def _separate_full_ce_kl_fwdbwd_kernel(*args, **kwargs):
        raise RuntimeError("Triton kernels are unavailable; install Triton and use CUDA/HIP tensors.")


# ── Python wrappers ──────────────────────────────────────────────────────────
def _separate_student_fwdbwd(
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
) -> None:
    """Forward + in-place student backward for separate student-only distillation."""
    if not HAS_TRITON:
        raise RuntimeError("_separate_student_fwdbwd requires Triton and CUDA/HIP tensors.")
    if logits_student_chunk.ndim != 2:
        raise ValueError(f"logits_student_chunk must be 2D, got {tuple(logits_student_chunk.shape)}")
    if logits_teacher_chunk.ndim != 2:
        raise ValueError(f"logits_teacher_chunk must be 2D, got {tuple(logits_teacher_chunk.shape)}")
    if labels_chunk.ndim != 1:
        raise ValueError(f"labels_chunk must be 1D, got {tuple(labels_chunk.shape)}")
    if student_loss.ndim != 1 or kl_loss.ndim != 1:
        raise ValueError("student_loss and kl_loss must be 1D tensors")
    if not logits_student_chunk.is_cuda:
        raise ValueError(f"logits_student_chunk must be a CUDA/HIP tensor, got {logits_student_chunk.device}")
    if logits_teacher_chunk.device != logits_student_chunk.device:
        raise ValueError("logits_teacher_chunk and logits_student_chunk must be on the same device")
    if labels_chunk.device != logits_student_chunk.device:
        raise ValueError("labels_chunk and logits chunks must be on the same device")
    if student_loss.device != logits_student_chunk.device or kl_loss.device != logits_student_chunk.device:
        raise ValueError("loss buffers must be on the same device as logits chunks")
    if not logits_student_chunk.is_contiguous():
        raise ValueError("logits_student_chunk must be contiguous because gradients are written in-place")
    if not logits_teacher_chunk.is_contiguous():
        raise ValueError("logits_teacher_chunk must be contiguous")
    if not student_loss.is_contiguous() or not kl_loss.is_contiguous():
        raise ValueError("student_loss and kl_loss must be contiguous")

    student_rows, vocab_size = logits_student_chunk.shape
    if n_rows is None:
        n_rows = student_rows
    if student_rows != n_rows:
        raise ValueError(f"logits_student_chunk rows must equal n_rows, got {student_rows} and {n_rows}")
    if logits_teacher_chunk.shape != (n_rows, vocab_size):
        raise ValueError(
            f"logits_teacher_chunk must have shape ({n_rows}, {vocab_size}), "
            f"got {tuple(logits_teacher_chunk.shape)}"
        )
    if labels_chunk.shape[0] != n_rows:
        raise ValueError(f"labels_chunk must have {n_rows} rows, got {labels_chunk.shape[0]}")
    if student_loss.shape[0] != n_rows or kl_loss.shape[0] != n_rows:
        raise ValueError("student_loss and kl_loss must have n_rows elements")

    labels_chunk = labels_chunk.contiguous()
    if block_size is None or num_warps is None:
        launch_config = default_config(SEPARATE_STUDENT, vocab_size, int(max_fused_size))
        block_size = launch_config.block_size
        num_warps = launch_config.num_warps

    _separate_student_ce_kl_fwdbwd_kernel[(n_rows,)](
        logits_student_chunk,
        logits_student_chunk.stride(0),
        logits_teacher_chunk,
        logits_teacher_chunk.stride(0),
        labels_chunk,
        labels_chunk.stride(0),
        student_loss,
        kl_loss,
        n_rows,
        vocab_size,
        ignore_index,
        student_scale=float(student_scale),
        T_inv=1.0 / float(kl_temperature),
        kl_grad_scale=float(kl_weight) * float(kl_temperature),
        T_sq=float(kl_temperature) * float(kl_temperature),
        BLOCK_SIZE=int(block_size),
        num_warps=int(num_warps),
    )


def _separate_full_fwdbwd(
    logits_student_chunk: torch.Tensor,
    logits_teacher_chunk: torch.Tensor,
    labels_chunk: torch.Tensor,
    student_loss: torch.Tensor,
    teacher_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    *,
    kl_weight: float,
    kl_temperature: float,
    n_rows: int | None = None,
    ignore_index: int = -100,
    student_scale: float = 1.0,
    teacher_scale: float = 1.0,
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> None:
    """Forward + in-place backward for separate full distillation."""
    if not HAS_TRITON:
        raise RuntimeError("_separate_full_fwdbwd requires Triton and CUDA/HIP tensors.")
    if logits_student_chunk.ndim != 2:
        raise ValueError(f"logits_student_chunk must be 2D, got {tuple(logits_student_chunk.shape)}")
    if logits_teacher_chunk.ndim != 2:
        raise ValueError(f"logits_teacher_chunk must be 2D, got {tuple(logits_teacher_chunk.shape)}")
    if labels_chunk.ndim != 1:
        raise ValueError(f"labels_chunk must be 1D, got {tuple(labels_chunk.shape)}")
    if student_loss.ndim != 1 or teacher_loss.ndim != 1 or kl_loss.ndim != 1:
        raise ValueError("student_loss, teacher_loss and kl_loss must be 1D tensors")
    if not logits_student_chunk.is_cuda:
        raise ValueError(f"logits_student_chunk must be a CUDA/HIP tensor, got {logits_student_chunk.device}")
    if logits_teacher_chunk.device != logits_student_chunk.device:
        raise ValueError("logits_teacher_chunk and logits_student_chunk must be on the same device")
    if labels_chunk.device != logits_student_chunk.device:
        raise ValueError("labels_chunk and logits chunks must be on the same device")
    if (
        student_loss.device != logits_student_chunk.device
        or teacher_loss.device != logits_student_chunk.device
        or kl_loss.device != logits_student_chunk.device
    ):
        raise ValueError("loss buffers must be on the same device as logits chunks")
    if not logits_student_chunk.is_contiguous():
        raise ValueError("logits_student_chunk must be contiguous because gradients are written in-place")
    if not logits_teacher_chunk.is_contiguous():
        raise ValueError("logits_teacher_chunk must be contiguous because gradients are written in-place")
    if not student_loss.is_contiguous() or not teacher_loss.is_contiguous() or not kl_loss.is_contiguous():
        raise ValueError("loss buffers must be contiguous")

    student_rows, vocab_size = logits_student_chunk.shape
    if n_rows is None:
        n_rows = student_rows
    if student_rows != n_rows:
        raise ValueError(f"logits_student_chunk rows must equal n_rows, got {student_rows} and {n_rows}")
    if logits_teacher_chunk.shape != (n_rows, vocab_size):
        raise ValueError(
            f"logits_teacher_chunk must have shape ({n_rows}, {vocab_size}), "
            f"got {tuple(logits_teacher_chunk.shape)}"
        )
    if labels_chunk.shape[0] != n_rows:
        raise ValueError(f"labels_chunk must have {n_rows} rows, got {labels_chunk.shape[0]}")
    if student_loss.shape[0] != n_rows or teacher_loss.shape[0] != n_rows or kl_loss.shape[0] != n_rows:
        raise ValueError("loss buffers must have n_rows elements")

    labels_chunk = labels_chunk.contiguous()
    if block_size is None or num_warps is None:
        launch_config = default_config(SEPARATE_FULL, vocab_size, int(max_fused_size))
        block_size = launch_config.block_size
        num_warps = launch_config.num_warps

    _separate_full_ce_kl_fwdbwd_kernel[(n_rows,)](
        logits_student_chunk,
        logits_student_chunk.stride(0),
        logits_teacher_chunk,
        logits_teacher_chunk.stride(0),
        labels_chunk,
        labels_chunk.stride(0),
        student_loss,
        teacher_loss,
        kl_loss,
        n_rows,
        vocab_size,
        ignore_index,
        student_scale=float(student_scale),
        teacher_scale=float(teacher_scale),
        T_inv=1.0 / float(kl_temperature),
        kl_grad_scale=float(kl_weight) * float(kl_temperature),
        T_sq=float(kl_temperature) * float(kl_temperature),
        BLOCK_SIZE=int(block_size),
        num_warps=int(num_warps),
    )


# ── Autograd Functions ───────────────────────────────────────────────────────
class SeparateStudentCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        student_ce_weight=1.0,
        ignore_index=-100,
        reduction="mean",
        chunk_size=None,
        use_fp32_accum=None,
        kl_weight=0.0,
        kl_temperature=1.0,
        teacher_ce_weight=0.0,
        max_chunks=None,
        max_fused_size=DEFAULT_MAX_FUSED_SIZE,
        autotune=False,
    ):
        # ── Validate mode constraints ────────────────────────────────────────
        if float(teacher_ce_weight) != 0.0:
            raise ValueError("SeparateStudentCEFunction requires teacher_ce_weight=0.0")
        if student_ce_weight < 0.0:
            raise ValueError(f"student_ce_weight must be >= 0.0, got {student_ce_weight}")
        if kl_weight < 0.0:
            raise ValueError(f"kl_weight must be >= 0.0, got {kl_weight}")
        if kl_temperature <= 0.0:
            raise ValueError(f"kl_temperature must be > 0.0, got {kl_temperature}")
        if max_fused_size < 1:
            raise ValueError(f"max_fused_size must be >= 1, got {max_fused_size}")
        if not is_power_of_two(int(max_fused_size)):
            raise ValueError(f"max_fused_size must be a power of two, got {max_fused_size}")

        BT, _ = student_hidden.shape
        V = weight.shape[0]
        device = student_hidden.device

        need_grad_student_hidden = student_hidden.requires_grad
        need_grad_weight = weight.requires_grad

        # ── Normalize inputs and compute dtype ──────────────────────────────
        student_hidden = student_hidden.contiguous()
        teacher_hidden = teacher_hidden.contiguous()
        weight = weight.contiguous()
        teacher_weight = teacher_weight.contiguous()
        if labels.dtype != torch.long:
            raise ValueError(f"labels must have dtype torch.long, got {labels.dtype}")
        labels = labels.contiguous()

        compute_dtype = default_compute_dtype(weight, teacher_weight, student_hidden, teacher_hidden)
        weight_cast = weight.to(compute_dtype)
        teacher_weight_cast = teacher_weight.to(compute_dtype)
        student_hidden_c = student_hidden if student_hidden.dtype == compute_dtype else student_hidden.to(compute_dtype)
        teacher_hidden_c = teacher_hidden if teacher_hidden.dtype == compute_dtype else teacher_hidden.to(compute_dtype)

        # ── Allocate saved gradients ────────────────────────────────────────
        compute_grad = need_grad_student_hidden or need_grad_weight
        grad_student_hidden = torch.zeros_like(student_hidden) if need_grad_student_hidden else None
        grad_weight = None

        # ── Resolve chunking and accumulators ───────────────────────────────
        chunk_size_actual, num_chunks = resolve_chunk_size(BT, chunk_size, V=V, max_chunks=max_chunks)
        actual_use_fp32_accum = False if use_fp32_accum is None else bool(use_fp32_accum)

        loss_student_accum = torch.zeros((), dtype=torch.float32, device=device)
        kl_accum = torch.zeros((), dtype=torch.float32, device=device)

        student_scale = float(student_ce_weight)

        with torch.no_grad():
            for chunk_id in range(num_chunks):
                # ── Slice current chunk ─────────────────────────────────────
                start = chunk_id * chunk_size_actual
                end = min(start + chunk_size_actual, BT)
                n_rows = end - start

                student_hidden_chunk = student_hidden_c[start:end]
                teacher_hidden_chunk = teacher_hidden_c[start:end]
                labels_chunk = labels[start:end]
                bench_fn = None
                if bool(autotune):
                    # ── Autotune trial: fresh GEMM output per config ────────
                    def bench_fn(block_size: int, num_warps: int):
                        logits_s_bench = student_hidden_chunk @ weight_cast.t()
                        logits_t_bench = teacher_hidden_chunk @ teacher_weight_cast.t()
                        student_bench = torch.empty(n_rows, dtype=torch.float32, device=device)
                        kl_bench = torch.empty(n_rows, dtype=torch.float32, device=device)
                        _separate_student_fwdbwd(
                            logits_s_bench,
                            logits_t_bench,
                            labels_chunk,
                            student_bench,
                            kl_bench,
                            n_rows=n_rows,
                            kl_weight=kl_weight,
                            kl_temperature=kl_temperature,
                            ignore_index=ignore_index,
                            student_scale=student_scale,
                            max_fused_size=max_fused_size,
                            block_size=block_size,
                            num_warps=num_warps,
                        )
                        outputs = [logits_s_bench, logits_t_bench]
                        if compute_grad:
                            student_logits_grad = logits_s_bench
                            if grad_student_hidden is not None:
                                outputs.append(student_logits_grad @ weight_cast)
                            if need_grad_weight:
                                outputs.append(student_logits_grad.t() @ student_hidden_chunk)
                        return tuple(outputs)

                # ── Real chunk launch ──────────────────────────────────────
                launch_config = select_config(
                    mode=SEPARATE_STUDENT,
                    device=device,
                    dtype=compute_dtype,
                    vocab_size=V,
                    n_rows=n_rows,
                    max_fused_size=max_fused_size,
                    shape_key=(student_hidden.shape[1], teacher_hidden.shape[1]),
                    autotune=bool(autotune),
                    bench_fn=bench_fn,
                )
                logits_s_chunk = student_hidden_chunk @ weight_cast.t()
                logits_t_chunk = teacher_hidden_chunk @ teacher_weight_cast.t()

                student_loss_buf = torch.empty(n_rows, dtype=torch.float32, device=device)
                kl_buf = torch.empty(n_rows, dtype=torch.float32, device=device)
                _separate_student_fwdbwd(
                    logits_s_chunk,
                    logits_t_chunk,
                    labels_chunk,
                    student_loss_buf,
                    kl_buf,
                    n_rows=n_rows,
                    kl_weight=kl_weight,
                    kl_temperature=kl_temperature,
                    ignore_index=ignore_index,
                    student_scale=student_scale,
                    max_fused_size=max_fused_size,
                    block_size=launch_config.block_size,
                    num_warps=launch_config.num_warps,
                )
                del logits_t_chunk

                loss_student_accum = loss_student_accum + student_loss_buf.sum()
                kl_accum = kl_accum + kl_buf.sum()
                del student_loss_buf, kl_buf

                # ── Materialize student hidden/weight gradients ────────────
                if compute_grad:
                    student_logits_grad = logits_s_chunk
                    if grad_student_hidden is not None:
                        grad_student_hidden[start:end] = student_logits_grad @ weight_cast

                    if need_grad_weight:
                        if actual_use_fp32_accum:
                            chunk_contrib = torch.mm(student_logits_grad.t(), student_hidden_chunk).float()
                            grad_weight = chunk_contrib if grad_weight is None else grad_weight.add_(chunk_contrib)
                        else:
                            if grad_weight is None:
                                grad_weight = student_logits_grad.t() @ student_hidden_chunk
                            else:
                                grad_weight.addmm_(student_logits_grad.t(), student_hidden_chunk)
                    del student_logits_grad

                del logits_s_chunk

        # ── Reduce losses ──────────────────────────────────────────────────
        if reduction == "mean":
            denom = torch.clamp((labels != ignore_index).sum(), min=1)
            mean_scale = denom.to(torch.float32).reciprocal()
            loss_s = loss_student_accum / denom
            kl_loss = kl_accum / denom
        else:
            mean_scale = loss_student_accum.new_ones(())
            loss_s = loss_student_accum
            kl_loss = kl_accum

        loss_t = loss_s.new_zeros(())
        loss = loss_s * float(student_ce_weight) + kl_weight * kl_loss

        # ── Save precomputed backward buffers ──────────────────────────────
        ctx.save_for_backward(grad_student_hidden, grad_weight, mean_scale)
        ctx.compute_grad = compute_grad

        loss_s_out = loss_s.detach()
        loss_t_out = loss_t.detach()
        kl_loss_out = kl_loss.detach()
        ctx.mark_non_differentiable(loss_s_out, loss_t_out, kl_loss_out)
        return loss, loss_s_out, loss_t_out, kl_loss_out

    @staticmethod
    def backward(ctx, grad_output, _gs=None, _gt=None, _gkl=None):
        grad_student_hidden, grad_weight, mean_scale = ctx.saved_tensors

        # ── Scale cached gradients by upstream scalar ──────────────────────
        if grad_output is None or not ctx.compute_grad:
            return (None,) * 16

        effective_grad = grad_output * mean_scale

        if grad_student_hidden is not None:
            grad_student_hidden = grad_student_hidden * effective_grad
        if grad_weight is not None:
            grad_weight = grad_weight * effective_grad

        return (
            grad_student_hidden,
            None,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SeparateFullCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        student_ce_weight=1.0,
        ignore_index=-100,
        reduction="mean",
        chunk_size=None,
        use_fp32_accum=None,
        kl_weight=0.0,
        kl_temperature=1.0,
        teacher_ce_weight=1.0,
        max_chunks=None,
        max_fused_size=DEFAULT_MAX_FUSED_SIZE,
        autotune=False,
    ):
        # ── Validate mode constraints ────────────────────────────────────────
        if float(teacher_ce_weight) <= 0.0:
            raise ValueError("SeparateFullCEFunction requires teacher_ce_weight > 0.0")
        if student_ce_weight < 0.0:
            raise ValueError(f"student_ce_weight must be >= 0.0, got {student_ce_weight}")
        if kl_weight < 0.0:
            raise ValueError(f"kl_weight must be >= 0.0, got {kl_weight}")
        if kl_temperature <= 0.0:
            raise ValueError(f"kl_temperature must be > 0.0, got {kl_temperature}")
        if max_fused_size < 1:
            raise ValueError(f"max_fused_size must be >= 1, got {max_fused_size}")
        if not is_power_of_two(int(max_fused_size)):
            raise ValueError(f"max_fused_size must be a power of two, got {max_fused_size}")

        BT, _ = student_hidden.shape
        V = weight.shape[0]
        device = student_hidden.device

        need_grad_student_hidden = student_hidden.requires_grad
        need_grad_teacher_hidden = teacher_hidden.requires_grad
        need_grad_weight = weight.requires_grad
        need_grad_teacher_weight = teacher_weight.requires_grad

        # ── Normalize inputs and compute dtype ──────────────────────────────
        student_hidden = student_hidden.contiguous()
        teacher_hidden = teacher_hidden.contiguous()
        weight = weight.contiguous()
        teacher_weight = teacher_weight.contiguous()
        if labels.dtype != torch.long:
            raise ValueError(f"labels must have dtype torch.long, got {labels.dtype}")
        labels = labels.contiguous()

        compute_dtype = default_compute_dtype(weight, teacher_weight, student_hidden, teacher_hidden)
        weight_cast = weight.to(compute_dtype)
        teacher_weight_cast = teacher_weight.to(compute_dtype)
        student_hidden_c = student_hidden if student_hidden.dtype == compute_dtype else student_hidden.to(compute_dtype)
        teacher_hidden_c = teacher_hidden if teacher_hidden.dtype == compute_dtype else teacher_hidden.to(compute_dtype)

        # ── Allocate saved gradients ────────────────────────────────────────
        compute_grad = (
            need_grad_student_hidden
            or need_grad_teacher_hidden
            or need_grad_weight
            or need_grad_teacher_weight
        )
        grad_student_hidden = torch.zeros_like(student_hidden) if need_grad_student_hidden else None
        grad_teacher_hidden = torch.zeros_like(teacher_hidden) if need_grad_teacher_hidden else None
        grad_weight = None
        grad_teacher_weight = None

        # ── Resolve chunking and accumulators ───────────────────────────────
        chunk_size_actual, num_chunks = resolve_chunk_size(BT, chunk_size, V=V, max_chunks=max_chunks)
        actual_use_fp32_accum = False if use_fp32_accum is None else bool(use_fp32_accum)

        loss_student_accum = torch.zeros((), dtype=torch.float32, device=device)
        loss_teacher_accum = torch.zeros((), dtype=torch.float32, device=device)
        kl_accum = torch.zeros((), dtype=torch.float32, device=device)

        student_scale = float(student_ce_weight)
        teacher_scale = float(teacher_ce_weight)

        with torch.no_grad():
            for chunk_id in range(num_chunks):
                # ── Slice current chunk ─────────────────────────────────────
                start = chunk_id * chunk_size_actual
                end = min(start + chunk_size_actual, BT)
                n_rows = end - start

                student_hidden_chunk = student_hidden_c[start:end]
                teacher_hidden_chunk = teacher_hidden_c[start:end]
                labels_chunk = labels[start:end]
                bench_fn = None
                if bool(autotune):
                    # ── Autotune trial: fresh GEMM output per config ────────
                    def bench_fn(block_size: int, num_warps: int):
                        logits_s_bench = student_hidden_chunk @ weight_cast.t()
                        logits_t_bench = teacher_hidden_chunk @ teacher_weight_cast.t()
                        student_bench = torch.empty(n_rows, dtype=torch.float32, device=device)
                        teacher_bench = torch.empty(n_rows, dtype=torch.float32, device=device)
                        kl_bench = torch.empty(n_rows, dtype=torch.float32, device=device)
                        _separate_full_fwdbwd(
                            logits_s_bench,
                            logits_t_bench,
                            labels_chunk,
                            student_bench,
                            teacher_bench,
                            kl_bench,
                            n_rows=n_rows,
                            kl_weight=kl_weight,
                            kl_temperature=kl_temperature,
                            ignore_index=ignore_index,
                            student_scale=student_scale,
                            teacher_scale=teacher_scale,
                            max_fused_size=max_fused_size,
                            block_size=block_size,
                            num_warps=num_warps,
                        )
                        outputs = [logits_s_bench, logits_t_bench]
                        if compute_grad:
                            if grad_student_hidden is not None:
                                outputs.append(logits_s_bench @ weight_cast)
                            if grad_teacher_hidden is not None:
                                outputs.append(logits_t_bench @ teacher_weight_cast)
                            if need_grad_weight:
                                outputs.append(logits_s_bench.t() @ student_hidden_chunk)
                            if need_grad_teacher_weight:
                                outputs.append(logits_t_bench.t() @ teacher_hidden_chunk)
                        return tuple(outputs)

                # ── Real chunk launch ──────────────────────────────────────
                launch_config = select_config(
                    mode=SEPARATE_FULL,
                    device=device,
                    dtype=compute_dtype,
                    vocab_size=V,
                    n_rows=n_rows,
                    max_fused_size=max_fused_size,
                    shape_key=(student_hidden.shape[1], teacher_hidden.shape[1]),
                    autotune=bool(autotune),
                    bench_fn=bench_fn,
                )
                logits_s_chunk = student_hidden_chunk @ weight_cast.t()
                logits_t_chunk = teacher_hidden_chunk @ teacher_weight_cast.t()

                student_loss_buf = torch.empty(n_rows, dtype=torch.float32, device=device)
                teacher_loss_buf = torch.empty(n_rows, dtype=torch.float32, device=device)
                kl_buf = torch.empty(n_rows, dtype=torch.float32, device=device)
                _separate_full_fwdbwd(
                    logits_s_chunk,
                    logits_t_chunk,
                    labels_chunk,
                    student_loss_buf,
                    teacher_loss_buf,
                    kl_buf,
                    n_rows=n_rows,
                    kl_weight=kl_weight,
                    kl_temperature=kl_temperature,
                    ignore_index=ignore_index,
                    student_scale=student_scale,
                    teacher_scale=teacher_scale,
                    max_fused_size=max_fused_size,
                    block_size=launch_config.block_size,
                    num_warps=launch_config.num_warps,
                )
                loss_student_accum = loss_student_accum + student_loss_buf.sum()
                loss_teacher_accum = loss_teacher_accum + teacher_loss_buf.sum()
                kl_accum = kl_accum + kl_buf.sum()
                del student_loss_buf, teacher_loss_buf, kl_buf

                # ── Materialize hidden/weight gradients from dlogits ───────
                if compute_grad:
                    if grad_student_hidden is not None:
                        grad_student_hidden[start:end] = logits_s_chunk @ weight_cast
                    if grad_teacher_hidden is not None:
                        grad_teacher_hidden[start:end] = logits_t_chunk @ teacher_weight_cast

                    if need_grad_weight:
                        if actual_use_fp32_accum:
                            chunk_contrib = torch.mm(logits_s_chunk.t(), student_hidden_chunk).float()
                            grad_weight = chunk_contrib if grad_weight is None else grad_weight.add_(chunk_contrib)
                        else:
                            if grad_weight is None:
                                grad_weight = logits_s_chunk.t() @ student_hidden_chunk
                            else:
                                grad_weight.addmm_(logits_s_chunk.t(), student_hidden_chunk)

                    if need_grad_teacher_weight:
                        if actual_use_fp32_accum:
                            chunk_contrib = torch.mm(logits_t_chunk.t(), teacher_hidden_chunk).float()
                            grad_teacher_weight = (
                                chunk_contrib
                                if grad_teacher_weight is None
                                else grad_teacher_weight.add_(chunk_contrib)
                            )
                        else:
                            if grad_teacher_weight is None:
                                grad_teacher_weight = logits_t_chunk.t() @ teacher_hidden_chunk
                            else:
                                grad_teacher_weight.addmm_(logits_t_chunk.t(), teacher_hidden_chunk)

                del logits_s_chunk, logits_t_chunk

        # ── Reduce losses ──────────────────────────────────────────────────
        if reduction == "mean":
            denom = torch.clamp((labels != ignore_index).sum(), min=1)
            mean_scale = denom.to(torch.float32).reciprocal()
            loss_s = loss_student_accum / denom
            loss_t = loss_teacher_accum / denom
            kl_loss = kl_accum / denom
        else:
            mean_scale = loss_student_accum.new_ones(())
            loss_s = loss_student_accum
            loss_t = loss_teacher_accum
            kl_loss = kl_accum

        loss = loss_s * float(student_ce_weight) + loss_t * teacher_scale + kl_weight * kl_loss

        # ── Save precomputed backward buffers ──────────────────────────────
        ctx.save_for_backward(grad_student_hidden, grad_teacher_hidden, grad_weight, grad_teacher_weight, mean_scale)
        ctx.compute_grad = compute_grad

        loss_s_out = loss_s.detach()
        loss_t_out = loss_t.detach()
        kl_loss_out = kl_loss.detach()
        ctx.mark_non_differentiable(loss_s_out, loss_t_out, kl_loss_out)
        return loss, loss_s_out, loss_t_out, kl_loss_out

    @staticmethod
    def backward(ctx, grad_output, _gs=None, _gt=None, _gkl=None):
        grad_student_hidden, grad_teacher_hidden, grad_weight, grad_teacher_weight, mean_scale = ctx.saved_tensors

        # ── Scale cached gradients by upstream scalar ──────────────────────
        if grad_output is None or not ctx.compute_grad:
            return (None,) * 16

        effective_grad = grad_output * mean_scale

        if grad_student_hidden is not None:
            grad_student_hidden = grad_student_hidden * effective_grad
        if grad_teacher_hidden is not None:
            grad_teacher_hidden = grad_teacher_hidden * effective_grad
        if grad_weight is not None:
            grad_weight = grad_weight * effective_grad
        if grad_teacher_weight is not None:
            grad_teacher_weight = grad_teacher_weight * effective_grad

        return (
            grad_student_hidden,
            grad_teacher_hidden,
            grad_weight,
            grad_teacher_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ── Public entry point ────────────────────────────────────────────────────────
@torch._dynamo.disable
def separate_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    weight: torch.Tensor,
    teacher_weight: torch.Tensor | None,
    labels: torch.Tensor,
    student_ce_weight: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    chunk_size: str | int | None = None,
    use_fp32_accum: bool | None = None,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    teacher_ce_weight: float | None = None,
    max_chunks: int | None = None,
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    autotune: bool = False,
    validate_labels: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _teacher_ce_weight = 0.0 if teacher_ce_weight is None else float(teacher_ce_weight)

    if not HAS_TRITON:
        raise RuntimeError(
            "separate_distillation_loss requires Triton and CUDA/HIP tensors. "
            "Use distillation_loss(..., backend='auto' or 'torch') for fallback execution."
        )
    if student_hidden.ndim != 2:
        raise ValueError(f"student_hidden must be 2D, got shape={tuple(student_hidden.shape)}")
    BT, H = student_hidden.shape
    device = student_hidden.device
    if teacher_hidden is None:
        raise ValueError("separate_distillation_loss requires teacher_hidden")
    if teacher_hidden.ndim != 2 or teacher_hidden.shape[0] != BT:
        raise ValueError(f"teacher_hidden must have shape (BT, teacher_hidden_dim), got {tuple(teacher_hidden.shape)}")
    H_t = teacher_hidden.shape[1]
    if weight.ndim != 2 or weight.shape[1] != H:
        raise ValueError(f"weight must have shape (V, {H}), got {tuple(weight.shape)}")
    V = weight.shape[0]
    if teacher_weight is None:
        raise ValueError("separate_distillation_loss requires teacher_weight")
    if teacher_weight.shape != (V, H_t):
        raise ValueError(f"teacher_weight must have shape ({V}, {H_t}), got {tuple(teacher_weight.shape)}")
    if labels.shape != (BT,):
        raise ValueError(f"labels must have shape ({BT},), got {tuple(labels.shape)}")
    if labels.dtype != torch.long:
        raise ValueError(f"labels must have dtype torch.long, got {labels.dtype}")
    if student_ce_weight < 0.0:
        raise ValueError(f"student_ce_weight must be >= 0.0, got {student_ce_weight}")
    if kl_weight < 0.0:
        raise ValueError(f"kl_weight must be >= 0.0, got {kl_weight}")
    if kl_temperature <= 0.0:
        raise ValueError(f"kl_temperature must be > 0.0, got {kl_temperature}")
    if _teacher_ce_weight < 0.0:
        raise ValueError(f"teacher_ce_weight must be >= 0.0, got {_teacher_ce_weight}")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    if max_chunks is not None and max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")
    if max_fused_size < 1:
        raise ValueError(f"max_fused_size must be >= 1, got {max_fused_size}")
    if not is_power_of_two(int(max_fused_size)):
        raise ValueError(f"max_fused_size must be a power of two, got {max_fused_size}")
    if device.type != "cuda":
        raise ValueError(f"tensors must be CUDA/HIP tensors, got device={device}")
    if labels.device != device or weight.device != device:
        raise ValueError("labels and weight must be on the same device as student_hidden")
    if teacher_hidden.device != device or teacher_weight.device != device:
        raise ValueError("teacher_hidden and teacher_weight must be on the same device as student_hidden")

    if validate_labels:
        valid_mask = (labels == ignore_index) | ((labels >= 0) & (labels < V))
        if not valid_mask.all():
            bad_vals = labels[~valid_mask].tolist()[:5]
            raise ValueError(
                f"labels contains values outside [0, {V}) and not equal to ignore_index={ignore_index}: {bad_vals}"
            )

    function = SeparateFullCEFunction if _teacher_ce_weight > 0.0 else SeparateStudentCEFunction
    return function.apply(
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        student_ce_weight,
        ignore_index,
        reduction,
        chunk_size,
        use_fp32_accum,
        kl_weight,
        kl_temperature,
        _teacher_ce_weight,
        max_chunks,
        max_fused_size,
        autotune,
    )
