import torch

from .._runtime import HAS_TRITON, default_compute_dtype, tl, tl_highprec_exp, tl_highprec_log, triton
from ..utils._autotune import TIED, default_config, select_config
from ..utils.resolver import DEFAULT_MAX_FUSED_SIZE, is_power_of_two, resolve_chunk_size


# ── Kernel: CE + KL Fwd/Bwd ──────────────────────────────────────────────────
if HAS_TRITON:
    @triton.jit
    def _tied_ce_kl_fwdbwd_kernel(
        X_ptr, X_stride,
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
        COMPUTE_TEACHER_CE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Forward losses and in-place dlogits for tied-weight CE+KL."""
        i = tl.program_id(0).to(tl.int64)
        y = tl.load(Y_ptr + i * Y_stride)

        tl.device_assert(
            ((y >= 0) & (y < n_cols)) | (y == ignore_index),
            "labels out of range",
        )
        ignored = y == ignore_index
        y_safe = tl.where(ignored, 0, y)

        row_s = X_ptr + i * X_stride
        row_t = X_ptr + (i + n_rows) * X_stride
        offs = tl.arange(0, BLOCK_SIZE)

        # ── Forward: Online max + Σexp ───────────────────────────────────────
        m_s = float("-inf")
        d_s_ce = 0.0
        d_s_kl = 0.0
        m_t_kl = float("-inf")
        d_t_kl = 0.0
        m_t_ce = float("-inf")
        d_t_ce = 0.0

        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            x_s = tl.load(row_s + cols, mask=mask, other=float("-inf"), eviction_policy="evict_last").to(tl.float32)
            x_t = tl.load(row_t + cols, mask=mask, other=float("-inf"), eviction_policy="evict_last").to(tl.float32)
            x_t_kl = x_t * T_inv

            m_s_new = tl.maximum(m_s, tl.max(x_s))
            m_t_kl_new = tl.maximum(m_t_kl, tl.max(x_t_kl))
            if COMPUTE_TEACHER_CE:
                m_t_ce_new = tl.maximum(m_t_ce, tl.max(x_t))

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
            if COMPUTE_TEACHER_CE:
                d_t_ce = (
                    d_t_ce * tl_highprec_exp(m_t_ce - m_t_ce_new)
                    + tl.sum(tl_highprec_exp(x_t - m_t_ce_new))
                )

            m_s = m_s_new
            m_t_kl = m_t_kl_new
            if COMPUTE_TEACHER_CE:
                m_t_ce = m_t_ce_new

        m_s_kl = m_s * T_inv
        log_d_s_ce = tl_highprec_log(d_s_ce)
        log_d_s_kl = tl_highprec_log(d_s_kl)
        log_d_t_kl = tl_highprec_log(d_t_kl)
        # ── NLL ──────────────────────────────────────────────────────────────
        lse_s_ce = m_s + log_d_s_ce
        logit_label_s = tl.load(row_s + y_safe).to(tl.float32)
        nll_s = lse_s_ce - logit_label_s
        tl.store(L_ptr + i, tl.where(ignored, 0.0, nll_s))

        if COMPUTE_TEACHER_CE:
            lse_t_ce = m_t_ce + tl_highprec_log(d_t_ce)
            logit_label_t = tl.load(row_t + y_safe).to(tl.float32)
            nll_t = lse_t_ce - logit_label_t
            tl.store(T_L_ptr + i, tl.where(ignored, 0.0, nll_t))
        else:
            tl.store(T_L_ptr + i, 0.0)

        # ── Gradient pass ────────────────────────────────────────────────────
        kl_row = 0.0
        student_row_scale = tl.where(ignored, 0.0, student_scale)
        teacher_row_scale = tl.where(ignored, 0.0, teacher_scale)

        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            is_label = cols == y_safe

            x_s = tl.load(row_s + cols, mask=mask, other=float("-inf"), eviction_policy="evict_first").to(tl.float32)
            x_t = tl.load(row_t + cols, mask=mask, other=float("-inf"), eviction_policy="evict_first").to(tl.float32)

            if COMPUTE_TEACHER_CE:
                teacher_grad = tl_highprec_exp(x_t - m_t_ce) / d_t_ce
                teacher_grad = tl.where(is_label, teacher_grad - 1.0, teacher_grad)
                teacher_grad = teacher_grad * teacher_row_scale
            else:
                teacher_grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
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
    def _tied_ce_kl_fwdbwd_kernel(*args, **kwargs):
        raise RuntimeError("Triton kernels are unavailable; install Triton and use CUDA/HIP tensors.")


# ── Python wrapper ────────────────────────────────────────────────────────────
def _tied_fwdbwd(
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
) -> None:
    """Forward + in-place backward for tied-weight CE+KL distillation."""
    if not HAS_TRITON:
        raise RuntimeError("_tied_fwdbwd requires Triton and CUDA/HIP tensors.")

    if logits_chunk.ndim != 2:
        raise ValueError(f"logits_chunk must be 2D, got {tuple(logits_chunk.shape)}")
    if labels_chunk.ndim != 1:
        raise ValueError(f"labels_chunk must be 1D, got {tuple(labels_chunk.shape)}")
    if student_loss.ndim != 1 or teacher_loss.ndim != 1 or kl_loss.ndim != 1:
        raise ValueError("student_loss, teacher_loss and kl_loss must be 1D tensors")
    if not logits_chunk.is_cuda:
        raise ValueError(f"logits_chunk must be a CUDA/HIP tensor, got {logits_chunk.device}")
    if labels_chunk.device != logits_chunk.device:
        raise ValueError("labels_chunk and logits_chunk must be on the same device")
    if (
        student_loss.device != logits_chunk.device
        or teacher_loss.device != logits_chunk.device
        or kl_loss.device != logits_chunk.device
    ):
        raise ValueError("loss buffers must be on the same device as logits_chunk")
    if not logits_chunk.is_contiguous():
        raise ValueError("logits_chunk must be contiguous because gradients are written in-place")
    if not student_loss.is_contiguous() or not teacher_loss.is_contiguous() or not kl_loss.is_contiguous():
        raise ValueError("loss buffers must be contiguous")

    total_rows, vocab_size = logits_chunk.shape
    if n_rows is None:
        if total_rows % 2 != 0:
            raise ValueError("n_rows is required when logits_chunk has an odd row count")
        n_rows = total_rows // 2
    if total_rows != 2 * n_rows:
        raise ValueError(f"logits_chunk rows must equal 2*n_rows, got {total_rows} and {n_rows}")
    if labels_chunk.shape[0] != n_rows:
        raise ValueError(f"labels_chunk must have {n_rows} rows, got {labels_chunk.shape[0]}")
    if student_loss.shape[0] != n_rows or teacher_loss.shape[0] != n_rows or kl_loss.shape[0] != n_rows:
        raise ValueError("loss buffers must have n_rows elements")

    labels_chunk = labels_chunk.contiguous()
    if block_size is None or num_warps is None:
        launch_config = default_config(TIED, vocab_size, int(max_fused_size))
        block_size = launch_config.block_size
        num_warps = launch_config.num_warps
    compute_teacher_ce = float(teacher_scale) != 0.0

    _tied_ce_kl_fwdbwd_kernel[(n_rows,)](
        logits_chunk,
        logits_chunk.stride(0),
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
        COMPUTE_TEACHER_CE=compute_teacher_ce,
        BLOCK_SIZE=int(block_size),
        num_warps=int(num_warps),
    )


# ── Autograd Function ─────────────────────────────────────────────────────────
class TiedCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        student_hidden,
        teacher_hidden,
        weight,
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
        BT, _ = student_hidden.shape
        V = weight.shape[0]
        device = student_hidden.device

        # ── Validate mode constraints ────────────────────────────────────────
        if not HAS_TRITON:
            raise RuntimeError(
                "tied_distillation_loss requires Triton and CUDA/HIP tensors. "
                "Use distillation_loss(..., backend='auto' or 'torch') for fallback execution."
            )
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

        need_grad_student_hidden = student_hidden.requires_grad
        need_grad_teacher_hidden = teacher_hidden.requires_grad
        need_grad_weight = weight.requires_grad

        # ── Normalize inputs and compute dtype ──────────────────────────────
        student_hidden = student_hidden.contiguous()
        teacher_hidden = teacher_hidden.contiguous()
        weight = weight.contiguous()
        if labels.dtype != torch.long:
            raise ValueError(f"labels must have dtype torch.long, got {labels.dtype}")
        labels = labels.contiguous()

        compute_dtype = default_compute_dtype(weight, student_hidden, teacher_hidden)
        weight_cast = weight.to(compute_dtype)
        student_hidden_c = student_hidden if student_hidden.dtype == compute_dtype else student_hidden.to(compute_dtype)
        teacher_hidden_c = teacher_hidden if teacher_hidden.dtype == compute_dtype else teacher_hidden.to(compute_dtype)

        # ── Allocate saved gradients ────────────────────────────────────────
        compute_grad = need_grad_student_hidden or need_grad_teacher_hidden or need_grad_weight
        grad_student_hidden = torch.zeros_like(student_hidden) if need_grad_student_hidden else None
        grad_teacher_hidden = torch.zeros_like(teacher_hidden) if need_grad_teacher_hidden else None
        grad_weight = None

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
                        hidden_bench_concat = torch.cat([student_hidden_chunk, teacher_hidden_chunk], dim=0)
                        logits_bench = hidden_bench_concat @ weight_cast.t()
                        student_bench = torch.empty(n_rows, dtype=torch.float32, device=device)
                        teacher_bench = torch.empty(n_rows, dtype=torch.float32, device=device)
                        kl_bench = torch.empty(n_rows, dtype=torch.float32, device=device)
                        _tied_fwdbwd(
                            logits_bench,
                            labels_chunk,
                            student_bench,
                            kl_bench,
                            teacher_loss=teacher_bench,
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
                        outputs = [logits_bench]
                        if compute_grad:
                            student_logits_grad = logits_bench[:n_rows]
                            teacher_logits_grad = logits_bench[n_rows:]
                            if grad_student_hidden is not None:
                                outputs.append(student_logits_grad @ weight_cast)
                            if grad_teacher_hidden is not None:
                                outputs.append(teacher_logits_grad @ weight_cast)
                            if need_grad_weight:
                                if teacher_scale != 0.0:
                                    grad_src = logits_bench
                                    hidden_src = hidden_bench_concat
                                else:
                                    grad_src = student_logits_grad
                                    hidden_src = student_hidden_chunk
                                outputs.append(grad_src.t() @ hidden_src)
                        return tuple(outputs)

                # ── Real chunk launch ──────────────────────────────────────
                launch_config = select_config(
                    mode=TIED,
                    device=device,
                    dtype=compute_dtype,
                    vocab_size=V,
                    n_rows=n_rows,
                    max_fused_size=max_fused_size,
                    shape_key=(student_hidden.shape[1],),
                    autotune=bool(autotune),
                    bench_fn=bench_fn,
                )
                hidden_chunk_concat = torch.cat([student_hidden_chunk, teacher_hidden_chunk], dim=0)
                logits_chunk = hidden_chunk_concat @ weight_cast.t()

                student_loss_buf = torch.empty(n_rows, dtype=torch.float32, device=device)
                teacher_loss_buf = torch.empty(n_rows, dtype=torch.float32, device=device)
                kl_buf = torch.empty(n_rows, dtype=torch.float32, device=device)
                _tied_fwdbwd(
                    logits_chunk,
                    labels_chunk,
                    student_loss_buf,
                    kl_buf,
                    teacher_loss=teacher_loss_buf,
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
                    student_logits_grad = logits_chunk[:n_rows]
                    teacher_logits_grad = logits_chunk[n_rows:]
                    if grad_student_hidden is not None or grad_teacher_hidden is not None:
                        if grad_student_hidden is not None:
                            grad_student_hidden[start:end] = student_logits_grad @ weight_cast
                        if grad_teacher_hidden is not None:
                            grad_teacher_hidden[start:end] = teacher_logits_grad @ weight_cast

                    if need_grad_weight:
                        if teacher_scale != 0.0:
                            grad_src = logits_chunk
                            hidden_src = hidden_chunk_concat
                        else:
                            grad_src = student_logits_grad
                            hidden_src = student_hidden_chunk
                        if actual_use_fp32_accum:
                            chunk_contrib = torch.mm(grad_src.t(), hidden_src).float()
                            grad_weight = chunk_contrib if grad_weight is None else grad_weight.add_(chunk_contrib)
                        else:
                            if grad_weight is None:
                                grad_weight = grad_src.t() @ hidden_src
                            else:
                                grad_weight.addmm_(grad_src.t(), hidden_src)

                del logits_chunk, hidden_chunk_concat

        # ── Reduce losses ──────────────────────────────────────────────────
        if reduction == "mean":
            denom = torch.clamp((labels != ignore_index).sum(), min=1)
            mean_scale = denom.to(torch.float32).reciprocal()
            loss_s = loss_student_accum / denom
            loss_t_raw = loss_teacher_accum / denom
            kl_loss = kl_accum / denom
        else:
            mean_scale = loss_student_accum.new_ones(())
            loss_s = loss_student_accum
            loss_t_raw = loss_teacher_accum
            kl_loss = kl_accum

        loss_t = loss_t_raw if teacher_scale != 0.0 else loss_s.new_zeros(())
        loss = loss_s * float(student_ce_weight) + loss_t * teacher_scale + kl_weight * kl_loss

        # ── Save precomputed backward buffers ──────────────────────────────
        ctx.save_for_backward(grad_student_hidden, grad_teacher_hidden, grad_weight, mean_scale)
        ctx.compute_grad = compute_grad

        loss_s_out = loss_s.detach()
        loss_t_out = loss_t.detach()
        kl_loss_out = kl_loss.detach()
        ctx.mark_non_differentiable(loss_s_out, loss_t_out, kl_loss_out)
        return loss, loss_s_out, loss_t_out, kl_loss_out

    @staticmethod
    def backward(ctx, grad_output, _gs=None, _gt=None, _gkl=None):
        grad_student_hidden, grad_teacher_hidden, grad_weight, mean_scale = ctx.saved_tensors

        # ── Scale cached gradients by upstream scalar ──────────────────────
        if grad_output is None or not ctx.compute_grad:
            return (None,) * 15

        effective_grad = grad_output * mean_scale

        if grad_student_hidden is not None:
            grad_student_hidden = grad_student_hidden * effective_grad
        if grad_teacher_hidden is not None:
            grad_teacher_hidden = grad_teacher_hidden * effective_grad
        if grad_weight is not None:
            grad_weight = grad_weight * effective_grad

        return (
            grad_student_hidden,
            grad_teacher_hidden,
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
        )


# ── Public entry point ────────────────────────────────────────────────────────
@torch._dynamo.disable
def tied_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    weight: torch.Tensor,
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
    _teacher_ce_weight = 1.0 if teacher_ce_weight is None else float(teacher_ce_weight)

    if not HAS_TRITON:
        raise RuntimeError(
            "tied_distillation_loss requires Triton and CUDA/HIP tensors. "
            "Use distillation_loss(..., backend='auto' or 'torch') for fallback execution."
        )
    if student_hidden.ndim != 2:
        raise ValueError(f"student_hidden must be 2D, got shape={tuple(student_hidden.shape)}")
    BT, H = student_hidden.shape
    device = student_hidden.device
    if teacher_hidden is None:
        raise ValueError("tied_distillation_loss requires teacher_hidden")
    if teacher_hidden.shape != (BT, H):
        raise ValueError(f"teacher_hidden.shape={tuple(teacher_hidden.shape)} does not match student_hidden.shape={(BT, H)}")
    if weight.ndim != 2 or weight.shape[1] != H:
        raise ValueError(f"weight must have shape (V, {H}), got {tuple(weight.shape)}")
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
    if labels.device != device or weight.device != device or teacher_hidden.device != device:
        raise ValueError("labels, weight and teacher_hidden must be on the same device as student_hidden")

    V = weight.shape[0]
    if validate_labels:
        valid_mask = (labels == ignore_index) | ((labels >= 0) & (labels < V))
        if not valid_mask.all():
            bad_vals = labels[~valid_mask].tolist()[:5]
            raise ValueError(
                f"labels contains values outside [0, {V}) and not equal to ignore_index={ignore_index}: {bad_vals}"
            )

    return TiedCEFunction.apply(
        student_hidden,
        teacher_hidden,
        weight,
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
