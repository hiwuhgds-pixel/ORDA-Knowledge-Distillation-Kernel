"""Triton KL distillation kernel for ORDA.

Style flags: synchronized with CE kernel (ONLINE_SOFTMAX / FAST_MATH_EXP / LOG / MUL).

Public API:
    kl_from_logits_chunk(logits_chunk, targets_chunk, ...) -> (kl_loss, grad_kl_student)
    add_kl_grad_to_logits_chunk_(logits_chunk, targets_chunk, ...)  -> kl_loss
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    triton = None  # type: ignore[assignment]
    tl = None      # type: ignore[assignment]
    _HAS_TRITON = False

from .._runtime import _LOG2E
from ..utils.resolver import DEFAULT_MAX_FUSED_SIZE, is_power_of_two


# ── High-precision math helpers ────────────────────────────────────────────────

if _HAS_TRITON:
    @triton.jit
    def _kl_highprec_exp(x):
        return tl.math.exp(x)

    @triton.jit
    def _kl_highprec_log(x):
        return tl.math.log(x)


# ── Triton kernel ─────────────────────────────────────────────────────────────

if _HAS_TRITON:
    @triton.jit
    def _kl_from_logits_chunk_kernel(
        X_ptr, X_stride,      # [2*n_rows, V] — student rows first, teacher rows after
        G_ptr, G_stride,      # [n_rows, V]   — KL grad for student logits (output)
        K_ptr,                # [n_rows]      — per-row KL loss (output, fp32)
        Y_ptr, Y_stride,      # [n_rows]      — target ids (to apply ignore_index mask)
        n_rows,
        n_cols,
        ignore_index,
        T_inv,                # 1.0 / temperature
        grad_scale,           # kl_weight * temperature / grad_denom  (grad_denom=n_non_ignore if mean, else 1)
        T_sq,                 # temperature ** 2 (used to scale kl_row → K_ptr)
        ONLINE_SOFTMAX: tl.constexpr,  # True = 2-pass online, False = 3-pass fixed-shift
        FAST_MATH_EXP:  tl.constexpr,  # True = exp2.approx (~4 ULP), False = libdevice exp
        FAST_MATH_LOG:  tl.constexpr,  # True = tl.math.log,          False = libdevice log
        FAST_MATH_MUL:  tl.constexpr,  # True = precompute inv_d → mul instead of div
        BLOCK_SIZE:     tl.constexpr,
    ):
        """Per-row KL(teacher || student) + dKL/d(student logits).

        Two modes:
          ONLINE_SOFTMAX=True  → 2 pass (Pass 1: online max+Σexp fused, Pass 2: KL+grad)
          ONLINE_SOFTMAX=False → 3 pass (Pass 1: max, Pass 2: Σexp, Pass 3: KL+grad)
        """
        i = tl.program_id(0).to(tl.int64)

        y = tl.load(Y_ptr + i * Y_stride)
        ignored = y == ignore_index

        row_s = X_ptr + i * X_stride
        row_t = X_ptr + (i + n_rows) * X_stride
        row_gs = G_ptr + i * G_stride
        offs  = tl.arange(0, BLOCK_SIZE)

        # ── Pass 1 & 2: max + Σexp ────────────────────────────────────────────────
        if ONLINE_SOFTMAX:
            # 2-pass (Milakov): fused max + Σexp
            m_s = float("-inf")
            m_t = float("-inf")
            d_s = 0.0
            d_t = 0.0
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offs
                mask = cols < n_cols
                x_s   = tl.load(row_s + cols, mask=mask, other=float("-inf"))
                x_t   = tl.load(row_t + cols, mask=mask, other=float("-inf"))
                x_s_t = x_s * T_inv
                x_t_t = x_t * T_inv

                m_s_new = tl.maximum(m_s, tl.max(x_s_t))
                m_t_new = tl.maximum(m_t, tl.max(x_t_t))

                if FAST_MATH_EXP:
                    d_s = (d_s * tl.math.exp2((m_s - m_s_new) * _LOG2E)
                           + tl.sum(tl.where(mask, tl.math.exp2((x_s_t - m_s_new) * _LOG2E), 0.0)))
                    d_t = (d_t * tl.math.exp2((m_t - m_t_new) * _LOG2E)
                           + tl.sum(tl.where(mask, tl.math.exp2((x_t_t - m_t_new) * _LOG2E), 0.0)))
                else:
                    d_s = (d_s * _kl_highprec_exp(m_s - m_s_new)
                           + tl.sum(tl.where(mask, _kl_highprec_exp(x_s_t - m_s_new), 0.0)))
                    d_t = (d_t * _kl_highprec_exp(m_t - m_t_new)
                           + tl.sum(tl.where(mask, _kl_highprec_exp(x_t_t - m_t_new), 0.0)))
                m_s = m_s_new
                m_t = m_t_new
        else:
            # 3-pass: Pass 1 = max
            m_s = float("-inf")
            m_t = float("-inf")
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offs
                mask = cols < n_cols
                x_s = tl.load(row_s + cols, mask=mask, other=float("-inf"))
                x_t = tl.load(row_t + cols, mask=mask, other=float("-inf"))
                m_s = tl.maximum(m_s, tl.max(x_s * T_inv))
                m_t = tl.maximum(m_t, tl.max(x_t * T_inv))

            # 3-pass: Pass 2 = Σexp
            d_s = 0.0
            d_t = 0.0
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offs
                mask = cols < n_cols
                x_s = tl.load(row_s + cols, mask=mask, other=float("-inf"))
                x_t = tl.load(row_t + cols, mask=mask, other=float("-inf"))
                v_s = x_s * T_inv - m_s
                v_t = x_t * T_inv - m_t
                if FAST_MATH_EXP:
                    d_s += tl.sum(tl.where(mask, tl.math.exp2(v_s * _LOG2E), 0.0))
                    d_t += tl.sum(tl.where(mask, tl.math.exp2(v_t * _LOG2E), 0.0))
                else:
                    d_s += tl.sum(tl.where(mask, _kl_highprec_exp(v_s), 0.0))
                    d_t += tl.sum(tl.where(mask, _kl_highprec_exp(v_t), 0.0))

        # log(Σexp) for both student and teacher
        if FAST_MATH_LOG:
            log_d_s = tl.math.log(d_s)
            log_d_t = tl.math.log(d_t)
        else:
            log_d_s = _kl_highprec_log(d_s)
            log_d_t = _kl_highprec_log(d_t)

        # Precompute inv_d if FAST_MATH_MUL (mul instead of div, faster on CUDA)
        if FAST_MATH_MUL:
            inv_d_s = 1.0 / d_s
            inv_d_t = 1.0 / d_t

        # ── Final pass: KL(teacher || student) + dKL/d(student logits) ───────────────────
        kl_row = 0.0
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            x_s = tl.load(row_s + cols, mask=mask, other=float("-inf"))
            x_t = tl.load(row_t + cols, mask=mask, other=float("-inf"))
            v_s = x_s * T_inv - m_s
            v_t = x_t * T_inv - m_t

            if FAST_MATH_EXP:
                exp_v_s = tl.math.exp2(v_s * _LOG2E)
                exp_v_t = tl.math.exp2(v_t * _LOG2E)
            else:
                exp_v_s = _kl_highprec_exp(v_s)
                exp_v_t = _kl_highprec_exp(v_t)

            if FAST_MATH_MUL:
                p_s = exp_v_s * inv_d_s
                p_t = exp_v_t * inv_d_t
            else:
                p_s = exp_v_s / d_s
                p_t = exp_v_t / d_t

            log_p_s = v_s - log_d_s
            log_p_t = v_t - log_d_t
            kl_row += tl.sum(tl.where(mask, p_t * (log_p_t - log_p_s), 0.0))

            grad_s = grad_scale * (p_s - p_t)
            grad_s = tl.where(ignored, 0.0, grad_s)
            grad_s = tl.where(mask, grad_s, 0.0)
            tl.store(row_gs + cols, grad_s, mask=mask)

        kl_row = tl.where(ignored, 0.0, kl_row)
        tl.store(K_ptr + i, kl_row * T_sq)


# ── Python wrapper ─────────────────────────────────────────────────────────────

def kl_from_logits_chunk(
    logits_chunk:       torch.Tensor,
    targets_chunk:      torch.Tensor,
    n_rows:             int | None  = None,
    kl_weight:          float       = 0.4,
    kl_temperature:     float       = 1.5,
    n_non_ignore:       int | None  = None,
    ignore_index:       int         = -100,
    use_fast_math_exp:  bool | None = None,
    use_fast_math_log:  bool | None = None,
    use_fast_math_mul:  bool | None = None,
    use_online_softmax: bool | None = None,
    max_fused_size:     int         = DEFAULT_MAX_FUSED_SIZE,
    reduction:          str         = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL(teacher || student) from logits_chunk created by CE.

    This function does **not** project hidden states and does **not** create full-batch logits.
    Reuses ``logits_chunk`` ``[2*n_rows, V]`` already present in the CE chunk loop.

    Args:
        logits_chunk: ``[2*n_rows, V]`` fp16. First half = student logits, second half = teacher logits.
        targets_chunk: ``[n_rows]`` target ids.
        n_rows: number of student rows. If None → inferred from logits_chunk.shape[0] // 2.
        kl_weight: KL coefficient (used to compute grad_scale).
        kl_temperature: softmax temperature T.
        n_non_ignore: denominator (number of non-ignored tokens) of the **entire batch**.
            If None → counted from targets_chunk (only correct for single-chunk).
        ignore_index: target value to ignore.
        use_fast_math_exp:  None → False.
        use_fast_math_log:  None → False.
        use_fast_math_mul:  None → False.
        use_online_softmax: None → False.
        reduction: ``"mean"`` divides by ``n_non_ignore``; ``"sum"`` returns the summed KL.

    Returns:
        ``(kl_loss, grad_kl_student)``
        - ``kl_loss``: scalar fp32, reduced according to ``reduction``, scaled by T².
        - ``grad_kl_student``: ``[n_rows, V]`` fp16, scaled by grad_scale. Ready to be added
          to ``logits_chunk[:n_rows]``.
    """
    # ── Triton availability check ───────────────────────────────────────────────
    if not _HAS_TRITON:
        raise RuntimeError(
            "kl_from_logits_chunk requires Triton and CUDA/HIP tensors."
        )

    # ── Validation ─────────────────────────────────────────────────────────────
    if logits_chunk.ndim != 2:
        raise ValueError(f"logits_chunk must be 2D, got {tuple(logits_chunk.shape)}")
    if targets_chunk.ndim != 1:
        raise ValueError(f"targets_chunk must be 1D, got {tuple(targets_chunk.shape)}")
    if not logits_chunk.is_cuda:
        raise ValueError(f"logits_chunk must be a CUDA/HIP tensor, got {logits_chunk.device}")
    if targets_chunk.device != logits_chunk.device:
        raise ValueError("targets_chunk and logits_chunk must be on the same device")
    if targets_chunk.dtype == torch.bool or targets_chunk.is_floating_point():
        raise ValueError(f"targets_chunk must be an integer class-index tensor, got dtype={targets_chunk.dtype}")
    if kl_temperature <= 0.0:
        raise ValueError(f"kl_temperature must be > 0, got {kl_temperature}")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    if max_fused_size < 1:
        raise ValueError(f"max_fused_size must be >= 1, got {max_fused_size}")
    if not is_power_of_two(int(max_fused_size)):
        raise ValueError(f"max_fused_size must be a power of two, got {max_fused_size}")

    total_rows, vocab_size = logits_chunk.shape
    if n_rows is None:
        if total_rows % 2 != 0:
            raise ValueError("n_rows is required when logits_chunk has an odd row count")
        n_rows = total_rows // 2
    if total_rows != 2 * n_rows:
        raise ValueError(
            f"logits_chunk rows must equal 2*n_rows, got {total_rows} and {n_rows}"
        )
    if targets_chunk.shape[0] != n_rows:
        raise ValueError(
            f"targets_chunk must have {n_rows} rows, got {targets_chunk.shape[0]}"
        )

    # ── Resolve flags ──────────────────────────────────────────────────────────
    actual_exp    = False if use_fast_math_exp  is None else use_fast_math_exp
    actual_log    = False if use_fast_math_log  is None else use_fast_math_log
    actual_mul    = False if use_fast_math_mul  is None else use_fast_math_mul
    actual_online = False if use_online_softmax is None else use_online_softmax

    # ── Contiguous + denom ─────────────────────────────────────────────────────
    logits_chunk   = logits_chunk.contiguous()
    targets_chunk  = targets_chunk.contiguous()

    if n_non_ignore is None:
        n_non_ignore = int((targets_chunk != ignore_index).sum().item())
    denom = max(int(n_non_ignore), 1)
    grad_denom = denom if reduction == "mean" else 1

    # ── Output buffers ─────────────────────────────────────────────────────────
    grad_kl_student = torch.empty(
        (n_rows, vocab_size),
        dtype=logits_chunk.dtype,
        device=logits_chunk.device,
    )
    kl_per_row = torch.empty(n_rows, dtype=torch.float32, device=logits_chunk.device)

    # ── Dispatch kernel ────────────────────────────────────────────────────────
    block_size = min(int(max_fused_size), triton.next_power_of_2(vocab_size))
    _kl_from_logits_chunk_kernel[(n_rows,)](
        logits_chunk,
        logits_chunk.stride(0),
        grad_kl_student,
        grad_kl_student.stride(0),
        kl_per_row,
        targets_chunk,
        targets_chunk.stride(0),
        n_rows,
        vocab_size,
        ignore_index,
        T_inv=1.0 / float(kl_temperature),
        grad_scale=float(kl_weight) * float(kl_temperature) / grad_denom,
        T_sq=float(kl_temperature) * float(kl_temperature),
        ONLINE_SOFTMAX=actual_online,
        FAST_MATH_EXP=actual_exp,
        FAST_MATH_LOG=actual_log,
        FAST_MATH_MUL=actual_mul,
        BLOCK_SIZE=block_size,
        num_warps=16,
    )

    kl_sum = kl_per_row.sum()
    if reduction == "mean":
        return kl_sum / denom, grad_kl_student
    return kl_sum, grad_kl_student


def add_kl_grad_to_logits_chunk_(
    logits_chunk:       torch.Tensor,
    targets_chunk:      torch.Tensor,
    n_rows:             int | None  = None,
    kl_weight:          float       = 0.4,
    kl_temperature:     float       = 1.5,
    n_non_ignore:       int | None  = None,
    ignore_index:       int         = -100,
    use_fast_math_exp:  bool | None = None,
    use_fast_math_log:  bool | None = None,
    use_fast_math_mul:  bool | None = None,
    use_online_softmax: bool | None = None,
    max_fused_size:     int         = DEFAULT_MAX_FUSED_SIZE,
    reduction:          str         = "mean",
) -> torch.Tensor:
    """Computes KL and adds student grad to ``logits_chunk`` in-place.

    Utility to directly add KL gradient before CE backward.

    Returns:
        ``kl_loss`` scalar fp32 (reduced and scaled by T²).
    """
    kl_loss, grad_kl_student = kl_from_logits_chunk(
        logits_chunk,
        targets_chunk,
        n_rows=n_rows,
        kl_weight=kl_weight,
        kl_temperature=kl_temperature,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        use_fast_math_exp=use_fast_math_exp,
        use_fast_math_log=use_fast_math_log,
        use_fast_math_mul=use_fast_math_mul,
        use_online_softmax=use_online_softmax,
        max_fused_size=max_fused_size,
        reduction=reduction,
    )
    logits_chunk[: grad_kl_student.shape[0]].add_(grad_kl_student.to(logits_chunk.dtype))
    return kl_loss
