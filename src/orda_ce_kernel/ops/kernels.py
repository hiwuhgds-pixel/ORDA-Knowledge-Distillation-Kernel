try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    triton = None  # type: ignore[assignment]
    tl = None      # type: ignore[assignment]
    _HAS_TRITON = False

from .._runtime import tl_highprec_exp, tl_highprec_log, _LOG2E


# ── Kernel: Exact CE Forward + Backward Merged ────────────────────────────────
if _HAS_TRITON:
    @triton.jit
    def _exact_ce_fwdbwd_kernel_merged(
        X_ptr,    X_stride,   # [grid_rows, V] logits → gradient written in-place
        Y_ptr,    Y_stride,   # [n_rows]   targets
        L_ptr,                # [grid_rows] output: per-row NLL loss (float32)
        n_cols,
        n_rows,
        ignore_index,
        label_smoothing: tl.constexpr,
        student_scale,
        teacher_scale,
        ONLINE_SOFTMAX:  tl.constexpr,  # True → 2-pass Milakov | False → 3-pass fixed-shift
        FAST_MATH_EXP:   tl.constexpr,  # True → tl.math.exp2.approx  | False → libdevice exp
        FAST_MATH_LOG:   tl.constexpr,  # True → tl.math.log (native)  | False → libdevice log
        FAST_MATH_MUL:   tl.constexpr,  # True → multiply-not-divide (inv_d) | False → divide
        STUDENT_ONLY:    tl.constexpr,  # True → grid=[n_rows], drop teacher branch (KD pure)
        BLOCK_SIZE:      tl.constexpr,
    ):
        """CE kernel, specialised via STUDENT_ONLY × ONLINE_SOFTMAX × FAST_MATH_EXP × FAST_MATH_LOG × FAST_MATH_MUL constexpr flags."""
        i          = tl.program_id(0).to(tl.int64)
        if STUDENT_ONLY:
            is_student = True
            y_idx      = i
        else:
            is_student = i < n_rows
            y_idx      = tl.where(is_student, i, i - n_rows)
        y          = tl.load(Y_ptr + y_idx * Y_stride)

        row_ptr  = X_ptr + i * X_stride
        offs     = tl.arange(0, BLOCK_SIZE)

        # ── Ignore index ──────────────────────────────────────────────────────────
        tl.device_assert(
            ((y >= 0) & (y < n_cols)) | (y == ignore_index),
            "target out of range",
        )
        ignored = (y == ignore_index)
        y_safe  = tl.where(ignored, 0, y)   # clamp index; ignored rows zeroed via scale

        if STUDENT_ONLY:
            scale = student_scale
        else:
            scale = tl.where(is_student, student_scale, teacher_scale)
        scale = tl.where(ignored, 0.0, scale)

        # ── Forward: Max + Σexp ───────────────────────────────────────────────────
        m     = float("-inf")
        d     = 0.0
        sum_x = 0.0

        if ONLINE_SOFTMAX:
            # ── 2-pass Milakov: combined max + Σexp (1 memory pass) ───────────────
            for start in range(0, n_cols, BLOCK_SIZE):
                cols  = start + offs
                mask  = cols < n_cols
                x     = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
                m_new = tl.maximum(m, tl.max(x))

                if FAST_MATH_EXP:
                    d = d * tl.math.exp2((m - m_new) * _LOG2E) \
                        + tl.sum(tl.where(mask, tl.math.exp2((x - m_new) * _LOG2E), 0.0))
                else:
                    d = d * tl_highprec_exp(m - m_new) \
                        + tl.sum(tl.where(mask, tl_highprec_exp(x - m_new), 0.0))

                m = m_new
                if label_smoothing > 0.0:
                    sum_x += tl.sum(tl.where(mask, x, 0.0))

        else:
            # ── 3-pass fixed-shift: pass 1 — find global max ──────────────────────
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offs
                mask = cols < n_cols
                x    = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
                m    = tl.maximum(m, tl.max(x))

            # pass 2 — Σexp(x - m)
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offs
                mask = cols < n_cols
                x    = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)

                if FAST_MATH_EXP:
                    d += tl.sum(tl.where(mask, tl.math.exp2((x - m) * _LOG2E), 0.0))
                else:
                    d += tl.sum(tl.where(mask, tl_highprec_exp(x - m), 0.0))

                if label_smoothing > 0.0:
                    sum_x += tl.sum(tl.where(mask, x, 0.0))

        # ── NLL ───────────────────────────────────────────────────────────────────
        if FAST_MATH_LOG:
            lse = m + tl.math.log(d)
        else:
            lse = m + tl_highprec_log(d)

        logit_tgt = tl.load(row_ptr + y_safe).to(tl.float32)

        if label_smoothing > 0.0:
            nll = lse - (1.0 - label_smoothing) * logit_tgt \
                      - (label_smoothing / n_cols) * sum_x
        else:
            nll = lse - logit_tgt

        tl.store(L_ptr + i, tl.where(ignored, 0.0, nll))

        # ── Gradient pass ─────────────────────────────────────────────────────────
        eps = label_smoothing / n_cols
        if FAST_MATH_MUL:
            inv_d = 1.0 / d   # precompute 1/d → V muls instead of V divides

        for start in range(0, n_cols, BLOCK_SIZE):
            cols      = start + offs
            mask      = cols < n_cols
            is_target = cols == y_safe

            x = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)

            if FAST_MATH_EXP:
                if FAST_MATH_MUL:
                    prob = tl.math.exp2((x - m) * _LOG2E) * inv_d
                else:
                    prob = tl.math.exp2((x - m) * _LOG2E) / d
            else:
                if FAST_MATH_MUL:
                    prob = tl_highprec_exp(x - m) * inv_d
                else:
                    prob = tl_highprec_exp(x - m) / d

            if label_smoothing > 0.0:
                prob = prob - eps
            prob = tl.where(is_target, prob - (1.0 - label_smoothing), prob)
            prob = prob * scale
            prob = tl.where(mask, prob, 0.0)

            tl.store(row_ptr + cols, prob, mask=mask)

else:
    def _exact_ce_fwdbwd_kernel_merged(*args, **kwargs):
        raise RuntimeError("Triton kernels are unavailable; install Triton and use CUDA/HIP tensors.")
