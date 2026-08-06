"""Small inline Triton helpers for the fused CE/KL kernel."""

from __future__ import annotations

from ..._runtime import HAS_TRITON, tl, tl_highprec_exp, tl_highprec_log, triton

# ── Inline KL math ───────────────────────────────────────────────────────────
if HAS_TRITON:

    @triton.jit
    def zero_dlogits_rows(
        row_s,
        row_t,
        n_cols,
        offsets,
        WRITE_TEACHER: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offsets
            mask = cols < n_cols
            tl.store(row_s + cols, 0.0, mask=mask)
            if WRITE_TEACHER:
                tl.store(row_t + cols, 0.0, mask=mask)

    @triton.jit
    def aligned_span_grad(
        row_s,
        T_L_ptr,
        K_ptr,
        row_index,
        n_cols,
        label,
        offsets,
        gradient_materialization_factor_ptr,
        SCALE_GRADIENT_MATERIALIZATION: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        lse = tl.load(T_L_ptr + row_index).to(tl.float32)
        coefficient = tl.load(K_ptr + row_index).to(tl.float32)
        active = coefficient != 0.0
        safe_lse = tl.where(active, lse, 0.0)
        coefficient = tl.where(active, coefficient, 0.0)
        if SCALE_GRADIENT_MATERIALIZATION:
            gradient_materialization_factor = tl.load(
                gradient_materialization_factor_ptr
            ).to(tl.float32)

        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offsets
            mask = cols < n_cols
            logits = tl.load(
                row_s + cols,
                mask=mask,
                other=float("-inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            logits = tl.where(active, logits, 0.0)
            probability = tl_highprec_exp(logits - safe_lse)
            gradient = coefficient * (probability - tl.where(cols == label, 1.0, 0.0))
            gradient = tl.where(active, gradient, 0.0)
            if SCALE_GRADIENT_MATERIALIZATION:
                gradient = gradient * gradient_materialization_factor
            tl.store(row_s + cols, gradient, mask=mask)

    @triton.jit
    def zero_preserving_mul(weight, value, active):
        return tl.where(active & (weight > 0.0), weight * value, 0.0)

    @triton.jit
    def support_safe_logaddexp(left, right):
        maximum = tl.maximum(left, right)
        live = maximum > float("-inf")
        safe_maximum = tl.where(live, maximum, 0.0)
        value = safe_maximum + tl_highprec_log(
            tl_highprec_exp(left - safe_maximum) + tl_highprec_exp(right - safe_maximum)
        )
        return tl.where(live, value, float("-inf")), live

    @triton.jit
    def scaled_log_prob(logit, lse, temperature_inv, BOTH_T_ONE: tl.constexpr):
        # ``lse`` must normalize logits in the same temperature domain used here.
        if BOTH_T_ONE:
            return logit - lse
        return logit * temperature_inv - lse

else:

    def _triton_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "Triton kernels are unavailable; install Triton and use CUDA/HIP tensors."
        )

    def zero_preserving_mul(*args, **kwargs):
        return _triton_unavailable(*args, **kwargs)

    def zero_dlogits_rows(*args, **kwargs):
        return _triton_unavailable(*args, **kwargs)

    def aligned_span_grad(*args, **kwargs):
        return _triton_unavailable(*args, **kwargs)

    def support_safe_logaddexp(*args, **kwargs):
        return _triton_unavailable(*args, **kwargs)

    def scaled_log_prob(*args, **kwargs):
        return _triton_unavailable(*args, **kwargs)
