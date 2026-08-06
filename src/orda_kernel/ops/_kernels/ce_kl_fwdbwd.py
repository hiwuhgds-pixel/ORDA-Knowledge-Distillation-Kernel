"""Canonical fused CE/KL Triton forward-backward kernel."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from ..._runtime import (
    HAS_TRITON,
    supports_bfloat16,
    tl,
    tl_highprec_exp,
    tl_highprec_expm1,
    tl_highprec_log,
    triton,
)
from ._kl_math import (
    aligned_span_grad,
    scaled_log_prob,
    support_safe_logaddexp,
    zero_dlogits_rows,
    zero_preserving_mul,
)

# ── KL specialization ────────────────────────────────────────────────────────
KL_FORWARD = 0
KL_JEFFREYS = 1
KL_MUTUAL = 2
KL_COJEFFREYS = 3
KL_REVERSE = 4
KL_ALPHA_DIVERGENCE = 5
KL_SKEW_FORWARD = 6
KL_SKEW_JS = 7
KL_ENTROPY_ADAPTIVE = 8
KL_SKEW_REVERSE = 9
KL_RENYI = 10
KL_RENYI_REVERSE = 11
KL_DKD = 12
KL_ABKD = 13
KL_TRIANGULAR = 14

KERNEL_STAGE_TOKENWISE = 0
KERNEL_STAGE_ALIGNED_STATS = 1
KERNEL_STAGE_ALIGNED_GRAD = 2

_KL_STYLES = {
    "forward": KL_FORWARD,
    "jeffreys": KL_JEFFREYS,
    "mutual": KL_MUTUAL,
    "cojeffreys": KL_COJEFFREYS,
    "reverse": KL_REVERSE,
    "alpha_divergence": KL_ALPHA_DIVERGENCE,
    "skew_forward": KL_SKEW_FORWARD,
    "skew_js": KL_SKEW_JS,
    "entropy_adaptive": KL_ENTROPY_ADAPTIVE,
    "skew_reverse": KL_SKEW_REVERSE,
    "renyi": KL_RENYI,
    "renyi_reverse": KL_RENYI_REVERSE,
    "dkd": KL_DKD,
    "abkd": KL_ABKD,
    "triangular": KL_TRIANGULAR,
}


def kl_style_of(kl_divergence: str) -> int:
    try:
        return _KL_STYLES[kl_divergence]
    except KeyError as exc:
        raise ValueError(f"unknown KL variant {kl_divergence!r}") from exc


# ── GEMM autocast guard ──────────────────────────────────────────────────────
def compute_autocast_context(
    device: torch.device,
    compute_dtype: torch.dtype,
):
    if device.type != "cuda" or compute_dtype not in (torch.float16, torch.float32):
        return nullcontext()
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return nullcontext()
    try:
        active_dtype = torch.get_autocast_dtype("cuda")
    except (AttributeError, TypeError):
        active_dtype = torch.get_autocast_gpu_dtype()
    if active_dtype != torch.bfloat16 or supports_bfloat16(device):
        return nullcontext()
    try:
        return torch.autocast(device_type="cuda", enabled=False)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=False)


# ── Canonical Triton kernel ──────────────────────────────────────────────────
if HAS_TRITON:
    _KL_FORWARD_C = tl.constexpr(KL_FORWARD)
    _KL_JEFFREYS_C = tl.constexpr(KL_JEFFREYS)
    _KL_MUTUAL_C = tl.constexpr(KL_MUTUAL)
    _KL_COJEFFREYS_C = tl.constexpr(KL_COJEFFREYS)
    _KL_REVERSE_C = tl.constexpr(KL_REVERSE)
    _KL_ALPHA_DIVERGENCE_C = tl.constexpr(KL_ALPHA_DIVERGENCE)
    _KL_SKEW_FORWARD_C = tl.constexpr(KL_SKEW_FORWARD)
    _KL_SKEW_JS_C = tl.constexpr(KL_SKEW_JS)
    _KL_ENTROPY_ADAPTIVE_C = tl.constexpr(KL_ENTROPY_ADAPTIVE)
    _KL_SKEW_REVERSE_C = tl.constexpr(KL_SKEW_REVERSE)
    _KL_RENYI_C = tl.constexpr(KL_RENYI)
    _KL_RENYI_REVERSE_C = tl.constexpr(KL_RENYI_REVERSE)
    _KL_DKD_C = tl.constexpr(KL_DKD)
    _KL_ABKD_C = tl.constexpr(KL_ABKD)
    _KL_TRIANGULAR_C = tl.constexpr(KL_TRIANGULAR)
    _KERNEL_STAGE_ALIGNED_STATS_C = tl.constexpr(KERNEL_STAGE_ALIGNED_STATS)
    _KERNEL_STAGE_ALIGNED_GRAD_C = tl.constexpr(KERNEL_STAGE_ALIGNED_GRAD)

    @triton.jit
    def ce_kl_fwdbwd_kernel(
        S_ptr,
        S_stride,
        T_ptr,
        T_stride,
        Y_ptr,
        Y_stride,
        L_ptr,
        T_L_ptr,
        K_ptr,
        gradient_materialization_factor_ptr,
        n_rows,
        n_cols,
        ignore_index,
        student_scale,
        teacher_scale,
        T_inv_s,
        T_inv_t,
        kl_grad_scale_s,
        kl_loss_scale,
        kl_alpha,
        kl_beta,
        gate_alpha_min,
        gate_alpha_span,
        gate_tau,
        gate_inv_scale,
        inv_log_vocab,
        TIED_LAYOUT: tl.constexpr,
        COMPUTE_TEACHER_CE: tl.constexpr,
        COMPUTE_KL: tl.constexpr,
        TEACHER_KL_GRAD: tl.constexpr,
        BOTH_T_ONE: tl.constexpr,
        KL_STYLE: tl.constexpr,
        KERNEL_STAGE: tl.constexpr,
        ALLOW_TEACHER_NEG_INF_SUPPORT: tl.constexpr,
        RENYI_NEAR_ONE: tl.constexpr,
        SCALE_GRADIENT_MATERIALIZATION: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # TIED_LAYOUT appends teacher rows to S_ptr; otherwise T_ptr is independent.
        # Disabled teacher pointers may alias live buffers and are never accessed.
        i = tl.program_id(0).to(tl.int64)
        y = tl.load(Y_ptr + i * Y_stride)
        ignored = y == ignore_index

        row_s = S_ptr + i * S_stride
        if TIED_LAYOUT:
            row_t = S_ptr + (i + n_rows) * S_stride
        else:
            row_t = T_ptr + i * T_stride
        offs = tl.arange(0, BLOCK_SIZE)

        if ignored:
            if KERNEL_STAGE == _KERNEL_STAGE_ALIGNED_STATS_C:
                tl.store(L_ptr + i, 0.0)
                tl.store(T_L_ptr + i, 0.0)
                return
            # Gradient stages reuse logits as dlogits and must overwrite the row.
            zero_dlogits_rows(
                row_s,
                row_t,
                n_cols,
                offs,
                (KERNEL_STAGE != _KERNEL_STAGE_ALIGNED_GRAD_C)
                and (COMPUTE_TEACHER_CE or TEACHER_KL_GRAD),
                BLOCK_SIZE,
            )
            if KERNEL_STAGE != _KERNEL_STAGE_ALIGNED_GRAD_C:
                tl.store(L_ptr + i, 0.0)
                if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                    tl.store(T_L_ptr + i, 0.0)
                tl.store(K_ptr + i, 0.0)
            return

        in_range = (y >= 0) & (y < n_cols)
        y_safe = tl.where(~in_range, 0, y)

        if KERNEL_STAGE == _KERNEL_STAGE_ALIGNED_GRAD_C:
            aligned_span_grad(
                row_s,
                T_L_ptr,
                K_ptr,
                i,
                n_cols,
                y_safe,
                offs,
                gradient_materialization_factor_ptr,
                SCALE_GRADIENT_MATERIALIZATION,
                BLOCK_SIZE,
            )
            return

        if SCALE_GRADIENT_MATERIALIZATION:
            gradient_materialization_factor = tl.load(
                gradient_materialization_factor_ptr
            ).to(tl.float32)

        student_inactive = student_scale == 0.0
        teacher_inactive = teacher_scale == 0.0
        student_numeric_invalid = False
        if COMPUTE_TEACHER_CE or COMPUTE_KL or TEACHER_KL_GRAD:
            teacher_numeric_invalid = False

        m_s = float("-inf")
        d_s_ce = 0.0
        s_has_value = False
        if COMPUTE_KL and not BOTH_T_ONE:
            d_s_kl = 0.0
        if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
            m_t_ce = float("-inf")
            d_t_ce = 0.0
            t_ce_has_value = False
        if COMPUTE_KL:
            if BOTH_T_ONE:
                if not (COMPUTE_TEACHER_CE or TEACHER_KL_GRAD):
                    m_t_kl = float("-inf")
                    d_t_kl = 0.0
                    t_kl_has_value = False
            else:
                m_t_kl = float("-inf")
                d_t_kl = 0.0
                t_kl_has_value = False
        if (
            KL_STYLE == _KL_JEFFREYS_C
            or KL_STYLE == _KL_COJEFFREYS_C
            or KL_STYLE == _KL_REVERSE_C
            or KL_STYLE == _KL_ENTROPY_ADAPTIVE_C
        ):
            n_diff = 0.0
        if (
            KL_STYLE == _KL_JEFFREYS_C
            or KL_STYLE == _KL_REVERSE_C
            or KL_STYLE == _KL_ENTROPY_ADAPTIVE_C
        ):
            reverse_unsafe = False
        if KL_STYLE == _KL_COJEFFREYS_C:
            n_diff_t = 0.0
        if KL_STYLE == _KL_RENYI_C or KL_STYLE == _KL_RENYI_REVERSE_C:
            if KL_STYLE == _KL_RENYI_C:
                mix_a = 1.0 - kl_alpha
            else:
                mix_a = kl_alpha
            m_renyi_mix = float("-inf")
            d_renyi_mix = 0.0
            renyi_has_finite = False
            renyi_unsafe = False
        if KL_STYLE == _KL_DKD_C:
            m_s_nt = float("-inf")
            d_s_nt = 0.0
            s_nt_has_finite = False
            m_t_nt = float("-inf")
            d_t_nt = 0.0
            t_nt_has_finite = False
            n_diff_t_nt = 0.0
            dkd_nt_unsafe = False
        if KL_STYLE == _KL_ENTROPY_ADAPTIVE_C:
            n_ent = 0.0

        # ── Streaming LSE sweep ──────────────────────────────
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            x_s = tl.load(
                row_s + cols,
                mask=mask,
                other=float("-inf"),
                eviction_policy="evict_last",
            ).to(tl.float32)
            invalid_student_value = (
                (x_s == float("inf")) | (x_s == float("-inf")) | (x_s != x_s)
            )
            student_numeric_invalid = student_numeric_invalid | (
                tl.sum(tl.where(mask & invalid_student_value, 1.0, 0.0)) > 0.0
            )
            if COMPUTE_TEACHER_CE or COMPUTE_KL or TEACHER_KL_GRAD:
                x_t = tl.load(
                    row_t + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                if ALLOW_TEACHER_NEG_INF_SUPPORT:
                    invalid_teacher_value = (x_t == float("inf")) | (x_t != x_t)
                else:
                    invalid_teacher_value = (
                        (x_t == float("inf")) | (x_t == float("-inf")) | (x_t != x_t)
                    )
                teacher_numeric_invalid = teacher_numeric_invalid | (
                    tl.sum(
                        tl.where(
                            mask & invalid_teacher_value,
                            1.0,
                            0.0,
                        )
                    )
                    > 0.0
                )

            s_present = mask & (x_s != float("-inf"))
            s_block_live = tl.sum(tl.where(s_present, 1.0, 0.0)) > 0.0
            s_block_max = tl.max(tl.where(mask, x_s, float("-inf")))
            m_s_candidate = tl.maximum(m_s, s_block_max)
            m_s_new = tl.where(s_block_live, m_s_candidate, m_s)
            old_m_s = tl.where(s_has_value, m_s, 0.0)
            new_m_s = tl.where(s_has_value, m_s_new, 0.0)
            s_old_scale_ce = tl.where(
                s_has_value, tl_highprec_exp(old_m_s - new_m_s), 0.0
            )
            block_m_s = tl.where(s_block_live, m_s_new, 0.0)
            safe_x_s = tl.where(s_present, x_s, 0.0)
            s_block_sum_ce = tl.sum(
                tl.where(s_present, tl_highprec_exp(safe_x_s - block_m_s), 0.0)
            )
            d_s_ce_candidate = d_s_ce * s_old_scale_ce + s_block_sum_ce
            d_s_ce = tl.where(s_block_live, d_s_ce_candidate, d_s_ce)
            if COMPUTE_KL and not BOTH_T_ONE:
                s_old_scale_kl = tl.where(
                    s_has_value,
                    tl_highprec_exp(T_inv_s * (old_m_s - new_m_s)),
                    0.0,
                )
                s_block_sum_kl = tl.sum(
                    tl.where(
                        s_present,
                        tl_highprec_exp((safe_x_s - block_m_s) * T_inv_s),
                        0.0,
                    )
                )
                d_s_kl_candidate = d_s_kl * s_old_scale_kl + s_block_sum_kl
                d_s_kl = tl.where(s_block_live, d_s_kl_candidate, d_s_kl)

            if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                t_ce_present = mask & (x_t != float("-inf"))
                t_ce_block_live = tl.sum(tl.where(t_ce_present, 1.0, 0.0)) > 0.0
                t_ce_block_max = tl.max(tl.where(mask, x_t, float("-inf")))
                m_t_ce_candidate = tl.maximum(m_t_ce, t_ce_block_max)
                m_t_ce_new = tl.where(t_ce_block_live, m_t_ce_candidate, m_t_ce)
                old_m_t_ce = tl.where(t_ce_has_value, m_t_ce, 0.0)
                new_m_t_ce = tl.where(t_ce_has_value, m_t_ce_new, 0.0)
                t_ce_old_scale = tl.where(
                    t_ce_has_value,
                    tl_highprec_exp(old_m_t_ce - new_m_t_ce),
                    0.0,
                )
                block_m_t_ce = tl.where(t_ce_block_live, m_t_ce_new, 0.0)
                safe_x_t_ce = tl.where(t_ce_present, x_t, 0.0)
                t_ce_block_sum = tl.sum(
                    tl.where(
                        t_ce_present,
                        tl_highprec_exp(safe_x_t_ce - block_m_t_ce),
                        0.0,
                    )
                )
                d_t_ce_candidate = d_t_ce * t_ce_old_scale + t_ce_block_sum
                d_t_ce = tl.where(t_ce_block_live, d_t_ce_candidate, d_t_ce)

            if COMPUTE_KL:
                if BOTH_T_ONE:
                    if not (COMPUTE_TEACHER_CE or TEACHER_KL_GRAD):
                        t_kl_present = mask & (x_t != float("-inf"))
                        t_kl_block_live = tl.sum(tl.where(t_kl_present, 1.0, 0.0)) > 0.0
                        t_kl_block_max = tl.max(tl.where(mask, x_t, float("-inf")))
                        m_t_kl_candidate = tl.maximum(m_t_kl, t_kl_block_max)
                        m_t_kl_new = tl.where(t_kl_block_live, m_t_kl_candidate, m_t_kl)
                        old_m_t_kl = tl.where(t_kl_has_value, m_t_kl, 0.0)
                        new_m_t_kl = tl.where(t_kl_has_value, m_t_kl_new, 0.0)
                        t_kl_old_scale = tl.where(
                            t_kl_has_value,
                            tl_highprec_exp(old_m_t_kl - new_m_t_kl),
                            0.0,
                        )
                        block_m_t_kl = tl.where(t_kl_block_live, m_t_kl_new, 0.0)
                        safe_x_t_kl = tl.where(t_kl_present, x_t, 0.0)
                        t_kl_block_sum = tl.sum(
                            tl.where(
                                t_kl_present,
                                tl_highprec_exp(safe_x_t_kl - block_m_t_kl),
                                0.0,
                            )
                        )
                        d_t_kl_candidate = d_t_kl * t_kl_old_scale + t_kl_block_sum
                        d_t_kl = tl.where(t_kl_block_live, d_t_kl_candidate, d_t_kl)
                        m_t_kl = m_t_kl_new
                        t_kl_has_value = t_kl_has_value | t_kl_block_live
                else:
                    x_t_kl = x_t * T_inv_t
                    t_kl_present = mask & (x_t_kl != float("-inf"))
                    t_kl_block_live = tl.sum(tl.where(t_kl_present, 1.0, 0.0)) > 0.0
                    t_kl_block_max = tl.max(tl.where(mask, x_t_kl, float("-inf")))
                    m_t_kl_candidate = tl.maximum(m_t_kl, t_kl_block_max)
                    m_t_kl_new = tl.where(t_kl_block_live, m_t_kl_candidate, m_t_kl)
                    old_m_t_kl = tl.where(t_kl_has_value, m_t_kl, 0.0)
                    new_m_t_kl = tl.where(t_kl_has_value, m_t_kl_new, 0.0)
                    t_kl_old_scale = tl.where(
                        t_kl_has_value,
                        tl_highprec_exp(old_m_t_kl - new_m_t_kl),
                        0.0,
                    )
                    block_m_t_kl = tl.where(t_kl_block_live, m_t_kl_new, 0.0)
                    safe_x_t_kl = tl.where(t_kl_present, x_t_kl, 0.0)
                    t_kl_block_sum = tl.sum(
                        tl.where(
                            t_kl_present,
                            tl_highprec_exp(safe_x_t_kl - block_m_t_kl),
                            0.0,
                        )
                    )
                    d_t_kl_candidate = d_t_kl * t_kl_old_scale + t_kl_block_sum
                    d_t_kl = tl.where(t_kl_block_live, d_t_kl_candidate, d_t_kl)
                    m_t_kl = m_t_kl_new
                    t_kl_has_value = t_kl_has_value | t_kl_block_live

            if (
                KL_STYLE == _KL_JEFFREYS_C
                or KL_STYLE == _KL_COJEFFREYS_C
                or KL_STYLE == _KL_REVERSE_C
                or KL_STYLE == _KL_ENTROPY_ADAPTIVE_C
            ):
                # w > 0 pins 0 * (±inf) to the zero-mass limit; a positive
                # weight against a -inf teacher logit still sums to +inf.
                if BOTH_T_ONE:
                    w_diff = tl.where(
                        s_present,
                        tl_highprec_exp(safe_x_s - block_m_s),
                        0.0,
                    )
                    diff_old_scale = s_old_scale_ce
                else:
                    w_diff = tl.where(
                        s_present,
                        tl_highprec_exp((safe_x_s - block_m_s) * T_inv_s),
                        0.0,
                    )
                    diff_old_scale = s_old_scale_kl
                diff_live = mask & (w_diff > 0)
                if BOTH_T_ONE:
                    safe_diff_s = tl.where(diff_live, x_s, 0.0)
                    safe_diff_t = tl.where(diff_live, x_t, 0.0)
                else:
                    safe_diff_s = tl.where(diff_live, x_s * T_inv_s, 0.0)
                    safe_diff_t = tl.where(diff_live, x_t * T_inv_t, 0.0)
                diff_block_sum = tl.sum(
                    tl.where(
                        diff_live,
                        w_diff * (safe_diff_s - safe_diff_t),
                        0.0,
                    )
                )
                n_diff_candidate = n_diff * diff_old_scale + diff_block_sum
                n_diff = tl.where(s_block_live, n_diff_candidate, n_diff)
                if (
                    KL_STYLE == _KL_JEFFREYS_C
                    or KL_STYLE == _KL_REVERSE_C
                    or KL_STYLE == _KL_ENTROPY_ADAPTIVE_C
                ):
                    reverse_unsafe = reverse_unsafe | (
                        tl.sum(
                            tl.where(
                                mask
                                & (x_s > float("-inf"))
                                & (x_s < float("inf"))
                                & (x_t == float("-inf")),
                                1.0,
                                0.0,
                            )
                        )
                        > 0.0
                    )
            if KL_STYLE == _KL_COJEFFREYS_C:
                if BOTH_T_ONE:
                    w_diff_t = tl.where(
                        t_ce_present,
                        tl_highprec_exp(safe_x_t_ce - block_m_t_ce),
                        0.0,
                    )
                    diff_t_old_scale = t_ce_old_scale
                else:
                    w_diff_t = tl.where(
                        t_ce_present,
                        tl_highprec_exp((safe_x_t_ce - block_m_t_ce) * T_inv_t),
                        0.0,
                    )
                    diff_t_old_scale = tl.where(
                        t_ce_has_value,
                        tl_highprec_exp((old_m_t_ce - new_m_t_ce) * T_inv_t),
                        0.0,
                    )
                diff_t_live = mask & (w_diff_t > 0)
                if BOTH_T_ONE:
                    safe_diff_t = tl.where(diff_t_live, x_t, 0.0)
                    safe_diff_s = tl.where(diff_t_live, x_s, 0.0)
                else:
                    safe_diff_t = tl.where(diff_t_live, x_t * T_inv_t, 0.0)
                    safe_diff_s = tl.where(diff_t_live, x_s * T_inv_s, 0.0)
                diff_t_block_sum = tl.sum(
                    tl.where(
                        diff_t_live,
                        w_diff_t * (safe_diff_t - safe_diff_s),
                        0.0,
                    )
                )
                n_diff_t_candidate = n_diff_t * diff_t_old_scale + diff_t_block_sum
                n_diff_t = tl.where(t_ce_block_live, n_diff_t_candidate, n_diff_t)

            if KL_STYLE == _KL_ENTROPY_ADAPTIVE_C:
                if BOTH_T_ONE:
                    if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                        ent_present = t_ce_present
                        ent_block_live = t_ce_block_live
                        ent_block_m = block_m_t_ce
                        ent_old_scale = t_ce_old_scale
                        ent_x = x_t
                        ent_safe_x = safe_x_t_ce
                    else:
                        ent_present = t_kl_present
                        ent_block_live = t_kl_block_live
                        ent_block_m = block_m_t_kl
                        ent_old_scale = t_kl_old_scale
                        ent_x = x_t
                        ent_safe_x = safe_x_t_kl
                else:
                    ent_present = t_kl_present
                    ent_block_live = t_kl_block_live
                    ent_block_m = block_m_t_kl
                    ent_old_scale = t_kl_old_scale
                    ent_x = x_t_kl
                    ent_safe_x = safe_x_t_kl
                w_ent = tl.where(
                    ent_present,
                    tl_highprec_exp(ent_safe_x - ent_block_m),
                    0.0,
                )
                ent_live = mask & (w_ent > 0)
                safe_ent_x = tl.where(ent_live, ent_x, 0.0)
                ent_block_sum = tl.sum(tl.where(ent_live, w_ent * safe_ent_x, 0.0))
                n_ent_candidate = n_ent * ent_old_scale + ent_block_sum
                n_ent = tl.where(ent_block_live, n_ent_candidate, n_ent)

            if KL_STYLE == _KL_RENYI_C or KL_STYLE == _KL_RENYI_REVERSE_C:
                both_dead = (x_s == float("-inf")) & (x_t == float("-inf"))
                x_s_mix = tl.where(both_dead, 0.0, x_s)
                x_t_mix = tl.where(both_dead, 0.0, x_t)
                if BOTH_T_ONE:
                    u = mix_a * x_s_mix + (1.0 - mix_a) * x_t_mix
                else:
                    u = mix_a * x_s_mix * T_inv_s + (1.0 - mix_a) * x_t_mix * T_inv_t
                u = tl.where(both_dead, float("-inf"), u)
                finite_u = mask & (u > float("-inf")) & (u < float("inf"))
                block_has_finite = tl.sum(tl.where(finite_u, 1.0, 0.0)) > 0.0
                block_unsafe = (
                    tl.sum(tl.where(mask & (u == float("inf")), 1.0, 0.0)) > 0.0
                )

                u_for_max = tl.where(finite_u, u, float("-inf"))
                block_max = tl.max(u_for_max)
                m_candidate = tl.maximum(m_renyi_mix, block_max)
                m_renyi_mix_new = tl.where(block_has_finite, m_candidate, m_renyi_mix)
                old_m_safe = tl.where(renyi_has_finite, m_renyi_mix, 0.0)
                new_m_safe = tl.where(renyi_has_finite, m_renyi_mix_new, 0.0)
                old_scale = tl.where(
                    renyi_has_finite,
                    tl_highprec_exp(old_m_safe - new_m_safe),
                    0.0,
                )
                block_m_safe = tl.where(block_has_finite, m_renyi_mix_new, 0.0)
                u_for_exp = tl.where(finite_u, u, 0.0)
                block_sum = tl.sum(
                    tl.where(
                        finite_u,
                        tl_highprec_exp(u_for_exp - block_m_safe),
                        0.0,
                    )
                )
                d_candidate = d_renyi_mix * old_scale + block_sum
                d_renyi_mix = tl.where(block_has_finite, d_candidate, d_renyi_mix)
                m_renyi_mix = m_renyi_mix_new
                renyi_has_finite = renyi_has_finite | block_has_finite
                renyi_unsafe = renyi_unsafe | block_unsafe

            if KL_STYLE == _KL_DKD_C:
                nt_mask = mask & (cols != y_safe)
                if BOTH_T_ONE:
                    z_s_nt = x_s
                    z_t_nt = x_t
                else:
                    z_s_nt = x_s * T_inv_s
                    z_t_nt = x_t * T_inv_t
                finite_s_nt = (
                    nt_mask & (z_s_nt > float("-inf")) & (z_s_nt < float("inf"))
                )
                finite_t_nt = (
                    nt_mask & (z_t_nt > float("-inf")) & (z_t_nt < float("inf"))
                )
                block_s_nt = tl.sum(tl.where(finite_s_nt, 1.0, 0.0)) > 0.0
                block_t_nt = tl.sum(tl.where(finite_t_nt, 1.0, 0.0)) > 0.0

                s_nt_max = tl.max(tl.where(finite_s_nt, z_s_nt, float("-inf")))
                m_s_nt_candidate = tl.maximum(m_s_nt, s_nt_max)
                m_s_nt_new = tl.where(block_s_nt, m_s_nt_candidate, m_s_nt)
                old_m_s_nt = tl.where(s_nt_has_finite, m_s_nt, 0.0)
                new_m_s_nt = tl.where(s_nt_has_finite, m_s_nt_new, 0.0)
                s_nt_old_scale = tl.where(
                    s_nt_has_finite,
                    tl_highprec_exp(old_m_s_nt - new_m_s_nt),
                    0.0,
                )
                block_m_s_nt = tl.where(block_s_nt, m_s_nt_new, 0.0)
                safe_z_s_nt = tl.where(finite_s_nt, z_s_nt, 0.0)
                s_nt_block_sum = tl.sum(
                    tl.where(
                        finite_s_nt,
                        tl_highprec_exp(safe_z_s_nt - block_m_s_nt),
                        0.0,
                    )
                )
                d_s_nt_candidate = d_s_nt * s_nt_old_scale + s_nt_block_sum
                d_s_nt = tl.where(block_s_nt, d_s_nt_candidate, d_s_nt)
                m_s_nt = m_s_nt_new

                t_nt_max = tl.max(tl.where(finite_t_nt, z_t_nt, float("-inf")))
                m_t_nt_candidate = tl.maximum(m_t_nt, t_nt_max)
                m_t_nt_new = tl.where(block_t_nt, m_t_nt_candidate, m_t_nt)
                old_m_t_nt = tl.where(t_nt_has_finite, m_t_nt, 0.0)
                new_m_t_nt = tl.where(t_nt_has_finite, m_t_nt_new, 0.0)
                t_nt_old_scale = tl.where(
                    t_nt_has_finite,
                    tl_highprec_exp(old_m_t_nt - new_m_t_nt),
                    0.0,
                )
                block_m_t_nt = tl.where(block_t_nt, m_t_nt_new, 0.0)
                safe_z_t_nt = tl.where(finite_t_nt, z_t_nt, 0.0)
                t_nt_weights = tl.where(
                    finite_t_nt,
                    tl_highprec_exp(safe_z_t_nt - block_m_t_nt),
                    0.0,
                )
                t_nt_block_sum = tl.sum(t_nt_weights)
                d_t_nt_candidate = d_t_nt * t_nt_old_scale + t_nt_block_sum
                d_t_nt = tl.where(block_t_nt, d_t_nt_candidate, d_t_nt)

                diff_live = finite_t_nt & finite_s_nt
                safe_diff_t = tl.where(diff_live, z_t_nt, 0.0)
                safe_diff_s = tl.where(diff_live, z_s_nt, 0.0)
                n_diff_block = tl.sum(
                    tl.where(
                        diff_live,
                        t_nt_weights * (safe_diff_t - safe_diff_s),
                        0.0,
                    )
                )
                n_diff_candidate = n_diff_t_nt * t_nt_old_scale + n_diff_block
                n_diff_t_nt = tl.where(block_t_nt, n_diff_candidate, n_diff_t_nt)
                m_t_nt = m_t_nt_new
                s_nt_has_finite = s_nt_has_finite | block_s_nt
                t_nt_has_finite = t_nt_has_finite | block_t_nt
                dkd_nt_unsafe = dkd_nt_unsafe | (
                    tl.sum(tl.where(finite_t_nt & (~finite_s_nt), 1.0, 0.0)) > 0.0
                )

            if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                m_t_ce = m_t_ce_new
                t_ce_has_value = t_ce_has_value | t_ce_block_live
            m_s = m_s_new
            s_has_value = s_has_value | s_block_live

        # ── Row scalars and CE ───────────────────────────────
        teacher_row_dead = False
        if ALLOW_TEACHER_NEG_INF_SUPPORT:
            # An all--inf raw cache row has no teacher support.
            if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                teacher_row_dead = m_t_ce == float("-inf")
            elif COMPUTE_KL:
                teacher_row_dead = m_t_kl == float("-inf")
        elif COMPUTE_TEACHER_CE or COMPUTE_KL or TEACHER_KL_GRAD:
            teacher_row_dead = False

        log_d_s_ce = tl_highprec_log(d_s_ce)
        lse_s_ce = m_s + log_d_s_ce
        logit_label_s = tl.load(row_s + y_safe).to(tl.float32)
        if KERNEL_STAGE == _KERNEL_STAGE_ALIGNED_STATS_C or COMPUTE_KL:
            student_numeric_active = student_numeric_invalid
        else:
            student_numeric_active = student_numeric_invalid & (~student_inactive)
        student_ce_row = tl.where(
            student_numeric_active,
            float("nan"),
            lse_s_ce - logit_label_s,
        )
        if KERNEL_STAGE == _KERNEL_STAGE_ALIGNED_STATS_C:
            tl.store(L_ptr + i, student_ce_row)
            tl.store(
                T_L_ptr + i,
                tl.where(student_numeric_active, float("nan"), lse_s_ce),
            )
            return
        tl.store(
            L_ptr + i,
            tl.where(student_inactive, 0.0, student_ce_row),
        )

        if COMPUTE_TEACHER_CE or (COMPUTE_KL and KL_STYLE == _KL_DKD_C):
            logit_label_t = tl.load(row_t + y_safe).to(tl.float32)
        if COMPUTE_TEACHER_CE:
            log_d_t_ce = tl_highprec_log(d_t_ce)
            lse_t_ce = m_t_ce + log_d_t_ce
            teacher_ce_row = tl.where(
                teacher_numeric_invalid,
                float("nan"),
                lse_t_ce - logit_label_t,
            )
            tl.store(
                T_L_ptr + i,
                tl.where(
                    teacher_inactive | teacher_row_dead,
                    0.0,
                    teacher_ce_row,
                ),
            )
        elif TEACHER_KL_GRAD:
            tl.store(T_L_ptr + i, 0.0)

        if COMPUTE_KL:
            if BOTH_T_ONE:
                m_s_kl = m_s
                log_d_s_kl = log_d_s_ce
                if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                    m_t_kl = m_t_ce
                    log_d_t_kl = tl_highprec_log(d_t_ce)
                else:
                    log_d_t_kl = tl_highprec_log(d_t_kl)
            else:
                m_s_kl = m_s * T_inv_s
                log_d_s_kl = tl_highprec_log(d_s_kl)
                log_d_t_kl = tl_highprec_log(d_t_kl)

        # ── Per-variant row terms ────────────────────────────
        if (
            KL_STYLE == _KL_JEFFREYS_C
            or KL_STYLE == _KL_COJEFFREYS_C
            or KL_STYLE == _KL_REVERSE_C
            or KL_STYLE == _KL_ENTROPY_ADAPTIVE_C
        ):
            lse_s_kl = m_s_kl + log_d_s_kl
            lse_t_kl = m_t_kl + log_d_t_kl
            if KL_STYLE == _KL_COJEFFREYS_C:
                if BOTH_T_ONE:
                    rkl_row = n_diff / d_s_ce - (lse_s_kl - lse_t_kl)
                else:
                    rkl_row = n_diff / d_s_kl - (lse_s_kl - lse_t_kl)
            else:
                reverse_invalid = reverse_unsafe & (~teacher_row_dead)
                reverse_live = (~reverse_invalid) & (~teacher_row_dead)
                safe_n_diff = tl.where(reverse_live, n_diff, 0.0)
                safe_lse_s_rkl = tl.where(reverse_live, lse_s_kl, 0.0)
                safe_lse_t_rkl = tl.where(reverse_live, lse_t_kl, 0.0)
                if BOTH_T_ONE:
                    safe_d_rkl = tl.where(reverse_live, d_s_ce, 1.0)
                    finite_rkl = safe_n_diff / safe_d_rkl - (
                        safe_lse_s_rkl - safe_lse_t_rkl
                    )
                else:
                    safe_d_rkl = tl.where(reverse_live, d_s_kl, 1.0)
                    finite_rkl = safe_n_diff / safe_d_rkl - (
                        safe_lse_s_rkl - safe_lse_t_rkl
                    )
                rkl_grad_row = tl.where(reverse_live, finite_rkl, 0.0)
                rkl_row = tl.where(reverse_invalid, float("inf"), rkl_grad_row)
        if KL_STYLE == _KL_COJEFFREYS_C:
            if BOTH_T_ONE:
                fkl_row = n_diff_t / d_t_ce - (lse_t_kl - lse_s_kl)
            else:
                fkl_row = n_diff_t / d_t_kl - (lse_t_kl - lse_s_kl)
        if KL_STYLE == _KL_ALPHA_DIVERGENCE_C:
            lse_s_kl = m_s_kl + log_d_s_kl
            lse_t_kl = m_t_kl + log_d_t_kl
            alpha_live = ~teacher_row_dead
        if KL_STYLE == _KL_RENYI_C or KL_STYLE == _KL_RENYI_REVERSE_C:
            lse_s_renyi = m_s_kl + log_d_s_kl
            lse_t_renyi = m_t_kl + log_d_t_kl
            renyi_invalid = renyi_unsafe | (renyi_has_finite == 0)
            safe_lse_s = tl.where(renyi_invalid, 0.0, lse_s_renyi)
            safe_lse_t = tl.where(renyi_invalid, 0.0, lse_t_renyi)
            safe_m_mix = tl.where(renyi_invalid, 0.0, m_renyi_mix)
            safe_d_mix = tl.where(renyi_invalid, 1.0, d_renyi_mix)
            lse_renyi_mix = safe_m_mix + tl_highprec_log(safe_d_mix)
            renyi_log_a_lse = (
                lse_renyi_mix - mix_a * safe_lse_s - (1.0 - mix_a) * safe_lse_t
            )
            renyi_valid = (renyi_invalid == 0) & (~teacher_row_dead)
        if KL_STYLE == _KL_DKD_C:
            dkd_base_live = ~teacher_row_dead
            lse_s_dkd = m_s_kl + log_d_s_kl
            lse_t_dkd = m_t_kl + log_d_t_kl
            safe_lse_s_dkd = tl.where(dkd_base_live, lse_s_dkd, 0.0)
            safe_lse_t_dkd = tl.where(dkd_base_live, lse_t_dkd, 0.0)
            safe_logit_s_y = tl.where(dkd_base_live, logit_label_s, 0.0)
            safe_logit_t_y = tl.where(dkd_base_live, logit_label_t, 0.0)
            log_p_y = tl.where(
                dkd_base_live,
                safe_logit_s_y * T_inv_s - safe_lse_s_dkd,
                float("-inf"),
            )
            log_q_y = tl.where(
                dkd_base_live,
                safe_logit_t_y * T_inv_t - safe_lse_t_dkd,
                float("-inf"),
            )

            safe_m_s_nt = tl.where(s_nt_has_finite, m_s_nt, 0.0)
            safe_d_s_nt = tl.where(s_nt_has_finite, d_s_nt, 1.0)
            safe_m_t_nt = tl.where(t_nt_has_finite, m_t_nt, 0.0)
            safe_d_t_nt = tl.where(t_nt_has_finite, d_t_nt, 1.0)
            lse_s_nt = safe_m_s_nt + tl_highprec_log(safe_d_s_nt)
            lse_t_nt = safe_m_t_nt + tl_highprec_log(safe_d_t_nt)
            log_p_not = tl.where(
                dkd_base_live & s_nt_has_finite,
                lse_s_nt - safe_lse_s_dkd,
                float("-inf"),
            )
            log_q_not = tl.where(
                dkd_base_live & t_nt_has_finite,
                lse_t_nt - safe_lse_t_dkd,
                float("-inf"),
            )

            p_y_support = (log_p_y > float("-inf")) & (log_p_y < float("inf"))
            q_y_support = (log_q_y > float("-inf")) & (log_q_y < float("inf"))
            dkd_unsafe = dkd_base_live & (
                dkd_nt_unsafe
                | (q_y_support & (~p_y_support))
                | (t_nt_has_finite & (~s_nt_has_finite))
            )
            dkd_valid = dkd_base_live & (dkd_unsafe == 0)

            q_y_live = dkd_valid & q_y_support
            q_not_live = dkd_valid & t_nt_has_finite
            safe_log_q_y = tl.where(q_y_live, log_q_y, 0.0)
            safe_log_p_y = tl.where(q_y_live, log_p_y, 0.0)
            safe_log_q_not = tl.where(q_not_live, log_q_not, 0.0)
            safe_log_p_not = tl.where(q_not_live, log_p_not, 0.0)
            q_y_row = tl.where(q_y_live, tl_highprec_exp(safe_log_q_y), 0.0)
            q_not_row = tl.where(q_not_live, tl_highprec_exp(safe_log_q_not), 0.0)
            p_y_live = dkd_valid & p_y_support
            safe_log_p_y_row = tl.where(p_y_live, log_p_y, 0.0)
            p_y_row = tl.where(p_y_live, tl_highprec_exp(safe_log_p_y_row), 0.0)
            tckd_row = q_y_row * (safe_log_q_y - safe_log_p_y) + q_not_row * (
                safe_log_q_not - safe_log_p_not
            )

            nckd_live = dkd_valid & t_nt_has_finite
            safe_n_diff_t_nt = tl.where(nckd_live, n_diff_t_nt, 0.0)
            safe_nckd_denom = tl.where(nckd_live, d_t_nt, 1.0)
            nckd_lse_s_nt = tl.where(nckd_live, lse_s_nt, 0.0)
            nckd_lse_t_nt = tl.where(nckd_live, lse_t_nt, 0.0)
            nckd_row = tl.where(
                nckd_live,
                safe_n_diff_t_nt / safe_nckd_denom - (nckd_lse_t_nt - nckd_lse_s_nt),
                0.0,
            )
            finite_dkd_loss = kl_alpha * tckd_row + kl_beta * nckd_row
            dkd_loss_row = tl.where(
                dkd_unsafe,
                float("inf"),
                tl.where(dkd_base_live, finite_dkd_loss, 0.0),
            )
        if KL_STYLE == _KL_ALPHA_DIVERGENCE_C:
            alpha_correction = 0.0
            alpha_base_is_teacher = kl_alpha <= 0.5
            alpha_exponent = tl.where(
                alpha_base_is_teacher,
                kl_alpha,
                1.0 - kl_alpha,
            )
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offs
                mask = cols < n_cols
                x_s = tl.load(
                    row_s + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                x_t = tl.load(
                    row_t + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                log_p_s = scaled_log_prob(x_s, lse_s_kl, T_inv_s, BOTH_T_ONE)
                log_p_t = scaled_log_prob(x_t, lse_t_kl, T_inv_t, BOTH_T_ONE)
                log_base = tl.where(alpha_base_is_teacher, log_p_t, log_p_s)
                log_other = tl.where(alpha_base_is_teacher, log_p_s, log_p_t)
                base_live = (
                    mask
                    & alpha_live
                    & (log_base > float("-inf"))
                    & (log_base < float("inf"))
                )
                safe_base = tl.where(base_live, log_base, 0.0)
                safe_delta = tl.where(base_live, log_other - log_base, 0.0)
                base_mass = tl.where(base_live, tl_highprec_exp(safe_base), 0.0)
                alpha_correction += tl.sum(
                    zero_preserving_mul(
                        base_mass,
                        tl_highprec_expm1(alpha_exponent * safe_delta),
                        base_live,
                    )
                )
            finite_alpha_loss = -alpha_correction / (kl_alpha * (1.0 - kl_alpha))
            alpha_affinity = 1.0 + alpha_correction
            alpha_loss_row = tl.where(
                alpha_live,
                tl.maximum(finite_alpha_loss, 0.0),
                0.0,
            )
        if KL_STYLE == _KL_RENYI_C or KL_STYLE == _KL_RENYI_REVERSE_C:
            if RENYI_NEAR_ONE:
                renyi_s = 0.0
                renyi_outside_mass = 0.0
                for start in range(0, n_cols, BLOCK_SIZE):
                    cols = start + offs
                    mask = cols < n_cols
                    x_s = tl.load(
                        row_s + cols,
                        mask=mask,
                        other=float("-inf"),
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                    x_t = tl.load(
                        row_t + cols,
                        mask=mask,
                        other=float("-inf"),
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                    log_p_s = scaled_log_prob(x_s, lse_s_renyi, T_inv_s, BOTH_T_ONE)
                    log_p_t = scaled_log_prob(x_t, lse_t_renyi, T_inv_t, BOTH_T_ONE)
                    if KL_STYLE == _KL_RENYI_C:
                        log_base = log_p_t
                        log_other = log_p_s
                    else:
                        log_base = log_p_s
                        log_other = log_p_t
                    base_support = (log_base > float("-inf")) & (
                        log_base < float("inf")
                    )
                    other_support = (log_other > float("-inf")) & (
                        log_other < float("inf")
                    )
                    term_live = mask & renyi_valid & base_support
                    outside_live = mask & renyi_valid & (~base_support) & other_support
                    safe_log_base = tl.where(term_live, log_base, 0.0)
                    safe_log_other = tl.where(term_live, log_other, 0.0)
                    base = tl.where(term_live, tl_highprec_exp(safe_log_base), 0.0)
                    safe_outside_log = tl.where(outside_live, log_other, 0.0)
                    renyi_outside_mass += tl.sum(
                        tl.where(
                            outside_live,
                            tl_highprec_exp(safe_outside_log),
                            0.0,
                        )
                    )
                    renyi_delta = safe_log_other - safe_log_base
                    renyi_core = tl_highprec_expm1((1.0 - kl_alpha) * renyi_delta) - (
                        1.0 - kl_alpha
                    ) * tl_highprec_expm1(renyi_delta)
                    renyi_s += tl.sum(
                        tl.where(
                            term_live,
                            base * renyi_core,
                            0.0,
                        )
                    )
                corrected_renyi_s = renyi_s - (1.0 - kl_alpha) * renyi_outside_mass
                renyi_cv_valid = (
                    renyi_valid
                    & (corrected_renyi_s == corrected_renyi_s)
                    & (corrected_renyi_s > -1.0)
                    & (corrected_renyi_s < float("inf"))
                )
                safe_renyi_s = tl.where(renyi_cv_valid, corrected_renyi_s, 0.0)
                renyi_s2 = safe_renyi_s * safe_renyi_s
                renyi_log_a_taylor = (
                    safe_renyi_s
                    - 0.5 * renyi_s2
                    + renyi_s2 * safe_renyi_s / 3.0
                    - renyi_s2 * renyi_s2 / 4.0
                )
                renyi_log_a_direct = tl_highprec_log(1.0 + safe_renyi_s)
                renyi_log_a_stable = tl.where(
                    tl.abs(safe_renyi_s) <= 1.0e-2,
                    renyi_log_a_taylor,
                    renyi_log_a_direct,
                )
                renyi_log_a = tl.where(
                    renyi_cv_valid,
                    renyi_log_a_stable,
                    renyi_log_a_lse,
                )
            else:
                renyi_log_a = renyi_log_a_lse
            # Rényi is non-negative; pin residual fp32 sign noise at identity.
            finite_renyi_loss = tl.maximum(renyi_log_a / (kl_alpha - 1.0), 0.0)
            renyi_loss_row = tl.where(
                renyi_invalid,
                float("inf"),
                tl.where(renyi_valid, finite_renyi_loss, 0.0),
            )
        if KL_STYLE == _KL_ABKD_C:
            abkd_live = ~teacher_row_dead
            ab_power = kl_alpha + kl_beta
            lse_s_ab = m_s_kl + log_d_s_kl
            lse_t_ab = m_t_kl + log_d_t_kl
            abkd_loss_row = 0.0
            abkd_h_row = 0.0
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offs
                mask = cols < n_cols
                x_s = tl.load(
                    row_s + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                x_t = tl.load(
                    row_t + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                log_p_s = scaled_log_prob(x_s, lse_s_ab, T_inv_s, BOTH_T_ONE)
                log_p_t = scaled_log_prob(x_t, lse_t_ab, T_inv_t, BOTH_T_ONE)
                both_dead = (log_p_s == float("-inf")) & (log_p_t == float("-inf"))
                safe_log_p = tl.where(both_dead, 0.0, log_p_s)
                safe_log_q = tl.where(both_dead, 0.0, log_p_t)
                q_le_p = safe_log_q <= safe_log_p
                low_delta = safe_log_q - safe_log_p
                high_delta = safe_log_p - safe_log_q
                low_d2 = low_delta * low_delta
                high_d2 = high_delta * high_delta
                low_taylor = (
                    0.5 * low_d2
                    + ((kl_alpha + 2.0 * kl_beta) / 6.0) * low_d2 * low_delta
                    + (
                        (
                            kl_alpha * kl_alpha
                            + 3.0 * kl_alpha * kl_beta
                            + 3.0 * kl_beta * kl_beta
                        )
                        / 24.0
                    )
                    * low_d2
                    * low_d2
                )
                high_taylor = (
                    0.5 * high_d2
                    + ((2.0 * kl_alpha + kl_beta) / 6.0) * high_d2 * high_delta
                    + (
                        (
                            3.0 * kl_alpha * kl_alpha
                            + 3.0 * kl_alpha * kl_beta
                            + kl_beta * kl_beta
                        )
                        / 24.0
                    )
                    * high_d2
                    * high_d2
                )
                low_exact = (
                    (kl_beta / ab_power) * tl_highprec_expm1(ab_power * low_delta)
                    - tl_highprec_expm1(kl_beta * low_delta)
                ) / (kl_alpha * kl_beta)
                high_exact = (
                    (kl_alpha / ab_power) * tl_highprec_expm1(ab_power * high_delta)
                    - tl_highprec_expm1(kl_alpha * high_delta)
                ) / (kl_alpha * kl_beta)
                low_core = tl.where(tl.abs(low_delta) <= 1.0e-2, low_taylor, low_exact)
                high_core = tl.where(
                    tl.abs(high_delta) <= 1.0e-2, high_taylor, high_exact
                )
                p_power = tl_highprec_exp(ab_power * safe_log_p)
                q_power = tl_highprec_exp(ab_power * safe_log_q)
                d_col = tl.where(q_le_p, p_power * low_core, q_power * high_core)
                h_low = -p_power * tl_highprec_expm1(kl_beta * low_delta)
                h_high = (
                    q_power
                    * tl_highprec_exp(kl_alpha * high_delta)
                    * tl_highprec_expm1(kl_beta * high_delta)
                )
                live = mask & abkd_live & (~both_dead)
                abkd_loss_row += tl.sum(tl.where(live, d_col, 0.0))
                abkd_h_row += tl.sum(
                    tl.where(live, tl.where(q_le_p, h_low, h_high), 0.0)
                )
        if KL_STYLE == _KL_ENTROPY_ADAPTIVE_C:
            # Teacher entropy rides the same LSE that fed ``n_ent`` in pass 1.
            if BOTH_T_ONE:
                if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                    teacher_moment = n_ent / d_t_ce
                else:
                    teacher_moment = n_ent / d_t_kl
            else:
                teacher_moment = n_ent / d_t_kl
            entropy_row = lse_t_kl - teacher_moment
            gate_row = gate_alpha_min + gate_alpha_span / (
                1.0
                + tl_highprec_exp(
                    -(entropy_row * inv_log_vocab - gate_tau) * gate_inv_scale
                )
            )
            entropy_invalid = reverse_invalid & (gate_row < 1.0)
        if (
            KL_STYLE == _KL_SKEW_FORWARD_C
            or KL_STYLE == _KL_SKEW_JS_C
            or KL_STYLE == _KL_SKEW_REVERSE_C
        ):
            # Third logical vocab sweep: accumulate the mixture row scalar.
            lse_s_kl = m_s_kl + log_d_s_kl
            lse_t_kl = m_t_kl + log_d_t_kl
            log_beta = tl_highprec_log(kl_beta)
            log_1m_beta = tl_highprec_log(1.0 - kl_beta)
            skew_row = 0.0
            if KL_STYLE == _KL_SKEW_REVERSE_C:
                skew_ratio = 0.0
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offs
                mask = cols < n_cols
                x_s = tl.load(
                    row_s + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                x_t = tl.load(
                    row_t + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                log_p_s = scaled_log_prob(x_s, lse_s_kl, T_inv_s, BOTH_T_ONE)
                log_p_t = scaled_log_prob(x_t, lse_t_kl, T_inv_t, BOTH_T_ONE)
                a = log_1m_beta + log_p_t
                b = log_beta + log_p_s
                log_m, mix_live = support_safe_logaddexp(a, b)
                if KL_STYLE == _KL_SKEW_FORWARD_C:
                    skew_row += tl.sum(
                        tl.where(
                            mask & mix_live,
                            tl_highprec_exp(log_beta + log_p_t + log_p_s - log_m),
                            0.0,
                        )
                    )
                else:
                    w_js = tl_highprec_exp(log_p_s)
                    skew_row += tl.sum(zero_preserving_mul(w_js, log_p_s - log_m, mask))
                    if KL_STYLE == _KL_SKEW_REVERSE_C:
                        ratio_live = mask & mix_live & (w_js > 0)
                        ratio_log = tl.where(
                            ratio_live,
                            log_beta + 2.0 * log_p_s - log_m,
                            0.0,
                        )
                        skew_ratio += tl.sum(
                            tl.where(
                                ratio_live,
                                tl_highprec_exp(ratio_log),
                                0.0,
                            )
                        )
        if KL_STYLE == _KL_TRIANGULAR_C:
            lse_s_tri = m_s_kl + log_d_s_kl
            lse_t_tri = m_t_kl + log_d_t_kl
            delta_row = 0.0
            epg_row = 0.0
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offs
                mask = cols < n_cols
                x_s = tl.load(
                    row_s + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                x_t = tl.load(
                    row_t + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                log_p_s = scaled_log_prob(x_s, lse_s_tri, T_inv_s, BOTH_T_ONE)
                log_p_t = scaled_log_prob(x_t, lse_t_tri, T_inv_t, BOTH_T_ONE)
                p_tri = tl_highprec_exp(log_p_s)
                q_tri = tl_highprec_exp(log_p_t)
                sum_pq = p_tri + q_tri
                tri_live = mask & (sum_pq > 0.0)
                safe_sum_pq = tl.where(tri_live, sum_pq, 1.0)
                diff_pq = tl.where(tri_live, p_tri - q_tri, 0.0)
                tri_g = tl.where(
                    tri_live,
                    diff_pq * (p_tri + 3.0 * q_tri) / (safe_sum_pq * safe_sum_pq),
                    0.0,
                )
                delta_row += tl.sum(
                    tl.where(
                        tri_live,
                        diff_pq * diff_pq / safe_sum_pq,
                        0.0,
                    )
                )
                epg_row += tl.sum(tl.where(tri_live, p_tri * tri_g, 0.0))

        # ── KL loss and gradients ────────────────────────────
        kl_row = 0.0
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            is_label = cols == y_safe

            x_s = tl.load(
                row_s + cols,
                mask=mask,
                other=float("-inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            if COMPUTE_TEACHER_CE or COMPUTE_KL or TEACHER_KL_GRAD:
                x_t = tl.load(
                    row_t + cols,
                    mask=mask,
                    other=float("-inf"),
                    eviction_policy="evict_first",
                ).to(tl.float32)

            # CE gradients.
            student_prob_ce = tl_highprec_exp(x_s - m_s) / d_s_ce
            student_grad = student_prob_ce
            student_grad = tl.where(is_label, student_grad - 1.0, student_grad)
            student_grad = tl.where(
                student_inactive,
                0.0,
                student_grad * student_scale,
            )

            if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                teacher_prob_ce = tl_highprec_exp(x_t - m_t_ce) / d_t_ce
                if COMPUTE_TEACHER_CE:
                    teacher_grad = teacher_prob_ce
                    teacher_grad = tl.where(is_label, teacher_grad - 1.0, teacher_grad)
                    teacher_grad = tl.where(
                        teacher_inactive,
                        0.0,
                        teacher_grad * teacher_scale,
                    )
                else:
                    teacher_grad = 0.0

            # KL loss and student gradient.
            grad_s = student_grad
            if COMPUTE_KL:
                if BOTH_T_ONE:
                    p_s_kl = student_prob_ce
                    log_p_s = x_s - m_s_kl - log_d_s_kl
                    if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                        p_t_kl = teacher_prob_ce
                        log_p_t = x_t - m_t_kl - log_d_t_kl
                    else:
                        log_p_t = x_t - m_t_kl - log_d_t_kl
                        p_t_kl = tl_highprec_exp(log_p_t)
                else:
                    log_p_s = x_s * T_inv_s - m_s_kl - log_d_s_kl
                    p_s_kl = tl_highprec_exp(log_p_s)
                    log_p_t = x_t * T_inv_t - m_t_kl - log_d_t_kl
                    p_t_kl = tl_highprec_exp(log_p_t)

                if (
                    KL_STYLE == _KL_FORWARD_C
                    or KL_STYLE == _KL_JEFFREYS_C
                    or KL_STYLE == _KL_MUTUAL_C
                    or KL_STYLE == _KL_COJEFFREYS_C
                    or KL_STYLE == _KL_ENTROPY_ADAPTIVE_C
                ):
                    # p == 0 pins p * log-ratio to the zero-mass limit (raw is
                    # 0 * -inf = NaN); p > 0 vs a zero opposite still gives +inf.
                    kl_forward = tl.sum(
                        zero_preserving_mul(p_t_kl, log_p_t - log_p_s, mask)
                    )
                    kl_row += kl_forward
                if (
                    KL_STYLE == _KL_FORWARD_C
                    or KL_STYLE == _KL_JEFFREYS_C
                    or KL_STYLE == _KL_MUTUAL_C
                    or KL_STYLE == _KL_COJEFFREYS_C
                ):
                    grad_s = student_grad + kl_grad_scale_s * (p_s_kl - p_t_kl)

                if KL_STYLE == _KL_MUTUAL_C or KL_STYLE == _KL_COJEFFREYS_C:
                    kl_row += tl.sum(
                        zero_preserving_mul(p_s_kl, log_p_s - log_p_t, mask)
                    )
                if KL_STYLE == _KL_JEFFREYS_C or KL_STYLE == _KL_REVERSE_C:
                    grad_s = grad_s + kl_grad_scale_s * zero_preserving_mul(
                        p_s_kl,
                        log_p_s - log_p_t - rkl_grad_row,
                        mask,
                    )
                    grad_s = tl.where(reverse_invalid, student_grad, grad_s)
                if KL_STYLE == _KL_COJEFFREYS_C:
                    grad_s = grad_s + kl_grad_scale_s * zero_preserving_mul(
                        p_s_kl,
                        log_p_s - log_p_t - rkl_row,
                        mask,
                    )
                if KL_STYLE == _KL_ENTROPY_ADAPTIVE_C:
                    entropy_delta = tl.where(
                        gate_row > 0.0, gate_row * (p_s_kl - p_t_kl), 0.0
                    ) + tl.where(
                        gate_row < 1.0,
                        (1.0 - gate_row)
                        * zero_preserving_mul(
                            p_s_kl,
                            log_p_s - log_p_t - rkl_grad_row,
                            mask,
                        ),
                        0.0,
                    )
                    grad_s = tl.where(
                        entropy_invalid,
                        student_grad,
                        student_grad + kl_grad_scale_s * entropy_delta,
                    )
                if KL_STYLE == _KL_ALPHA_DIVERGENCE_C:
                    alpha_student_live = (
                        mask
                        & alpha_live
                        & (log_p_s > float("-inf"))
                        & (log_p_s < float("inf"))
                    )
                    alpha_joint_live = (
                        alpha_student_live
                        & (log_p_t > float("-inf"))
                        & (log_p_t < float("inf"))
                    )
                    alpha_p = tl.where(alpha_student_live, p_s_kl, 0.0)
                    safe_alpha_joint_p = tl.where(alpha_joint_live, log_p_s, 0.0)
                    safe_alpha_joint_q = tl.where(alpha_joint_live, log_p_t, 0.0)
                    alpha_ratio_delta = (1.0 - kl_alpha) * (
                        safe_alpha_joint_q - safe_alpha_joint_p
                    )
                    alpha_escort = tl.where(
                        alpha_joint_live,
                        tl_highprec_exp(safe_alpha_joint_p + alpha_ratio_delta),
                        0.0,
                    )
                    alpha_numerator = alpha_affinity * alpha_p - alpha_escort
                    alpha_ratio_live = alpha_joint_live & (
                        tl.abs(alpha_ratio_delta) <= 1.0e-2
                    )
                    safe_alpha_ratio_delta = tl.where(
                        alpha_ratio_live, alpha_ratio_delta, 0.0
                    )
                    alpha_numerator = tl.where(
                        alpha_ratio_live,
                        alpha_p
                        * (
                            alpha_correction - tl_highprec_expm1(safe_alpha_ratio_delta)
                        ),
                        alpha_numerator,
                    )
                    alpha_delta = tl.where(
                        alpha_student_live,
                        alpha_numerator / (1.0 - kl_alpha),
                        0.0,
                    )
                    grad_s = student_grad + kl_grad_scale_s * alpha_delta
                if KL_STYLE == _KL_RENYI_C or KL_STYLE == _KL_RENYI_REVERSE_C:
                    renyi_student_live = (
                        mask
                        & renyi_valid
                        & (log_p_s > float("-inf"))
                        & (log_p_s < float("inf"))
                    )
                    renyi_p = tl.where(renyi_student_live, p_s_kl, 0.0)
                    renyi_joint_live = (
                        renyi_student_live
                        & (log_p_t > float("-inf"))
                        & (log_p_t < float("inf"))
                    )
                    safe_renyi_joint_p = tl.where(renyi_joint_live, log_p_s, 0.0)
                    safe_renyi_joint_q = tl.where(renyi_joint_live, log_p_t, 0.0)
                    if KL_STYLE == _KL_RENYI_REVERSE_C:
                        renyi_teacher_power = 1.0 - kl_alpha
                    else:
                        renyi_teacher_power = kl_alpha
                    safe_renyi_log_a = tl.where(renyi_valid, renyi_log_a, 0.0)
                    renyi_escort = tl.where(
                        renyi_joint_live,
                        tl_highprec_exp(
                            safe_renyi_joint_p
                            + renyi_teacher_power
                            * (safe_renyi_joint_q - safe_renyi_joint_p)
                            - safe_renyi_log_a
                        ),
                        0.0,
                    )
                    if KL_STYLE == _KL_RENYI_C:
                        renyi_delta = renyi_p - renyi_escort
                    else:
                        renyi_delta = (kl_alpha / (kl_alpha - 1.0)) * (
                            renyi_escort - renyi_p
                        )
                    grad_s = student_grad + kl_grad_scale_s * renyi_delta
                if KL_STYLE == _KL_DKD_C:
                    non_target = mask & (~is_label) & dkd_valid
                    p_hat_live = (
                        non_target
                        & s_nt_has_finite
                        & (x_s > float("-inf"))
                        & (x_s < float("inf"))
                    )
                    q_hat_live = (
                        non_target
                        & t_nt_has_finite
                        & (x_t > float("-inf"))
                        & (x_t < float("inf"))
                    )
                    safe_x_s_nt = tl.where(p_hat_live, x_s * T_inv_s, 0.0)
                    safe_x_t_nt = tl.where(q_hat_live, x_t * T_inv_t, 0.0)
                    safe_lse_s_nt = tl.where(p_hat_live, lse_s_nt, 0.0)
                    safe_lse_t_nt = tl.where(q_hat_live, lse_t_nt, 0.0)
                    p_hat = tl.where(
                        p_hat_live,
                        tl_highprec_exp(safe_x_s_nt - safe_lse_s_nt),
                        0.0,
                    )
                    q_hat = tl.where(
                        q_hat_live,
                        tl_highprec_exp(safe_x_t_nt - safe_lse_t_nt),
                        0.0,
                    )
                    target_grad = kl_alpha * (p_y_row - q_y_row)
                    nckd_grad = tl.where(t_nt_has_finite, p_hat - q_hat, 0.0)
                    non_target_grad = (
                        kl_alpha * (q_y_row - p_y_row) * p_hat + kl_beta * nckd_grad
                    )
                    dkd_delta = tl.where(is_label, target_grad, non_target_grad)
                    grad_s = student_grad + kl_grad_scale_s * tl.where(
                        mask & dkd_valid, dkd_delta, 0.0
                    )
                if KL_STYLE == _KL_ABKD_C:
                    both_dead = (log_p_s == float("-inf")) & (log_p_t == float("-inf"))
                    safe_log_p = tl.where(both_dead, 0.0, log_p_s)
                    safe_log_q = tl.where(both_dead, 0.0, log_p_t)
                    q_le_p = safe_log_q <= safe_log_p
                    low_delta = safe_log_q - safe_log_p
                    high_delta = safe_log_p - safe_log_q
                    p_power = tl_highprec_exp(ab_power * safe_log_p)
                    q_power = tl_highprec_exp(ab_power * safe_log_q)
                    h_low = -p_power * tl_highprec_expm1(kl_beta * low_delta)
                    h_high = (
                        q_power
                        * tl_highprec_exp(kl_alpha * high_delta)
                        * tl_highprec_expm1(kl_beta * high_delta)
                    )
                    abkd_col_live = mask & abkd_live & (~both_dead)
                    h_col = tl.where(
                        abkd_col_live,
                        tl.where(q_le_p, h_low, h_high),
                        0.0,
                    )
                    abkd_delta = tl.where(
                        abkd_col_live,
                        (h_col - p_s_kl * abkd_h_row) / kl_beta,
                        0.0,
                    )
                    grad_s = student_grad + kl_grad_scale_s * abkd_delta
                if KL_STYLE == _KL_TRIANGULAR_C:
                    sum_pq = p_s_kl + p_t_kl
                    tri_live = mask & (sum_pq > 0.0)
                    safe_sum_pq = tl.where(tri_live, sum_pq, 1.0)
                    diff_pq = tl.where(tri_live, p_s_kl - p_t_kl, 0.0)
                    tri_g = tl.where(
                        tri_live,
                        diff_pq * (p_s_kl + 3.0 * p_t_kl) / (safe_sum_pq * safe_sum_pq),
                        0.0,
                    )
                    safe_epg = tl.where(tri_live, epg_row, 0.0)
                    grad_s = student_grad + kl_grad_scale_s * tl.where(
                        tri_live,
                        p_s_kl * (tri_g - safe_epg),
                        0.0,
                    )
                if KL_STYLE == _KL_SKEW_FORWARD_C:
                    a = log_1m_beta + log_p_t
                    b = log_beta + log_p_s
                    log_m, mix_live = support_safe_logaddexp(a, b)
                    kl_row += tl.sum(zero_preserving_mul(p_t_kl, log_p_t - log_m, mask))
                    cross_term = tl.where(
                        mix_live,
                        tl_highprec_exp(log_beta + log_p_t + log_p_s - log_m),
                        0.0,
                    )
                    grad_s = student_grad + kl_grad_scale_s * tl.where(
                        mask,
                        p_s_kl * skew_row - cross_term,
                        0.0,
                    )
                if KL_STYLE == _KL_SKEW_JS_C:
                    a = log_1m_beta + log_p_t
                    b = log_beta + log_p_s
                    log_m, mix_live = support_safe_logaddexp(a, b)
                    kl_row += tl.sum(zero_preserving_mul(p_t_kl, log_p_t - log_m, mask))
                    grad_s = (
                        student_grad
                        + kl_grad_scale_s
                        * kl_beta
                        * zero_preserving_mul(
                            p_s_kl,
                            log_p_s - log_m - skew_row,
                            mask,
                        )
                    )
                if KL_STYLE == _KL_SKEW_REVERSE_C:
                    a = log_1m_beta + log_p_t
                    b = log_beta + log_p_s
                    log_m, mix_live = support_safe_logaddexp(a, b)
                    reverse_live = mask & mix_live & (p_s_kl > 0)
                    safe_log_ratio = tl.where(reverse_live, log_p_s - log_m, 0.0)
                    safe_center = tl.where(reverse_live, skew_row - skew_ratio, 0.0)
                    cross_log = tl.where(
                        reverse_live,
                        log_beta + 2.0 * log_p_s - log_m,
                        0.0,
                    )
                    cross_term = tl.where(
                        reverse_live,
                        tl_highprec_exp(cross_log),
                        0.0,
                    )
                    safe_p_s = tl.where(reverse_live, p_s_kl, 0.0)
                    skew_reverse_grad = (
                        safe_p_s * (safe_log_ratio - safe_center) - cross_term
                    )
                    grad_s = student_grad + kl_grad_scale_s * skew_reverse_grad
                if TEACHER_KL_GRAD:
                    teacher_grad = teacher_grad + kl_grad_scale_s * (p_t_kl - p_s_kl)
                    if KL_STYLE == _KL_COJEFFREYS_C:
                        teacher_grad = (
                            teacher_grad
                            + kl_grad_scale_s
                            * zero_preserving_mul(
                                p_t_kl,
                                log_p_t - log_p_s - fkl_row,
                                mask,
                            )
                        )

            # Gradient stores.
            if COMPUTE_TEACHER_CE or TEACHER_KL_GRAD:
                teacher_grad = tl.where(teacher_row_dead, 0.0, teacher_grad)
                teacher_grad = tl.where(
                    teacher_numeric_invalid,
                    float("nan"),
                    teacher_grad,
                )
                if SCALE_GRADIENT_MATERIALIZATION:
                    teacher_grad = teacher_grad * gradient_materialization_factor
                tl.store(row_t + cols, teacher_grad, mask=mask)

            if COMPUTE_KL:
                # Dead teacher row: drop the (NaN) KL contribution and keep
                # the student's plain CE gradient.
                grad_s = tl.where(teacher_row_dead, student_grad, grad_s)
                grad_s = tl.where(
                    teacher_numeric_invalid,
                    float("nan"),
                    grad_s,
                )
            grad_s = tl.where(student_numeric_active, float("nan"), grad_s)
            if SCALE_GRADIENT_MATERIALIZATION:
                grad_s = grad_s * gradient_materialization_factor
            tl.store(row_s + cols, grad_s, mask=mask)

        # ── KL epilogue and store ────────────────────────────
        if KL_STYLE == _KL_JEFFREYS_C or KL_STYLE == _KL_REVERSE_C:
            kl_row += rkl_row
        if KL_STYLE == _KL_ALPHA_DIVERGENCE_C:
            kl_row += alpha_loss_row
        if KL_STYLE == _KL_RENYI_C or KL_STYLE == _KL_RENYI_REVERSE_C:
            kl_row += renyi_loss_row
        if KL_STYLE == _KL_DKD_C:
            kl_row += dkd_loss_row
        if KL_STYLE == _KL_ABKD_C:
            kl_row += abkd_loss_row
        if KL_STYLE == _KL_ENTROPY_ADAPTIVE_C:
            finite_entropy = tl.where(
                gate_row > 0.0, gate_row * kl_row, 0.0
            ) + tl.where(
                gate_row < 1.0,
                (1.0 - gate_row) * rkl_grad_row,
                0.0,
            )
            kl_row = tl.where(entropy_invalid, float("inf"), finite_entropy)
        # Evaluate JSD as weighted KLs to the log-space mixture. This avoids
        # subtracting nearly equal entropy sums when the distributions are
        # close, preserving a small divergence more accurately.
        if KL_STYLE == _KL_SKEW_JS_C:
            kl_row = (1.0 - kl_beta) * kl_row + kl_beta * skew_row
        if KL_STYLE == _KL_SKEW_REVERSE_C:
            kl_row += skew_row
        if KL_STYLE == _KL_TRIANGULAR_C:
            kl_row += delta_row
        if COMPUTE_KL:
            kl_row = tl.where(teacher_row_dead, 0.0, kl_row)
            kl_row = tl.where(teacher_numeric_invalid, float("nan"), kl_row)
            kl_row = tl.where(student_numeric_active, float("nan"), kl_row)
        tl.store(K_ptr + i, kl_row * kl_loss_scale)

else:

    def ce_kl_fwdbwd_kernel(*args, **kwargs):
        raise RuntimeError(
            "Triton kernels are unavailable; install Triton and use CUDA/HIP tensors."
        )
