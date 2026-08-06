from __future__ import annotations

import math


# ── Numerical primitives ─────────────────────────────────────────────────────
def zero_preserving_mul(torch, probability, log_ratio):
    supported = probability > 0
    safe_ratio = torch.where(supported, log_ratio, torch.zeros_like(log_ratio))
    return torch.where(
        supported,
        probability * safe_ratio,
        torch.zeros_like(probability),
    )


def renyi_log_affinity(torch, log_first, log_second, order: float):
    terms = order * log_first + (1.0 - order) * log_second
    first_dead = torch.isneginf(log_first)
    second_dead = torch.isneginf(log_second)
    if order < 1.0:
        terms = torch.where(
            first_dead | second_dead,
            torch.full_like(terms, float("-inf")),
            terms,
        )
    else:
        terms = torch.where(
            first_dead,
            torch.full_like(terms, float("-inf")),
            terms,
        )
        terms = torch.where(
            (~first_dead) & second_dead,
            torch.full_like(terms, float("inf")),
            terms,
        )
    return torch.logsumexp(terms, dim=-1)


def dkd_rows(torch, log_p_s, log_p_t, labels, mask, alpha: float, beta: float):
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
    gather_index = safe_labels.unsqueeze(-1)
    log_p_y = log_p_s.gather(-1, gather_index).squeeze(-1)
    log_q_y = log_p_t.gather(-1, gather_index).squeeze(-1)
    p_t = log_p_t.exp()
    q_y = log_q_y.exp()
    one_m_p_y = -torch.expm1(log_p_y)
    one_m_q_y = -torch.expm1(log_q_y)

    tckd = zero_preserving_mul(torch, q_y, log_q_y - log_p_y)
    tckd = tckd + zero_preserving_mul(
        torch,
        one_m_q_y,
        one_m_q_y.log() - one_m_p_y.log(),
    )

    classes = torch.arange(log_p_s.shape[-1], device=log_p_s.device)
    non_target = classes.unsqueeze(0) != safe_labels.unsqueeze(-1)
    safe_q_mass = torch.where(one_m_q_y > 0, one_m_q_y, torch.ones_like(one_m_q_y))
    safe_p_mass = torch.where(one_m_p_y > 0, one_m_p_y, torch.ones_like(one_m_p_y))
    q_hat = torch.where(
        non_target,
        p_t / safe_q_mass.unsqueeze(-1),
        torch.zeros_like(p_t),
    )
    log_q_hat = log_p_t - safe_q_mass.log().unsqueeze(-1)
    log_p_hat = log_p_s - safe_p_mass.log().unsqueeze(-1)
    nckd = zero_preserving_mul(torch, q_hat, log_q_hat - log_p_hat).sum(dim=-1)
    has_non_target = one_m_q_y > 0
    missing_support = has_non_target & (one_m_p_y <= 0)
    nckd = torch.where(
        has_non_target & (one_m_p_y > 0),
        nckd,
        torch.zeros_like(nckd),
    )
    row = alpha * tckd + beta * nckd
    return torch.where(missing_support, torch.full_like(row, float("inf")), row)


# ── Tokenwise divergence oracle ──────────────────────────────────────────────
def tokenwise_kl_rows(
    torch,
    logits_s,
    logits_t,
    labels,
    *,
    student_temperature: float,
    teacher_temperature: float | None = None,
    kl_divergence: str = "forward",
    kl_alpha: float | None = None,
    kl_beta: float | None = None,
    entropy_gate=None,
    ignore_index: int = -100,
):
    """Independent per-row oracle shared by tests and benchmark baselines."""

    F = torch.nn.functional
    student_t = float(student_temperature)
    teacher_t = student_t if teacher_temperature is None else float(teacher_temperature)
    scale = student_t * student_t
    alpha = (
        1.0
        if kl_divergence == "dkd" and kl_alpha is None
        else 0.5
        if kl_alpha is None
        else float(kl_alpha)
    )
    beta = (
        1.0
        if kl_divergence == "dkd" and kl_beta is None
        else 0.5
        if kl_beta is None
        else float(kl_beta)
    )
    mask = labels != ignore_index
    log_p_s = F.log_softmax(logits_s / student_t, dim=-1)

    if kl_divergence in ("mutual", "cojeffreys"):
        log_p_t = F.log_softmax(logits_t / teacher_t, dim=-1)
    else:
        log_p_t = F.log_softmax(logits_t.detach() / teacher_t, dim=-1)

    if kl_divergence == "mutual":
        target_t = log_p_t.detach()
        target_s = log_p_s.detach()
        rows = (
            zero_preserving_mul(torch, target_t.exp(), target_t - log_p_s)
            + zero_preserving_mul(torch, target_s.exp(), target_s - log_p_t)
        ).sum(dim=-1)
    elif kl_divergence == "cojeffreys":
        rows = (
            zero_preserving_mul(torch, log_p_t.exp(), log_p_t - log_p_s)
            + zero_preserving_mul(torch, log_p_s.exp(), log_p_s - log_p_t)
        ).sum(dim=-1)
    elif kl_divergence == "reverse":
        rows = zero_preserving_mul(torch, log_p_s.exp(), log_p_s - log_p_t).sum(dim=-1)
    elif kl_divergence == "jeffreys":
        rows = (
            zero_preserving_mul(torch, log_p_t.exp(), log_p_t - log_p_s)
            + zero_preserving_mul(torch, log_p_s.exp(), log_p_s - log_p_t)
        ).sum(dim=-1)
    elif kl_divergence == "alpha_divergence":
        log_affinity = torch.logsumexp(
            alpha * log_p_s + (1.0 - alpha) * log_p_t,
            dim=-1,
        )
        rows = -torch.expm1(log_affinity) / (alpha * (1.0 - alpha))
    elif kl_divergence in ("skew_forward", "skew_js", "skew_reverse"):
        log_m = torch.logaddexp(
            math.log1p(-beta) + log_p_t,
            math.log(beta) + log_p_s,
        )
        forward = zero_preserving_mul(torch, log_p_t.exp(), log_p_t - log_m).sum(dim=-1)
        reverse = zero_preserving_mul(torch, log_p_s.exp(), log_p_s - log_m).sum(dim=-1)
        if kl_divergence == "skew_forward":
            rows = forward
        elif kl_divergence == "skew_reverse":
            rows = reverse
        else:
            rows = (1.0 - beta) * forward + beta * reverse
    elif kl_divergence in ("renyi", "renyi_reverse"):
        base, other = (
            (log_p_t, log_p_s) if kl_divergence == "renyi" else (log_p_s, log_p_t)
        )
        base_live = torch.isfinite(base)
        safe_base = torch.where(base_live, base, torch.zeros_like(base))
        safe_delta = torch.where(base_live, other - base, torch.zeros_like(base))
        correction = torch.where(
            base_live,
            safe_base.exp() * torch.expm1((1.0 - alpha) * safe_delta),
            torch.zeros_like(base),
        )
        rows = torch.log1p(correction.sum(dim=-1)) / (alpha - 1.0)
    elif kl_divergence == "dkd":
        rows = dkd_rows(torch, log_p_s, log_p_t, labels, mask, alpha, beta)
    elif kl_divergence == "abkd":
        total = alpha + beta
        both_zero = torch.isneginf(log_p_s) & torch.isneginf(log_p_t)
        zero = torch.zeros_like(log_p_s)
        safe_p = torch.where(both_zero, zero, log_p_s)
        safe_q = torch.where(both_zero, zero, log_p_t)
        q_le_p = safe_q <= safe_p
        low_delta = safe_q - safe_p
        high_delta = safe_p - safe_q
        low = (
            torch.exp(total * safe_p)
            * (
                (beta / total) * torch.expm1(total * low_delta)
                - torch.expm1(beta * low_delta)
            )
            / (alpha * beta)
        )
        high = (
            torch.exp(total * safe_q)
            * (
                (alpha / total) * torch.expm1(total * high_delta)
                - torch.expm1(alpha * high_delta)
            )
            / (alpha * beta)
        )
        rows = torch.where(both_zero, zero, torch.where(q_le_p, low, high)).sum(dim=-1)
    elif kl_divergence == "triangular":
        p_s = log_p_s.exp()
        p_t = log_p_t.exp()
        denominator = p_s + p_t
        live = denominator > 0
        safe_denominator = torch.where(live, denominator, torch.ones_like(denominator))
        delta = torch.where(live, p_s - p_t, torch.zeros_like(p_s))
        rows = (delta.square() / safe_denominator).sum(dim=-1)
    elif kl_divergence == "entropy_adaptive":
        if entropy_gate is None:
            gate_lo, gate_hi, gate_tau, gate_scale = 0.0, 1.0, 0.5, 0.1
        else:
            gate_lo = float(entropy_gate.alpha_min)
            gate_hi = float(entropy_gate.alpha_max)
            gate_tau = float(entropy_gate.tau)
            gate_scale = float(entropy_gate.scale)
        p_t = log_p_t.exp()
        entropy = -zero_preserving_mul(torch, p_t, log_p_t).sum(dim=-1)
        normalized = (
            entropy / math.log(log_p_t.shape[-1])
            if log_p_t.shape[-1] > 1
            else torch.zeros_like(entropy)
        )
        gate = gate_lo + (gate_hi - gate_lo) * torch.sigmoid(
            (normalized - gate_tau) / gate_scale
        )
        forward = zero_preserving_mul(
            torch, gate.unsqueeze(-1) * p_t, log_p_t - log_p_s
        ).sum(dim=-1)
        reverse = zero_preserving_mul(
            torch,
            (1.0 - gate).unsqueeze(-1) * log_p_s.exp(),
            log_p_s - log_p_t,
        ).sum(dim=-1)
        rows = forward + reverse
    elif kl_divergence == "forward":
        rows = zero_preserving_mul(torch, log_p_t.exp(), log_p_t - log_p_s).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported kl_divergence {kl_divergence!r}")

    return rows * scale


def cosine_sim(torch, a, b) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0)
    ).item()


# ── Full loss oracle ─────────────────────────────────────────────────────────
def reference_distillation_loss(
    torch,
    student_hidden,
    weight,
    labels,
    *,
    teacher_mode: str,
    teacher_hidden=None,
    teacher_weight=None,
    teacher_logits=None,
    student_ce_weight: float = 1.0,
    teacher_ce_weight: float | None = None,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    kl_teacher_temperature: float | None = None,
    kl_divergence: str = "forward",
    kl_alpha: float | None = None,
    kl_beta: float | None = None,
    entropy_gate=None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    if teacher_mode not in {"tied", "separate", "precomputed"}:
        raise ValueError(f"Unsupported teacher_mode {teacher_mode!r}")

    F = torch.nn.functional
    hs_ref = student_hidden.detach().double().clone().requires_grad_(True)
    w_ref = weight.detach().double().clone().requires_grad_(True)
    labels_ref = labels.detach().to(device=hs_ref.device)
    compute_student_ce = float(student_ce_weight) != 0.0
    compute_teacher_ce = False
    compute_kl = float(kl_weight) != 0.0
    logits_s = hs_ref @ w_ref.t() if compute_student_ce or compute_kl else None

    ht_ref = None
    wt_ref = None
    if teacher_mode == "tied":
        ht_ref = teacher_hidden.detach().double().clone().requires_grad_(True)
        effective_teacher_ce_weight = (
            1.0 if teacher_ce_weight is None else float(teacher_ce_weight)
        )
        compute_teacher_ce = effective_teacher_ce_weight != 0.0
        logits_t = ht_ref @ w_ref.t() if compute_teacher_ce or compute_kl else None
    elif teacher_mode == "separate":
        ht_ref = teacher_hidden.detach().double().clone()
        wt_ref = teacher_weight.detach().double().clone()
        effective_teacher_ce_weight = (
            0.0 if teacher_ce_weight is None else float(teacher_ce_weight)
        )
        teacher_receives_kl = kl_weight != 0.0 and kl_divergence in (
            "mutual",
            "cojeffreys",
        )
        if effective_teacher_ce_weight != 0.0 or teacher_receives_kl:
            ht_ref = ht_ref.requires_grad_(True)
            wt_ref = wt_ref.requires_grad_(True)
        compute_teacher_ce = effective_teacher_ce_weight != 0.0
        logits_t = ht_ref @ wt_ref.t() if compute_teacher_ce or compute_kl else None
    else:
        effective_teacher_ce_weight = (
            0.0 if teacher_ce_weight is None else float(teacher_ce_weight)
        )
        compute_teacher_ce = effective_teacher_ce_weight != 0.0
        if not (compute_teacher_ce or compute_kl):
            logits_t = None
        elif teacher_logits is not None:
            logits_t = teacher_logits.detach().double().clone()
        else:
            logits_t = (
                teacher_hidden.detach().double().clone()
                @ teacher_weight.detach().double().clone().t()
            )

    ce_s_all = None
    if compute_student_ce:
        assert logits_s is not None
        ce_s_all = F.cross_entropy(
            logits_s,
            labels_ref,
            ignore_index=ignore_index,
            reduction="none",
        )
    ce_t_all = None
    if compute_teacher_ce:
        assert logits_t is not None
        ce_t_all = F.cross_entropy(
            logits_t,
            labels_ref,
            ignore_index=ignore_index,
            reduction="none",
        )

    mask = labels_ref != ignore_index
    denom = max(int(mask.sum().item()), 1)
    if compute_kl:
        assert logits_s is not None and logits_t is not None
        kl_all = tokenwise_kl_rows(
            torch,
            logits_s,
            logits_t,
            labels_ref,
            student_temperature=kl_temperature,
            teacher_temperature=kl_teacher_temperature,
            kl_divergence=kl_divergence,
            kl_alpha=kl_alpha,
            kl_beta=kl_beta,
            entropy_gate=entropy_gate,
            ignore_index=ignore_index,
        )
        kl_all = kl_all.masked_fill(~mask, 0.0)
    else:
        kl_all = None

    zero = hs_ref.new_zeros(())
    if reduction == "mean":
        student_ce = zero if ce_s_all is None else ce_s_all.sum() / denom
        teacher_ce = zero if ce_t_all is None else ce_t_all.sum() / denom
        kl = zero if kl_all is None else kl_all.sum() / denom
    elif reduction == "sum":
        student_ce = zero if ce_s_all is None else ce_s_all.sum()
        teacher_ce = zero if ce_t_all is None else ce_t_all.sum()
        kl = zero if kl_all is None else kl_all.sum()
    else:
        raise ValueError(f"Unsupported reduction {reduction!r}")

    loss = zero
    if compute_student_ce:
        loss = loss + float(student_ce_weight) * student_ce
    if compute_teacher_ce:
        loss = loss + effective_teacher_ce_weight * teacher_ce
    if compute_kl:
        loss = loss + float(kl_weight) * kl
    if not (compute_student_ce or compute_teacher_ce or compute_kl):
        loss = hs_ref[:0].sum() + w_ref[:0].sum()
    return loss, student_ce, teacher_ce, kl, hs_ref, ht_ref, w_ref, wt_ref
