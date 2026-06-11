from __future__ import annotations


def cosine_sim(torch, a, b) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def max_abs_diff(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a, b) -> float:
    return (a.float() - b.float()).abs().mean().item()


_VALID_TEACHER_MODES = ("tied", "separate", "precomputed")


def _resolve_teacher_loss_weight(mode: str, override):
    if override is not None:
        return float(override)
    return 1.0 if mode == "tied" else 0.0


def reference_distill_loss(
    torch,
    hs,
    ht,
    weight,
    target,
    *,
    lambda_student: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    detach_teacher_kl: bool = True,
    teacher_mode: str = "tied",
    weight_teacher=None,
    logits_teacher=None,
    teacher_loss_weight=None,
):
    """FP64 reference for DistillCEFunction supporting all 3 teacher modes.

    Modes:
      - 'tied': student and teacher project through the same `weight`.
      - 'separate': teacher uses `weight_teacher` (independent tensor).
      - 'precomputed': teacher logits are given as `logits_teacher [BT, V]`.

    `teacher_loss_weight` scales `loss_t` contribution to total loss:
      - None → 1.0 (tied) or 0.0 (separate/precomputed)
      - 0.0 → pure KD
      - > 0.0 → co-distillation / monitoring
    """
    if teacher_mode not in _VALID_TEACHER_MODES:
        raise ValueError(f"teacher_mode must be one of {_VALID_TEACHER_MODES}")

    F = torch.nn.functional
    hs_ref = hs.detach().double().clone().requires_grad_(True)
    w_ref = weight.detach().double().clone().requires_grad_(True)
    target_ref = target.detach().to(device=hs_ref.device)
    eff_tlw = _resolve_teacher_loss_weight(teacher_mode, teacher_loss_weight)

    logits_s = hs_ref @ w_ref.t()

    if teacher_mode == "tied":
        ht_ref = ht.detach().double().clone().requires_grad_(True)
        wt_ref = None
        logits_t = ht_ref @ w_ref.t()
    elif teacher_mode == "separate":
        ht_ref = ht.detach().double().clone().requires_grad_(True)
        wt_ref = weight_teacher.detach().double().clone()
        # Only require grad if teacher_loss_weight > 0 (matches kernel semantics).
        if eff_tlw > 0.0:
            wt_ref = wt_ref.requires_grad_(True)
        logits_t = ht_ref @ wt_ref.t()
    else:  # precomputed
        ht_ref = None
        wt_ref = None
        logits_t = logits_teacher.detach().double().clone()

    ce_s_all = F.cross_entropy(
        logits_s,
        target_ref,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    ce_t_all = F.cross_entropy(
        logits_t,
        target_ref,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )

    mask = target_ref != ignore_index
    denom = max(int(mask.sum().item()), 1)

    if kl_weight > 0.0:
        T = float(kl_temperature)
        log_p_s = F.log_softmax(logits_s / T, dim=-1)
        teacher_logits = logits_t.detach() if detach_teacher_kl else logits_t
        p_t = F.softmax(teacher_logits / T, dim=-1)
        kl_all = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * (T * T)
        kl_all = kl_all.masked_fill(~mask, 0.0)
    else:
        kl_all = logits_s.new_zeros(logits_s.shape[0])

    if reduction == "mean":
        loss_s = ce_s_all.sum() / denom
        loss_t = ce_t_all.sum() / denom
        kl_loss = kl_all.sum() / denom
    elif reduction == "sum":
        loss_s = ce_s_all.sum()
        loss_t = ce_t_all.sum()
        kl_loss = kl_all.sum()
    else:
        raise ValueError(f"Unsupported reduction {reduction!r}")

    loss = lambda_student * loss_s + eff_tlw * loss_t + kl_weight * kl_loss

    return loss, loss_s, loss_t, kl_loss, hs_ref, ht_ref, w_ref, wt_ref


