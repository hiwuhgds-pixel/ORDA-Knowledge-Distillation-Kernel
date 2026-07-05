from __future__ import annotations


def cosine_sim(torch, a, b) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


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
    ignore_index: int = -100,
    reduction: str = "mean",
):
    if teacher_mode not in {"tied", "separate", "precomputed"}:
        raise ValueError(f"Unsupported teacher_mode {teacher_mode!r}")

    F = torch.nn.functional
    hs_ref = student_hidden.detach().double().clone().requires_grad_(True)
    w_ref = weight.detach().double().clone().requires_grad_(True)
    labels_ref = labels.detach().to(device=hs_ref.device)
    logits_s = hs_ref @ w_ref.t()

    ht_ref = None
    wt_ref = None
    if teacher_mode == "tied":
        ht_ref = teacher_hidden.detach().double().clone().requires_grad_(True)
        logits_t = ht_ref @ w_ref.t()
        effective_teacher_ce_weight = 1.0 if teacher_ce_weight is None else float(teacher_ce_weight)
    elif teacher_mode == "separate":
        ht_ref = teacher_hidden.detach().double().clone()
        wt_ref = teacher_weight.detach().double().clone()
        effective_teacher_ce_weight = 0.0 if teacher_ce_weight is None else float(teacher_ce_weight)
        if effective_teacher_ce_weight != 0.0:
            ht_ref = ht_ref.requires_grad_(True)
            wt_ref = wt_ref.requires_grad_(True)
        logits_t = ht_ref @ wt_ref.t()
    else:
        effective_teacher_ce_weight = 0.0 if teacher_ce_weight is None else float(teacher_ce_weight)
        if teacher_logits is not None:
            logits_t = teacher_logits.detach().double().clone()
        else:
            logits_t = (
                teacher_hidden.detach().double().clone()
                @ teacher_weight.detach().double().clone().t()
            )

    ce_s_all = F.cross_entropy(
        logits_s,
        labels_ref,
        ignore_index=ignore_index,
        reduction="none",
    )
    ce_t_all = F.cross_entropy(
        logits_t,
        labels_ref,
        ignore_index=ignore_index,
        reduction="none",
    )

    mask = labels_ref != ignore_index
    denom = max(int(mask.sum().item()), 1)
    t = float(kl_temperature)
    log_p_s = F.log_softmax(logits_s / t, dim=-1)
    p_t = F.softmax(logits_t.detach() / t, dim=-1)
    kl_all = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * (t * t)
    kl_all = kl_all.masked_fill(~mask, 0.0)

    if reduction == "mean":
        student_ce = ce_s_all.sum() / denom
        teacher_ce_raw = ce_t_all.sum() / denom
        kl = kl_all.sum() / denom
    elif reduction == "sum":
        student_ce = ce_s_all.sum()
        teacher_ce_raw = ce_t_all.sum()
        kl = kl_all.sum()
    else:
        raise ValueError(f"Unsupported reduction {reduction!r}")

    teacher_ce = teacher_ce_raw if effective_teacher_ce_weight != 0.0 else student_ce.new_zeros(())
    loss = student_ce_weight * student_ce + effective_teacher_ce_weight * teacher_ce + kl_weight * kl
    return loss, student_ce, teacher_ce, kl, hs_ref, ht_ref, w_ref, wt_ref
