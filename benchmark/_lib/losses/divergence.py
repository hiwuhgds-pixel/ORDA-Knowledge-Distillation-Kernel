from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ── Divergence registry ──────────────────────────────────────────────────────
IGNORE_INDEX = -100

DIVERGENCES = (
    "forward",
    "reverse",
    "jeffreys",
    "alpha_divergence",
    "skew_forward",
    "skew_js",
    "skew_reverse",
    "renyi",
    "renyi_reverse",
    "dkd",
    "abkd",
    "triangular",
    "entropy_adaptive",
)

# Differentiate through the teacher too; only reachable from co-trained states.
COTRAINED_DIVERGENCES = ("mutual", "cojeffreys")

ALL_DIVERGENCES = (*DIVERGENCES, *COTRAINED_DIVERGENCES)


# ── Divergence evaluation ────────────────────────────────────────────────────
def kl_term(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    *,
    temperature: float,
    divergence: str,
    labels: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    early_row_scaling: bool = False,
) -> torch.Tensor:
    """Exclude ``ignore_index`` rows from the mean when labels are supplied."""
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    t2 = t * t
    log_p_s = F.log_softmax(logits_s / t, dim=-1)

    if labels is None:
        valid_rows = None
        valid_count = None
        safe_labels = None
    else:
        safe_labels = labels.reshape(-1)
        if safe_labels.numel() != logits_s.shape[0]:
            raise ValueError(
                f"labels contain {safe_labels.numel()} rows, "
                f"expected {logits_s.shape[0]}"
            )
        valid_rows = safe_labels != ignore_index
        valid_count = valid_rows.sum().clamp_min(1)

    def mean_rows(rows: torch.Tensor) -> torch.Tensor:
        if valid_rows is None:
            return rows.mean()
        return rows.masked_fill(~valid_rows, 0.0).sum() / valid_count

    def finish(rows: torch.Tensor) -> torch.Tensor:
        if early_row_scaling:
            return mean_rows(rows * t2)
        return mean_rows(rows) * t2

    if divergence == "mutual":
        log_p_t = F.log_softmax(logits_t / t, dim=-1)
        lpt_d = log_p_t.detach()
        lps_d = log_p_s.detach()
        kl_row = lpt_d.exp() * (lpt_d - log_p_s) + lps_d.exp() * (lps_d - log_p_t)
        return finish(kl_row.sum(dim=-1))

    if divergence == "cojeffreys":
        log_p_t = F.log_softmax(logits_t / t, dim=-1)
        kl_row = log_p_t.exp() * (log_p_t - log_p_s) + log_p_s.exp() * (
            log_p_s - log_p_t
        )
        return finish(kl_row.sum(dim=-1))

    log_p_t = F.log_softmax(logits_t.detach() / t, dim=-1)

    if divergence == "forward":
        return finish((log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1))

    if divergence == "reverse":
        return finish((log_p_s.exp() * (log_p_s - log_p_t)).sum(dim=-1))

    if divergence == "jeffreys":
        kl_row = log_p_t.exp() * (log_p_t - log_p_s) + log_p_s.exp() * (
            log_p_s - log_p_t
        )
        return finish(kl_row.sum(dim=-1))

    if divergence == "alpha_divergence":
        alpha = 0.5
        log_affinity = torch.logsumexp(
            alpha * log_p_s + (1.0 - alpha) * log_p_t,
            dim=-1,
        )
        # Alpha divergence is reduced before row scaling, so the result is
        # independent of the selected reduction order.
        return mean_rows(-torch.expm1(log_affinity) * t2 / (alpha * (1.0 - alpha)))

    if divergence in ("skew_forward", "skew_js", "skew_reverse"):
        beta = 0.5
        log_m = torch.logaddexp(
            math.log1p(-beta) + log_p_t,
            math.log(beta) + log_p_s,
        )
        forward = (log_p_t.exp() * (log_p_t - log_m)).sum(dim=-1)
        reverse = (log_p_s.exp() * (log_p_s - log_m)).sum(dim=-1)
        if divergence == "skew_forward":
            rows = forward
        elif divergence == "skew_reverse":
            rows = reverse
        else:
            rows = (1.0 - beta) * forward + beta * reverse
        return finish(rows)

    if divergence in ("renyi", "renyi_reverse"):
        order = 0.5
        base, other = (
            (log_p_t, log_p_s) if divergence == "renyi" else (log_p_s, log_p_t)
        )
        correction = base.exp() * torch.expm1((1.0 - order) * (other - base))
        return finish(torch.log1p(correction.sum(dim=-1)) / (order - 1.0))

    if divergence == "dkd":
        if labels is None:
            raise ValueError("labels are required for dkd")
        if logits_s.shape[-1] < 2:
            return logits_s.sum() * 0.0
        target_labels = torch.where(
            valid_rows, safe_labels, torch.zeros_like(safe_labels)
        )
        classes = torch.arange(log_p_s.shape[-1], device=log_p_s.device)
        non_target = classes.unsqueeze(0) != target_labels.unsqueeze(-1)
        scaled_s = logits_s / t
        scaled_t = logits_t.detach() / t
        target_s = scaled_s.gather(-1, target_labels.unsqueeze(-1)).squeeze(-1)
        target_t = scaled_t.gather(-1, target_labels.unsqueeze(-1)).squeeze(-1)
        non_target_s = scaled_s.masked_fill(~non_target, float("-inf"))
        non_target_t = scaled_t.masked_fill(~non_target, float("-inf"))
        binary_s = torch.stack(
            (target_s, torch.logsumexp(non_target_s, dim=-1)), dim=-1
        )
        binary_t = torch.stack(
            (target_t, torch.logsumexp(non_target_t, dim=-1)), dim=-1
        )
        log_binary_s = F.log_softmax(binary_s, dim=-1)
        log_binary_t = F.log_softmax(binary_t, dim=-1)
        tckd = (log_binary_t.exp() * (log_binary_t - log_binary_s)).sum(dim=-1)
        log_p_hat = F.log_softmax(non_target_s, dim=-1)
        log_q_hat = F.log_softmax(non_target_t, dim=-1)
        nckd = torch.where(
            non_target,
            log_q_hat.exp() * (log_q_hat - log_p_hat),
            torch.zeros_like(log_q_hat),
        ).sum(dim=-1)
        return finish(tckd + nckd)

    if divergence == "abkd":
        alpha = beta = 0.5
        total = alpha + beta
        q_le_p = log_p_t <= log_p_s
        low_delta = log_p_t - log_p_s
        high_delta = log_p_s - log_p_t
        low = (
            torch.exp(total * log_p_s)
            * (
                (beta / total) * torch.expm1(total * low_delta)
                - torch.expm1(beta * low_delta)
            )
            / (alpha * beta)
        )
        high = (
            torch.exp(total * log_p_t)
            * (
                (alpha / total) * torch.expm1(total * high_delta)
                - torch.expm1(alpha * high_delta)
            )
            / (alpha * beta)
        )
        return finish(torch.where(q_le_p, low, high).sum(dim=-1))

    if divergence == "triangular":
        p_s = log_p_s.exp()
        p_t = log_p_t.exp()
        denominator = p_s + p_t
        safe_denominator = torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
        return finish(((p_s - p_t).square() / safe_denominator).sum(dim=-1))

    if divergence == "entropy_adaptive":
        p_t = log_p_t.exp()
        entropy = -(p_t * log_p_t).sum(dim=-1)
        normalized = (
            entropy / math.log(log_p_t.shape[-1])
            if log_p_t.shape[-1] > 1
            else torch.zeros_like(entropy)
        )
        gate = torch.sigmoid((normalized - 0.5) / 0.1)
        if early_row_scaling:
            gate = gate.unsqueeze(-1)
            kl_all = gate * p_t * (log_p_t - log_p_s) + (1.0 - gate) * log_p_s.exp() * (
                log_p_s - log_p_t
            )
            return finish(kl_all.sum(dim=-1))
        forward = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
        reverse = (log_p_s.exp() * (log_p_s - log_p_t)).sum(dim=-1)
        return finish(gate * forward + (1.0 - gate) * reverse)

    raise ValueError(f"Unknown kl_divergence: {divergence}")
