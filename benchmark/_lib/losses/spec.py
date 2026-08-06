from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .divergence import IGNORE_INDEX, kl_term


# ── Loss specification ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class LossSpec:
    temperature: float
    kl_divergence: str = "forward"
    student_ce_weight: float = 1.0
    teacher_ce_weight: float = 0.0
    kl_weight: float = 1.0
    tied_mode: str | None = None

    @property
    def needs_teacher_logits(self) -> bool:
        return self.kl_weight != 0.0 or self.teacher_ce_weight != 0.0


# ── Torch loss terms ─────────────────────────────────────────────────────────
def torch_loss_terms(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels: torch.Tensor,
    spec: LossSpec,
) -> torch.Tensor:
    terms = []
    if spec.student_ce_weight != 0.0:
        terms.append(
            spec.student_ce_weight
            * F.cross_entropy(logits_s, labels, ignore_index=IGNORE_INDEX)
        )
    if spec.teacher_ce_weight != 0.0:
        terms.append(
            spec.teacher_ce_weight
            * F.cross_entropy(logits_t, labels, ignore_index=IGNORE_INDEX)
        )
    if spec.kl_weight != 0.0:
        terms.append(
            spec.kl_weight
            * kl_term(
                logits_s,
                logits_t,
                temperature=spec.temperature,
                divergence=spec.kl_divergence,
                labels=labels,
            )
        )
    if not terms:
        raise ValueError("loss spec has no active component")
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total
