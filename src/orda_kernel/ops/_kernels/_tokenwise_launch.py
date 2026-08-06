from __future__ import annotations

import math
from dataclasses import dataclass

from ..._api_types import EntropyGate
from .ce_kl_fwdbwd import kl_style_of


@dataclass(frozen=True)
class TokenwiseKLLaunch:
    teacher_temperature: float
    both_t_one: bool
    kl_style: int
    renyi_near_one: bool
    gate_alpha_min: float
    gate_alpha_span: float
    gate_tau: float
    gate_inv_scale: float
    inv_log_vocab: float


def is_renyi_near_one(
    *,
    compute_kl: bool,
    kl_divergence: str,
    kl_alpha: float,
) -> bool:
    return (
        compute_kl
        and kl_divergence in ("renyi", "renyi_reverse")
        and abs(float(kl_alpha) - 1.0) <= 0.125
    )


def resolve_tokenwise_kl_launch(
    *,
    compute_kl: bool,
    kl_temperature: float,
    kl_teacher_temperature: float | None,
    kl_divergence: str,
    kl_alpha: float,
    vocab_size: int,
    entropy_gate: EntropyGate | None,
) -> TokenwiseKLLaunch:
    student_temperature = float(kl_temperature)
    teacher_temperature = float(
        kl_temperature if kl_teacher_temperature is None else kl_teacher_temperature
    )
    both_t_one = not compute_kl or (
        student_temperature == 1.0 and teacher_temperature == 1.0
    )
    if entropy_gate is None:
        gate_alpha_min = gate_alpha_span = gate_tau = gate_inv_scale = 0.0
    else:
        gate_alpha_min = float(entropy_gate.alpha_min)
        gate_alpha_span = float(entropy_gate.alpha_max) - gate_alpha_min
        gate_tau = float(entropy_gate.tau)
        gate_inv_scale = 1.0 / float(entropy_gate.scale)
    return TokenwiseKLLaunch(
        teacher_temperature=teacher_temperature,
        both_t_one=both_t_one,
        kl_style=kl_style_of(kl_divergence if compute_kl else "forward"),
        renyi_near_one=is_renyi_near_one(
            compute_kl=compute_kl,
            kl_divergence=kl_divergence,
            kl_alpha=kl_alpha,
        ),
        gate_alpha_min=gate_alpha_min,
        gate_alpha_span=gate_alpha_span,
        gate_tau=gate_tau,
        gate_inv_scale=gate_inv_scale,
        inv_log_vocab=(
            1.0 / math.log(vocab_size)
            if compute_kl and kl_divergence == "entropy_adaptive" and vocab_size > 1
            else 0.0
        ),
    )
