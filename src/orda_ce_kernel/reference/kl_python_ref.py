"""KL distillation Python reference implementation.

Extracted from DistillCEFunction (ops/cross_entropy.py) to:
  - Use as a reference for original logic
  - Easily import for testing / comparing with the Triton kernel

Usage:
    from orda_ce_kernel.reference.kl_python_ref import kl_python_chunk
"""

import torch
import torch.nn.functional as F


def kl_python_chunk(
    logits_chunk: torch.Tensor,
    t_c: torch.Tensor,
    n_rows: int,
    kl_weight: float,
    kl_temperature: float,
    n_non_ignore: int,
    ignore_index: int,
    compute_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """KL(teacher || student) Python reference implementation.

    Args:
        logits_chunk: ``[2*n_rows, V]`` fp16/bf16. First half = student, second = teacher.
        t_c: ``[n_rows]`` target ids (int).
        n_rows: number of student rows (= n_rows in the chunk loop).
        kl_weight: KL loss coefficient (used to compute grad_scale).
        kl_temperature: softmax temperature T.
        n_non_ignore: number of non-ignored tokens (denominator of the entire batch).
        ignore_index: target value to ignore.
        compute_grad: whether to compute grad_kl_student.

    Returns:
        ``(kl_accum_delta, grad_kl_student)`` where:
        - ``kl_accum_delta``: scalar fp32, scaled by T² (ready to be added to kl_accum).
        - ``grad_kl_student``: ``[n_rows, V]`` fp32 if compute_grad, else None.
    """
    T_inv_h = 1.0 / kl_temperature

    # Compute softmax in compute_dtype (fp16) — do not upcast to fp32
    logits_s_kl = logits_chunk[:n_rows].mul(T_inv_h)
    logits_t_kl = logits_chunk[n_rows:].mul(T_inv_h)
    log_p_s = F.log_softmax(logits_s_kl, dim=-1)       # fp16
    p_t     = F.softmax(logits_t_kl, dim=-1)            # fp16
    del logits_s_kl, logits_t_kl

    # kl_div_row: [n_rows] — cast to fp32 only when summing
    kl_div_row = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1).float()
    if ignore_index is not None:
        kl_div_row = kl_div_row.masked_fill((t_c == ignore_index), 0.0)
    kl_accum_delta = kl_div_row.sum() * (kl_temperature * kl_temperature)
    del kl_div_row

    grad_kl_student = None
    if compute_grad:
        p_s = log_p_s.exp()
        # grad_kl_student: cast to fp32 to add accurately into logits_chunk
        grad_kl_student = (
            (kl_weight * kl_temperature / max(n_non_ignore, 1)) * (p_s - p_t)
        ).float()
        del p_s
        if ignore_index is not None:
            grad_kl_student = grad_kl_student.masked_fill(
                (t_c == ignore_index).unsqueeze(-1), 0.0
            )
    del log_p_s, p_t

    return kl_accum_delta, grad_kl_student
