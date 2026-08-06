from __future__ import annotations

import torch

from .aligned import cross_vocab_teacher_vocab_size


def precomputed_cross_vocab_logits(
    n_rows: int,
    student_vocab_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    from orda_kernel import project_teacher_logits

    teacher_vocab_size = cross_vocab_teacher_vocab_size(student_vocab_size)
    # -1 marks a teacher token with no student counterpart.
    token_map = torch.arange(
        -1,
        student_vocab_size,
        dtype=torch.int64,
        device=device,
    )
    teacher_logits = torch.randn(
        n_rows,
        teacher_vocab_size,
        dtype=dtype,
        device=device,
    )
    return project_teacher_logits(
        teacher_logits,
        token_map,
        student_vocab_size,
    )


def precomputed_multi_teacher_logits(
    n_rows: int,
    vocab_size: int,
    *,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    from orda_kernel import mix_teacher_logits

    teachers = [
        torch.randn(n_rows, vocab_size, dtype=dtype, device=device),
        torch.randn(n_rows, vocab_size, dtype=dtype, device=device),
    ]
    return mix_teacher_logits(teachers, (0.4, 0.6), mode=mode)
