from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..engine.train import BuiltStep, TrainContext, TrainWorkload, optimizer_step
from ..losses.aligned import aligned_components, build_cross_vocab_span_cache
from ..model import ModelConfig, build_student
from .train import orda_grad_scaler


@dataclass
class AlignedTrainWorkload(TrainWorkload):
    student_config: ModelConfig
    teacher_dim: int
    objective: str

    def build(self, ctx: TrainContext) -> BuiltStep:
        # ── Workload setup ───────────────────────────────────
        student_raw = build_student(
            ctx.vocab, self.student_config, seq=ctx.seq, device=ctx.device
        )
        if not ctx.use_amp:
            student_raw.to(dtype=ctx.dtype)
        optimizer = ctx.sgd(student_raw.parameters())
        scaler = ctx.grad_scaler()

        tokens = torch.randint(0, ctx.vocab, (ctx.batch, ctx.seq), device=ctx.device)
        labels = torch.randint(0, ctx.vocab, (ctx.batch * ctx.seq,), device=ctx.device)
        student_ranges, teacher_ranges, teacher = build_cross_vocab_span_cache(
            ctx.batch * ctx.seq,
            ctx.vocab,
            self.teacher_dim,
            dtype=ctx.dtype,
            device=ctx.device,
        )
        student_dim = self.student_config.dim
        objective = self.objective

        if ctx.mode == "torch-compile":
            # ── Torch backend ────────────────────────────────

            def raw_loss(x_in, labels_in, ranges_in, teacher_in):
                hidden = student_raw(x_in, return_hidden=True).view(-1, student_dim)
                logits = F.linear(hidden, student_raw.head.weight)
                return aligned_components(
                    logits, labels_in, ranges_in, teacher_in, objective
                )[0]

            compiled = torch.compile(raw_loss, mode="default")

            def forward_loss():
                return compiled(tokens, labels, student_ranges, teacher.span_log_probs)

        elif ctx.mode == "orda":
            # ── ORDA backend ─────────────────────────────────
            from orda_kernel import (
                AlignedLikelihoodConfig,
                AlignedSpanPairs,
                aligned_span_distillation_loss,
            )

            student = torch.compile(student_raw, mode="default")
            weight = student_raw.head.weight
            pairs = AlignedSpanPairs(student_ranges, teacher_ranges)
            config = AlignedLikelihoodConfig(objective=objective)
            grad_scaler = orda_grad_scaler(scaler, ctx.dtype)

            def forward_loss():
                hidden = student(tokens, return_hidden=True).view(-1, student_dim)
                return aligned_span_distillation_loss(
                    hidden,
                    weight,
                    labels,
                    teacher,
                    pairs,
                    config,
                    profile="fast",
                    backend="triton",
                    grad_scaler=grad_scaler,
                ).loss

        else:
            raise ValueError(f"Unknown mode: {ctx.mode}")

        return BuiltStep(
            step=optimizer_step(
                optimizer,
                scaler,
                forward_loss,
                dtype=ctx.dtype,
                device=ctx.device,
                use_amp=ctx.use_amp,
            ),
            optimizer=optimizer,
            scaler=scaler,
        )
