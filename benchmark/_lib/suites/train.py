from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..core.device import release_memory, tensor_bytes
from ..engine.train import BuiltStep, TrainContext, TrainWorkload, optimizer_step
from ..losses.orda_loss import (
    orda_loss,
    precomputed_teacher,
    separate_teacher,
)
from ..losses.spec import LossSpec
from ..losses.torch_loss import torch_loss_terms
from ..model import ModelConfig, build_student, build_teacher_student


def orda_grad_scaler(scaler: torch.amp.GradScaler, dtype: torch.dtype):
    return scaler if dtype == torch.float16 and scaler.is_enabled() else None


# ── Model workload ───────────────────────────────────────────────────────────
@dataclass
class ModelTrainWorkload(TrainWorkload):
    teacher_config: ModelConfig
    student_config: ModelConfig
    spec: LossSpec
    teacher_requires_grad: bool
    precomputed: bool = False
    precomputed_source: str = "logits"
    precomputed_cache: dict[str, torch.Tensor] | None = None

    def build(self, ctx: TrainContext) -> BuiltStep:
        if self.precomputed:
            if self.precomputed_cache is None:
                raise ValueError("precomputed benchmark requires precomputed_cache.")
            if self.precomputed_source not in ("logits", "hidden"):
                raise ValueError(
                    f"Unknown precomputed_source: {self.precomputed_source}"
                )

        teacher_raw = None
        if self.precomputed:
            student_raw = build_student(
                ctx.vocab, self.student_config, seq=ctx.seq, device=ctx.device
            )
        else:
            teacher_raw, student_raw = build_teacher_student(
                ctx.vocab,
                self.teacher_config,
                self.student_config,
                seq=ctx.seq,
                device=ctx.device,
                freeze_teacher_model=not self.teacher_requires_grad,
            )
        if not ctx.use_amp:
            if teacher_raw is not None:
                teacher_raw.to(dtype=ctx.dtype)
            student_raw.to(dtype=ctx.dtype)

        parameters = list(student_raw.parameters())
        if teacher_raw is not None and self.teacher_requires_grad:
            parameters.extend(teacher_raw.parameters())
        optimizer = ctx.sgd(parameters)
        scaler = ctx.grad_scaler()

        if self.precomputed:
            tokens = self.precomputed_cache["x"].to(device=ctx.device)
            labels = self.precomputed_cache["labels"].to(device=ctx.device)
        else:
            tokens = torch.randint(
                0, ctx.vocab, (ctx.batch, ctx.seq), device=ctx.device
            )
            labels = torch.randint(
                0, ctx.vocab, (ctx.batch, ctx.seq), device=ctx.device
            )
        labels_flat = labels.view(-1)

        if self.precomputed:
            forward_loss, cache_bytes = self._precomputed_forward(
                ctx, student_raw, tokens, labels_flat, scaler
            )
            release_memory(ctx.device, collect_ipc=True)
        else:
            forward_loss, cache_bytes = self._live_forward(
                ctx, teacher_raw, student_raw, tokens, labels_flat, scaler
            )

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
            teacher_cache_bytes=cache_bytes,
        )

    # ── Precomputed teacher ──────────────────────────────────
    def _teacher_cache_tensors(self, ctx: TrainContext):
        cache = self.precomputed_cache
        if self.precomputed_source == "logits":
            logits_t = cache["logits"].to(device=ctx.device)
            return (logits_t,), tensor_bytes(logits_t)
        hidden_t = cache["hidden"].to(device=ctx.device)
        weight_t = cache["weight"].to(device=ctx.device)
        return (hidden_t, weight_t), tensor_bytes(hidden_t, weight_t)

    def _precomputed_forward(self, ctx, student_raw, tokens, labels_flat, scaler):
        teacher_tensors, cache_bytes = self._teacher_cache_tensors(ctx)
        student_dim = self.student_config.dim
        from_logits = self.precomputed_source == "logits"

        if ctx.mode == "torch-compile":

            def raw_loss(x_in, labels_in, *teacher):
                h_s = student_raw(x_in, return_hidden=True).view(-1, student_dim)
                logits_s = F.linear(h_s, student_raw.head.weight)
                logits_t = teacher[0] if from_logits else F.linear(*teacher)
                return torch_loss_terms(logits_s, logits_t, labels_in, self.spec)

            compiled = torch.compile(raw_loss, mode="default")
            return (
                lambda: compiled(tokens, labels_flat, *teacher_tensors),
                cache_bytes,
            )

        if ctx.mode != "orda":
            raise ValueError(f"Unknown mode: {ctx.mode}")

        student = torch.compile(student_raw, mode="default")
        head_weight = student_raw.head.weight
        teacher = precomputed_teacher(
            teacher_tensors[0],
            teacher_tensors[1] if len(teacher_tensors) > 1 else None,
            self.precomputed_source,
        )
        grad_scaler = orda_grad_scaler(scaler, ctx.dtype)

        def forward_loss():
            h_s = student(tokens, return_hidden=True).view(-1, student_dim)
            return orda_loss(
                h_s,
                head_weight,
                labels_flat,
                teacher,
                self.spec,
                profile="fast",
                grad_scaler=grad_scaler,
            )

        return forward_loss, cache_bytes

    # ── Live teacher ─────────────────────────────────────────
    def _live_forward(self, ctx, teacher_raw, student_raw, tokens, labels_flat, scaler):
        student_dim = self.student_config.dim
        teacher_dim = self.teacher_config.dim
        requires_grad = self.teacher_requires_grad

        def teacher_hidden(model, x_in):
            if requires_grad:
                return model(x_in, return_hidden=True).view(-1, teacher_dim)
            with torch.no_grad():
                return model(x_in, return_hidden=True).view(-1, teacher_dim)

        if ctx.mode == "torch-compile":

            def raw_loss(x_in, labels_in):
                h_t = teacher_hidden(teacher_raw, x_in)
                h_s = student_raw(x_in, return_hidden=True).view(-1, student_dim)
                logits_s = F.linear(h_s, student_raw.head.weight)
                logits_t = F.linear(h_t, teacher_raw.head.weight)
                return torch_loss_terms(logits_s, logits_t, labels_in, self.spec)

            compiled = torch.compile(raw_loss, mode="default")
            return (lambda: compiled(tokens, labels_flat)), 0

        if ctx.mode != "orda":
            raise ValueError(f"Unknown mode: {ctx.mode}")

        student = torch.compile(student_raw, mode="default")
        teacher = torch.compile(teacher_raw, mode="default")
        head_weight = student_raw.head.weight
        teacher_weight = teacher_raw.head.weight
        grad_scaler = orda_grad_scaler(scaler, ctx.dtype)

        def forward_loss():
            h_t = teacher_hidden(teacher, tokens)
            h_s = student(tokens, return_hidden=True).view(-1, student_dim)
            return orda_loss(
                h_s,
                head_weight,
                labels_flat,
                separate_teacher(h_t, teacher_weight),
                self.spec,
                profile="fast",
                grad_scaler=grad_scaler,
            )

        return forward_loss, 0
