from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config.profiles import VramProfile
from ..engine.vram import VramWorkload, backward_inputs
from ..losses.aligned import aligned_inputs, make_aligned_loss_fn, run_aligned_loss_only
from ..losses.orda_loss import (
    precomputed_orda_loss_fn,
    separate_orda_loss_fn,
)
from ..losses.spec import LossSpec
from ..losses.torch_loss import (
    precomputed_torch_loss_fn,
    separate_torch_loss_fn,
)
from .specs import precomputed_mode_parts

TORCH_MODES = ("torch", "torch-compile")


@dataclass
class _Shape:
    vocab: int
    seq: int
    dim: int
    teacher_dim: int
    batch: int
    dtype: torch.dtype
    device: torch.device

    @property
    def rows(self) -> int:
        return self.batch * self.seq


# ── Separate teacher ─────────────────────────────────────────────────────────
@dataclass
class SeparateVramWorkload(VramWorkload):
    shape: _Shape
    spec: LossSpec
    mode: str
    num_chunks: int
    teacher_requires_grad: bool = False
    cache_indices: tuple[int, ...] = ()

    def loss_fn(self):
        if self.mode in TORCH_MODES:
            return separate_torch_loss_fn(
                self.spec, compile=self.mode == "torch-compile"
            )
        return separate_orda_loss_fn(self.spec, num_chunks=self.num_chunks)

    def inputs(self):
        s = self.shape
        h_s = torch.randn(s.rows, s.dim, device=s.device, dtype=s.dtype)
        h_t = torch.randn(s.rows, s.teacher_dim, device=s.device, dtype=s.dtype)
        student_weight = torch.randn(
            s.vocab, s.dim, device=s.device, dtype=s.dtype, requires_grad=True
        )
        teacher_weight = torch.randn(
            s.vocab,
            s.teacher_dim,
            device=s.device,
            dtype=s.dtype,
            requires_grad=self.teacher_requires_grad,
        )
        labels = torch.randint(0, s.vocab, (s.rows,), device=s.device)
        return h_s, h_t, student_weight, teacher_weight, labels

    def run(self, loss_fn, inputs):
        backward_inputs(
            loss_fn,
            inputs,
            differentiable=((0, True), (1, self.teacher_requires_grad)),
            clear_grads=(2, 3),
            device=self.shape.device,
        )


# ── Precomputed teacher ──────────────────────────────────────────────────────
@dataclass
class PrecomputedVramWorkload(VramWorkload):
    shape: _Shape
    spec: LossSpec
    mode: str
    num_chunks: int
    cache_kind: str = "standard"
    cache_indices: tuple[int, ...] = (2, 3)

    @property
    def _parts(self) -> tuple[str, str]:
        return precomputed_mode_parts(self.mode)

    def loss_fn(self):
        backend, source = self._parts
        if backend in TORCH_MODES:
            return precomputed_torch_loss_fn(
                self.spec, source, compile=backend == "torch-compile"
            )
        return precomputed_orda_loss_fn(self.spec, source, num_chunks=self.num_chunks)

    def inputs(self):
        s = self.shape
        _, source = self._parts
        h_s = torch.randn(s.rows, s.dim, device=s.device, dtype=s.dtype)
        student_weight = torch.randn(
            s.vocab, s.dim, device=s.device, dtype=s.dtype, requires_grad=True
        )
        teacher_cache, teacher_weight = self._teacher_cache(source)
        labels = torch.randint(0, s.vocab, (s.rows,), device=s.device)
        return h_s, student_weight, teacher_cache, teacher_weight, labels

    def _teacher_cache(self, source: str):
        from ..losses.precomputed import (
            precomputed_cross_vocab_logits,
            precomputed_multi_teacher_logits,
        )

        s = self.shape
        if source == "logits":
            if self.cache_kind == "cross_vocab":
                cache = precomputed_cross_vocab_logits(
                    s.rows, s.vocab, dtype=s.dtype, device=s.device
                )
            elif self.cache_kind in (
                "multi_teacher_arithmetic",
                "multi_teacher_geometric",
            ):
                cache = precomputed_multi_teacher_logits(
                    s.rows,
                    s.vocab,
                    mode=self.cache_kind.removeprefix("multi_teacher_"),
                    dtype=s.dtype,
                    device=s.device,
                )
            elif self.cache_kind == "standard":
                cache = torch.randn(s.rows, s.vocab, device=s.device, dtype=s.dtype)
            else:
                raise ValueError(f"Unknown precomputed cache kind: {self.cache_kind}")
            return cache, None
        if source != "hidden":
            raise ValueError(f"Unknown precomputed source: {source}")
        if self.cache_kind != "standard":
            raise ValueError(f"{self.cache_kind} requires precomputed_source='logits'")
        cache = torch.randn(s.rows, s.teacher_dim, device=s.device, dtype=s.dtype)
        weight = torch.randn(s.vocab, s.teacher_dim, device=s.device, dtype=s.dtype)
        return cache, weight

    def run(self, loss_fn, inputs):
        backward_inputs(
            loss_fn,
            inputs,
            differentiable=((0, True),),
            clear_grads=(1,),
            device=self.shape.device,
        )


# ── Aligned span ─────────────────────────────────────────────────────────────
@dataclass
class AlignedVramWorkload(VramWorkload):
    shape: _Shape
    objective: str
    mode: str
    num_chunks: int
    cache_indices: tuple[int, ...] = ()

    def loss_fn(self):
        return make_aligned_loss_fn(
            self.objective,
            self.mode,
            num_chunks=self.num_chunks,
            autotune=False,
        )

    def inputs(self):
        s = self.shape
        return aligned_inputs(s.rows, s.vocab, s.dim, dtype=s.dtype, device=s.device)

    def run(self, loss_fn, inputs):
        run_aligned_loss_only(
            loss_fn,
            *inputs,
            dtype=self.shape.dtype,
            device=self.shape.device,
            use_amp=False,
        )


def shape_for(
    profile: VramProfile,
    *,
    vocab: int,
    seq: int,
    dim: int,
    teacher_dim: int | None,
    dtype: torch.dtype,
    device: torch.device,
) -> _Shape:
    return _Shape(
        vocab=vocab,
        seq=seq,
        dim=dim,
        teacher_dim=dim if teacher_dim is None else teacher_dim,
        batch=profile.batch,
        dtype=dtype,
        device=device,
    )
