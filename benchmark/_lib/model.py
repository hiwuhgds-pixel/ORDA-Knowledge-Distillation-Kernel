from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.validation import validate_positive_int


def _sdpa_supports_gqa() -> bool:
    try:
        query = torch.zeros(1, 2, 1, 2)
        key_value = torch.zeros(1, 1, 1, 2)
        F.scaled_dot_product_attention(
            query,
            key_value,
            key_value,
            enable_gqa=True,
        )
        return True
    except (TypeError, RuntimeError):
        return False


_SDPA_SUPPORTS_GQA = _sdpa_supports_gqa()


# ── Model configuration ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    dim: int = 1024
    q_heads: int = 8
    kv_heads: int = 2
    n_layers: int = 16
    ffn_dim: int = 2816

    @classmethod
    def from_mapping(cls, values: Any) -> ModelConfig:
        if isinstance(values, ModelConfig):
            return values
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"unknown model config keys: {sorted(unknown)}; "
                f"Transformer reads only {sorted(known)}"
            )
        return cls(**values)

    def replace(self, **changes: Any) -> ModelConfig:
        return ModelConfig.from_mapping({**self.as_dict(), **changes})

    def as_dict(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def __str__(self) -> str:
        return str(self.as_dict())


# ── RMSNorm module ───────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


# ── RoPE helpers ─────────────────────────────────────────────────────────────
def _precompute_rope(dh: int, seq: int):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
    t = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    rope = torch.cat([freqs, freqs], dim=-1)
    return rope.cos(), rope.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dh: int, seq: int):
        super().__init__()
        cos, sin = _precompute_rope(dh, seq)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("_cos_fp16", cos.to(torch.float16), persistent=False)
        self.register_buffer("_sin_fp16", sin.to(torch.float16), persistent=False)
        self.register_buffer("_cos_bf16", cos.to(torch.bfloat16), persistent=False)
        self.register_buffer("_sin_bf16", sin.to(torch.bfloat16), persistent=False)

    def _apply(self, fn):
        super()._apply(fn)
        self.cos = self.cos.float()
        self.sin = self.sin.float()
        self._cos_fp16 = self.cos.to(torch.float16)
        self._sin_fp16 = self.sin.to(torch.float16)
        self._cos_bf16 = self.cos.to(torch.bfloat16)
        self._sin_bf16 = self.sin.to(torch.bfloat16)
        return self

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        length = q.shape[-2]
        if length > self.cos.shape[0]:
            raise ValueError(
                f"input sequence length {length} exceeds configured maximum "
                f"{self.cos.shape[0]}"
            )
        if q.dtype == torch.float16:
            cos, sin = self._cos_fp16, self._sin_fp16
        elif q.dtype == torch.bfloat16:
            cos, sin = self._cos_bf16, self._sin_bf16
        elif q.dtype == torch.float32:
            cos, sin = self.cos, self.sin
        else:
            cos, sin = self.cos.to(q.dtype), self.sin.to(q.dtype)
        cos = cos[:length].unsqueeze(0).unsqueeze(0)
        sin = sin[:length].unsqueeze(0).unsqueeze(0)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ── SwiGLU module ────────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ── Attention (GQA + SDPA) ───────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, d: int, q_heads: int, kv_heads: int, seq: int):
        super().__init__()
        validate_positive_int("dim", d)
        validate_positive_int("q_heads", q_heads)
        validate_positive_int("kv_heads", kv_heads)
        validate_positive_int("seq", seq)
        if d % q_heads != 0:
            raise ValueError(f"dim={d} must be divisible by q_heads={q_heads}")
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}"
            )
        if (d // q_heads) % 2 != 0:
            raise ValueError(
                f"head dimension must be even for RoPE, got dim/q_heads={d // q_heads}"
            )
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.dh = d // q_heads
        self.kv_group = q_heads // kv_heads

        self.qkv_proj = nn.Linear(d, (q_heads + 2 * kv_heads) * self.dh, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.kv_group == 1:
            return x
        B, H, T, D = x.shape
        return (
            x.unsqueeze(2)
            .expand(B, H, self.kv_group, T, D)
            .reshape(B, H * self.kv_group, T, D)
        )

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        q_size = self.q_heads * self.dh
        kv_size = self.kv_heads * self.dh
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(B, T, self.q_heads, self.dh).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, self.dh).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.dh).transpose(1, 2)
        q, k = rope(q, k)
        if self.kv_group != 1 and _SDPA_SUPPORTS_GQA:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
        else:
            out = F.scaled_dot_product_attention(
                q,
                self._repeat_kv(k),
                self._repeat_kv(v),
                is_causal=True,
            )
        return self.out(out.transpose(1, 2).reshape(B, T, D))


# ── Block (Pre-norm) ─────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, d: int, q_heads: int, kv_heads: int, ff: int, seq: int):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.attn = Attention(d, q_heads, kv_heads, seq)
        self.ff = SwiGLU(d, ff)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        x = x + self.attn(self.n1(x), rope)
        return x + self.ff(self.n2(x))


# ── Transformer module ───────────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(
        self,
        vocab: int,
        model_config: ModelConfig | dict,
        seq: int = 1024,
    ):
        super().__init__()
        validate_positive_int("vocab", vocab)
        validate_positive_int("seq", seq)
        config = ModelConfig.from_mapping(model_config)

        self.vocab = vocab
        self.seq = seq
        self.config = config

        self.tok = nn.Embedding(vocab, config.dim)
        self.rope = RotaryEmbedding(config.dim // config.q_heads, seq)
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.dim,
                    config.q_heads,
                    config.kv_heads,
                    config.ffn_dim,
                    seq,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, vocab, bias=False)
        self.head.weight = self.tok.weight

        nn.init.normal_(self.tok.weight, std=0.02)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"input tokens must have shape [batch, seq], got {tuple(x.shape)}"
            )
        if x.shape[1] > self.seq:
            raise ValueError(
                f"input sequence length {x.shape[1]} exceeds configured maximum {self.seq}"
            )
        h = self.tok(x)
        for block in self.blocks:
            h = block(h, self.rope)
        h = self.norm(h)
        if return_hidden:
            return h
        return self.head(h)


# ── Model builders ───────────────────────────────────────────────────────────
def freeze_teacher(model: nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def build_student(
    vocab: int,
    student_config: ModelConfig | dict,
    *,
    seq: int,
    device: torch.device,
) -> Transformer:
    return Transformer(vocab, student_config, seq=seq).to(
        dtype=torch.float32, device=device
    )


def build_teacher_student(
    vocab: int,
    teacher_config: ModelConfig | dict,
    student_config: ModelConfig | dict,
    *,
    seq: int,
    device: torch.device,
    freeze_teacher_model: bool = True,
) -> tuple[Transformer, Transformer]:
    teacher = build_student(vocab, teacher_config, seq=seq, device=device)
    student = build_student(vocab, student_config, seq=seq, device=device)
    if freeze_teacher_model:
        freeze_teacher(teacher)
    return teacher, student
