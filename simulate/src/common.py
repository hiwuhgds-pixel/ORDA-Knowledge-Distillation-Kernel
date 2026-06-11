import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RMSNorm ───────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


# ── RoPE helpers ──────────────────────────────────────────────────────────────
def _precompute_rope(dh: int, seq: int):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
    t        = torch.arange(seq, dtype=torch.float32)
    freqs    = torch.outer(t, inv_freq)
    rope     = torch.cat([freqs, freqs], dim=-1)
    return rope.cos(), rope.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin, pos_ids=None):
    if pos_ids is None:
        T = q.shape[-2]
        c = cos[:T].unsqueeze(0).unsqueeze(0)
        s = sin[:T].unsqueeze(0).unsqueeze(0)
    else:
        c = F.embedding(pos_ids, cos).unsqueeze(1)
        s = F.embedding(pos_ids, sin).unsqueeze(1)
    return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s


# ── SwiGLU ────────────────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

