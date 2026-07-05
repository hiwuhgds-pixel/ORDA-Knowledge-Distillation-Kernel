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


def _apply_rope(q, k, cos, sin):
    T = q.shape[-2]
    c = cos[:T].unsqueeze(0).unsqueeze(0)
    s = sin[:T].unsqueeze(0).unsqueeze(0)
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


# ── Attention (GQA + SDPA) ───────────────────────────────────────────────────
class Attention(nn.Module):

    def __init__(self, d: int, q_heads: int, kv_heads: int, seq: int):
        super().__init__()
        assert d % q_heads == 0
        assert q_heads % kv_heads == 0
        self.q_heads  = q_heads
        self.kv_heads = kv_heads
        self.dh       = d // q_heads
        self.kv_group = q_heads // kv_heads

        self.qkv_proj = nn.Linear(d, (q_heads + 2 * kv_heads) * self.dh, bias=False)
        self.out      = nn.Linear(d, d, bias=False)

        cos, sin = _precompute_rope(self.dh, seq)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.kv_group == 1:
            return x
        B, H, T, D = x.shape
        return x.unsqueeze(2).expand(B, H, self.kv_group, T, D).reshape(B, H * self.kv_group, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv    = self.qkv_proj(x)
        q_size = self.q_heads * self.dh
        kv_size = self.kv_heads * self.dh
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(B, T, self.q_heads,  self.dh).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, self.dh).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.dh).transpose(1, 2)
        q, k  = _apply_rope(q, k, self.cos, self.sin)
        k_rep = self._repeat_kv(k)
        v_rep = self._repeat_kv(v)
        out   = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(B, T, D))


# ── Block (Pre-norm) ──────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, d: int, q_heads: int, kv_heads: int, ff: int, seq: int):
        super().__init__()
        self.n1   = RMSNorm(d)
        self.n2   = RMSNorm(d)
        self.attn = Attention(d, q_heads, kv_heads, seq)
        self.ff   = SwiGLU(d, ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.ff(self.n2(x))


# ── Transformer ──────────────────────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(
        self,
        vocab: int,
        model_config: dict,
        seq: int = 1024,
    ):
        super().__init__()
        d        = model_config.get("dim", 1024)
        q_heads  = model_config.get("q_heads", 8)
        kv_heads = model_config.get("kv_heads", 2)
        layers   = model_config.get("n_layers", 16)
        ff       = model_config.get("ffn_dim", 2816)

        self.vocab = vocab
        self.seq   = seq

        self.tok    = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, q_heads, kv_heads, ff, seq) for _ in range(layers)])
        self.norm   = RMSNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

        nn.init.normal_(self.tok.weight, std=0.02)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        h = self.tok(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        if return_hidden:
            return h
        return self.head(h)
