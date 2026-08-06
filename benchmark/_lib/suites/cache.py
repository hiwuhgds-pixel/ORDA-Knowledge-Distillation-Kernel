"""Build the teacher cache once so every backend shares the same unmeasured inputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.device import release_memory
from ..losses.precomputed import (
    precomputed_cross_vocab_logits,
    precomputed_multi_teacher_logits,
)
from ..model import ModelConfig, Transformer, freeze_teacher

CACHE_KINDS = (
    "standard",
    "cross_vocab",
    "multi_teacher_arithmetic",
    "multi_teacher_geometric",
)


def build_precomputed_cache(
    *,
    vocab: int,
    seq: int,
    teacher_config: ModelConfig,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
    precomputed_source: str = "all",
    cache_kind: str = "standard",
) -> dict[str, torch.Tensor]:
    if precomputed_source not in ("all", "logits", "hidden_weight"):
        raise ValueError(f"Unknown precomputed_source: {precomputed_source}")
    if cache_kind not in CACHE_KINDS:
        raise ValueError(f"Unknown precomputed cache kind: {cache_kind}")

    x = torch.randint(0, vocab, (batch, seq), device=device)
    labels = torch.randint(0, vocab, (batch, seq), device=device)

    if cache_kind != "standard":
        if precomputed_source != "logits":
            raise ValueError(f"{cache_kind} requires precomputed_source='logits'")
        n_rows = batch * seq
        if cache_kind == "cross_vocab":
            logits = precomputed_cross_vocab_logits(
                n_rows, vocab, dtype=dtype, device=device
            )
        else:
            logits = precomputed_multi_teacher_logits(
                n_rows,
                vocab,
                mode=cache_kind.removeprefix("multi_teacher_"),
                dtype=dtype,
                device=device,
            )
        cache = {"x": x.cpu(), "labels": labels.cpu(), "logits": logits.cpu()}
        del x, labels, logits
        release_memory(device, collect_ipc=True)
        return cache

    teacher = Transformer(vocab, teacher_config, seq=seq).to(
        dtype=torch.float32, device=device
    )
    freeze_teacher(teacher)
    if not use_amp:
        teacher.to(dtype=dtype)

    h_t = logits_t = teacher_weight = teacher_weight_cache = None
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        with torch.no_grad():
            h_t = (
                teacher(x, return_hidden=True)
                .view(-1, teacher_config.dim)
                .detach()
                .to(dtype)
            )
            teacher_weight = teacher.head.weight.detach().to(dtype)
            if precomputed_source in ("all", "logits"):
                logits_t = F.linear(h_t, teacher_weight).detach()
            if precomputed_source in ("all", "hidden_weight"):
                teacher_weight_cache = teacher_weight

    cache = {"x": x.cpu(), "labels": labels.cpu()}
    if precomputed_source in ("all", "logits"):
        cache["logits"] = logits_t.cpu()
    if precomputed_source in ("all", "hidden_weight"):
        cache["hidden"] = h_t.cpu()
        cache["weight"] = teacher_weight_cache.cpu()

    del x, labels, h_t, logits_t, teacher_weight, teacher_weight_cache, teacher
    release_memory(device, collect_ipc=True)
    return cache
