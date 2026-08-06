from __future__ import annotations

import gc

import torch


# ── OOM classification and recovery ──────────────────────────────────────────
class NonChunkableWorkspaceOOM(RuntimeError):
    """OOM from a workspace allocation whose size does not shrink by chunking."""


def is_oom_error(exc: BaseException) -> bool:
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    message = str(exc).lower()
    return (
        "cuda out of memory" in message
        or "cuda error: out of memory" in message
        or "hip out of memory" in message
        or "hip error: out of memory" in message
    )


def recover_after_oom(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
