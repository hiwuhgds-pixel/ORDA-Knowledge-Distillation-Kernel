from __future__ import annotations

import gc

import torch


# ── Device policy ────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def precision_config(device: torch.device) -> tuple[torch.dtype, bool, bool]:
    dtype = device_dtype(device)
    use_amp = device.type == "cuda"
    return dtype, use_amp, use_amp and dtype == torch.float16


def device_vram_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_properties(device).total_memory / 1024**3


def seed_all(device: torch.device, seed: int) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "-"


def require_cuda(device: torch.device, benchmark: str) -> None:
    if device.type != "cuda":
        raise RuntimeError(f"{benchmark} benchmark requires a CUDA device")


# ── CUDA accounting ──────────────────────────────────────────────────────────
def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_allocated(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return 0


def peak_extra_mb(device: torch.device, base_bytes: int) -> float:
    if device.type != "cuda":
        return 0.0
    return (torch.cuda.max_memory_allocated(device) - base_bytes) / 1024**2


# ── Memory cleanup ───────────────────────────────────────────────────────────
def release_memory(
    device: torch.device,
    *,
    reset_compiler: bool = False,
    collect_ipc: bool = False,
) -> None:
    if reset_compiler:
        compiler = getattr(torch, "compiler", None)
        if compiler is not None and hasattr(compiler, "reset"):
            compiler.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        if collect_ipc:
            torch.cuda.ipc_collect()


def full_cleanup(device: torch.device) -> None:
    release_memory(device, reset_compiler=True, collect_ipc=True)


def tensor_bytes(*tensors: torch.Tensor | None) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)
