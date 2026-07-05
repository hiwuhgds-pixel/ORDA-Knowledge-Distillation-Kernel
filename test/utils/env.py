from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Runtime:
    torch: object
    orda: object


def import_torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required. Install the package environment before running tests."
        ) from exc


def import_orda():
    try:
        return importlib.import_module("orda_ce_kernel")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "orda_ce_kernel is not importable. Run tests from the project environment."
        ) from exc


def get_runtime() -> Runtime:
    return Runtime(torch=import_torch(), orda=import_orda())


def pytest_skip_if_no_cuda_kernel(pytest):
    try:
        runtime = get_runtime()
    except RuntimeError as exc:
        pytest.skip(str(exc), allow_module_level=True)
        raise
    torch = runtime.torch
    if not torch.cuda.is_available() or not runtime.orda.is_available():
        pytest.skip("CUDA/Triton kernels are not available", allow_module_level=True)
    return runtime


def parse_configs(value: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise argparse.ArgumentTypeError(f"Config must be BxS, got {item!r}")
        b_raw, s_raw = item.split("x", 1)
        try:
            batch = int(b_raw)
            seq = int(s_raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Config must be BxS, got {item!r}") from exc
        if batch <= 0 or seq <= 0:
            raise argparse.ArgumentTypeError(f"Config values must be positive, got {item!r}")
        configs.append((batch, seq))
    if not configs:
        raise argparse.ArgumentTypeError("At least one BxS config is required")
    return configs


def torch_dtype(torch, name: str):
    normalized = name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise argparse.ArgumentTypeError(f"Unsupported dtype {name!r}")


def validate_cuda_dtype(torch, dtype) -> None:
    if dtype is torch.bfloat16 and hasattr(torch.cuda, "is_bf16_supported"):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support bf16")


def set_seed(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
