from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol

import torch

from .models import DeviceFingerprint


class _DeviceProperties(Protocol):
    name: str
    major: int
    minor: int
    multi_processor_count: int
    total_memory: int


def physical_device_key_from_properties(
    index: int,
    backend: str,
    properties: _DeviceProperties,
) -> str:
    """Return the most stable physical GPU identity exposed by PyTorch."""
    raw_uuid = getattr(properties, "uuid", None)
    if raw_uuid is not None:
        physical_uuid = str(raw_uuid)
        if physical_uuid:
            return f"{backend}:uuid:{physical_uuid}"

    raw_pci_bus_id = getattr(properties, "pci_bus_id", None)
    if raw_pci_bus_id is not None:
        pci_bus_id = str(raw_pci_bus_id)
        if pci_bus_id:
            return f"{backend}:pci:{pci_bus_id}"

    return (
        f"{backend}:legacy:{index}:{properties.name}:{properties.major}:"
        f"{properties.minor}:{properties.multi_processor_count}:"
        f"{properties.total_memory}"
    )


# ── Ambient toolchain identity ───────────────────────────────────────────────
def _backend_name() -> str:
    if torch.version.hip:
        return "hip"
    if torch.version.cuda:
        return "cuda"
    return "unknown"


def _runtime_version() -> str | None:
    if torch.version.hip:
        return str(torch.version.hip)
    if torch.version.cuda:
        return str(torch.version.cuda)
    return None


# Probe common CUDA and ROCm Triton distribution names.
_TRITON_DISTRIBUTIONS = (
    "triton",
    "pytorch-triton-rocm",
    "triton-rocm",
    "pytorch-triton",
)


@lru_cache(maxsize=1)
def _triton_version() -> str | None:
    for distribution in _TRITON_DISTRIBUTIONS:
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return None


def _is_t4_properties(properties: _DeviceProperties) -> bool:
    return re.search(r"(?<![a-z0-9])t4(?![a-z0-9])", properties.name, re.I) is not None


def _max_num_warps_from_properties(
    backend: str,
    properties: _DeviceProperties,
) -> int:
    warp_size = int(getattr(properties, "warp_size", 64 if backend == "hip" else 32))
    max_threads = int(getattr(properties, "max_threads_per_block", 1024))
    return max(1, min(32, max_threads // max(1, warp_size)))


# ── Unified device identity ──────────────────────────────────────────────────
@dataclass(frozen=True)
class DeviceIdentity:
    """Everything a caller needs about one CUDA device from a single props read."""

    fingerprint: DeviceFingerprint
    physical_key: str
    is_t4: bool
    max_num_warps: int


@lru_cache(maxsize=32)
def _cuda_device_identity(
    index: int,
    backend: str,
    runtime_version: str | None,
    torch_version: str,
    triton_version: str | None,
) -> DeviceIdentity:
    """Read device properties once and derive every identity a caller needs."""
    props = torch.cuda.get_device_properties(index)
    physical_key = physical_device_key_from_properties(index, backend, props)
    fingerprint = DeviceFingerprint(
        device_type="cuda",
        # UUID/PCI identity is stable when CUDA_VISIBLE_DEVICES remaps indices.
        device_index=None,
        backend=backend,
        runtime_version=runtime_version,
        torch_version=torch_version,
        triton_version=triton_version,
        name=props.name,
        major=props.major,
        minor=props.minor,
        multi_processor_count=props.multi_processor_count,
        total_memory=props.total_memory,
        physical_id=physical_key,
    )
    return DeviceIdentity(
        fingerprint=fingerprint,
        physical_key=physical_key,
        is_t4=_is_t4_properties(props),
        max_num_warps=_max_num_warps_from_properties(backend, props),
    )


def device_identity(index: int) -> DeviceIdentity:
    """Return the unified identity for a CUDA device index (single cache)."""
    return _cuda_device_identity(
        int(index),
        _backend_name(),
        _runtime_version(),
        str(torch.__version__),
        _triton_version(),
    )


def reset_device_identity_cache() -> None:
    """Clear every device-identity cache in one place after a reset."""
    _cuda_device_identity.cache_clear()
    _triton_version.cache_clear()
