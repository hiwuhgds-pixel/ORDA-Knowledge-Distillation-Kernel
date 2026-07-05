from __future__ import annotations

import json
import gc
import os
import threading
import warnings
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import torch


AUTOTUNE_VERSION = 3
AUTOTUNE_WARMUP_MS = int(os.environ.get("ORDA_AUTOTUNE_WARMUP_MS", "25"))
AUTOTUNE_REP_MS = int(os.environ.get("ORDA_AUTOTUNE_REP_MS", "100"))


# ── Launch config model ──────────────────────────────────────────────────────
class LaunchConfig(NamedTuple):
    block_size: int
    num_warps: int


TIED = "tied"
SEPARATE_FULL = "separate_full"
SEPARATE_STUDENT = "separate_student"
PRECOMPUTED = "precomputed"


# ── Defaults and candidate space ─────────────────────────────────────────────
DEFAULT_MAX_FUSED_SIZE = 65536 // 2

T4_DEFAULT_CONFIGS: dict[str, LaunchConfig] = {
    PRECOMPUTED: LaunchConfig(16384, 8),
    SEPARATE_FULL: LaunchConfig(8192, 8),
    SEPARATE_STUDENT: LaunchConfig(16384, 8),
    TIED: LaunchConfig(8192, 16),
}

DEFAULT_CONFIGS: dict[str, LaunchConfig] = {
    PRECOMPUTED: LaunchConfig(8192, 8),
    SEPARATE_FULL: LaunchConfig(16384, 32),
    SEPARATE_STUDENT: LaunchConfig(8192, 8),
    TIED: LaunchConfig(16384, 32),
}

# Do not use @triton.autotune here: ORDA kernels overwrite logits buffers in-place,
# so each trial must build fresh GEMM outputs before launching the kernel.
_CANDIDATES: tuple[LaunchConfig, ...] = tuple(
    LaunchConfig(block_size, num_warps)
    for block_size in (8192, 16384, 32768)
    for num_warps in (8, 16, 32)
)

_LOCK = threading.Lock()
_MEMORY_CACHE: dict[str, LaunchConfig] = {}
_DISK_CACHE: dict[str, list[int]] | None = None


# ── Candidate filtering ──────────────────────────────────────────────────────
def _next_power_of_2(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _max_block_size(vocab_size: int, max_fused_size: int) -> int:
    return min(int(max_fused_size), _next_power_of_2(int(vocab_size)))


def valid_candidates(vocab_size: int, max_fused_size: int) -> list[LaunchConfig]:
    max_block = _max_block_size(vocab_size, max_fused_size)
    return [config for config in _CANDIDATES if config.block_size <= max_block]


def _is_t4_device() -> bool:
    if not torch.cuda.is_available():
        return False

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if "t4" in props.name.lower():
        return True

    mem_gb = props.total_memory / 1024**3
    return props.major == 7 and props.minor == 5 and 14 <= mem_gb <= 17


def _default_configs_for_device() -> dict[str, LaunchConfig]:
    return T4_DEFAULT_CONFIGS if _is_t4_device() else DEFAULT_CONFIGS


def default_config(mode: str, vocab_size: int, max_fused_size: int) -> LaunchConfig:
    config = _default_configs_for_device()[mode]
    max_block = _max_block_size(vocab_size, max_fused_size)
    if config.block_size <= max_block:
        return config

    candidates = valid_candidates(vocab_size, max_fused_size)
    same_warps = [candidate for candidate in candidates if candidate.num_warps == config.num_warps]
    if same_warps:
        return same_warps[-1]
    if candidates:
        return candidates[-1]
    return LaunchConfig(max_block, config.num_warps)


# ── Disk cache ───────────────────────────────────────────────────────────────
def _cache_path() -> Path:
    root = os.environ.get("ORDA_AUTOTUNE_CACHE_DIR")
    if root:
        cache_dir = Path(root)
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        cache_dir = Path(os.environ["LOCALAPPDATA"]) / "orda_ce_kernel"
    else:
        cache_dir = Path.home() / ".cache" / "orda_ce_kernel"
    return cache_dir / f"autotune_v{AUTOTUNE_VERSION}.json"


def _read_cache_file(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, list) and len(value) == 2
    }


def _load_disk_cache() -> dict[str, list[int]]:
    global _DISK_CACHE
    if _DISK_CACHE is not None:
        return _DISK_CACHE

    path = _cache_path()
    try:
        _DISK_CACHE = _read_cache_file(path)
    except FileNotFoundError:
        _DISK_CACHE = {}
    except Exception as exc:
        warnings.warn(f"Failed to read ORDA autotune cache: {exc}", RuntimeWarning, stacklevel=2)
        _DISK_CACHE = {}
    return _DISK_CACHE


def _save_disk_cache(cache: dict[str, list[int]]) -> None:
    global _DISK_CACHE
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            merged = _read_cache_file(path)
            merged.update(cache)
        except FileNotFoundError:
            merged = dict(cache)
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)
        _DISK_CACHE = merged
    except Exception as exc:
        warnings.warn(f"Failed to write ORDA autotune cache: {exc}", RuntimeWarning, stacklevel=2)


# ── Cache key ────────────────────────────────────────────────────────────────
def _backend_name() -> str:
    if torch.version.hip:
        return "hip"
    if torch.version.cuda:
        return "cuda"
    return "unknown"


def _device_key(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"type": device.type, "index": device.index}

    index = torch.cuda.current_device() if device.index is None else int(device.index)
    props = torch.cuda.get_device_properties(index)
    return {
        "type": device.type,
        "index": index,
        "backend": _backend_name(),
        "name": props.name,
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
        "total_memory": props.total_memory,
    }


def _cache_key(
    *,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    n_rows: int,
    max_fused_size: int,
    shape_key: Iterable[int],
    compute_kl: bool = True,
    t_is_one: bool = False,
    compute_teacher_ce: bool = True,
) -> str:
    payload = {
        "version": AUTOTUNE_VERSION,
        "mode": mode,
        "device": _device_key(device),
        "dtype": str(dtype),
        "vocab_size": int(vocab_size),
        "n_rows": int(n_rows),
        "max_fused_size": int(max_fused_size),
        "shape_key": [int(item) for item in shape_key],
        "compute_kl": bool(compute_kl),
        "t_is_one": bool(t_is_one),
        "compute_teacher_ce": bool(compute_teacher_ce),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ── Benchmark runner ─────────────────────────────────────────────────────────
def _is_oom_error(exc: BaseException) -> bool:
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "cuda error: out of memory" in msg
        or "hip out of memory" in msg
        or "hip error: out of memory" in msg
    )


def _bench_config(bench_fn: Callable[[int, int], object], config: LaunchConfig) -> float:
    import triton

    def run_once():
        outputs = bench_fn(config.block_size, config.num_warps)
        del outputs

    return float(
        triton.testing.do_bench(
            run_once,
            warmup=AUTOTUNE_WARMUP_MS,
            rep=AUTOTUNE_REP_MS,
        )
    )


# ── Config selection ─────────────────────────────────────────────────────────
def select_config(
    *,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    n_rows: int,
    max_fused_size: int,
    shape_key: Iterable[int],
    autotune: bool,
    bench_fn: Callable[[int, int], object] | None = None,
    compute_kl: bool = True,
    t_is_one: bool = False,
    compute_teacher_ce: bool = True,
) -> LaunchConfig:
    fallback = default_config(mode, vocab_size, max_fused_size)
    if not autotune:
        return fallback
    if bench_fn is None:
        return fallback

    key = _cache_key(
        mode=mode,
        device=device,
        dtype=dtype,
        vocab_size=vocab_size,
        n_rows=n_rows,
        max_fused_size=max_fused_size,
        shape_key=shape_key,
        compute_kl=compute_kl,
        t_is_one=t_is_one,
        compute_teacher_ce=compute_teacher_ce,
    )

    with _LOCK:
        cached = _MEMORY_CACHE.get(key)
        if cached is not None:
            return cached

        disk_cache = _load_disk_cache()
        disk_value = disk_cache.get(key)
        if disk_value is not None:
            cached = LaunchConfig(int(disk_value[0]), int(disk_value[1]))
            if cached in valid_candidates(vocab_size, max_fused_size) or cached == fallback:
                _MEMORY_CACHE[key] = cached
                return cached

        candidates = valid_candidates(vocab_size, max_fused_size)
        if not candidates:
            _MEMORY_CACHE[key] = fallback
            return fallback

        timings: list[tuple[float, LaunchConfig]] = []
        for config in candidates:
            try:
                timing = _bench_config(bench_fn, config)
            except Exception as exc:
                if _is_oom_error(exc):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                continue
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            timings.append((timing, config))

        if not timings:
            selected = fallback
            warnings.warn(
                f"ORDA autotune failed for mode={mode}; using default "
                f"{fallback.block_size}/{fallback.num_warps}.",
                RuntimeWarning,
                stacklevel=2,
            )
            _MEMORY_CACHE[key] = selected
            return selected
        else:
            selected = min(timings, key=lambda item: item[0])[1]

        _MEMORY_CACHE[key] = selected
        disk_cache[key] = [selected.block_size, selected.num_warps]
        _save_disk_cache(disk_cache)
        return selected


# ── Public cache control ─────────────────────────────────────────────────────
def clear_autotune_cache(*, disk: bool = False) -> None:
    global _DISK_CACHE
    with _LOCK:
        _MEMORY_CACHE.clear()
        _DISK_CACHE = None
        if disk:
            try:
                _cache_path().unlink()
            except FileNotFoundError:
                pass
