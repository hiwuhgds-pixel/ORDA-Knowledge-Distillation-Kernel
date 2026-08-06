from __future__ import annotations

import os

import torch

try:
    from orda_kernel import is_available as _orda_is_available

    ORDA_AVAILABLE = bool(_orda_is_available())
    ORDA_IMPORT_ERROR = None
except Exception as exc:
    ORDA_AVAILABLE = False
    ORDA_IMPORT_ERROR = repr(exc)


# ── Availability policy ──────────────────────────────────────────────────────
def _detail() -> str:
    return "" if ORDA_IMPORT_ERROR is None else f" ({ORDA_IMPORT_ERROR})"


def orda_unavailable_warning(fallback: str) -> str:
    return f"[WARN] orda_kernel is unavailable{_detail()} - {fallback}"


# ── Mode selection ───────────────────────────────────────────────────────────
def backend_modes(source: str | None = None) -> tuple[list[str], str | None]:
    suffix = "" if source is None else f"-{source}"
    modes = [f"torch{suffix}", f"torch-compile{suffix}"]
    if ORDA_AVAILABLE:
        modes.append(f"orda{suffix}")
        return modes, None
    fallback = (
        "running torch modes only"
        if source is None
        else "running torch and torch-compile modes only"
    )
    return modes, orda_unavailable_warning(fallback)


def train_modes() -> tuple[list[str], str | None]:
    if ORDA_AVAILABLE:
        return ["torch-compile", "orda"], None
    return ["torch-compile"], orda_unavailable_warning(
        "running torch-compile mode only"
    )


def precomputed_train_modes() -> tuple[list[str], str | None]:
    if ORDA_AVAILABLE:
        return [
            "orda-hidden",
            "torch-compile-hidden",
            "orda-logits",
            "torch-compile-logits",
        ], None
    return ["torch-compile-hidden", "torch-compile-logits"], orda_unavailable_warning(
        "running torch-compile modes only"
    )


# ── Runtime gate ─────────────────────────────────────────────────────────────
def _ci() -> bool:
    return os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}


def runtime_available(device: torch.device, *, require_cuda: bool = True) -> bool:
    """Raise in CI when ORDA is unavailable so a skipped run cannot look like pass."""
    reason = None
    if require_cuda and device.type != "cuda":
        reason = "CUDA is unavailable - Triton accuracy benchmark cannot run"
    elif not ORDA_AVAILABLE:
        reason = f"orda_kernel is unavailable{_detail()}"
    if reason is None:
        return True
    if _ci():
        raise RuntimeError(reason)
    print(f"[WARN] {reason}")
    return False
