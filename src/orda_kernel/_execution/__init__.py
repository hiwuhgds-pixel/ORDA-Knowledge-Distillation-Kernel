"""Chunk planning, dispatch, and recovery internals."""

from __future__ import annotations

from importlib import import_module

# ── Lazy subsystem modules ───────────────────────────────────────────────────
__all__ = ["chunking", "dispatcher", "recovery"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
