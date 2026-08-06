from __future__ import annotations

from importlib import import_module

# ── Lazy public surface ──────────────────────────────────────────────────────
__all__ = [
    "aligned_span_distillation_loss",
    "precomputed_distillation_loss",
    "separate_distillation_loss",
    "tied_distillation_loss",
]

_MODULE_BY_EXPORT = {
    "aligned_span_distillation_loss": ".precomputed_inputs.aligned_span",
    "precomputed_distillation_loss": ".precomputed_teacher",
    "separate_distillation_loss": ".separate_teacher",
    "tied_distillation_loss": ".tied_teacher",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_EXPORT.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
