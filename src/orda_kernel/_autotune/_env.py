from __future__ import annotations

import os
import warnings


# ── Environment controls ─────────────────────────────────────────────────────
def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not an integer; using default {default}",
            RuntimeWarning,
            stacklevel=2,
        )
        return default
    bounded = min(maximum, max(minimum, value))
    if bounded != value:
        warnings.warn(
            f"{name}={value} is outside [{minimum}, {maximum}]; using {bounded}",
            RuntimeWarning,
            stacklevel=2,
        )
    return bounded
