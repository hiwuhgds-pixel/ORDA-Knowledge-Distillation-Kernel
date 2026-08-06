from __future__ import annotations


# ── Block-size helpers ───────────────────────────────────────────────────────
def _next_power_of_2(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _max_block_size(vocab_size: int, max_fused_size: int) -> int:
    """Return the only max-fused value that can affect a launch candidate."""
    return min(int(max_fused_size), _next_power_of_2(int(vocab_size)))
