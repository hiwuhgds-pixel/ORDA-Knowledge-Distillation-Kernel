from ._autotune import DEFAULT_MAX_FUSED_SIZE


# ── Basic predicates ─────────────────────────────────────────────────────────
def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _is_auto_chunk_size(chunk_size) -> bool:
    return (
        chunk_size is None
        or chunk_size == "auto"
        or chunk_size == "dynamic"
        or chunk_size == -2
        or (isinstance(chunk_size, (int, float)) and chunk_size <= 0)
    )


# ── Dynamic chunk heuristic ──────────────────────────────────────────────────
def _chunks_from_raw(raw: float) -> int:
    if raw < 1.5:
        return 1
    import math
    return 1 << (math.floor(math.log2(raw / 1.5)) + 1)


# ── Public chunk resolver ────────────────────────────────────────────────────
def resolve_chunk_size(BT, chunk_size_arg, V=None, max_chunks=None, num_chunks=None):
    """Compute chunk_size and num_chunks from either chunk_size or num_chunks."""
    if int(BT) == 0:
        raise ValueError("BT (batch x sequence length) must be > 0.")

    if num_chunks is not None:
        if not _is_auto_chunk_size(chunk_size_arg):
            raise ValueError("chunk_size and num_chunks are mutually exclusive.")
        requested_chunks = int(num_chunks)
        if requested_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        effective_chunks = min(requested_chunks, int(BT))
        if max_chunks is not None and effective_chunks > int(max_chunks):
            raise ValueError(
                f"num_chunks must be <= max_chunks, got num_chunks={num_chunks}, max_chunks={max_chunks}"
            )
        return (BT + effective_chunks - 1) // effective_chunks, effective_chunks

    if _is_auto_chunk_size(chunk_size_arg):
        if V is None:
            return BT, 1

        max_useful_chunks = max(1, int(BT) // 512)
        if max_chunks is None:
            max_chunks = 2 * max_useful_chunks

        raw_pressure = (float(BT) / 1024.0) * ((float(V) / 32768.0) ** 2)
        raw_bt_floor = float(BT) / 4096.0
        num_chunks = _chunks_from_raw(max(raw_pressure, raw_bt_floor))

        num_chunks = min(max_chunks, max_useful_chunks, num_chunks, int(BT))
        return (BT + num_chunks - 1) // num_chunks, num_chunks

    cs = min(int(chunk_size_arg), BT)
    return cs, (BT + cs - 1) // cs
