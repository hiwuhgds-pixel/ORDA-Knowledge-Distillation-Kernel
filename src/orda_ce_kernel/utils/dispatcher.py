import threading

import torch

from .resolver import DEFAULT_MAX_FUSED_SIZE


# ── Chunk helpers ────────────────────────────────────────────────────────────
def _chunk_size_from_num_chunks(BT: int, num_chunks: int) -> int:
    return (BT + num_chunks - 1) // num_chunks


def _is_oom_error(exc) -> bool:
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


def _tensor_sig(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    device = tensor.device
    return (tuple(tensor.shape), tensor.dtype, device.type, device.index)


def _cache_key(
    BT: int,
    V: int,
    H: int,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    teacher_mode: str = "tied",
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    max_chunks: int | None = None,
    teacher_hidden: torch.Tensor | None = None,
    teacher_weight: torch.Tensor | None = None,
    logits_teacher: torch.Tensor | None = None,
    compute_kl: bool = True,
):
    device = student_hidden.device
    return (
        BT,
        V,
        H,
        device.type,
        device.index,
        student_hidden.dtype,
        weight.dtype,
        teacher_mode,
        int(max_fused_size),
        max_chunks,
        bool(compute_kl),
        _tensor_sig(teacher_hidden),
        _tensor_sig(teacher_weight),
        _tensor_sig(logits_teacher),
    )


_CACHE_LOCK = threading.Lock()
_MAX_CHUNK_CACHE_SIZE = 1024
_cached_num_chunks: dict = {}


# ── Chunk cache API ──────────────────────────────────────────────────────────
def clear_chunk_cache():
    with _CACHE_LOCK:
        _cached_num_chunks.clear()


def get_chunk_cache() -> dict:
    with _CACHE_LOCK:
        return dict(_cached_num_chunks)


# ── Kernel dispatch ──────────────────────────────────────────────────────────
def _resolve_kernel_fn(teacher_mode: str):
    if teacher_mode == "tied":
        from ..ops.tied_teacher import tied_distillation_loss

        return tied_distillation_loss
    if teacher_mode == "separate":
        from ..ops.separate_teacher import separate_distillation_loss

        return separate_distillation_loss
    if teacher_mode == "precomputed":
        from ..ops.precomputed_teacher import precomputed_distillation_loss

        return precomputed_distillation_loss
    raise ValueError(f"teacher_mode must be 'tied', 'separate', or 'precomputed', got {teacher_mode!r}")


def _call_kernel_fn(
    kernel_fn,
    teacher_mode: str,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_ce_weight: float,
    ignore_index: int,
    reduction: str,
    chunk_size: int,
    use_fp32_accum: bool | None,
    kl_weight: float,
    kl_temperature: float,
    max_chunks: int | None,
    max_fused_size: int,
    autotune: bool,
    teacher_weight: torch.Tensor | None,
    logits_teacher: torch.Tensor | None,
    teacher_ce_weight: float | None,
    validate_labels: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    common_kwargs = dict(
        student_ce_weight=student_ce_weight,
        ignore_index=ignore_index,
        reduction=reduction,
        chunk_size=chunk_size,
        use_fp32_accum=use_fp32_accum,
        kl_weight=kl_weight,
        kl_temperature=kl_temperature,
        teacher_ce_weight=teacher_ce_weight,
        max_chunks=max_chunks,
        max_fused_size=max_fused_size,
        autotune=autotune,
        validate_labels=validate_labels,
    )
    if teacher_mode == "tied":
        return kernel_fn(student_hidden, teacher_hidden, weight, labels, **common_kwargs)
    if teacher_mode == "separate":
        return kernel_fn(
            student_hidden,
            teacher_hidden,
            weight,
            teacher_weight,
            labels,
            **common_kwargs,
        )
    return kernel_fn(
        student_hidden, weight, logits_teacher, labels,
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        **common_kwargs,
    )


# ── Dynamic chunk dispatcher ─────────────────────────────────────────────────
@torch._dynamo.disable
def dynamic_chunk(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    weight: torch.Tensor,
    labels: torch.Tensor,
    student_ce_weight: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    chunk_size: str | int | None = None,
    num_chunks: int | None = None,
    use_fp32_accum: bool | None = None,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    max_chunks: int | None = None,
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    autotune: bool = False,
    teacher_mode: str | None = None,
    teacher_weight: torch.Tensor | None = None,
    logits_teacher: torch.Tensor | None = None,
    teacher_ce_weight: float | None = None,
    validate_labels: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunk dispatcher with OOM fallback for the per-mode Triton paths."""
    from .resolver import _is_auto_chunk_size, resolve_chunk_size

    BT, H = student_hidden.shape
    V = weight.shape[0]
    effective_mode = "tied" if teacher_mode is None else teacher_mode
    kernel_fn = _resolve_kernel_fn(effective_mode)

    if num_chunks is not None or not _is_auto_chunk_size(chunk_size):
        # ── Fixed chunk path ────────────────────────────────────────────────
        cs_actual, _ = resolve_chunk_size(
            BT,
            chunk_size,
            max_chunks=max_chunks,
            num_chunks=num_chunks,
        )
        return _call_kernel_fn(
            kernel_fn,
            effective_mode,
            student_hidden,
            teacher_hidden,
            weight,
            labels,
            student_ce_weight=student_ce_weight,
            ignore_index=ignore_index,
            reduction=reduction,
            chunk_size=cs_actual,
            use_fp32_accum=use_fp32_accum,
            kl_weight=kl_weight,
            kl_temperature=kl_temperature,
            max_chunks=max_chunks,
            max_fused_size=max_fused_size,
            autotune=autotune,
            teacher_weight=teacher_weight,
            logits_teacher=logits_teacher,
            teacher_ce_weight=teacher_ce_weight,
            validate_labels=validate_labels,
        )

    # ── Dynamic OOM fallback path ───────────────────────────────────────────
    max_useful_chunks = max(1, BT // 512)
    max_chunks_limit = max_chunks if max_chunks is not None else 2 * max_useful_chunks

    _, dynamic_num_chunks = resolve_chunk_size(
        BT, chunk_size, V=V, max_chunks=max_chunks_limit
    )
    key = _cache_key(
        BT,
        V,
        H,
        student_hidden,
        weight,
        effective_mode,
        max_fused_size,
        max_chunks_limit,
        teacher_hidden,
        teacher_weight,
        logits_teacher,
        compute_kl=float(kl_weight) != 0.0,
    )
    with _CACHE_LOCK:
        cached_num_chunks = _cached_num_chunks.get(key)
    num_chunks = max(cached_num_chunks or dynamic_num_chunks, dynamic_num_chunks)
    num_chunks = min(num_chunks, BT, max_chunks_limit)

    while True:
        # ── Retry with more chunks after OOM ────────────────────────────────
        cs_attempt = _chunk_size_from_num_chunks(BT, num_chunks)
        try:
            result = _call_kernel_fn(
                kernel_fn,
                effective_mode,
                student_hidden,
                teacher_hidden,
                weight,
                labels,
                student_ce_weight=student_ce_weight,
                ignore_index=ignore_index,
                reduction=reduction,
                chunk_size=cs_attempt,
                use_fp32_accum=use_fp32_accum,
                kl_weight=kl_weight,
                kl_temperature=kl_temperature,
                max_chunks=max_chunks,
                max_fused_size=max_fused_size,
                autotune=autotune,
                teacher_weight=teacher_weight,
                logits_teacher=logits_teacher,
                teacher_ce_weight=teacher_ce_weight,
                validate_labels=validate_labels,
            )
            with _CACHE_LOCK:
                if key not in _cached_num_chunks and len(_cached_num_chunks) >= _MAX_CHUNK_CACHE_SIZE:
                    _cached_num_chunks.clear()
                _cached_num_chunks[key] = num_chunks
            return result
        except Exception as exc:
            if not _is_oom_error(exc):
                raise

            torch.cuda.empty_cache()

            if num_chunks >= BT or num_chunks >= max_chunks_limit:
                fp32_status = "enabled" if use_fp32_accum else "disabled"
                raise RuntimeError(
                    f"Out of memory with max_chunks={max_chunks_limit}, "
                    f"fp32_accum={fp32_status}\n"
                    f"BT={BT}, V={V}, H={H}\n"
                    f"Options:\n"
                    f"  - Increase max_chunks to {max_chunks_limit * 2}\n"
                    f"  - Enable fp32 grad-weight accumulation if numerical stability is required\n"
                    f"  - Reduce batch size or sequence length"
                ) from exc

            num_chunks = min(num_chunks * 2, BT, max_chunks_limit)
