import torch

from .resolver import DEFAULT_MAX_FUSED_SIZE


def _chunk_size_from_num_chunks(BT: int, num_chunks: int) -> int:
    return (BT + num_chunks - 1) // num_chunks


def _is_oom_error(exc) -> bool:
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(exc).lower():
        return True
    return False


def _tensor_sig(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    device = tensor.device
    return (
        tuple(tensor.shape),
        tensor.dtype,
        device.type,
        device.index,
    )


def _cache_key(
    BT: int,
    V: int,
    H: int,
    h_student: torch.Tensor,
    weight: torch.Tensor,
    teacher_mode: str = "tied",
    student_only: bool = False,
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    max_chunks: int | None = None,
    h_teacher: torch.Tensor | None = None,
    weight_teacher: torch.Tensor | None = None,
    logits_teacher: torch.Tensor | None = None,
):
    device = h_student.device
    return (
        BT,
        V,
        H,
        device.type,
        device.index,
        h_student.dtype,
        weight.dtype,
        teacher_mode,
        student_only,
        int(max_fused_size),
        max_chunks,
        _tensor_sig(h_teacher),
        _tensor_sig(weight_teacher),
        _tensor_sig(logits_teacher),
    )


_cached_num_chunks: dict = {}


def clear_chunk_cache():
    _cached_num_chunks.clear()


def get_chunk_cache() -> dict:
    return dict(_cached_num_chunks)


@torch._dynamo.disable
def dynamic_chunk(
    h_student:       torch.Tensor,
    h_teacher:       torch.Tensor | None,
    weight:          torch.Tensor,
    target:          torch.Tensor,
    lambda_student:  float = 1.0,
    ignore_index:    int   = -100,
    reduction:       str   = "mean",
    label_smoothing: float = 0.0,
    chunk_size:      str | int | None = None,
    use_int8_quant:       bool | None = None,
    use_stochastic_quant: bool | None = None,
    use_fast_math_exp:    bool | None = None,
    use_fast_math_log:    bool | None = None,
    use_fast_math_mul:    bool | None = None,
    use_online_softmax:   bool | None = None,
    use_fp32_accum:       bool | None = None,
    use_kl_in_kernel:     bool | None = None,
    kl_weight:            float       = 0.0,
    kl_temperature:       float       = 1.0,
    max_chunks:           int | None  = None,
    max_fused_size:       int         = DEFAULT_MAX_FUSED_SIZE,
    teacher_mode:         str | None  = None,
    weight_teacher:       torch.Tensor | None = None,
    logits_teacher:       torch.Tensor | None = None,
    teacher_loss_weight:  float | None = None,
    stochastic_seed:      int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunk dispatcher with OOM fallback.

    On OOM: double num_chunks and retry.
    """
    from ..ops.cross_entropy import distill_cross_entropy as _distill_cross_entropy
    from .resolver import resolve_chunk_size, _is_auto_chunk_size

    BT, H = h_student.shape
    V     = weight.shape[0]

    common_kwargs = dict(
        lambda_student=lambda_student, ignore_index=ignore_index,
        reduction=reduction, label_smoothing=label_smoothing,
        use_int8_quant=use_int8_quant, use_stochastic_quant=use_stochastic_quant,
        use_fast_math_exp=use_fast_math_exp, use_fast_math_log=use_fast_math_log,
        use_fast_math_mul=use_fast_math_mul, use_online_softmax=use_online_softmax,
        use_fp32_accum=use_fp32_accum,
        use_kl_in_kernel=use_kl_in_kernel,
        kl_weight=kl_weight, kl_temperature=kl_temperature,
        max_chunks=max_chunks, max_fused_size=max_fused_size,
        teacher_mode=teacher_mode, weight_teacher=weight_teacher,
        logits_teacher=logits_teacher, teacher_loss_weight=teacher_loss_weight,
        stochastic_seed=stochastic_seed,
    )

    if not _is_auto_chunk_size(chunk_size):
        cs_actual, _ = resolve_chunk_size(BT, chunk_size, max_chunks=max_chunks)
        return _distill_cross_entropy(h_student, h_teacher, weight, target,
                                      chunk_size=cs_actual, **common_kwargs)

    max_useful_chunks = max(1, BT // 512)
    max_chunks_limit = max_chunks
    if max_chunks_limit is None:
        max_chunks_limit = 2 * max_useful_chunks

    _, dynamic_num_chunks = resolve_chunk_size(BT, chunk_size, V=V, max_chunks=max_chunks_limit)
    _effective_mode = "tied" if teacher_mode is None else teacher_mode
    _eff_tlw = teacher_loss_weight
    if _eff_tlw is None:
        _eff_tlw = 1.0 if _effective_mode == "tied" else 0.0
    _student_only = (float(_eff_tlw) == 0.0)
    key        = _cache_key(
        BT,
        V,
        H,
        h_student,
        weight,
        _effective_mode,
        _student_only,
        max_fused_size,
        max_chunks_limit,
        h_teacher,
        weight_teacher,
        logits_teacher,
    )
    num_chunks = max(_cached_num_chunks.get(key, dynamic_num_chunks), dynamic_num_chunks)
    num_chunks = min(num_chunks, BT, max_chunks_limit)

    while True:
        cs_attempt = _chunk_size_from_num_chunks(BT, num_chunks)
        try:
            result = _distill_cross_entropy(h_student, h_teacher, weight, target,
                                            chunk_size=cs_attempt, **common_kwargs)
            _cached_num_chunks[key] = num_chunks
            return result

        except Exception as exc:
            if not _is_oom_error(exc):
                raise

            torch.cuda.empty_cache()

            if num_chunks >= BT or num_chunks >= max_chunks_limit:
                fp32_status = "enabled" if use_fp32_accum else "disabled"
                raise RuntimeError(
                    f"Out of memory with max_chunks={max_chunks_limit}, fp32_accum={fp32_status}\n"
                    f"BT={BT}, V={V}, H={H}\n"
                    f"Options:\n"
                    f"  - Increase max_chunks to {max_chunks_limit * 2}\n"
                    f"  - Enable fp32 grad-weight accumulation if numerical stability is required\n"
                    f"  - Reduce batch size or sequence length"
                ) from exc

            num_chunks = min(num_chunks * 2, BT, max_chunks_limit)
