from __future__ import annotations

from functools import lru_cache

import torch

# ── Runtime and dtype policy ─────────────────────────────────────────────────
TRITON_IMPORT_ERROR: BaseException | None = None

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except (ImportError, ModuleNotFoundError, OSError) as exc:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False
    TRITON_IMPORT_ERROR = exc


def triton_unavailable_reason() -> str:
    """Describe why the optional Triton runtime is unavailable."""
    if TRITON_IMPORT_ERROR is None:
        return "Triton is not installed"
    return (
        "Triton import failed with "
        f"{type(TRITON_IMPORT_ERROR).__name__}: {TRITON_IMPORT_ERROR}"
    )


TORCH_MODEL_DTYPES = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
TRITON_MODEL_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))


def cast_logits_for_compute(
    logits: torch.Tensor,
    dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cast logits after source-dtype row centering to prevent positive overflow."""
    if logits.dtype == dtype:
        if out is None:
            return logits
        out.copy_(logits)
        return out

    row_max = logits.amax(dim=-1, keepdim=True)
    center = torch.where(torch.isneginf(row_max), torch.zeros_like(row_max), row_max)
    centered = logits - center
    centered = torch.where(
        torch.isfinite(centered),
        centered.clamp_min(torch.finfo(dtype).min),
        centered,
    )
    if out is None:
        return centered.to(dtype)
    out.copy_(centered)
    return out


# ── Device capabilities ──────────────────────────────────────────────────────
def is_hip() -> bool:
    """Return True on AMD ROCm/HIP."""
    return getattr(torch.version, "hip", None) is not None


@lru_cache(maxsize=16)
def _supports_bfloat16_cuda(index: int) -> bool:
    try:
        with torch.cuda.device(index):
            supports_bf16 = torch.cuda.is_bf16_supported
            try:
                # Newer PyTorch versions may report emulation by default; the
                # Triton path requires native BF16 hardware support.
                return bool(supports_bf16(including_emulation=False))
            except TypeError:
                if is_hip():
                    return bool(supports_bf16())
                major, _ = torch.cuda.get_device_capability(index)
                return major >= 8
    except (AttributeError, RuntimeError):
        if is_hip():
            return False
        try:
            major, _ = torch.cuda.get_device_capability(index)
            return major >= 8
        except (AttributeError, RuntimeError):
            return False


def supports_bfloat16(device: torch.device | None = None) -> bool:
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    index = (
        torch.cuda.current_device()
        if device is None or device.index is None
        else int(device.index)
    )
    return _supports_bfloat16_cuda(index)


def triton_dtype_is_supported(dtype: torch.dtype, device: torch.device) -> bool:
    if dtype not in TRITON_MODEL_DTYPES:
        return False
    if dtype == torch.bfloat16 and device.type == "cuda":
        return supports_bfloat16(device)
    return True


def _active_autocast_dtype(device: torch.device | None) -> torch.dtype | None:
    device_type = device.type if device is not None else "cuda"
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:
        enabled = (
            torch.is_autocast_cpu_enabled()
            if device_type == "cpu"
            else torch.is_autocast_enabled()
        )
    if not enabled:
        return None

    try:
        dtype = torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):
        dtype = (
            torch.get_autocast_cpu_dtype()
            if device_type == "cpu"
            else torch.get_autocast_gpu_dtype()
        )
    if dtype == torch.bfloat16:
        # A requested BF16 autocast mode is not a usable Triton compute dtype on
        # pre-Ampere CUDA. Falling back to the tensor dtype preserves FP16 and
        # FP32 execution instead of silently coercing either one to FP16.
        if device_type == "cuda" and not supports_bfloat16(device):
            return None
        return torch.bfloat16
    if dtype == torch.float16:
        return torch.float16
    return None


# ── Dtype resolution ─────────────────────────────────────────────────────────
def resolve_uniform_model_dtype(
    *named_tensors: tuple[str, torch.Tensor],
) -> torch.dtype:
    """Resolve one dtype, allowing supported mixtures only under autocast."""
    if not named_tensors:
        raise ValueError("at least one model tensor is required to resolve dtype")

    dtype_items = [(name, tensor.dtype) for name, tensor in named_tensors]
    invalid = [
        (name, dtype) for name, dtype in dtype_items if dtype not in TORCH_MODEL_DTYPES
    ]
    if invalid:
        details = ", ".join(f"{name}={dtype}" for name, dtype in invalid)
        raise ValueError(
            "model tensors must use a floating dtype from "
            "{torch.float16, torch.bfloat16, torch.float32, torch.float64}; "
            f"got {details}"
        )

    unique_dtypes = {dtype for _, dtype in dtype_items}
    autocast_dtype = _active_autocast_dtype(named_tensors[0][1].device)
    if autocast_dtype is not None and unique_dtypes <= TRITON_MODEL_DTYPES:
        return autocast_dtype
    if len(unique_dtypes) != 1:
        details = ", ".join(f"{name}={dtype}" for name, dtype in dtype_items)
        raise ValueError(
            f"all model tensors must use the same dtype outside autocast; got {details}"
        )
    return dtype_items[0][1]


def default_compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    model_tensors = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
    dtype = resolve_uniform_model_dtype(
        *((f"tensor_{index}", tensor) for index, tensor in enumerate(model_tensors))
    )
    device = model_tensors[0].device

    if not triton_dtype_is_supported(dtype, device):
        raise ValueError(
            "Triton compute dtype must be one of "
            "{torch.float16, torch.bfloat16, torch.float32} and supported by "
            f"the target device; got {dtype} on {device}"
        )

    return dtype


def resolve_grad_weight_accum_dtype(
    weight_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    policy: str,
) -> torch.dtype:
    """Resolve the dtype of a weight-gradient accumulator buffer.

    ``auto`` promotes to float32 whenever the parameter dtype differs from the
    compute dtype, and for float16 parameters because chunk contributions are
    summed before the reduction scale is applied in ``backward``.
    """
    if policy == "fp32":
        return torch.float32
    if policy == "compute":
        return compute_dtype
    if policy != "auto":
        raise ValueError(
            "grad_weight_accumulation must be 'auto', 'compute', or 'fp32', got "
            f"{policy!r}"
        )
    if weight_dtype != compute_dtype or weight_dtype == torch.float16:
        return torch.float32
    return weight_dtype


def resolve_gradient_materialization_scale(
    student_hidden: torch.Tensor,
    gradient_materialization_scale: torch.Tensor | None,
    grad_scaler: torch.amp.GradScaler | None,
) -> torch.Tensor | None:
    if grad_scaler is None:
        return gradient_materialization_scale
    if gradient_materialization_scale is not None:
        raise ValueError(
            "grad_scaler and gradient_materialization_scale cannot be used together"
        )
    if not grad_scaler.is_enabled():
        return None
    unit = torch.ones((), dtype=torch.float32, device=student_hidden.device)
    return grad_scaler.scale(unit)


# ── High-precision Triton math ───────────────────────────────────────────────
if HAS_TRITON:
    try:
        from triton.language.extra.libdevice import exp as tl_highprec_exp
        from triton.language.extra.libdevice import expm1 as tl_highprec_expm1
        from triton.language.extra.libdevice import log as tl_highprec_log
    except (ImportError, ModuleNotFoundError, OSError):
        try:
            from triton.language.extra.cuda.libdevice import exp as tl_highprec_exp
            from triton.language.extra.cuda.libdevice import expm1 as tl_highprec_expm1
            from triton.language.extra.cuda.libdevice import log as tl_highprec_log
        except (ImportError, ModuleNotFoundError, OSError):

            @triton.jit
            def tl_highprec_exp(x):
                return tl.exp(x)

            @triton.jit
            def tl_highprec_log(x):
                return tl.log(x)

            @triton.jit
            def tl_highprec_expm1(x):
                x2 = x * x
                small = x + x2 * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x / 120.0)))
                return tl.where(tl.abs(x) <= 1.0e-2, small, tl_highprec_exp(x) - 1.0)
else:

    def tl_highprec_exp(x) -> None:  # type: ignore[misc]
        raise RuntimeError(
            "Triton kernels are unavailable; install Triton and use CUDA/HIP tensors."
        )

    def tl_highprec_log(x) -> None:  # type: ignore[misc]
        raise RuntimeError(
            "Triton kernels are unavailable; install Triton and use CUDA/HIP tensors."
        )

    def tl_highprec_expm1(x) -> None:  # type: ignore[misc]
        raise RuntimeError(
            "Triton kernels are unavailable; install Triton and use CUDA/HIP tensors."
        )
