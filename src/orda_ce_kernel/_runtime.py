import torch


# ── Triton availability ──────────────────────────────────────────────────────
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    triton = None  # type: ignore[assignment]
    tl = None      # type: ignore[assignment]
    HAS_TRITON = False


# ── Backend dtype helpers ────────────────────────────────────────────────────
def is_hip() -> bool:
    """Return True on AMD ROCm/HIP."""
    return getattr(torch.version, "hip", None) is not None


def supports_bfloat16(device: torch.device | None = None) -> bool:
    if is_hip():
        return True
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        if device is not None and device.index is not None:
            with torch.cuda.device(device):
                return bool(torch.cuda.is_bf16_supported())
        return bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        try:
            major, _ = torch.cuda.get_device_capability(device)
            return major >= 8
        except RuntimeError:
            return False


def _active_autocast_dtype(device: torch.device | None) -> torch.dtype | None:
    device_type = device.type if device is not None else "cuda"
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return None

    try:
        dtype = torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):
        dtype = torch.get_autocast_gpu_dtype()
    if dtype == torch.bfloat16 and supports_bfloat16(device):
        return torch.bfloat16
    if dtype == torch.float16:
        return torch.float16
    return None


def default_compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    device = next((tensor.device for tensor in tensors if isinstance(tensor, torch.Tensor)), None)

    autocast_dtype = _active_autocast_dtype(device)
    if autocast_dtype is not None:
        return autocast_dtype

    floating_dtypes = [
        tensor.dtype
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
    ]
    if torch.bfloat16 in floating_dtypes:
        return torch.bfloat16 if supports_bfloat16(device) else torch.float16
    if torch.float16 in floating_dtypes:
        return torch.float16
    if torch.float32 in floating_dtypes:
        return torch.float32

    return torch.bfloat16 if is_hip() else torch.float16


# ── High-precision Triton math ───────────────────────────────────────────────
if HAS_TRITON:
    try:
        from triton.language.extra.libdevice import exp as tl_highprec_exp
        from triton.language.extra.libdevice import log as tl_highprec_log
    except (ImportError, ModuleNotFoundError):
        try:
            from triton.language.extra.cuda.libdevice import exp as tl_highprec_exp
            from triton.language.extra.cuda.libdevice import log as tl_highprec_log
        except (ImportError, ModuleNotFoundError):
            @triton.jit
            def tl_highprec_exp(x):
                return tl.exp(x)

            @triton.jit
            def tl_highprec_log(x):
                return tl.log(x)
else:
    def tl_highprec_exp(x):  # type: ignore[misc]
        raise RuntimeError(
            "Triton kernels are unavailable; install Triton and use CUDA/HIP tensors."
        )

    def tl_highprec_log(x):  # type: ignore[misc]
        raise RuntimeError(
            "Triton kernels are unavailable; install Triton and use CUDA/HIP tensors."
        )
