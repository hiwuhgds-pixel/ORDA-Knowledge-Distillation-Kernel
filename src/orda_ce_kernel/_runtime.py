import torch


# ── Triton availability guard ─────────────────────────────────────────────────
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    triton = None  # type: ignore[assignment]
    tl = None      # type: ignore[assignment]
    HAS_TRITON = False


# ── Device Detection ──────────────────────────────────────────────────────────
def is_hip() -> bool:
    """Returns True if the platform is AMD ROCm/HIP."""
    return getattr(torch.version, "hip", None) is not None


# ── High-Precision CUDA/HIP Libdevice Math Functions ──────────────────────────
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


_LOG2E = tl.constexpr(1.4426950408889634) if HAS_TRITON else 1.4426950408889634
