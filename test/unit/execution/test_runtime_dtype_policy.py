"""Capability-aware autocast dtype policy contracts."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch

from orda_kernel import _runtime as runtime
from orda_kernel.ops._kernels import ce_kl_fwdbwd as canonical_kernel


# ── Dtype-only tensor stub ───────────────────────────────────────────────────
@dataclass(frozen=True)
class _TensorDtypeStub:
    dtype: torch.dtype
    device: torch.device


def _enable_bf16_autocast(monkeypatch, *, bf16_supported: bool) -> None:
    monkeypatch.setattr(runtime.torch, "is_autocast_enabled", lambda *_args: True)
    monkeypatch.setattr(
        runtime.torch,
        "get_autocast_dtype",
        lambda _device_type: torch.bfloat16,
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "supports_bfloat16",
        lambda _device: bf16_supported,
    )


# ── Legacy CPU autocast compatibility ────────────────────────────────────────
def test_cpu_autocast_uses_cpu_specific_legacy_api(monkeypatch):
    calls = []

    def legacy_cuda_enabled():
        calls.append("cuda_enabled")
        return False

    def legacy_cpu_enabled():
        calls.append("cpu_enabled")
        return True

    def legacy_get_autocast_dtype():
        calls.append("generic_dtype")
        return torch.float16

    def legacy_cpu_dtype():
        calls.append("cpu_dtype")
        return torch.bfloat16

    monkeypatch.setattr(runtime.torch, "is_autocast_enabled", legacy_cuda_enabled)
    monkeypatch.setattr(
        runtime.torch,
        "is_autocast_cpu_enabled",
        legacy_cpu_enabled,
        raising=False,
    )
    monkeypatch.setattr(
        runtime.torch,
        "get_autocast_dtype",
        legacy_get_autocast_dtype,
        raising=False,
    )
    monkeypatch.setattr(
        runtime.torch,
        "get_autocast_cpu_dtype",
        legacy_cpu_dtype,
        raising=False,
    )

    assert runtime._active_autocast_dtype(torch.device("cpu")) == torch.bfloat16
    assert calls == ["cpu_enabled", "cpu_dtype"]


# ── Pre-Ampere BF16 autocast ─────────────────────────────────────────────────
def test_unsupported_bf16_autocast_preserves_native_fp16_and_fp32(monkeypatch):
    cuda = torch.device("cuda", 0)
    _enable_bf16_autocast(monkeypatch, bf16_supported=False)

    assert runtime._active_autocast_dtype(cuda) is None
    assert runtime.triton_dtype_is_supported(torch.float16, cuda)
    assert runtime.triton_dtype_is_supported(torch.float32, cuda)
    assert (
        runtime.resolve_uniform_model_dtype(
            ("student_hidden", _TensorDtypeStub(torch.float16, cuda)),
            ("weight", _TensorDtypeStub(torch.float16, cuda)),
        )
        == torch.float16
    )
    assert (
        runtime.resolve_uniform_model_dtype(
            ("student_hidden", _TensorDtypeStub(torch.float32, cuda)),
            ("weight", _TensorDtypeStub(torch.float32, cuda)),
        )
        == torch.float32
    )


def test_unsupported_bf16_autocast_does_not_coerce_mixed_inputs(monkeypatch):
    cuda = torch.device("cuda", 0)
    _enable_bf16_autocast(monkeypatch, bf16_supported=False)

    with pytest.raises(ValueError, match="same dtype"):
        runtime.resolve_uniform_model_dtype(
            ("student_hidden", _TensorDtypeStub(torch.float16, cuda)),
            ("weight", _TensorDtypeStub(torch.float32, cuda)),
        )


def test_supported_bf16_autocast_remains_the_effective_compute_dtype(monkeypatch):
    cuda = torch.device("cuda", 0)
    _enable_bf16_autocast(monkeypatch, bf16_supported=True)

    assert runtime._active_autocast_dtype(cuda) == torch.bfloat16
    assert (
        runtime.resolve_uniform_model_dtype(
            ("student_hidden", _TensorDtypeStub(torch.float32, cuda)),
            ("weight", _TensorDtypeStub(torch.float32, cuda)),
        )
        == torch.bfloat16
    )


def test_supported_bf16_autocast_normalizes_mixed_model_inputs(monkeypatch):
    cuda = torch.device("cuda", 0)
    _enable_bf16_autocast(monkeypatch, bf16_supported=True)

    assert (
        runtime.resolve_uniform_model_dtype(
            ("student_hidden", _TensorDtypeStub(torch.float16, cuda)),
            ("weight", _TensorDtypeStub(torch.float32, cuda)),
        )
        == torch.bfloat16
    )


# ── Native BF16 capability ───────────────────────────────────────────────────
def test_native_bf16_query_disables_emulation(monkeypatch):
    runtime._supports_bfloat16_cuda.cache_clear()
    calls = []
    monkeypatch.setattr(runtime.torch.cuda, "device", lambda _index: nullcontext())

    def supports_bf16(*, including_emulation=True):
        calls.append(including_emulation)
        return including_emulation

    monkeypatch.setattr(runtime.torch.cuda, "is_bf16_supported", supports_bf16)
    try:
        assert runtime._supports_bfloat16_cuda(0) is False
        assert calls == [False]
    finally:
        runtime._supports_bfloat16_cuda.cache_clear()


@pytest.mark.parametrize(("major", "expected"), ((7, False), (8, True)))
def test_native_bf16_query_supports_pytorch_23_api(monkeypatch, major, expected):
    runtime._supports_bfloat16_cuda.cache_clear()
    monkeypatch.setattr(runtime.torch.cuda, "device", lambda _index: nullcontext())
    monkeypatch.setattr(runtime, "is_hip", lambda: False)
    monkeypatch.setattr(runtime.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        runtime.torch.cuda,
        "get_device_capability",
        lambda _index: (major, 0),
    )
    try:
        assert runtime._supports_bfloat16_cuda(0) is expected
    finally:
        runtime._supports_bfloat16_cuda.cache_clear()


def test_rocm_bf16_uses_the_runtime_capability_predicate(monkeypatch):
    runtime._supports_bfloat16_cuda.cache_clear()
    monkeypatch.setattr(runtime, "is_hip", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "device", lambda _index: nullcontext())
    monkeypatch.setattr(
        runtime.torch.cuda,
        "is_bf16_supported",
        lambda *, including_emulation=True: False,
    )
    try:
        assert runtime._supports_bfloat16_cuda(0) is False
    finally:
        runtime._supports_bfloat16_cuda.cache_clear()


# ── Internal GEMM autocast guard ─────────────────────────────────────────────
def test_unsupported_bf16_autocast_disables_only_explicit_fp16_fp32_gemms(monkeypatch):
    cuda = torch.device("cuda", 0)
    calls = []

    monkeypatch.setattr(
        canonical_kernel.torch, "is_autocast_enabled", lambda *_args: True
    )
    monkeypatch.setattr(
        canonical_kernel.torch,
        "get_autocast_dtype",
        lambda _device_type: torch.bfloat16,
        raising=False,
    )
    monkeypatch.setattr(canonical_kernel, "supports_bfloat16", lambda _device: False)
    monkeypatch.setattr(
        canonical_kernel.torch,
        "autocast",
        lambda **kwargs: calls.append(kwargs) or nullcontext(),
    )

    with canonical_kernel.compute_autocast_context(cuda, torch.float16):
        pass
    assert calls == [{"device_type": "cuda", "enabled": False}]

    with canonical_kernel.compute_autocast_context(cuda, torch.bfloat16):
        pass
    assert len(calls) == 1


# ── Fallback expm1 polynomial ────────────────────────────────────────────────
def _fallback_expm1(x: torch.Tensor) -> torch.Tensor:
    x2 = x * x
    small = x + x2 * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x / 120.0)))
    return torch.where(x.abs() <= 1.0e-2, small, torch.exp(x) - 1.0)


def test_fallback_expm1_polynomial_matches_expm1_on_small_domain():
    x = torch.linspace(-1.0e-2, 1.0e-2, 4097, dtype=torch.float64)

    approx = _fallback_expm1(x)

    # 5th-order Taylor truncation over |x| <= 1e-2 lands well under fp32 eps.
    torch.testing.assert_close(approx, torch.expm1(x), rtol=0.0, atol=1e-14)


def test_fallback_expm1_beats_naive_cancellation_near_zero():
    x = torch.linspace(-1.0e-3, 1.0e-3, 2049, dtype=torch.float32)
    exact = torch.expm1(x.double())

    fallback_err = (_fallback_expm1(x).double() - exact).abs().max()
    naive_err = ((torch.exp(x) - 1.0).double() - exact).abs().max()

    assert fallback_err <= naive_err
