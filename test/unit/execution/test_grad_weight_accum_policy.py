"""Weight-gradient accumulator dtype policy contracts."""

from __future__ import annotations

import pytest
import torch

from orda_kernel._runtime import resolve_grad_weight_accum_dtype
from orda_kernel.ops._common import accumulate_weight_gradient_, grad_accum_dtype_key


# ── Policy resolution ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("weight_dtype", "compute_dtype", "expected"),
    [
        (torch.float32, torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16, torch.float32),
        (torch.float32, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float32, torch.float32),
    ],
)
def test_auto_promotes_only_mixed_and_float16(weight_dtype, compute_dtype, expected):
    assert (
        resolve_grad_weight_accum_dtype(weight_dtype, compute_dtype, "auto") is expected
    )


@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_compute_policy_never_promotes(weight_dtype):
    assert (
        resolve_grad_weight_accum_dtype(weight_dtype, torch.bfloat16, "compute")
        is torch.bfloat16
    )


@pytest.mark.parametrize("compute_dtype", [torch.float16, torch.bfloat16])
def test_fp32_policy_always_promotes(compute_dtype):
    assert (
        resolve_grad_weight_accum_dtype(torch.float16, compute_dtype, "fp32")
        is torch.float32
    )


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="must be 'auto', 'compute', or 'fp32'"):
        resolve_grad_weight_accum_dtype(torch.float32, torch.float32, "float32")


def test_autotune_key_distinguishes_accumulator_dtypes():
    assert grad_accum_dtype_key(None) is None
    assert grad_accum_dtype_key(torch.bfloat16) != grad_accum_dtype_key(torch.float32)


# ── Mixed parameter/compute accumulation ─────────────────────────────────────
def test_float16_parameter_under_bf16_compute_accumulates_in_float32():
    accum_dtype = resolve_grad_weight_accum_dtype(torch.float16, torch.bfloat16, "auto")
    dlogits = torch.randn(4, 3, dtype=torch.bfloat16)
    hidden = torch.randn(4, 2, dtype=torch.bfloat16)
    destination = torch.empty((3, 2), dtype=accum_dtype)

    grad = accumulate_weight_gradient_(
        destination, dlogits, hidden, row_active=None, overwrite=True
    )

    assert grad.dtype is torch.float32
    torch.testing.assert_close(grad, (dlogits.t() @ hidden).float(), rtol=0.0, atol=0.0)


def test_float16_accumulation_overflows_where_auto_promotion_does_not():
    # Chunk contributions are summed before the reduction scale is applied in
    # backward, so an unscaled float16 buffer saturates on large batches.
    # Each chunk contribution (8 * 30 * 30 = 7200) is representable in float16;
    # only their sum over 64 chunks exceeds the float16 range.
    chunks = 64
    dlogits = torch.full((8, 4), 30.0, dtype=torch.float16)
    hidden = torch.full((8, 2), 30.0, dtype=torch.float16)

    unscaled = torch.empty((4, 2), dtype=torch.float16)
    promoted = torch.empty((4, 2), dtype=torch.float32)
    for index in range(chunks):
        for destination in (unscaled, promoted):
            accumulate_weight_gradient_(
                destination,
                dlogits,
                hidden,
                row_active=None,
                overwrite=index == 0,
            )

    assert torch.isinf(unscaled).all()
    assert torch.isfinite(promoted).all()
