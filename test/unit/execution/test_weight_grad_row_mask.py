from __future__ import annotations

import pytest
import torch

from orda_kernel._execution.recovery import NonChunkableWorkspaceOOM
from orda_kernel.ops import _common
from orda_kernel.ops._common import (
    accumulate_weight_gradient_,
    claim_cached_gradients,
    scale_cached_gradient_,
)


def _destination(dlogits, hidden, dtype=None):
    return torch.empty(
        (dlogits.shape[1], hidden.shape[1]),
        dtype=dlogits.dtype if dtype is None else dtype,
    )


# ── Hidden row mask ──────────────────────────────────────────────────────────
def test_ignored_row_with_nonfinite_hidden_does_not_poison_weight_grad():
    # The kernel zeros dlogits on the ignored row, but the hidden operand still
    # carries a NaN there.  Without the mask, ``0 * NaN = NaN`` corrupts every
    # column of the shared weight grad.
    dlogits = torch.tensor([[1.0, -2.0], [0.0, 0.0]])  # row 1 ignored -> zeroed
    hidden = torch.tensor([[0.5, 1.5, -0.5], [float("nan"), float("nan"), 1.0]])
    row_active = torch.tensor([True, False])

    grad = accumulate_weight_gradient_(
        _destination(dlogits, hidden),
        dlogits,
        hidden,
        row_active=row_active,
        overwrite=True,
    )

    assert torch.isfinite(grad).all()
    expected = dlogits[:1].t() @ hidden[:1]
    assert torch.allclose(grad, expected)


def test_row_mask_leaves_valid_rows_bit_identical():
    dlogits = torch.randn(6, 4)
    hidden = torch.randn(6, 3)
    row_active = torch.tensor([True, True, False, True, False, True])

    masked = accumulate_weight_gradient_(
        _destination(dlogits, hidden),
        dlogits,
        hidden,
        row_active=row_active,
        overwrite=True,
    )
    # Zeroing dlogits on the same rows (as the kernel does) is equivalent, so the
    # helper must reproduce that reference exactly.
    reference = (dlogits * row_active.unsqueeze(-1)).t() @ hidden
    assert torch.equal(masked, reference)


def test_row_mask_applies_when_accumulating_into_existing_buffer():
    dlogits = torch.tensor([[2.0, 1.0], [0.0, 0.0]])
    hidden = torch.tensor([[1.0, 0.0, 3.0], [float("inf"), 0.0, 0.0]])
    row_active = torch.tensor([True, False])
    current = torch.ones(2, 3, dtype=torch.float32)

    grad = accumulate_weight_gradient_(
        current,
        dlogits,
        hidden,
        row_active=row_active,
        overwrite=False,
    )

    assert grad is current  # accumulates in place
    assert torch.isfinite(grad).all()
    expected = torch.ones(2, 3) + dlogits[:1].t() @ hidden[:1]
    assert torch.allclose(grad, expected)


def test_row_mask_neutralizes_nonfinite_dlogits_too():
    dlogits = torch.tensor([[2.0, 1.0], [float("nan"), float("inf")]])
    hidden = torch.tensor([[1.0, 0.0, 3.0], [4.0, 5.0, 6.0]])
    row_active = torch.tensor([True, False])

    grad = accumulate_weight_gradient_(
        _destination(dlogits, hidden),
        dlogits,
        hidden,
        row_active=row_active,
        overwrite=True,
    )

    assert torch.isfinite(grad).all()
    assert torch.equal(grad, dlogits[:1].t() @ hidden[:1])


def test_trusted_zero_dlogits_path_masks_only_hidden_operand(monkeypatch):
    dlogits = torch.tensor([[2.0, 1.0], [0.0, 0.0]])
    hidden = torch.tensor([[1.0, 0.0, 3.0], [float("nan"), 5.0, 6.0]])
    row_active = torch.tensor([True, False])
    real_where = torch.where
    where_calls = 0

    def count_where(*args, **kwargs):
        nonlocal where_calls
        where_calls += 1
        return real_where(*args, **kwargs)

    monkeypatch.setattr(torch, "where", count_where)
    grad = accumulate_weight_gradient_(
        _destination(dlogits, hidden),
        dlogits,
        hidden,
        row_active=row_active,
        overwrite=True,
        dlogits_zero_for_inactive=True,
    )

    assert where_calls == 1
    assert torch.isfinite(grad).all()
    assert torch.equal(grad, dlogits[:1].t() @ hidden[:1])


def test_overwrite_ignores_uninitialized_destination_contents():
    dlogits = torch.randn(4, 3)
    hidden = torch.randn(4, 2)
    destination = torch.full((3, 2), float("nan"))

    grad = accumulate_weight_gradient_(
        destination,
        dlogits,
        hidden,
        row_active=None,
        overwrite=True,
    )

    assert grad is destination
    assert torch.equal(grad, dlogits.t() @ hidden)


# ── Input validation ─────────────────────────────────────────────────────────
def test_operand_dtype_mismatch_is_rejected():
    dlogits = torch.randn(4, 3, dtype=torch.float32)
    hidden = torch.randn(4, 2, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="must share a dtype"):
        accumulate_weight_gradient_(
            torch.empty(3, 2),
            dlogits,
            hidden,
            row_active=None,
            overwrite=True,
        )


def test_destination_shape_mismatch_is_rejected():
    dlogits = torch.randn(4, 3)
    hidden = torch.randn(4, 2)

    with pytest.raises(ValueError, match=r"must have shape \(3, 2\)"):
        accumulate_weight_gradient_(
            torch.empty(2, 3),
            dlogits,
            hidden,
            row_active=None,
            overwrite=True,
        )


# ── Mixed-dtype dispatch ─────────────────────────────────────────────────────
def test_out_dtype_path_uses_beta_zero_then_beta_one(monkeypatch):
    calls: list[tuple[int, torch.dtype, torch.dtype]] = []

    monkeypatch.setattr(_common, "_addmm_supports_out_dtype", lambda *_: True)

    def fake_addmm(
        input_tensor,
        mat1,
        mat2,
        *,
        beta,
        out_dtype,
        out,
    ):
        calls.append((beta, mat1.dtype, mat2.dtype))
        contribution = (mat1.float() @ mat2.float()).to(out_dtype)
        out.copy_(contribution if beta == 0 else input_tensor + contribution)
        return out

    monkeypatch.setattr(torch, "addmm", fake_addmm)
    dlogits = torch.randn(4, 3, dtype=torch.bfloat16)
    hidden = torch.randn(4, 2, dtype=torch.bfloat16)
    destination = _destination(dlogits, hidden, torch.float32)

    pointer = destination.data_ptr()
    accumulate_weight_gradient_(
        destination, dlogits, hidden, row_active=None, overwrite=True
    )
    accumulate_weight_gradient_(
        destination, dlogits, hidden, row_active=None, overwrite=False
    )

    assert destination.data_ptr() == pointer
    assert calls == [
        (0, torch.bfloat16, torch.bfloat16),
        (1, torch.bfloat16, torch.bfloat16),
    ]


def test_fallback_writes_first_contribution_without_float_upcast(monkeypatch):
    monkeypatch.setattr(_common, "_addmm_supports_out_dtype", lambda *_: False)

    products: list[torch.dtype] = []
    real_mm = torch.mm

    def spy_mm(*args, **kwargs):
        result = real_mm(*args, **kwargs)
        products.append(result.dtype)
        return result

    monkeypatch.setattr(torch, "mm", spy_mm)
    dlogits = torch.randn(4, 3, dtype=torch.bfloat16)
    hidden = torch.randn(4, 2, dtype=torch.bfloat16)
    destination = _destination(dlogits, hidden, torch.float32)

    accumulate_weight_gradient_(
        destination, dlogits, hidden, row_active=None, overwrite=True
    )
    expected = real_mm(dlogits.t(), hidden).float()
    assert destination.dtype == torch.float32
    assert torch.equal(destination, expected)

    pointer = destination.data_ptr()
    accumulate_weight_gradient_(
        destination, dlogits, hidden, row_active=None, overwrite=False
    )
    assert destination.data_ptr() == pointer
    assert torch.equal(destination, expected + expected)
    # The contribution stays in the operand dtype; only the buffer is float32.
    assert products == [torch.bfloat16, torch.bfloat16]


def test_fallback_classifies_fixed_workspace_oom(monkeypatch):
    monkeypatch.setattr(_common, "_addmm_supports_out_dtype", lambda *_: False)

    def raise_oom(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")

    monkeypatch.setattr(torch, "mm", raise_oom)
    dlogits = torch.randn(4, 3, dtype=torch.bfloat16)
    hidden = torch.randn(4, 2, dtype=torch.bfloat16)
    destination = _destination(dlogits, hidden, torch.float32)

    with pytest.raises(NonChunkableWorkspaceOOM, match="fixed"):
        accumulate_weight_gradient_(
            destination, dlogits, hidden, row_active=None, overwrite=True
        )


def test_same_dtype_path_matches_reference():
    dlogits = torch.randn(5, 4)
    hidden = torch.randn(5, 3)
    grad = accumulate_weight_gradient_(
        _destination(dlogits, hidden),
        dlogits,
        hidden,
        row_active=None,
        overwrite=True,
    )
    assert torch.equal(grad, dlogits.t() @ hidden)


# ── Cached-gradient scaling ──────────────────────────────────────────────────
def test_cached_gradient_scaling_reuses_storage_without_cast():
    gradient = torch.tensor([1.0, -2.0])
    pointer = gradient.data_ptr()

    scaled = scale_cached_gradient_(gradient, torch.tensor(0.5))

    assert scaled is gradient
    assert scaled.data_ptr() == pointer
    assert torch.equal(scaled, torch.tensor([0.5, -1.0]))


def test_cached_gradient_scaling_casts_after_in_place_scale():
    gradient = torch.tensor([2.0, -4.0], dtype=torch.float32)

    scaled = scale_cached_gradient_(
        gradient,
        torch.tensor(0.25),
        output_dtype=torch.float16,
    )

    assert torch.equal(gradient, torch.tensor([0.5, -1.0]))
    assert scaled is not None and scaled.dtype == torch.float16
    assert torch.equal(scaled, torch.tensor([0.5, -1.0], dtype=torch.float16))


def test_cached_gradient_scaling_preserves_none():
    assert scale_cached_gradient_(None, torch.tensor(1.0)) is None


def test_cached_gradient_claim_reports_repeated_backward():
    class CachedGradientFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(torch.ones_like(value))
            return value.sum()

        @staticmethod
        def backward(ctx, grad_output):
            claim_cached_gradients(ctx, owner="CachedGradientFunction")
            (gradient,) = ctx.saved_tensors
            return scale_cached_gradient_(gradient, grad_output)

    value = torch.ones(2, requires_grad=True)
    loss = CachedGradientFunction.apply(value)
    loss.backward(retain_graph=True)

    with pytest.raises(RuntimeError, match="one backward pass per forward"):
        loss.backward()
