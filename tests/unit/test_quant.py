from __future__ import annotations

import torch

from orda_ce_kernel.ops.quant import (
    dequantize_grad_w,
    dequantize_rowwise_int8,
    quantize_grad_w,
    quantize_rowwise_int8,
    quantize_rowwise_int8_stochastic,
)


def test_rowwise_int8_quantization_shapes_and_similarity():
    torch.manual_seed(7)
    x = torch.randn(64, 128, dtype=torch.float32)
    q, scale = quantize_rowwise_int8(x.clone())
    x_hat = dequantize_rowwise_int8(q, scale)

    assert q.dtype == torch.int8
    assert q.shape == x.shape
    assert scale.shape == (x.shape[0], 1)
    assert torch.nn.functional.cosine_similarity(x.flatten(), x_hat.flatten(), dim=0) > 0.999


def test_stochastic_quantization_accepts_generator():
    torch.manual_seed(7)
    x = torch.randn(8, 16, dtype=torch.float32)
    generator = torch.Generator().manual_seed(123)
    q, scale = quantize_rowwise_int8_stochastic(x.clone(), generator=generator)

    assert q.dtype == torch.int8
    assert scale.shape == (8, 1)


def test_stochastic_quantization_is_reproducible_with_generator_seed():
    x = torch.linspace(-1, 1, 32, dtype=torch.float32).reshape(4, 8)
    q1, scale1 = quantize_rowwise_int8_stochastic(x.clone(), generator=torch.Generator().manual_seed(99))
    q2, scale2 = quantize_rowwise_int8_stochastic(x.clone(), generator=torch.Generator().manual_seed(99))

    assert torch.equal(q1, q2)
    assert torch.equal(scale1, scale2)


def test_rowwise_int8_quantization_handles_zero_rows():
    x = torch.zeros(3, 5, dtype=torch.float32)
    q, scale = quantize_rowwise_int8(x.clone())
    x_hat = dequantize_rowwise_int8(q, scale)

    assert torch.isfinite(scale).all()
    assert torch.equal(q, torch.zeros_like(q))
    assert torch.equal(x_hat, x)


def test_grad_w_quantization_restores_target_rows_exactly():
    torch.manual_seed(7)
    grad_w = torch.randn(16, 8, dtype=torch.float32)
    target = torch.tensor([0, 3, 3, 7, -100, 10], dtype=torch.long)

    q, scale, target_rows, unique_targets = quantize_grad_w(grad_w.clone(), target, -100)
    restored = dequantize_grad_w(q, scale, target_rows, unique_targets, torch.tensor(1.0))

    assert torch.equal(unique_targets, torch.tensor([0, 3, 7, 10]))
    assert torch.allclose(restored[unique_targets], grad_w[unique_targets])


def test_grad_w_dequantization_accepts_tensor_grad_output():
    grad_w = torch.randn(8, 4, dtype=torch.float32)
    target = torch.tensor([1, 3, -100], dtype=torch.long)
    grad_output = torch.full_like(grad_w, 0.25)

    q, scale, target_rows, unique_targets = quantize_grad_w(grad_w.clone(), target, -100)
    restored = dequantize_grad_w(q, scale, target_rows, unique_targets, grad_output)

    assert restored.shape == grad_w.shape
    assert torch.allclose(restored[unique_targets], grad_w[unique_targets] * 0.25)


