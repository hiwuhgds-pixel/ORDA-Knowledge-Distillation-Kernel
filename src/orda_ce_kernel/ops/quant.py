import torch

# ── Quantize / Dequantize Row-wise ────────────────────────────────────────────

def quantize_rowwise_int8(tensor):
    """Deterministic INT8 quantization (banker's rounding)."""
    scale     = tensor.abs().amax(dim=1, keepdim=True).clamp_min_(1e-12).div_(127.0)
    scaled    = tensor / scale
    scaled.round_().clamp_(-127, 127)
    quantized = scaled.to(torch.int8)
    return quantized, scale


def quantize_rowwise_int8_stochastic(tensor, generator=None):
    """Stochastic INT8 quantization. Unbiased in expectation: E[Q(x)] = x."""
    scale     = tensor.abs().amax(dim=1, keepdim=True).clamp_min_(1e-12).div_(127.0)
    scaled    = tensor / scale
    floor_val = scaled.floor()
    scaled.sub_(floor_val)                           # scaled ← frac (in-place)
    if generator is not None:
        rand  = torch.rand(scaled.shape, generator=generator,
                           dtype=scaled.dtype, device=scaled.device)
    else:
        rand  = torch.rand_like(scaled)               # uniform[0, 1)
    floor_val.add_((scaled > rand).to(floor_val.dtype))
    floor_val.clamp_(-127, 127)
    quantized = floor_val.to(torch.int8)
    return quantized, scale


def dequantize_rowwise_int8(quantized, scale):
    return quantized.to(scale.dtype) * scale


# ── Grad-W Quant / Dequant Ops ────────────────────────────────────────────────

def quantize_grad_w(grad_W, target, ignore_index, quantize_fn=None):
    """Quantize grad_W to INT8; store target rows in FP16 exact."""
    if quantize_fn is None:
        quantize_fn = quantize_rowwise_int8
    unique_targets = torch.unique(target)
    unique_targets = unique_targets[unique_targets != ignore_index]
    grad_W_target  = grad_W[unique_targets].clone()
    grad_W_q, grad_W_scale = quantize_fn(grad_W)
    del grad_W
    return grad_W_q, grad_W_scale, grad_W_target, unique_targets


def dequantize_grad_w(grad_W_a, grad_W_scale, grad_W_target, unique_targets, grad_output):
    """Dequantize INT8 grad_W, restore target rows in FP16 exact, apply grad_output."""
    if grad_output.ndim == 0:
        combined_scale = grad_W_scale * grad_output
        grad_W         = grad_W_a.to(combined_scale.dtype) * combined_scale
        grad_W[unique_targets] = grad_W_target.to(grad_W.dtype) * grad_output
    else:
        grad_W         = dequantize_rowwise_int8(grad_W_a, grad_W_scale)
        grad_W[unique_targets] = grad_W_target.to(grad_W.dtype)
        grad_W         = grad_W * grad_output
    return grad_W
