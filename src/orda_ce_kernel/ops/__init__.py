from .kernels import _exact_ce_fwdbwd_kernel_merged
from .quant import (
    quantize_rowwise_int8,
    quantize_rowwise_int8_stochastic,
    dequantize_rowwise_int8,
    quantize_grad_w,
    dequantize_grad_w,
)
from .cross_entropy import distill_cross_entropy, DistillCEFunction, DistillCrossEntropyLoss
from .kl_kernel import kl_from_logits_chunk, add_kl_grad_to_logits_chunk_
