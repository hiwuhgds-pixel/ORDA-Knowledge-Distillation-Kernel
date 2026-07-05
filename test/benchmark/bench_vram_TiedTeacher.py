#!/usr/bin/env python3
"""
VRAM benchmark for TiedTeacher CE+KL.

This script measures only backend memory. It directly allocates
student/teacher hidden states and the tied head weight, then compares PyTorch
compiled loss vs ORDA distillation_loss.
"""

from itertools import product

import torch

torch._dynamo.config.recompile_limit = 64

from src.helper import (
    fmt_precision,
    get_device,
    precision_config,
    print_banner,
    print_config,
    print_group_header,
    script_log,
)
from src._vram import (
    BATCH,
    DIMS,
    MEASURE_ITERS,
    NUM_CHUNKS,
    ORDA_AVAILABLE,
    SEQS,
    TEMP,
    VOCABS,
    WARMUP,
    measure_tied_once,
    print_backend_header,
    print_backend_row,
)

device = get_device()
DTYPE, _, _ = precision_config(device)
USE_AMP = False
MODES = ["torch", "torch-compile"]
ORDA_WARN = None
if ORDA_AVAILABLE:
    MODES.append("orda")
else:
    ORDA_WARN = "[WARN] orda_ce_kernel khong kha dung - chi chay torch modes only"


def main():
    if ORDA_WARN:
        print(ORDA_WARN)
    print_banner("BENCHMARK: VRAM backend · TiedTeacher")
    print_config({
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "-",
        "dtype": fmt_precision(DTYPE, USE_AMP, False),
        "batch": f"{BATCH}   T: {TEMP}",
        "num_chunks": str(NUM_CHUNKS),
    })

    for dim in DIMS:
        print_group_header(f"dim={dim}")
        print_backend_header()
        for vocab, seq in product(VOCABS, SEQS):
            config_rows = []
            for mode in MODES:
                result = measure_tied_once(
                    vocab=vocab,
                    seq=seq,
                    dim=dim,
                    mode=mode,
                    batch=BATCH,
                    warmup=WARMUP,
                    measure_iters=MEASURE_ITERS,
                    temp=TEMP,
                    dtype=DTYPE,
                    device=device,
                    use_amp=USE_AMP,
                )
                config_rows.append((vocab, seq, mode, result))
            print_backend_row(
                vocab,
                seq,
                config_rows,
                torch_mode="torch",
                compile_mode="torch-compile",
                orda_mode="orda",
            )


if __name__ == "__main__":
    with script_log(device, __file__):
        main()
