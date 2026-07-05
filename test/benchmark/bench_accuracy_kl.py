#!/usr/bin/env python3
"""Loss accuracy benchmark for KL component against fp32 baseline."""

import torch

from src._accuracy import (
    ACCURACY_BATCH_SEQS,
    ACCURACY_VOCAB,
    ORDA_AVAILABLE,
    ORDA_IMPORT_ERROR,
    TEMP,
    accuracy_configs,
    max_errors,
    measure_accuracy_rows,
    print_accuracy_table,
)
from src.helper import benchmark_profile, fmt_precision, get_device, precision_config, print_banner, print_config, script_log


device = get_device()
DTYPE, USE_AMP, USE_GRAD_SCALER = precision_config(device)
PROFILE = benchmark_profile(device)
CONFIGS = accuracy_configs(PROFILE)


def main() -> None:
    print_banner("BENCHMARK: loss accuracy KL")
    print_config({
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "-",
        "dtype": fmt_precision(DTYPE, USE_AMP, USE_GRAD_SCALER),
        "baseline": "fp32",
        "component": "kl",
        "backends": "torch-compile, orda",
        "chunk": "dynamic",
        "configs": ", ".join(f"{batch}x{seq}" for batch, seq in ACCURACY_BATCH_SEQS),
        "vocab": str(ACCURACY_VOCAB),
        "dim": str(PROFILE["student_config"]["dim"]),
        "T": str(TEMP),
    })

    if device.type != "cuda":
        print("[WARN] CUDA is unavailable - triton benchmark skipped")
        return
    if not ORDA_AVAILABLE:
        print(f"[WARN] orda_ce_kernel is unavailable - skipped ({ORDA_IMPORT_ERROR})")
        return

    rows = measure_accuracy_rows(
        component="kl",
        configs=CONFIGS,
        dtype=DTYPE,
        device=device,
        use_amp=USE_AMP,
        use_grad_scaler=USE_GRAD_SCALER,
    )
    print_accuracy_table(rows)
    tc_abs, tc_rel, orda_abs, orda_rel = max_errors(rows)
    print()
    print(f"torch-compile max abs_err: {tc_abs:.6e}")
    print(f"torch-compile max rel_err: {tc_rel:.6e}")
    print(f"orda max abs_err: {orda_abs:.6e}")
    print(f"orda max rel_err: {orda_rel:.6e}")


if __name__ == "__main__":
    with script_log(device, __file__):
        main()
