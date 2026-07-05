#!/usr/bin/env python3
"""
Benchmark: (student CE + KL) — torch.compile (fused full graph) vs ORDA kernel.

2 modes:
  - torch-compile : torch.compile merges model + (student CE + KL) into one compiled region
  - orda          : torch.compile model + distillation_loss (Triton, kl_weight=1)

"""

from itertools import product

import torch

from src.helper import (
    benchmark_profile,
    fmt_mb,
    fmt_ms,
    fmt_precision,
    fmt_vocab,
    get_device,
    precision_config,
    print_banner,
    print_config,
    print_group_header,
    print_row,
    print_table,
    train_log,
)
from src._train import run_mode_isolated

try:
    from orda_ce_kernel import is_available as _orda_is_available
    ORDA_AVAILABLE = _orda_is_available()
except Exception:
    ORDA_AVAILABLE = False

device = get_device()

# ── Configuration ─────────────────────────────────────────────────────────────
PROFILE = benchmark_profile(device)
TEACHER_CFG = PROFILE["teacher_config"]
STUDENT_CFG = PROFILE["student_config"]

VOCABS         = PROFILE["vocabs"]
SEQS           = PROFILE["seqs"]
BATCH          = PROFILE["batch"]
WARMUP         = PROFILE["warmup"]
UPDATE_STEPS   = PROFILE["update_steps"]
PASSES         = PROFILE["passes"]
TEMP           = PROFILE["temp"]
DTYPE, USE_AMP, USE_GRAD_SCALER = precision_config(device)

ORDA_WARN = None
MODES = ["torch-compile"]
if ORDA_AVAILABLE:
    MODES += ["orda"]
else:
    ORDA_WARN = "[WARN] orda_ce_kernel is unavailable - running torch-compile mode only"


def _table_ms(value: float) -> str:
    return "OOM" if value == float("inf") else f"{value:.2f}"


def _table_mb(value: float) -> str:
    return "OOM" if value == float("inf") else f"{value:.1f}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if ORDA_WARN:
        print(ORDA_WARN)
    print_banner("BENCHMARK: Train · SeparateTeacher student")
    print_config({
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "-",
        "dtype": fmt_precision(DTYPE, USE_AMP, USE_GRAD_SCALER),
        "batch": f"{BATCH}   warmup: {WARMUP}   steps: {UPDATE_STEPS}   passes: {PASSES}   T: {TEMP}",
        "teacher": str(TEACHER_CFG),
        "student": str(STUDENT_CFG),
    })

    mode_w = max(len(m) for m in MODES)
    rows: list[tuple] = []

    for vocab, seq in product(VOCABS, SEQS):
        label = f"vocab={fmt_vocab(vocab)}  seq={seq}"
        print_group_header(label)

        pass_orders = [MODES, list(reversed(MODES))]
        pass_results: list[dict[str, tuple[float, float]]] = []

        for p_idx, order in enumerate(pass_orders[:PASSES]):
            direction = "↓ top-down" if p_idx == 0 else "↑ bottom-up"
            print(f"  pass {p_idx + 1}/{PASSES} ({direction})")
            p_res: dict[str, tuple[float, float]] = {}
            for mode in order:
                ms, vram = run_mode_isolated(
                    vocab=vocab,
                    seq=seq,
                    mode=mode,
                    teacher_config=TEACHER_CFG,
                    student_config=STUDENT_CFG,
                    batch=BATCH,
                    warmup=WARMUP,
                    update_steps=UPDATE_STEPS,
                    temp=TEMP,
                    dtype=DTYPE,
                    device=device,
                    use_amp=USE_AMP,
                    use_grad_scaler=USE_GRAD_SCALER,
                    teacher_mode="separate_student",
                )
                p_res[mode] = (ms, vram)
                print_row(4, [f"{mode:>{mode_w}}", f"{fmt_ms(ms)} ms", f"{fmt_mb(vram)} MB"])
            pass_results.append(p_res)

        print("  average")
        for mode in MODES:
            ms_vals = [pr[mode][0] for pr in pass_results]
            vr_vals = [pr[mode][1] for pr in pass_results]
            if any(m == float("inf") for m in ms_vals):
                ms_avg, vr_avg = float("inf"), float("inf")
            else:
                ms_avg = sum(ms_vals) / len(ms_vals)
                vr_avg = sum(vr_vals) / len(vr_vals)
            rows.append((vocab, seq, mode, ms_avg, vr_avg))
            print_row(4, [f"{mode:>{mode_w}}", f"{fmt_ms(ms_avg)} ms", f"{fmt_mb(vr_avg)} MB"])

        # Direct ORDA vs torch-compile comparison when available.
        combo = {m: (ms, vr) for v, s, m, ms, vr in rows if v == vocab and s == seq}
        if ORDA_AVAILABLE and "orda" in combo and "torch-compile" in combo:
            ms_o, vr_o = combo["orda"]
            ms_c, vr_c = combo["torch-compile"]
            if ms_o != float("inf") and ms_c != float("inf"):
                print(
                    f"  Δ orda vs torch-compile   {ms_o - ms_c:+.2f} ms "
                    f"({(ms_o / ms_c - 1) * 100:+.1f}%)   "
                    f"{vr_o - vr_c:+.1f} MB ({(vr_o / vr_c - 1) * 100:+.1f}%)"
                )

    print_banner("SUMMARY")
    summary_rows = [
        [fmt_vocab(vocab), str(seq), mode, _table_ms(ms), _table_mb(vram)]
        for vocab, seq, mode, ms, vram in rows
    ]
    print_table(["vocab", "seq", "mode", "ms/step", "peakVRAM MB"], summary_rows, align_right={0, 1, 3, 4})

    if ORDA_AVAILABLE:
        delta_rows: list[list[str]] = []
        for vocab, seq in product(VOCABS, SEQS):
            tc_entry = next(
                ((ms, vr) for v, s, m, ms, vr in rows
                 if v == vocab and s == seq and m == "torch-compile"),
                None,
            )
            orda_entry = next(
                ((ms, vr) for v, s, m, ms, vr in rows
                 if v == vocab and s == seq and m == "orda"),
                None,
            )
            if tc_entry is None or orda_entry is None:
                continue
            tc_ms, tc_vr = tc_entry
            o_ms,  o_vr  = orda_entry
            if tc_ms == float("inf") or o_ms == float("inf"):
                delta_rows.append([fmt_vocab(vocab), str(seq), "OOM", "OOM", "OOM", "OOM"])
            else:
                delta_rows.append([
                    fmt_vocab(vocab),
                    str(seq),
                    f"{o_ms - tc_ms:+.2f}",
                    f"{(o_ms / tc_ms - 1) * 100:+.1f}%",
                    f"{o_vr - tc_vr:+.1f}",
                    f"{(o_vr / tc_vr - 1) * 100:+.1f}%",
                ])
        print_table(
            ["vocab", "seq", "Δms", "Δms%", "ΔVRAM MB", "ΔVRAM%"],
            delta_rows,
            title="ORDA vs torch-compile",
            align_right={0, 1, 2, 3, 4, 5},
        )


if __name__ == "__main__":
    with train_log(device, __file__):
        main()
