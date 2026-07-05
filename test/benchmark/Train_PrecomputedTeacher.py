#!/usr/bin/env python3
"""
Benchmark: (student CE + KL) — torch.compile (fused full graph) vs ORDA kernel.

4 modes:
  - torch-compile-logits : torch.compile + cached full teacher logits
  - orda-logits          : ORDA precomputed + cached full teacher logits
  - torch-compile-hidden : torch.compile + cached teacher hidden/weight
  - orda-hidden          : ORDA precomputed + cached teacher hidden/weight

"""

from itertools import product
from pathlib import Path

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
from src._precomputed_worker import run_source_in_worker, source_seed

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
if ORDA_AVAILABLE:
    MODES = ["orda-hidden", "torch-compile-hidden", "orda-logits", "torch-compile-logits"]
else:
    MODES = ["torch-compile-hidden", "torch-compile-logits"]
    ORDA_WARN = "[WARN] orda_ce_kernel is unavailable - running torch-compile modes only"


def _table_ms(value: float) -> str:
    return "OOM" if value == float("inf") else f"{value:.2f}"


def _table_mb(value: float) -> str:
    return "OOM" if value == float("inf") else f"{value:.1f}"


def _mode_backend_and_source(mode: str) -> tuple[str, str]:
    if mode.startswith("torch-compile"):
        backend = "torch-compile"
    elif mode.startswith("orda"):
        backend = "orda"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode.endswith("hidden"):
        source = "hidden_weight"
    elif mode.endswith("logits"):
        source = "logits"
    else:
        raise ValueError(f"Unknown precomputed source in mode: {mode}")
    return backend, source


def _entry(rows: list[tuple], vocab: int, seq: int, mode: str):
    return next(
        ((ms, work, teacher, total) for v, s, m, ms, work, teacher, total in rows
         if v == vocab and s == seq and m == mode),
        None,
    )


def _worker_config(vocab: int, seq: int, source_label: str, source_modes: list[str]) -> dict:
    return {
        "vocab": vocab,
        "seq": seq,
        "source_label": source_label,
        "source_modes": source_modes,
        "seed": source_seed(vocab, seq, BATCH),
        "device": str(device),
        "dtype": str(DTYPE),
        "teacher_config": TEACHER_CFG,
        "student_config": STUDENT_CFG,
        "batch": BATCH,
        "warmup": WARMUP,
        "update_steps": UPDATE_STEPS,
        "passes": PASSES,
        "temp": TEMP,
        "use_amp": USE_AMP,
        "use_grad_scaler": USE_GRAD_SCALER,
    }


def _print_source_result(
    *,
    result: dict,
    rows: list[tuple],
    vocab: int,
    seq: int,
    source_label: str,
    backend_w: int,
    ms_col_w: int,
    mb_col_w: int,
) -> None:
    print(f"  {source_label}")
    print_row(
        4,
        [
            f"{'backend':>{backend_w}}",
            f"{'ms/step':>{ms_col_w}}",
            f"{'peakVRAM':>{mb_col_w}}",
            f"{'cacheTeacher':>{mb_col_w}}",
            f"{'totalVRAM':>{mb_col_w}}",
        ],
    )
    for pass_info in result["passes"]:
        print(f"    pass {pass_info['index']}/{PASSES} ({pass_info['direction']})")
        for row in pass_info["rows"]:
            print_row(
                4,
                [
                    f"{row['backend']:>{backend_w}}",
                    f"{fmt_ms(row['ms'])} ms",
                    f"{fmt_mb(row['work_vram'])} MB",
                    f"{fmt_mb(row['teacher_vram'])} MB",
                    f"{fmt_mb(row['total_vram'])} MB",
                ],
            )

    print("    average")
    for row in result["averages"]:
        rows.append((
            vocab,
            seq,
            row["mode"],
            row["ms"],
            row["work_vram"],
            row["teacher_vram"],
            row["total_vram"],
        ))
        print_row(
            4,
            [
                f"{row['backend']:>{backend_w}}",
                f"{fmt_ms(row['ms'])} ms",
                f"{fmt_mb(row['work_vram'])} MB",
                f"{fmt_mb(row['teacher_vram'])} MB",
                f"{fmt_mb(row['total_vram'])} MB",
            ],
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if ORDA_WARN:
        print(ORDA_WARN)
    print_banner("BENCHMARK: Train · PrecomputedTeacher")
    print_config({
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "-",
        "dtype": fmt_precision(DTYPE, USE_AMP, USE_GRAD_SCALER),
        "batch": f"{BATCH}   warmup: {WARMUP}   steps: {UPDATE_STEPS}   passes: {PASSES}   T: {TEMP}",
        "teacher": str(TEACHER_CFG),
        "student": str(STUDENT_CFG),
    })

    backend_w = max(len(_mode_backend_and_source(mode)[0]) for mode in MODES)
    ms_col_w = len(f"{fmt_ms(0.0)} ms")
    mb_col_w = len(f"{fmt_mb(0.0)} MB")
    rows: list[tuple] = []
    source_groups = [
        ("hidden", [mode for mode in MODES if mode.endswith("hidden")]),
        ("logits", [mode for mode in MODES if mode.endswith("logits")]),
    ]
    benchmark_dir = Path(__file__).resolve().parent

    for vocab, seq in product(VOCABS, SEQS):
        label = f"vocab={fmt_vocab(vocab)}  seq={seq}"
        print_group_header(label)

        for group_idx, (source_label, source_modes) in enumerate(source_groups):
            result = run_source_in_worker(
                _worker_config(vocab, seq, source_label, source_modes),
                cwd=benchmark_dir,
            )
            _print_source_result(
                result=result,
                rows=rows,
                vocab=vocab,
                seq=seq,
                source_label=source_label,
                backend_w=backend_w,
                ms_col_w=ms_col_w,
                mb_col_w=mb_col_w,
            )

            if ORDA_AVAILABLE:
                orda_mode = f"orda-{source_label}"
                compile_mode = f"torch-compile-{source_label}"
                combo = {
                    m: (ms, total)
                    for v, s, m, ms, _, _, total in rows
                    if v == vocab and s == seq
                }
                if orda_mode in combo and compile_mode in combo:
                    ms_o, total_o = combo[orda_mode]
                    ms_c, total_c = combo[compile_mode]
                    if ms_o != float("inf") and ms_c != float("inf"):
                        print(
                            f"    Δ orda vs torch-compile   {ms_o - ms_c:+.2f} ms "
                            f"({(ms_o / ms_c - 1) * 100:+.1f}%)   "
                            f"{total_o - total_c:+.1f} MB ({(total_o / total_c - 1) * 100:+.1f}%)"
                        )

            if group_idx != len(source_groups) - 1:
                print()

        if ORDA_AVAILABLE:
            hidden_entry = _entry(rows, vocab, seq, "orda-hidden")
            logits_entry = _entry(rows, vocab, seq, "orda-logits")
            if hidden_entry is not None and logits_entry is not None:
                ms_h, _, _, total_h = hidden_entry
                ms_l, _, _, total_l = logits_entry
                if ms_h != float("inf") and ms_l != float("inf"):
                    print(
                        f"  Δ orda-hidden vs orda-logits   {ms_h - ms_l:+.2f} ms "
                        f"({(ms_h / ms_l - 1) * 100:+.1f}%)   "
                        f"{total_h - total_l:+.1f} MB ({(total_h / total_l - 1) * 100:+.1f}%)"
                    )

    print_banner("SUMMARY")
    summary_rows = []
    for vocab, seq, mode, ms, work_vram, teacher_vram, total_vram in rows:
        backend, source = _mode_backend_and_source(mode)
        source = "hidden" if source == "hidden_weight" else source
        summary_rows.append([
            fmt_vocab(vocab),
            str(seq),
            source,
            backend,
            _table_ms(ms),
            _table_mb(work_vram),
            _table_mb(teacher_vram),
            _table_mb(total_vram),
        ])
    print_table(
        ["vocab", "seq", "source", "backend", "ms/step", "peakVRAM MB", "cacheTeacher MB", "totalVRAM MB"],
        summary_rows,
        align_right={0, 1, 4, 5, 6, 7},
    )

    if ORDA_AVAILABLE:
        delta_rows: list[list[str]] = []
        for vocab, seq in product(VOCABS, SEQS):
            for source_label in ("hidden", "logits"):
                tc_entry = _entry(rows, vocab, seq, f"torch-compile-{source_label}")
                orda_entry = _entry(rows, vocab, seq, f"orda-{source_label}")
                if tc_entry is None or orda_entry is None:
                    continue
                tc_ms, _, _, tc_total = tc_entry
                o_ms, _, _, o_total = orda_entry
                if tc_ms == float("inf") or o_ms == float("inf"):
                    delta_rows.append([fmt_vocab(vocab), str(seq), source_label, "OOM", "OOM", "OOM", "OOM"])
                else:
                    delta_rows.append([
                        fmt_vocab(vocab),
                        str(seq),
                        source_label,
                        f"{o_ms - tc_ms:+.2f}",
                        f"{(o_ms / tc_ms - 1) * 100:+.1f}%",
                        f"{o_total - tc_total:+.1f}",
                        f"{(o_total / tc_total - 1) * 100:+.1f}%",
                    ])
        print_table(
            ["vocab", "seq", "source", "Δms", "Δms%", "ΔVRAM MB", "ΔVRAM%"],
            delta_rows,
            title="ORDA vs torch-compile",
            align_right={0, 1, 3, 4, 5, 6},
        )

        hidden_vs_logits_rows: list[list[str]] = []
        for vocab, seq in product(VOCABS, SEQS):
            hidden_entry = _entry(rows, vocab, seq, "orda-hidden")
            logits_entry = _entry(rows, vocab, seq, "orda-logits")
            if hidden_entry is None or logits_entry is None:
                continue
            h_ms, _, _, h_total = hidden_entry
            l_ms, _, _, l_total = logits_entry
            if h_ms == float("inf") or l_ms == float("inf"):
                hidden_vs_logits_rows.append([fmt_vocab(vocab), str(seq), "OOM", "OOM", "OOM", "OOM"])
            else:
                hidden_vs_logits_rows.append([
                    fmt_vocab(vocab),
                    str(seq),
                    f"{h_ms - l_ms:+.2f}",
                    f"{(h_ms / l_ms - 1) * 100:+.1f}%",
                    f"{h_total - l_total:+.1f}",
                    f"{(h_total / l_total - 1) * 100:+.1f}%",
                ])
        print_table(
            ["vocab", "seq", "Δms", "Δms%", "ΔVRAM MB", "ΔVRAM%"],
            hidden_vs_logits_rows,
            title="ORDA hidden vs ORDA logits",
            align_right={0, 1, 2, 3, 4, 5},
        )


if __name__ == "__main__":
    with train_log(device, __file__):
        main()
