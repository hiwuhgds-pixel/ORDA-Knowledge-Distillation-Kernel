from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

import torch

from ..config.availability import ORDA_AVAILABLE
from ..core.console import (
    print_banner,
    print_config,
    print_group_header,
    print_row,
)
from ..core.device import gpu_name
from ..core.metrics import (
    fmt_mb,
    fmt_ms,
    fmt_precision,
    fmt_table_mb,
    fmt_table_ms,
    fmt_vocab,
)
from ..report.render import (
    MS_METRIC,
    VRAM_METRIC,
    DeltaMetric,
    delta_table,
    print_delta_line,
    print_record_table,
)
from ..report.schema import Column, RecordTable, TableSchema
from .train import TrainMeasurement, run_passes

# ── Train schemas ────────────────────────────────────────────────────────────
TRAIN_SCHEMA = TableSchema(
    (
        Column("vocab", "vocab", align_right=True, fmt=fmt_vocab),
        Column("seq", "seq", align_right=True),
        Column("mode", "mode"),
        Column(
            "ms",
            "ms/step",
            align_right=True,
            fmt=fmt_table_ms,
            csv_header="ms_per_step",
        ),
        Column(
            "vram",
            "peakVRAM MB",
            align_right=True,
            fmt=fmt_table_mb,
            csv_header="peak_vram_mb",
        ),
    )
)

PRECOMPUTED_TRAIN_SCHEMA = TableSchema(
    (
        Column("vocab", "vocab", align_right=True, fmt=fmt_vocab),
        Column("seq", "seq", align_right=True),
        Column("mode", "mode"),
        Column("source", "source"),
        Column("backend", "backend"),
        Column(
            "ms",
            "ms/step",
            align_right=True,
            fmt=fmt_table_ms,
            csv_header="ms_per_step",
        ),
        Column(
            "work_vram",
            "peakVRAM MB",
            align_right=True,
            fmt=fmt_table_mb,
            csv_header="work_vram_mb",
        ),
        Column(
            "teacher_vram",
            "cacheTeacher MB",
            align_right=True,
            fmt=fmt_table_mb,
            csv_header="cache_teacher_mb",
        ),
        Column(
            "total_vram",
            "totalVRAM MB",
            align_right=True,
            fmt=fmt_table_mb,
            csv_header="total_vram_mb",
        ),
    )
)


# ── Shape sweep ──────────────────────────────────────────────────────────────
def print_train_header(
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
    use_grad_scaler: bool,
    case_label: str,
    batch: int,
    warmup: int,
    update_steps: int,
    passes: int,
    temperature: float | None = None,
    teacher: str | None = None,
    student: str | None = None,
    kl_divergence: str | None = None,
    objective: str | None = None,
    cache_kind: str | None = None,
    orda_warn: str | None = None,
) -> None:
    if orda_warn:
        print(orda_warn)
    print_banner("BENCHMARK: train")
    batch_line = (
        f"{batch}   warmup: {warmup}   steps: {update_steps}   passes: {passes}"
    )
    if temperature is not None:
        batch_line += f"   T: {temperature}"
    config = {
        "device": str(device),
        "gpu": gpu_name(device),
        "dtype": fmt_precision(dtype, use_amp, use_grad_scaler),
        "batch": batch_line,
    }
    if teacher is not None:
        config["teacher"] = teacher
    if student is not None:
        config["student"] = student
    if cache_kind is not None and cache_kind != "standard":
        config["cache"] = cache_kind
        config["scope"] = "cached loss/backward; cache build excluded"
    config["case"] = case_label
    if objective is not None:
        config["objective"] = objective
    if kl_divergence is not None:
        config["kl_divergence"] = kl_divergence
    print_config(config)


def sweep_shapes(
    *,
    vocabs: Sequence[int],
    seqs: Sequence[int],
    modes: Sequence[str],
    passes: int,
    measure: Callable[[int, int, str, int], TrainMeasurement],
) -> RecordTable:
    table = RecordTable(TRAIN_SCHEMA)
    mode_width = max(len(mode) for mode in modes)

    for vocab, seq in product(vocabs, seqs):
        print_group_header(f"vocab={fmt_vocab(vocab)}  seq={seq}")

        def run(mode: str, pass_index: int, *, vocab=vocab, seq=seq):
            return measure(vocab, seq, mode, pass_index)

        averages = run_passes(
            modes,
            passes,
            run,
            on_pass_start=lambda index, direction: print(
                f"  pass {index}/{passes} ({direction})"
            ),
            on_result=lambda mode, result: print_row(
                4,
                [
                    f"{mode:>{mode_width}}",
                    f"{fmt_ms(result.ms)} ms",
                    f"{fmt_mb(result.work_vram_mb)} MB",
                ],
            ),
            on_averages_start=lambda: print("  average"),
            on_average=lambda mode, values: print_row(
                4,
                [
                    f"{mode:>{mode_width}}",
                    f"{fmt_ms(values['ms'])} ms",
                    f"{fmt_mb(values['work_vram_mb'])} MB",
                ],
            ),
        )
        for mode in modes:
            table.add(
                vocab=vocab,
                seq=seq,
                mode=mode,
                ms=averages[mode]["ms"],
                vram=averages[mode]["work_vram_mb"],
            )
        _print_inline_delta(averages, modes)

    return table


# ── Summary rendering ────────────────────────────────────────────────────────
def _print_inline_delta(averages: dict, modes: Sequence[str]) -> None:
    if not (ORDA_AVAILABLE and "orda" in averages and "torch-compile" in averages):
        return
    orda = averages["orda"]
    compiled = averages["torch-compile"]
    print_delta_line(
        2,
        "orda vs torch-compile",
        [
            (MS_METRIC, compiled["ms"], orda["ms"]),
            (VRAM_METRIC, compiled["work_vram_mb"], orda["work_vram_mb"]),
        ],
    )


def print_train_summary(table: RecordTable) -> None:
    print_banner("SUMMARY")
    print_record_table(table.schema, table.records)


def print_orda_delta_table(
    table: RecordTable,
    *,
    vocabs: Sequence[int],
    seqs: Sequence[int],
    baseline_mode: str = "torch-compile",
    candidate_mode: str = "orda",
    title: str = "ORDA vs torch-compile",
    metrics: Sequence[DeltaMetric] = (MS_METRIC, VRAM_METRIC),
    metric_keys: Sequence[str] = ("ms", "vram"),
) -> None:
    if not ORDA_AVAILABLE:
        return
    rows = []
    for vocab, seq in product(vocabs, seqs):
        baseline = table.filter(vocab=vocab, seq=seq, mode=baseline_mode)
        candidate = table.filter(vocab=vocab, seq=seq, mode=candidate_mode)
        if not baseline or not candidate:
            continue
        rows.append(
            (
                [fmt_vocab(vocab), str(seq)],
                [(baseline[0][key], candidate[0][key]) for key in metric_keys],
            )
        )
    delta_table(
        title=title,
        key_headers=["vocab", "seq"],
        metrics=metrics,
        rows=rows,
    )
