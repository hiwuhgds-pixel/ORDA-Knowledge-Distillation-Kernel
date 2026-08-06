from __future__ import annotations

import torch

from ..config.availability import ORDA_AVAILABLE, precomputed_train_modes
from ..config.profiles import TrainProfile
from ..core.console import print_group_header, print_row
from ..core.metrics import fmt_mb, fmt_ms, fmt_vocab
from ..core.paths import BENCHMARK_ROOT
from ..engine.selection import BenchmarkCase
from ..engine.train_runner import (
    PRECOMPUTED_TRAIN_SCHEMA,
    print_train_header,
    print_train_summary,
)
from ..process.worker import run_source_in_worker
from ..report.render import (
    MS_METRIC,
    VRAM_METRIC,
    delta_table,
    print_delta_line,
)
from ..report.schema import RecordTable
from .specs import precomputed_mode_parts, stable_seed

SOURCE_LABELS = ("hidden", "logits")


# ── Worker configuration ─────────────────────────────────────────────────────
def _worker_config(
    *,
    profile: TrainProfile,
    vocab: int,
    seq: int,
    source_label: str,
    source_modes: list[str],
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
    use_grad_scaler: bool,
    kl_divergence: str | None,
    cache_kind: str,
) -> dict:
    config = {
        "vocab": vocab,
        "seq": seq,
        "source_label": source_label,
        "source_modes": source_modes,
        "seed": stable_seed(
            "precomputed-cache",
            cache_kind,
            source_label,
            vocab,
            seq,
            profile.batch,
        ),
        "device": str(device),
        "dtype": str(dtype),
        "teacher_config": profile.teacher_config.as_dict(),
        "student_config": profile.student_config.as_dict(),
        "batch": profile.batch,
        "warmup": profile.warmup,
        "update_steps": profile.update_steps,
        "passes": profile.passes,
        "temp": profile.temperature,
        "use_amp": use_amp,
        "use_grad_scaler": use_grad_scaler,
        "cache_kind": cache_kind,
    }
    if kl_divergence is not None:
        config["kl_divergence"] = kl_divergence
    return config


def _print_source_result(
    *,
    result: dict,
    table: RecordTable,
    vocab: int,
    seq: int,
    source_label: str,
    widths: tuple[int, int, int],
    passes: int,
) -> None:
    backend_w, ms_w, mb_w = widths
    print(f"  {source_label}")
    print_row(
        4,
        [
            f"{'backend':>{backend_w}}",
            f"{'ms/step':>{ms_w}}",
            f"{'peakVRAM':>{mb_w}}",
            f"{'cacheTeacher':>{mb_w}}",
            f"{'totalVRAM':>{mb_w}}",
        ],
    )
    for pass_info in result["passes"]:
        print(f"    pass {pass_info['index']}/{passes} ({pass_info['direction']})")
        for row in pass_info["rows"]:
            _print_metric_row(row, backend_w)

    print("    average")
    for row in result["averages"]:
        backend, source = precomputed_mode_parts(row["mode"])
        table.add(
            vocab=vocab,
            seq=seq,
            mode=row["mode"],
            source=source,
            backend=backend,
            ms=row["ms"],
            work_vram=row["work_vram"],
            teacher_vram=row["teacher_vram"],
            total_vram=row["total_vram"],
        )
        _print_metric_row(row, backend_w)


def _print_metric_row(row: dict, backend_w: int) -> None:
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


# ── Case execution ───────────────────────────────────────────────────────────
def precomputed_train_case(
    *,
    case_id: str,
    divergence: str,
    label: str = "PrecomputedTeacher",
    cache_kind: str = "standard",
    source_labels: tuple[str, ...] = SOURCE_LABELS,
    profile: TrainProfile,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
    use_grad_scaler: bool,
) -> BenchmarkCase:
    def run():
        modes, orda_warn = precomputed_train_modes()
        kl_divergence = None if divergence == "forward" else divergence
        print_train_header(
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            use_grad_scaler=use_grad_scaler,
            case_label=f"{label} · {divergence}",
            batch=profile.batch,
            warmup=profile.warmup,
            update_steps=profile.update_steps,
            passes=profile.passes,
            temperature=profile.temperature,
            teacher=(str(profile.teacher_config) if cache_kind == "standard" else None),
            student=str(profile.student_config),
            cache_kind=cache_kind,
            kl_divergence=kl_divergence,
            orda_warn=orda_warn,
        )

        backend_w = max(len(precomputed_mode_parts(mode)[0]) for mode in modes)
        widths = (
            backend_w,
            len(f"{fmt_ms(0.0)} ms"),
            len(f"{fmt_mb(0.0)} MB"),
        )
        table = RecordTable(PRECOMPUTED_TRAIN_SCHEMA)
        source_groups = [
            (source, [mode for mode in modes if mode.endswith(source)])
            for source in source_labels
        ]

        for vocab in profile.vocabs:
            for seq in profile.seqs:
                print_group_header(f"vocab={fmt_vocab(vocab)}  seq={seq}")
                for index, (source_label, source_modes) in enumerate(source_groups):
                    result = run_source_in_worker(
                        _worker_config(
                            profile=profile,
                            vocab=vocab,
                            seq=seq,
                            source_label=source_label,
                            source_modes=source_modes,
                            device=device,
                            dtype=dtype,
                            use_amp=use_amp,
                            use_grad_scaler=use_grad_scaler,
                            kl_divergence=kl_divergence,
                            cache_kind=cache_kind,
                        ),
                        cwd=BENCHMARK_ROOT,
                    )
                    _print_source_result(
                        result=result,
                        table=table,
                        vocab=vocab,
                        seq=seq,
                        source_label=source_label,
                        widths=widths,
                        passes=profile.passes,
                    )
                    _print_source_delta(table, vocab, seq, source_label)
                    if index != len(source_groups) - 1:
                        print()

                _print_hidden_vs_logits_line(table, vocab, seq)

        print_train_summary(table)
        _print_delta_tables(table, profile, source_labels)
        return PRECOMPUTED_TRAIN_SCHEMA, table.records

    return BenchmarkCase(case_id, f"{label} · {divergence}", run)


# ── Delta reports ────────────────────────────────────────────────────────────
def _entry(table: RecordTable, vocab: int, seq: int, mode: str):
    rows = table.filter(vocab=vocab, seq=seq, mode=mode)
    return rows[0] if rows else None


def _print_source_delta(table: RecordTable, vocab: int, seq: int, source: str) -> None:
    if not ORDA_AVAILABLE:
        return
    orda = _entry(table, vocab, seq, f"orda-{source}")
    compiled = _entry(table, vocab, seq, f"torch-compile-{source}")
    if orda is None or compiled is None:
        return
    print_delta_line(
        4,
        "orda vs torch-compile",
        [
            (MS_METRIC, compiled["ms"], orda["ms"]),
            (VRAM_METRIC, compiled["total_vram"], orda["total_vram"]),
        ],
    )


def _print_hidden_vs_logits_line(table: RecordTable, vocab: int, seq: int) -> None:
    if not ORDA_AVAILABLE:
        return
    hidden = _entry(table, vocab, seq, "orda-hidden")
    logits = _entry(table, vocab, seq, "orda-logits")
    if hidden is None or logits is None:
        return
    print_delta_line(
        2,
        "orda-hidden vs orda-logits",
        [
            (MS_METRIC, logits["ms"], hidden["ms"]),
            (VRAM_METRIC, logits["total_vram"], hidden["total_vram"]),
        ],
    )


def _print_delta_tables(
    table: RecordTable, profile: TrainProfile, source_labels: tuple[str, ...]
) -> None:
    if not ORDA_AVAILABLE:
        return
    rows = []
    for vocab in profile.vocabs:
        for seq in profile.seqs:
            for source in source_labels:
                compiled = _entry(table, vocab, seq, f"torch-compile-{source}")
                orda = _entry(table, vocab, seq, f"orda-{source}")
                if compiled is None or orda is None:
                    continue
                rows.append(
                    (
                        [fmt_vocab(vocab), str(seq), source],
                        [
                            (compiled["ms"], orda["ms"]),
                            (compiled["total_vram"], orda["total_vram"]),
                        ],
                    )
                )
    delta_table(
        title="ORDA vs torch-compile",
        key_headers=["vocab", "seq", "source"],
        metrics=[MS_METRIC, VRAM_METRIC],
        rows=rows,
    )

    hidden_rows = []
    for vocab in profile.vocabs:
        for seq in profile.seqs:
            hidden = _entry(table, vocab, seq, "orda-hidden")
            logits = _entry(table, vocab, seq, "orda-logits")
            if hidden is None or logits is None:
                continue
            hidden_rows.append(
                (
                    [fmt_vocab(vocab), str(seq)],
                    [
                        (logits["ms"], hidden["ms"]),
                        (logits["total_vram"], hidden["total_vram"]),
                    ],
                )
            )
    delta_table(
        title="ORDA hidden vs ORDA logits",
        key_headers=["vocab", "seq"],
        metrics=[MS_METRIC, VRAM_METRIC],
        rows=hidden_rows,
    )
