"""Keep sweep records as the single source for console and CSV output."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product

import torch

from ..config.profiles import VramProfile
from ..core.console import print_banner, print_config, print_group_header
from ..core.device import gpu_name, seed_all
from ..core.metrics import fmt_precision
from ..report.render import BACKEND_GRID, config_label
from ..report.schema import Column, RecordTable, TableSchema
from ..suites.specs import stable_seed
from .vram import VramMeasurement, VramWorkload, measure_once, validate_shape

# ── VRAM schemas ─────────────────────────────────────────────────────────────
_BASE_COLUMNS = (
    Column("dim", "dim", align_right=True),
    Column("teacher_dim", "teacher_dim", align_right=True),
    Column("vocab", "vocab", align_right=True),
    Column("seq", "seq", align_right=True),
    Column("mode", "mode"),
    Column("status", "status"),
    Column("backend_extra_mb", "backend_extra_mb", align_right=True),
)

VRAM_SCHEMA = TableSchema(_BASE_COLUMNS)

PRECOMPUTED_VRAM_SCHEMA = TableSchema(
    (
        *_BASE_COLUMNS,
        Column("cache_teacher_mb", "cache_teacher_mb", align_right=True),
        Column("total_extra_mb", "total_extra_mb", align_right=True),
    )
)


@dataclass(frozen=True)
class VramSweep:
    profile: VramProfile
    device: torch.device
    dtype: torch.dtype
    modes: Sequence[str]
    make_workload: Callable[..., VramWorkload]
    seed_parts: Callable[..., tuple]
    torch_mode: str
    compile_mode: str
    orda_mode: str
    dim_pairs: Sequence[tuple[int, int]]
    reports_cache: bool = False
    reports_teacher_dim: bool = True


# ── Sweep execution ──────────────────────────────────────────────────────────
def run_sweep(sweep: VramSweep) -> RecordTable:
    schema = PRECOMPUTED_VRAM_SCHEMA if sweep.reports_cache else VRAM_SCHEMA
    table = RecordTable(schema)
    profile = sweep.profile

    for dim, teacher_dim in sweep.dim_pairs:
        group = (
            f"student_dim={dim}  teacher_dim={teacher_dim}"
            if teacher_dim != dim
            else f"dim={dim}"
        )
        print_group_header(group)
        BACKEND_GRID.header()

        for vocab, seq in product(profile.vocabs, profile.seqs):
            results: dict[str, VramMeasurement] = {}
            for mode in sweep.modes:
                seed_all(
                    sweep.device,
                    stable_seed(
                        *sweep.seed_parts(
                            dim=dim, teacher_dim=teacher_dim, vocab=vocab, seq=seq
                        )
                    ),
                )
                validate_shape(
                    vocab, seq, profile.batch, profile.warmup, profile.measure_iters
                )
                workload = sweep.make_workload(
                    vocab=vocab,
                    seq=seq,
                    dim=dim,
                    teacher_dim=teacher_dim,
                    mode=mode,
                )
                result = measure_once(
                    workload,
                    warmup=profile.warmup,
                    measure_iters=profile.measure_iters,
                    device=sweep.device,
                )
                results[mode] = result

                record = dict(
                    dim=dim,
                    teacher_dim=teacher_dim if sweep.reports_teacher_dim else "",
                    vocab=vocab,
                    seq=seq,
                    mode=mode,
                    status=result.status,
                    backend_extra_mb=result.backend_extra_mb,
                )
                if sweep.reports_cache:
                    record["cache_teacher_mb"] = result.cache_teacher_mb
                    record["total_extra_mb"] = result.total_extra_mb
                table.add(**record)

            _print_backend_row(sweep, vocab, seq, results)

    return table


# ── Summary rendering ────────────────────────────────────────────────────────
def _print_backend_row(
    sweep: VramSweep,
    vocab: int,
    seq: int,
    results: dict[str, VramMeasurement],
) -> None:
    compile_result = results.get(sweep.compile_mode)
    if compile_result is None:
        return
    torch_result = results.get(sweep.torch_mode)
    orda_result = results.get(sweep.orda_mode)
    BACKEND_GRID.row(
        config_label(vocab, seq),
        torch_mb=None if torch_result is None else torch_result.backend_extra_mb,
        compile_mb=compile_result.backend_extra_mb,
        orda_mb=None if orda_result is None else orda_result.backend_extra_mb,
    )


def print_vram_header(
    *,
    device: torch.device,
    dtype: torch.dtype,
    profile: VramProfile,
    case_label: str,
    temperature: float,
    orda_warn: str | None = None,
    kl_divergence: str | None = None,
    cache_kind: str | None = None,
    unequal_dim_pairs: Sequence[tuple[int, int]] | None = None,
) -> None:
    if orda_warn:
        print(orda_warn)
    print_banner("BENCHMARK: VRAM")
    config = {
        "device": str(device),
        "gpu": gpu_name(device),
        "dtype": fmt_precision(dtype, False, False),
        "batch": f"{profile.batch}   T: {temperature}",
        "case": case_label,
    }
    if kl_divergence is not None:
        config["kl_divergence"] = kl_divergence
    if cache_kind is not None and cache_kind != "standard":
        config["cache"] = cache_kind
        config["scope"] = "cached loss/backward; cache-build peak excluded"
    if unequal_dim_pairs:
        config["hidden_dims"] = ", ".join(
            f"{student}→{teacher}" for student, teacher in unequal_dim_pairs
        )
    print_config(config)
