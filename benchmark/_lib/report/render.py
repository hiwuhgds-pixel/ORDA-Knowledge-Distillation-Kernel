from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..core.console import print_table
from ..core.metrics import (
    OOM_TEXT,
    delta,
    fmt_backend_mb,
    fmt_delta_absolute,
    fmt_delta_percent,
    fmt_delta_percent_na,
    fmt_vocab,
    is_oom,
)
from .schema import Record, TableSchema


def print_record_table(schema: TableSchema, records: Sequence[Record]) -> None:
    print_table(
        schema.headers,
        schema.console_rows(records),
        title=schema.title,
        align_right=schema.align_right,
        separate_on=schema.separate_index,
    )


# ── Delta lines and tables ───────────────────────────────────────────────────
@dataclass(frozen=True)
class DeltaMetric:
    key: str
    unit: str
    precision: int
    header: str
    percent_header: str


MS_METRIC = DeltaMetric("ms", "ms", 2, "Δms", "Δms%")
VRAM_METRIC = DeltaMetric("vram", "MB", 1, "ΔVRAM MB", "ΔVRAM%")


def print_delta_line(
    indent: int,
    label: str,
    pairs: Sequence[tuple[DeltaMetric, float, float]],
) -> None:
    """Skip partial delta lines when any metric is OOM."""
    deltas = [
        (metric, delta(baseline, candidate)) for metric, baseline, candidate in pairs
    ]
    if any(not value.defined for _, value in deltas):
        return
    parts = [
        f"{value.absolute:+.{metric.precision}f} {metric.unit} ({value.percent:+.1f}%)"
        for metric, value in deltas
    ]
    print(f"{' ' * indent}Δ {label}   {'   '.join(parts)}")


def delta_table(
    *,
    title: str,
    key_headers: Sequence[str],
    metrics: Sequence[DeltaMetric],
    rows: Sequence[tuple[Sequence[str], Sequence[tuple[float, float]]]],
) -> None:
    """Render every delta cell as OOM when any value in its row is OOM."""
    if not rows:
        return
    headers = [*key_headers]
    for metric in metrics:
        headers.extend((metric.header, metric.percent_header))
    table_rows: list[list[str]] = []
    for keys, pairs in rows:
        cells = [*keys]
        # The timing metric gates the whole row: a mode that OOMs has no
        # comparable VRAM figure either, so the row reads OOM across.
        row_is_oom = any(is_oom(value) for value in pairs[0])
        for metric, (baseline, candidate) in zip(metrics, pairs, strict=True):
            if row_is_oom:
                cells.extend((OOM_TEXT, OOM_TEXT))
                continue
            value = delta(baseline, candidate)
            cells.extend(
                (
                    fmt_delta_absolute(value, precision=metric.precision),
                    fmt_delta_percent(value),
                )
            )
        table_rows.append(cells)
    align_right = set(range(len(headers))) - {
        index for index, header in enumerate(key_headers) if header == "source"
    }
    print_table(headers, table_rows, title=title, align_right=align_right)


# ── VRAM backend grid ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BackendGrid:
    config: int = len("vocab=64k  seq=2048")
    torch: int = len("99999.9 MB")
    compile: int = len("torch-compile")
    orda: int = len("99999.9 MB")
    delta_torch: int = len("deltaTorch")
    delta_compile: int = len("deltaCompile")

    def header(self) -> None:
        print(
            f"{'config':<{self.config}}  "
            f"{'torch':^{self.torch}}  "
            f"{'torch-compile':^{self.compile}}  "
            f"{'orda':^{self.orda}}  "
            f"{'deltaTorch':^{self.delta_torch}}  "
            f"{'deltaCompile':^{self.delta_compile}}"
        )

    def row(
        self,
        config_label: str,
        *,
        torch_mb: float | None,
        compile_mb: float,
        orda_mb: float | None,
    ) -> None:
        torch_text = "NA" if torch_mb is None else fmt_backend_mb(torch_mb)
        if orda_mb is None:
            orda_text = delta_torch_text = delta_compile_text = "NA"
        else:
            orda_text = fmt_backend_mb(orda_mb)
            delta_torch_text = (
                "NA" if torch_mb is None else _backend_delta(torch_mb, orda_mb)
            )
            delta_compile_text = _backend_delta(compile_mb, orda_mb)
        print(
            f"{config_label:<{self.config}}  "
            f"{torch_text:^{self.torch}}  "
            f"{fmt_backend_mb(compile_mb):^{self.compile}}  "
            f"{orda_text:^{self.orda}}  "
            f"{delta_torch_text:^{self.delta_torch}}  "
            f"{delta_compile_text:^{self.delta_compile}}"
        )


def _backend_delta(baseline_mb: float, orda_mb: float) -> str:
    return fmt_delta_percent_na(delta(baseline_mb, orda_mb))


BACKEND_GRID = BackendGrid()


def config_label(vocab: int, seq: int) -> str:
    return f"vocab={fmt_vocab(vocab)}  seq={seq}"
