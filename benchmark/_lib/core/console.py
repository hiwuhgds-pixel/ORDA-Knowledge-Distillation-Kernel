from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable, Sequence

from .paths import LOGGING_UTILS_PATH


# ── Logging utilities ────────────────────────────────────────────────────────
def _load_logging_utils():
    spec = importlib.util.spec_from_file_location(
        "_test_logging_utils", LOGGING_UTILS_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load logging utils from {LOGGING_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_logging_utils = _load_logging_utils()
banner_width = _logging_utils.banner_width
hrule = _logging_utils.hrule
print_banner = _logging_utils.print_banner
print_config = _logging_utils.print_config
print_group_header = _logging_utils.print_group_header
safe_log_name = _logging_utils.safe_log_name


# ── Output encoding ──────────────────────────────────────────────────────────
def _configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = getattr(stream, "encoding", "") or ""
        if reconfigure is not None and encoding.lower() != "utf-8":
            reconfigure(encoding="utf-8")


_configure_utf8_output()


# ── Table rendering ──────────────────────────────────────────────────────────
def print_row(indent: int, columns: Iterable[str]) -> None:
    print(" " * indent + "  ".join(columns))


def _is_numeric_text(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped in {"OOM", "NA"}:
        return True
    if stripped.endswith("%"):
        stripped = stripped[:-1]
    if stripped.startswith(("+", "-")):
        stripped = stripped[1:]
    stripped = stripped.replace(",", "")
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    title: str | None = None,
    align_right: set[int] | None = None,
    separate_on: int | None = None,
) -> None:
    align_right = set() if align_right is None else set(align_right)
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    if title:
        print()
        print(f" {title}")
    header_line = " | ".join(
        headers[idx].ljust(widths[idx]) for idx in range(len(headers))
    )
    print(f" {header_line}")
    separator = " " + "─|─".join("─" * width for width in widths)
    print(separator)
    previous_group = None
    for row_index, row in enumerate(rows):
        if (
            separate_on is not None
            and row_index > 0
            and row[separate_on] != previous_group
        ):
            print(separator)
        cells: list[str] = []
        for idx, cell in enumerate(row):
            text = str(cell)
            if idx in align_right or _is_numeric_text(text):
                cells.append(text.rjust(widths[idx]))
            else:
                cells.append(text.ljust(widths[idx]))
        print(" " + " | ".join(cells))
        if separate_on is not None:
            previous_group = row[separate_on]
