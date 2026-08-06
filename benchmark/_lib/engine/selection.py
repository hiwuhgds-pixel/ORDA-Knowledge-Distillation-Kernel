from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from ..core.console import print_banner
from ..report.schema import TableSchema
from ..report.sink import case_log, write_csv

CaseResult = tuple[TableSchema | None, Sequence[dict] | None]
CaseRunner = Callable[[], CaseResult]
CaseFactory = Callable[[int | None], tuple[torch.device, Sequence["BenchmarkCase"]]]


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    label: str
    run: CaseRunner


# ── Case selection ───────────────────────────────────────────────────────────
def _selection_tokens(values: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(token for token in re.split(r"[\s,]+", value.strip()) if token)
    return tokens


def _resolve_selection(
    tokens: Sequence[str],
    cases: Sequence[BenchmarkCase],
) -> list[BenchmarkCase]:
    by_id = {case.id: case for case in cases}
    selected: list[BenchmarkCase] = []
    selected_ids: set[str] = set()

    def add(case: BenchmarkCase) -> None:
        if case.id not in selected_ids:
            selected.append(case)
            selected_ids.add(case.id)

    for token in tokens:
        if token.lower() == "all":
            for case in cases:
                add(case)
            continue
        if token in by_id:
            add(by_id[token])
            continue
        range_match = re.fullmatch(r"(\d+)-(\d+)", token)
        if range_match:
            start, end = (int(value) for value in range_match.groups())
            if start > end or start < 1 or end > len(cases):
                raise ValueError(f"invalid case range: {token}")
            for index in range(start - 1, end):
                add(cases[index])
            continue
        if token.isdigit():
            index = int(token)
            if index < 1 or index > len(cases):
                raise ValueError(f"invalid case number: {token}")
            add(cases[index - 1])
            continue
        raise ValueError(f"unknown case: {token}")
    return selected


# ── Interactive selection ────────────────────────────────────────────────────
def _print_case_list(title: str, cases: Sequence[BenchmarkCase]) -> None:
    print_banner(title)
    width = len(str(len(cases)))
    id_width = max(len(case.id) for case in cases)
    for index, case in enumerate(cases, start=1):
        print(f" {index:>{width}}. {case.id:<{id_width}}  {case.label}")


def _interactive_selection(
    parser: argparse.ArgumentParser,
    title: str,
    cases: Sequence[BenchmarkCase],
) -> list[BenchmarkCase]:
    _print_case_list(title, cases)
    while True:
        try:
            value = input("\n Select case(s) [1,3,5-7 or all]: ").strip()
        except EOFError:
            parser.error("provide --select or --all")
        try:
            selected = _resolve_selection(_selection_tokens([value]), cases)
        except ValueError as exc:
            print(f" Invalid selection: {exc}")
            continue
        if selected:
            return selected
        print(" Invalid selection: choose at least one case")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


# ── Router execution ─────────────────────────────────────────────────────────
def run_router(
    *,
    router: str,
    title: str,
    case_factory: CaseFactory,
    supports_batch: bool = False,
    argv: Sequence[str] | None = None,
    preflight: Callable[[torch.device], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument(
        "--select",
        action="append",
        nargs="+",
        default=[],
        metavar="LIST",
        help="case IDs, numbers, or ranges; for example tied_forward 1,3,5-7",
    )
    parser.add_argument("--all", action="store_true", help="run every case")
    parser.add_argument("--list", action="store_true", help="list case IDs and exit")
    if supports_batch:
        parser.add_argument(
            "--batch",
            type=_positive_int,
            metavar="N",
            help="override the profile batch size",
        )
    args = parser.parse_args(argv)

    device, cases = case_factory(args.batch if supports_batch else None)

    if args.list:
        if args.select or args.all:
            parser.error("--list cannot be combined with a case selection")
        _print_case_list(title, cases)
        return 0

    tokens = [value for values in args.select for value in values]
    if args.all:
        if tokens:
            parser.error("--all cannot be combined with another case selection")
        selected = list(cases)
    elif tokens:
        try:
            selected = _resolve_selection(_selection_tokens(tokens), cases)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        selected = _interactive_selection(parser, title, cases)

    if selected and preflight is not None:
        preflight(device)

    for index, case in enumerate(selected):
        if index:
            print()
        with case_log(device, router, case.id) as csv_path:
            schema, records = case.run()
            if schema is not None and records is not None:
                write_csv(csv_path, schema.csv_headers, schema.csv_rows(records))
            elif schema is None and records is None:
                csv_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    "benchmark case must return both a table schema and records"
                )
    return 0
