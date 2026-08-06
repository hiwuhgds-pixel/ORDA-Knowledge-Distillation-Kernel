"""Route GPU correctness, autotune, and bounded Cartesian suites."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from logging_utils import (
    configure_utf8_output,
    print_banner,
    print_config,
    print_group_header,
    run_logged,
    test_log,
)
from utils.matrices import (
    API_LEVELS,
    CARTESIAN_ROUTES,
    CARTESIAN_SELECTION_ENVIRONMENT_KEYS,
    CHUNK_POLICIES,
    DTYPES,
    KL_VARIANTS,
    LABEL_PATTERNS,
    TOTAL_CARTESIAN_CASES,
    CartesianRoute,
    ExpectedOutcome,
    cartesian_case_from_ordinal,
    cartesian_route_for_ordinal,
    parse_cartesian_route,
)
from utils.sharding import DIRECT_GPU_TESTS

ROOT = Path(__file__).resolve().parents[1]
CARTESIAN_PATH = ROOT / "test" / "correctness" / "test_cartesian_cuda.py"
GPU_TESTS = {
    **{name: ROOT / path for name, path in DIRECT_GPU_TESTS.items()},
    "autotune": ROOT / "test" / "nightly" / "test_autotune_full_batch_cuda.py",
    **{f"cartesian-{route.route_id}": CARTESIAN_PATH for route in CARTESIAN_ROUTES},
}
_CARTESIAN_OPTIONS = {
    f"cartesian-{route.route_id}": route for route in CARTESIAN_ROUTES
}
_T = TypeVar("_T")


# ── Runner arguments ─────────────────────────────────────────────────────────
def _route_argument(value: str) -> CartesianRoute:
    try:
        return parse_cartesian_route(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _ordinal_argument(value: str) -> int:
    try:
        ordinal = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ordinal must be an integer") from exc
    try:
        cartesian_case_from_ordinal(ordinal)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return ordinal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected GPU test files in sequential pytest subprocesses. "
            "Unknown arguments are forwarded to pytest."
        )
    )
    parser.add_argument(
        "--test",
        action="append",
        choices=tuple(GPU_TESTS),
        help="GPU test to run; repeat to select multiple tests",
    )
    parser.add_argument(
        "--route",
        action="append",
        type=_route_argument,
        metavar="TEACHER",
        help="compatibility shorthand for a Cartesian test option; repeatable",
    )
    parser.add_argument(
        "--kl",
        dest="kl_variant",
        action="append",
        choices=KL_VARIANTS,
        help="KL variant filter; repeat to select multiple variants",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run all nine GPU test options",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list GPU tests and Cartesian routes, then exit",
    )
    parser.add_argument("--dtype", action="append", choices=DTYPES)
    parser.add_argument("--api", action="append", choices=API_LEVELS)
    parser.add_argument("--chunk", action="append", choices=CHUNK_POLICIES)
    parser.add_argument("--labels", action="append", choices=LABEL_PATTERNS)
    parser.add_argument(
        "--outcome",
        action="append",
        choices=tuple(outcome.value for outcome in ExpectedOutcome),
    )
    parser.add_argument(
        "--ordinal",
        action="append",
        type=_ordinal_argument,
        help="exact global ordinal to run; repeat for multiple cases",
    )
    return parser


def _unique(values: Iterable[_T] | None) -> tuple[_T, ...]:
    if values is None:
        return ()
    return tuple(dict.fromkeys(values))


def _set_csv(environment: dict[str, str], name: str, values: Iterable[object]) -> None:
    rendered = ",".join(str(value) for value in values)
    if rendered:
        environment[name] = rendered


# ── Route environment ────────────────────────────────────────────────────────
def _route_environment(
    args,
    route: CartesianRoute,
    dtypes: tuple[str, ...],
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in CARTESIAN_SELECTION_ENVIRONMENT_KEYS:
        environment.pop(name, None)
    environment["ORDA_CARTESIAN_ROUTES"] = route.route_id
    _set_csv(
        environment,
        "ORDA_CARTESIAN_KL_VARIANTS",
        _unique(args.kl_variant),
    )
    _set_csv(environment, "ORDA_CARTESIAN_DTYPES", dtypes)
    _set_csv(environment, "ORDA_CARTESIAN_API_LEVELS", _unique(args.api))
    _set_csv(environment, "ORDA_CARTESIAN_CHUNK_POLICIES", _unique(args.chunk))
    _set_csv(environment, "ORDA_CARTESIAN_LABEL_PATTERNS", _unique(args.labels))
    _set_csv(environment, "ORDA_CARTESIAN_OUTCOMES", _unique(args.outcome))
    _set_csv(environment, "ORDA_CARTESIAN_ORDINALS", _unique(args.ordinal))
    return environment


def _supported_dtypes() -> tuple[str, ...]:
    import torch

    from orda_kernel._runtime import supports_bfloat16

    if supports_bfloat16(torch.device("cuda")):
        return DTYPES
    return tuple(dtype for dtype in DTYPES if dtype != "bfloat16")


def _device_fields() -> tuple[str, str]:
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    return "cuda", gpu


# ── Runner output ────────────────────────────────────────────────────────────
def _print_header(
    *,
    tests: tuple[str, ...] | str,
    routes: str,
    dtypes: tuple[str, ...] = (),
    kl_variants: tuple[str, ...] = (),
) -> None:
    device, gpu = _device_fields()
    print_banner("TEST: GPU suite")
    fields = {
        "device": device,
        "gpu": gpu,
        "tests": tests if isinstance(tests, str) else ", ".join(tests),
    }
    if routes:
        fields["routes"] = routes
        fields["dtype"] = ", ".join(dtypes)
    if kl_variants:
        fields["KL"] = ", ".join(kl_variants)
    print_config(fields)


def _print_unsupported_dtypes(dtypes: tuple[str, ...]) -> None:
    if dtypes:
        print(
            "[cartesian] skipping unsupported dtype(s): " + ", ".join(dtypes),
            flush=True,
        )


def _print_routes(total_cases: int) -> None:
    print(
        f"Cartesian routes ({len(CARTESIAN_ROUTES)} routes; "
        f"{total_cases:,} total cases):"
    )
    width = len(str(len(CARTESIAN_ROUTES)))
    for index, route in enumerate(CARTESIAN_ROUTES, start=1):
        start_percent = (index - 1) * 100 // len(CARTESIAN_ROUTES)
        end_percent = index * 100 // len(CARTESIAN_ROUTES)
        print(
            f" {index:>{width}}. {route.route_id:<27}  {start_percent}-{end_percent}%"
        )


def _print_tests() -> None:
    print(f"GPU tests ({len(GPU_TESTS)} groups):")
    width = len(str(len(GPU_TESTS)))
    for index, name in enumerate(GPU_TESTS, start=1):
        print(f" {index:>{width}}. {name}")


def _parse_selection(
    raw: str,
    choices: tuple[str, ...],
) -> tuple[str, ...]:
    by_name = set(choices)
    selected: list[str] = []
    for token in (
        part for value in raw.split(",") for part in value.strip().split() if part
    ):
        if token.lower() == "all":
            return choices
        if token in by_name:
            selected.append(token)
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end or start < 1 or end > len(choices):
                raise ValueError
            selected.extend(choices[start - 1 : end])
            continue
        index = int(token)
        if index < 1 or index > len(choices):
            raise ValueError
        selected.append(choices[index - 1])
    result = _unique(selected)
    if not result:
        raise ValueError
    return result


def _interactive_tests(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    _print_tests()
    while True:
        try:
            raw = input("\n Select test(s) [1,3-5, name, or all]: ").strip()
        except EOFError:
            parser.error("provide --test, --route, --all, or --list")
        try:
            return _parse_selection(raw, tuple(GPU_TESTS))
        except (ValueError, IndexError):
            print(" Invalid selection")


# ── Suite execution ──────────────────────────────────────────────────────────
def _run_pytest(
    path: Path,
    pytest_args: list[str],
    environment: dict[str, str],
) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        str(path),
        *pytest_args,
    ]
    return run_logged(command, cwd=ROOT, env=environment)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    with test_log(ROOT, Path(__file__).stem):
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args, pytest_args = parser.parse_known_args(argv)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    cartesian_filters = any(
        (
            args.kl_variant,
            args.dtype,
            args.api,
            args.chunk,
            args.labels,
            args.outcome,
            args.ordinal,
        )
    )
    if args.list:
        if any(
            (
                args.test,
                args.route,
                args.all,
                args.dtype,
                args.kl_variant,
                args.api,
                args.chunk,
                args.labels,
                args.outcome,
                args.ordinal,
                pytest_args,
            )
        ):
            parser.error(
                "--list cannot be combined with selections or pytest arguments"
            )
        supported_dtypes = _supported_dtypes()
        total_cases = TOTAL_CARTESIAN_CASES * len(supported_dtypes) // len(DTYPES)
        _print_header(
            tests="available",
            routes="available",
            dtypes=supported_dtypes,
            kl_variants=(),
        )
        _print_tests()
        print()
        _print_routes(total_cases)
        return 0

    selector_count = sum(bool(value) for value in (args.test, args.route, args.all))
    if selector_count > 1:
        parser.error("choose only one of --test, --route, or --all")

    if args.all:
        selected_tests = tuple(GPU_TESTS)
    elif args.test:
        selected_tests = _unique(args.test)
    elif args.route:
        selected_tests = tuple(
            f"cartesian-{route.route_id}" for route in _unique(args.route)
        )
    elif args.ordinal:
        selected_tests = tuple(
            f"cartesian-{route.route_id}"
            for route in _unique(
                cartesian_route_for_ordinal(ordinal) for ordinal in args.ordinal
            )
        )
    else:
        _print_header(tests="select below", routes="", kl_variants=())
        selected_tests = _interactive_tests(parser)

    routes = tuple(
        _CARTESIAN_OPTIONS[name]
        for name in selected_tests
        if name in _CARTESIAN_OPTIONS
    )
    if cartesian_filters and not routes:
        parser.error("Cartesian filters require a cartesian-* test option")
    if args.ordinal:
        ordinal_routes = frozenset(
            cartesian_route_for_ordinal(ordinal) for ordinal in args.ordinal
        )
        if not ordinal_routes.issubset(routes):
            parser.error("--ordinal must belong to a selected Cartesian route")

    effective_dtypes: tuple[str, ...] = ()
    unsupported_dtypes: tuple[str, ...] = ()
    if routes:
        supported_dtypes = _supported_dtypes()
        requested_dtypes = _unique(args.dtype) or DTYPES
        effective_dtypes = tuple(
            dtype for dtype in requested_dtypes if dtype in supported_dtypes
        )
        unsupported_dtypes = tuple(
            dtype for dtype in requested_dtypes if dtype not in supported_dtypes
        )
        if not effective_dtypes:
            parser.error(
                "none of the requested dtypes are supported by the current GPU: "
                + ", ".join(unsupported_dtypes)
            )
    _print_header(
        tests=selected_tests,
        routes=", ".join(route.route_id for route in routes),
        dtypes=effective_dtypes,
        kl_variants=_unique(args.kl_variant),
    )
    _print_unsupported_dtypes(unsupported_dtypes)

    first_failure = 0
    direct_tests = tuple(
        name for name in selected_tests if name not in _CARTESIAN_OPTIONS
    )
    total_runs = len(direct_tests) + len(routes)
    run_index = 0
    base_environment = os.environ.copy()
    for name in direct_tests:
        run_index += 1
        print_group_header(f"test {run_index}/{total_runs} · {name}")
        status = _run_pytest(GPU_TESTS[name], pytest_args, base_environment)
        if status != 0 and first_failure == 0:
            first_failure = status

    for route_index, route in enumerate(routes, start=1):
        run_index += 1
        print_group_header(
            f"test {run_index}/{total_runs} · cartesian route "
            f"{route_index}/{len(routes)} · {route.route_id}"
        )
        status = _run_pytest(
            CARTESIAN_PATH,
            pytest_args,
            _route_environment(args, route, effective_dtypes),
        )
        if status != 0 and first_failure == 0:
            first_failure = status
    return first_failure


if __name__ == "__main__":
    raise SystemExit(main())
