"""One-click runner for the full Orda test+bench gauntlet on Colab/T4.

Layout under ``--output-dir`` (default ``benchmark_results``):

    benchmark_results/
        json/    # JSON artifacts from every benchmark
        csv/     # CSV artifacts from every benchmark
        logs/    # stdout + stderr for every step. One .log file per
                 # step plus a master ``run_all.log``.

Usage on Colab::

    !cd /content/orda_ce_kernel && python scripts/run_all_test_colab.py

Use ``--skip-large`` to drop the ``--mode orda-large`` end-to-end run
when free VRAM is tight.

Every step is captured so failures in one step do not abort the rest of the
gauntlet — the master log records the exit code per step at the end.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_step(name: str, cmd: list[str], *, log_dir: Path, master_log) -> int:
    step_log = log_dir / f"{name}.log"
    header = f"\n{'=' * 80}\n[{_utc_now()}] STEP: {name}\nCMD: {subprocess.list2cmdline(cmd)}\n{'=' * 80}\n"
    print(header, end="")
    master_log.write(header)
    master_log.flush()

    with step_log.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            master_log.write(line)
        rc = proc.wait()
        footer = f"\n[{_utc_now()}] STEP {name} exit={rc}\n"
        print(footer, end="")
        fh.write(footer)
        master_log.write(footer)
        master_log.flush()
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Orda test+bench gauntlet")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Root directory for json/, csv/, logs/.")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip bench_end_to_end --mode orda-large.")
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Skip GPU correctness pytest suite.")
    parser.add_argument("--skip-unit", action="store_true",
                        help="Skip CPU-safe unit pytest suite.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned step list without executing.")
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    json_dir = root / "json"
    csv_dir  = root / "csv"
    log_dir  = root / "logs"
    for d in (json_dir, csv_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    def _bench(name: str, module: str, extra: list[str] | None = None) -> tuple[str, list[str]]:
        cmd = [py, "-m", module,
               "--output-json", str(json_dir / f"{name}.json"),
               "--output-csv",  str(csv_dir  / f"{name}.csv")]
        if extra:
            cmd.extend(extra)
        return (name, cmd)

    steps: list[tuple[str, list[str]]] = []

    steps.append(("gate0_ast_imports", [py, "scripts/check_ast_imports.py"]))

    if not args.skip_unit:
        steps.append(("gate1_unit", [py, "-m", "pytest", "tests/unit", "-q", "-rs"]))

    if not args.skip_correctness:
        steps.append(("gate2_correctness", [py, "-m", "pytest", "tests/correctness", "-q", "-rs"]))

    steps.append(_bench("bench_ce_only",        "tests.benchmarks.bench_ce_only"))
    steps.append(_bench("bench_ce_kl",          "tests.benchmarks.bench_ce_kl"))
    steps.append(_bench("bench_kl_accuracy",    "tests.benchmarks.bench_kl_accuracy"))
    steps.append(_bench("bench_kl_throughput",  "tests.benchmarks.bench_kl_throughput"))
    steps.append(_bench("bench_memory_bandwidth","tests.benchmarks.bench_memory_bandwidth", ["--verify"]))
    steps.append(_bench("bench_end_to_end_compare",    "tests.benchmarks.bench_end_to_end", ["--mode", "compare"]))
    if not args.skip_large:
        steps.append(_bench("bench_end_to_end_orda_large", "tests.benchmarks.bench_end_to_end", ["--mode", "orda-large"]))

    steps.append(("gate5_profile_dry_run",
                  [py, "-m", "tests.benchmarks.profile_wrapper", "--dry-run"]))

    if args.dry_run:
        for name, cmd in steps:
            print(f"[PLAN] {name}: {subprocess.list2cmdline(cmd)}")
        print(f"\n[PLAN] artifacts -> {root}")
        return

    master_path = log_dir / "run_all.log"
    summary: list[tuple[str, int]] = []
    with master_path.open("w", encoding="utf-8") as master_log:
        master_log.write(f"[{_utc_now()}] Orda full gauntlet — output_dir={root}\n")
        master_log.write(f"python={py}  cwd={os.getcwd()}\n")
        master_log.write(f"PATH-style which-pytest: {shutil.which('pytest')}\n\n")
        for name, cmd in steps:
            rc = _run_step(name, cmd, log_dir=log_dir, master_log=master_log)
            summary.append((name, rc))

        master_log.write("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80 + "\n")
        for name, rc in summary:
            tag = "OK" if rc == 0 else f"FAIL(exit={rc})"
            master_log.write(f"  [{tag:>10}] {name}\n")
        master_log.write(f"\n[{_utc_now()}] Done.\n")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, rc in summary:
        tag = "OK" if rc == 0 else f"FAIL(exit={rc})"
        print(f"  [{tag:>10}] {name}")
    print(f"\nArtifacts:")
    print(f"  JSON: {json_dir}")
    print(f"  CSV : {csv_dir}")
    print(f"  Logs: {log_dir}")
    print(f"  Master log: {master_path}")

    sys.exit(0 if all(rc == 0 for _, rc in summary) else 1)


if __name__ == "__main__":
    main()
