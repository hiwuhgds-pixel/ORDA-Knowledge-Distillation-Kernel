from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from logging_utils import (
    configure_utf8_output,
    print_banner,
    print_config,
    print_group_header,
    run_logged,
    test_log,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
_SHARD_ENVIRONMENT_KEYS = (
    "ORDA_CARTESIAN_SHARD_COUNT",
    "ORDA_CARTESIAN_SHARD_INDEX",
    "ORDA_GPU_DIRECT_SHARD_COUNT",
    "ORDA_GPU_DIRECT_SHARD_INDEX",
    "ORDA_VERIFY_DIRECT_SHARD_COUNT",
    "ORDA_CARTESIAN_ROUTES",
    "ORDA_CARTESIAN_KL_VARIANTS",
    "ORDA_CARTESIAN_DTYPES",
    "ORDA_CARTESIAN_API_LEVELS",
    "ORDA_CARTESIAN_CHUNK_POLICIES",
    "ORDA_CARTESIAN_LABEL_PATTERNS",
    "ORDA_CARTESIAN_OUTCOMES",
    "ORDA_CARTESIAN_ORDINALS",
)


# ── Runner state ─────────────────────────────────────────────────────────────
def _device_fields() -> tuple[str, str]:
    return "cpu", "-"


def _unsharded_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in _SHARD_ENVIRONMENT_KEYS:
        env.pop(name, None)
    return env


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    argv = [] if argv is None else list(argv)
    with test_log(ROOT, Path(__file__).stem, device_name="cpu", group=None):
        return _main(argv)


def _main(argv: list[str]) -> int:
    device, gpu = _device_fields()
    print_banner("TEST: CPU suite")
    fields = {
        "device": device,
        "gpu": gpu,
        "scope": "unit, CPU correctness, multiprocess",
    }
    print_config(fields)

    if importlib.util.find_spec("pytest") is None:
        print("pytest is not installed in this Python environment.", file=sys.stderr)
        print(
            "Install pytest, then run: python test/run_test_cpu.py",
            file=sys.stderr,
        )
        return 2

    base_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        str(TEST_ROOT / "unit"),
        str(TEST_ROOT / "correctness" / "test_torch_backend_cpu.py"),
        str(TEST_ROOT / "nightly" / "test_autotune_multiprocess.py"),
        *argv,
    ]
    base_env = _unsharded_environment()
    print_group_header("CPU tests")
    return run_logged(base_cmd, cwd=ROOT, env=base_env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
