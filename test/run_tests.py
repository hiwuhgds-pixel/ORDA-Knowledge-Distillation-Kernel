from __future__ import annotations

from datetime import datetime
import importlib.util
import subprocess
import sys
from pathlib import Path

from logging_utils import safe_log_name


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"


def _log_gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def _log_path(script_path: str) -> Path:
    logs_root = ROOT / "logs" / safe_log_name(_log_gpu_name())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(script_path).stem
    path = logs_root / f"{stem}_{stamp}.log"
    suffix = 1
    while path.exists():
        path = logs_root / f"{stem}_{stamp}_{suffix}.log"
        suffix += 1
    return path


def _write_both(log_file, text: str, *, stream=None) -> None:
    stream = sys.stdout if stream is None else stream
    stream.write(text)
    stream.flush()
    log_file.write(text)
    log_file.flush()


def _run_logged(cmd: list[str], log_file) -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _write_both(log_file, line)
    return proc.wait()


def main(argv: list[str] | None = None) -> int:
    argv = [] if argv is None else list(argv)
    path = _log_path(__file__)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", buffering=1) as log_file:
        return _main(argv, log_file)


def _main(argv: list[str], log_file) -> int:

    if importlib.util.find_spec("pytest") is None:
        _write_both(log_file, "pytest is not installed in this Python environment.\n", stream=sys.stderr)
        _write_both(log_file, "Install pytest, then run: python test/run_tests.py\n", stream=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TEST_ROOT / "unit"),
        str(TEST_ROOT / "correctness"),
        "--ignore",
        str(TEST_ROOT / "benchmark"),
        *argv,
    ]
    return _run_logged(cmd, log_file)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
