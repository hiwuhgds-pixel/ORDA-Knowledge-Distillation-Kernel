from __future__ import annotations

import re
import subprocess
import sys
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

RULE = "─"
Emitter = Callable[[str], object]


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def safe_log_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "unknown"


def active_gpu_log_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return safe_log_name(torch.cuda.get_device_name(0))
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def test_log_path(
    root: Path,
    runner: str,
    *,
    device_name: str | None = None,
    group: str | None = "test",
) -> Path:
    directory = root / "logs" / safe_log_name(device_name or active_gpu_log_name())
    if group is not None:
        directory /= safe_log_name(group)
    directory.mkdir(parents=True, exist_ok=True)
    stem = safe_log_name(runner)
    candidate = stem
    suffix = 1
    while True:
        path = directory / f"{candidate}.log"
        try:
            with path.open("x", encoding="utf-8"):
                pass
            return path
        except FileExistsError:
            candidate = f"{stem}_{suffix}"
            suffix += 1


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self.streams
        )


@contextmanager
def test_log(
    root: Path,
    runner: str,
    *,
    device_name: str | None = None,
    group: str | None = "test",
):
    path = test_log_path(root, runner, device_name=device_name, group=group)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = _TeeStream(old_stdout, log_file)
        sys.stderr = _TeeStream(old_stderr, log_file)
        try:
            yield path
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            tee_stdout, tee_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr
            tee_stdout.flush()
            tee_stderr.flush()


def run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return proc.wait()


def hrule(width: int) -> str:
    return RULE * max(width, 0)


def banner_width(title: str, min_width: int = 56) -> int:
    return max(min_width, len(title) + 2)


def print_banner(
    title: str,
    width: int | None = None,
    *,
    emit: Emitter = print,
) -> None:
    width = banner_width(title) if width is None else width
    emit(hrule(width))
    emit(f" {title}")
    emit(hrule(width))


def print_config(fields: dict[str, str], *, emit: Emitter = print) -> None:
    key_w = max(len(key) for key in fields)
    for key, value in fields.items():
        emit(f" {key:<{key_w}} : {value}")
    emit("")


def print_group_header(
    label: str,
    width: int | None = None,
    *,
    emit: Emitter = print,
) -> None:
    width = max(56, len(label) + 6) if width is None else width
    tail = hrule(width - len(label) - 4)
    emit(f"\n── {label} {tail}")
