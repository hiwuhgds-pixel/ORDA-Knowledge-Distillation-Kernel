from __future__ import annotations

import csv
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch

from ..core.console import safe_log_name
from ..core.metrics import csv_cell
from ..core.paths import LOGS_ROOT


# ── Output tee ───────────────────────────────────────────────────────────────
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


# ── Artifact paths ───────────────────────────────────────────────────────────
def case_log_paths(
    device: torch.device,
    router: str,
    case_id: str,
) -> tuple[Path, Path]:
    """Add a numeric suffix on collision so an artifact is not overwritten."""
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = str(device)
    root = LOGS_ROOT / safe_log_name(device_name) / safe_log_name(router)
    log_dir = root / "logs"
    csv_dir = root / "csv"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_log_name(case_id)
    candidate = stem
    suffix = 1
    while True:
        log_path = log_dir / f"{candidate}.log"
        csv_path = csv_dir / f"{candidate}.csv"
        try:
            with log_path.open("x", encoding="utf-8"):
                pass
            try:
                with csv_path.open("x", encoding="utf-8"):
                    pass
            except BaseException:
                log_path.unlink(missing_ok=True)
                raise
            return log_path, csv_path
        except FileExistsError:
            candidate = f"{stem}_{suffix}"
            suffix += 1


# ── Artifact persistence ─────────────────────────────────────────────────────
def write_csv(
    path: Path,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([csv_cell(value) for value in row])


@contextmanager
def case_log(device: torch.device, router: str, case_id: str):
    log_path, csv_path = case_log_paths(device, router, case_id)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = _TeeStream(old_stdout, log_file)
        sys.stderr = _TeeStream(old_stderr, log_file)
        try:
            yield csv_path
        except BaseException:
            csv_path.unlink(missing_ok=True)
            raise
        finally:
            tee_stdout, tee_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr
            tee_stdout.flush()
            tee_stderr.flush()
