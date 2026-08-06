"""Keep text OOM detection torch-free for the worker's early failure path."""

from __future__ import annotations

import signal

OOM_MARKERS = (
    "out of memory",
    "cuda oom",
    "defaultcpuallocator",
    "not enough memory",
    "cannot allocate memory",
    "bad allocation",
)

# STATUS_NO_MEMORY, STATUS_COMMITMENT_LIMIT: a Windows Job Object memory cap
# kills the child with these instead of raising inside Python.
WINDOWS_OOM_EXIT_CODES = {0xC0000017, 0xC000009A}
SIGKILL_RETURN_CODE = -getattr(signal, "SIGKILL", 9)


def text_is_oom(*texts: str) -> bool:
    joined = "\n".join(texts).lower().replace("not out of memory", "")
    return any(marker in joined for marker in OOM_MARKERS)


def is_oom_error(exc: BaseException) -> bool:
    import torch

    return isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) or text_is_oom(
        str(exc)
    )


def exception_is_oom_without_torch(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or text_is_oom(str(exc))


def process_failure_is_oom(
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    cgroup_oom_kill: bool = False,
) -> bool:
    if text_is_oom(stderr, stdout):
        return True
    if returncode == SIGKILL_RETURN_CODE and cgroup_oom_kill:
        return True
    return returncode & 0xFFFFFFFF in WINDOWS_OOM_EXIT_CODES
