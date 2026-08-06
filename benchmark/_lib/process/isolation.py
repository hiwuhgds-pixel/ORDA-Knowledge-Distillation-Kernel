"""Keep worker OOM isolated so one failed allocation cannot terminate the sweep."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path


# ── Linux cgroup accounting ──────────────────────────────────────────────────
def _linux_cgroup_directories() -> list[Path]:
    if not sys.platform.startswith("linux"):
        return []
    directories = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")]
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, OSError):
        return directories
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, relative = parts
        relative = relative.lstrip("/")
        if hierarchy == "0":
            directories.append(Path("/sys/fs/cgroup") / relative)
        elif "memory" in controllers.split(","):
            directories.extend(
                (
                    Path("/sys/fs/cgroup/memory") / relative,
                    Path("/sys/fs/cgroup") / relative,
                )
            )
    return list(dict.fromkeys(directories))


def _read_cgroup_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return None if parsed >= 1 << 60 else parsed


def linux_cgroup_memory_remaining() -> int | None:
    remaining = []
    for directory in _linux_cgroup_directories():
        for limit_name, usage_name in (
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            limit = _read_cgroup_integer(directory / limit_name)
            usage = _read_cgroup_integer(directory / usage_name)
            if limit is not None and usage is not None:
                remaining.append(max(0, limit - usage))
    return min(remaining) if remaining else None


def linux_cgroup_oom_kill_count() -> int | None:
    """Use a rising cgroup counter as evidence of a kernel OOM kill."""
    counts = []
    for directory in _linux_cgroup_directories():
        for name in ("memory.events", "memory.events.local"):
            try:
                lines = (directory / name).read_text(encoding="utf-8").splitlines()
            except (FileNotFoundError, PermissionError, OSError):
                continue
            for line in lines:
                key, value = line.split(maxsplit=1)
                if key == "oom_kill":
                    counts.append(int(value))
                    break
    return max(counts) if counts else None


# ── Memory budget ────────────────────────────────────────────────────────────
def available_physical_memory_bytes() -> int | None:
    if sys.platform != "win32":
        try:
            host_available = int(os.sysconf("SC_AVPHYS_PAGES")) * int(
                os.sysconf("SC_PAGE_SIZE")
            )
        except (ValueError, OSError):
            host_available = None
        cgroup_remaining = linux_cgroup_memory_remaining()
        available = [
            value for value in (host_available, cgroup_remaining) if value is not None
        ]
        return min(available) if available else None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullAvailPhys)
    return None


def worker_memory_limit_bytes() -> int | None:
    explicit_gb = os.environ.get("PRECOMPUTED_WORKER_MEMORY_LIMIT_GB")
    if explicit_gb is not None:
        value = float(explicit_gb)
        return None if value <= 0 else int(value * 1024**3)

    available = available_physical_memory_bytes()
    if available is None:
        return None

    fraction = float(os.environ.get("PRECOMPUTED_WORKER_MEMORY_FRACTION", "0.85"))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("PRECOMPUTED_WORKER_MEMORY_FRACTION must be in (0, 1]")
    reserve = int(
        float(os.environ.get("PRECOMPUTED_WORKER_MEMORY_RESERVE_GB", "2")) * 1024**3
    )
    if reserve < 0:
        raise ValueError("PRECOMPUTED_WORKER_MEMORY_RESERVE_GB must be >= 0")
    fraction_limit = int(available * fraction)
    reserve_limit = available - reserve
    usable = reserve_limit if reserve_limit > 0 else fraction_limit
    return max(1, min(fraction_limit, usable))


def _last_win32_error() -> str:
    code = ctypes.get_last_error()
    return f"{code}: {ctypes.FormatError(code)}"


# ── Windows Job Object ───────────────────────────────────────────────────────
def configure_windows_worker(proc: subprocess.Popen) -> int | None:
    if sys.platform != "win32":
        return None

    from ctypes import wintypes

    limit_bytes = worker_memory_limit_bytes()
    if limit_bytes is None:
        return None

    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    JobObjectExtendedLimitInformation = 9

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise RuntimeError(f"CreateJobObjectW failed: {_last_win32_error()}")

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | JOB_OBJECT_LIMIT_JOB_MEMORY
        | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    info.ProcessMemoryLimit = limit_bytes
    info.JobMemoryLimit = limit_bytes
    ok = kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        kernel32.CloseHandle(wintypes.HANDLE(job))
        raise RuntimeError(f"SetInformationJobObject failed: {_last_win32_error()}")

    process_handle = wintypes.HANDLE(proc._handle)
    ok = kernel32.AssignProcessToJobObject(job, process_handle)
    if not ok:
        kernel32.CloseHandle(wintypes.HANDLE(job))
        raise RuntimeError(f"AssignProcessToJobObject failed: {_last_win32_error()}")

    kernel32.SetPriorityClass(process_handle, BELOW_NORMAL_PRIORITY_CLASS)
    return int(job)


def close_windows_handle(handle: int | None) -> None:
    if not handle or sys.platform != "win32":
        return

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(wintypes.HANDLE(handle))


# ── POSIX memory limits ──────────────────────────────────────────────────────
def configure_posix_worker_memory() -> None:
    """Use the RSS monitor on Linux because ``RLIMIT_RSS`` is not enforced."""
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        return
    limit = worker_memory_limit_bytes()
    if limit is None:
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_RSS, (limit, limit))
    except (AttributeError, ImportError, OSError, ValueError):
        return


def linux_process_tree_rss_bytes(pid: int) -> int:
    pending = [pid]
    seen = set()
    total = 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text(encoding="utf-8")
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
            children = Path(f"/proc/{current}/task/{current}/children").read_text(
                encoding="utf-8"
            )
            pending.extend(int(child) for child in children.split())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return total


def start_linux_memory_monitor(proc: subprocess.Popen):
    stop = threading.Event()
    exceeded = threading.Event()
    limit = worker_memory_limit_bytes()
    if not sys.platform.startswith("linux") or limit is None:
        return stop, exceeded, None

    def monitor() -> None:
        while not stop.wait(0.25):
            if linux_process_tree_rss_bytes(proc.pid) > limit:
                exceeded.set()
                terminate_worker_tree(proc)
                return

    thread = threading.Thread(
        target=monitor, name="benchmark-worker-memory", daemon=True
    )
    thread.start()
    return stop, exceeded, thread


def terminate_worker_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def spawn_kwargs() -> dict:
    """Keep workers low-priority and independently killable."""
    if sys.platform == "win32":
        return {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
            "start_new_session": False,
        }
    return {"creationflags": 0, "start_new_session": True}


def process_exit_description(returncode: int) -> str:
    if returncode < 0:
        return f"signal {-returncode}"
    return f"exit code {returncode}"
