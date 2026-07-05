import ctypes
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


RESULT_PREFIX = "__PRECOMPUTED_WORKER_RESULT__ "
METRIC_KEYS = ("ms", "work_vram", "teacher_vram", "total_vram")


def source_seed(vocab: int, seq: int, batch: int) -> int:
    return (int(vocab) * 1_000_003 + int(seq) * 97 + int(batch)) % (2**31)


def _mode_backend_and_source(mode: str) -> tuple[str, str]:
    if mode.startswith("torch-compile"):
        backend = "torch-compile"
    elif mode.startswith("orda"):
        backend = "orda"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode.endswith("hidden"):
        source = "hidden_weight"
    elif mode.endswith("logits"):
        source = "logits"
    else:
        raise ValueError(f"Unknown precomputed source in mode: {mode}")
    return backend, source


def _pack_metric(value: float):
    return "OOM" if value == float("inf") else value


def _unpack_metric(value) -> float:
    return float("inf") if value == "OOM" else float(value)


def _map_result_metrics(result: dict, map_value) -> dict:
    def map_row(row: dict) -> dict:
        return {
            key: map_value(value) if key in METRIC_KEYS else value
            for key, value in row.items()
        }

    return {
        "passes": [
            {
                "index": pass_info["index"],
                "direction": pass_info["direction"],
                "rows": [map_row(row) for row in pass_info["rows"]],
            }
            for pass_info in result["passes"]
        ],
        "averages": [map_row(row) for row in result["averages"]],
    }


def _oom_result(source_modes: list[str], passes: int) -> dict:
    pass_orders = [source_modes, list(reversed(source_modes))]
    pass_results = []
    for p_idx, order in enumerate(pass_orders[:passes]):
        direction = "↓ top-down" if p_idx == 0 else "↑ bottom-up"
        pass_results.append({
            "index": p_idx + 1,
            "direction": direction,
            "rows": [
                {
                    "mode": mode,
                    "backend": _mode_backend_and_source(mode)[0],
                    "ms": float("inf"),
                    "work_vram": float("inf"),
                    "teacher_vram": float("inf"),
                    "total_vram": float("inf"),
                }
                for mode in order
            ],
        })

    return {
        "passes": pass_results,
        "averages": [
            {
                "mode": mode,
                "backend": _mode_backend_and_source(mode)[0],
                "ms": float("inf"),
                "work_vram": float("inf"),
                "teacher_vram": float("inf"),
                "total_vram": float("inf"),
            }
            for mode in source_modes
        ],
    }


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "out of memory",
        "cuda oom",
        "defaultcpuallocator",
        "not enough memory",
        "cannot allocate memory",
        "bad allocation",
        "winerror 1114",
        "dynamic link library (dll) initialization routine failed",
    )
    return isinstance(exc, MemoryError) or any(marker in text for marker in markers)


def _dtype_from_name(torch, name: str):
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    return getattr(torch, name)


def _set_seed(torch, device, seed: int) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _run_source_group(config: dict) -> dict:
    import torch

    from ._train import build_precomputed_cache, run_mode_isolated

    device = torch.device(config["device"])
    dtype = _dtype_from_name(torch, config["dtype"])
    _set_seed(torch, device, int(config["seed"]))

    source_label = config["source_label"]
    source = "hidden_weight" if source_label == "hidden" else "logits"
    source_modes = config["source_modes"]
    passes = int(config["passes"])

    precomputed_cache = build_precomputed_cache(
        vocab=config["vocab"],
        seq=config["seq"],
        teacher_config=config["teacher_config"],
        batch=config["batch"],
        dtype=dtype,
        device=device,
        use_amp=config["use_amp"],
        precomputed_source=source,
    )

    try:
        pass_orders = [source_modes, list(reversed(source_modes))]
        pass_rows = []
        pass_results: list[dict[str, tuple[float, float, float, float]]] = []

        for p_idx, order in enumerate(pass_orders[:passes]):
            direction = "↓ top-down" if p_idx == 0 else "↑ bottom-up"
            p_res: dict[str, tuple[float, float, float, float]] = {}
            rows = []
            for mode in order:
                backend, mode_source = _mode_backend_and_source(mode)
                ms, work_vram, teacher_vram, total_vram = run_mode_isolated(
                    vocab=config["vocab"],
                    seq=config["seq"],
                    mode=backend,
                    teacher_config=config["teacher_config"],
                    student_config=config["student_config"],
                    batch=config["batch"],
                    warmup=config["warmup"],
                    update_steps=config["update_steps"],
                    temp=config["temp"],
                    dtype=dtype,
                    device=device,
                    use_amp=config["use_amp"],
                    use_grad_scaler=config["use_grad_scaler"],
                    teacher_mode="precomputed",
                    precomputed_source=mode_source,
                    precomputed_cache=precomputed_cache,
                    return_details=True,
                )
                p_res[mode] = (ms, work_vram, teacher_vram, total_vram)
                rows.append({
                    "mode": mode,
                    "backend": backend,
                    "ms": ms,
                    "work_vram": work_vram,
                    "teacher_vram": teacher_vram,
                    "total_vram": total_vram,
                })
            pass_results.append(p_res)
            pass_rows.append({
                "index": p_idx + 1,
                "direction": direction,
                "rows": rows,
            })

        averages = []
        for mode in source_modes:
            backend, _ = _mode_backend_and_source(mode)
            ms_vals = [pr[mode][0] for pr in pass_results]
            work_vals = [pr[mode][1] for pr in pass_results]
            teacher_vals = [pr[mode][2] for pr in pass_results]
            total_vals = [pr[mode][3] for pr in pass_results]
            if any(value == float("inf") for value in ms_vals):
                ms_avg = work_avg = teacher_avg = total_avg = float("inf")
            else:
                ms_avg = sum(ms_vals) / len(ms_vals)
                work_avg = sum(work_vals) / len(work_vals)
                teacher_avg = sum(teacher_vals) / len(teacher_vals)
                total_avg = sum(total_vals) / len(total_vals)
            averages.append({
                "mode": mode,
                "backend": backend,
                "ms": ms_avg,
                "work_vram": work_avg,
                "teacher_vram": teacher_avg,
                "total_vram": total_avg,
            })

        return {"passes": pass_rows, "averages": averages}
    finally:
        del precomputed_cache


def _available_physical_memory_bytes() -> int | None:
    if sys.platform != "win32":
        return None

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


def _worker_memory_limit_bytes() -> int | None:
    explicit_gb = os.environ.get("PRECOMPUTED_WORKER_MEMORY_LIMIT_GB")
    if explicit_gb is not None:
        value = float(explicit_gb)
        return None if value <= 0 else int(value * 1024**3)

    available = _available_physical_memory_bytes()
    if available is None:
        return None

    fraction = float(os.environ.get("PRECOMPUTED_WORKER_MEMORY_FRACTION", "0.85"))
    reserve = int(float(os.environ.get("PRECOMPUTED_WORKER_MEMORY_RESERVE_GB", "2")) * 1024**3)
    return max(512 * 1024**2, min(int(available * fraction), max(0, available - reserve)))


def _last_win32_error() -> str:
    code = ctypes.get_last_error()
    return f"{code}: {ctypes.FormatError(code)}"


def _configure_windows_worker(proc: subprocess.Popen) -> int | None:
    if sys.platform != "win32":
        return None

    from ctypes import wintypes

    limit_bytes = _worker_memory_limit_bytes()
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
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
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


def _close_windows_handle(handle: int | None) -> None:
    if not handle or sys.platform != "win32":
        return

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(wintypes.HANDLE(handle))


def _worker_payload(stdout: str) -> dict | None:
    payload = None
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX):])
    return payload


def _tail(text: str, max_chars: int = 2000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_source_in_worker(config: dict, *, cwd: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if sys.platform == "win32":
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        )
    else:
        creationflags = 0

    proc = subprocess.Popen(
        [sys.executable, "-m", "src._precomputed_worker"],
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=creationflags,
    )

    job_handle = None
    try:
        job_handle = _configure_windows_worker(proc)
        stdout, stderr = proc.communicate(json.dumps(config) + "\n")
    except Exception:
        proc.kill()
        proc.wait()
        raise
    finally:
        _close_windows_handle(job_handle)

    payload = _worker_payload(stdout)
    if payload and payload.get("ok"):
        return _map_result_metrics(payload["result"], _unpack_metric)

    if payload and payload.get("kind") == "oom":
        return _oom_result(config["source_modes"], int(config["passes"]))

    if payload:
        raise RuntimeError(
            f"worker failed for vocab={config['vocab']} seq={config['seq']} "
            f"source={config['source_label']}: {payload.get('error', 'unknown error')}\n"
            f"{payload.get('traceback', '')}"
        )

    if proc.returncode != 0:
        return _oom_result(config["source_modes"], int(config["passes"]))

    details = _tail(stderr) or _tail(stdout)
    raise RuntimeError(f"worker exited without result: {details}")


def worker_main() -> int:
    try:
        config = json.loads(sys.stdin.readline())
        result = _run_source_group(config)
        payload = {"ok": True, "result": _map_result_metrics(result, _pack_metric)}
        print(RESULT_PREFIX + json.dumps(payload), flush=True)
        return 0
    except BaseException as exc:
        kind = "oom" if _is_oom_error(exc) else "error"
        payload = {
            "ok": False,
            "kind": kind,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(RESULT_PREFIX + json.dumps(payload), flush=True)
        return 0 if kind == "oom" else 1


if __name__ == "__main__":
    raise SystemExit(worker_main())
