"""Run each teacher-cache source group separately to bound host memory."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

from ..core.metrics import pack_metric, unpack_metric
from ..core.oom import (
    exception_is_oom_without_torch,
    process_failure_is_oom,
)
from ..core.stats import OOM, average_passes
from ..suites.specs import alternating_passes, precomputed_mode_parts, stable_seed
from .isolation import (
    close_windows_handle,
    configure_posix_worker_memory,
    configure_windows_worker,
    linux_cgroup_oom_kill_count,
    process_exit_description,
    spawn_kwargs,
    start_linux_memory_monitor,
    terminate_worker_tree,
)

RESULT_PREFIX = "__PRECOMPUTED_WORKER_RESULT__ "
PROGRESS_PREFIX = "__PRECOMPUTED_WORKER_PROGRESS__ "
METRIC_KEYS = ("ms", "work_vram", "teacher_vram", "total_vram")


# ── Protocol encoding ────────────────────────────────────────────────────────
def _map_row_metrics(row: dict, map_value) -> dict:
    return {
        key: map_value(value) if key in METRIC_KEYS else value
        for key, value in row.items()
    }


def _map_result_metrics(result: dict, map_value) -> dict:
    return {
        "passes": [
            {
                "index": pass_info["index"],
                "direction": pass_info["direction"],
                "rows": [_map_row_metrics(row, map_value) for row in pass_info["rows"]],
            }
            for pass_info in result["passes"]
        ],
        "averages": [_map_row_metrics(row, map_value) for row in result["averages"]],
    }


def _oom_row(mode: str) -> dict:
    return {
        "mode": mode,
        "backend": precomputed_mode_parts(mode)[0],
        **{key: OOM for key in METRIC_KEYS},
    }


def oom_result(
    source_modes: list[str],
    passes: int,
    completed: dict[tuple[int, str], dict] | None = None,
) -> dict:
    """Preserve completed rows when a worker dies from OOM."""
    completed = {} if completed is None else completed
    pass_results = []
    for pass_index, direction, order in alternating_passes(source_modes, passes):
        rows = [completed.get((pass_index, mode)) or _oom_row(mode) for mode in order]
        pass_results.append({"index": pass_index, "direction": direction, "rows": rows})

    pass_maps = [
        {row["mode"]: row for row in pass_result["rows"]}
        for pass_result in pass_results
    ]
    averages = [
        {
            "mode": mode,
            "backend": precomputed_mode_parts(mode)[0],
            **average_passes(
                [pass_map[mode] for pass_map in pass_maps],
                METRIC_KEYS,
                guard_key="ms",
            ),
        }
        for mode in source_modes
    ]
    return {"passes": pass_results, "averages": averages}


# ── Worker child ─────────────────────────────────────────────────────────────
def _dtype_from_name(torch, name: str):
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    return getattr(torch, name)


def run_source_group(config: dict) -> dict:
    import torch

    from ..engine.train import TrainContext, run_isolated
    from ..losses.spec import LossSpec
    from ..model import ModelConfig
    from ..suites.cache import build_precomputed_cache
    from ..suites.train import ModelTrainWorkload

    device = torch.device(config["device"])
    dtype = _dtype_from_name(torch, config["dtype"])
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))

    source_label = config["source_label"]
    source = "hidden_weight" if source_label == "hidden" else "logits"
    source_modes = config["source_modes"]
    passes = int(config["passes"])
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")

    teacher_config = ModelConfig.from_mapping(config["teacher_config"])
    student_config = ModelConfig.from_mapping(config["student_config"])
    divergence = config.get("kl_divergence", "forward")
    spec = LossSpec(
        temperature=config["temp"],
        kl_divergence=divergence,
        student_ce_weight=1.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
    )

    precomputed_cache = build_precomputed_cache(
        vocab=config["vocab"],
        seq=config["seq"],
        teacher_config=teacher_config,
        batch=config["batch"],
        dtype=dtype,
        device=device,
        use_amp=config["use_amp"],
        precomputed_source=source,
        cache_kind=config.get("cache_kind", "standard"),
    )

    try:
        pass_rows = []
        samples: dict[str, list[dict[str, float]]] = {mode: [] for mode in source_modes}

        for pass_index, direction, order in alternating_passes(source_modes, passes):
            rows = []
            for mode in order:
                backend, mode_source = precomputed_mode_parts(mode)
                result = run_isolated(
                    ModelTrainWorkload(
                        teacher_config=teacher_config,
                        student_config=student_config,
                        spec=spec,
                        teacher_requires_grad=False,
                        precomputed=True,
                        precomputed_source=mode_source,
                        precomputed_cache=precomputed_cache,
                    ),
                    TrainContext(
                        vocab=config["vocab"],
                        seq=config["seq"],
                        batch=config["batch"],
                        warmup=config["warmup"],
                        update_steps=config["update_steps"],
                        dtype=dtype,
                        device=device,
                        use_amp=config["use_amp"],
                        use_grad_scaler=config["use_grad_scaler"],
                        mode=backend,
                    ),
                    seed=stable_seed(config["seed"], pass_index),
                )
                row = {
                    "mode": mode,
                    "backend": backend,
                    "ms": result.ms,
                    "work_vram": result.work_vram_mb,
                    "teacher_vram": result.teacher_cache_mb,
                    "total_vram": result.total_vram_mb,
                }
                samples[mode].append({key: row[key] for key in METRIC_KEYS})
                rows.append(row)
                print(
                    PROGRESS_PREFIX
                    + json.dumps(
                        {
                            "pass_index": pass_index,
                            "row": _map_row_metrics(row, pack_metric),
                        }
                    ),
                    flush=True,
                )
            pass_rows.append(
                {"index": pass_index, "direction": direction, "rows": rows}
            )

        averages = [
            {
                "mode": mode,
                "backend": precomputed_mode_parts(mode)[0],
                **average_passes(samples[mode], METRIC_KEYS, guard_key="ms"),
            }
            for mode in source_modes
        ]
        return {"passes": pass_rows, "averages": averages}
    finally:
        del precomputed_cache


# ── Parent transport ─────────────────────────────────────────────────────────
def _worker_payload(stdout: str) -> dict | None:
    payload = None
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX) :])
    return payload


def _worker_progress(stdout: str) -> dict[tuple[int, str], dict]:
    completed = {}
    for line in stdout.splitlines():
        if not line.startswith(PROGRESS_PREFIX):
            continue
        payload = json.loads(line[len(PROGRESS_PREFIX) :])
        row = _map_row_metrics(payload["row"], unpack_metric)
        completed[(int(payload["pass_index"]), row["mode"])] = row
    return completed


def _tail(text: str, max_chars: int = 2000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_source_in_worker(config: dict, *, cwd: Path) -> dict:
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cgroup_oom_before = linux_cgroup_oom_kill_count()

    proc = subprocess.Popen(
        [sys.executable, "-m", "_lib.process.worker"],
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        **spawn_kwargs(),
    )

    job_handle = None
    monitor_stop = monitor_exceeded = monitor_thread = None
    try:
        job_handle = configure_windows_worker(proc)
        monitor_stop, monitor_exceeded, monitor_thread = start_linux_memory_monitor(
            proc
        )
        timeout = float(os.environ.get("PRECOMPUTED_WORKER_TIMEOUT_SECONDS", "3600"))
        if timeout <= 0:
            raise ValueError("PRECOMPUTED_WORKER_TIMEOUT_SECONDS must be > 0")
        try:
            stdout, stderr = proc.communicate(
                json.dumps(config) + "\n", timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            terminate_worker_tree(proc)
            stdout, stderr = proc.communicate()
            raise TimeoutError(
                f"worker timed out after {timeout:g}s for "
                f"vocab={config['vocab']} seq={config['seq']} "
                f"source={config['source_label']}"
            ) from exc
    except BaseException:
        if proc.poll() is None:
            terminate_worker_tree(proc)
            proc.wait()
        raise
    finally:
        if monitor_stop is not None:
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join()
        close_windows_handle(job_handle)

    completed = _worker_progress(stdout)
    payload = _worker_payload(stdout)
    cgroup_oom_after = linux_cgroup_oom_kill_count()
    cgroup_oom_kill = (
        cgroup_oom_before is not None
        and cgroup_oom_after is not None
        and cgroup_oom_after > cgroup_oom_before
    )

    if monitor_exceeded is not None and monitor_exceeded.is_set():
        return oom_result(config["source_modes"], int(config["passes"]), completed)

    if proc.returncode != 0:
        if process_failure_is_oom(
            proc.returncode,
            stdout,
            stderr,
            cgroup_oom_kill=cgroup_oom_kill,
        ):
            return oom_result(config["source_modes"], int(config["passes"]), completed)
        details = _tail(stderr) or _tail(stdout) or "no worker output"
        raise RuntimeError(
            f"worker exited with {process_exit_description(proc.returncode)} "
            f"for vocab={config['vocab']} "
            f"seq={config['seq']} source={config['source_label']}: {details}"
        )

    if payload and payload.get("ok"):
        return _map_result_metrics(payload["result"], unpack_metric)

    if payload and payload.get("kind") == "oom":
        return oom_result(config["source_modes"], int(config["passes"]), completed)

    if payload:
        raise RuntimeError(
            f"worker failed for vocab={config['vocab']} seq={config['seq']} "
            f"source={config['source_label']}: {payload.get('error', 'unknown error')}\n"
            f"{payload.get('traceback', '')}"
        )

    details = _tail(stderr) or _tail(stdout)
    raise RuntimeError(f"worker exited without result: {details}")


# ── Child entrypoint ─────────────────────────────────────────────────────────
def worker_main() -> int:
    try:
        configure_posix_worker_memory()
        config = json.loads(sys.stdin.readline())
        result = run_source_group(config)
        payload = {"ok": True, "result": _map_result_metrics(result, pack_metric)}
        print(RESULT_PREFIX + json.dumps(payload), flush=True)
        return 0
    except Exception as exc:
        kind = "oom" if exception_is_oom_without_torch(exc) else "error"
        payload = {
            "ok": False,
            "kind": kind,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(RESULT_PREFIX + json.dumps(payload), flush=True)
        # An OOM is a result, not a failure: exit 0 so the parent reads it.
        return 0 if kind == "oom" else 1


if __name__ == "__main__":
    raise SystemExit(worker_main())
