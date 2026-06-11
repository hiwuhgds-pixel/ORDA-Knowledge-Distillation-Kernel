from __future__ import annotations

import argparse
import glob
import os
import shlex
import shutil
import subprocess
import sys

from tests.utils.env import import_torch


def locate_tool(name: str) -> str | None:
    path_match = shutil.which(name)
    if path_match:
        return path_match

    if sys.platform.startswith("win"):
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        if name == "ncu":
            matches = glob.glob(os.path.join(program_files, "NVIDIA Corporation", "Nsight Compute *", "ncu.exe"))
            return sorted(matches)[-1] if matches else None
        if name == "nsys":
            for pattern in [
                os.path.join(program_files, "NVIDIA Corporation", "Nsight Systems *", "target-windows-x64", "nsys.exe"),
                os.path.join(program_files, "NVIDIA Corporation", "Nsight Systems *", "bin", "nsys.exe"),
            ]:
                matches = glob.glob(pattern)
                if matches:
                    return sorted(matches)[-1]
    else:
        candidates = [f"/usr/local/cuda/bin/{name}"]
        if name == "ncu":
            candidates.extend(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
        if name == "nsys":
            candidates.extend(glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys"))
        for candidate in candidates:
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def split_target_args(raw: str) -> list[str]:
    if not raw:
        return []
    return shlex.split(raw, posix=not sys.platform.startswith("win"))


def construct_commands(ncu: str, nsys: str, target: list[str], output_dir: str, kernel: str | None):
    ncu_occ = [ncu, "--section", "Occupancy", "-o", os.path.join(output_dir, "profile_occupancy"), "-f"]
    ncu_cache = [
        ncu,
        "--metrics",
        "dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "-o",
        os.path.join(output_dir, "profile_memory"),
        "-f",
    ]
    if kernel:
        ncu_occ.extend(["-k", kernel])
        ncu_cache.extend(["-k", kernel])
    ncu_occ.extend(target)
    ncu_cache.extend(target)
    nsys_cmd = [
        nsys,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--stats=true",
        "-o",
        os.path.join(output_dir, "profile_nsys"),
        "-f",
        *target,
    ]
    return {"occupancy": ncu_occ, "memory": ncu_cache, "nsys": nsys_cmd}


def run(cmd: list[str], dry_run: bool) -> int:
    print("\n[COMMAND]", subprocess.list2cmdline(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Nsight profiling wrapper")
    parser.add_argument("--target-module", default="tests.benchmarks.bench_ce_only")
    parser.add_argument("--target-args", default="")
    parser.add_argument("--ncu")
    parser.add_argument("--nsys")
    parser.add_argument("--output-dir", default="profile_results")
    parser.add_argument("--kernel")
    parser.add_argument("--mode", choices=["all", "occupancy", "memory", "nsys"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ncu = args.ncu or locate_tool("ncu")
    nsys = args.nsys or locate_tool("nsys")
    cuda_available = False
    if not args.dry_run:
        torch = import_torch()
        cuda_available = torch.cuda.is_available()
    else:
        try:
            torch = import_torch()
            cuda_available = torch.cuda.is_available()
        except RuntimeError:
            cuda_available = False
    dry_run = args.dry_run or not cuda_available or not ncu or not nsys
    ncu_exec = ncu or ("ncu.exe" if sys.platform.startswith("win") else "ncu")
    nsys_exec = nsys or ("nsys.exe" if sys.platform.startswith("win") else "nsys")

    target = [sys.executable, "-m", args.target_module, *split_target_args(args.target_args)]
    commands = construct_commands(ncu_exec, nsys_exec, target, args.output_dir, args.kernel)
    if not dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    selected = ["occupancy", "memory", "nsys"] if args.mode == "all" else [args.mode]
    exit_code = 0
    for mode in selected:
        exit_code = exit_code or run(commands[mode], dry_run)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


