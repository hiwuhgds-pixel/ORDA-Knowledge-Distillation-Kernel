import gc
import importlib.util
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import torch

from . import Transformer


def _load_safe_log_name():
    path = Path(__file__).resolve().parents[2] / "logging_utils.py"
    spec = importlib.util.spec_from_file_location("_test_logging_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load logging utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.safe_log_name


safe_log_name = _load_safe_log_name()


# ── Log formatting ───────────────────────────────────────────────────────────
RULE = "─"


def hrule(width: int) -> str:
    return RULE * max(width, 0)


def banner_width(title: str, min_width: int = 56) -> int:
    return max(min_width, len(title) + 2)


def print_banner(title: str, width: int | None = None) -> None:
    width = banner_width(title) if width is None else width
    print(hrule(width))
    print(f" {title}")
    print(hrule(width))


def print_config(fields: dict[str, str]) -> None:
    key_w = max(len(key) for key in fields)
    for key, value in fields.items():
        print(f" {key:<{key_w}} : {value}")
    print()


def print_group_header(label: str, width: int | None = None) -> None:
    width = max(56, len(label) + 6) if width is None else width
    tail = hrule(width - len(label) - 4)
    print(f"\n── {label} {tail}")


def print_row(indent: int, columns: Iterable[str]) -> None:
    print(" " * indent + "  ".join(columns))


def _is_numeric_text(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped in {"OOM", "NA"}:
        return True
    if stripped.endswith("%"):
        stripped = stripped[:-1]
    if stripped.startswith(("+", "-")):
        stripped = stripped[1:]
    stripped = stripped.replace(",", "")
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    title: str | None = None,
    align_right: set[int] | None = None,
) -> None:
    align_right = set() if align_right is None else set(align_right)
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    if title:
        print()
        print(f" {title}")
    header_line = " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    print(f" {header_line}")
    print(" " + "-|-".join("-" * width for width in widths))
    for row in rows:
        cells: list[str] = []
        for idx, cell in enumerate(row):
            text = str(cell)
            if idx in align_right or _is_numeric_text(text):
                cells.append(text.rjust(widths[idx]))
            else:
                cells.append(text.ljust(widths[idx]))
        print(" " + " | ".join(cells))


# ── File logging ─────────────────────────────────────────────────────────────
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
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def script_log_path(device: torch.device, script_path: str) -> Path:
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = str(device)
    logs_root = Path(__file__).resolve().parents[3] / "logs" / safe_log_name(gpu_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(script_path).stem
    path = logs_root / f"{stem}_{stamp}.log"
    suffix = 1
    while path.exists():
        path = logs_root / f"{stem}_{stamp}_{suffix}.log"
        suffix += 1
    return path


@contextmanager
def script_log(device: torch.device, script_path: str):
    path = script_log_path(device, script_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = _TeeStream(old_stdout, log_file)
        sys.stderr = _TeeStream(old_stderr, log_file)
        try:
            yield path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr


@contextmanager
def train_log(device: torch.device, script_path: str):
    with script_log(device, script_path) as path:
        yield path


# ── Benchmark profiles ───────────────────────────────────────────────────────
SMALL_PROFILE = {
    "name": "t4",
    "teacher_config": dict(dim=1024, q_heads=8, kv_heads=2, n_layers=8, ffn_dim=2816),
    "student_config": dict(dim=1024, q_heads=8, kv_heads=2, n_layers=4, ffn_dim=2816),
    "vocabs": [32_768, 65_536, 131_072],
    "seqs": [256, 512, 1024, 2048],
    "batch": 16,
    "warmup": 5,
    "update_steps": 50,
    "passes": 2,
    "temp": 3.0,
}

LARGE_PROFILE = {
    "name": "large",
    "teacher_config": dict(dim=1024, q_heads=8, kv_heads=2, n_layers=16, ffn_dim=2816),
    "student_config": dict(dim=1024, q_heads=8, kv_heads=2, n_layers=8, ffn_dim=2816),
    "vocabs": [32_768, 65_536, 131_072, 262_144],
    "seqs": [512, 1024, 2048, 4096],
    "batch": 16,
    "warmup": 5,
    "update_steps": 50,
    "passes": 2,
    "temp": 3.0,
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark_profile(device: torch.device) -> dict:
    if device.type != "cuda":
        return LARGE_PROFILE

    props = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory / 1024**3
    return SMALL_PROFILE if vram_gb < 16 else LARGE_PROFILE


def precision_config(device: torch.device) -> tuple[torch.dtype, bool, bool]:
    if device.type != "cuda":
        return torch.float32, False, False

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, False
    return torch.float16, True, True


def fmt_precision(dtype: torch.dtype, use_amp: bool, use_grad_scaler: bool) -> str:
    suffix = []
    if use_amp:
        suffix.append("amp")
    if use_grad_scaler:
        suffix.append("gradscale")
    if suffix:
        return f"{dtype} ({', '.join(suffix)})"
    return str(dtype)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_extra_mb(device: torch.device, base_bytes: int) -> float:
    if device.type != "cuda":
        return 0.0
    return (torch.cuda.max_memory_allocated(device) - base_bytes) / 1024**2


def full_cleanup(device: torch.device) -> None:
    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def freeze_teacher(model: torch.nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def build_teacher_student(
    vocab: int,
    teacher_config: dict,
    student_config: dict,
    *,
    seq: int,
    device: torch.device,
) -> tuple[Transformer, Transformer]:
    teacher = Transformer(vocab, teacher_config, seq=seq).to(dtype=torch.float32, device=device)
    student = Transformer(vocab, student_config, seq=seq).to(dtype=torch.float32, device=device)
    freeze_teacher(teacher)
    return teacher, student


def trimmed_mean(values: list[float], trim_ratio: float = 0.10) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    cut = int(len(values) * trim_ratio)
    if cut > 0 and len(values) > 2 * cut:
        values = values[cut:-cut]
    return sum(values) / len(values)


def fmt_vocab(vocab: int) -> str:
    if vocab >= 262_144:
        return "256k"
    if vocab >= 131_072:
        return "128k"
    if vocab >= 65_536:
        return "64k"
    if vocab >= 32_768:
        return "32k"
    if vocab >= 16_384:
        return "16k"
    return "32k"


def fmt_ms(value: float) -> str:
    return f"{value:>9.2f}" if value != float("inf") else f"{'OOM':>9}"


def fmt_mb(value: float) -> str:
    return f"{value:>12.1f}" if value != float("inf") else f"{'OOM':>12}"
