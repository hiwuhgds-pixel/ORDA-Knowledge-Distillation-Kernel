#!/usr/bin/env python3
"""
VRAM benchmark for TiedTeacher CE+KL.

This script intentionally separates:
  - compiled model forward memory: teacher/student produce hidden states
  - CE+KL loss/backend memory: PyTorch compiled loss vs ORDA distillation_loss

The reported delta only compares the CE+KL loss/backend portion. It does not use
total-step VRAM deltas as the ORDA saving claim.
"""

import gc
import sys
from itertools import product

import torch
import torch.nn.functional as F

torch._dynamo.config.recompile_limit = 64

sys.path.insert(0, ".")
sys.path.insert(0, "..")
from src import Transformer

try:
    from orda_ce_kernel import TiedTeacher, distillation_loss, is_available as _orda_is_available

    ORDA_AVAILABLE = _orda_is_available()
except Exception:
    ORDA_AVAILABLE = False


BATCH = 16
VOCABS = [32_768, 65_536, 131_072]
SEQS = [256, 512, 1024, 2048]
DIMS = [512, 1024]
TEACHER_LAYERS = 8
STUDENT_LAYERS = 4
WARMUP = 1
MEASURE_ITERS = 1
TEMP = 3.0
DTYPE = torch.float16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODES = ["torch-compile"]
if ORDA_AVAILABLE:
    MODES.append("orda")
else:
    print("[WARN] orda_ce_kernel khong kha dung - chi chay torch-compile mode")


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def allocated_mb() -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024**2


def peak_extra_mb(base_bytes: int) -> float:
    if device.type != "cuda":
        return 0.0
    return (torch.cuda.max_memory_allocated(device) - base_bytes) / 1024**2


def full_cleanup():
    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def model_config(dim: int, layers: int) -> dict:
    if dim == 512:
        return dict(dim=512, q_heads=8, kv_heads=2, n_layers=layers, ffn_dim=1408)
    if dim == 1024:
        return dict(dim=1024, q_heads=8, kv_heads=2, n_layers=layers, ffn_dim=2816)
    raise ValueError(f"Unsupported dim: {dim}")


def build_models(vocab: int, dim: int, seq: int):
    teacher = Transformer(vocab, model_config(dim, TEACHER_LAYERS), seq=seq).to(
        dtype=torch.float32,
        device=device,
    )
    student = Transformer(vocab, model_config(dim, STUDENT_LAYERS), seq=seq).to(
        dtype=torch.float32,
        device=device,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, student


def make_torch_loss_fn():
    temperature = TEMP

    @torch.compile(mode="default")
    def loss_fn(h_s, h_t, head_weight, labels_flat):
        logits_s = F.linear(h_s, head_weight)
        logits_t = F.linear(h_t, head_weight)
        ce = F.cross_entropy(logits_s, labels_flat) + F.cross_entropy(logits_t, labels_flat)
        log_p_s = F.log_softmax(logits_s / temperature, dim=-1)
        p_t = F.softmax((logits_t / temperature).detach(), dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature * temperature)
        return ce + kl

    return loss_fn


def make_orda_loss_fn():
    def loss_fn(h_s, h_t, head_weight, labels_flat):
        return distillation_loss(
            h_s,
            head_weight,
            labels_flat,
            TiedTeacher(h_t),
            student_ce_weight=1.0,
            kd_weight=1.0,
            temperature=TEMP,
            backend="triton",
        ).loss

    return loss_fn


def zero_loss_grads(h_s, head_weight):
    h_s.grad = None
    head_weight.grad = None


def run_model_forward(teacher, student, x, dim: int):
    with torch.autocast(device_type=device.type, dtype=DTYPE):
        with torch.no_grad():
            h_t = teacher(x, return_hidden=True).view(-1, dim)
        h_s = student(x, return_hidden=True).view(-1, dim)
    return h_s, h_t


def run_loss_only(loss_fn, h_s, h_t, head_weight, labels_flat):
    # Detach h_s so the measurement covers CE+KL forward/backward and gradients
    # to hidden/head_weight, but not full student-model backward activation cost.
    h_s_loss = h_s.detach().requires_grad_(True)
    h_t_loss = h_t.detach()
    zero_loss_grads(h_s_loss, head_weight)
    with torch.autocast(device_type=device.type, dtype=DTYPE):
        loss = loss_fn(h_s_loss, h_t_loss, head_weight, labels_flat)
    loss.backward()
    sync()


def warmup_once(teacher, student, loss_fn, head_weight, vocab: int, dim: int, seq: int):
    x = torch.randint(0, vocab, (BATCH, seq), device=device)
    labels = torch.randint(0, vocab, (BATCH, seq), device=device).view(-1)
    h_s, h_t = run_model_forward(teacher, student, x, dim)
    run_loss_only(loss_fn, h_s, h_t, head_weight, labels)
    head_weight.grad = None
    del x, labels, h_s, h_t
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sync()


def measure_once(vocab: int, seq: int, dim: int, mode: str) -> dict:
    full_cleanup()
    teacher_raw, student_raw = build_models(vocab, dim, seq)
    teacher = torch.compile(teacher_raw, mode="default")
    student = torch.compile(student_raw, mode="default")
    head_weight = student_raw.head.weight
    loss_fn = make_torch_loss_fn() if mode == "torch-compile" else make_orda_loss_fn()

    try:
        for _ in range(WARMUP):
            warmup_once(teacher, student, loss_fn, head_weight, vocab, dim, seq)
        head_weight.grad = None

        x = torch.randint(0, vocab, (BATCH, seq), device=device)
        labels = torch.randint(0, vocab, (BATCH, seq), device=device).view(-1)
        sync()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            base_model = torch.cuda.memory_allocated(device)
        else:
            base_model = 0
        h_s, h_t = run_model_forward(teacher, student, x, dim)
        sync()
        model_extra = peak_extra_mb(base_model)
        after_model_alloc = allocated_mb()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            base_loss = torch.cuda.memory_allocated(device)
        else:
            base_loss = 0
        for _ in range(MEASURE_ITERS):
            run_loss_only(loss_fn, h_s, h_t, head_weight, labels)
        loss_extra = peak_extra_mb(base_loss)

        return {
            "status": "ok",
            "model_extra_mb": model_extra,
            "loss_ce_kl_extra_mb": loss_extra,
            "after_model_alloc_mb": after_model_alloc,
        }

    except torch.cuda.OutOfMemoryError:
        return {
            "status": "oom",
            "model_extra_mb": float("inf"),
            "loss_ce_kl_extra_mb": float("inf"),
            "after_model_alloc_mb": float("inf"),
        }
    finally:
        del teacher_raw, student_raw, teacher, student
        full_cleanup()


def fmt_vocab(vocab: int) -> str:
    if vocab >= 131_072:
        return "128k"
    if vocab >= 65_536:
        return "64k"
    return "32k"


def fmt_mb(value: float) -> str:
    if value == float("inf"):
        return f"{'OOM':>12}"
    return f"{value:>12.1f}"


def main():
    print(f"device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(device)}")
    print(
        f"dtype  : {DTYPE} | batch={BATCH} | teacher_layers={TEACHER_LAYERS} | "
        f"student_layers={STUDENT_LAYERS} | warmup={WARMUP} | measured_iters={MEASURE_ITERS} | T={TEMP}"
    )
    print(f"modes  : {MODES}")
    print("delta  : chi so sanh loss_ce_kl_extra_mb, khong so sanh tong VRAM")
    print()

    rows = []
    mode_w = max(len(mode) for mode in MODES)
    header = (
        f"{'dim':>5}  {'vocab':>6}  {'seq':>5}  {'mode':>{mode_w}}  "
        f"{'model MB':>12}  {'CE+KL MB':>12}  {'alloc after model MB':>20}"
    )
    sep = "-" * len(header)

    for dim, vocab, seq in product(DIMS, VOCABS, SEQS):
        print(f"\n-- dim={dim} vocab={fmt_vocab(vocab)} seq={seq} --")
        for mode in MODES:
            print(f"  {mode:>{mode_w}} ...", end=" ", flush=True)
            result = measure_once(vocab, seq, dim, mode)
            rows.append((dim, vocab, seq, mode, result))
            print(
                f"model {fmt_mb(result['model_extra_mb'])} MB  "
                f"CE+KL {fmt_mb(result['loss_ce_kl_extra_mb'])} MB"
            )

        combo = {
            mode: result
            for d, v, s, mode, result in rows
            if d == dim and v == vocab and s == seq
        }
        if "torch-compile" in combo and "orda" in combo:
            tc = combo["torch-compile"]["loss_ce_kl_extra_mb"]
            od = combo["orda"]["loss_ce_kl_extra_mb"]
            if tc == float("inf") or od == float("inf"):
                print("  CE+KL delta orda vs torch-compile: OOM")
            else:
                print(
                    f"  CE+KL delta orda vs torch-compile: {od - tc:+.1f} MB "
                    f"({(od / tc - 1) * 100:+.1f}%)"
                )

    print(f"\nSUMMARY\n{header}\n{sep}")
    for dim, vocab, seq, mode, result in rows:
        print(
            f"{dim:>5}  {fmt_vocab(vocab):>6}  {seq:>5}  {mode:>{mode_w}}  "
            f"{fmt_mb(result['model_extra_mb'])}  "
            f"{fmt_mb(result['loss_ce_kl_extra_mb'])}  "
            f"{fmt_mb(result['after_model_alloc_mb']):>20}"
        )

    if ORDA_AVAILABLE:
        delta_header = (
            f"{'dim':>5}  {'vocab':>6}  {'seq':>5}  "
            f"{'torch CE+KL MB':>15}  {'orda CE+KL MB':>15}  {'delta MB':>10}  {'delta %':>9}"
        )
        print(f"\nCE+KL VRAM DELTA ONLY\n{delta_header}\n{'-' * len(delta_header)}")
        for dim, vocab, seq in product(DIMS, VOCABS, SEQS):
            tc_entry = next(
                (
                    result
                    for d, v, s, mode, result in rows
                    if d == dim and v == vocab and s == seq and mode == "torch-compile"
                ),
                None,
            )
            od_entry = next(
                (
                    result
                    for d, v, s, mode, result in rows
                    if d == dim and v == vocab and s == seq and mode == "orda"
                ),
                None,
            )
            if tc_entry is None or od_entry is None:
                continue
            tc = tc_entry["loss_ce_kl_extra_mb"]
            od = od_entry["loss_ce_kl_extra_mb"]
            if tc == float("inf") or od == float("inf"):
                print(
                    f"{dim:>5}  {fmt_vocab(vocab):>6}  {seq:>5}  "
                    f"{'OOM':>15}  {'OOM':>15}  {'OOM':>10}  {'OOM':>9}"
                )
            else:
                print(
                    f"{dim:>5}  {fmt_vocab(vocab):>6}  {seq:>5}  "
                    f"{tc:>15.1f}  {od:>15.1f}  {od - tc:>+10.1f}  "
                    f"{(od / tc - 1) * 100:>+8.1f}%"
                )


if __name__ == "__main__":
    main()
