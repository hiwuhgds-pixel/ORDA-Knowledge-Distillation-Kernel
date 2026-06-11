#!/usr/bin/env python3
"""
Benchmark: (CE + KL) — torch.compile (fused full graph) vs ORDA kernel.

2 modes:
  - torch-compile : torch.compile merges model + (CE + KL) into one compiled region
  - orda          : torch.compile model + distillation_loss (Triton, kd_weight=1)

"""

import gc
import sys
import time
from itertools import product

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "..")
from src import Transformer

try:
    from orda_ce_kernel import TiedTeacher, distillation_loss, is_available as _orda_is_available
    ORDA_AVAILABLE = _orda_is_available()
except Exception:
    ORDA_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
TEACHER_CFG = dict(dim=1024, q_heads=8, kv_heads=2, n_layers=8, ffn_dim=2816)
STUDENT_CFG = dict(dim=1024, q_heads=8, kv_heads=2, n_layers=4,  ffn_dim=2816)

VOCABS         = [32_768, 65_536, 131_072]
SEQS           = [256, 512, 1024, 2048]
BATCH          = 16
WARMUP         = 5
UPDATE_STEPS   = 50
PASSES         = 2
TEMP           = 3.0
DTYPE = torch.float16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODES = ["torch-compile"]
if ORDA_AVAILABLE:
    MODES += ["orda"]
else:
    print("[WARN] orda_ce_kernel is unavailable - running torch-compile mode only")


# ── Helpers ───────────────────────────────────────────────────────────────────
def sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_mb() -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return 0.0


def full_cleanup():
    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_models(vocab: int, seq: int):
    teacher = Transformer(vocab, TEACHER_CFG, seq=seq).to(dtype=torch.float32, device=device)
    student = Transformer(vocab, STUDENT_CFG, seq=seq).to(dtype=torch.float32, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, student


def make_step(teacher_raw, student_raw, opt, scaler, vocab: int, seq: int, mode: str):
    x      = torch.randint(0, vocab, (BATCH, seq), device=device)
    labels = torch.randint(0, vocab, (BATCH, seq), device=device)
    H      = TEACHER_CFG["dim"]
    T      = TEMP

    if mode == "torch-compile":
        # Merge model + (CE + KL) loss into one compiled region so Inductor can fuse through it.
        def forward_and_loss(x_in, labels_flat):
            with torch.no_grad():
                h_t = teacher_raw(x_in, return_hidden=True).view(-1, H)
            h_s = student_raw(x_in, return_hidden=True).view(-1, H)
            w = student_raw.head.weight
            logits_s = F.linear(h_s, w)
            logits_t = F.linear(h_t, w)
            ce = F.cross_entropy(logits_s, labels_flat) + F.cross_entropy(logits_t, labels_flat)
            log_p_s = F.log_softmax(logits_s / T, dim=-1)
            p_t = F.softmax((logits_t / T).detach(), dim=-1)
            kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
            return ce + kl

        compiled_fn = torch.compile(forward_and_loss, mode="default")

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=DTYPE):
                loss = compiled_fn(x, labels.view(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        return step

    if mode == "orda":
        # ORDA: compile the models; run distillation_loss under autocast.
        teacher = torch.compile(teacher_raw, mode="default")
        student = torch.compile(student_raw, mode="default")
        head_weight = student_raw.head.weight
        labels_flat = labels.view(-1)

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=DTYPE):
                with torch.no_grad():
                    h_t = teacher(x, return_hidden=True).view(-1, H)
                h_s = student(x, return_hidden=True).view(-1, H)
                out = distillation_loss(
                    h_s,
                    head_weight,
                    labels_flat,
                    TiedTeacher(h_t),
                    student_ce_weight=1.0,
                    kd_weight=1.0,
                    temperature=T,
                    backend="triton",
                )
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        return step

    raise ValueError(f"Unknown mode: {mode}")


def trimmed_mean(values: list[float], trim_ratio: float = 0.10) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    cut = int(len(values) * trim_ratio)
    if cut > 0 and len(values) > 2 * cut:
        values = values[cut:-cut]
    return sum(values) / len(values)


def run_mode_isolated(vocab: int, seq: int, mode: str) -> tuple[float, float]:
    full_cleanup()
    teacher_raw, student_raw = build_models(vocab, seq)
    opt = torch.optim.SGD(student_raw.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    try:
        step = make_step(teacher_raw, student_raw, opt, scaler, vocab, seq, mode)

        for _ in range(WARMUP):
            step()
        sync()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync()

        samples = []
        if device.type == "cuda":
            for _ in range(UPDATE_STEPS):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()
                step()
                end_event.record()
                sync()
                samples.append(start_event.elapsed_time(end_event))
            ms = trimmed_mean(samples, 0.10)
        else:
            for _ in range(UPDATE_STEPS):
                t0 = time.perf_counter()
                step()
                sync()
                samples.append((time.perf_counter() - t0) * 1_000)
            ms = trimmed_mean(samples, 0.10)

        vram = peak_mb()
        return ms, vram

    except torch.cuda.OutOfMemoryError:
        return float("inf"), float("inf")
    finally:
        del teacher_raw, student_raw, opt
        full_cleanup()


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt_vocab(v: int) -> str:
    if v >= 131_072: return "128k"
    if v >= 65_536:  return "64k"
    return "32k"


def fmt_ms(v: float) -> str:
    return f"{v:>9.2f}" if v != float("inf") else f"{'OOM':>9}"


def fmt_mb(v: float) -> str:
    return f"{v:>12.1f}" if v != float("inf") else f"{'OOM':>12}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(device)}")
    print(f"dtype  : {DTYPE} | batch={BATCH} | warmup={WARMUP} | "
          f"update_steps={UPDATE_STEPS} | passes={PASSES} | T={TEMP}")
    print(f"modes  : {MODES}  (KD: CE_s + CE_t + KL)")
    print()

    mode_w = max(len(m) for m in MODES)
    col = f"{'vocab':>6}  {'seq':>5}  {'mode':>{mode_w}}  {'ms/step':>9}  {'peakVRAM MB':>12}"
    sep = "─" * len(col)
    rows: list[tuple] = []

    for vocab, seq in product(VOCABS, SEQS):
        label = f"vocab={fmt_vocab(vocab)}  seq={seq}"
        print(f"\n── {label} ──")

        pass_orders = [MODES, list(reversed(MODES))]
        pass_results: list[dict[str, tuple[float, float]]] = []

        for p_idx, order in enumerate(pass_orders[:PASSES]):
            direction = "↓ top-down" if p_idx == 0 else "↑ bottom-up"
            print(f"  pass {p_idx + 1}/{PASSES} ({direction})")
            p_res: dict[str, tuple[float, float]] = {}
            for mode in order:
                print(f"    {mode:>{mode_w}} ...", end=" ", flush=True)
                ms, vram = run_mode_isolated(vocab, seq, mode)
                p_res[mode] = (ms, vram)
                print(f"{fmt_ms(ms)} ms  {fmt_mb(vram)} MB")
            pass_results.append(p_res)

        print(f"  ── average ──")
        for mode in MODES:
            ms_vals = [pr[mode][0] for pr in pass_results]
            vr_vals = [pr[mode][1] for pr in pass_results]
            if any(m == float("inf") for m in ms_vals):
                ms_avg, vr_avg = float("inf"), float("inf")
            else:
                ms_avg = sum(ms_vals) / len(ms_vals)
                vr_avg = sum(vr_vals) / len(vr_vals)
            rows.append((vocab, seq, mode, ms_avg, vr_avg))
            print(f"    {mode:>{mode_w}}  {fmt_ms(ms_avg)} ms  {fmt_mb(vr_avg)} MB")

        # Direct ORDA vs torch-compile comparison when available.
        combo = {m: (ms, vr) for v, s, m, ms, vr in rows if v == vocab and s == seq}
        if ORDA_AVAILABLE and "orda" in combo and "torch-compile" in combo:
            ms_o, vr_o = combo["orda"]
            ms_c, vr_c = combo["torch-compile"]
            if ms_o != float("inf") and ms_c != float("inf"):
                print(f"\n    orda vs torch-compile: "
                      f"Δms {ms_o - ms_c:>+7.2f}  ({(ms_o / ms_c - 1) * 100:>+6.1f}%)  "
                      f"ΔVRAM {vr_o - vr_c:>+7.1f} MB  ({(vr_o / vr_c - 1) * 100:>+6.1f}%)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'═' * len(col)}")
    print("SUMMARY")
    print(col)
    print(sep)
    for vocab, seq, mode, ms, vram in rows:
        print(f"{fmt_vocab(vocab):>6}  {seq:>5}  {mode:>{mode_w}}  {fmt_ms(ms)}  {fmt_mb(vram)}")

    # ── ORDA vs torch-compile table ───────────────────────────────────────────
    if ORDA_AVAILABLE:
        print(sep)
        print("\nORDA vs torch-compile")
        hdr = (f"{'vocab':>6}  {'seq':>5}  "
               f"{'Δms':>10}  {'Δms%':>8}  {'ΔVRAM MB':>10}  {'ΔVRAM%':>8}")
        print(hdr)
        print("─" * len(hdr))

        for vocab, seq in product(VOCABS, SEQS):
            tc_entry = next(
                ((ms, vr) for v, s, m, ms, vr in rows
                 if v == vocab and s == seq and m == "torch-compile"),
                None,
            )
            orda_entry = next(
                ((ms, vr) for v, s, m, ms, vr in rows
                 if v == vocab and s == seq and m == "orda"),
                None,
            )
            if tc_entry is None or orda_entry is None:
                continue
            tc_ms, tc_vr = tc_entry
            o_ms,  o_vr  = orda_entry
            if tc_ms == float("inf") or o_ms == float("inf"):
                print(f"{fmt_vocab(vocab):>6}  {seq:>5}"
                      f"  {'OOM':>10}  {'OOM':>8}  {'OOM':>10}  {'OOM':>8}")
            else:
                print(f"{fmt_vocab(vocab):>6}  {seq:>5}"
                      f"  {o_ms - tc_ms:>+10.2f}"
                      f"  {(o_ms / tc_ms - 1) * 100:>+7.1f}%"
                      f"  {o_vr - tc_vr:>+10.1f}"
                      f"  {(o_vr / tc_vr - 1) * 100:>+7.1f}%")


if __name__ == "__main__":
    main()
