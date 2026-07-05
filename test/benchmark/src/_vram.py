import torch
import torch.nn.functional as F

from .helper import fmt_vocab, full_cleanup, peak_extra_mb, sync


# ── Shared benchmark config ─────────────────────────────────────────────────
BATCH = 16
VOCABS = [16_384, 32_768, 65_536]
SEQS = [256, 512, 1024, 2048]
DIMS = [512, 1024]
WARMUP = 1
MEASURE_ITERS = 1
TEMP = 3.0
NUM_CHUNKS = 16
CONFIG_W = len("vocab=64k  seq=2048")
TORCH_W = len("99999.9 MB")
MB_W = len("torch-compile")
ORDA_W = len("99999.9 MB")
DELTA_TORCH_W = len("deltaTorch")
DELTA_COMPILE_W = len("deltaCompile")


def fmt_backend_mb(value: float) -> str:
    return "OOM" if value == float("inf") else f"{value:.1f} MB"


def fmt_delta(baseline_mb: float, orda_mb: float) -> str:
    if baseline_mb == float("inf") or orda_mb == float("inf"):
        return "NA"
    if baseline_mb == 0.0:
        return "NA"
    return f"{(orda_mb / baseline_mb - 1) * 100:+.1f}%"


def print_backend_header() -> None:
    print(
        f"{'config':<{CONFIG_W}}  "
        f"{'torch':^{TORCH_W}}  "
        f"{'torch-compile':^{MB_W}}  "
        f"{'orda':^{ORDA_W}}  "
        f"{'deltaTorch':^{DELTA_TORCH_W}}  "
        f"{'deltaCompile':^{DELTA_COMPILE_W}}"
    )


def print_backend_row(
    vocab: int,
    seq: int,
    rows: list[tuple[int, int, str, dict]],
    *,
    torch_mode: str,
    compile_mode: str,
    orda_mode: str,
) -> None:
    compile_entry = next((r for v, s, mode, r in rows if v == vocab and s == seq and mode == compile_mode), None)
    if compile_entry is None:
        return
    torch_entry = next((r for v, s, mode, r in rows if v == vocab and s == seq and mode == torch_mode), None)
    torch_text = "NA" if torch_entry is None else fmt_backend_mb(torch_entry["backend_extra_mb"])
    torch_mb = None if torch_entry is None else torch_entry["backend_extra_mb"]
    compile_mb = compile_entry["backend_extra_mb"]
    orda_entry = next((r for v, s, mode, r in rows if v == vocab and s == seq and mode == orda_mode), None)
    if orda_entry is None:
        orda_text = "NA"
        delta_torch_text = "NA"
        delta_compile_text = "NA"
    else:
        orda_mb = orda_entry["backend_extra_mb"]
        orda_text = fmt_backend_mb(orda_mb)
        delta_torch_text = "NA" if torch_mb is None else fmt_delta(torch_mb, orda_mb)
        delta_compile_text = fmt_delta(compile_mb, orda_mb)
    config = f"vocab={fmt_vocab(vocab)}  seq={seq}"
    print(
        f"{config:<{CONFIG_W}}  "
        f"{torch_text:^{TORCH_W}}  "
        f"{fmt_backend_mb(compile_mb):^{MB_W}}  "
        f"{orda_text:^{ORDA_W}}  "
        f"{delta_torch_text:^{DELTA_TORCH_W}}  "
        f"{delta_compile_text:^{DELTA_COMPILE_W}}"
    )


try:
    from orda_ce_kernel import (
        KernelConfig,
        PrecomputedTeacher,
        SeparateTeacher,
        TiedTeacher,
        distillation_loss,
        is_available as _orda_is_available,
    )

    ORDA_AVAILABLE = _orda_is_available()
except Exception:
    ORDA_AVAILABLE = False


def _tensor_bytes(*tensors: torch.Tensor | None) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


# ── Mode helpers ─────────────────────────────────────────────────────────────
def precomputed_backend(mode: str) -> str:
    if mode.startswith("torch-compile"):
        return "torch-compile"
    elif mode.startswith("torch"):
        return "torch"
    elif mode.startswith("orda"):
        return "orda"
    raise ValueError(f"Unknown mode: {mode}")


# ── TiedTeacher loss/input builders ──────────────────────────────────────────
def _make_tied_torch_loss_fn(temp: float, *, compile: bool):
    def loss_fn(h_s, h_t, head_weight, labels_flat):
        logits_s = F.linear(h_s, head_weight)
        logits_t = F.linear(h_t, head_weight)
        ce = F.cross_entropy(logits_s, labels_flat) + F.cross_entropy(logits_t, labels_flat)
        log_p_s = F.log_softmax(logits_s / temp, dim=-1)
        p_t = F.softmax((logits_t / temp).detach(), dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temp * temp)
        return ce + kl

    return torch.compile(loss_fn, mode="default") if compile else loss_fn


def _make_tied_orda_loss_fn(temp: float):
    def loss_fn(h_s, h_t, head_weight, labels_flat):
        return distillation_loss(
            h_s,
            head_weight,
            labels_flat,
            TiedTeacher(h_t),
            student_ce_weight=1.0,
            kl_weight=1.0,
            kl_temperature=temp,
            backend="triton",
            config=KernelConfig(num_chunks=NUM_CHUNKS),
        ).loss

    return loss_fn


def _make_tied_inputs(
    vocab: int,
    dim: int,
    seq: int,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
):
    n_rows = batch * seq
    h_s = torch.randn(n_rows, dim, device=device, dtype=dtype)
    h_t = torch.randn(n_rows, dim, device=device, dtype=dtype)
    head_weight = torch.randn(vocab, dim, device=device, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, vocab, (n_rows,), device=device)
    return h_s, h_t, head_weight, labels


def _run_tied_loss_only(
    loss_fn,
    h_s,
    h_t,
    head_weight,
    labels_flat,
    *,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> None:
    h_s_loss = h_s.detach().requires_grad_(True)
    h_t_loss = h_t.detach()
    h_s_loss.grad = None
    head_weight.grad = None
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        loss = loss_fn(h_s_loss, h_t_loss, head_weight, labels_flat)
    loss.backward()
    sync(device)


def _warmup_tied_once(
    loss_fn,
    vocab: int,
    dim: int,
    seq: int,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> None:
    h_s, h_t, head_weight, labels = _make_tied_inputs(
        vocab,
        dim,
        seq,
        batch=batch,
        dtype=dtype,
        device=device,
    )
    _run_tied_loss_only(
        loss_fn,
        h_s,
        h_t,
        head_weight,
        labels,
        dtype=dtype,
        device=device,
        use_amp=use_amp,
    )
    head_weight.grad = None
    del labels, h_s, h_t, head_weight
    sync(device)


# ── SeparateTeacher loss/input builders ──────────────────────────────────────
def _make_separate_torch_loss_fn(temp: float, teacher_ce_weight: float, *, compile: bool):
    def loss_fn(h_s, h_t, student_weight, teacher_weight, labels_flat):
        logits_s = F.linear(h_s, student_weight)
        logits_t = F.linear(h_t, teacher_weight)
        ce = F.cross_entropy(logits_s, labels_flat)
        if teacher_ce_weight != 0.0:
            ce = ce + F.cross_entropy(logits_t, labels_flat)
        log_p_s = F.log_softmax(logits_s / temp, dim=-1)
        p_t = F.softmax((logits_t / temp).detach(), dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temp * temp)
        return ce + kl

    return torch.compile(loss_fn, mode="default") if compile else loss_fn


def _make_separate_orda_loss_fn(temp: float, teacher_ce_weight: float):
    def loss_fn(h_s, h_t, student_weight, teacher_weight, labels_flat):
        return distillation_loss(
            h_s,
            student_weight,
            labels_flat,
            SeparateTeacher(h_t, teacher_weight),
            student_ce_weight=1.0,
            teacher_ce_weight=teacher_ce_weight,
            kl_weight=1.0,
            kl_temperature=temp,
            backend="triton",
            config=KernelConfig(num_chunks=NUM_CHUNKS),
        ).loss

    return loss_fn


def _make_separate_inputs(
    vocab: int,
    dim: int,
    seq: int,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
):
    n_rows = batch * seq
    h_s = torch.randn(n_rows, dim, device=device, dtype=dtype)
    h_t = torch.randn(n_rows, dim, device=device, dtype=dtype)
    student_weight = torch.randn(vocab, dim, device=device, dtype=dtype, requires_grad=True)
    teacher_weight = torch.randn(vocab, dim, device=device, dtype=dtype)
    labels = torch.randint(0, vocab, (n_rows,), device=device)
    return h_s, h_t, student_weight, teacher_weight, labels


def _run_separate_loss_only(
    loss_fn,
    h_s,
    h_t,
    student_weight,
    teacher_weight,
    labels_flat,
    *,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> None:
    h_s_loss = h_s.detach().requires_grad_(True)
    h_t_loss = h_t.detach()
    h_s_loss.grad = None
    student_weight.grad = None
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        loss = loss_fn(h_s_loss, h_t_loss, student_weight, teacher_weight, labels_flat)
    loss.backward()
    sync(device)


def _warmup_separate_once(
    loss_fn,
    vocab: int,
    dim: int,
    seq: int,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> None:
    h_s, h_t, student_weight, teacher_weight, labels = _make_separate_inputs(
        vocab,
        dim,
        seq,
        batch=batch,
        dtype=dtype,
        device=device,
    )
    _run_separate_loss_only(
        loss_fn,
        h_s,
        h_t,
        student_weight,
        teacher_weight,
        labels,
        dtype=dtype,
        device=device,
        use_amp=use_amp,
    )
    student_weight.grad = None
    del labels, h_s, h_t, student_weight, teacher_weight
    sync(device)


# ── PrecomputedTeacher loss/input builders ───────────────────────────────────
def _make_precomputed_torch_loss_fn(temp: float, *, compile: bool):
    def loss_fn(h_s, student_weight, teacher_cache, teacher_weight, labels_flat):
        logits_s = F.linear(h_s, student_weight)
        logits_t = F.linear(teacher_cache, teacher_weight).detach()
        ce = F.cross_entropy(logits_s, labels_flat, ignore_index=-100)
        log_p_s = F.log_softmax(logits_s / temp, dim=-1)
        p_t = F.softmax(logits_t / temp, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temp * temp)
        return ce + kl

    return torch.compile(loss_fn, mode="default") if compile else loss_fn


def _make_precomputed_orda_loss_fn(temp: float):
    def loss_fn(h_s, student_weight, teacher_cache, teacher_weight, labels_flat):
        teacher = PrecomputedTeacher(teacher_hidden=teacher_cache, teacher_weight=teacher_weight)
        return distillation_loss(
            h_s,
            student_weight,
            labels_flat,
            teacher,
            student_ce_weight=1.0,
            teacher_ce_weight=0.0,
            kl_weight=1.0,
            kl_temperature=temp,
            backend="triton",
            config=KernelConfig(num_chunks=NUM_CHUNKS),
        ).loss

    return loss_fn


def _make_precomputed_inputs(
    vocab: int,
    dim: int,
    seq: int,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
):
    n_rows = batch * seq
    h_s = torch.randn(n_rows, dim, device=device, dtype=dtype)
    student_weight = torch.randn(vocab, dim, device=device, dtype=dtype, requires_grad=True)
    teacher_cache = torch.randn(n_rows, dim, device=device, dtype=dtype)
    teacher_weight = torch.randn(vocab, dim, device=device, dtype=dtype)
    labels = torch.randint(0, vocab, (n_rows,), device=device)
    return h_s, student_weight, teacher_cache, teacher_weight, labels


def _run_precomputed_loss_only(
    loss_fn,
    h_s,
    student_weight,
    teacher_cache,
    teacher_weight,
    labels_flat,
    *,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> None:
    h_s_loss = h_s.detach().requires_grad_(True)
    teacher_cache_loss = teacher_cache.detach()
    h_s_loss.grad = None
    student_weight.grad = None
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        loss = loss_fn(h_s_loss, student_weight, teacher_cache_loss, teacher_weight, labels_flat)
    loss.backward()
    sync(device)


def _warmup_precomputed_once(
    loss_fn,
    vocab: int,
    dim: int,
    seq: int,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> None:
    h_s, student_weight, teacher_cache, teacher_weight, labels = _make_precomputed_inputs(
        vocab,
        dim,
        seq,
        batch=batch,
        dtype=dtype,
        device=device,
    )
    _run_precomputed_loss_only(
        loss_fn,
        h_s,
        student_weight,
        teacher_cache,
        teacher_weight,
        labels,
        dtype=dtype,
        device=device,
        use_amp=use_amp,
    )
    student_weight.grad = None
    del labels, h_s, student_weight, teacher_cache, teacher_weight
    sync(device)


# ── Measurement entry points ─────────────────────────────────────────────────
def measure_tied_once(
    *,
    vocab: int,
    seq: int,
    dim: int,
    mode: str,
    batch: int,
    warmup: int,
    measure_iters: int,
    temp: float,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> dict:
    full_cleanup(device)
    if mode == "torch-compile":
        loss_fn = _make_tied_torch_loss_fn(temp, compile=True)
    elif mode == "torch":
        loss_fn = _make_tied_torch_loss_fn(temp, compile=False)
    else:
        loss_fn = _make_tied_orda_loss_fn(temp)

    try:
        for _ in range(warmup):
            _warmup_tied_once(
                loss_fn,
                vocab,
                dim,
                seq,
                batch=batch,
                dtype=dtype,
                device=device,
                use_amp=use_amp,
            )

        h_s, h_t, head_weight, labels = _make_tied_inputs(
            vocab,
            dim,
            seq,
            batch=batch,
            dtype=dtype,
            device=device,
        )
        sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            base_loss = torch.cuda.memory_allocated(device)
        else:
            base_loss = 0
        for _ in range(measure_iters):
            _run_tied_loss_only(
                loss_fn,
                h_s,
                h_t,
                head_weight,
                labels,
                dtype=dtype,
                device=device,
                use_amp=use_amp,
            )
        loss_extra = peak_extra_mb(device, base_loss)

        return {
            "status": "ok",
            "backend_extra_mb": loss_extra,
        }

    except torch.cuda.OutOfMemoryError:
        return {
            "status": "oom",
            "backend_extra_mb": float("inf"),
        }
    finally:
        full_cleanup(device)


def measure_separate_once(
    *,
    vocab: int,
    seq: int,
    dim: int,
    mode: str,
    batch: int,
    warmup: int,
    measure_iters: int,
    temp: float,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
    teacher_ce_weight: float,
) -> dict:
    full_cleanup(device)
    if mode == "torch-compile":
        loss_fn = _make_separate_torch_loss_fn(temp, teacher_ce_weight, compile=True)
    elif mode == "torch":
        loss_fn = _make_separate_torch_loss_fn(temp, teacher_ce_weight, compile=False)
    else:
        loss_fn = _make_separate_orda_loss_fn(temp, teacher_ce_weight)

    try:
        for _ in range(warmup):
            _warmup_separate_once(
                loss_fn,
                vocab,
                dim,
                seq,
                batch=batch,
                dtype=dtype,
                device=device,
                use_amp=use_amp,
            )

        h_s, h_t, student_weight, teacher_weight, labels = _make_separate_inputs(
            vocab,
            dim,
            seq,
            batch=batch,
            dtype=dtype,
            device=device,
        )
        sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            base_loss = torch.cuda.memory_allocated(device)
        else:
            base_loss = 0
        for _ in range(measure_iters):
            _run_separate_loss_only(
                loss_fn,
                h_s,
                h_t,
                student_weight,
                teacher_weight,
                labels,
                dtype=dtype,
                device=device,
                use_amp=use_amp,
            )
        loss_extra = peak_extra_mb(device, base_loss)

        return {
            "status": "ok",
            "backend_extra_mb": loss_extra,
        }

    except torch.cuda.OutOfMemoryError:
        return {
            "status": "oom",
            "backend_extra_mb": float("inf"),
        }
    finally:
        full_cleanup(device)


def measure_precomputed_once(
    *,
    vocab: int,
    seq: int,
    dim: int,
    mode: str,
    batch: int,
    warmup: int,
    measure_iters: int,
    temp: float,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> dict:
    full_cleanup(device)
    backend = precomputed_backend(mode)

    try:
        if backend == "torch-compile":
            loss_fn = _make_precomputed_torch_loss_fn(temp, compile=True)
        elif backend == "torch":
            loss_fn = _make_precomputed_torch_loss_fn(temp, compile=False)
        else:
            loss_fn = _make_precomputed_orda_loss_fn(temp)

        for _ in range(warmup):
            _warmup_precomputed_once(
                loss_fn,
                vocab,
                dim,
                seq,
                batch=batch,
                dtype=dtype,
                device=device,
                use_amp=use_amp,
            )

        h_s, student_weight, teacher_cache, teacher_weight, labels = _make_precomputed_inputs(
            vocab,
            dim,
            seq,
            batch=batch,
            dtype=dtype,
            device=device,
        )
        teacher_cache_mb = _tensor_bytes(teacher_cache, teacher_weight) / 1024**2
        sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            base_loss = torch.cuda.memory_allocated(device)
        else:
            base_loss = 0
        for _ in range(measure_iters):
            _run_precomputed_loss_only(
                loss_fn,
                h_s,
                student_weight,
                teacher_cache,
                teacher_weight,
                labels,
                dtype=dtype,
                device=device,
                use_amp=use_amp,
            )
        loss_extra = peak_extra_mb(device, base_loss)

        return {
            "status": "ok",
            "backend_extra_mb": loss_extra,
            "cache_teacher_mb": teacher_cache_mb,
            "total_extra_mb": loss_extra + teacher_cache_mb,
        }

    except torch.cuda.OutOfMemoryError:
        return {
            "status": "oom",
            "backend_extra_mb": float("inf"),
            "cache_teacher_mb": float("inf"),
            "total_extra_mb": float("inf"),
        }
    finally:
        full_cleanup(device)
