import time
import gc

import torch
import torch.nn.functional as F

from .helper import Transformer, build_teacher_student, freeze_teacher, full_cleanup, peak_mb, reset_peak_memory, sync, trimmed_mean


def _tensor_bytes(*tensors: torch.Tensor | None) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    oom_markers = (
        "out of memory",
        "cuda oom",
        "defaultcpuallocator",
        "not enough memory",
        "cannot allocate memory",
        "bad allocation",
    )
    return isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) or any(marker in text for marker in oom_markers)


def _torch_compile_precomputed_would_oom(
    *,
    vocab: int,
    seq: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    teacher_cache_bytes: int,
) -> bool:
    if device.type != "cuda":
        return False

    full_logits_bytes = int(batch) * int(seq) * int(vocab) * torch.empty((), dtype=dtype).element_size()
    estimated_bytes = 4 * full_logits_bytes + teacher_cache_bytes
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    return estimated_bytes > int(total_bytes * 0.90)


def _release_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _build_student(vocab: int, student_config: dict, *, seq: int, device: torch.device) -> Transformer:
    return Transformer(vocab, student_config, seq=seq).to(dtype=torch.float32, device=device)


def build_precomputed_cache(
    *,
    vocab: int,
    seq: int,
    teacher_config: dict,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
    precomputed_source: str = "all",
) -> dict[str, torch.Tensor]:
    if precomputed_source not in ("all", "logits", "hidden_weight"):
        raise ValueError(f"Unknown precomputed_source: {precomputed_source}")

    x = torch.randint(0, vocab, (batch, seq), device=device)
    labels = torch.randint(0, vocab, (batch, seq), device=device)
    teacher = Transformer(vocab, teacher_config, seq=seq).to(dtype=torch.float32, device=device)
    freeze_teacher(teacher)
    if not use_amp:
        teacher.to(dtype=dtype)

    h_t = logits_t = teacher_weight = teacher_weight_cache = None
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        with torch.no_grad():
            h_t = teacher(x, return_hidden=True).view(-1, teacher_config["dim"]).detach().to(dtype)
            teacher_weight = teacher.head.weight.detach().to(dtype)
            if precomputed_source in ("all", "logits"):
                logits_t = F.linear(h_t, teacher_weight).detach()
            if precomputed_source in ("all", "hidden_weight"):
                teacher_weight_cache = teacher_weight.clone()

    cache = {
        "x": x.cpu(),
        "labels": labels.cpu(),
    }
    if precomputed_source in ("all", "logits"):
        cache["logits"] = logits_t.cpu()
    if precomputed_source in ("all", "hidden_weight"):
        cache["hidden"] = h_t.cpu()
        cache["weight"] = teacher_weight_cache.cpu()

    del x, labels, h_t, logits_t, teacher_weight, teacher_weight_cache, teacher
    _release_cuda_cache(device)
    return cache


def make_step(
    teacher_raw,
    student_raw,
    opt,
    scaler,
    *,
    vocab: int,
    seq: int,
    mode: str,
    batch: int,
    teacher_dim: int,
    temp: float,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool = True,
    teacher_mode: str = "tied",
    precomputed_source: str = "logits",
    precomputed_cache: dict[str, torch.Tensor] | None = None,
):
    if precomputed_cache is not None:
        x = precomputed_cache["x"].to(device=device, non_blocking=True)
        labels = precomputed_cache["labels"].to(device=device, non_blocking=True)
    else:
        x = torch.randint(0, vocab, (batch, seq), device=device)
        labels = torch.randint(0, vocab, (batch, seq), device=device)

    if teacher_mode == "precomputed" and precomputed_source not in ("logits", "hidden_weight"):
        raise ValueError(f"Unknown precomputed_source: {precomputed_source}")
    if teacher_mode == "precomputed" and precomputed_cache is None:
        raise ValueError("precomputed benchmark requires precomputed_cache.")

    if mode == "torch-compile":
        labels_flat = labels.view(-1)

        if teacher_mode == "precomputed":
            if precomputed_source == "logits":
                logits_t_precomputed = precomputed_cache["logits"].to(device=device, non_blocking=True)
                teacher_cache_bytes = _tensor_bytes(logits_t_precomputed)

                def forward_and_loss_precomputed(x_in, labels_flat, logits_t):
                    h_s = student_raw(x_in, return_hidden=True).view(-1, teacher_dim)
                    weight = student_raw.head.weight
                    logits_s = F.linear(h_s, weight)
                    ce = F.cross_entropy(logits_s, labels_flat)
                    log_p_s = F.log_softmax(logits_s / temp, dim=-1)
                    p_t = F.softmax((logits_t / temp).detach(), dim=-1)
                    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temp * temp)
                    return ce + kl

                compiled_fn = torch.compile(forward_and_loss_precomputed, mode="default")

                def step():
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                        loss = compiled_fn(x, labels_flat, logits_t_precomputed)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                step._teacher_cache_bytes = teacher_cache_bytes
                return step

            h_t_precomputed = precomputed_cache["hidden"].to(device=device, non_blocking=True)
            teacher_weight_precomputed = precomputed_cache["weight"].to(device=device, non_blocking=True)
            teacher_cache_bytes = _tensor_bytes(h_t_precomputed, teacher_weight_precomputed)

            def forward_and_loss_precomputed_hidden(x_in, labels_flat, h_t, w_t):
                h_s = student_raw(x_in, return_hidden=True).view(-1, teacher_dim)
                weight = student_raw.head.weight
                logits_s = F.linear(h_s, weight)
                logits_t = F.linear(h_t, w_t).detach()
                ce = F.cross_entropy(logits_s, labels_flat, ignore_index=-100)
                log_p_s = F.log_softmax(logits_s / temp, dim=-1)
                p_t = F.softmax(logits_t / temp, dim=-1)
                kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temp * temp)
                return ce + kl

            compiled_fn = torch.compile(forward_and_loss_precomputed_hidden, mode="default")

            def step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    loss = compiled_fn(
                        x,
                        labels_flat,
                        h_t_precomputed,
                        teacher_weight_precomputed,
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            step._teacher_cache_bytes = teacher_cache_bytes
            return step

        def forward_and_loss(x_in, labels_flat):
            with torch.no_grad():
                h_t = teacher_raw(x_in, return_hidden=True).view(-1, teacher_dim)
            h_s = student_raw(x_in, return_hidden=True).view(-1, teacher_dim)
            weight = student_raw.head.weight
            logits_s = F.linear(h_s, weight)
            if teacher_mode == "tied":
                logits_t = F.linear(h_t, weight)
                ce = F.cross_entropy(logits_s, labels_flat) + F.cross_entropy(logits_t, labels_flat)
            elif teacher_mode in ("separate_student", "separate_full"):
                logits_t = F.linear(h_t, teacher_raw.head.weight)
                if teacher_mode == "separate_full":
                    ce = F.cross_entropy(logits_s, labels_flat) + F.cross_entropy(logits_t, labels_flat)
                else:
                    ce = F.cross_entropy(logits_s, labels_flat)
            else:
                raise ValueError(f"Unknown teacher_mode: {teacher_mode}")
            log_p_s = F.log_softmax(logits_s / temp, dim=-1)
            p_t = F.softmax((logits_t / temp).detach(), dim=-1)
            kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temp * temp)
            return ce + kl

        compiled_fn = torch.compile(forward_and_loss, mode="default")

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                loss = compiled_fn(x, labels_flat)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        return step

    if mode == "orda":
        from orda_ce_kernel import PrecomputedTeacher, SeparateTeacher, TiedTeacher, distillation_loss

        student = torch.compile(student_raw, mode="default")
        head_weight = student_raw.head.weight
        labels_flat = labels.view(-1)
        teacher_cache_bytes = 0
        precomputed_config = None

        if teacher_mode == "precomputed":
            if precomputed_source == "logits":
                logits_t_precomputed = precomputed_cache["logits"].to(device=device, non_blocking=True)
                precomputed_teacher_arg = PrecomputedTeacher(logits=logits_t_precomputed)
                teacher_cache_bytes = _tensor_bytes(logits_t_precomputed)
            else:
                h_t_precomputed = precomputed_cache["hidden"].to(device=device, non_blocking=True)
                teacher_weight_precomputed = precomputed_cache["weight"].to(device=device, non_blocking=True)
                precomputed_teacher_arg = PrecomputedTeacher(
                    teacher_hidden=h_t_precomputed,
                    teacher_weight=teacher_weight_precomputed,
                )
                teacher_cache_bytes = _tensor_bytes(h_t_precomputed, teacher_weight_precomputed)
                precomputed_config = None

            def step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    h_s = student(x, return_hidden=True).view(-1, teacher_dim)
                    out = distillation_loss(
                        h_s,
                        head_weight,
                        labels_flat,
                        precomputed_teacher_arg,
                        student_ce_weight=1.0,
                        teacher_ce_weight=0.0,
                        kl_weight=1.0,
                        kl_temperature=temp,
                        backend="triton",
                        config=precomputed_config,
                    )
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            step._teacher_cache_bytes = teacher_cache_bytes
            return step

        teacher = torch.compile(teacher_raw, mode="default")
        teacher_weight = teacher_raw.head.weight

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                with torch.no_grad():
                    h_t = teacher(x, return_hidden=True).view(-1, teacher_dim)
                h_s = student(x, return_hidden=True).view(-1, teacher_dim)
                if teacher_mode == "tied":
                    teacher_arg = TiedTeacher(h_t)
                    teacher_ce_weight = None
                elif teacher_mode == "separate_student":
                    teacher_arg = SeparateTeacher(h_t, teacher_weight)
                    teacher_ce_weight = 0.0
                elif teacher_mode == "separate_full":
                    teacher_arg = SeparateTeacher(h_t, teacher_weight)
                    teacher_ce_weight = 1.0
                else:
                    raise ValueError(f"Unknown teacher_mode: {teacher_mode}")
                out = distillation_loss(
                    h_s,
                    head_weight,
                    labels_flat,
                    teacher_arg,
                    student_ce_weight=1.0,
                    teacher_ce_weight=teacher_ce_weight,
                    kl_weight=1.0,
                    kl_temperature=temp,
                    backend="triton",
                    config=precomputed_config,
                )
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        step._teacher_cache_bytes = teacher_cache_bytes
        return step

    raise ValueError(f"Unknown mode: {mode}")


def run_mode_isolated(
    *,
    vocab: int,
    seq: int,
    mode: str,
    teacher_config: dict,
    student_config: dict,
    batch: int,
    warmup: int,
    update_steps: int,
    temp: float,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool = True,
    use_grad_scaler: bool = True,
    teacher_mode: str = "tied",
    precomputed_source: str = "logits",
    precomputed_cache: dict[str, torch.Tensor] | None = None,
    return_details: bool = False,
) -> tuple[float, float] | tuple[float, float, float, float]:
    full_cleanup(device)
    if teacher_mode == "precomputed" and mode == "torch-compile" and precomputed_cache is not None:
        if precomputed_source == "logits":
            teacher_cache_bytes = _tensor_bytes(precomputed_cache["logits"])
        else:
            teacher_cache_bytes = _tensor_bytes(precomputed_cache["hidden"], precomputed_cache["weight"])
        if _torch_compile_precomputed_would_oom(
            vocab=vocab,
            seq=seq,
            batch=batch,
            dtype=dtype,
            device=device,
            teacher_cache_bytes=teacher_cache_bytes,
        ):
            if return_details:
                teacher_cache_mb = teacher_cache_bytes / 1024**2
                return float("inf"), float("inf"), teacher_cache_mb, float("inf")
            return float("inf"), float("inf")

    if teacher_mode == "precomputed" and precomputed_cache is not None:
        teacher_raw = None
        student_raw = _build_student(vocab, student_config, seq=seq, device=device)
    else:
        teacher_raw, student_raw = build_teacher_student(
            vocab,
            teacher_config,
            student_config,
            seq=seq,
            device=device,
        )
    if not use_amp:
        if teacher_raw is not None:
            teacher_raw.to(dtype=dtype)
        student_raw.to(dtype=dtype)
    opt = torch.optim.SGD(student_raw.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_grad_scaler))

    try:
        step = make_step(
            teacher_raw,
            student_raw,
            opt,
            scaler,
            vocab=vocab,
            seq=seq,
            mode=mode,
            batch=batch,
            teacher_dim=teacher_config["dim"],
            temp=temp,
            dtype=dtype,
            device=device,
            use_amp=use_amp,
            teacher_mode=teacher_mode,
            precomputed_source=precomputed_source,
            precomputed_cache=precomputed_cache,
        )
        if teacher_mode == "precomputed":
            teacher_raw = None
            _release_cuda_cache(device)

        for _ in range(warmup):
            step()
        sync(device)

        reset_peak_memory(device)
        sync(device)

        samples = []
        if device.type == "cuda":
            for _ in range(update_steps):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                step()
                end_event.record()
                sync(device)
                samples.append(start_event.elapsed_time(end_event))
            ms = trimmed_mean(samples, 0.10)
        else:
            for _ in range(update_steps):
                t0 = time.perf_counter()
                step()
                sync(device)
                samples.append((time.perf_counter() - t0) * 1_000)
            ms = trimmed_mean(samples, 0.10)

        teacher_cache_mb = getattr(
            step,
            "_teacher_cache_bytes",
            0,
        ) / 1024**2
        total_vram_mb = peak_mb(device)
        work_vram_mb = max(0.0, total_vram_mb - teacher_cache_mb)
        if return_details:
            return (
                ms,
                work_vram_mb,
                teacher_cache_mb,
                total_vram_mb,
            )
        return ms, work_vram_mb

    except Exception as exc:
        if not _is_oom_error(exc):
            raise
        if return_details:
            return float("inf"), float("inf"), float("inf"), float("inf")
        return float("inf"), float("inf")
    finally:
        del teacher_raw, student_raw, opt
        full_cleanup(device)
