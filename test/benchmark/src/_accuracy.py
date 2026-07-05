from __future__ import annotations

from dataclasses import dataclass

import torch

from .helper import fmt_vocab, print_table, sync


try:
    from orda_ce_kernel import (
        KernelConfig,
        PrecomputedTeacher,
        SeparateTeacher,
        TiedTeacher,
        distillation_loss,
        is_available as _orda_is_available,
    )
    from orda_ce_kernel.utils.dispatcher import clear_chunk_cache

    ORDA_AVAILABLE = _orda_is_available()
    ORDA_IMPORT_ERROR = None
except Exception as exc:
    ORDA_AVAILABLE = False
    ORDA_IMPORT_ERROR = repr(exc)


TEMP = 1.7
IGNORE_INDEX = -100
GRAD_SCALE = 8192.0
ACCURACY_BATCH_SEQS = [
    (32, 256),
    (16, 512),
    (8, 1_024),
]
ACCURACY_VOCAB = 32_768
MODES = [
    "tied",
    "separate-student",
    "separate-full",
    "precomputed-logits",
    "precomputed-hidden",
]


def accuracy_configs(profile: dict) -> list[dict]:
    configs = []
    for batch, seq in ACCURACY_BATCH_SEQS:
        configs.append(
            dict(
                batch=batch,
                seq=seq,
                BT=batch * seq,
                Hs=profile["student_config"]["dim"],
                Ht=profile["teacher_config"]["dim"],
                V=ACCURACY_VOCAB,
            )
        )
    return configs


@dataclass
class Case:
    mode: str
    student_hidden: torch.Tensor
    weight: torch.Tensor
    labels: torch.Tensor
    teacher: object
    ref_mode: str
    teacher_hidden: torch.Tensor | None = None
    teacher_weight: torch.Tensor | None = None
    teacher_logits: torch.Tensor | None = None
    teacher_ce_weight: float | None = None


def fmt_config(config: dict) -> str:
    return (
        f"vocab={fmt_vocab(config['V'])}  seq={config['seq']}  "
        f"batch={config['batch']}  dim={config['Hs']}"
    )


def _labels(torch_mod, BT: int, V: int, *, device: torch.device) -> torch.Tensor:
    labels = torch_mod.randint(0, V, (BT,), device=device)
    labels[:: max(1, BT // 8)] = IGNORE_INDEX
    return labels


def _randn(torch_mod, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch_mod.randn(*shape, device=device, dtype=dtype) * 0.1


def make_case(mode: str, config: dict, *, dtype: torch.dtype, device: torch.device, seed: int) -> Case:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    BT, Hs, Ht, V = config["BT"], config["Hs"], config["Ht"], config["V"]
    labels = _labels(torch, BT, V, device=device)

    if mode == "tied":
        hidden = _randn(torch, (BT, Hs), dtype=dtype, device=device).requires_grad_(True)
        teacher_hidden = _randn(torch, (BT, Hs), dtype=dtype, device=device).requires_grad_(True)
        weight = _randn(torch, (V, Hs), dtype=dtype, device=device).requires_grad_(True)
        return Case(
            mode=mode,
            student_hidden=hidden,
            weight=weight,
            labels=labels,
            teacher=TiedTeacher(teacher_hidden),
            ref_mode="tied",
            teacher_hidden=teacher_hidden,
            teacher_ce_weight=None,
        )

    if mode in {"separate-student", "separate-full"}:
        hidden = _randn(torch, (BT, Hs), dtype=dtype, device=device).requires_grad_(True)
        teacher_hidden = _randn(torch, (BT, Ht), dtype=dtype, device=device)
        weight = _randn(torch, (V, Hs), dtype=dtype, device=device).requires_grad_(True)
        teacher_weight = _randn(torch, (V, Ht), dtype=dtype, device=device)
        teacher_ce_weight = 1.0 if mode == "separate-full" else 0.0
        return Case(
            mode=mode,
            student_hidden=hidden,
            weight=weight,
            labels=labels,
            teacher=SeparateTeacher(teacher_hidden, teacher_weight),
            ref_mode="separate",
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
            teacher_ce_weight=teacher_ce_weight,
        )

    if mode == "precomputed-logits":
        hidden = _randn(torch, (BT, Hs), dtype=dtype, device=device).requires_grad_(True)
        weight = _randn(torch, (V, Hs), dtype=dtype, device=device).requires_grad_(True)
        teacher_logits = _randn(torch, (BT, V), dtype=dtype, device=device)
        return Case(
            mode=mode,
            student_hidden=hidden,
            weight=weight,
            labels=labels,
            teacher=PrecomputedTeacher(logits=teacher_logits),
            ref_mode="precomputed",
            teacher_logits=teacher_logits,
            teacher_ce_weight=0.0,
        )

    if mode == "precomputed-hidden":
        hidden = _randn(torch, (BT, Hs), dtype=dtype, device=device).requires_grad_(True)
        weight = _randn(torch, (V, Hs), dtype=dtype, device=device).requires_grad_(True)
        teacher_hidden = _randn(torch, (BT, Ht), dtype=dtype, device=device)
        teacher_weight = _randn(torch, (V, Ht), dtype=dtype, device=device)
        return Case(
            mode=mode,
            student_hidden=hidden,
            weight=weight,
            labels=labels,
            teacher=PrecomputedTeacher(teacher_hidden=teacher_hidden, teacher_weight=teacher_weight),
            ref_mode="precomputed",
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
            teacher_ce_weight=0.0,
        )

    raise ValueError(f"Unknown mode: {mode}")


def _actual_component(out, component: str) -> torch.Tensor:
    if component == "ce":
        return out.student_ce + out.teacher_ce
    if component == "kl":
        return out.kl
    raise ValueError(f"Unknown component: {component}")


def _component_weights(case: Case, component: str) -> tuple[float, float | None, float]:
    if component == "ce":
        return 1.0, case.teacher_ce_weight, 0.0
    if component == "kl":
        return 0.0, 0.0, 1.0
    raise ValueError(f"Unknown component: {component}")


def _native_pytorch_component(case: Case, component: str, *, backend: str, grad_scale: float = 1.0):
    student_ce_weight, teacher_ce_weight, kl_weight = _component_weights(case, component)
    out = distillation_loss(
        case.student_hidden,
        case.weight,
        case.labels,
        case.teacher,
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
        kl_temperature=TEMP,
        backend=backend,
        config=KernelConfig(chunk_size="dynamic") if backend == "triton" else None,
    )
    (out.loss * grad_scale).backward()
    return _actual_component(out, component).detach(), case.student_hidden.grad.detach().float() / grad_scale


def _components(logits_s, logits_t, labels, teacher_ce_weight: float):
    F = torch.nn.functional
    mask = labels != IGNORE_INDEX
    denom = torch.clamp(mask.sum(), min=1)
    ce_s = F.cross_entropy(
        logits_s,
        labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).sum() / denom
    if teacher_ce_weight != 0.0:
        ce_t = F.cross_entropy(
            logits_t,
            labels,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).sum() / denom
    else:
        ce_t = ce_s.new_zeros(())

    temp = float(TEMP)
    log_p_s = F.log_softmax(logits_s / temp, dim=-1)
    p_t = F.softmax(logits_t.detach() / temp, dim=-1)
    kl_all = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * (temp * temp)
    kl = kl_all.masked_fill(~mask, 0.0).sum() / denom
    return ce_s + ce_t, kl


def _torch_compile_component(
    case: Case,
    *,
    component: str,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
    grad_scale: float = 1.0,
):
    mode = case.mode
    if mode == "tied":
        def loss_fn(h_s, h_t, weight, labels):
            logits_s = h_s @ weight.t()
            logits_t = h_t @ weight.t()
            return _components(logits_s, logits_t, labels, 1.0)

        compiled = torch.compile(loss_fn, mode="default")
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            ce, kl = compiled(case.student_hidden, case.teacher_hidden, case.weight, case.labels)
        loss = ce if component == "ce" else kl
        (loss * grad_scale).backward()
        return loss.detach(), case.student_hidden.grad.detach().float() / grad_scale

    if mode in {"separate-student", "separate-full"}:
        teacher_ce_weight = 1.0 if mode == "separate-full" else 0.0

        def loss_fn(h_s, h_t, weight, teacher_weight, labels):
            logits_s = h_s @ weight.t()
            logits_t = h_t @ teacher_weight.t()
            return _components(logits_s, logits_t, labels, teacher_ce_weight)

        compiled = torch.compile(loss_fn, mode="default")
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            ce, kl = compiled(
                case.student_hidden,
                case.teacher_hidden,
                case.weight,
                case.teacher_weight,
                case.labels,
            )
        loss = ce if component == "ce" else kl
        (loss * grad_scale).backward()
        return loss.detach(), case.student_hidden.grad.detach().float() / grad_scale

    if mode == "precomputed-logits":
        def loss_fn(h_s, weight, logits_t, labels):
            logits_s = h_s @ weight.t()
            return _components(logits_s, logits_t, labels, 0.0)

        compiled = torch.compile(loss_fn, mode="default")
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            ce, kl = compiled(case.student_hidden, case.weight, case.teacher_logits, case.labels)
        loss = ce if component == "ce" else kl
        (loss * grad_scale).backward()
        return loss.detach(), case.student_hidden.grad.detach().float() / grad_scale

    if mode == "precomputed-hidden":
        def loss_fn(h_s, weight, h_t, teacher_weight, labels):
            logits_s = h_s @ weight.t()
            logits_t = h_t @ teacher_weight.t()
            return _components(logits_s, logits_t, labels, 0.0)

        compiled = torch.compile(loss_fn, mode="default")
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            ce, kl = compiled(
                case.student_hidden,
                case.weight,
                case.teacher_hidden,
                case.teacher_weight,
                case.labels,
            )
        loss = ce if component == "ce" else kl
        (loss * grad_scale).backward()
        return loss.detach(), case.student_hidden.grad.detach().float() / grad_scale

    raise ValueError(f"Unknown mode: {mode}")


def _fmt_float(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.6e}"


def _grad_metrics(expected: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float]:
    exp = expected.detach().float().flatten()
    act = actual.detach().float().flatten()
    cosine = torch.nn.functional.cosine_similarity(exp, act, dim=0, eps=1e-12).item()
    diff = (exp - act).abs()
    return cosine, diff.mean().item(), diff.max().item()


def _backend_na(value: str) -> list[str]:
    return [value, "NA", "NA", "NA", "NA", "NA"]


def _accuracy_row(config: dict, mode: str, baseline: str, torch_compile: list[str], orda: list[str]) -> list[str]:
    return [fmt_config(config), mode, baseline, *torch_compile, *orda]


def measure_accuracy_rows(
    *,
    component: str,
    configs: list[dict],
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
    use_grad_scaler: bool = False,
) -> list[list[str]]:
    rows: list[list[str]] = []
    grad_scale = GRAD_SCALE if use_grad_scaler else 1.0
    for cfg_idx, config in enumerate(configs):
        for mode_idx, mode in enumerate(MODES):
            seed = 10_000 + cfg_idx * 100 + mode_idx
            clear_chunk_cache()
            sync(device)
            try:
                ref_case = make_case(mode, config, dtype=torch.float32, device=device, seed=seed)
                expected, expected_grad = _native_pytorch_component(ref_case, component, backend="torch")
                expected = expected.detach().double().cpu()
                expected_grad = expected_grad.detach().float().cpu()
            except torch.cuda.OutOfMemoryError:
                rows.append(_accuracy_row(config, mode, "OOM", _backend_na("NA"), _backend_na("NA")))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            sync(device)
            denom = max(abs(float(expected.item())), 1e-12)

            try:
                tc_case = make_case(mode, config, dtype=dtype, device=device, seed=seed)
                torch_compile_actual, torch_compile_grad = _torch_compile_component(
                    tc_case,
                    component=component,
                    dtype=dtype,
                    device=device,
                    use_amp=use_amp,
                    grad_scale=grad_scale,
                )
                sync(device)
                torch_compile_actual = torch_compile_actual.detach().double().cpu()
                torch_compile_grad = torch_compile_grad.detach().float().cpu()
                tc_abs_err = abs(float(torch_compile_actual.item() - expected.item()))
                tc_grad_cos, tc_grad_mean, tc_grad_max = _grad_metrics(expected_grad, torch_compile_grad)
                tc_value = _fmt_float(float(torch_compile_actual.item()))
                tc_abs = _fmt_float(tc_abs_err)
                tc_rel = _fmt_float(tc_abs_err / denom)
                tc_gcos = f"{tc_grad_cos:.8f}"
                tc_gmean = _fmt_float(tc_grad_mean)
                tc_gmax = _fmt_float(tc_grad_max)
                torch_compile_cols = [tc_value, tc_abs, tc_rel, tc_gcos, tc_gmean, tc_gmax]
                del tc_case, torch_compile_actual, torch_compile_grad
            except torch.cuda.OutOfMemoryError:
                torch_compile_cols = _backend_na("OOM")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            try:
                orda_case = make_case(mode, config, dtype=dtype, device=device, seed=seed)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    orda_actual, orda_grad = _native_pytorch_component(
                        orda_case,
                        component,
                        backend="triton",
                        grad_scale=grad_scale,
                    )
                sync(device)
                orda_actual = orda_actual.detach().double().cpu()
                orda_grad = orda_grad.detach().float().cpu()
                orda_abs_err = abs(float(orda_actual.item() - expected.item()))
                orda_grad_cos, orda_grad_mean, orda_grad_max = _grad_metrics(expected_grad, orda_grad)
                orda_value = _fmt_float(float(orda_actual.item()))
                orda_abs = _fmt_float(orda_abs_err)
                orda_rel = _fmt_float(orda_abs_err / denom)
                orda_gcos = f"{orda_grad_cos:.8f}"
                orda_gmean = _fmt_float(orda_grad_mean)
                orda_gmax = _fmt_float(orda_grad_max)
                orda_cols = [orda_value, orda_abs, orda_rel, orda_gcos, orda_gmean, orda_gmax]
                del orda_case, orda_actual, orda_grad
            except torch.cuda.OutOfMemoryError:
                orda_cols = _backend_na("OOM")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            rows.append(_accuracy_row(config, mode, _fmt_float(float(expected.item())), torch_compile_cols, orda_cols))

            del ref_case, expected, expected_grad
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def print_accuracy_table(rows: list[list[str]]) -> None:
    loss_rows = [[row[idx] for idx in (0, 1, 2, 3, 4, 5, 9, 10, 11)] for row in rows]
    grad_rows = [[row[idx] for idx in (0, 1, 6, 7, 8, 12, 13, 14)] for row in rows]
    print_table(
        [
            "config",
            "mode",
            "fp32",
            "torch-compile",
            "tc_abs_err",
            "tc_rel_err",
            "orda",
            "orda_abs_err",
            "orda_rel_err",
        ],
        loss_rows,
        title="loss",
        align_right={2, 3, 4, 5, 6, 7, 8},
    )
    print_table(
        [
            "config",
            "mode",
            "tc_grad_cos",
            "tc_grad_mean",
            "tc_grad_max",
            "orda_grad_cos",
            "orda_grad_mean",
            "orda_grad_max",
        ],
        grad_rows,
        title="grad student_hidden",
        align_right={2, 3, 4, 5, 6, 7},
    )


def max_errors(rows: list[list[str]]) -> tuple[float, float, float, float]:
    def parse(value: str) -> float | None:
        if value in {"NA", "OOM"}:
            return None
        return float(value) if value != "0" else 0.0

    tc_max_abs = 0.0
    tc_max_rel = 0.0
    orda_max_abs = 0.0
    orda_max_rel = 0.0
    for row in rows:
        tc_abs = parse(row[4])
        tc_rel = parse(row[5])
        orda_abs = parse(row[10])
        orda_rel = parse(row[11])
        if tc_abs is not None:
            tc_max_abs = max(tc_max_abs, tc_abs)
        if tc_rel is not None:
            tc_max_rel = max(tc_max_rel, tc_rel)
        if orda_abs is not None:
            orda_max_abs = max(orda_max_abs, orda_abs)
        if orda_rel is not None:
            orda_max_rel = max(orda_max_rel, orda_rel)
    return tc_max_abs, tc_max_rel, orda_max_abs, orda_max_rel
