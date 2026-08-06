from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F

from ..core.device import device_dtype, release_memory, seed_all
from ..core.metrics import fmt_precision
from ..losses.divergence import IGNORE_INDEX, kl_term
from ..model import ModelConfig, Transformer, freeze_teacher


class RefMode(str, Enum):
    __str__ = str.__str__

    TIED = "tied"
    SEPARATE = "separate"
    PRECOMPUTED_HIDDEN = "precomputed-hidden"


@dataclass
class Case:
    mode: str
    student_hidden: torch.Tensor
    weight: torch.Tensor
    labels: torch.Tensor
    teacher: object
    ref_mode: RefMode
    teacher_hidden: torch.Tensor | None = None
    teacher_weight: torch.Tensor | None = None
    student_ce_weight: float = 1.0
    teacher_ce_weight: float = 0.0
    kl_weight: float = 1.0
    kl_divergence: str = "forward"
    temperature: float = 1.7

    def reset_gradients(self) -> Case:
        for tensor in (
            self.student_hidden,
            self.weight,
            self.teacher_hidden,
            self.teacher_weight,
        ):
            if tensor is not None:
                tensor.grad = None
        return self


@dataclass(frozen=True)
class Gradients:
    names: tuple[str, ...]
    tensors: tuple[torch.Tensor, ...]
    scale: float | torch.Tensor = 1.0


def teacher_needs_grad(teacher_ce_weight: float, divergence: str) -> bool:
    return teacher_ce_weight != 0.0 or divergence in {"cojeffreys", "mutual"}


# ── Logits construction ──────────────────────────────────────────────────────
def logits_pair(
    case: Case,
    *,
    component: str,
    tensors: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Alias unused teacher logits to avoid a second (rows, vocab) allocation."""
    if tensors is None:
        student_hidden = case.student_hidden
        weight = case.weight
        teacher_hidden = case.teacher_hidden
        teacher_weight = case.teacher_weight
    else:
        student_hidden, weight, teacher_hidden, teacher_weight = tensors

    logits_s = student_hidden @ weight.t()

    if case.ref_mode is RefMode.TIED:
        if teacher_hidden is None:
            raise ValueError("tied accuracy case requires teacher_hidden")
        needs_teacher = component == "kl" or case.teacher_ce_weight != 0.0
        logits_t = teacher_hidden @ weight.t() if needs_teacher else logits_s
        return logits_s, logits_t, case.teacher_ce_weight

    if teacher_hidden is None or teacher_weight is None:
        raise ValueError(
            f"{case.ref_mode} accuracy case requires teacher_hidden and teacher_weight"
        )

    if case.ref_mode is RefMode.SEPARATE:
        needs_teacher = component == "kl" or case.teacher_ce_weight != 0.0
        logits_t = teacher_hidden @ teacher_weight.t() if needs_teacher else logits_s
        return logits_s, logits_t, case.teacher_ce_weight

    if case.ref_mode is RefMode.PRECOMPUTED_HIDDEN:
        # A precomputed teacher is never trained, so its CE never contributes.
        logits_t = (
            teacher_hidden @ teacher_weight.t() if component == "kl" else logits_s
        )
        return logits_s, logits_t, 0.0

    raise ValueError(f"Unknown ref_mode: {case.ref_mode}")


def case_tensors(case: Case) -> tuple[torch.Tensor, ...]:
    return (case.student_hidden, case.weight, case.teacher_hidden, case.teacher_weight)


# ── Component values ─────────────────────────────────────────────────────────
def components(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_ce_weight: float,
    teacher_ce_weight: float,
    kl_weight: float,
    kl_divergence: str,
    component: str,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use one valid-row denominator so CE and divergence remain comparable."""
    if component == "kl":
        return logits_s.new_zeros(()), kl_weight * kl_term(
            logits_s,
            logits_t,
            temperature=temperature,
            divergence=kl_divergence,
            labels=labels,
            early_row_scaling=True,
        )
    if component != "ce":
        raise ValueError(f"Unknown component: {component}")

    mask = labels != IGNORE_INDEX
    denominator = torch.clamp(mask.sum(), min=1)
    ce_s = (
        F.cross_entropy(
            logits_s, labels, ignore_index=IGNORE_INDEX, reduction="none"
        ).sum()
        / denominator
    )
    if teacher_ce_weight != 0.0:
        ce_t = (
            F.cross_entropy(
                logits_t, labels, ignore_index=IGNORE_INDEX, reduction="none"
            ).sum()
            / denominator
        )
    else:
        ce_t = ce_s.new_zeros(())
    return (
        student_ce_weight * ce_s + teacher_ce_weight * ce_t,
        logits_s.new_zeros(()),
    )


def active_gradients(case: Case, grad_scale: float | torch.Tensor) -> Gradients:
    names = []
    gradients = []
    for name, tensor in (
        ("student_hidden", case.student_hidden),
        ("student_weight", case.weight),
        ("teacher_hidden", case.teacher_hidden),
        ("teacher_weight", case.teacher_weight),
    ):
        if tensor is not None and tensor.grad is not None:
            names.append(name)
            gradients.append(tensor.grad.detach())
    if not gradients:
        raise RuntimeError(f"accuracy case {case.mode!r} produced no active gradient")
    return Gradients(tuple(names), tuple(gradients), grad_scale)


def backward_with_grad_scaler(
    loss: torch.Tensor, grad_scaler: torch.amp.GradScaler | None
) -> float | torch.Tensor:
    if grad_scaler is None or not grad_scaler.is_enabled():
        loss.backward()
        return 1.0
    grad_scale = grad_scaler.scale(
        torch.ones((), dtype=torch.float32, device=loss.device)
    )
    grad_scaler.scale(loss).backward()
    return grad_scale


def grad_scaler_for(
    device: torch.device, dtype: torch.dtype
) -> torch.amp.GradScaler | None:
    if dtype != torch.float16:
        return None
    return torch.amp.GradScaler(device.type)


# ── Backend execution ────────────────────────────────────────────────────────
def eager_reference_component(case: Case, *, component: str):
    logits_s, logits_t, teacher_ce_weight = logits_pair(case, component=component)
    ce, kl = components(
        logits_s,
        logits_t,
        case.labels,
        student_ce_weight=case.student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=case.kl_weight,
        kl_divergence=case.kl_divergence,
        component=component,
        temperature=case.temperature,
    )
    loss = ce if component == "ce" else kl
    loss.backward()
    return loss.detach(), active_gradients(case, 1.0)


def torch_component(
    case: Case,
    *,
    component: str,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
    compile_graph: bool,
    grad_scaler: torch.amp.GradScaler | None = None,
):
    def loss_fn(student_hidden, weight, teacher_hidden, teacher_weight, labels):
        logits_s, logits_t, teacher_ce_weight = logits_pair(
            case,
            component=component,
            tensors=(student_hidden, weight, teacher_hidden, teacher_weight),
        )
        return components(
            logits_s,
            logits_t,
            labels,
            student_ce_weight=case.student_ce_weight,
            teacher_ce_weight=teacher_ce_weight,
            kl_weight=case.kl_weight,
            kl_divergence=case.kl_divergence,
            component=component,
            temperature=case.temperature,
        )

    run_loss = torch.compile(loss_fn, mode="default") if compile_graph else loss_fn
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        ce, kl = run_loss(*case_tensors(case), case.labels)
    loss = ce if component == "ce" else kl
    grad_scale = backward_with_grad_scaler(loss, grad_scaler)
    return loss.detach(), active_gradients(case, grad_scale)


def orda_component(
    case: Case,
    component: str,
    *,
    backend: str,
    num_chunks: int,
    grad_scaler: torch.amp.GradScaler | None = None,
):
    from orda_kernel import distillation_loss

    from ..losses.orda_loss import kernel_config

    if component == "ce":
        weights = (case.student_ce_weight, case.teacher_ce_weight, 0.0)
    elif component == "kl":
        weights = (0.0, 0.0, case.kl_weight)
    else:
        raise ValueError(f"Unknown component: {component}")
    student_ce_weight, teacher_ce_weight, kl_weight = weights

    out = distillation_loss(
        case.student_hidden,
        case.weight,
        case.labels,
        case.teacher,
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
        kl_temperature=case.temperature,
        kl_divergence=case.kl_divergence,
        backend=backend,
        config=(kernel_config(num_chunks=num_chunks) if backend == "triton" else None),
        grad_scaler=grad_scaler,
    )
    grad_scale = backward_with_grad_scaler(out.loss, grad_scaler)
    if component == "ce":
        value = (
            case.student_ce_weight * out.student_ce
            + case.teacher_ce_weight * out.teacher_ce
        )
    else:
        value = case.kl_weight * out.kl
    return value.detach(), active_gradients(case, grad_scale)


# ── Metric scoring ───────────────────────────────────────────────────────────
def gradients_to_cpu(gradients: Gradients) -> Gradients:
    scale = gradients.scale
    if isinstance(scale, torch.Tensor):
        scale = float(scale.detach().float().cpu())
    else:
        scale = float(scale)
    tensors = []
    for tensor in gradients.tensors:
        copied = tensor.detach().float().cpu()
        if scale != 1.0:
            copied.div_(scale)
        tensors.append(copied)
    return Gradients(gradients.names, tuple(tensors))


def grad_metrics(
    expected: Gradients,
    actual: Gradients,
) -> tuple[float, float, float, int, int]:
    """Compare gradient blocks to avoid materializing a full FP64 vocabulary."""
    if expected.names != actual.names:
        expected_names = ", ".join(expected.names) or "none"
        actual_names = ", ".join(actual.names) or "none"
        raise RuntimeError(
            f"active gradient mismatch: expected [{expected_names}], "
            f"got [{actual_names}]"
        )
    dot = exp_norm = act_norm = 0.0
    diff_sum = diff_max = 0.0
    element_count = 0
    underflow = underflow_total = 0
    finite = True
    block = 1_048_576
    for name, exp_tensor, act_tensor in zip(
        expected.names,
        expected.tensors,
        actual.tensors,
        strict=True,
    ):
        exp = exp_tensor.flatten()
        act = act_tensor.flatten()
        if exp.shape != act.shape:
            raise RuntimeError(
                f"gradient shape mismatch for {name}: "
                f"expected {tuple(exp.shape)}, got {tuple(act.shape)}"
            )
        for start in range(0, exp.numel(), block):
            exp_block = exp[start : start + block]
            act_block = act[start : start + block]
            finite = finite and bool(
                torch.isfinite(exp_block).all() and torch.isfinite(act_block).all()
            )
            difference = (exp_block - act_block).abs()
            diff_sum += difference.double().sum().item()
            diff_max = max(diff_max, difference.max().item())
            element_count += difference.numel()
            expected_nonzero = exp_block != 0
            underflow += ((act_block == 0) & expected_nonzero).sum().item()
            underflow_total += expected_nonzero.sum().item()
            if finite:
                exp64 = exp_block.double()
                act64 = act_block.double()
                dot += torch.dot(exp64, act64).item()
                exp_norm += torch.dot(exp64, exp64).item()
                act_norm += torch.dot(act64, act64).item()
    grad_mean = diff_sum / max(element_count, 1)
    if not finite:
        return (
            math.nan,
            grad_mean,
            math.nan if math.isnan(diff_sum) else diff_max,
            underflow,
            underflow_total,
        )
    if exp_norm == 0.0 and act_norm == 0.0:
        cosine = 1.0
    elif exp_norm == 0.0 or act_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / math.sqrt(exp_norm * act_norm)
    cosine = min(1.0, max(-1.0, cosine))
    return cosine, grad_mean, diff_max, underflow, underflow_total


# ── Case construction ────────────────────────────────────────────────────────
def labels_with_ignored(rows: int, vocab: int, *, device: torch.device):
    labels = torch.randint(0, vocab, (rows,), device=device)
    labels[:: max(1, rows // 8)] = IGNORE_INDEX
    return labels


def case_model_inputs(
    spec,
    config: dict,
    *,
    profile,
    device: torch.device,
    seed: int,
):
    seed_all(device, seed)
    student_dim, teacher_dim = config["Hs"], config["Ht"]
    student_base = (
        profile.teacher_config if spec.family == "tied" else profile.student_config
    )
    student_config = ModelConfig.from_mapping(student_base).replace(dim=student_dim)
    teacher_config = profile.teacher_config.replace(dim=teacher_dim)
    teacher_model = Transformer(config["V"], teacher_config, seq=config["seq"]).to(
        device=device, dtype=torch.float32
    )
    student_model = Transformer(config["V"], student_config, seq=config["seq"]).to(
        device=device, dtype=torch.float32
    )
    freeze_teacher(teacher_model)
    if spec.family == "tied":
        if student_dim != teacher_dim:
            raise ValueError("tied accuracy requires equal hidden dimensions")
        teacher_model.head.weight = student_model.head.weight
    seed_all(device, seed + 1)
    tokens = torch.randint(
        0, config["V"], (config["batch"], config["seq"]), device=device
    )
    labels = labels_with_ignored(config["BT"], config["V"], device=device)
    return teacher_model, student_model, tokens, labels


def make_case(
    spec,
    config: dict,
    *,
    device: torch.device,
    model_inputs,
    temperature: float,
    use_amp: bool = False,
) -> Case:
    from orda_kernel import PrecomputedTeacher, SeparateTeacher, TiedTeacher

    rows, student_dim, teacher_dim = config["BT"], config["Hs"], config["Ht"]
    teacher_model, student_model, tokens, labels = model_inputs
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=device_dtype(device),
            enabled=use_amp,
        ),
    ):
        hidden = student_model(tokens, return_hidden=True).view(rows, student_dim)
        teacher_hidden = teacher_model(tokens, return_hidden=True).view(
            rows, teacher_dim
        )

    hidden = hidden.detach().requires_grad_(True)
    weight = student_model.head.weight.requires_grad_(spec.head_requires_grad)
    teacher_hidden = teacher_hidden.detach()
    shared = dict(
        mode=spec.mode,
        student_hidden=hidden,
        weight=weight,
        labels=labels,
        student_ce_weight=spec.student_ce_weight,
        teacher_ce_weight=spec.teacher_ce_weight,
        kl_weight=spec.kl_weight,
        kl_divergence=spec.divergence,
        temperature=temperature,
    )
    requires_grad = teacher_needs_grad(spec.teacher_ce_weight, spec.divergence)

    if spec.family == "tied":
        if student_dim != teacher_dim:
            raise ValueError("tied accuracy requires equal hidden dimensions")
        teacher_hidden = teacher_hidden.requires_grad_(requires_grad)
        return Case(
            teacher=TiedTeacher(teacher_hidden),
            ref_mode=RefMode.TIED,
            teacher_hidden=teacher_hidden,
            **shared,
        )

    teacher_weight = teacher_model.head.weight.detach().clone()

    if spec.family == "separate":
        teacher_hidden = teacher_hidden.requires_grad_(requires_grad)
        teacher_weight = teacher_weight.requires_grad_(requires_grad)
        return Case(
            teacher=SeparateTeacher(teacher_hidden, teacher_weight),
            ref_mode=RefMode.SEPARATE,
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
            **shared,
        )

    if spec.family == "precomputed-hidden":
        return Case(
            teacher=PrecomputedTeacher(
                teacher_hidden=teacher_hidden, teacher_weight=teacher_weight
            ),
            ref_mode=RefMode.PRECOMPUTED_HIDDEN,
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
            **shared,
        )

    raise ValueError(f"Unknown mode: {spec.mode}")


def release_case_memory(device: torch.device) -> None:
    release_memory(device, reset_compiler=True)


def dtype_text(dtype: torch.dtype) -> str:
    return fmt_precision(dtype, True, dtype == torch.float16)


DTYPE_LABELS = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


def normalize_dtype(dtype: torch.dtype) -> str:
    try:
        return DTYPE_LABELS[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported accuracy dtype: {dtype}") from exc
