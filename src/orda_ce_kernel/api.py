from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from ._runtime import HAS_TRITON
from .utils.dispatcher import dynamic_chunk
from .utils.resolver import DEFAULT_MAX_FUSED_SIZE, is_power_of_two


Reduction = Literal["mean", "sum"]
Profile = Literal["fast", "balanced", "debug"]
Backend = Literal["auto", "triton", "torch"]
_UNSET = object()


@dataclass(frozen=True)
class TiedTeacher:
    hidden: torch.Tensor


@dataclass(frozen=True)
class SeparateTeacher:
    hidden: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class PrecomputedTeacher:
    logits: torch.Tensor


@dataclass(frozen=True, init=False)
class KernelConfig:
    online_softmax: bool = True
    fast_math: bool = False
    quantize_grad_weight: bool = False
    stochastic_rounding: bool = False
    fp32_grad_weight_accumulation: bool = False
    fp32_accumulation: bool | None = None
    stochastic_seed: int | None = None
    chunk_size: int | Literal["auto", "dynamic"] | None = None
    max_chunks: int | None = None
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE

    def __init__(
        self,
        *args,
        online_softmax: bool = True,
        fast_math: bool = False,
        quantize_grad_weight: bool = False,
        stochastic_rounding: bool = False,
        fp32_grad_weight_accumulation: bool | object = _UNSET,
        fp32_accumulation: bool | None | object = _UNSET,
        stochastic_seed: int | None = None,
        chunk_size: int | Literal["auto", "dynamic"] | None = None,
        max_chunks: int | None = None,
        max_fused_size: int = DEFAULT_MAX_FUSED_SIZE,
    ) -> None:
        positional_names = (
            "online_softmax",
            "fast_math",
            "quantize_grad_weight",
            "stochastic_rounding",
            "fp32_accumulation",
            "chunk_size",
            "max_chunks",
            "max_fused_size",
        )
        if len(args) > len(positional_names):
            raise TypeError(f"KernelConfig expected at most {len(positional_names)} positional arguments")
        values = {
            "online_softmax": online_softmax,
            "fast_math": fast_math,
            "quantize_grad_weight": quantize_grad_weight,
            "stochastic_rounding": stochastic_rounding,
            "fp32_accumulation": fp32_accumulation,
            "chunk_size": chunk_size,
            "max_chunks": max_chunks,
            "max_fused_size": max_fused_size,
        }
        for name, value in zip(positional_names, args):
            values[name] = value

        online_softmax = values["online_softmax"]
        fast_math = values["fast_math"]
        quantize_grad_weight = values["quantize_grad_weight"]
        stochastic_rounding = values["stochastic_rounding"]
        fp32_accumulation = values["fp32_accumulation"]
        chunk_size = values["chunk_size"]
        max_chunks = values["max_chunks"]
        max_fused_size = values["max_fused_size"]

        canonical_set = fp32_grad_weight_accumulation is not _UNSET
        canonical = False if not canonical_set else bool(fp32_grad_weight_accumulation)
        alias_value = None if fp32_accumulation is _UNSET else fp32_accumulation
        if alias_value is not None and canonical_set and bool(alias_value) != canonical:
            raise ValueError(
                "fp32_accumulation and fp32_grad_weight_accumulation cannot disagree"
            )
        effective_fp32 = bool(alias_value) if alias_value is not None else canonical

        object.__setattr__(self, "online_softmax", bool(online_softmax))
        object.__setattr__(self, "fast_math", bool(fast_math))
        object.__setattr__(self, "quantize_grad_weight", bool(quantize_grad_weight))
        object.__setattr__(self, "stochastic_rounding", bool(stochastic_rounding))
        object.__setattr__(self, "fp32_grad_weight_accumulation", effective_fp32)
        object.__setattr__(self, "fp32_accumulation", effective_fp32)
        object.__setattr__(self, "stochastic_seed", stochastic_seed)
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "max_chunks", max_chunks)
        object.__setattr__(self, "max_fused_size", max_fused_size)

    @property
    def effective_fp32_grad_weight_accumulation(self) -> bool:
        return bool(self.fp32_grad_weight_accumulation)


class DistillationLossOutput(NamedTuple):
    loss: torch.Tensor
    student_ce: torch.Tensor
    teacher_ce: torch.Tensor
    kl: torch.Tensor


def _resolve_profile(profile: Profile) -> KernelConfig:
    if profile == "balanced":
        return KernelConfig(
            online_softmax=True,
            fast_math=False,
            quantize_grad_weight=False,
            stochastic_rounding=False,
            fp32_grad_weight_accumulation=False,
            chunk_size=None,
        )
    if profile == "fast":
        return KernelConfig(
            online_softmax=True,
            fast_math=True,
            quantize_grad_weight=False,
            stochastic_rounding=False,
            fp32_grad_weight_accumulation=False,
            chunk_size=None,
        )
    if profile == "debug":
        return KernelConfig(
            online_softmax=False,
            fast_math=False,
            quantize_grad_weight=False,
            stochastic_rounding=False,
            fp32_grad_weight_accumulation=True,
            chunk_size=None,
        )
    raise ValueError(f"profile must be 'fast', 'balanced', or 'debug', got {profile!r}")


def _resolve_config(profile: Profile, config: KernelConfig | None) -> KernelConfig:
    return config if config is not None else _resolve_profile(profile)


def _teacher_mode_and_tensors(
    teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
) -> tuple[str, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(teacher, TiedTeacher):
        return "tied", teacher.hidden, None, None
    if isinstance(teacher, SeparateTeacher):
        return "separate", teacher.hidden, teacher.weight, None
    if isinstance(teacher, PrecomputedTeacher):
        return "precomputed", None, None, teacher.logits
    raise TypeError(
        "teacher must be TiedTeacher, SeparateTeacher, or PrecomputedTeacher"
    )


def _resolve_teacher_ce_weight(mode: str, teacher_ce_weight: float | None) -> float:
    if teacher_ce_weight is not None:
        return float(teacher_ce_weight)
    return 1.0 if mode == "tied" else 0.0


def _validate_public_args(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    student_ce_weight: float,
    teacher_ce_weight: float | None,
    kd_weight: float,
    temperature: float,
    label_smoothing: float,
    reduction: str,
    backend: str,
    config: KernelConfig,
) -> None:
    if student_ce_weight < 0.0:
        raise ValueError(f"student_ce_weight must be >= 0.0, got {student_ce_weight}")
    if teacher_ce_weight is not None and teacher_ce_weight < 0.0:
        raise ValueError(f"teacher_ce_weight must be >= 0.0, got {teacher_ce_weight}")
    if kd_weight < 0.0:
        raise ValueError(f"kd_weight must be >= 0.0, got {kd_weight}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0.0, got {temperature}")
    if not (0.0 <= label_smoothing <= 1.0):
        raise ValueError(f"label_smoothing must be in [0.0, 1.0], got {label_smoothing}")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    if backend not in ("auto", "triton", "torch"):
        raise ValueError(f"backend must be 'auto', 'triton', or 'torch', got {backend!r}")
    if student_hidden.ndim != 2:
        raise ValueError(f"student_hidden must be 2D, got shape={tuple(student_hidden.shape)}")
    if weight.ndim != 2 or weight.shape[1] != student_hidden.shape[1]:
        raise ValueError(
            f"weight must have shape (vocab, {student_hidden.shape[1]}), "
            f"got {tuple(weight.shape)}"
        )
    if labels.shape != (student_hidden.shape[0],):
        raise ValueError(
            f"labels must have shape ({student_hidden.shape[0]},), got {tuple(labels.shape)}"
        )
    if labels.dtype == torch.bool or labels.is_floating_point():
        raise ValueError(
            f"labels must be an integer class-index tensor, got dtype={labels.dtype}"
        )
    if config.max_chunks is not None and config.max_chunks < 1:
        raise ValueError(f"config.max_chunks must be >= 1, got {config.max_chunks}")
    if config.max_fused_size < 1:
        raise ValueError(f"config.max_fused_size must be >= 1, got {config.max_fused_size}")
    if not is_power_of_two(int(config.max_fused_size)):
        raise ValueError(
            f"config.max_fused_size must be a power of two, got {config.max_fused_size}"
        )
    if config.stochastic_seed is not None and int(config.stochastic_seed) < 0:
        raise ValueError(f"config.stochastic_seed must be >= 0, got {config.stochastic_seed}")


def _validate_teacher(
    *,
    mode: str,
    teacher_hidden: torch.Tensor | None,
    teacher_weight: torch.Tensor | None,
    teacher_logits: torch.Tensor | None,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    bt, hidden = student_hidden.shape
    vocab = weight.shape[0]
    if mode == "tied":
        assert teacher_hidden is not None
        if teacher_hidden.shape != (bt, hidden):
            raise ValueError(
                f"TiedTeacher.hidden must have shape {(bt, hidden)}, "
                f"got {tuple(teacher_hidden.shape)}"
            )
    elif mode == "separate":
        assert teacher_hidden is not None and teacher_weight is not None
        if teacher_hidden.ndim != 2 or teacher_hidden.shape[0] != bt:
            raise ValueError(
                f"SeparateTeacher.hidden must have shape (BT, H_teacher), "
                f"got {tuple(teacher_hidden.shape)}"
            )
        if teacher_weight.shape != (vocab, teacher_hidden.shape[1]):
            raise ValueError(
                f"SeparateTeacher.weight must have shape {(vocab, teacher_hidden.shape[1])}, "
                f"got {tuple(teacher_weight.shape)}"
            )
    else:
        assert teacher_logits is not None
        if teacher_logits.shape != (bt, vocab):
            raise ValueError(
                f"PrecomputedTeacher.logits must have shape {(bt, vocab)}, "
                f"got {tuple(teacher_logits.shape)}"
            )
        if teacher_logits.requires_grad:
            raise ValueError(
                "PrecomputedTeacher.logits must not require gradients; use "
                "SeparateTeacher when the teacher path needs gradients"
            )


def _supports_triton(student_hidden: torch.Tensor) -> bool:
    return bool(HAS_TRITON and student_hidden.device.type == "cuda")


def _torch_reference(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
    student_ce_weight: float,
    teacher_ce_weight: float,
    kd_weight: float,
    temperature: float,
    ignore_index: int,
    label_smoothing: float,
    reduction: Reduction,
) -> DistillationLossOutput:
    mode, teacher_hidden, teacher_weight, teacher_logits = _teacher_mode_and_tensors(teacher)
    logits_s = student_hidden @ weight.t()
    if mode == "tied":
        assert teacher_hidden is not None
        logits_t = teacher_hidden @ weight.t()
    elif mode == "separate":
        assert teacher_hidden is not None and teacher_weight is not None
        logits_t = teacher_hidden @ teacher_weight.t()
    else:
        assert teacher_logits is not None
        logits_t = teacher_logits.detach()

    ce_s_all = F.cross_entropy(
        logits_s,
        labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    ce_t_all = F.cross_entropy(
        logits_t,
        labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    mask = labels != ignore_index
    denom = max(int(mask.sum().item()), 1)

    if kd_weight > 0.0:
        t = float(temperature)
        log_p_s = F.log_softmax(logits_s / t, dim=-1)
        p_t = F.softmax(logits_t.detach() / t, dim=-1)
        kl_all = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * (t * t)
        kl_all = kl_all.masked_fill(~mask, 0.0)
    else:
        kl_all = logits_s.new_zeros(logits_s.shape[0])

    if reduction == "mean":
        student_ce = ce_s_all.sum() / denom
        teacher_ce_raw = ce_t_all.sum() / denom
        kl = kl_all.sum() / denom
    else:
        student_ce = ce_s_all.sum()
        teacher_ce_raw = ce_t_all.sum()
        kl = kl_all.sum()

    if teacher_ce_weight == 0.0:
        teacher_ce = student_ce.new_zeros(())
    else:
        teacher_ce = teacher_ce_raw
    loss = student_ce_weight * student_ce + teacher_ce_weight * teacher_ce + kd_weight * kl
    return DistillationLossOutput(loss, student_ce.detach(), teacher_ce.detach(), kl.detach())


@torch._dynamo.disable
def distillation_loss(
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
    *,
    student_ce_weight: float = 1.0,
    teacher_ce_weight: float | None = None,
    kd_weight: float = 0.0,
    temperature: float = 1.0,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Reduction = "mean",
    profile: Profile = "balanced",
    backend: Backend = "auto",
    config: KernelConfig | None = None,
) -> DistillationLossOutput:
    resolved = _resolve_config(profile, config)
    _validate_public_args(
        student_hidden=student_hidden,
        weight=weight,
        labels=labels,
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kd_weight=kd_weight,
        temperature=temperature,
        label_smoothing=label_smoothing,
        reduction=reduction,
        backend=backend,
        config=resolved,
    )

    mode, teacher_hidden, teacher_weight, teacher_logits = _teacher_mode_and_tensors(teacher)
    _validate_teacher(
        mode=mode,
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        teacher_logits=teacher_logits,
        student_hidden=student_hidden,
        weight=weight,
    )
    effective_teacher_ce_weight = _resolve_teacher_ce_weight(mode, teacher_ce_weight)

    if backend == "torch" or (backend == "auto" and not _supports_triton(student_hidden)):
        return _torch_reference(
            student_hidden=student_hidden,
            weight=weight,
            labels=labels,
            teacher=teacher,
            student_ce_weight=student_ce_weight,
            teacher_ce_weight=effective_teacher_ce_weight,
            kd_weight=kd_weight,
            temperature=temperature,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
    if backend == "triton" and not _supports_triton(student_hidden):
        raise RuntimeError(
            "backend='triton' requires Triton and CUDA/HIP tensors; use backend='auto' "
            "or backend='torch' for fallback execution"
        )

    loss, student_ce, teacher_ce, kl = dynamic_chunk(
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        lambda_student=student_ce_weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
        chunk_size=resolved.chunk_size,
        use_int8_quant=resolved.quantize_grad_weight,
        use_stochastic_quant=resolved.stochastic_rounding,
        use_fast_math_exp=resolved.fast_math,
        use_fast_math_log=resolved.fast_math,
        use_fast_math_mul=resolved.fast_math,
        use_online_softmax=resolved.online_softmax,
        use_fp32_accum=resolved.effective_fp32_grad_weight_accumulation,
        use_kl_in_kernel=True,
        kl_weight=kd_weight,
        kl_temperature=temperature,
        stochastic_seed=resolved.stochastic_seed,
        max_chunks=resolved.max_chunks,
        max_fused_size=resolved.max_fused_size,
        teacher_mode=mode,
        weight_teacher=teacher_weight,
        logits_teacher=teacher_logits,
        teacher_loss_weight=effective_teacher_ce_weight,
    )
    return DistillationLossOutput(loss, student_ce, teacher_ce, kl)


class DistillationLoss(nn.Module):
    def __init__(
        self,
        *,
        student_ce_weight: float = 1.0,
        teacher_ce_weight: float | None = None,
        kd_weight: float = 0.0,
        temperature: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Reduction = "mean",
        profile: Profile = "balanced",
        backend: Backend = "auto",
        config: KernelConfig | None = None,
    ) -> None:
        super().__init__()
        self.student_ce_weight = student_ce_weight
        self.teacher_ce_weight = teacher_ce_weight
        self.kd_weight = kd_weight
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.profile = profile
        self.backend = backend
        self.config = config

    def forward(
        self,
        student_hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
    ) -> DistillationLossOutput:
        return distillation_loss(
            student_hidden,
            weight,
            labels,
            teacher,
            student_ce_weight=self.student_ce_weight,
            teacher_ce_weight=self.teacher_ce_weight,
            kd_weight=self.kd_weight,
            temperature=self.temperature,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
            profile=self.profile,
            backend=self.backend,
            config=self.config,
        )
