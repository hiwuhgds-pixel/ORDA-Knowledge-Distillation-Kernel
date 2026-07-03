from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from ._runtime import HAS_TRITON
from .utils.dispatcher import dynamic_chunk
from .utils.resolver import DEFAULT_MAX_FUSED_SIZE, _is_auto_chunk_size, is_power_of_two


Reduction = Literal["mean", "sum"]
Profile = Literal["balanced", "debug"]
Backend = Literal["auto", "triton", "torch"]


# ── Teacher descriptors ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class TiedTeacher:
    hidden: torch.Tensor


@dataclass(frozen=True)
class SeparateTeacher:
    hidden: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class PrecomputedTeacher:
    logits: torch.Tensor | None = None
    teacher_hidden: torch.Tensor | None = None
    teacher_weight: torch.Tensor | None = None

    def __post_init__(self):
        has_logits = self.logits is not None
        has_hidden_weight = self.teacher_hidden is not None or self.teacher_weight is not None
        if has_logits and has_hidden_weight:
            raise ValueError(
                "PrecomputedTeacher: provide logits or (teacher_hidden, teacher_weight), not both"
            )
        if not has_logits and not has_hidden_weight:
            raise ValueError(
                "PrecomputedTeacher: must provide logits or (teacher_hidden, teacher_weight)"
            )
        if has_hidden_weight and (self.teacher_hidden is None or self.teacher_weight is None):
            raise ValueError(
                "PrecomputedTeacher: teacher_hidden and teacher_weight must both be provided"
            )


# ── Public config and output types ───────────────────────────────────────────
@dataclass(frozen=True)
class KernelConfig:
    fp32_grad_weight_accumulation: bool = False
    chunk_size: int | Literal["auto", "dynamic"] | None = None
    num_chunks: int | None = None
    max_chunks: int | None = None
    max_fused_size: int = DEFAULT_MAX_FUSED_SIZE
    kl_weight: float | None = None
    kl_temperature: float | None = None
    autotune: bool = False

    @property
    def effective_fp32_grad_weight_accumulation(self) -> bool:
        return bool(self.fp32_grad_weight_accumulation)


class DistillationLossOutput(NamedTuple):
    loss: torch.Tensor
    student_ce: torch.Tensor
    teacher_ce: torch.Tensor
    kl: torch.Tensor


# ── Config resolution ────────────────────────────────────────────────────────
def _resolve_profile(profile: Profile) -> KernelConfig:
    if profile == "balanced":
        return KernelConfig()
    if profile == "debug":
        return KernelConfig(fp32_grad_weight_accumulation=True)
    raise ValueError(f"profile must be 'balanced' or 'debug', got {profile!r}")


def _resolve_config(profile: Profile, config: KernelConfig | None) -> KernelConfig:
    return config if config is not None else _resolve_profile(profile)


def _resolve_kl_config(
    config: KernelConfig,
    kl_weight: float,
    kl_temperature: float,
) -> tuple[float, float]:
    effective_kl_weight = kl_weight if config.kl_weight is None else float(config.kl_weight)
    effective_kl_temperature = kl_temperature if config.kl_temperature is None else float(config.kl_temperature)
    return effective_kl_weight, effective_kl_temperature


def _teacher_mode_and_tensors(
    teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
) -> tuple[str, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(teacher, TiedTeacher):
        return "tied", teacher.hidden, None, None
    if isinstance(teacher, SeparateTeacher):
        return "separate", teacher.hidden, teacher.weight, None
    if isinstance(teacher, PrecomputedTeacher):
        return "precomputed", teacher.teacher_hidden, teacher.teacher_weight, teacher.logits
    raise TypeError("teacher must be TiedTeacher, SeparateTeacher, or PrecomputedTeacher")


def _resolve_teacher_ce_weight(mode: str, teacher_ce_weight: float | None) -> float:
    if teacher_ce_weight is not None:
        return float(teacher_ce_weight)
    return 1.0 if mode == "tied" else 0.0


# ── Public argument validation ───────────────────────────────────────────────
def _validate_public_args(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    student_ce_weight: float,
    teacher_ce_weight: float | None,
    kl_weight: float,
    kl_temperature: float,
    reduction: str,
    backend: str,
    config: KernelConfig,
) -> None:
    if student_ce_weight < 0.0:
        raise ValueError(f"student_ce_weight must be >= 0.0, got {student_ce_weight}")
    if teacher_ce_weight is not None and teacher_ce_weight < 0.0:
        raise ValueError(f"teacher_ce_weight must be >= 0.0, got {teacher_ce_weight}")
    if kl_weight < 0.0:
        raise ValueError(f"kl_weight must be >= 0.0, got {kl_weight}")
    if kl_temperature <= 0.0:
        raise ValueError(f"kl_temperature must be > 0.0, got {kl_temperature}")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    if backend not in ("auto", "triton", "torch"):
        raise ValueError(f"backend must be 'auto', 'triton', or 'torch', got {backend!r}")
    if student_hidden.ndim != 2:
        raise ValueError(f"student_hidden must be 2D, got shape={tuple(student_hidden.shape)}")
    if weight.ndim != 2 or weight.shape[1] != student_hidden.shape[1]:
        raise ValueError(
            f"weight must have shape (vocab, {student_hidden.shape[1]}), got {tuple(weight.shape)}"
        )
    if labels.shape != (student_hidden.shape[0],):
        raise ValueError(
            f"labels must have shape ({student_hidden.shape[0]},), got {tuple(labels.shape)}"
        )
    if labels.dtype != torch.long:
        raise ValueError(f"labels must have dtype torch.long, got {labels.dtype}")
    if config.max_chunks is not None and config.max_chunks < 1:
        raise ValueError(f"config.max_chunks must be >= 1, got {config.max_chunks}")
    if config.num_chunks is not None and config.num_chunks < 1:
        raise ValueError(f"config.num_chunks must be >= 1, got {config.num_chunks}")
    if config.num_chunks is not None and not _is_auto_chunk_size(config.chunk_size):
        raise ValueError("config.chunk_size and config.num_chunks are mutually exclusive")
    effective_num_chunks = (
        None
        if config.num_chunks is None
        else min(int(config.num_chunks), int(student_hidden.shape[0]))
    )
    if (
        effective_num_chunks is not None
        and config.max_chunks is not None
        and effective_num_chunks > config.max_chunks
    ):
        raise ValueError(
            f"config.num_chunks must be <= config.max_chunks, got "
            f"num_chunks={config.num_chunks}, max_chunks={config.max_chunks}"
        )
    if config.max_fused_size < 1:
        raise ValueError(f"config.max_fused_size must be >= 1, got {config.max_fused_size}")
    if not is_power_of_two(int(config.max_fused_size)):
        raise ValueError(f"config.max_fused_size must be a power of two, got {config.max_fused_size}")


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
                f"TiedTeacher.hidden must have shape {(bt, hidden)}, got {tuple(teacher_hidden.shape)}"
            )
    elif mode == "separate":
        assert teacher_hidden is not None and teacher_weight is not None
        if teacher_hidden.ndim != 2 or teacher_hidden.shape[0] != bt:
            raise ValueError(
                f"SeparateTeacher.hidden must have shape (BT, teacher_hidden_dim), got {tuple(teacher_hidden.shape)}"
            )
        if teacher_weight.shape != (vocab, teacher_hidden.shape[1]):
            raise ValueError(
                f"SeparateTeacher.weight must have shape {(vocab, teacher_hidden.shape[1])}, "
                f"got {tuple(teacher_weight.shape)}"
            )
    else:
        has_hidden_weight = teacher_hidden is not None and teacher_weight is not None
        if teacher_logits is not None:
            if teacher_logits.shape != (bt, vocab):
                raise ValueError(
                    f"PrecomputedTeacher.logits must have shape {(bt, vocab)}, got {tuple(teacher_logits.shape)}"
                )
            if teacher_logits.requires_grad:
                raise ValueError(
                    "PrecomputedTeacher.logits must not require gradients; use SeparateTeacher "
                    "when the teacher path needs gradients"
                )
        elif has_hidden_weight:
            if teacher_hidden.ndim != 2 or teacher_hidden.shape[0] != bt:
                raise ValueError(
                    f"PrecomputedTeacher.teacher_hidden must have shape (BT, D_t), got {tuple(teacher_hidden.shape)}"
                )
            if teacher_weight.shape != (vocab, teacher_hidden.shape[1]):
                raise ValueError(
                    f"PrecomputedTeacher.teacher_weight must have shape {(vocab, teacher_hidden.shape[1])}, "
                    f"got {tuple(teacher_weight.shape)}"
                )
            if teacher_hidden.requires_grad or teacher_weight.requires_grad:
                raise ValueError(
                    "PrecomputedTeacher.teacher_hidden and teacher_weight must not require gradients"
                )
        else:
            raise ValueError(
                "PrecomputedTeacher requires logits or (teacher_hidden, teacher_weight)"
            )


# ── Backend selection ────────────────────────────────────────────────────────
def _supports_triton(student_hidden: torch.Tensor) -> bool:
    return bool(HAS_TRITON and student_hidden.device.type == "cuda")


# ── Torch reference fallback ─────────────────────────────────────────────────
def _torch_reference(
    *,
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
    student_ce_weight: float,
    teacher_ce_weight: float,
    kl_weight: float,
    kl_temperature: float,
    ignore_index: int,
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
        if teacher_logits is not None:
            logits_t = teacher_logits.detach()
        else:
            assert teacher_hidden is not None and teacher_weight is not None
            logits_t = (teacher_hidden @ teacher_weight.t()).detach()

    ce_s_all = F.cross_entropy(
        logits_s,
        labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    ce_t_all = F.cross_entropy(
        logits_t,
        labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    mask = labels != ignore_index
    denom = max(int(mask.sum().item()), 1)

    t = float(kl_temperature)
    log_p_s = F.log_softmax(logits_s / t, dim=-1)
    p_t = F.softmax(logits_t.detach() / t, dim=-1)
    kl_all = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * (t * t)
    kl_all = kl_all.masked_fill(~mask, 0.0)

    if reduction == "mean":
        student_ce = ce_s_all.sum() / denom
        teacher_ce_raw = ce_t_all.sum() / denom
        kl = kl_all.sum() / denom
    else:
        student_ce = ce_s_all.sum()
        teacher_ce_raw = ce_t_all.sum()
        kl = kl_all.sum()

    teacher_ce = teacher_ce_raw if teacher_ce_weight != 0.0 else student_ce.new_zeros(())
    loss = student_ce_weight * student_ce + teacher_ce_weight * teacher_ce + kl_weight * kl
    return DistillationLossOutput(loss, student_ce.detach(), teacher_ce.detach(), kl.detach())


# ── Public functional API ────────────────────────────────────────────────────
@torch._dynamo.disable
def distillation_loss(
    student_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher: TiedTeacher | SeparateTeacher | PrecomputedTeacher,
    *,
    student_ce_weight: float = 1.0,
    teacher_ce_weight: float | None = None,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
    profile: Profile = "balanced",
    backend: Backend = "auto",
    config: KernelConfig | None = None,
) -> DistillationLossOutput:
    resolved = _resolve_config(profile, config)
    effective_kl_weight, effective_kl_temperature = _resolve_kl_config(
        resolved,
        kl_weight,
        kl_temperature,
    )
    _validate_public_args(
        student_hidden=student_hidden,
        weight=weight,
        labels=labels,
        student_ce_weight=student_ce_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=effective_kl_weight,
        kl_temperature=effective_kl_temperature,
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
    supports_triton = _supports_triton(student_hidden)

    if backend == "triton" and mode == "precomputed" and effective_teacher_ce_weight != 0.0:
        raise ValueError(
            "backend='triton' with PrecomputedTeacher requires teacher_ce_weight=0.0"
        )

    auto_fallback = (
        not supports_triton
        or (mode == "precomputed" and effective_teacher_ce_weight != 0.0)
    )
    if backend == "torch" or (backend == "auto" and auto_fallback):
        return _torch_reference(
            student_hidden=student_hidden,
            weight=weight,
            labels=labels,
            teacher=teacher,
            student_ce_weight=student_ce_weight,
            teacher_ce_weight=effective_teacher_ce_weight,
            kl_weight=effective_kl_weight,
            kl_temperature=effective_kl_temperature,
            ignore_index=ignore_index,
            reduction=reduction,
        )
    if backend == "triton" and not supports_triton:
        raise RuntimeError(
            "backend='triton' requires Triton and CUDA/HIP tensors; use backend='auto' "
            "or backend='torch' for fallback execution"
        )

    loss, student_ce, teacher_ce, kl = dynamic_chunk(
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        student_ce_weight=student_ce_weight,
        ignore_index=ignore_index,
        reduction=reduction,
        chunk_size=resolved.chunk_size,
        num_chunks=resolved.num_chunks,
        use_fp32_accum=resolved.effective_fp32_grad_weight_accumulation,
        kl_weight=effective_kl_weight,
        kl_temperature=effective_kl_temperature,
        max_chunks=resolved.max_chunks,
        max_fused_size=resolved.max_fused_size,
        autotune=resolved.autotune,
        teacher_mode=mode,
        teacher_weight=teacher_weight,
        logits_teacher=teacher_logits,
        teacher_ce_weight=effective_teacher_ce_weight,
        validate_labels=profile == "debug",
    )
    return DistillationLossOutput(loss, student_ce, teacher_ce, kl)


# ── nn.Module wrapper ────────────────────────────────────────────────────────
class DistillationLoss(nn.Module):
    def __init__(
        self,
        *,
        student_ce_weight: float = 1.0,
        teacher_ce_weight: float | None = None,
        kl_weight: float = 0.0,
        kl_temperature: float = 1.0,
        ignore_index: int = -100,
        reduction: Reduction = "mean",
        profile: Profile = "balanced",
        backend: Backend = "auto",
        config: KernelConfig | None = None,
    ) -> None:
        super().__init__()
        self.student_ce_weight = student_ce_weight
        self.teacher_ce_weight = teacher_ce_weight
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature
        self.ignore_index = ignore_index
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
            kl_weight=self.kl_weight,
            kl_temperature=self.kl_temperature,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            profile=self.profile,
            backend=self.backend,
            config=self.config,
        )
