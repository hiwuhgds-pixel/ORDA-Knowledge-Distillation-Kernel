from __future__ import annotations

from dataclasses import dataclass

import torch

from .._api_types import AutotuneMode, EntropyGate, GradWeightAccumulation

# ── Planning vocabulary ──────────────────────────────────────────────────────
TIED = "tied"
SEPARATE = "separate"
PRECOMPUTED = "precomputed"

PRECOMPUTED_TOKENWISE = "precomputed_tokenwise"
PRECOMPUTED_ALIGNED_SPAN = "precomputed_aligned_span"
SEPARATE_TRAINED = "separate_trained"
SEPARATE_FROZEN = "separate_frozen"
SEPARATE_COTRAINED = "separate_cotrained"
TIED_FROZEN = "tied_frozen"
TIED_TRAINED = "tied_trained"
TIED_COTRAINED = "tied_cotrained"

TEACHER_GRAD_NONE = "none"
TEACHER_GRAD_CE = "ce"
TEACHER_GRAD_KL = "kl"
TEACHER_GRAD_CE_KL = "ce_kl"

VALID_KL_VARIANTS = frozenset(
    {
        "forward",
        "jeffreys",
        "mutual",
        "cojeffreys",
        "reverse",
        "alpha_divergence",
        "skew_forward",
        "skew_js",
        "entropy_adaptive",
        "skew_reverse",
        "renyi",
        "renyi_reverse",
        "dkd",
        "abkd",
        "triangular",
    }
)
TEACHER_KL_GRAD_VARIANTS = frozenset({"mutual", "cojeffreys"})
# Teacher-detached variants run wherever the teacher never receives KL gradient.
STUDENT_GRAD_KL_VARIANTS = VALID_KL_VARIANTS - TEACHER_KL_GRAD_VARIANTS
GATED_KL_VARIANTS = frozenset({"entropy_adaptive"})
CANONICAL_INACTIVE_KL_VARIANT = "forward"
DEFAULT_KL_ALPHA = 0.5
DEFAULT_KL_BETA = 0.5


# ── KL hyperparameters ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class KLHyperparamSpec:
    lo: float
    hi: float
    default: float
    open_lo: bool = True
    open_hi: bool = True
    exclude: tuple[float, float] | None = None


KL_HYPERPARAM_SPECS: dict[str, dict[str, KLHyperparamSpec]] = {
    "alpha_divergence": {
        "kl_alpha": KLHyperparamSpec(lo=0.0, hi=1.0, default=0.5),
    },
    "skew_forward": {
        "kl_beta": KLHyperparamSpec(lo=0.0, hi=1.0, default=0.5),
    },
    "skew_js": {
        "kl_beta": KLHyperparamSpec(lo=0.0, hi=1.0, default=0.5),
    },
    "skew_reverse": {
        "kl_beta": KLHyperparamSpec(lo=0.0, hi=1.0, default=0.5),
    },
    "renyi": {
        "kl_alpha": KLHyperparamSpec(
            lo=0.0,
            hi=2.0,
            default=0.5,
            open_hi=False,
            exclude=(0.99, 1.01),
        ),
    },
    "renyi_reverse": {
        "kl_alpha": KLHyperparamSpec(
            lo=0.0,
            hi=2.0,
            default=0.5,
            open_hi=False,
            exclude=(0.99, 1.01),
        ),
    },
    "dkd": {
        "kl_alpha": KLHyperparamSpec(
            lo=0.0,
            hi=16.0,
            default=1.0,
            open_hi=False,
        ),
        "kl_beta": KLHyperparamSpec(
            lo=0.0,
            hi=16.0,
            default=1.0,
            open_hi=False,
        ),
    },
    "abkd": {
        "kl_alpha": KLHyperparamSpec(
            lo=0.05,
            hi=2.0,
            default=0.5,
            open_lo=False,
            open_hi=False,
        ),
        "kl_beta": KLHyperparamSpec(
            lo=0.05,
            hi=2.0,
            default=0.5,
            open_lo=False,
            open_hi=False,
        ),
    },
}
ALPHA_KL_VARIANTS = frozenset(
    variant for variant, specs in KL_HYPERPARAM_SPECS.items() if "kl_alpha" in specs
)
BETA_KL_VARIANTS = frozenset(
    variant for variant, specs in KL_HYPERPARAM_SPECS.items() if "kl_beta" in specs
)


# ── Workload records ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class KernelSpecialization:
    compute_kl: bool
    t_is_one: bool
    compute_teacher_ce: bool
    kl_variant: str
    teacher_grad_policy: str


@dataclass(frozen=True)
class ExecutionPolicy:
    teacher_mode: str
    kernel_path: str
    specialization: KernelSpecialization


# ── Immutable execution handoff ──────────────────────────────────────────────
@dataclass(frozen=True)
class ChunkPolicy:
    initial_chunk_size: int
    initial_num_chunks: int
    max_chunks: int | None
    max_chunks_limit: int
    dynamic: bool
    fast: bool = False


@dataclass(frozen=True)
class ChunkAttempt:
    chunk_size: int
    num_chunks: int


@dataclass(frozen=True)
class ExecutionPlan:
    backend: str
    teacher_mode: str
    kernel_path: str
    specialization: KernelSpecialization
    compute_dtype: torch.dtype
    chunk_policy: ChunkPolicy
    max_fused_size: int
    autotune: AutotuneMode
    refresh_autotune: bool
    grad_weight_accumulation: GradWeightAccumulation
    teacher_grad_policy: str
    student_ce_weight: float
    teacher_ce_weight: float
    kl_weight: float
    kl_temperature: float
    kl_teacher_temperature: float
    ignore_index: int
    reduction: str
    assume_valid_labels: bool
    # KL hyperparameters are runtime scalars: they change the loss value (so
    # they live on the serialized plan) but never the kernel specialization or
    # the autotune performance key.  Unused values hold canonical defaults.
    kl_alpha: float = DEFAULT_KL_ALPHA
    kl_beta: float = DEFAULT_KL_BETA
    entropy_gate: EntropyGate | None = None


# ── Aligned-span execution handoff ───────────────────────────────────────────
@dataclass(frozen=True)
class AlignedSpanSpecialization:
    compute_student_ce: bool
    compute_span: bool


@dataclass(frozen=True)
class AlignedSpanExecutionPlan:
    backend: str
    teacher_mode: str
    kernel_path: str
    specialization: AlignedSpanSpecialization
    compute_dtype: torch.dtype
    chunk_policy: ChunkPolicy
    max_fused_size: int
    autotune: AutotuneMode
    refresh_autotune: bool
    grad_weight_accumulation: GradWeightAccumulation
    student_ce_weight: float
    span_kl_weight: float
    ignore_index: int
    reduction: str
    objective: str
    assume_valid_labels: bool
