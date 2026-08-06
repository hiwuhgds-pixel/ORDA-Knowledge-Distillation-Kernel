from __future__ import annotations

import zlib
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from ..losses.divergence import DIVERGENCES

# ── Divergence orderings ─────────────────────────────────────────────────────
# These orders define benchmark identity: one controls case listing and the
# other feeds deterministic accuracy seeds. Append entries; do not reorder.
CASE_ID_ORDER = (
    "abkd",
    "alpha_divergence",
    "dkd",
    "entropy_adaptive",
    "forward",
    "jeffreys",
    "renyi_reverse",
    "renyi",
    "reverse",
    "skew_forward",
    "skew_js",
    "skew_reverse",
    "triangular",
)

ACCURACY_ORDER = (
    "forward",
    "jeffreys",
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
)


def _check_orders() -> None:
    if set(CASE_ID_ORDER) != set(DIVERGENCES):
        raise RuntimeError("CASE_ID_ORDER must cover exactly DIVERGENCES")
    if set(ACCURACY_ORDER) != set(DIVERGENCES):
        raise RuntimeError("ACCURACY_ORDER must cover exactly DIVERGENCES")


_check_orders()


# ── Teacher states ───────────────────────────────────────────────────────────
class TeacherState(str, Enum):
    __str__ = str.__str__

    TRAINED = "trained"
    FROZEN = "frozen"
    COTRAINED = "cotrained"
    COTRAINED_TEACHER_CE = "cotrained_teacher_ce"


TEACHER_STATE_LABELS = {
    TeacherState.TRAINED: "trained",
    TeacherState.FROZEN: "frozen",
    TeacherState.COTRAINED: "cotrained",
    TeacherState.COTRAINED_TEACHER_CE: "cotrained teacher CE",
}

# Co-training is only meaningful for symmetric divergences.
COTRAINED_PAIRS = (
    (TeacherState.COTRAINED, "mutual"),
    (TeacherState.COTRAINED, "cojeffreys"),
    (TeacherState.COTRAINED_TEACHER_CE, "mutual"),
    (TeacherState.COTRAINED_TEACHER_CE, "cojeffreys"),
)


class TeacherPolicy(NamedTuple):
    teacher_ce_weight: float
    teacher_requires_grad: bool


def teacher_policy(state: str, divergence: str) -> TeacherPolicy:
    state = TeacherState(state)
    teacher_ce_weight = (
        1.0
        if state in {TeacherState.TRAINED, TeacherState.COTRAINED_TEACHER_CE}
        else 0.0
    )
    teacher_requires_grad = teacher_ce_weight != 0.0 or divergence in {
        "mutual",
        "cojeffreys",
    }
    return TeacherPolicy(teacher_ce_weight, teacher_requires_grad)


def teacher_state_label(state: str, benchmark: str) -> str:
    try:
        return TEACHER_STATE_LABELS[TeacherState(state)]
    except ValueError as exc:
        raise ValueError(f"Unknown {benchmark} state: {state}") from exc


def teacher_name(family: str, benchmark: str) -> str:
    if family == "tied":
        return "TiedTeacher"
    if family == "separate":
        return "SeparateTeacher"
    raise ValueError(f"Unknown {benchmark} family: {family}")


def teacher_case_label(
    benchmark: str,
    family: str,
    state: str,
    divergence: str,
) -> str:
    teacher = teacher_name(family, benchmark)
    state_label = teacher_state_label(state, benchmark)
    return f"{teacher} {state_label} · {divergence}"


class TeacherCaseSpec(NamedTuple):
    case_id: str
    state: str
    divergence: str


def teacher_case_specs():
    for state in (TeacherState.TRAINED, TeacherState.FROZEN):
        for divergence in CASE_ID_ORDER:
            yield TeacherCaseSpec(f"{state}_{divergence}", state, divergence)
    for state, divergence in COTRAINED_PAIRS:
        yield TeacherCaseSpec(f"{state}_{divergence}", state, divergence)


# ── Special workloads ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SpecialCase:
    """Keep accuracy temperature separate because it intentionally differs from train/VRAM."""

    id: str
    label: str
    tied_mode: str
    kl_divergence: str
    student_ce_weight: float
    teacher_ce_weight: float
    kl_weight: float
    temperature: float = 1.0
    accuracy_temperature: float = 1.7

    @property
    def teacher_requires_grad(self) -> bool:
        return self.tied_mode != "frozen"

    @property
    def head_requires_grad(self) -> bool:
        return self.tied_mode != "frozen"


SPECIAL_CASES = (
    SpecialCase(
        id="kangaroo",
        label="Kangaroo · frozen · forward",
        tied_mode="frozen",
        kl_divergence="forward",
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
    ),
    SpecialCase(
        id="recursive_transformer",
        label="Recursive Transformers · trained · forward",
        tied_mode="trained",
        kl_divergence="forward",
        student_ce_weight=0.1,
        teacher_ce_weight=1.0,
        kl_weight=1.0,
    ),
    SpecialCase(
        id="r_drop",
        label="R-Drop · cotrained · cojeffreys",
        tied_mode="cotrained",
        kl_divergence="cojeffreys",
        student_ce_weight=1.0,
        teacher_ce_weight=1.0,
        # cojeffreys is KL(p||q) + KL(q||p); R-Drop uses one half.
        kl_weight=0.5,
    ),
)


# ── Aligned-span and precomputed-cache cases ─────────────────────────────────
class AlignedCaseSpec(NamedTuple):
    case_id: str
    objective: str
    label: str


ALIGNED_CASE_SPECS = (
    AlignedCaseSpec(
        "aligned_span_binary_ce",
        "binary_ce",
        "precomputed cross-vocab · binary CE",
    ),
    AlignedCaseSpec(
        "aligned_span_binary_kl",
        "binary_kl",
        "precomputed cross-vocab · binary KL",
    ),
)


class PrecomputedCacheCaseSpec(NamedTuple):
    case_id: str
    label: str
    cache_kind: str


PRECOMPUTED_CACHE_CASE_SPECS = (
    PrecomputedCacheCaseSpec(
        "cross_vocab_end_to_end_precomputed_loss",
        "cross-vocab projection → PrecomputedTeacher",
        "cross_vocab",
    ),
    PrecomputedCacheCaseSpec(
        "multi_teacher_arithmetic",
        "multi-teacher arithmetic → PrecomputedTeacher",
        "multi_teacher_arithmetic",
    ),
    PrecomputedCacheCaseSpec(
        "multi_teacher_geometric",
        "multi-teacher geometric → PrecomputedTeacher",
        "multi_teacher_geometric",
    ),
)

PRECOMPUTED_INPUT_ACCURACY_CASES = (
    ("cross_vocab_projection", "cross-vocab projection"),
    (
        "cross_vocab_end_to_end_precomputed_loss",
        "cross-vocab projection → PrecomputedTeacher loss",
    ),
    ("aligned_span_live_cross_vocab", "aligned span · live cross-vocab"),
    (
        "precomputed_span_teacher_cache_build",
        "PrecomputedSpanTeacher cache build",
    ),
    ("multi_teacher_arithmetic", "multi-teacher arithmetic"),
    ("multi_teacher_geometric", "multi-teacher geometric"),
    ("multi_teacher_gjs", "multi-teacher GJS"),
)


# ── Mode plumbing ────────────────────────────────────────────────────────────
def precomputed_mode_parts(mode: str) -> tuple[str, str]:
    backend, _, source = mode.rpartition("-")
    if backend not in ("torch", "torch-compile", "orda") or source not in (
        "logits",
        "hidden",
    ):
        raise ValueError(f"Unknown precomputed mode: {mode}")
    return backend, source


def alternating_passes(modes, passes: int):
    """Alternate backend order to prevent warm-cache effects favoring one backend."""
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")
    modes = tuple(modes)
    for index in range(passes):
        descending = index % 2 == 0
        yield (
            index + 1,
            "↓ top-down" if descending else "↑ bottom-up",
            modes if descending else tuple(reversed(modes)),
        )


def stable_seed(*parts) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return zlib.crc32(payload) & 0x7FFF_FFFF
