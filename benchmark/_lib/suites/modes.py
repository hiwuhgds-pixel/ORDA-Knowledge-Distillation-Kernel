from __future__ import annotations

from dataclasses import dataclass

from .specs import ACCURACY_ORDER, COTRAINED_PAIRS, SPECIAL_CASES, TeacherState


@dataclass(frozen=True)
class AccuracyMode:
    mode: str
    family: str
    teacher_ce_weight: float
    divergence: str
    student_ce_weight: float = 1.0
    kl_weight: float = 1.0
    head_requires_grad: bool = True
    label: str | None = None
    temperature: float | None = None

    @property
    def display_label(self) -> str:
        return self.label or self.mode


# ── Mode generation ──────────────────────────────────────────────────────────
def _teacher_modes(family: str, states: tuple[tuple[str, float], ...]):
    for state, teacher_ce_weight in states:
        for divergence in ACCURACY_ORDER:
            yield AccuracyMode(
                f"{family}-{state}-{divergence}",
                family,
                teacher_ce_weight,
                divergence,
            )


def _cotrained_modes(family: str):
    for state, divergence in COTRAINED_PAIRS:
        state_id = str(state).replace("_", "-")
        teacher_ce_weight = 1.0 if state == TeacherState.COTRAINED_TEACHER_CE else 0.0
        yield AccuracyMode(
            f"{family}-{state_id}-{divergence}",
            family,
            teacher_ce_weight,
            divergence,
        )


def _special_modes():
    for case in SPECIAL_CASES:
        yield AccuracyMode(
            case.id,
            "tied",
            case.teacher_ce_weight,
            case.kl_divergence,
            student_ce_weight=case.student_ce_weight,
            kl_weight=case.kl_weight,
            head_requires_grad=case.head_requires_grad,
            label=case.label,
            temperature=case.accuracy_temperature,
        )


def accuracy_modes():
    yield from _teacher_modes("tied", (("trained", 1.0), ("frozen", 0.0)))
    yield from _cotrained_modes("tied")
    yield from _special_modes()
    yield from _teacher_modes("separate", (("frozen", 0.0), ("trained", 1.0)))
    yield from _cotrained_modes("separate")
    for divergence in ACCURACY_ORDER:
        yield AccuracyMode(
            f"precomputed-hidden-{divergence}",
            "precomputed-hidden",
            0.0,
            divergence,
        )


ACCURACY_MODES = {mode.mode: mode for mode in accuracy_modes()}

MODE_INDEX = {mode: index for index, mode in enumerate(ACCURACY_MODES)}

_FAMILY_TAGS = {
    "tied": "tied",
    "separate": "separate",
    "precomputed": "precomputed-hidden",
}


# ── Mode lookup ──────────────────────────────────────────────────────────────
def accuracy_mode(mode: str) -> AccuracyMode:
    try:
        return ACCURACY_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown mode: {mode}") from exc


def accuracy_family_modes(family: str) -> list[str]:
    try:
        tag = _FAMILY_TAGS[family]
    except KeyError as exc:
        raise ValueError(f"Unknown accuracy family: {family}") from exc
    return [mode for mode, spec in ACCURACY_MODES.items() if spec.family == tag]


def active_components(mode: str) -> tuple[str, ...]:
    spec = accuracy_mode(mode)
    components = []
    if spec.student_ce_weight != 0.0 or spec.teacher_ce_weight != 0.0:
        components.append("ce")
    if spec.kl_weight != 0.0:
        components.append("kl")
    if not components:
        raise ValueError(f"accuracy mode {mode!r} has no active loss component")
    return tuple(components)
