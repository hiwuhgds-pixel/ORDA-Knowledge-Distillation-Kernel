"""Full Cartesian correctness matrix and deterministic shard metadata."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import prod

# ── Matrix dimensions ────────────────────────────────────────────────────────
TEACHER_REPRESENTATIONS = (
    "tied",
    "separate",
    "precomputed_logits",
    "precomputed_hidden_weight",
)
KL_VARIANTS = (
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
)
KL_WEIGHT_STATES = ("zero", "positive")
STUDENT_CE_STATES = ("zero", "positive")
TEACHER_CE_STATES = ("default", "zero", "positive")
TEMPERATURE_STATES = ("one", "non_one")
REDUCTIONS = ("mean", "sum")
DTYPES = ("float16", "bfloat16", "float32")
INPUT_LAYOUTS = ("contiguous", "noncontiguous_hidden", "noncontiguous_weight")
CHUNK_POLICIES = (
    "fixed_chunk_size",
    "fixed_num_chunks",
    "dynamic",
    "autotune_off",
)
LABEL_PATTERNS = (
    "all_valid",
    "with_ignore_index",
    "all_ignored",
    "invalid_negative",
    "invalid_high",
)
API_LEVELS = (
    "high_level",
    "direct_tied",
    "direct_separate",
    "direct_precomputed",
)

MATRIX_DIMENSIONS = (
    TEACHER_REPRESENTATIONS,
    KL_VARIANTS,
    KL_WEIGHT_STATES,
    STUDENT_CE_STATES,
    TEACHER_CE_STATES,
    TEMPERATURE_STATES,
    REDUCTIONS,
    DTYPES,
    INPUT_LAYOUTS,
    CHUNK_POLICIES,
    LABEL_PATTERNS,
    API_LEVELS,
)
MATRIX_FIELD_NAMES = (
    "teacher_representation",
    "kl_variant",
    "kl_weight_state",
    "student_ce_state",
    "teacher_ce_state",
    "temperature_state",
    "reduction",
    "dtype",
    "input_layout",
    "chunk_policy",
    "label_pattern",
    "api_level",
)
TOTAL_CARTESIAN_CASES = prod(len(dimension) for dimension in MATRIX_DIMENSIONS)
MATRIX_STRIDES = tuple(
    prod(len(dimension) for dimension in MATRIX_DIMENSIONS[index + 1 :])
    for index in range(len(MATRIX_DIMENSIONS))
)
CARTESIAN_ROUTE_CASE_COUNT = MATRIX_STRIDES[0]
TOTAL_CARTESIAN_ROUTES = len(TEACHER_REPRESENTATIONS)

assert len(MATRIX_FIELD_NAMES) == len(MATRIX_DIMENSIONS)
assert CARTESIAN_ROUTE_CASE_COUNT * TOTAL_CARTESIAN_ROUTES == TOTAL_CARTESIAN_CASES


# ── Expected outcomes ────────────────────────────────────────────────────────
class ExpectedOutcome(str, Enum):
    SUCCESS = "success"
    REJECT_API_TEACHER_MISMATCH = "reject_api_teacher_mismatch"
    REJECT_UNSUPPORTED_VARIANT = "reject_unsupported_variant"
    REJECT_INVALID_TEACHER_STATE = "reject_invalid_teacher_state"
    REJECT_INVALID_LABEL = "reject_invalid_label"


@dataclass(frozen=True, slots=True)
class RejectionContract:
    exception_type: type[BaseException]
    message_pattern: str


REJECTION_CONTRACTS = {
    ExpectedOutcome.REJECT_API_TEACHER_MISMATCH: RejectionContract(
        ValueError,
        (
            r"(?i)(?:teacher_hidden.*(?:does not match|shape)|"
            r"requires teacher_weight|requires logits_teacher|"
            r"teacher_hidden and teacher_weight must both be provided|"
            r"teacher_hidden.*must not require gradients)"
        ),
    ),
    ExpectedOutcome.REJECT_UNSUPPORTED_VARIANT: RejectionContract(
        ValueError,
        (
            r"(?i)(?:(?:kl_divergence|kl variant|variant).*"
            r"(?:mutual|cojeffreys)|(?:mutual|cojeffreys).*"
            r"(?:requires|supported|valid).*separate)"
        ),
    ),
    ExpectedOutcome.REJECT_INVALID_TEACHER_STATE: RejectionContract(
        ValueError,
        r"(?i)teacher_ce_weight\s*=\s*0(?:\.0)?",
    ),
    ExpectedOutcome.REJECT_INVALID_LABEL: RejectionContract(
        ValueError,
        (
            r"(?i)(?:labels?|targets?).*"
            r"(?:outside|out(?:-| )of(?:-| )range|out of bounds|"
            r"invalid|vocab|ignore_index)"
        ),
    ),
}

assert set(REJECTION_CONTRACTS) == set(ExpectedOutcome) - {
    ExpectedOutcome.SUCCESS,
}


EXPECTED_OUTCOME_COUNTS = {
    ExpectedOutcome.SUCCESS: 490_752,
    ExpectedOutcome.REJECT_API_TEACHER_MISMATCH: 1_002_240,
    ExpectedOutcome.REJECT_UNSUPPORTED_VARIANT: 103_680,
    ExpectedOutcome.REJECT_INVALID_TEACHER_STATE: 149_760,
    ExpectedOutcome.REJECT_INVALID_LABEL: 327_168,
}


# ── Route and filter records ─────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CartesianRoute:
    teacher_representation: str

    def __post_init__(self) -> None:
        if self.teacher_representation not in TEACHER_REPRESENTATIONS:
            raise ValueError(
                f"unknown teacher representation {self.teacher_representation!r}"
            )

    @property
    def route_id(self) -> str:
        return self.teacher_representation

    def __str__(self) -> str:
        return self.route_id


CARTESIAN_ROUTES = tuple(
    CartesianRoute(teacher_representation)
    for teacher_representation in TEACHER_REPRESENTATIONS
)


def parse_cartesian_route(value: str) -> CartesianRoute:
    return CartesianRoute(value)


def cartesian_route_ordinal_range(route: CartesianRoute) -> range:
    teacher_index = TEACHER_REPRESENTATIONS.index(route.teacher_representation)
    start = teacher_index * CARTESIAN_ROUTE_CASE_COUNT
    return range(start, start + CARTESIAN_ROUTE_CASE_COUNT)


def _validate_filter_values(
    name: str,
    values: tuple[str, ...],
    allowed: tuple[str, ...],
) -> None:
    unknown = tuple(value for value in values if value not in allowed)
    if unknown:
        raise ValueError(f"unknown {name}: {unknown!r}")


@dataclass(frozen=True, slots=True)
class CartesianSelection:
    routes: tuple[CartesianRoute, ...] = ()
    kl_variants: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    api_levels: tuple[str, ...] = ()
    chunk_policies: tuple[str, ...] = ()
    label_patterns: tuple[str, ...] = ()
    outcomes: tuple[ExpectedOutcome, ...] = ()
    ordinals: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.routes)) != len(self.routes):
            raise ValueError("Cartesian routes must not contain duplicates")
        _validate_filter_values("KL variant", self.kl_variants, KL_VARIANTS)
        _validate_filter_values("dtype", self.dtypes, DTYPES)
        _validate_filter_values("API level", self.api_levels, API_LEVELS)
        _validate_filter_values("chunk policy", self.chunk_policies, CHUNK_POLICIES)
        _validate_filter_values("label pattern", self.label_patterns, LABEL_PATTERNS)
        unknown_outcomes = tuple(
            outcome
            for outcome in self.outcomes
            if not isinstance(outcome, ExpectedOutcome)
        )
        if unknown_outcomes:
            raise ValueError(f"unknown expected outcome: {unknown_outcomes!r}")
        for ordinal in self.ordinals:
            _validate_cartesian_ordinal(ordinal)


CARTESIAN_SELECTION_ENVIRONMENT_KEYS = (
    "ORDA_CARTESIAN_ROUTES",
    "ORDA_CARTESIAN_KL_VARIANTS",
    "ORDA_CARTESIAN_DTYPES",
    "ORDA_CARTESIAN_API_LEVELS",
    "ORDA_CARTESIAN_CHUNK_POLICIES",
    "ORDA_CARTESIAN_LABEL_PATTERNS",
    "ORDA_CARTESIAN_OUTCOMES",
    "ORDA_CARTESIAN_ORDINALS",
)


# ── Case records ─────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CartesianCase:
    ordinal: int
    teacher_representation: str
    kl_variant: str
    kl_weight_state: str
    student_ce_state: str
    teacher_ce_state: str
    temperature_state: str
    reduction: str
    dtype: str
    input_layout: str
    chunk_policy: str
    label_pattern: str
    api_level: str

    @property
    def kl_weight(self) -> float:
        return 0.0 if self.kl_weight_state == "zero" else 0.35

    @property
    def student_ce_weight(self) -> float:
        return 0.0 if self.student_ce_state == "zero" else 0.8

    @property
    def teacher_ce_weight(self) -> float | None:
        if self.teacher_ce_state == "default":
            return None
        return 0.0 if self.teacher_ce_state == "zero" else 0.5

    @property
    def kl_temperature(self) -> float:
        return 1.0 if self.temperature_state == "one" else 1.4

    @property
    def case_id(self) -> str:
        dimensions = (
            self.teacher_representation,
            self.kl_variant,
            self.kl_weight_state,
            self.student_ce_state,
            self.teacher_ce_state,
            self.temperature_state,
            self.reduction,
            self.dtype,
            self.input_layout,
            self.chunk_policy,
            self.label_pattern,
            self.api_level,
        )
        return f"{self.ordinal:06d}-" + "-".join(dimensions)


def _validate_cartesian_ordinal(ordinal: int) -> None:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise TypeError("Cartesian ordinal must be an integer")
    if not 0 <= ordinal < TOTAL_CARTESIAN_CASES:
        raise ValueError(
            f"Cartesian ordinal must satisfy 0 <= ordinal < {TOTAL_CARTESIAN_CASES}"
        )


def cartesian_case_from_ordinal(ordinal: int) -> CartesianCase:
    """Decode one global ordinal using the original product dimension order."""
    _validate_cartesian_ordinal(ordinal)
    remainder = ordinal
    values: list[str] = []
    for dimension, stride in zip(MATRIX_DIMENSIONS, MATRIX_STRIDES, strict=True):
        index, remainder = divmod(remainder, stride)
        values.append(dimension[index])
    return CartesianCase(ordinal, *values)


def cartesian_ordinal_for_case(case: CartesianCase) -> int:
    """Encode a case's dimensions independently of its stored ordinal."""
    return sum(
        dimension.index(getattr(case, name)) * stride
        for name, dimension, stride in zip(
            MATRIX_FIELD_NAMES,
            MATRIX_DIMENSIONS,
            MATRIX_STRIDES,
            strict=True,
        )
    )


def cartesian_route_for_case(case: CartesianCase) -> CartesianRoute:
    return CartesianRoute(case.teacher_representation)


def cartesian_route_for_ordinal(ordinal: int) -> CartesianRoute:
    return cartesian_route_for_case(cartesian_case_from_ordinal(ordinal))


# ── Case classification ──────────────────────────────────────────────────────
def representation_mode(case: CartesianCase) -> str:
    if case.teacher_representation in (
        "precomputed_logits",
        "precomputed_hidden_weight",
    ):
        return "precomputed"
    return case.teacher_representation


def invoked_mode(case: CartesianCase) -> str:
    if case.api_level == "high_level":
        return representation_mode(case)
    return {
        "direct_tied": "tied",
        "direct_separate": "separate",
        "direct_precomputed": "precomputed",
    }[case.api_level]


def api_accepts_representation(case: CartesianCase) -> bool:
    if case.api_level == "high_level":
        return True
    if case.api_level == "direct_tied":
        return case.teacher_representation == "tied"
    if case.api_level == "direct_separate":
        return case.teacher_representation == "separate"
    return case.teacher_representation in (
        "precomputed_logits",
        "precomputed_hidden_weight",
    )


def effective_teacher_ce_weight(case: CartesianCase) -> float:
    explicit_weight = case.teacher_ce_weight
    if explicit_weight is not None:
        return explicit_weight
    return 1.0 if invoked_mode(case) == "tied" else 0.0


def classify_case(case: CartesianCase) -> ExpectedOutcome:
    mode = invoked_mode(case)
    if mode == "precomputed" and case.kl_variant in (
        "mutual",
        "cojeffreys",
    ):
        return ExpectedOutcome.REJECT_UNSUPPORTED_VARIANT

    if not api_accepts_representation(case):
        return ExpectedOutcome.REJECT_API_TEACHER_MISMATCH

    teacher_ce_weight = effective_teacher_ce_weight(case)
    if mode == "precomputed" and teacher_ce_weight != 0.0:
        return ExpectedOutcome.REJECT_INVALID_TEACHER_STATE

    if case.label_pattern in ("invalid_negative", "invalid_high"):
        return ExpectedOutcome.REJECT_INVALID_LABEL
    return ExpectedOutcome.SUCCESS


def cartesian_outcome_counts(
    routes: Iterable[CartesianRoute] = CARTESIAN_ROUTES,
) -> dict[ExpectedOutcome, int]:
    """Count outcomes from only the dimensions used by ``classify_case``."""
    counts = dict.fromkeys(ExpectedOutcome, 0)
    independent_multiplier = prod(
        len(MATRIX_DIMENSIONS[index]) for index in (2, 3, 5, 6, 7, 8, 9)
    )
    for route, kl_variant, teacher_ce_state, label_pattern, api_level in product(
        routes,
        KL_VARIANTS,
        TEACHER_CE_STATES,
        LABEL_PATTERNS,
        API_LEVELS,
    ):
        case = CartesianCase(
            0,
            route.teacher_representation,
            kl_variant,
            KL_WEIGHT_STATES[0],
            STUDENT_CE_STATES[0],
            teacher_ce_state,
            TEMPERATURE_STATES[0],
            REDUCTIONS[0],
            DTYPES[0],
            INPUT_LAYOUTS[0],
            CHUNK_POLICIES[0],
            label_pattern,
            api_level,
        )
        outcome = classify_case(case)
        counts[outcome] += independent_multiplier
    return counts


def _split_environment_values(
    environ: Mapping[str, str],
    name: str,
) -> tuple[str, ...]:
    raw = environ.get(name)
    if raw is None:
        return ()
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def cartesian_selection_from_environment(
    environ: Mapping[str, str] | None = None,
) -> CartesianSelection:
    values = os.environ if environ is None else environ
    route_values = _split_environment_values(values, "ORDA_CARTESIAN_ROUTES")
    ordinal_values = _split_environment_values(values, "ORDA_CARTESIAN_ORDINALS")
    routes = tuple(parse_cartesian_route(value) for value in route_values)
    try:
        ordinals = tuple(int(value) for value in ordinal_values)
    except ValueError as exc:
        raise ValueError("ORDA_CARTESIAN_ORDINALS must contain integers") from exc
    if not routes and not ordinals:
        raise ValueError(
            "Cartesian execution requires ORDA_CARTESIAN_ROUTES or "
            "ORDA_CARTESIAN_ORDINALS; use python test/run_test_gpu.py"
        )
    try:
        outcomes = tuple(
            ExpectedOutcome(value)
            for value in _split_environment_values(
                values,
                "ORDA_CARTESIAN_OUTCOMES",
            )
        )
    except ValueError as exc:
        raise ValueError("ORDA_CARTESIAN_OUTCOMES contains an unknown outcome") from exc
    return CartesianSelection(
        routes=routes,
        kl_variants=_split_environment_values(
            values,
            "ORDA_CARTESIAN_KL_VARIANTS",
        ),
        dtypes=_split_environment_values(values, "ORDA_CARTESIAN_DTYPES"),
        api_levels=_split_environment_values(values, "ORDA_CARTESIAN_API_LEVELS"),
        chunk_policies=_split_environment_values(
            values,
            "ORDA_CARTESIAN_CHUNK_POLICIES",
        ),
        label_patterns=_split_environment_values(
            values,
            "ORDA_CARTESIAN_LABEL_PATTERNS",
        ),
        outcomes=outcomes,
        ordinals=ordinals,
    )


def _case_matches_selection(
    case: CartesianCase,
    selection: CartesianSelection,
    route_set: frozenset[CartesianRoute],
) -> bool:
    if route_set and cartesian_route_for_case(case) not in route_set:
        return False
    if selection.kl_variants and case.kl_variant not in selection.kl_variants:
        return False
    if selection.dtypes and case.dtype not in selection.dtypes:
        return False
    if selection.api_levels and case.api_level not in selection.api_levels:
        return False
    if selection.chunk_policies and case.chunk_policy not in selection.chunk_policies:
        return False
    if selection.label_patterns and case.label_pattern not in selection.label_patterns:
        return False
    return not selection.outcomes or classify_case(case) in selection.outcomes


def iter_selected_cartesian_cases(
    selection: CartesianSelection,
    shard: ShardSpec | None = None,
) -> Iterator[CartesianCase]:
    """Generate selected cases from route-local mixed-radix digits."""
    if not selection.routes and not selection.ordinals:
        raise ValueError("Cartesian selection requires at least one route or ordinal")
    route_set = frozenset(selection.routes)
    if selection.ordinals:
        for ordinal in sorted(set(selection.ordinals)):
            if shard is not None and ordinal % shard.count != shard.index:
                continue
            case = cartesian_case_from_ordinal(ordinal)
            if _case_matches_selection(case, selection, route_set):
                yield case
        return

    routes = selection.routes or CARTESIAN_ROUTES
    routes = tuple(
        sorted(routes, key=lambda route: cartesian_route_ordinal_range(route).start)
    )
    allowed_by_dimension = {
        1: frozenset(selection.kl_variants),
        7: frozenset(selection.dtypes),
        9: frozenset(selection.chunk_policies),
        10: frozenset(selection.label_patterns),
        11: frozenset(selection.api_levels),
    }
    index_choices = tuple(
        tuple(
            index
            for index, value in enumerate(dimension)
            if not allowed_by_dimension.get(dimension_index)
            or value in allowed_by_dimension[dimension_index]
        )
        for dimension_index, dimension in enumerate(MATRIX_DIMENSIONS[1:], start=1)
    )
    for route in routes:
        start = cartesian_route_ordinal_range(route).start
        for tail_indices in product(*index_choices):
            ordinal = start + sum(
                index * MATRIX_STRIDES[dimension_index]
                for dimension_index, index in enumerate(tail_indices, start=1)
            )
            if shard is not None and ordinal % shard.count != shard.index:
                continue
            tail_values = tuple(
                MATRIX_DIMENSIONS[dimension_index][index]
                for dimension_index, index in enumerate(tail_indices, start=1)
            )
            case = CartesianCase(
                ordinal,
                route.teacher_representation,
                *tail_values,
            )
            if not selection.outcomes or classify_case(case) in selection.outcomes:
                yield case


# ── Case generation and sharding ─────────────────────────────────────────────
def iter_cartesian_cases() -> Iterator[CartesianCase]:
    for ordinal, values in enumerate(product(*MATRIX_DIMENSIONS)):
        yield CartesianCase(ordinal, *values)


@dataclass(frozen=True, slots=True)
class ShardSpec:
    count: int
    index: int


def shard_spec_from_environment(
    environ: Mapping[str, str] | None = None,
) -> ShardSpec:
    values = os.environ if environ is None else environ
    raw_count = values.get("ORDA_CARTESIAN_SHARD_COUNT", "1")
    raw_index = values.get("ORDA_CARTESIAN_SHARD_INDEX", "0")
    try:
        count = int(raw_count)
        index = int(raw_index)
    except ValueError as exc:
        raise ValueError(
            "ORDA_CARTESIAN_SHARD_COUNT and ORDA_CARTESIAN_SHARD_INDEX must be integers"
        ) from exc
    if count < 1:
        raise ValueError("ORDA_CARTESIAN_SHARD_COUNT must be >= 1")
    if not 0 <= index < count:
        raise ValueError(
            "ORDA_CARTESIAN_SHARD_INDEX must satisfy "
            "0 <= index < ORDA_CARTESIAN_SHARD_COUNT"
        )
    return ShardSpec(count=count, index=index)


def selected_cartesian_cases_for_shard(
    selection: CartesianSelection,
    shard: ShardSpec,
) -> tuple[CartesianCase, ...]:
    """Reject a globally empty selection while allowing an empty shard."""
    if next(iter_selected_cartesian_cases(selection), None) is None:
        raise ValueError("Cartesian selection matched no cases before sharding")
    return tuple(iter_selected_cartesian_cases(selection, shard=shard))


def iter_cartesian_shard(spec: ShardSpec) -> Iterator[CartesianCase]:
    for ordinal in range(spec.index, TOTAL_CARTESIAN_CASES, spec.count):
        yield cartesian_case_from_ordinal(ordinal)


def shard_case_count(spec: ShardSpec) -> int:
    if spec.index >= TOTAL_CARTESIAN_CASES:
        return 0
    return (TOTAL_CARTESIAN_CASES - 1 - spec.index) // spec.count + 1


# ── Grouped execution ────────────────────────────────────────────────────────
def cartesian_group_key(case: CartesianCase) -> tuple[str, str, str, str]:
    return (
        case.teacher_representation,
        case.kl_variant,
        case.dtype,
        case.api_level,
    )


def group_cartesian_cases(
    cases: Iterable[CartesianCase],
) -> dict[tuple[str, str, str, str], tuple[CartesianCase, ...]]:
    groups: dict[tuple[str, str, str, str], list[CartesianCase]] = {}
    for case in cases:
        groups.setdefault(cartesian_group_key(case), []).append(case)
    return {key: tuple(members) for key, members in groups.items()}
