from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, TypeAlias, final

import torch

from ._scalar_validation import _is_normal_float32

# ── Public type aliases ──────────────────────────────────────────────────────
Reduction = Literal["mean", "sum"]
AutotuneMode: TypeAlias = bool | Literal["Max"]
Profile = Literal["balanced", "fast", "debug"]
Backend = Literal["auto", "triton", "torch"]
GradWeightAccumulation = Literal["auto", "compute", "fp32"]
KLVariant = Literal[
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
]
TiedTeacherMode = Literal["trained", "frozen", "cotrained"]


# ── KL hyperparameters ───────────────────────────────────────────────────────
@final
@dataclass(frozen=True, slots=True)
class EntropyGate:
    """Teacher-entropy gate for the ``entropy_adaptive`` KL variant.

    The per-row blend weight is ``alpha_min + (alpha_max - alpha_min) *
    sigmoid((h - tau) / scale)`` where ``h`` is the teacher entropy normalized
    by ``log(vocab)`` (``0`` when ``vocab == 1``).  The gate is computed from
    the detached teacher distribution and never receives gradient.
    """

    alpha_min: float = 0.0
    alpha_max: float = 1.0
    tau: float = 0.5
    scale: float = 0.1

    def __post_init__(self) -> None:
        for name in ("alpha_min", "alpha_max", "tau", "scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"EntropyGate.{name} must be a float, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"EntropyGate.{name} must be finite, got {value!r}")
        if not 0.0 <= float(self.alpha_min) <= float(self.alpha_max) <= 1.0:
            raise ValueError(
                "EntropyGate requires 0 <= alpha_min <= alpha_max <= 1, got "
                f"alpha_min={self.alpha_min}, alpha_max={self.alpha_max}"
            )
        if float(self.scale) <= 0.0:
            raise ValueError(f"EntropyGate.scale must be > 0, got {self.scale}")
        try:
            inverse_scale = 1.0 / float(self.scale)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError(
                "EntropyGate.scale and its reciprocal must be finite normal "
                f"float32 values, got {self.scale}"
            ) from exc
        if not all(
            _is_normal_float32(value) for value in (float(self.scale), inverse_scale)
        ):
            raise ValueError(
                "EntropyGate.scale and its reciprocal must be finite normal "
                f"float32 values, got {self.scale}"
            )


# ── Chunk strategy ───────────────────────────────────────────────────────────
_FAST_CHUNK_SENTINEL = "fast"


@final
@dataclass(frozen=True, slots=True)
class ChunkSize:
    """Nominal chunk strategy used at public API boundaries."""

    _nrows: int | str | None

    def __post_init__(self) -> None:
        if self._nrows is None:
            return
        if isinstance(self._nrows, str):
            if self._nrows != _FAST_CHUNK_SENTINEL:
                raise ValueError(
                    "ChunkSize must be created with ChunkSize.nrows(...), "
                    f"ChunkSize.fast(), or ChunkSize.dynamic(), got {self._nrows!r}"
                )
            return
        if isinstance(self._nrows, bool) or not isinstance(self._nrows, int):
            raise TypeError(
                f"ChunkSize.nrows() requires an int, got {type(self._nrows).__name__}"
            )
        if self._nrows < 1:
            raise ValueError(f"chunk size must be >= 1, got {self._nrows}")

    @classmethod
    def nrows(cls, n: int) -> ChunkSize:
        """Use exactly ``n`` rows per chunk, clamped to the batch size."""
        return cls(_nrows=n)

    @classmethod
    def dynamic(cls) -> ChunkSize:
        """Choose the initial partition from the runtime workload shape."""
        return cls(_nrows=None)

    @classmethod
    def fast(cls) -> ChunkSize:
        """Target 1024 rows per chunk, with at least 2 and at most 16 chunks.

        Unlike :meth:`dynamic`, the partition ignores the vocabulary size and is
        never retried: an out-of-memory failure is reported so the caller can
        choose a new ``num_chunks``.
        """
        return cls(_nrows=_FAST_CHUNK_SENTINEL)

    @property
    def is_dynamic(self) -> bool:
        return self._nrows is None

    @property
    def is_fast(self) -> bool:
        return self._nrows == _FAST_CHUNK_SENTINEL

    @property
    def value(self) -> int:
        if self._nrows is None:
            raise TypeError("dynamic ChunkSize has no nrows value")
        if isinstance(self._nrows, str):
            raise TypeError("fast ChunkSize has no nrows value")
        return self._nrows

    def __repr__(self) -> str:
        if self.is_dynamic:
            return "ChunkSize.dynamic()"
        if self.is_fast:
            return "ChunkSize.fast()"
        return f"ChunkSize.nrows({self.value})"


_DEFAULT_CHUNK_SIZE = ChunkSize.dynamic()


# ── Precomputed representation invariant ─────────────────────────────────────
def _validate_precomputed_representation(
    *,
    logits: torch.Tensor | None,
    teacher_hidden: torch.Tensor | None,
    teacher_weight: torch.Tensor | None,
    owner: str,
    logits_name: str,
) -> Literal["logits", "hidden_weight"]:
    for name, tensor in (
        (logits_name, logits),
        ("teacher_hidden", teacher_hidden),
        ("teacher_weight", teacher_weight),
    ):
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{owner}: {name} must be a torch.Tensor, got {type(tensor).__name__}"
            )

    has_logits = logits is not None
    has_hidden = teacher_hidden is not None
    has_weight = teacher_weight is not None

    if has_logits and (has_hidden or has_weight):
        raise ValueError(
            f"{owner}: provide {logits_name} or "
            "(teacher_hidden, teacher_weight), not both"
        )
    if not has_logits and not has_hidden and not has_weight:
        raise ValueError(
            f"{owner}: must provide {logits_name} or (teacher_hidden, teacher_weight)"
        )
    if has_hidden != has_weight:
        raise ValueError(
            f"{owner}: teacher_hidden and teacher_weight must both be provided"
        )

    cached_tensors = (
        (logits_name, logits),
        ("teacher_hidden", teacher_hidden),
        ("teacher_weight", teacher_weight),
    )
    for name, tensor in cached_tensors:
        if tensor is not None and tensor.requires_grad:
            raise ValueError(
                f"{owner}: cached {name} must not require gradients; use "
                "SeparateTeacher when the teacher path needs gradients"
            )
    return "logits" if has_logits else "hidden_weight"


def _validate_precomputed_dtypes(
    *,
    logits: torch.Tensor | None,
    teacher_hidden: torch.Tensor | None,
    teacher_weight: torch.Tensor | None,
    owner: str = "PrecomputedTeacher",
    logits_name: str = "logits",
) -> None:
    named = (
        (logits_name, logits),
        ("teacher_hidden", teacher_hidden),
        ("teacher_weight", teacher_weight),
    )
    for name, tensor in named:
        if tensor is None:
            continue
        if tensor.layout != torch.strided:
            raise ValueError(
                f"{owner}.{name} must use torch.strided layout, got {tensor.layout}"
            )
        if not tensor.is_floating_point():
            raise ValueError(f"{owner}.{name} must be floating, got {tensor.dtype}")


def _validate_precomputed_values(
    *,
    logits: torch.Tensor | None,
    teacher_hidden: torch.Tensor | None,
    teacher_weight: torch.Tensor | None,
    owner: str = "PrecomputedTeacher",
    logits_name: str = "logits",
) -> None:
    _validate_precomputed_dtypes(
        logits=logits,
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        owner=owner,
        logits_name=logits_name,
    )

    if logits is not None:
        if logits.device.type == "meta" or logits.numel() == 0:
            return
        maximum = float(logits.amax())
        if math.isnan(maximum) or maximum == math.inf:
            raise ValueError(
                f"{owner}.{logits_name} may contain only finite values or -inf"
            )
        return
    for name, tensor in (
        ("teacher_hidden", teacher_hidden),
        ("teacher_weight", teacher_weight),
    ):
        if tensor is None or tensor.device.type == "meta" or tensor.numel() == 0:
            continue
        lower_bound, upper_bound = torch.aminmax(tensor)
        if not bool(torch.isfinite(lower_bound) & torch.isfinite(upper_bound)):
            raise ValueError(f"{owner}.{name} must contain only finite values")


def _validate_precomputed_span_values(span_log_probs: torch.Tensor) -> None:
    owner = "PrecomputedSpanTeacher.span_log_probs"
    if span_log_probs.layout != torch.strided:
        raise ValueError(
            f"{owner} must use torch.strided layout, got {span_log_probs.layout}"
        )
    if not span_log_probs.is_floating_point():
        raise ValueError(f"{owner} must be floating, got {span_log_probs.dtype}")
    if span_log_probs.device.type == "meta" or span_log_probs.numel() == 0:
        return
    maximum = float(span_log_probs.amax())
    if math.isnan(maximum):
        raise ValueError(f"{owner} contains NaN")
    if maximum == math.inf:
        raise ValueError(f"{owner} contains +inf (-inf is allowed)")
    if maximum > 0.0:
        raise ValueError(f"{owner} must contain values <= 0 (-inf is allowed)")


@dataclass(frozen=True, slots=True)
class _TensorValidationStamp:
    object_id: int
    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    requires_grad: bool
    version: int | None


def _capture_tensor_validation_stamp(
    tensor: torch.Tensor,
    *,
    owner: str,
    name: str,
    trust_cache_immutability: bool,
) -> _TensorValidationStamp:
    try:
        version = int(tensor._version)
    except RuntimeError as exc:
        if not trust_cache_immutability:
            raise ValueError(
                f"{owner}.{name} does not expose a version counter; pass "
                "trust_cache_immutability=True only when the cache will remain immutable"
            ) from exc
        version = None
    return _TensorValidationStamp(
        object_id=id(tensor),
        data_ptr=int(tensor.data_ptr()),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        dtype=tensor.dtype,
        device=tensor.device,
        layout=tensor.layout,
        requires_grad=bool(tensor.requires_grad),
        version=version,
    )


def _capture_precomputed_stamps(
    *,
    owner: str,
    tensors: tuple[tuple[str, torch.Tensor | None], ...],
    trust_cache_immutability: bool,
) -> tuple[tuple[str, _TensorValidationStamp], ...]:
    return tuple(
        (
            name,
            _capture_tensor_validation_stamp(
                tensor,
                owner=owner,
                name=name,
                trust_cache_immutability=trust_cache_immutability,
            ),
        )
        for name, tensor in tensors
        if tensor is not None
    )


def _validate_precomputed_stamps(
    *,
    owner: str,
    tensors: tuple[tuple[str, torch.Tensor | None], ...],
    expected: tuple[tuple[str, _TensorValidationStamp], ...],
) -> None:
    current_tensors = tuple(
        (name, tensor) for name, tensor in tensors if tensor is not None
    )
    if tuple(name for name, _tensor in current_tensors) != tuple(
        name for name, _stamp in expected
    ):
        raise ValueError(f"{owner} cache representation was modified after validation")

    for (name, tensor), (expected_name, stamp) in zip(
        current_tensors, expected, strict=True
    ):
        assert tensor is not None and name == expected_name
        if tensor.requires_grad:
            raise ValueError(
                f"{owner}: cached {name} must not require gradients; the cache "
                "was modified after validation"
            )
        current = _capture_tensor_validation_stamp(
            tensor,
            owner=owner,
            name=name,
            trust_cache_immutability=stamp.version is None,
        )
        if current != stamp:
            raise ValueError(
                f"{owner}.{name} was modified after validation; construct a new descriptor"
            )


# ── Teacher descriptors ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class TiedTeacher:
    hidden: torch.Tensor
    # ``mode`` is a declarative consistency check, not a routing switch: the
    # effective teacher role still derives from the loss weights and variant.
    mode: TiedTeacherMode | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hidden, torch.Tensor):
            raise TypeError(
                f"TiedTeacher.hidden must be a torch.Tensor, got {type(self.hidden).__name__}"
            )
        if self.mode is not None and self.mode not in (
            "trained",
            "frozen",
            "cotrained",
        ):
            raise ValueError(
                "TiedTeacher.mode must be None, 'trained', 'frozen', or "
                f"'cotrained', got {self.mode!r}"
            )


@dataclass(frozen=True)
class SeparateTeacher:
    hidden: torch.Tensor
    weight: torch.Tensor

    def __post_init__(self) -> None:
        for name, tensor in (("hidden", self.hidden), ("weight", self.weight)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"SeparateTeacher.{name} must be a torch.Tensor, "
                    f"got {type(tensor).__name__}"
                )


@dataclass(frozen=True)
class PrecomputedTeacher:
    """Validated, gradient-free teacher cache.

    Cache tensors are not cloned. Normal in-place mutations are detected from
    their version counters; unsafe ``.data`` mutation is outside this guarantee.

    ``logits_temperature=None`` declares raw logits: the loss applies its own
    teacher temperature. A float declares logits already tempered at that
    temperature (e.g. ``project_teacher_logits(..., temperature=T)``, where
    temperature must precede many-to-one aggregation): a defaulted
    ``kl_teacher_temperature`` adopts it, an explicit conflicting value is
    rejected, and the loss applies none of its own.
    """

    logits: torch.Tensor | None = None
    teacher_hidden: torch.Tensor | None = None
    teacher_weight: torch.Tensor | None = None
    trust_cache_immutability: bool = False
    logits_temperature: float | None = None
    _validation_stamps: tuple[tuple[str, _TensorValidationStamp], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.trust_cache_immutability, bool):
            raise TypeError(
                "PrecomputedTeacher.trust_cache_immutability must be bool, got "
                f"{self.trust_cache_immutability!r}"
            )
        if self.logits_temperature is not None:
            if isinstance(self.logits_temperature, bool) or not isinstance(
                self.logits_temperature, (int, float)
            ):
                raise TypeError(
                    "PrecomputedTeacher.logits_temperature must be a float or "
                    f"None, got {self.logits_temperature!r}"
                )
            normalized = float(self.logits_temperature)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(
                    "PrecomputedTeacher.logits_temperature must be finite and "
                    f"> 0, got {self.logits_temperature!r}"
                )
            if self.logits is None:
                raise ValueError(
                    "PrecomputedTeacher.logits_temperature requires the logits "
                    "representation; hidden/weight caches hold raw logits"
                )
            object.__setattr__(self, "logits_temperature", normalized)
        _validate_precomputed_representation(
            logits=self.logits,
            teacher_hidden=self.teacher_hidden,
            teacher_weight=self.teacher_weight,
            owner="PrecomputedTeacher",
            logits_name="logits",
        )
        _validate_precomputed_values(
            logits=self.logits,
            teacher_hidden=self.teacher_hidden,
            teacher_weight=self.teacher_weight,
        )
        object.__setattr__(
            self,
            "_validation_stamps",
            _capture_precomputed_stamps(
                owner="PrecomputedTeacher",
                tensors=(
                    ("logits", self.logits),
                    ("teacher_hidden", self.teacher_hidden),
                    ("teacher_weight", self.teacher_weight),
                ),
                trust_cache_immutability=self.trust_cache_immutability,
            ),
        )

    def _validate_cache_unchanged(self) -> None:
        _validate_precomputed_stamps(
            owner="PrecomputedTeacher",
            tensors=(
                ("logits", self.logits),
                ("teacher_hidden", self.teacher_hidden),
                ("teacher_weight", self.teacher_weight),
            ),
            expected=self._validation_stamps,
        )

    def prepare_logits(self, dtype: torch.dtype) -> PrecomputedTeacher:
        """Return an explicitly materialized logits cache for ``dtype``.

        A mismatched cache is row-centered in its source dtype before casting,
        matching the chunk-wise runtime conversion. The caller owns the
        returned descriptor and its persistent tensor; nothing is retained
        implicitly.
        """
        from ._runtime import TORCH_MODEL_DTYPES, cast_logits_for_compute

        if not isinstance(dtype, torch.dtype):
            raise TypeError(
                "PrecomputedTeacher.prepare_logits dtype must be a torch.dtype, "
                f"got {dtype!r}"
            )
        if dtype not in TORCH_MODEL_DTYPES:
            raise ValueError(
                "PrecomputedTeacher.prepare_logits dtype must be one of "
                f"{sorted(map(str, TORCH_MODEL_DTYPES))}, got {dtype}"
            )

        self._validate_cache_unchanged()
        if self.logits is None:
            raise ValueError(
                "PrecomputedTeacher.prepare_logits requires the logits representation"
            )
        if self.logits.dtype == dtype and self.logits.is_contiguous():
            return self

        # Preparation may be called from an inference-mode setup block. Build a
        # normal tensor so the new descriptor keeps version-based mutation checks.
        with torch.inference_mode(False):
            prepared_logits = cast_logits_for_compute(
                self.logits,
                dtype,
            ).contiguous()
        return PrecomputedTeacher(
            logits=prepared_logits,
            trust_cache_immutability=self.trust_cache_immutability,
            logits_temperature=self.logits_temperature,
        )


# ── Cross-vocab span descriptors ─────────────────────────────────────────────
def _validate_range_tensor(owner: str, name: str, ranges: torch.Tensor) -> None:
    if not isinstance(ranges, torch.Tensor):
        raise TypeError(
            f"{owner}.{name} must be a torch.Tensor, got {type(ranges).__name__}"
        )
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(
            f"{owner}.{name} must have shape (C, 2), got {tuple(ranges.shape)}"
        )
    if ranges.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{owner}.{name} must be int32 or int64, got {ranges.dtype}")


@final
@dataclass(frozen=True)
class AlignedSpanPairs:
    """Half-open token ranges pairing student and teacher spans of one text.

    Row ``c`` declares that student rows ``student_ranges[c, 0]:student_ranges
    [c, 1]`` and teacher rows ``teacher_ranges[c, 0]:teacher_ranges[c, 1]``
    tokenize the same span. Bounds are validated against the batch shapes at
    call time.
    """

    student_ranges: torch.Tensor
    teacher_ranges: torch.Tensor

    def __post_init__(self) -> None:
        _validate_range_tensor(
            "AlignedSpanPairs", "student_ranges", self.student_ranges
        )
        _validate_range_tensor(
            "AlignedSpanPairs", "teacher_ranges", self.teacher_ranges
        )
        if self.student_ranges.shape[0] != self.teacher_ranges.shape[0]:
            raise ValueError(
                "AlignedSpanPairs requires one teacher range per student range, "
                f"got {self.student_ranges.shape[0]} and "
                f"{self.teacher_ranges.shape[0]}"
            )


@final
@dataclass(frozen=True)
class CrossVocabTeacher:
    """Online teacher over its own vocabulary for span-aligned distillation.

    The teacher is a reference model: its span likelihoods are detached, so
    none of these tensors ever receive gradient from the span objective.
    """

    hidden: torch.Tensor
    weight: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        for name, tensor in (
            ("hidden", self.hidden),
            ("weight", self.weight),
            ("labels", self.labels),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"CrossVocabTeacher.{name} must be a torch.Tensor, "
                    f"got {type(tensor).__name__}"
                )


@final
@dataclass(frozen=True)
class PrecomputedSpanTeacher:
    """Validated per-span teacher log-likelihoods ``[C]``.

    Values must be floating log-probabilities no greater than zero; ``-inf``
    represents zero support. Unsafe ``.data`` mutation is outside the stamp.

    ``teacher_ranges`` records the ranges the cache was materialized from and
    is checked against ``AlignedSpanPairs.teacher_ranges`` at call time;
    ``trust_teacher_ranges=True`` skips that binding for self-built caches.
    """

    span_log_probs: torch.Tensor
    trust_cache_immutability: bool = False
    teacher_ranges: torch.Tensor | None = None
    trust_teacher_ranges: bool = False
    _validation_stamps: tuple[tuple[str, _TensorValidationStamp], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.span_log_probs, torch.Tensor):
            raise TypeError(
                "PrecomputedSpanTeacher.span_log_probs must be a torch.Tensor, "
                f"got {type(self.span_log_probs).__name__}"
            )
        if self.span_log_probs.ndim != 1:
            raise ValueError(
                "PrecomputedSpanTeacher.span_log_probs must be 1D, got "
                f"{tuple(self.span_log_probs.shape)}"
            )
        if self.span_log_probs.requires_grad:
            raise ValueError(
                "PrecomputedSpanTeacher: cached span_log_probs must not require "
                "gradients; use CrossVocabTeacher when the teacher is live"
            )
        if not isinstance(self.trust_cache_immutability, bool):
            raise TypeError(
                "PrecomputedSpanTeacher.trust_cache_immutability must be bool, got "
                f"{self.trust_cache_immutability!r}"
            )
        if not isinstance(self.trust_teacher_ranges, bool):
            raise TypeError(
                "PrecomputedSpanTeacher.trust_teacher_ranges must be bool, got "
                f"{self.trust_teacher_ranges!r}"
            )
        if self.teacher_ranges is None:
            if not self.trust_teacher_ranges:
                raise ValueError(
                    "PrecomputedSpanTeacher requires the teacher_ranges the cache "
                    "was built from (build_precomputed_span_teacher provides "
                    "them); pass trust_teacher_ranges=True to accept an unbound "
                    "cache"
                )
        else:
            if self.trust_teacher_ranges:
                raise ValueError(
                    "PrecomputedSpanTeacher: teacher_ranges and "
                    "trust_teacher_ranges are mutually exclusive"
                )
            _validate_range_tensor(
                "PrecomputedSpanTeacher", "teacher_ranges", self.teacher_ranges
            )
            if self.teacher_ranges.shape[0] != self.span_log_probs.shape[0]:
                raise ValueError(
                    "PrecomputedSpanTeacher.teacher_ranges must have one range "
                    f"per span, got {self.teacher_ranges.shape[0]} for "
                    f"{self.span_log_probs.shape[0]} spans"
                )
            if (
                self.teacher_ranges.device.type != "meta"
                and self.teacher_ranges.numel() != 0
            ):
                starts = self.teacher_ranges[:, 0]
                ends = self.teacher_ranges[:, 1]
                if bool((starts < 0).any()) or bool((starts >= ends).any()):
                    raise ValueError(
                        "PrecomputedSpanTeacher.teacher_ranges must satisfy "
                        "0 <= start < end for every span"
                    )
        _validate_precomputed_span_values(self.span_log_probs)
        object.__setattr__(
            self,
            "_validation_stamps",
            _capture_precomputed_stamps(
                owner="PrecomputedSpanTeacher",
                tensors=(
                    ("span_log_probs", self.span_log_probs),
                    ("teacher_ranges", self.teacher_ranges),
                ),
                trust_cache_immutability=self.trust_cache_immutability,
            ),
        )

    def _validate_cache_unchanged(self) -> None:
        _validate_precomputed_stamps(
            owner="PrecomputedSpanTeacher",
            tensors=(
                ("span_log_probs", self.span_log_probs),
                ("teacher_ranges", self.teacher_ranges),
            ),
            expected=self._validation_stamps,
        )


@final
@dataclass(frozen=True, slots=True)
class AlignedLikelihoodConfig:
    """Span-likelihood objective configuration.

    The event space per span is binary — the observed continuation versus any
    other — so ``objective`` selects binary forward KL or binary cross-entropy
    between the teacher and student span likelihoods.
    """

    student_ce_weight: float = 1.0
    span_kl_weight: float = 1.0
    ignore_index: int = -100
    reduction: Reduction = "mean"
    objective: Literal["binary_kl", "binary_ce"] = "binary_kl"

    def __post_init__(self) -> None:
        for name in ("student_ce_weight", "span_kl_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"AlignedLikelihoodConfig.{name} must be a float, got {value!r}"
                )
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"AlignedLikelihoodConfig.{name} must be finite and >= 0, "
                    f"got {value!r}"
                )
        if isinstance(self.ignore_index, bool) or not isinstance(
            self.ignore_index, int
        ):
            raise ValueError(
                "AlignedLikelihoodConfig.ignore_index must be an integer, got "
                f"{self.ignore_index!r}"
            )
        if self.reduction not in ("mean", "sum"):
            raise ValueError(
                "AlignedLikelihoodConfig.reduction must be 'mean' or 'sum', got "
                f"{self.reduction!r}"
            )
        if self.objective not in ("binary_kl", "binary_ce"):
            raise ValueError(
                "AlignedLikelihoodConfig.objective must be 'binary_kl' or "
                f"'binary_ce', got {self.objective!r}"
            )


# ── Kernel configuration ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class KernelConfig:
    """Cold-path execution overrides.

    ``assume_valid_labels=True`` skips the host range scan. Kernel masking and
    ``ignore_index``-safe indexing remain active.

    ``grad_weight_accumulation`` selects the dtype of the weight-gradient
    accumulators, resolved per parameter against its own dtype:

    ``auto``
        float32 when the parameter dtype differs from the compute dtype, or for
        float16 parameters; otherwise the parameter dtype.
    ``compute``
        always the compute dtype, minimising accumulator memory.
    ``fp32``
        always float32. Each chunk's weight-gradient contribution is added to
        ``grad_weight`` in float32, limiting accumulated rounding error across
        many chunks. This is useful when training in BF16 but grad-weight
        accuracy needs to be preserved.
    """

    grad_weight_accumulation: GradWeightAccumulation | None = None
    chunk_size: ChunkSize | None = None
    num_chunks: int | None = None
    max_chunks: int | None = None
    max_fused_size: int | None = None
    kl_weight: float | None = None
    kl_temperature: float | None = None
    autotune: AutotuneMode | None = None
    refresh_autotune: bool | None = None
    kl_divergence: KLVariant | None = None
    assume_valid_labels: bool | None = None
    kl_alpha: float | None = None
    kl_beta: float | None = None
    entropy_gate: EntropyGate | None = None
    kl_teacher_temperature: float | None = None

    def __post_init__(self) -> None:
        # Keep this local to avoid importing _validation back into this module.
        if self.chunk_size is not None and not isinstance(self.chunk_size, ChunkSize):
            raise TypeError(
                "KernelConfig.chunk_size must be created with "
                "ChunkSize.nrows(...), ChunkSize.fast(), or ChunkSize.dynamic(), got "
                f"{self.chunk_size!r}"
            )
        if self.autotune is not None and not (
            isinstance(self.autotune, bool) or self.autotune == "Max"
        ):
            raise ValueError(
                "KernelConfig.autotune must be bool, 'Max', or None, got "
                f"{self.autotune!r}"
            )
        for name, value in (
            ("refresh_autotune", self.refresh_autotune),
            ("assume_valid_labels", self.assume_valid_labels),
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError(
                    f"KernelConfig.{name} must be bool or None, got {value!r}"
                )
        if self.grad_weight_accumulation not in (None, "auto", "compute", "fp32"):
            raise ValueError(
                "KernelConfig.grad_weight_accumulation must be 'auto', 'compute', "
                f"'fp32', or None, got {self.grad_weight_accumulation!r}"
            )
        if self.entropy_gate is not None and not isinstance(
            self.entropy_gate, EntropyGate
        ):
            raise TypeError(
                "KernelConfig.entropy_gate must be an EntropyGate or None, got "
                f"{self.entropy_gate!r}"
            )

    @property
    def effective_assume_valid_labels(self) -> bool:
        return bool(self.assume_valid_labels)

    @property
    def effective_grad_weight_accumulation(self) -> GradWeightAccumulation:
        return self.grad_weight_accumulation or "auto"


# ── Result container ─────────────────────────────────────────────────────────
class DistillationLossOutput(NamedTuple):
    """Components of the distillation objective.

    ``loss`` is the single scalar that is actually optimized: the weighted sum
    ``student_ce_weight * student_ce + teacher_ce_weight * teacher_ce +
    kl_weight * kl``.

    ``student_ce``, ``teacher_ce`` and ``kl`` are unweighted diagnostic
    components with one important caveat: a component reads exactly ``0`` when
    its corresponding weight is ``0``. Zeroing a weight removes that term from
    the graph, so its raw value is never computed and is reported as ``0``
    rather than as the (unused) cross-entropy / KL it would have had. This does
    not affect ``loss`` — a zero-weighted term contributes nothing regardless.
    To inspect a component's true value, give its weight a positive value.
    """

    loss: torch.Tensor
    student_ce: torch.Tensor
    teacher_ce: torch.Tensor
    kl: torch.Tensor
