"""Precomputed-cache value, immutability, and low-level API contracts."""

from __future__ import annotations

import pytest
import torch

import orda_kernel as ock
from orda_kernel import _api_types
from orda_kernel.ops import aligned_span_distillation_loss
from orda_kernel.ops import precomputed_teacher as low_level


def _public_inputs():
    student_hidden = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    return student_hidden, weight, labels


def _run_public(teacher: ock.PrecomputedTeacher):
    student_hidden, weight, labels = _public_inputs()
    return ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        teacher,
        student_ce_weight=0.0,
        kl_weight=0.5,
        backend="torch",
    )


# ── Descriptor value contracts ───────────────────────────────────────────────
def test_cpu_reductions_propagate_nan_for_cache_validation():
    values = torch.tensor([0.0, float("nan"), float("-inf")])

    assert torch.isnan(values.amax())
    minimum, maximum = torch.aminmax(values)
    assert torch.isnan(minimum)
    assert torch.isnan(maximum)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_precomputed_span_teacher_requires_floating_values(dtype):
    values = torch.ones(2, dtype=dtype)

    with pytest.raises(ValueError, match="must be floating"):
        ock.PrecomputedSpanTeacher(values, trust_teacher_ranges=True)


@pytest.mark.parametrize(
    "values,match",
    [
        (torch.tensor([-0.2, float("nan")]), "contains NaN"),
        (torch.tensor([-0.2, float("inf")]), r"contains \+inf"),
        (torch.tensor([-0.2, 0.01]), "values <= 0"),
    ],
)
def test_precomputed_span_teacher_rejects_invalid_log_probabilities(values, match):
    with pytest.raises(ValueError, match=match):
        ock.PrecomputedSpanTeacher(values, trust_teacher_ranges=True)


def test_precomputed_span_teacher_accepts_zero_negative_and_negative_inf():
    values = torch.tensor([0.0, -0.5, float("-inf")])

    teacher = ock.PrecomputedSpanTeacher(values, trust_teacher_ranges=True)

    assert teacher.span_log_probs is values


@pytest.mark.parametrize(
    "logits_temperature,match",
    [
        (0.0, "finite and > 0"),
        (-1.0, "finite and > 0"),
        (float("inf"), "finite and > 0"),
        (float("nan"), "finite and > 0"),
        (True, "float or"),
        ("2", "float or"),
    ],
)
def test_precomputed_teacher_rejects_invalid_logits_temperature(
    logits_temperature, match
):
    with pytest.raises((TypeError, ValueError), match=match):
        ock.PrecomputedTeacher(
            logits=torch.zeros(2, 3),
            logits_temperature=logits_temperature,
        )


def test_logits_temperature_requires_logits_representation():
    with pytest.raises(ValueError, match="requires the logits representation"):
        ock.PrecomputedTeacher(
            teacher_hidden=torch.zeros(2, 3),
            teacher_weight=torch.zeros(4, 3),
            logits_temperature=2.0,
        )


def test_tempered_cache_rejects_teacher_ce():
    student_hidden, weight, labels = _public_inputs()
    teacher = ock.PrecomputedTeacher(
        logits=torch.randn(3, 5),
        logits_temperature=2.0,
    )

    with pytest.raises(ValueError, match="teacher_ce_weight=0.0"):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            teacher,
            teacher_ce_weight=0.5,
            kl_weight=0.5,
            kl_teacher_temperature=2.0,
            backend="torch",
        )


def test_empty_and_meta_caches_skip_value_reductions():
    empty = ock.PrecomputedTeacher(logits=torch.empty(0, 5))
    meta = ock.PrecomputedTeacher(logits=torch.empty(2, 5, device="meta"))
    empty_span = ock.PrecomputedSpanTeacher(torch.empty(0), trust_teacher_ranges=True)
    meta_span = ock.PrecomputedSpanTeacher(
        torch.empty(2, device="meta"), trust_teacher_ranges=True
    )

    assert empty.logits is not None and empty.logits.numel() == 0
    assert meta.logits is not None and meta.logits.device.type == "meta"
    assert empty_span.span_log_probs.numel() == 0
    assert meta_span.span_log_probs.device.type == "meta"


@pytest.mark.parametrize("field", ["teacher_hidden", "teacher_weight"])
def test_precomputed_hidden_weight_scalar_reduction_rejects_nan(field):
    hidden = torch.randn(2, 3)
    weight = torch.randn(5, 3)
    target = hidden if field == "teacher_hidden" else weight
    target[0, 0] = float("nan")

    with pytest.raises(ValueError, match=rf"{field}.*finite"):
        ock.PrecomputedTeacher(teacher_hidden=hidden, teacher_weight=weight)


# ── Validation stamps ────────────────────────────────────────────────────────
@pytest.mark.parametrize("mutation", ["descriptor", "original", "view"])
def test_precomputed_teacher_detects_alias_mutation(mutation):
    cache = torch.randn(3, 5)
    teacher = ock.PrecomputedTeacher(logits=cache)

    if mutation == "descriptor":
        assert teacher.logits is not None
        teacher.logits.add_(0.25)
    elif mutation == "original":
        cache.add_(0.25)
    else:
        cache.view(-1)[0].add_(0.25)

    with pytest.raises(ValueError, match="modified after validation"):
        _run_public(teacher)


@pytest.mark.parametrize("replacement", [1.25, float("nan")])
def test_precomputed_teacher_detects_finite_and_invalid_mutation(replacement):
    cache = torch.randn(3, 5)
    teacher = ock.PrecomputedTeacher(logits=cache)
    cache[0, 0] = replacement

    with pytest.raises(ValueError, match="modified after validation"):
        _run_public(teacher)


def test_precomputed_teacher_detects_hidden_weight_mutation():
    hidden = torch.randn(3, 2)
    weight = torch.randn(5, 2)
    teacher = ock.PrecomputedTeacher(
        teacher_hidden=hidden,
        teacher_weight=weight,
    )
    weight.view(-1)[0].add_(1.0)

    with pytest.raises(ValueError, match="teacher_weight.*modified after validation"):
        _run_public(teacher)


@pytest.mark.parametrize("mutation", ["identity", "metadata", "representation"])
def test_precomputed_teacher_stamp_covers_tensor_metadata(mutation):
    cache = torch.randn(3, 5)
    teacher = ock.PrecomputedTeacher(logits=cache)

    if mutation == "identity":
        object.__setattr__(teacher, "logits", cache.clone())
    elif mutation == "metadata":
        cache.transpose_(0, 1)
    else:
        object.__setattr__(teacher, "logits", None)
        object.__setattr__(teacher, "teacher_hidden", torch.randn(3, 2))
        object.__setattr__(teacher, "teacher_weight", torch.randn(5, 2))

    with pytest.raises(ValueError, match="modified after validation"):
        _run_public(teacher)


def test_precomputed_span_teacher_detects_alias_mutation_at_call_time():
    student_hidden = torch.randn(2, 3)
    student_weight = torch.randn(4, 3)
    labels = torch.tensor([0, 1])
    pairs = ock.AlignedSpanPairs(
        torch.tensor([[0, 2]], dtype=torch.int64),
        torch.tensor([[0, 2]], dtype=torch.int64),
    )
    cache = torch.tensor([-0.5])
    teacher = ock.PrecomputedSpanTeacher(cache, trust_teacher_ranges=True)
    cache.add_(-0.1)

    with pytest.raises(ValueError, match="modified after validation"):
        aligned_span_distillation_loss(
            student_hidden,
            student_weight,
            labels,
            teacher,
            pairs,
        )


def test_inference_tensor_requires_explicit_immutability_trust():
    with torch.inference_mode():
        cache = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="does not expose a version counter"):
        ock.PrecomputedTeacher(logits=cache)

    teacher = ock.PrecomputedTeacher(
        logits=cache,
        trust_cache_immutability=True,
    )
    assert teacher.logits is cache


def test_public_repeated_calls_do_not_rescan_cache(monkeypatch):
    teacher = ock.PrecomputedTeacher(logits=torch.randn(3, 5))

    def fail_rescan(**_kwargs):
        raise AssertionError("cache content was scanned after construction")

    monkeypatch.setattr(_api_types, "_validate_precomputed_values", fail_rescan)

    assert torch.isfinite(_run_public(teacher).loss)
    assert torch.isfinite(_run_public(teacher).loss)


# ── Raw and descriptor paths ─────────────────────────────────────────────────
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_low_level_raw_logits_reject_invalid_values_before_triton(invalid):
    student_hidden, weight, labels = _public_inputs()
    logits = torch.randn(3, 5)
    logits[0, 0] = invalid

    with pytest.raises(ValueError, match="finite values or -inf"):
        low_level.precomputed_distillation_loss(
            student_hidden,
            weight,
            logits_teacher=logits,
            labels=labels,
            kl_weight=0.5,
        )


def test_low_level_raw_hidden_weight_rejects_invalid_values_before_triton():
    student_hidden, weight, labels = _public_inputs()
    teacher_hidden = torch.randn(3, 2)
    teacher_weight = torch.randn(5, 2)
    teacher_weight[0, 0] = float("nan")

    with pytest.raises(ValueError, match="teacher_weight.*finite"):
        low_level.precomputed_distillation_loss(
            student_hidden,
            weight,
            labels=labels,
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
            kl_weight=0.5,
        )


def test_low_level_trusted_raw_skips_content_but_keeps_dtype_checks(monkeypatch):
    student_hidden, weight, labels = _public_inputs()
    logits = torch.full((3, 5), float("nan"))
    monkeypatch.setattr(low_level, "HAS_TRITON", False)

    with pytest.raises(RuntimeError, match="requires Triton"):
        low_level.precomputed_distillation_loss(
            student_hidden,
            weight,
            logits_teacher=logits,
            labels=labels,
            kl_weight=0.5,
            assume_valid_teacher_cache=True,
        )

    with pytest.raises(ValueError, match="must be floating"):
        low_level.precomputed_distillation_loss(
            student_hidden,
            weight,
            logits_teacher=torch.ones(3, 5, dtype=torch.int64),
            labels=labels,
            kl_weight=0.5,
            assume_valid_teacher_cache=True,
        )


def test_low_level_descriptor_path_uses_stamp_without_rescan(monkeypatch):
    student_hidden, weight, labels = _public_inputs()
    teacher = ock.PrecomputedTeacher(logits=torch.randn(3, 5))
    monkeypatch.setattr(low_level, "HAS_TRITON", False)

    def fail_rescan(**_kwargs):
        raise AssertionError("descriptor cache content was rescanned")

    monkeypatch.setattr(low_level, "_validate_precomputed_values", fail_rescan)

    with pytest.raises(RuntimeError, match="requires Triton"):
        low_level.precomputed_distillation_loss(
            student_hidden,
            weight,
            labels=labels,
            teacher=teacher,
            kl_weight=0.5,
        )


def test_low_level_descriptor_rejects_raw_representation_and_mutation():
    student_hidden, weight, labels = _public_inputs()
    cache = torch.randn(3, 5)
    teacher = ock.PrecomputedTeacher(logits=cache)

    with pytest.raises(ValueError, match="mutually exclusive"):
        low_level.precomputed_distillation_loss(
            student_hidden,
            weight,
            logits_teacher=cache,
            labels=labels,
            teacher=teacher,
            kl_weight=0.5,
        )

    cache.add_(0.5)
    with pytest.raises(ValueError, match="modified after validation"):
        low_level.precomputed_distillation_loss(
            student_hidden,
            weight,
            labels=labels,
            teacher=teacher,
            kl_weight=0.5,
        )


# ── Torch dead-row semantics ─────────────────────────────────────────────────
def test_all_negative_inf_teacher_row_is_the_only_zero_support_row():
    student_hidden, weight, labels = _public_inputs()
    logits = torch.randn(3, 5)
    logits[1].fill_(float("-inf"))

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=logits),
        student_ce_weight=0.0,
        kl_weight=1.0,
        reduction="sum",
        backend="torch",
    )

    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.kl)
