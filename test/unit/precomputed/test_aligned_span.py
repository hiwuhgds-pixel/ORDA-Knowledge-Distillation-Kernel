"""Cross-vocabulary span-aligned distillation and token-map tests (CPU)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from utils.aligned import binary_span_rows, span_log_likelihoods

import orda_kernel as ock
from orda_kernel import (
    AlignedLikelihoodConfig,
    AlignedSpanPairs,
    CrossVocabTeacher,
    PrecomputedSpanTeacher,
    aligned_span_distillation_loss,
    build_precomputed_span_teacher,
)


# ── Test fixtures ────────────────────────────────────────────────────────────
def _make_case(seed=0, bt_s=8, bt_t=10, hs=5, ht=6, vs=11, vt=17):
    generator = torch.Generator().manual_seed(seed)
    student_hidden = torch.randn(
        bt_s, hs, generator=generator, dtype=torch.float64, requires_grad=True
    )
    student_weight = torch.randn(
        vs, hs, generator=generator, dtype=torch.float64, requires_grad=True
    )
    student_labels = torch.randint(0, vs, (bt_s,), generator=generator)
    teacher_hidden = torch.randn(bt_t, ht, generator=generator, dtype=torch.float64)
    teacher_weight = torch.randn(vt, ht, generator=generator, dtype=torch.float64)
    teacher_labels = torch.randint(0, vt, (bt_t,), generator=generator)
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 3], [3, bt_s]], dtype=torch.int32),
        torch.tensor([[0, 4], [4, bt_t]], dtype=torch.int32),
    )
    teacher = CrossVocabTeacher(teacher_hidden, teacher_weight, teacher_labels)
    return student_hidden, student_weight, student_labels, teacher, pairs


def test_explicit_triton_rejects_live_cross_vocab_teacher():
    hidden, weight, labels, teacher, pairs = _make_case()

    with pytest.raises(ValueError, match="requires PrecomputedSpanTeacher"):
        aligned_span_distillation_loss(
            hidden,
            weight,
            labels,
            teacher,
            pairs,
            backend="triton",
        )


def test_explicit_triton_rejects_cpu_precomputed_span_teacher():
    hidden, weight, labels, _, pairs = _make_case()
    teacher = PrecomputedSpanTeacher(
        torch.tensor([-0.8, -0.5]), teacher_ranges=pairs.teacher_ranges
    )

    with pytest.raises(RuntimeError, match="requires Triton and CUDA/HIP"):
        aligned_span_distillation_loss(
            hidden,
            weight,
            labels,
            teacher,
            pairs,
            backend="triton",
        )


# ── Reference parity ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("objective", ["binary_kl", "binary_ce"])
def test_aligned_span_matches_manual_oracle(reduction, objective):
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case()
    config = AlignedLikelihoodConfig(
        student_ce_weight=1.0,
        span_kl_weight=0.7,
        reduction=reduction,
        objective=objective,
    )
    out = aligned_span_distillation_loss(
        student_hidden, student_weight, student_labels, teacher, pairs, config
    )
    out.loss.backward()
    assert student_hidden.grad is not None
    assert student_weight.grad is not None

    hs_ref = student_hidden.detach().clone().requires_grad_(True)
    w_ref = student_weight.detach().clone().requires_grad_(True)
    ell_s = span_log_likelihoods(hs_ref, w_ref, student_labels, pairs.student_ranges)
    ell_t = span_log_likelihoods(
        teacher.hidden, teacher.weight, teacher.labels, pairs.teacher_ranges
    ).detach()
    p_t = ell_t.exp()
    if objective == "binary_kl":
        span_all = binary_span_rows(ell_s, ell_t, objective)
    else:
        span_all = -(p_t * ell_s + (1 - p_t) * torch.log1p(-ell_s.exp()))
    num_spans = pairs.student_ranges.shape[0]
    span = span_all.sum() / num_spans if reduction == "mean" else span_all.sum()
    ce_all = F.cross_entropy(hs_ref @ w_ref.t(), student_labels, reduction="none")
    denom = student_labels.numel()
    ce = ce_all.sum() / denom if reduction == "mean" else ce_all.sum()
    expected = 1.0 * ce + 0.7 * span
    expected.backward()

    assert torch.allclose(out.loss, expected)
    assert torch.allclose(out.kl, span.detach())
    assert torch.allclose(out.student_ce, ce.detach())
    assert torch.allclose(student_hidden.grad, hs_ref.grad, atol=1e-10, rtol=1e-8)
    assert torch.allclose(student_weight.grad, w_ref.grad, atol=1e-10, rtol=1e-8)


def test_identical_span_likelihoods_give_zero_divergence():
    student_hidden, student_weight, student_labels, _, _ = _make_case()
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 8]], dtype=torch.int32),
        torch.tensor([[0, 8]], dtype=torch.int32),
    )
    with torch.no_grad():
        ell = span_log_likelihoods(
            student_hidden, student_weight, student_labels, pairs.student_ranges
        )
    out = aligned_span_distillation_loss(
        student_hidden,
        student_weight,
        student_labels,
        PrecomputedSpanTeacher(ell, teacher_ranges=pairs.teacher_ranges),
        pairs,
        AlignedLikelihoodConfig(student_ce_weight=0.0),
    )
    assert abs(float(out.kl)) < 1e-12
    assert torch.isfinite(out.loss)


def test_binary_kl_non_negative_and_teacher_detached():
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case(seed=3)
    teacher_hidden = teacher.hidden.clone().requires_grad_(True)
    live_teacher = CrossVocabTeacher(teacher_hidden, teacher.weight, teacher.labels)
    out = aligned_span_distillation_loss(
        student_hidden,
        student_weight,
        student_labels,
        live_teacher,
        pairs,
        AlignedLikelihoodConfig(student_ce_weight=0.0),
    )
    assert float(out.kl) >= 0.0
    out.loss.backward()
    assert teacher_hidden.grad is None


def test_aligned_span_gradcheck():
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case(seed=5)
    with torch.no_grad():
        ell_t = span_log_likelihoods(
            teacher.hidden, teacher.weight, teacher.labels, pairs.teacher_ranges
        )
    span_teacher = PrecomputedSpanTeacher(ell_t, teacher_ranges=pairs.teacher_ranges)

    def objective(hidden, weight):
        return aligned_span_distillation_loss(
            hidden,
            weight,
            student_labels,
            span_teacher,
            pairs,
            AlignedLikelihoodConfig(span_kl_weight=0.5),
        ).loss

    assert torch.autograd.gradcheck(
        objective,
        (
            student_hidden.detach().clone().requires_grad_(True),
            student_weight.detach().clone().requires_grad_(True),
        ),
        atol=1e-8,
    )


def test_ignored_rows_do_not_contribute_to_span_sums():
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case(seed=7)
    labels_ignored = student_labels.clone()
    labels_ignored[1] = -100
    out = aligned_span_distillation_loss(
        student_hidden,
        student_weight,
        labels_ignored,
        teacher,
        pairs,
        AlignedLikelihoodConfig(student_ce_weight=0.0),
    )

    with torch.no_grad():
        ell_s = span_log_likelihoods(
            student_hidden, student_weight, labels_ignored, pairs.student_ranges
        )
        ell_t = span_log_likelihoods(
            teacher.hidden, teacher.weight, teacher.labels, pairs.teacher_ranges
        )
        expected = binary_span_rows(ell_s, ell_t, "binary_kl").mean()
    assert torch.allclose(out.kl, expected)


# ── Ignored spans ────────────────────────────────────────────────────────────
def test_all_ignored_span_contributes_zero_and_does_not_skew_mean():
    # span0 rows [0,2) are all ignored (no student evidence); span1 is valid.
    hidden = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([-100, -100, 2, 3])
    both = AlignedSpanPairs(
        torch.tensor([[0, 2], [2, 4]], dtype=torch.int32),
        torch.tensor([[0, 2], [2, 4]], dtype=torch.int32),
    )
    only_valid = AlignedSpanPairs(
        torch.tensor([[2, 4]], dtype=torch.int32),
        torch.tensor([[2, 4]], dtype=torch.int32),
    )
    t_hidden = torch.randn(4, 3, dtype=torch.float64)
    t_weight = torch.randn(6, 3, dtype=torch.float64)
    t_labels = torch.tensor([0, 1, 2, 3])
    config = AlignedLikelihoodConfig(student_ce_weight=0.0, span_kl_weight=1.0)
    out_both = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        CrossVocabTeacher(t_hidden, t_weight, t_labels),
        both,
        config,
    )
    out_valid = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        CrossVocabTeacher(t_hidden, t_weight, t_labels),
        only_valid,
        config,
    )
    # The all-ignored span is dropped from the sum AND the mean denominator, so
    # the mean over {span0, span1} equals the mean over {span1} alone.
    assert torch.allclose(out_both.kl, out_valid.kl)


def test_all_ignored_span_emits_exactly_zero_gradients():
    hidden = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([-100, -100])  # the only span is fully ignored
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 2]], dtype=torch.int32),
        torch.tensor([[0, 2]], dtype=torch.int32),
    )
    t_hidden = torch.randn(2, 3, dtype=torch.float64)
    t_weight = torch.randn(6, 3, dtype=torch.float64)
    teacher = CrossVocabTeacher(t_hidden, t_weight, torch.tensor([0, 1]))
    out = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        teacher,
        pairs,
        AlignedLikelihoodConfig(student_ce_weight=0.0, span_kl_weight=1.0),
    )
    assert float(out.kl) == 0.0
    out.loss.backward()
    assert float(hidden.grad.abs().sum()) == 0.0
    assert float(weight.grad.abs().sum()) == 0.0


def test_all_ignored_teacher_span_is_dropped():
    # Student span is valid but the CrossVocabTeacher span is all-ignored, so
    # the teacher likelihood is evidence-free and the span must be dropped.
    hidden = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 4]], dtype=torch.int32),
        torch.tensor([[0, 4]], dtype=torch.int32),
    )
    t_hidden = torch.randn(4, 3, dtype=torch.float64)
    t_weight = torch.randn(6, 3, dtype=torch.float64)
    teacher = CrossVocabTeacher(
        t_hidden, t_weight, torch.tensor([-100, -100, -100, -100])
    )
    out = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        teacher,
        pairs,
        AlignedLikelihoodConfig(student_ce_weight=0.0, span_kl_weight=1.0),
    )
    assert float(out.kl) == 0.0
    out.loss.backward()
    assert float(hidden.grad.abs().sum()) == 0.0


def test_precomputed_teacher_span_gated_by_student_side_only():
    # A precomputed teacher supplies each span's log-prob directly, so only the
    # student side gates the span: an all-ignored student span is dropped even
    # though the teacher supplies a finite value for it.
    hidden = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([-100, -100, 2, 3])
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 2], [2, 4]], dtype=torch.int32),
        torch.tensor([[0, 2], [2, 4]], dtype=torch.int32),
    )
    teacher = PrecomputedSpanTeacher(
        torch.tensor([-0.4, -0.6], dtype=torch.float64),
        teacher_ranges=pairs.teacher_ranges,
    )
    out = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        teacher,
        pairs,
        AlignedLikelihoodConfig(student_ce_weight=0.0, span_kl_weight=1.0),
    )
    only_valid = AlignedSpanPairs(
        torch.tensor([[2, 4]], dtype=torch.int32),
        torch.tensor([[2, 4]], dtype=torch.int32),
    )
    out_valid = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        PrecomputedSpanTeacher(
            torch.tensor([-0.6], dtype=torch.float64),
            teacher_ranges=only_valid.teacher_ranges,
        ),
        only_valid,
        AlignedLikelihoodConfig(student_ce_weight=0.0, span_kl_weight=1.0),
    )
    assert torch.allclose(out.kl, out_valid.kl)


# ── Input validation ─────────────────────────────────────────────────────────
def test_span_descriptor_validation():
    good = torch.tensor([[0, 2]], dtype=torch.int32)
    with pytest.raises(ValueError, match=r"\(C, 2\)"):
        AlignedSpanPairs(torch.zeros(2, 3, dtype=torch.int32), good)
    with pytest.raises(ValueError, match="int32 or int64"):
        AlignedSpanPairs(torch.zeros(1, 2), good)
    with pytest.raises(ValueError, match="one teacher range"):
        AlignedSpanPairs(good, torch.tensor([[0, 2], [2, 3]], dtype=torch.int32))
    with pytest.raises(ValueError, match="must be 1D"):
        PrecomputedSpanTeacher(torch.zeros(1, 2), trust_teacher_ranges=True)
    with pytest.raises(ValueError, match="must not require gradients"):
        PrecomputedSpanTeacher(
            torch.zeros(2, requires_grad=True), trust_teacher_ranges=True
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        CrossVocabTeacher(torch.zeros(2, 3), torch.zeros(4, 3), labels=None)


def test_entry_point_validation():
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case()
    out_of_range = AlignedSpanPairs(
        torch.tensor([[0, 99]], dtype=torch.int32),
        torch.tensor([[0, 4]], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="student_ranges"):
        aligned_span_distillation_loss(
            student_hidden, student_weight, student_labels, teacher, out_of_range
        )
    empty_span = AlignedSpanPairs(
        torch.tensor([[3, 3]], dtype=torch.int32),
        torch.tensor([[0, 4]], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="start < end"):
        aligned_span_distillation_loss(
            student_hidden, student_weight, student_labels, teacher, empty_span
        )
    with pytest.raises(ValueError, match="span_log_probs"):
        aligned_span_distillation_loss(
            student_hidden,
            student_weight,
            student_labels,
            PrecomputedSpanTeacher(torch.zeros(5), trust_teacher_ranges=True),
            pairs,
        )
    with pytest.raises(TypeError, match="AlignedLikelihoodConfig"):
        aligned_span_distillation_loss(
            student_hidden,
            student_weight,
            student_labels,
            teacher,
            pairs,
            config={"span_kl_weight": 1.0},
        )
    with pytest.raises(ValueError, match="objective"):
        AlignedLikelihoodConfig(objective="tokenwise")

    hidden = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    labels = torch.tensor([99, 99])
    ranges = torch.tensor([[0, 2]], dtype=torch.int32)
    pairs = AlignedSpanPairs(ranges, ranges)
    config = AlignedLikelihoodConfig(student_ce_weight=0.0, span_kl_weight=0.0)
    teacher = CrossVocabTeacher(hidden, weight, torch.tensor([88, 88]))

    with pytest.raises(ValueError, match="student_labels"):
        aligned_span_distillation_loss(hidden, weight, labels, teacher, pairs, config)

    out = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        teacher,
        pairs,
        config,
        kernel_config=ock.KernelConfig(assume_valid_labels=True),
    )
    assert float(out.loss) == 0.0

    labels = torch.tensor([-100, -100])

    with pytest.raises(ValueError, match="student_weight.*vocabulary"):
        aligned_span_distillation_loss(
            hidden,
            torch.empty(0, 3),
            labels,
            PrecomputedSpanTeacher(
                torch.tensor([-1.0]), teacher_ranges=pairs.teacher_ranges
            ),
            pairs,
        )

    teacher = CrossVocabTeacher(hidden, torch.empty(0, 3), labels)
    with pytest.raises(ValueError, match="CrossVocabTeacher.weight.*vocabulary"):
        aligned_span_distillation_loss(hidden, weight, labels, teacher, pairs)


def test_empty_batch_and_zero_weights_return_differentiable_zero():
    hidden = torch.zeros(0, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    labels = torch.zeros(0, dtype=torch.long)
    pairs = AlignedSpanPairs(
        torch.zeros(0, 2, dtype=torch.int32), torch.zeros(0, 2, dtype=torch.int32)
    )
    out = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        PrecomputedSpanTeacher(torch.zeros(0), teacher_ranges=pairs.teacher_ranges),
        pairs,
    )
    assert float(out.loss.detach()) == 0.0
    out.loss.backward()


# ── Label validation ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("invalid_label", [-1, 999])
def test_aligned_span_rejects_out_of_range_student_labels(invalid_label):
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case()
    invalid = student_labels.clone()
    invalid[0] = invalid_label
    with pytest.raises(ValueError, match="student_labels contains"):
        aligned_span_distillation_loss(
            student_hidden, student_weight, invalid, teacher, pairs
        )


@pytest.mark.parametrize("dtype", [torch.int32, torch.bool, torch.float32])
def test_aligned_span_rejects_non_long_student_labels(dtype):
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case()

    with pytest.raises(ValueError, match="student_labels must have dtype torch.long"):
        aligned_span_distillation_loss(
            student_hidden,
            student_weight,
            student_labels.to(dtype),
            teacher,
            pairs,
        )


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_aligned_span_rejects_non_long_teacher_labels(dtype):
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case()
    invalid_teacher = CrossVocabTeacher(
        teacher.hidden,
        teacher.weight,
        teacher.labels.to(dtype),
    )

    with pytest.raises(
        ValueError,
        match=r"CrossVocabTeacher.labels must have dtype torch.long",
    ):
        aligned_span_distillation_loss(
            student_hidden,
            student_weight,
            student_labels,
            invalid_teacher,
            pairs,
        )


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0, 999, 2, 999]),
        torch.tensor([999, 999, 999, 999]),
    ],
)
def test_aligned_span_positive_out_of_vocab_ignore_index_is_safe(labels):
    hidden = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    )
    teacher = PrecomputedSpanTeacher(
        torch.tensor([-0.5, -0.3], dtype=torch.float64),
        teacher_ranges=pairs.teacher_ranges,
    )
    config = AlignedLikelihoodConfig(
        student_ce_weight=1.0, span_kl_weight=1.0, ignore_index=999
    )
    out = aligned_span_distillation_loss(hidden, weight, labels, teacher, pairs, config)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert hidden.grad is not None
    assert weight.grad is not None


# ── Ignored-row safety ───────────────────────────────────────────────────────
def test_nan_in_ignored_student_row_does_not_poison_gradients():
    base = torch.randn(4, 3, dtype=torch.float64)
    base[1] = float("nan")
    hidden = base.clone().requires_grad_(True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    # Row 1 is ignored, and it sits inside the (single) span to prove the mask
    # zeroes it whether or not a span covers it.
    labels = torch.tensor([0, -100, 2, 3])
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 4]], dtype=torch.int32),
        torch.tensor([[0, 4]], dtype=torch.int32),
    )
    teacher = PrecomputedSpanTeacher(
        torch.tensor([-0.5], dtype=torch.float64),
        teacher_ranges=pairs.teacher_ranges,
    )
    out = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        teacher,
        pairs,
        AlignedLikelihoodConfig(student_ce_weight=1.0, span_kl_weight=1.0),
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_nan_in_ignored_teacher_row_keeps_loss_finite():
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case(seed=13)
    t_hidden = teacher.hidden.clone()
    t_labels = teacher.labels.clone()
    t_labels[2] = -100
    t_hidden[2] = float("nan")
    nan_teacher = CrossVocabTeacher(t_hidden, teacher.weight, t_labels)
    out = aligned_span_distillation_loss(
        student_hidden,
        student_weight,
        student_labels,
        nan_teacher,
        pairs,
        AlignedLikelihoodConfig(student_ce_weight=0.0),
    )
    assert torch.isfinite(out.loss)


@pytest.mark.parametrize("field", ["hidden", "weight"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_active_teacher_projection_is_rejected(field, value):
    student_hidden, student_weight, student_labels, teacher, pairs = _make_case()
    hidden = teacher.hidden.clone()
    weight = teacher.weight.clone()
    target = hidden if field == "hidden" else weight
    target[0, 0] = value

    with pytest.raises(ValueError, match="non-finite logits on active rows"):
        aligned_span_distillation_loss(
            student_hidden,
            student_weight,
            student_labels,
            CrossVocabTeacher(hidden, weight, teacher.labels),
            pairs,
        )


# ── Span teacher contract ────────────────────────────────────────────────────
def test_precomputed_span_teacher_neg_inf_is_legal_and_loss_stays_finite():
    hidden = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 4]], dtype=torch.int32),
        torch.tensor([[0, 4]], dtype=torch.int32),
    )
    # Teacher assigns the span probability zero: p_t = 0, and the
    # 0 * (-inf) product must resolve to the zero-mass limit, not NaN.
    teacher = PrecomputedSpanTeacher(
        torch.tensor([float("-inf")]), teacher_ranges=pairs.teacher_ranges
    )
    out = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        teacher,
        pairs,
        AlignedLikelihoodConfig(student_ce_weight=0.0, span_kl_weight=1.0),
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_span_teacher_size_validated_even_when_span_term_disabled():
    hidden = torch.randn(4, 3, dtype=torch.float64)
    weight = torch.randn(5, 3, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3])
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 4]], dtype=torch.int32),
        torch.tensor([[0, 4]], dtype=torch.int32),
    )
    bad_teacher = PrecomputedSpanTeacher(
        torch.zeros(3), trust_teacher_ranges=True
    )  # 3 entries, 1 span
    with pytest.raises(ValueError, match="must have 1 entries"):
        aligned_span_distillation_loss(
            hidden,
            weight,
            labels,
            bad_teacher,
            pairs,
            AlignedLikelihoodConfig(student_ce_weight=1.0, span_kl_weight=0.0),
        )


def test_span_teacher_requires_provenance_or_trust():
    values = torch.tensor([-0.5, -0.3], dtype=torch.float64)
    ranges = torch.tensor([[0, 2], [2, 4]], dtype=torch.int64)
    with pytest.raises(ValueError, match="trust_teacher_ranges"):
        PrecomputedSpanTeacher(values)
    with pytest.raises(ValueError, match="mutually exclusive"):
        PrecomputedSpanTeacher(values, teacher_ranges=ranges, trust_teacher_ranges=True)


def test_span_teacher_range_provenance_mismatch_rejected():
    hidden = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    teacher = PrecomputedSpanTeacher(
        torch.tensor([-0.5, -0.3], dtype=torch.float64),
        teacher_ranges=torch.tensor([[0, 2], [2, 4]], dtype=torch.int64),
    )
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 2], [2, 4]], dtype=torch.int32),
        torch.tensor([[0, 2], [1, 4]], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="does not match the ranges"):
        aligned_span_distillation_loss(
            hidden,
            weight,
            labels,
            teacher,
            pairs,
            AlignedLikelihoodConfig(student_ce_weight=0.0, span_kl_weight=1.0),
            backend="torch",
        )


def test_span_teacher_range_provenance_match_accepted():
    hidden = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    values = torch.tensor([-0.5, -0.3], dtype=torch.float64)
    pairs = AlignedSpanPairs(
        torch.tensor([[0, 2], [2, 4]], dtype=torch.int32),
        torch.tensor([[0, 2], [2, 4]], dtype=torch.int64),
    )
    config = AlignedLikelihoodConfig(student_ce_weight=0.0, span_kl_weight=1.0)
    # Descriptor holds int32 ranges while the call passes int64: still a match.
    bound = PrecomputedSpanTeacher(
        values,
        teacher_ranges=torch.tensor([[0, 2], [2, 4]], dtype=torch.int32),
    )
    trusted = PrecomputedSpanTeacher(values, trust_teacher_ranges=True)
    out_bound = aligned_span_distillation_loss(
        hidden, weight, labels, bound, pairs, config, backend="torch"
    )
    out_trusted = aligned_span_distillation_loss(
        hidden, weight, labels, trusted, pairs, config, backend="torch"
    )
    assert torch.isfinite(out_bound.loss)
    torch.testing.assert_close(out_bound.loss, out_trusted.loss)
    torch.testing.assert_close(out_bound.kl, out_trusted.kl)


def test_build_precomputed_span_teacher_binds_ranges():
    teacher_hidden = torch.randn(4, 3, dtype=torch.float64)
    teacher_weight = torch.randn(5, 3, dtype=torch.float64)
    teacher_labels = torch.tensor([0, 1, 2, 3])
    ranges = torch.tensor([[0, 2], [2, 4]], dtype=torch.int32)
    teacher = build_precomputed_span_teacher(
        teacher_hidden, teacher_weight, teacher_labels, ranges
    )
    assert teacher.teacher_ranges is not None
    assert torch.equal(teacher.teacher_ranges, ranges.to(torch.int64))
