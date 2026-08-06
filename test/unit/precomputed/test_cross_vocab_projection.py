"""Cross-vocabulary token mapping and projected-logit contracts."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import orda_kernel as ock
from orda_kernel import build_exact_token_map, project_teacher_logits


# ── Token-map projection ─────────────────────────────────────────────────────
def test_exact_token_map_marks_missing_and_reuses_duplicate_targets():
    token_map = build_exact_token_map(["a", "b", "c"], ["b", "zz", "a", "a"])

    assert token_map.tolist() == [1, -1, 0, 0]
    assert token_map.dtype == torch.int64
    with pytest.raises(TypeError, match="str"):
        build_exact_token_map(["a", 1], ["a"])


def test_projection_conventions_preserve_duplicate_mass():
    logits = torch.tensor([[0.0, math.log(3.0), 0.0, 0.0]])
    token_map = torch.tensor([1, -1, 0, 0])

    renormalized = project_teacher_logits(logits, token_map, 3)
    masked = project_teacher_logits(logits, token_map, 3, unmatched="mask")

    assert torch.allclose(renormalized.exp().sum(-1), torch.ones(1))
    assert renormalized.exp()[0, 0] > renormalized.exp()[0, 1]
    assert float(renormalized.exp()[0, 2]) == 0.0
    assert float(masked.exp().sum(-1)) == pytest.approx(0.5, rel=1e-6)


def test_projected_logits_are_detached_and_feed_precomputed_teacher():
    generator = torch.Generator().manual_seed(11)
    student_hidden = torch.randn(4, 3, generator=generator, requires_grad=True)
    weight = torch.randn(3, 3, generator=generator, requires_grad=True)
    labels = torch.randint(0, 3, (4,), generator=generator)
    teacher_logits = torch.randn(4, 4, generator=generator, requires_grad=True)
    token_map = build_exact_token_map(["a", "b", "c"], ["b", "zz", "a", "c"])

    projected = project_teacher_logits(teacher_logits, token_map, 3)
    assert not projected.requires_grad

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=projected),
        kl_weight=0.5,
        backend="torch",
    )
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert student_hidden.grad is not None
    assert weight.grad is not None
    assert teacher_logits.grad is None


# ── Many-to-one temperature ──────────────────────────────────────────────────
def test_projection_temperature_applies_before_aggregation():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    token_map = torch.tensor([0, 0, 1])
    temperature = 2.0

    tempered = project_teacher_logits(logits, token_map, 2, temperature=temperature)

    assert torch.equal(
        tempered,
        project_teacher_logits(logits / temperature, token_map, 2),
    )
    aggregated_then_tempered = F.log_softmax(
        project_teacher_logits(logits, token_map, 2) / temperature,
        dim=-1,
    )
    assert not torch.allclose(tempered, aggregated_then_tempered)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan, True, "2"])
def test_projection_rejects_invalid_temperature(temperature):
    with pytest.raises((TypeError, ValueError), match="temperature"):
        project_teacher_logits(
            torch.tensor([[0.0, -1.0]]),
            torch.tensor([0, 1]),
            2,
            temperature=temperature,
        )


def test_tempered_cache_pins_backend_teacher_temperature():
    generator = torch.Generator().manual_seed(23)
    student_hidden = torch.randn(4, 3, generator=generator)
    weight = torch.randn(3, 3, generator=generator)
    labels = torch.randint(0, 3, (4,), generator=generator)
    teacher_logits = torch.randn(4, 3, generator=generator)
    identity_map = torch.tensor([0, 1, 2])
    temperature = 2.0

    tempered = project_teacher_logits(
        teacher_logits, identity_map, 3, temperature=temperature
    )
    out_tempered = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=tempered, logits_temperature=temperature),
        kl_weight=0.7,
        kl_temperature=1.3,
        kl_teacher_temperature=temperature,
        backend="torch",
    )
    out_raw = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        kl_weight=0.7,
        kl_temperature=1.3,
        kl_teacher_temperature=temperature,
        backend="torch",
    )

    torch.testing.assert_close(out_tempered.loss, out_raw.loss)
    torch.testing.assert_close(out_tempered.kl, out_raw.kl)


def test_tempered_cache_rejects_explicit_mismatched_teacher_temperature():
    generator = torch.Generator().manual_seed(29)
    student_hidden = torch.randn(4, 3, generator=generator)
    weight = torch.randn(3, 3, generator=generator)
    labels = torch.randint(0, 3, (4,), generator=generator)
    projected = project_teacher_logits(
        torch.randn(4, 3, generator=generator),
        torch.tensor([0, 1, 2]),
        3,
        temperature=2.0,
    )
    teacher = ock.PrecomputedTeacher(logits=projected, logits_temperature=2.0)

    with pytest.raises(ValueError, match="does not match the call's teacher"):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            teacher,
            kl_weight=0.7,
            kl_temperature=1.7,
            kl_teacher_temperature=3.0,
            backend="torch",
        )


def test_tempered_cache_adopts_temperature_when_teacher_temp_defaulted():
    generator = torch.Generator().manual_seed(31)
    student_hidden = torch.randn(4, 3, generator=generator)
    weight = torch.randn(3, 3, generator=generator)
    labels = torch.randint(0, 3, (4,), generator=generator)
    teacher_logits = torch.randn(4, 3, generator=generator)
    identity_map = torch.tensor([0, 1, 2])
    temperature = 2.0
    tempered = project_teacher_logits(
        teacher_logits, identity_map, 3, temperature=temperature
    )

    out_adopted = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=tempered, logits_temperature=temperature),
        kl_weight=0.7,
        kl_temperature=1.3,
        backend="torch",
    )
    out_explicit = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=tempered, logits_temperature=temperature),
        kl_weight=0.7,
        kl_temperature=1.3,
        kl_teacher_temperature=temperature,
        backend="torch",
    )

    torch.testing.assert_close(out_adopted.loss, out_explicit.loss)
    torch.testing.assert_close(out_adopted.kl, out_explicit.kl)


# ── Projection validation ────────────────────────────────────────────────────
def test_projection_rejects_invalid_maps_logits_and_mode():
    logits = torch.randn(2, 3)

    with pytest.raises(ValueError, match="token_map must have 3 entries"):
        project_teacher_logits(logits, torch.tensor([0, 1]), 4)
    with pytest.raises(ValueError, match="matches no student token"):
        project_teacher_logits(logits, torch.tensor([-1, -1, -1]), 4)
    with pytest.raises(ValueError, match="student index"):
        project_teacher_logits(logits, torch.tensor([0, 1, 9]), 4)
    with pytest.raises(ValueError, match="sentinel"):
        project_teacher_logits(logits, torch.tensor([0, -2, 1]), 2)
    with pytest.raises(ValueError, match="unmatched"):
        project_teacher_logits(logits, torch.tensor([0, 1, 2]), 4, unmatched="drop")

    token_map = torch.tensor([0, 1, -1])
    with pytest.raises(ValueError, match="NaN"):
        project_teacher_logits(torch.tensor([[math.nan, 1.0, 2.0]]), token_map, 2)
    with pytest.raises(ValueError, match=r"\+inf"):
        project_teacher_logits(torch.tensor([[0.0, math.inf, 2.0]]), token_map, 2)
    with pytest.raises(ValueError, match="finite logit"):
        project_teacher_logits(
            torch.tensor([[-math.inf, -math.inf, 5.0]]),
            token_map,
            2,
        )


@pytest.mark.parametrize("dtype", (torch.int64, torch.complex64))
def test_projection_rejects_non_floating_logits(dtype):
    with pytest.raises(ValueError, match="teacher_logits must be floating"):
        project_teacher_logits(
            torch.tensor([[1, 2]], dtype=dtype),
            torch.tensor([0, 1]),
            2,
        )


@pytest.mark.parametrize("student_vocab_size", [2.9, "2", True, 0, -1])
def test_projection_rejects_invalid_student_vocab_size(student_vocab_size):
    with pytest.raises((TypeError, ValueError), match="student_vocab_size"):
        project_teacher_logits(
            torch.tensor([[0.0, -1.0]]),
            torch.tensor([0, 1]),
            student_vocab_size,
        )


def test_projection_accepts_integral_student_vocab_size():
    logits = torch.tensor([[0.0, -1.0]])
    token_map = torch.tensor([0, 1])

    assert project_teacher_logits(logits, token_map, 2).shape == (1, 2)
    numpy = pytest.importorskip("numpy")
    assert project_teacher_logits(logits, token_map, numpy.int64(2)).shape == (1, 2)


# ── Underflow stability ──────────────────────────────────────────────────────
@pytest.mark.parametrize("unmatched", ["renormalize", "mask"])
@pytest.mark.parametrize("dominant_unmatched", [88.0, 120.0, 1000.0])
def test_projection_preserves_conditional_shape_after_underflow(
    unmatched,
    dominant_unmatched,
):
    logits = torch.tensor([[dominant_unmatched, 0.0, -1.0]])
    token_map = torch.tensor([-1, 0, 1])

    projected = project_teacher_logits(logits, token_map, 2, unmatched=unmatched)
    downstream = torch.softmax(projected, dim=-1)
    expected = torch.softmax(torch.tensor([[0.0, -1.0]]), dim=-1)

    assert torch.isfinite(projected).all()
    assert torch.allclose(downstream, expected, atol=1e-6)


def test_projection_preserves_denormal_matched_ratios():
    projected = project_teacher_logits(
        torch.tensor([[0.0, -89.0, -90.0]]),
        torch.tensor([0, 1, 2]),
        3,
    )
    probs = projected.exp()

    assert torch.isfinite(projected).all()
    assert float(probs[0, 1] / probs[0, 2]) == pytest.approx(torch.e, rel=1e-3)


def test_mask_projection_matches_probability_space_without_underflow():
    generator = torch.Generator().manual_seed(29)
    logits = torch.randn(6, 8, generator=generator)
    token_map = torch.tensor([0, 1, -1, 2, 0, 1, 3, -1])

    projected = project_teacher_logits(logits, token_map, 4, unmatched="mask")
    matched = token_map >= 0
    expected_probs = torch.zeros(6, 4)
    expected_probs.index_add_(1, token_map[matched], F.softmax(logits, -1)[:, matched])
    positive = expected_probs > 0
    expected = torch.where(
        positive,
        torch.where(positive, expected_probs, torch.ones_like(expected_probs)).log(),
        torch.full_like(expected_probs, float("-inf")),
    )

    assert torch.allclose(projected, expected, atol=1e-6)


# ── Exact zero support ───────────────────────────────────────────────────────
def test_projection_preserves_zero_support_through_fp16_cast():
    projected = project_teacher_logits(
        torch.tensor([[2.0, -math.inf, 1.0]]),
        torch.tensor([0, 1, 0]),
        3,
        unmatched="mask",
    )
    half = projected.half()

    assert torch.isneginf(half[0, 1])
    assert torch.isneginf(half[0, 2])
    expected_live = torch.logsumexp(
        F.log_softmax(torch.tensor([2.0, -math.inf, 1.0]), dim=-1)[[0, 2]],
        dim=0,
    )
    assert float(projected[0, 0]) == pytest.approx(float(expected_live), abs=1e-6)
