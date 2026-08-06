"""Contracts for detached multi-teacher cached-logit utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import orda_kernel
from orda_kernel import (
    PrecomputedTeacher,
    distillation_loss,
)
from orda_kernel.ops.precomputed_inputs import (
    generalized_jensen_shannon,
    mix_teacher_logits,
)


# ── Distribution identities ──────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["arithmetic", "geometric"])
def test_single_teacher_preserves_its_distribution(mode: str):
    logits = torch.tensor(
        [[-1.0, 0.5, 2.0], [3.0, -2.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    mixed = mix_teacher_logits([logits], [1.0], mode=mode)  # type: ignore[arg-type]

    torch.testing.assert_close(mixed, F.log_softmax(logits.detach(), dim=-1))
    assert not mixed.requires_grad


def test_arithmetic_mixing_matches_manual_probability_mixture():
    first = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float64)
    second = torch.tensor([[-2.0, 3.0, 0.25]], dtype=torch.float64)
    weights = torch.tensor([2.0, 3.0], dtype=torch.float64)

    mixed = mix_teacher_logits([first, second], weights, mode="arithmetic")
    expected = torch.logsumexp(
        torch.stack(
            [
                weights[0].log() - weights.sum().log() + F.log_softmax(first, dim=-1),
                weights[1].log() - weights.sum().log() + F.log_softmax(second, dim=-1),
            ]
        ),
        dim=0,
    )

    torch.testing.assert_close(mixed, expected)
    torch.testing.assert_close(
        mixed.exp().sum(dim=-1), torch.ones(1, dtype=torch.float64)
    )


def test_geometric_mixing_matches_weighted_log_probabilities():
    first = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float64)
    second = torch.tensor([[-2.0, 3.0, 0.25]], dtype=torch.float64)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)

    mixed = mix_teacher_logits([first, second], weights, mode="geometric")
    expected = weights[0] / weights.sum() * F.log_softmax(first, dim=-1) + weights[
        1
    ] / weights.sum() * F.log_softmax(second, dim=-1)

    torch.testing.assert_close(mixed, expected)


def test_generalized_jensen_shannon_is_nonnegative_and_zero_for_identical_teachers():
    identical = torch.tensor(
        [[1.0, -1.5, 0.0], [-0.5, 2.0, 1.0]],
        dtype=torch.float64,
    )
    zero = generalized_jensen_shannon([identical, identical.clone()], [1.0, 3.0])
    different = generalized_jensen_shannon([identical, identical.flip(-1)], [1.0, 3.0])

    torch.testing.assert_close(zero, torch.zeros_like(zero))
    assert bool(torch.all(different >= 0))
    assert not different.requires_grad


# ── Detached precomputed path ────────────────────────────────────────────────
def test_mixed_logits_are_detached_and_feed_precomputed_torch_backend():
    torch.manual_seed(0)
    batch, hidden, vocab = 4, 3, 7
    student_hidden = torch.randn(batch, hidden, requires_grad=True)
    weight = torch.randn(vocab, hidden, requires_grad=True)
    labels = torch.tensor([0, 2, 4, 6])
    teacher_logits = [
        torch.randn(batch, vocab, requires_grad=True),
        torch.randn(batch, vocab, requires_grad=True),
    ]

    cached_logits = mix_teacher_logits(teacher_logits, [0.4, 0.6])
    teacher = PrecomputedTeacher(logits=cached_logits)
    out = distillation_loss(
        student_hidden,
        weight,
        labels,
        teacher,
        kl_weight=0.4,
        backend="torch",
    )
    out.loss.backward()

    assert not cached_logits.requires_grad
    assert torch.isfinite(out.loss)
    assert student_hidden.grad is not None
    assert weight.grad is not None
    assert all(logits.grad is None for logits in teacher_logits)


# ── Validation and zero support ──────────────────────────────────────────────
@pytest.mark.parametrize(
    ("logits_list", "weights", "match"),
    [
        ([], [1.0], "at least one teacher"),
        ([torch.randn(3)], [1.0], "2D"),
        ([torch.randn(2, 3), torch.randn(2, 4)], [1.0, 1.0], "same shape"),
        (
            [torch.randn(2, 3), torch.randn(2, 3, dtype=torch.float64)],
            [1.0, 1.0],
            "same dtype",
        ),
        (
            [torch.randn(2, 3), torch.empty(2, 3, device="meta")],
            [1.0, 1.0],
            "same device",
        ),
    ],
)
def test_mixing_validates_teacher_structure(logits_list, weights, match: str):
    with pytest.raises(ValueError, match=match):
        mix_teacher_logits(logits_list, weights)


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ([1.0], "exactly 2 entries"),
        ([1.0, 0.0], "strictly > 0"),
        ([1.0, -1.0], "strictly > 0"),
        ([1.0, float("nan")], "finite"),
        ([1.0, float("inf")], "finite"),
    ],
)
def test_mixing_validates_weights(weights, match: str):
    with pytest.raises(ValueError, match=match):
        mix_teacher_logits([torch.randn(2, 3), torch.randn(2, 3)], weights)


@pytest.mark.parametrize("weights", [[1e-300, 1.0], [1e100, 1.0]])
def test_extreme_finite_weights_follow_compute_dtype_contract(weights):
    teachers = [torch.randn(2, 3), torch.randn(2, 3)]
    assert torch.isfinite(
        mix_teacher_logits(teachers, weights, mode="arithmetic")
    ).all()
    with pytest.raises(ValueError, match="representable"):
        mix_teacher_logits(teachers, weights, mode="geometric")
    with pytest.raises(ValueError, match="representable"):
        generalized_jensen_shannon(teachers, weights)

    teachers64 = [teacher.double() for teacher in teachers]
    assert torch.isfinite(
        mix_teacher_logits(teachers64, weights, mode="geometric")
    ).all()
    assert torch.isfinite(generalized_jensen_shannon(teachers64, weights)).all()


@pytest.mark.parametrize("lost_value", [1e-300, 1e100])
def test_mixing_rejects_values_already_lost_in_float32_tensor(lost_value: float):
    weights = torch.tensor([lost_value, 1.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="finite|strictly > 0"):
        mix_teacher_logits([torch.randn(2, 3), torch.randn(2, 3)], weights)


@pytest.mark.parametrize(
    ("bad_logits", "match"),
    [
        (torch.tensor([[0.0, float("nan")]]), "NaN"),
        (torch.tensor([[0.0, float("inf")]]), "\\+inf"),
        (torch.full((1, 2), float("-inf")), "all-dead"),
    ],
)
def test_mixing_rejects_invalid_teacher_values(bad_logits, match: str):
    with pytest.raises(ValueError, match=match):
        mix_teacher_logits([bad_logits], [1.0])


def test_geometric_mixing_preserves_zero_support():
    first = torch.tensor([[0.0, float("-inf"), -2.0]])
    second = torch.tensor([[-3.0, -4.0, float("-inf")]])

    mixed = mix_teacher_logits([first, second], [1.0, 1.0], mode="geometric")

    assert bool(torch.isfinite(mixed[:, 0]).all())
    assert bool(torch.isneginf(mixed[:, 1:]).all())


def test_geometric_mixing_rejects_rows_without_common_support():
    first = torch.tensor([[0.0, float("-inf")]])
    second = torch.tensor([[float("-inf"), 0.0]])

    with pytest.raises(ValueError, match="no common teacher support"):
        mix_teacher_logits([first, second], [1.0, 1.0], mode="geometric")


def test_mixing_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        mix_teacher_logits([torch.randn(2, 3)], [1.0], mode="unknown")  # type: ignore[arg-type]


def test_multi_teacher_utilities_are_intentional_root_exports():
    expected = {"mix_teacher_logits", "generalized_jensen_shannon"}

    assert expected <= set(orda_kernel.__all__)
    assert all(callable(getattr(orda_kernel, name)) for name in expected)
