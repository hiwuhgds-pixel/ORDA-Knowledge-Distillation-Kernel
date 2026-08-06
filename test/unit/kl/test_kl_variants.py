"""Value, gradient, mathematical, and validation contracts for KL variants."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from utils.reference import reference_distillation_loss, zero_preserving_mul
from utils.variants import (
    ALL_KL_VARIANTS,
    COUPLED_KL_VARIANTS,
    DETACHED_KL_VARIANTS,
    variant_kwargs,
)

import orda_kernel as ock
from orda_kernel import api as public_api


# ── Test case inputs ─────────────────────────────────────────────────────────
def _make_inputs(dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    bt, hs_size, ht_size, vocab = 6, 5, 7, 13
    student_hidden = torch.randn(
        bt,
        hs_size,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    teacher_hidden = torch.randn(
        bt,
        ht_size,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    weight = torch.randn(
        vocab,
        hs_size,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    teacher_weight = torch.randn(
        vocab,
        ht_size,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    labels = torch.randint(0, vocab, (bt,), generator=generator)
    labels[1] = -100
    return student_hidden, teacher_hidden, weight, teacher_weight, labels


def _clone_trainable(*tensors):
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)


# ── CPU value parity matrix ──────────────────────────────────────────────────
@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
@pytest.mark.parametrize("kl_temperature", [1.0, 2.0])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("teacher_ce_weight", [0.0, 0.5])
def test_reference_value_parity(
    variant,
    kl_temperature,
    reduction,
    teacher_ce_weight,
):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs()
    extra = variant_kwargs(variant)
    kwargs = dict(
        kl_weight=0.3,
        kl_temperature=kl_temperature,
        teacher_ce_weight=teacher_ce_weight,
        kl_divergence=variant,
        reduction=reduction,
        backend="torch",
        **extra,
    )
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        **kwargs,
    )
    expected = reference_distillation_loss(
        torch,
        student_hidden,
        weight,
        labels,
        teacher_mode="separate",
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        kl_divergence=variant,
        kl_weight=0.3,
        kl_temperature=kl_temperature,
        teacher_ce_weight=teacher_ce_weight,
        reduction=reduction,
        **extra,
    )

    assert torch.allclose(out.loss, expected[0])
    assert torch.allclose(out.student_ce, expected[1])
    assert torch.allclose(out.teacher_ce, expected[2])
    assert torch.allclose(out.kl, expected[3])


def _make_tied_inputs(dtype=torch.float64, seed=1):
    generator = torch.Generator().manual_seed(seed)
    bt, hidden, vocab = 6, 5, 13
    student_hidden = torch.randn(
        bt, hidden, dtype=dtype, generator=generator, requires_grad=True
    )
    teacher_hidden = torch.randn(
        bt, hidden, dtype=dtype, generator=generator, requires_grad=True
    )
    weight = torch.randn(
        vocab, hidden, dtype=dtype, generator=generator, requires_grad=True
    )
    labels = torch.randint(0, vocab, (bt,), generator=generator)
    labels[1] = -100
    return student_hidden, teacher_hidden, weight, labels


@pytest.mark.parametrize("variant", DETACHED_KL_VARIANTS)
@pytest.mark.parametrize("kl_temperature", [1.0, 2.0])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_reference_value_parity_tied(variant, kl_temperature, reduction):
    student_hidden, teacher_hidden, weight, labels = _make_tied_inputs()
    extra = variant_kwargs(variant)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.TiedTeacher(teacher_hidden),
        kl_weight=0.3,
        kl_temperature=kl_temperature,
        teacher_ce_weight=0.0,
        kl_divergence=variant,
        reduction=reduction,
        backend="torch",
        **extra,
    )
    expected = reference_distillation_loss(
        torch,
        student_hidden,
        weight,
        labels,
        teacher_mode="tied",
        teacher_hidden=teacher_hidden,
        kl_divergence=variant,
        kl_weight=0.3,
        kl_temperature=kl_temperature,
        teacher_ce_weight=0.0,
        reduction=reduction,
        **extra,
    )
    assert torch.allclose(out.loss, expected[0])
    assert torch.allclose(out.kl, expected[3])


@pytest.mark.parametrize("variant", DETACHED_KL_VARIANTS)
@pytest.mark.parametrize("kl_temperature", [1.0, 2.0])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_reference_value_parity_precomputed(variant, kl_temperature, reduction):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs()
    extra = variant_kwargs(variant)
    teacher_logits = (teacher_hidden @ teacher_weight.t()).detach()
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        kl_weight=0.3,
        kl_temperature=kl_temperature,
        teacher_ce_weight=0.0,
        kl_divergence=variant,
        reduction=reduction,
        backend="torch",
        **extra,
    )
    expected = reference_distillation_loss(
        torch,
        student_hidden,
        weight,
        labels,
        teacher_mode="precomputed",
        teacher_logits=teacher_logits,
        kl_divergence=variant,
        kl_weight=0.3,
        kl_temperature=kl_temperature,
        teacher_ce_weight=0.0,
        reduction=reduction,
        **extra,
    )
    assert torch.allclose(out.loss, expected[0])
    assert torch.allclose(out.kl, expected[3])


@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_reference_respects_custom_ignore_index(variant, reduction):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs()
    labels = labels.clone()
    labels[1] = -7
    extra = variant_kwargs(variant)

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_weight=0.3,
        kl_temperature=2.0,
        teacher_ce_weight=0.0,
        kl_divergence=variant,
        ignore_index=-7,
        reduction=reduction,
        backend="torch",
        **extra,
    )
    expected = reference_distillation_loss(
        torch,
        student_hidden,
        weight,
        labels,
        teacher_mode="separate",
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        kl_divergence=variant,
        kl_weight=0.3,
        kl_temperature=2.0,
        teacher_ce_weight=0.0,
        reduction=reduction,
        ignore_index=-7,
        **extra,
    )

    assert torch.allclose(out.loss, expected[0])
    assert torch.allclose(out.student_ce, expected[1])
    assert torch.allclose(out.teacher_ce, expected[2])
    assert torch.allclose(out.kl, expected[3])


# ── CPU gradient routing matrix ──────────────────────────────────────────────
@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
@pytest.mark.parametrize("teacher_ce_weight", [0.0, 0.5])
def test_reference_grad_routing(variant, teacher_ce_weight):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs()
    extra = variant_kwargs(variant)
    kwargs = dict(
        kl_weight=0.3,
        kl_temperature=2.0,
        teacher_ce_weight=teacher_ce_weight,
        kl_divergence=variant,
        backend="torch",
        **extra,
    )
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        **kwargs,
    )
    out.loss.backward()

    expected = reference_distillation_loss(
        torch,
        student_hidden,
        weight,
        labels,
        teacher_mode="separate",
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        kl_divergence=variant,
        kl_weight=0.3,
        kl_temperature=2.0,
        teacher_ce_weight=teacher_ce_weight,
        **extra,
    )
    expected[0].backward()
    hs_ref, ht_ref, w_ref, tw_ref = expected[4:]

    assert student_hidden.grad is not None
    assert weight.grad is not None
    assert student_hidden.grad.abs().max() > 0
    assert weight.grad.abs().max() > 0
    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()
    assert torch.allclose(student_hidden.grad, hs_ref.grad)
    assert torch.allclose(weight.grad, w_ref.grad)

    teacher_should_have_grad = (
        teacher_ce_weight != 0.0 or variant in COUPLED_KL_VARIANTS
    )
    assert (teacher_hidden.grad is not None) == teacher_should_have_grad
    assert (teacher_weight.grad is not None) == teacher_should_have_grad
    assert (ht_ref.grad is not None) == teacher_should_have_grad
    assert (tw_ref.grad is not None) == teacher_should_have_grad
    if teacher_should_have_grad:
        assert teacher_hidden.grad.abs().max() > 0
        assert teacher_weight.grad.abs().max() > 0
        assert torch.allclose(teacher_hidden.grad, ht_ref.grad)
        assert torch.allclose(teacher_weight.grad, tw_ref.grad)


def test_mutual_teacher_grad_present_without_teacher_ce():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs()
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_weight=0.5,
        kl_temperature=2.0,
        teacher_ce_weight=0.0,
        kl_divergence="mutual",
        backend="torch",
    )

    assert torch.equal(out.teacher_ce, torch.zeros_like(out.teacher_ce))
    out.loss.backward()
    assert teacher_hidden.grad is not None
    assert teacher_weight.grad is not None
    assert teacher_hidden.grad.abs().max() > 0
    assert teacher_weight.grad.abs().max() > 0


# ── Detached-variant gradcheck ───────────────────────────────────────────────
@pytest.mark.parametrize("variant", DETACHED_KL_VARIANTS)
def test_detached_variant_gradcheck(variant):
    generator = torch.Generator().manual_seed(7)
    bt, hidden, vocab = 4, 3, 6
    student_hidden = torch.randn(bt, hidden, dtype=torch.float64, generator=generator)
    teacher_hidden = torch.randn(bt, hidden, dtype=torch.float64, generator=generator)
    weight = torch.randn(vocab, hidden, dtype=torch.float64, generator=generator)
    teacher_weight = torch.randn(
        vocab, hidden, dtype=torch.float64, generator=generator
    )
    labels = torch.randint(0, vocab, (bt,), generator=generator)
    labels[0] = -100
    extra = variant_kwargs(variant)

    def closure(hs, w):
        out = ock.distillation_loss(
            hs,
            w,
            labels,
            ock.SeparateTeacher(teacher_hidden, teacher_weight),
            kl_weight=0.4,
            kl_temperature=1.5,
            teacher_ce_weight=0.0,
            kl_divergence=variant,
            reduction="sum",
            backend="torch",
            **extra,
        )
        return out.loss

    hs = student_hidden.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(closure, (hs, w), eps=1e-6, atol=1e-7, rtol=1e-4)


# ── Math properties ──────────────────────────────────────────────────────────
def _kl_component(variant, sh, th, w, tw, labels, *, teacher=None, **kwargs):
    extra = variant_kwargs(variant)
    teacher = teacher if teacher is not None else ock.SeparateTeacher(th, tw)
    out = ock.distillation_loss(
        sh,
        w,
        labels,
        teacher,
        kl_weight=0.3,
        teacher_ce_weight=0.0,
        kl_divergence=variant,
        backend="torch",
        **{**extra, **kwargs},
    )
    return out.kl


@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
def test_identical_distributions_zero_kl(variant):
    """D(p, p) == 0: a student that matches the teacher has zero KL component."""
    generator = torch.Generator().manual_seed(11)
    bt, hidden, vocab = 6, 5, 13
    hidden_tensor = torch.randn(bt, hidden, dtype=torch.float64, generator=generator)
    weight = torch.randn(vocab, hidden, dtype=torch.float64, generator=generator)
    labels = torch.randint(0, vocab, (bt,), generator=generator)

    sh = hidden_tensor.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    kl = _kl_component(
        variant,
        sh,
        hidden_tensor.clone(),
        w,
        weight.clone(),
        labels,
        kl_temperature=1.0,
        reduction="sum",
    )
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-9)


@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
@pytest.mark.parametrize("kl_temperature", [1.0, 2.0])
def test_kl_component_non_negative(variant, kl_temperature):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=13
    )
    kl = _kl_component(
        variant,
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_temperature=kl_temperature,
        reduction="mean",
    )
    assert kl.item() >= -1e-12


@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
@pytest.mark.parametrize("kl_temperature", [1.0, 2.0])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_all_ignored_rows_have_zero_kl(variant, kl_temperature, reduction):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=15
    )
    labels = torch.full_like(labels, -100)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=0.3,
        kl_temperature=kl_temperature,
        kl_divergence=variant,
        reduction=reduction,
        backend="torch",
        **variant_kwargs(variant),
    )

    assert torch.equal(out.kl, torch.zeros_like(out.kl))
    assert torch.equal(out.loss, torch.zeros_like(out.loss))
    out.loss.backward()
    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()


# ── Alpha divergence properties ──────────────────────────────────────────────
def test_alpha_divergence_endpoints_approximate_forward_and_reverse():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=17
    )
    common = dict(reduction="sum", kl_temperature=1.0)

    forward = _kl_component(
        "forward",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        **common,
    )
    reverse = _kl_component(
        "reverse",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        **common,
    )
    alpha_lo = _kl_component(
        "alpha_divergence",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_alpha=1e-4,
        **common,
    )
    alpha_hi = _kl_component(
        "alpha_divergence",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_alpha=1.0 - 1e-4,
        **common,
    )
    assert torch.allclose(alpha_lo, forward, rtol=5e-3, atol=5e-3)
    assert torch.allclose(alpha_hi, reverse, rtol=5e-3, atol=5e-3)


# ── Skew divergence properties ───────────────────────────────────────────────
def test_skew_forward_endpoint_approximates_forward():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=19
    )
    common = dict(reduction="sum", kl_temperature=1.0)
    forward = _kl_component(
        "forward",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        **common,
    )
    skew = _kl_component(
        "skew_forward",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_beta=1.0 - 1e-6,
        **common,
    )
    assert torch.allclose(skew, forward, rtol=5e-3, atol=5e-3)


def test_skew_js_beta_half_equals_classic_jsd_and_bounded():
    generator = torch.Generator().manual_seed(23)
    bt, hs, ht, vocab = 6, 5, 7, 13
    student_hidden = torch.randn(bt, hs, dtype=torch.float64, generator=generator)
    teacher_hidden = torch.randn(bt, ht, dtype=torch.float64, generator=generator)
    weight = torch.randn(vocab, hs, dtype=torch.float64, generator=generator)
    teacher_weight = torch.randn(vocab, ht, dtype=torch.float64, generator=generator)
    labels = torch.zeros(bt, dtype=torch.long)  # all rows active

    sh = student_hidden.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    kl_sum = _kl_component(
        "skew_js",
        sh,
        teacher_hidden.clone(),
        w,
        teacher_weight.clone(),
        labels,
        kl_beta=0.5,
        kl_temperature=1.0,
        reduction="sum",
    )

    logits_s = student_hidden @ weight.t()
    logits_t = teacher_hidden @ teacher_weight.t()
    p = F.log_softmax(logits_s, dim=-1).exp()
    q = F.log_softmax(logits_t, dim=-1).exp()
    m = 0.5 * (p + q)
    jsd_row = 0.5 * (p * (p.log() - m.log())).sum(-1) + 0.5 * (
        q * (q.log() - m.log())
    ).sum(-1)
    assert torch.allclose(kl_sum, jsd_row.sum())
    # JSD is bounded by log 2 per row.
    assert bool((jsd_row <= math.log(2) + 1e-9).all())


def test_alpha_divergence_row_bound():
    generator = torch.Generator().manual_seed(29)
    bt, hs, ht, vocab = 5, 5, 7, 13
    student_hidden = torch.randn(bt, hs, dtype=torch.float64, generator=generator)
    teacher_hidden = torch.randn(bt, ht, dtype=torch.float64, generator=generator)
    weight = torch.randn(vocab, hs, dtype=torch.float64, generator=generator)
    teacher_weight = torch.randn(vocab, ht, dtype=torch.float64, generator=generator)
    labels = torch.zeros(bt, dtype=torch.long)
    a = 0.3
    kl_sum = _kl_component(
        "alpha_divergence",
        student_hidden.clone().requires_grad_(True),
        teacher_hidden,
        weight.clone().requires_grad_(True),
        teacher_weight,
        labels,
        kl_alpha=a,
        kl_temperature=1.0,
        reduction="sum",
    )
    # A_alpha >= 0 => D <= 1/(a(1-a)) per row; T == 1.
    assert kl_sum.item() <= bt / (a * (1.0 - a)) + 1e-9


def test_skew_forward_row_bound():
    generator = torch.Generator().manual_seed(31)
    bt, hs, ht, vocab = 5, 5, 7, 13
    student_hidden = torch.randn(bt, hs, dtype=torch.float64, generator=generator)
    teacher_hidden = torch.randn(bt, ht, dtype=torch.float64, generator=generator)
    weight = torch.randn(vocab, hs, dtype=torch.float64, generator=generator)
    teacher_weight = torch.randn(vocab, ht, dtype=torch.float64, generator=generator)
    labels = torch.zeros(bt, dtype=torch.long)
    b = 0.3
    kl_sum = _kl_component(
        "skew_forward",
        student_hidden.clone().requires_grad_(True),
        teacher_hidden,
        weight.clone().requires_grad_(True),
        teacher_weight,
        labels,
        kl_beta=b,
        kl_temperature=1.0,
        reduction="sum",
    )
    # KL(q || (1-b)q + bp) <= -log(1-b) per row; T == 1.
    assert kl_sum.item() <= -bt * math.log(1.0 - b) + 1e-9


def test_skew_reverse_row_bound_and_reverse_limit():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=33
    )
    labels = torch.zeros_like(labels)
    logits_s = student_hidden @ weight.t()
    logits_t = teacher_hidden @ teacher_weight.t()
    log_p_s = F.log_softmax(logits_s, dim=-1)
    log_p_t = F.log_softmax(logits_t, dim=-1)
    beta = 0.3
    log_m = torch.logaddexp(
        math.log1p(-beta) + log_p_t,
        math.log(beta) + log_p_s,
    )
    rows = zero_preserving_mul(torch, log_p_s.exp(), log_p_s - log_m).sum(dim=-1)

    skew_reverse = _kl_component(
        "skew_reverse",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_beta=beta,
        kl_temperature=1.0,
        reduction="sum",
    )
    reverse = _kl_component(
        "reverse",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_temperature=1.0,
        reduction="sum",
    )
    near_reverse = _kl_component(
        "skew_reverse",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_beta=1.0e-6,
        kl_temperature=1.0,
        reduction="sum",
    )

    assert bool((rows <= -math.log(beta) + 1.0e-10).all())
    assert torch.allclose(skew_reverse, rows.sum())
    assert torch.allclose(near_reverse, reverse, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("variant", ("renyi", "renyi_reverse"))
# ── Renyi properties ─────────────────────────────────────────────────────────
def test_renyi_is_monotone_in_order(variant):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=35
    )
    orders = (0.25, 0.5, 0.75, 1.01, 1.5, 2.0)
    values = [
        _kl_component(
            variant,
            student_hidden,
            teacher_hidden,
            weight,
            teacher_weight,
            labels,
            kl_alpha=order,
            kl_temperature=1.0,
            reduction="sum",
        )
        for order in orders
    ]

    assert all(
        right.item() >= left.item() - 1.0e-10
        for left, right in zip(values[:-1], values[1:], strict=True)
    )


@pytest.mark.parametrize("variant", ("renyi", "renyi_reverse"))
def test_renyi_order_two_matches_log_one_plus_chi_square(variant):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=39
    )
    labels = torch.zeros_like(labels)
    log_p_s = F.log_softmax(student_hidden @ weight.t(), dim=-1)
    log_p_t = F.log_softmax(teacher_hidden @ teacher_weight.t(), dim=-1)
    p_s = log_p_s.exp()
    p_t = log_p_t.exp()
    if variant == "renyi":
        chi_square = (p_t - p_s).square().div(p_s).sum(dim=-1)
    else:
        chi_square = (p_s - p_t).square().div(p_t).sum(dim=-1)
    actual = _kl_component(
        variant,
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_alpha=2.0,
        kl_temperature=1.0,
        reduction="sum",
    )

    assert torch.allclose(actual, torch.log1p(chi_square).sum())


@pytest.mark.parametrize(
    ("variant", "limit_variant"),
    (("renyi", "forward"), ("renyi_reverse", "reverse")),
)
@pytest.mark.parametrize("order", (0.99, 1.01))
def test_renyi_near_one_approximates_kl(variant, limit_variant, order):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=43
    )
    actual = _kl_component(
        variant,
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_alpha=order,
        kl_temperature=1.0,
        reduction="sum",
    )
    limit = _kl_component(
        limit_variant,
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_temperature=1.0,
        reduction="sum",
    )

    assert torch.allclose(actual, limit, rtol=3e-2, atol=3e-2)


def test_renyi_pair_obeys_order_symmetry():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=45
    )
    order = 0.3
    reverse = _kl_component(
        "renyi_reverse",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_alpha=order,
        kl_temperature=1.0,
        reduction="sum",
    )
    forward = _kl_component(
        "renyi",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_alpha=1.0 - order,
        kl_temperature=1.0,
        reduction="sum",
    )

    assert torch.allclose(reverse, order / (1.0 - order) * forward)


# ── Specialized properties ───────────────────────────────────────────────────
def test_dkd_defaults_and_nontrivial_weights_match_oracle():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=47
    )
    common = dict(
        kl_weight=0.3,
        kl_temperature=1.0,
        teacher_ce_weight=0.0,
        kl_divergence="dkd",
        reduction="sum",
        backend="torch",
    )
    default = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        **common,
    )
    explicit = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_alpha=1.0,
        kl_beta=1.0,
        **common,
    )
    nontrivial = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_alpha=1.0,
        kl_beta=8.0,
        **common,
    )
    expected = reference_distillation_loss(
        torch,
        student_hidden,
        weight,
        labels,
        teacher_mode="separate",
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        kl_divergence="dkd",
        kl_weight=0.3,
        kl_temperature=1.0,
        teacher_ce_weight=0.0,
        reduction="sum",
        kl_alpha=1.0,
        kl_beta=8.0,
    )

    assert torch.allclose(default.kl, explicit.kl)
    assert torch.allclose(nontrivial.loss, expected[0])
    assert torch.allclose(nontrivial.kl, expected[3])


def test_abkd_unit_sum_matches_alpha_value_and_grad():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=49
    )
    alpha = 0.3
    beta = 1.0 - alpha

    def run(variant, **extra):
        hs, ht, w, tw = _clone_trainable(
            student_hidden,
            teacher_hidden,
            weight,
            teacher_weight,
        )
        out = ock.distillation_loss(
            hs,
            w,
            labels,
            ock.SeparateTeacher(ht, tw),
            student_ce_weight=0.0,
            teacher_ce_weight=0.0,
            kl_weight=1.0,
            kl_temperature=1.0,
            kl_divergence=variant,
            reduction="sum",
            backend="torch",
            **extra,
        )
        out.loss.backward()
        return out, hs.grad, w.grad

    alpha_out, alpha_hs_grad, alpha_w_grad = run("alpha_divergence", kl_alpha=alpha)
    abkd_out, abkd_hs_grad, abkd_w_grad = run(
        "abkd",
        kl_alpha=alpha,
        kl_beta=beta,
    )

    assert torch.allclose(abkd_out.loss, alpha_out.loss)
    assert torch.allclose(abkd_out.kl, alpha_out.kl)
    assert torch.allclose(abkd_hs_grad, alpha_hs_grad)
    assert torch.allclose(abkd_w_grad, alpha_w_grad)


@pytest.mark.parametrize(
    ("alpha", "beta"),
    ((0.05, 0.05), (0.1, 0.5), (0.7, 0.9), (1.5, 1.5), (2.0, 2.0)),
)
def test_abkd_grid_is_nonnegative(alpha, beta):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=51
    )
    kl = _kl_component(
        "abkd",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_alpha=alpha,
        kl_beta=beta,
        kl_temperature=1.0,
        reduction="sum",
    )

    assert kl.item() >= -1e-12


def test_triangular_bound_symmetry_and_tv_sandwich():
    generator = torch.Generator().manual_seed(53)
    bt, hidden, vocab = 5, 4, 9
    student_hidden = torch.randn(bt, hidden, dtype=torch.float64, generator=generator)
    teacher_hidden = torch.randn(bt, hidden, dtype=torch.float64, generator=generator)
    weight = torch.randn(vocab, hidden, dtype=torch.float64, generator=generator)
    labels = torch.zeros(bt, dtype=torch.long)
    p_s = F.log_softmax(student_hidden @ weight.t(), dim=-1).exp()
    p_t = F.log_softmax(teacher_hidden @ weight.t(), dim=-1).exp()
    delta_rows = (p_s - p_t).square().div(p_s + p_t).sum(dim=-1)
    tv = 0.5 * (p_s - p_t).abs().sum(dim=-1)

    actual = _kl_component(
        "triangular",
        student_hidden.clone().requires_grad_(True),
        teacher_hidden,
        weight.clone().requires_grad_(True),
        weight,
        labels,
        kl_temperature=1.0,
        reduction="sum",
    )
    swapped = _kl_component(
        "triangular",
        teacher_hidden.clone().requires_grad_(True),
        student_hidden,
        weight.clone().requires_grad_(True),
        weight,
        labels,
        kl_temperature=1.0,
        reduction="sum",
    )

    assert bool((delta_rows <= 2.0 + 1.0e-12).all())
    assert bool((2.0 * tv.square() <= delta_rows + 1.0e-12).all())
    assert bool((delta_rows <= 2.0 * tv + 1.0e-12).all())
    assert torch.allclose(actual, delta_rows.sum())
    assert torch.allclose(actual, swapped)


@pytest.mark.parametrize("kl_temperature", [1.0, 2.0])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
# ── Entropy properties ───────────────────────────────────────────────────────
def test_entropy_adaptive_gate_endpoints(kl_temperature, reduction):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        seed=37
    )
    forward = _kl_component(
        "forward",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_temperature=kl_temperature,
        reduction=reduction,
    )
    reverse = _kl_component(
        "reverse",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        kl_temperature=kl_temperature,
        reduction=reduction,
    )
    gate_forward = _kl_component(
        "entropy_adaptive",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        entropy_gate=ock.EntropyGate(alpha_min=1.0, alpha_max=1.0),
        kl_temperature=kl_temperature,
        reduction=reduction,
    )
    gate_reverse = _kl_component(
        "entropy_adaptive",
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        entropy_gate=ock.EntropyGate(alpha_min=0.0, alpha_max=0.0),
        kl_temperature=kl_temperature,
        reduction=reduction,
    )
    assert torch.allclose(gate_forward, forward)
    assert torch.allclose(gate_reverse, reverse)


# ── Variant validation matrix ────────────────────────────────────────────────
def test_variant_validation_matrix():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        dtype=torch.float32
    )
    tied_hidden = torch.randn_like(student_hidden)
    teacher_logits = torch.randn(labels.shape[0], weight.shape[0])

    for variant in COUPLED_KL_VARIANTS:
        tied_out = ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.TiedTeacher(tied_hidden),
            kl_weight=0.3,
            kl_divergence=variant,
            backend="torch",
        )
        assert torch.isfinite(tied_out.loss)
        with pytest.raises(ValueError, match=variant):
            ock.distillation_loss(
                student_hidden,
                weight,
                labels,
                ock.PrecomputedTeacher(logits=teacher_logits),
                kl_weight=0.3,
                kl_divergence=variant,
                backend="torch",
            )

    with pytest.raises(
        ValueError,
        match="forward.*jeffreys.*mutual|kl_divergence",
    ):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.SeparateTeacher(teacher_hidden, teacher_weight),
            kl_weight=0.3,
            kl_divergence="does_not_exist",
            backend="torch",
        )

    jeffreys_out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_weight=0.3,
        teacher_ce_weight=0.5,
        kl_divergence="jeffreys",
        backend="torch",
    )
    assert torch.isfinite(jeffreys_out.loss)
    jeffreys_out.loss.backward()
    assert teacher_hidden.grad is not None
    assert teacher_weight.grad is not None

    for teacher in (
        ock.TiedTeacher(tied_hidden),
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        ock.PrecomputedTeacher(logits=teacher_logits),
    ):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            teacher,
            kl_weight=0.3,
            kl_divergence="jeffreys",
            backend="torch",
        )

    for teacher_ce_weight in (0.0, 0.5):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.SeparateTeacher(teacher_hidden, teacher_weight),
            kl_weight=0.3,
            teacher_ce_weight=teacher_ce_weight,
            kl_divergence="cojeffreys",
            backend="torch",
        )


def test_public_hyperparameter_validation():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs(
        dtype=torch.float32
    )
    teacher = ock.SeparateTeacher(teacher_hidden, teacher_weight)

    # kl_alpha out of the open interval (0, 1) with an active KL term.
    with pytest.raises(ValueError, match="kl_alpha"):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            teacher,
            kl_weight=0.3,
            kl_divergence="alpha_divergence",
            kl_alpha=1.5,
            backend="torch",
        )

    # entropy_gate must be an EntropyGate instance.
    with pytest.raises(TypeError):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            teacher,
            kl_weight=0.3,
            kl_divergence="entropy_adaptive",
            entropy_gate=object(),
            backend="torch",
        )


def test_entropy_gate_construction_validation():
    with pytest.raises(ValueError):
        ock.EntropyGate(alpha_min=0.9, alpha_max=0.1)


@pytest.mark.parametrize(
    "scale",
    [
        0.0,
        -1.0,
        torch.finfo(torch.float32).tiny / 2.0,
        torch.finfo(torch.float32).max,
    ],
)
def test_entropy_gate_rejects_scale_or_reciprocal_outside_normal_float32(scale):
    with pytest.raises(ValueError, match="EntropyGate.scale"):
        ock.EntropyGate(scale=scale)


@pytest.mark.parametrize(
    "scale",
    [
        torch.finfo(torch.float32).tiny,
        0.1,
        1.0 / torch.finfo(torch.float32).tiny,
    ],
)
def test_entropy_gate_accepts_scale_and_reciprocal_normal_float32_boundaries(scale):
    assert ock.EntropyGate(scale=scale).scale == scale


@pytest.mark.parametrize("variant", COUPLED_KL_VARIANTS)
@pytest.mark.parametrize("kl_temperature", [1.0, 1.6])
@pytest.mark.parametrize("teacher_ce_weight", [0.0, 0.5])
def test_tied_cotrained_matches_reference_and_routes_teacher_grad(
    variant, kl_temperature, teacher_ce_weight
):
    student_hidden, teacher_hidden, weight, labels = _make_tied_inputs()
    hs, ht, w = _clone_trainable(student_hidden, teacher_hidden, weight)
    out = ock.distillation_loss(
        hs,
        w,
        labels,
        ock.TiedTeacher(ht),
        kl_weight=0.3,
        kl_temperature=kl_temperature,
        teacher_ce_weight=teacher_ce_weight,
        kl_divergence=variant,
        backend="torch",
    )
    out.loss.backward()
    # The tied teacher receives KL gradient, and the shared projection weight
    # accumulates both the student and the teacher halves.
    assert ht.grad is not None
    assert w.grad is not None

    expected = reference_distillation_loss(
        torch,
        student_hidden,
        weight,
        labels,
        teacher_mode="tied",
        teacher_hidden=teacher_hidden,
        kl_divergence=variant,
        kl_weight=0.3,
        kl_temperature=kl_temperature,
        teacher_ce_weight=teacher_ce_weight,
    )
    expected[0].backward()
    hs_ref, ht_ref, w_ref = expected[4], expected[5], expected[6]
    assert torch.allclose(out.loss, expected[0])
    assert torch.allclose(out.kl, expected[3])
    assert torch.allclose(hs.grad, hs_ref.grad, atol=1e-9, rtol=1e-7)
    assert torch.allclose(ht.grad, ht_ref.grad, atol=1e-9, rtol=1e-7)
    assert torch.allclose(w.grad, w_ref.grad, atol=1e-9, rtol=1e-7)


def test_tied_teacher_mode_declaration_validation():
    student_hidden, tied_hidden, weight, labels = _make_tied_inputs()

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.TiedTeacher(tied_hidden, mode="cotrained"),
        kl_weight=0.3,
        kl_divergence="mutual",
        backend="torch",
    )
    assert torch.isfinite(out.loss)

    # The tied default teacher_ce_weight=None resolves to 1.0, contradicting
    # a declared frozen teacher.
    with pytest.raises(ValueError, match="frozen"):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.TiedTeacher(tied_hidden, mode="frozen"),
            kl_weight=0.3,
            backend="torch",
        )
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.TiedTeacher(tied_hidden, mode="frozen"),
        teacher_ce_weight=0.0,
        kl_weight=0.3,
        backend="torch",
    )
    assert torch.isfinite(out.loss)

    with pytest.raises(ValueError, match="trained"):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.TiedTeacher(tied_hidden, mode="trained"),
            teacher_ce_weight=0.0,
            kl_weight=0.3,
            backend="torch",
        )

    with pytest.raises(ValueError, match="cotrained"):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.TiedTeacher(tied_hidden, mode="cotrained"),
            kl_weight=0.3,
            kl_divergence="forward",
            backend="torch",
        )

    with pytest.raises(ValueError, match="mode"):
        ock.TiedTeacher(tied_hidden, mode="ema")


# ── Config and degenerate contracts ──────────────────────────────────────────
def test_config_kl_divergence_overrides_arg():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs()
    config = ock.KernelConfig(kl_divergence="jeffreys")
    out_config = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_weight=0.3,
        backend="torch",
        config=config,
    )
    out_argument = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_weight=0.3,
        kl_divergence="jeffreys",
        backend="torch",
    )

    assert torch.allclose(out_config.kl, out_argument.kl)
    merged = public_api._resolve_config(
        "balanced",
        ock.KernelConfig(kl_divergence="mutual"),
    )
    assert merged.kl_divergence == "mutual"
    assert public_api._resolve_config("balanced", None).kl_divergence is None


@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
def test_kl_weight_zero_ignores_variant(variant):
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _make_inputs()
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_weight=0.0,
        kl_divergence=variant,
        kl_alpha=0.5,
        kl_beta=0.5,
        backend="torch",
    )

    hs_forward, ht_forward, w_forward, tw_forward = _clone_trainable(
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
    )
    forward = ock.distillation_loss(
        hs_forward,
        w_forward,
        labels,
        ock.SeparateTeacher(ht_forward, tw_forward),
        kl_weight=0.0,
        kl_divergence="forward",
        backend="torch",
    )

    assert torch.allclose(out.loss, forward.loss)
    assert torch.allclose(out.student_ce, forward.student_ce)
    assert torch.allclose(out.teacher_ce, forward.teacher_ce)
    assert torch.equal(out.kl, torch.zeros_like(out.kl))
    assert torch.equal(forward.kl, torch.zeros_like(forward.kl))

    out.loss.backward()
    forward.loss.backward()
    assert torch.allclose(student_hidden.grad, hs_forward.grad)
    assert torch.allclose(weight.grad, w_forward.grad)
    assert teacher_hidden.grad is None
    assert teacher_weight.grad is None
    assert ht_forward.grad is None
    assert tw_forward.grad is None


def test_ops_level_validation_rejects_bad_variant():
    from orda_kernel.ops import (
        precomputed_teacher,
        separate_teacher,
        tied_teacher,
    )

    student_hidden = torch.zeros(2, 3)
    weight = torch.zeros(4, 3)
    labels = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError, match="forward.*jeffreys.*mutual"):
        tied_teacher.tied_distillation_loss(
            student_hidden,
            student_hidden,
            weight,
            labels,
            kl_divergence="does_not_exist",
        )
    with pytest.raises(ValueError, match="jeffreys"):
        precomputed_teacher.precomputed_distillation_loss(
            student_hidden,
            torch.zeros(2, 4),
            kl_divergence="mutual",
        )
    with pytest.raises(ValueError, match="forward.*jeffreys.*mutual"):
        separate_teacher.separate_distillation_loss(
            student_hidden,
            student_hidden,
            weight,
            torch.zeros(4, 3),
            labels,
            kl_divergence="does_not_exist",
        )
