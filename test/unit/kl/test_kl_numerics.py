"""Numerical stability and support-boundary contracts for KL variants."""

from __future__ import annotations

import pytest
import torch
from utils.variants import (
    ALL_KL_VARIANTS,
    COUPLED_KL_VARIANTS,
    DETACHED_KL_VARIANTS,
    variant_kwargs,
)

import orda_kernel as ock
from orda_kernel._torch_backend import (
    _alpha_divergence_rows,
    _support_safe_renyi_rows,
    _zero_preserving_mul,
)

# ── Shared shapes ────────────────────────────────────────────────────────────
_BT, _H, _V = 4, 8, 8
_DEAD = (2, 5)  # vocabulary columns forced to zero probability on both sides


def _precomputed_masked_inputs(seed: int = 3):
    """Create shared zero-support columns for student and cached teacher."""
    generator = torch.Generator().manual_seed(seed)
    student_hidden = (
        0.1 * torch.randn(_BT, _H, generator=generator) + 0.5
    ).requires_grad_(True)
    weight = (0.2 * torch.randn(_V, _H, generator=generator)).clone()
    for column in _DEAD:
        weight[column] = -1.0e4  # hs @ w[column] ~ -5e3 -> p_s underflows to 0
    weight = weight.requires_grad_(True)
    labels = torch.randint(0, _V, (_BT,), generator=generator)
    teacher_logits = torch.randn(_BT, _V, generator=generator)
    for column in _DEAD:
        teacher_logits[:, column] = float("-inf")
    return student_hidden, weight, labels, teacher_logits


def _coupled_masked_inputs(seed: int = 5, *, teacher_requires_grad: bool = True):
    """Create shared zero-support columns through separate-teacher overflow."""
    generator = torch.Generator().manual_seed(seed)
    big = 1.0e25
    student_hidden = (
        0.1 * torch.randn(_BT, _H, generator=generator) + 0.5
    ).requires_grad_(True)
    weight = (0.3 * torch.randn(_V, _H, generator=generator)).clone()
    for column in _DEAD:
        weight[column] = -1.0e4  # student p_s underflows to exactly 0
    weight = weight.requires_grad_(True)

    teacher_hidden = torch.zeros(_BT, 2)
    teacher_hidden[:, 0] = big
    teacher_hidden[:, 1] = 1.0
    teacher_weight = torch.zeros(_V, 2)
    teacher_weight[:, 1] = torch.randn(_V, generator=generator)  # small live logits
    for column in _DEAD:
        teacher_weight[column] = torch.tensor([-big, 0.0])  # dead column -> -inf
    teacher_hidden = teacher_hidden.requires_grad_(teacher_requires_grad)
    teacher_weight = teacher_weight.requires_grad_(teacher_requires_grad)

    labels = torch.randint(0, _V, (_BT,), generator=generator)
    return student_hidden, weight, labels, teacher_hidden, teacher_weight


def _assert_all_finite(out, *grad_tensors):
    assert torch.isfinite(out.loss), f"loss is not finite: {out.loss}"
    assert torch.isfinite(out.kl), f"kl is not finite: {out.kl}"
    out.loss.backward()
    for tensor in grad_tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all(), f"non-finite gradient: {tensor.grad}"


def _run_fp32_stability_case(
    variant: str,
    dtype: torch.dtype,
    *,
    kl_alpha: float,
    kl_beta: float | None = None,
    near: bool,
):
    student_hidden = torch.eye(2, dtype=dtype, requires_grad=True)
    weight = torch.tensor(
        [[0.25, -0.30], [-0.10, 0.40], [0.50, 0.10], [-0.45, -0.20], [0.05, 0.30]],
        dtype=dtype,
        requires_grad=True,
    )
    labels = torch.tensor([0, 1])
    teacher_logits = student_hidden.detach() @ weight.detach().t()
    if near:
        teacher_logits = teacher_logits + torch.tensor(
            [
                [2.0e-4, -1.0e-4, 1.5e-4, -2.0e-4, 0.5e-4],
                [-1.5e-4, 2.0e-4, -0.5e-4, 1.0e-4, -2.0e-4],
            ],
            dtype=dtype,
        )
    extra = {"kl_alpha": kl_alpha}
    if kl_beta is not None:
        extra["kl_beta"] = kl_beta
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_divergence=variant,
        reduction="sum",
        backend="torch",
        **extra,
    )
    out.loss.backward()
    return out, student_hidden.grad, weight.grad


# ── Zero-support guards ──────────────────────────────────────────────────────
_REVERSE_SUPPORT_VARIANTS = frozenset(("reverse", "jeffreys", "entropy_adaptive"))


@pytest.mark.parametrize(
    "variant",
    tuple(v for v in DETACHED_KL_VARIANTS if v not in _REVERSE_SUPPORT_VARIANTS),
)
@pytest.mark.parametrize("kl_temperature", [1.0, 1.7])
def test_precomputed_neg_inf_logits_stay_finite(variant, kl_temperature):
    student_hidden, weight, labels, teacher_logits = _precomputed_masked_inputs()
    extra = variant_kwargs(variant)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        kl_weight=0.7,
        kl_temperature=kl_temperature,
        teacher_ce_weight=0.0,
        kl_divergence=variant,
        backend="torch",
        **extra,
    )
    _assert_all_finite(out, student_hidden, weight)


@pytest.mark.parametrize("variant", tuple(sorted(_REVERSE_SUPPORT_VARIANTS)))
@pytest.mark.parametrize("kl_temperature", [1.0, 1.7])
def test_reverse_support_uses_finite_logits_not_underflowed_mass(
    variant,
    kl_temperature,
):
    student_hidden, weight, labels, teacher_logits = _precomputed_masked_inputs()
    extra = {}
    if variant == "entropy_adaptive":
        extra["entropy_gate"] = ock.EntropyGate(alpha_min=0.5, alpha_max=0.5)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        kl_weight=0.7,
        kl_temperature=kl_temperature,
        teacher_ce_weight=0.0,
        kl_divergence=variant,
        backend="torch",
        **extra,
    )
    assert torch.isposinf(out.loss)
    assert torch.isposinf(out.kl)
    out.loss.backward()
    assert torch.equal(student_hidden.grad, torch.zeros_like(student_hidden.grad))
    assert torch.equal(weight.grad, torch.zeros_like(weight.grad))


@pytest.mark.parametrize("variant", COUPLED_KL_VARIANTS)
@pytest.mark.parametrize("kl_temperature", [1.0, 1.7])
@pytest.mark.parametrize("teacher_requires_grad", [False, True])
def test_coupled_generated_nonfinite_is_visible(
    variant, kl_temperature, teacher_requires_grad
):
    (
        student_hidden,
        weight,
        labels,
        teacher_hidden,
        teacher_weight,
    ) = _coupled_masked_inputs(teacher_requires_grad=teacher_requires_grad)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_weight=0.7,
        kl_temperature=kl_temperature,
        teacher_ce_weight=0.0,
        kl_divergence=variant,
        backend="torch",
    )
    assert torch.isnan(out.loss)
    assert torch.isnan(out.kl)
    out.loss.backward()
    assert student_hidden.grad is not None
    assert weight.grad is not None
    assert torch.isnan(student_hidden.grad).any()
    assert torch.isnan(weight.grad).any()
    if teacher_requires_grad:
        assert teacher_hidden.grad is not None
        assert teacher_weight.grad is not None
        assert torch.isnan(teacher_hidden.grad).any()
        assert torch.isnan(teacher_weight.grad).any()


# ── Helper semantics ─────────────────────────────────────────────────────────
def test_zero_preserving_mul_pins_zero_and_preserves_inf():
    p = torch.tensor([0.0, 0.3, 0.0, 0.0, 0.5])
    diff = torch.tensor([float("inf"), float("inf"), float("-inf"), float("nan"), 2.0])
    out = _zero_preserving_mul(p, diff)
    # p == 0 columns are pinned to exactly zero regardless of diff.
    assert out[0].item() == 0.0
    assert out[2].item() == 0.0
    assert out[3].item() == 0.0
    # p > 0 columns keep the ordinary product, so a true +inf divergence survives.
    assert torch.isinf(out[1]) and out[1] > 0
    assert out[4].item() == pytest.approx(1.0)


def test_zero_preserving_mul_zero_probability_gradient_is_zero():
    log_p_s = torch.tensor([-4.8e4, -0.7, -0.7], requires_grad=True)
    log_p_t = torch.tensor([float("-inf"), -0.7, -0.7])
    p_s = log_p_s.exp()  # p_s[0] underflows to exactly 0
    diff = log_p_s - log_p_t  # diff[0] == +inf
    term = _zero_preserving_mul(p_s, diff)
    assert torch.equal(term, torch.zeros_like(term))
    term.sum().backward()
    assert log_p_s.grad is not None
    assert torch.isfinite(log_p_s.grad).all()
    assert log_p_s.grad[0].item() == 0.0


def test_zero_preserving_mul_matches_plain_product_on_finite_inputs():
    generator = torch.Generator().manual_seed(41)
    p = torch.rand(6, 5, generator=generator, dtype=torch.float64).requires_grad_(True)
    diff = torch.randn(6, 5, generator=generator, dtype=torch.float64).requires_grad_(
        True
    )
    guarded = _zero_preserving_mul(p, diff)
    plain = p * diff
    assert torch.equal(guarded, plain)

    (guarded.sum() * 2.0).backward()
    guarded_p_grad = p.grad.clone()
    guarded_diff_grad = diff.grad.clone()
    p.grad = None
    diff.grad = None
    (plain.sum() * 2.0).backward()
    assert torch.equal(guarded_p_grad, p.grad)
    assert torch.equal(guarded_diff_grad, diff.grad)


@pytest.mark.parametrize("variant", ("reverse", "jeffreys", "entropy_adaptive"))
def test_old_reverse_support_mismatch_is_inf_with_zero_kl_grad(variant):
    generator = torch.Generator().manual_seed(9)
    student_hidden = (0.3 * torch.randn(_BT, _H, generator=generator)).requires_grad_(
        True
    )
    weight = (0.3 * torch.randn(_V, _H, generator=generator)).requires_grad_(True)
    labels = torch.randint(0, _V, (_BT,), generator=generator)
    teacher_logits = torch.randn(_BT, _V, generator=generator)
    teacher_logits[:, _DEAD[0]] = float("-inf")  # teacher zero; student mass stays
    extra = {}
    if variant == "entropy_adaptive":
        extra["entropy_gate"] = ock.EntropyGate(alpha_min=0.5, alpha_max=0.5)

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        kl_weight=1.0,
        kl_temperature=1.0,
        teacher_ce_weight=0.0,
        kl_divergence=variant,
        reduction="sum",
        backend="torch",
        **extra,
    )
    assert torch.isinf(out.kl) and out.kl > 0
    assert torch.isinf(out.loss) and not torch.isnan(out.loss)
    out.loss.backward()
    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()
    assert torch.equal(student_hidden.grad, torch.zeros_like(student_hidden.grad))
    assert torch.equal(weight.grad, torch.zeros_like(weight.grad))


def test_forward_kl_remains_finite_when_teacher_has_partial_zero_support():
    generator = torch.Generator().manual_seed(10)
    student_hidden = torch.randn(_BT, _H, generator=generator, requires_grad=True)
    weight = torch.randn(_V, _H, generator=generator, requires_grad=True)
    labels = torch.randint(0, _V, (_BT,), generator=generator)
    teacher_logits = torch.randn(_BT, _V, generator=generator)
    teacher_logits[:, _DEAD[0]] = float("-inf")

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_divergence="forward",
        reduction="sum",
        backend="torch",
    )

    _assert_all_finite(out, student_hidden, weight)


@pytest.mark.parametrize(
    ("variant", "order"),
    (
        ("renyi", 0.5),
        ("renyi", 0.99),
        ("renyi", 1.01),
        ("renyi", 1.5),
        ("renyi_reverse", 0.5),
        ("renyi_reverse", 0.99),
        ("renyi_reverse", 1.01),
        ("renyi_reverse", 1.5),
    ),
)
# ── Renyi numerics ───────────────────────────────────────────────────────────
def test_renyi_generated_nonfinite_student_is_visible(variant, order):
    big = 1.0e20
    student_hidden = torch.tensor([[big, 1.0]], requires_grad=True)
    weight = torch.tensor([[0.0, 0.0], [-big, 0.0]], requires_grad=True)
    labels = torch.zeros(1, dtype=torch.long)
    teacher_logits = torch.tensor([[float("-inf"), 0.0]])

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_temperature=1.0,
        kl_divergence=variant,
        kl_alpha=order,
        reduction="sum",
        backend="torch",
    )

    assert torch.isnan(out.kl)
    assert torch.isnan(out.loss)
    out.loss.backward()
    assert torch.isnan(student_hidden.grad).any()
    assert torch.isnan(weight.grad).any()


@pytest.mark.parametrize("reverse", (False, True))
def test_renyi_partial_base_support_matches_definition(reverse):
    order = 0.9
    partial = torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float64)
    full = torch.tensor([[0.25, 0.5, 0.25]], dtype=torch.float64)
    student_probs, teacher_probs = (partial, full) if reverse else (full, partial)
    logits_s = student_probs.log().requires_grad_()
    log_p_s = torch.log_softmax(logits_s, dim=-1)
    log_p_t = teacher_probs.log()

    actual = _support_safe_renyi_rows(
        logits_s,
        log_p_s,
        log_p_t,
        order,
        reverse=reverse,
    )
    log_base, log_other = (log_p_s, log_p_t) if reverse else (log_p_t, log_p_s)
    expected = torch.logsumexp(
        order * log_base + (1.0 - order) * log_other,
        dim=-1,
    ) / (order - 1.0)
    actual_grad = torch.autograd.grad(actual.sum(), logits_s, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits_s)[0]

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("order", (0.8, 0.87, 0.875, 0.88, 1.12, 1.125, 1.13, 1.5))
def test_renyi_matches_definition_across_branch_switch(reverse, order):
    generator = torch.Generator().manual_seed(7)
    logits_s = torch.randn(5, 11, generator=generator, dtype=torch.float64) * 1.5
    logits_s = logits_s.requires_grad_()
    teacher = torch.randn(5, 11, generator=generator, dtype=torch.float64) * 1.5
    log_p_s = torch.log_softmax(logits_s, dim=-1)
    log_p_t = torch.log_softmax(teacher, dim=-1)

    actual = _support_safe_renyi_rows(
        logits_s, log_p_s, log_p_t, order, reverse=reverse
    )
    log_base, log_other = (log_p_s, log_p_t) if reverse else (log_p_t, log_p_s)
    expected = torch.logsumexp(order * log_base + (1.0 - order) * log_other, dim=-1) / (
        order - 1.0
    )
    actual_grad = torch.autograd.grad(actual.sum(), logits_s, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits_s)[0]

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-11)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0.0, atol=1e-11)


@pytest.mark.parametrize("variant", ("renyi", "renyi_reverse"))
@pytest.mark.parametrize("order", (0.99, 1.01))
def test_renyi_fp32_identity_has_exact_zero_value_and_grad(variant, order):
    out, hidden_grad, weight_grad = _run_fp32_stability_case(
        variant,
        torch.float32,
        kl_alpha=order,
        near=False,
    )

    assert torch.equal(out.loss, torch.zeros_like(out.loss))
    assert torch.equal(out.kl, torch.zeros_like(out.kl))
    assert torch.equal(hidden_grad, torch.zeros_like(hidden_grad))
    assert torch.equal(weight_grad, torch.zeros_like(weight_grad))


@pytest.mark.parametrize("variant", ("renyi", "renyi_reverse"))
@pytest.mark.parametrize("order", (0.99, 1.01))
def test_renyi_fp32_near_identity_matches_fp64_value_and_grad(variant, order):
    out32, hidden_grad32, weight_grad32 = _run_fp32_stability_case(
        variant,
        torch.float32,
        kl_alpha=order,
        near=True,
    )
    out64, hidden_grad64, weight_grad64 = _run_fp32_stability_case(
        variant,
        torch.float64,
        kl_alpha=order,
        near=True,
    )

    assert out32.loss >= 0
    assert out32.kl >= 0
    torch.testing.assert_close(out32.loss.double(), out64.loss, rtol=3e-2, atol=5e-8)
    torch.testing.assert_close(out32.kl.double(), out64.kl, rtol=3e-2, atol=5e-8)
    torch.testing.assert_close(
        hidden_grad32.double(), hidden_grad64, rtol=3e-2, atol=3e-6
    )
    torch.testing.assert_close(
        weight_grad32.double(), weight_grad64, rtol=3e-2, atol=3e-6
    )


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("order", (0.99, 1.01))
def test_renyi_control_variate_resolves_first_order_fp32_cancellation(reverse, order):
    generator = torch.Generator().manual_seed(0)
    logits64 = torch.randn(4, 13, generator=generator, dtype=torch.float64) * 2.0
    teacher64 = (
        logits64 + torch.randn(4, 13, generator=generator, dtype=torch.float64) * 2.0e-4
    )

    logits32 = logits64.float().requires_grad_()
    log_p_s32 = torch.log_softmax(logits32, dim=-1)
    log_p_t32 = torch.log_softmax(teacher64.float(), dim=-1)
    value32 = _support_safe_renyi_rows(
        logits32,
        log_p_s32,
        log_p_t32,
        order,
        reverse=reverse,
    ).sum()

    log_p_s64 = torch.log_softmax(logits64, dim=-1)
    log_p_t64 = torch.log_softmax(teacher64, dim=-1)
    log_base, log_other = (log_p_s64, log_p_t64) if reverse else (log_p_t64, log_p_s64)
    reference = (
        torch.logsumexp(order * log_base + (1.0 - order) * log_other, dim=-1)
        / (order - 1.0)
    ).sum()

    torch.testing.assert_close(value32.double(), reference, rtol=2e-3, atol=2e-9)


@pytest.mark.parametrize(
    ("variant", "order"),
    (("alpha_divergence", 1.0e-3), ("renyi", 0.99), ("renyi_reverse", 0.01)),
)
def test_log_space_escort_avoids_underflow_times_overflow(variant, order):
    logits_s = torch.tensor([[0.0, -100.0]], requires_grad=True)
    log_p_s = torch.log_softmax(logits_s, dim=-1)
    log_p_t = torch.log_softmax(torch.tensor([[-100.0, 0.0]]), dim=-1)
    log_p_s64 = log_p_s.detach().double()
    log_p_t64 = log_p_t.double()
    p_s = log_p_s64.exp()

    if variant == "alpha_divergence":
        rows = _alpha_divergence_rows(logits_s, log_p_s, log_p_t, order, 1.0)
        log_escort = order * log_p_s64 + (1.0 - order) * log_p_t64
        affinity = torch.logsumexp(log_escort, dim=-1, keepdim=True).exp()
        expected_grad = (affinity * p_s - log_escort.exp()) / (1.0 - order)
    else:
        reverse = variant == "renyi_reverse"
        rows = _support_safe_renyi_rows(
            logits_s, log_p_s, log_p_t, order, reverse=reverse
        )
        log_base, log_other = (
            (log_p_s64, log_p_t64) if reverse else (log_p_t64, log_p_s64)
        )
        log_terms = order * log_base + (1.0 - order) * log_other
        escort = (log_terms - torch.logsumexp(log_terms, dim=-1, keepdim=True)).exp()
        expected_grad = (
            (order / (order - 1.0)) * (escort - p_s) if reverse else p_s - escort
        )

    actual_grad = torch.autograd.grad(rows.sum(), logits_s)[0]
    assert torch.isfinite(rows).all()
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(
        actual_grad.double(), expected_grad, rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize(
    ("alpha", "beta"),
    ((0.05, 0.05), (0.1, 0.5), (0.7, 0.9), (2.0, 2.0)),
)
# ── Specialized numerics ─────────────────────────────────────────────────────
def test_abkd_fp32_identity_has_exact_zero_value_and_grad(alpha, beta):
    out, hidden_grad, weight_grad = _run_fp32_stability_case(
        "abkd",
        torch.float32,
        kl_alpha=alpha,
        kl_beta=beta,
        near=False,
    )

    assert torch.equal(out.loss, torch.zeros_like(out.loss))
    assert torch.equal(out.kl, torch.zeros_like(out.kl))
    assert torch.equal(hidden_grad, torch.zeros_like(hidden_grad))
    assert torch.equal(weight_grad, torch.zeros_like(weight_grad))


@pytest.mark.parametrize(
    ("alpha", "beta"),
    ((0.05, 0.05), (0.1, 0.5), (0.7, 0.9), (2.0, 2.0)),
)
def test_abkd_fp32_near_identity_matches_fp64_value_and_grad(alpha, beta):
    out32, hidden_grad32, weight_grad32 = _run_fp32_stability_case(
        "abkd",
        torch.float32,
        kl_alpha=alpha,
        kl_beta=beta,
        near=True,
    )
    out64, hidden_grad64, weight_grad64 = _run_fp32_stability_case(
        "abkd",
        torch.float64,
        kl_alpha=alpha,
        kl_beta=beta,
        near=True,
    )

    assert out32.loss >= 0
    assert out32.kl >= 0
    torch.testing.assert_close(out32.loss.double(), out64.loss, rtol=3e-2, atol=2e-8)
    torch.testing.assert_close(out32.kl.double(), out64.kl, rtol=3e-2, atol=2e-8)
    torch.testing.assert_close(
        hidden_grad32.double(), hidden_grad64, rtol=3e-2, atol=3e-6
    )
    torch.testing.assert_close(
        weight_grad32.double(), weight_grad64, rtol=3e-2, atol=3e-6
    )


@pytest.mark.parametrize("alpha", (1.0e-3, 0.5, 1.0 - 1.0e-3))
# ── Objective edge cases ─────────────────────────────────────────────────────
def test_alpha_divergence_fp32_identity_has_exact_zero_value_and_grad(alpha):
    out, hidden_grad, weight_grad = _run_fp32_stability_case(
        "alpha_divergence",
        torch.float32,
        kl_alpha=alpha,
        near=False,
    )

    assert torch.equal(out.loss, torch.zeros_like(out.loss))
    assert torch.equal(out.kl, torch.zeros_like(out.kl))
    assert torch.equal(hidden_grad, torch.zeros_like(hidden_grad))
    assert torch.equal(weight_grad, torch.zeros_like(weight_grad))


@pytest.mark.parametrize("alpha", (1.0e-3, 0.5, 1.0 - 1.0e-3))
def test_alpha_divergence_fp32_near_identity_matches_fp64_value_and_grad(alpha):
    out32, hidden_grad32, weight_grad32 = _run_fp32_stability_case(
        "alpha_divergence",
        torch.float32,
        kl_alpha=alpha,
        near=True,
    )
    out64, hidden_grad64, weight_grad64 = _run_fp32_stability_case(
        "alpha_divergence",
        torch.float64,
        kl_alpha=alpha,
        near=True,
    )

    assert out32.loss >= 0
    assert out32.kl >= 0
    # Near-identity values sit at the fp32 log_softmax noise floor (~1e-7).
    torch.testing.assert_close(out32.loss.double(), out64.loss, rtol=3e-2, atol=3e-7)
    torch.testing.assert_close(out32.kl.double(), out64.kl, rtol=3e-2, atol=3e-7)
    torch.testing.assert_close(
        hidden_grad32.double(), hidden_grad64, rtol=3e-2, atol=3e-6
    )
    torch.testing.assert_close(
        weight_grad32.double(), weight_grad64, rtol=3e-2, atol=3e-6
    )


@pytest.mark.parametrize("dtype", (torch.float64, torch.float32))
@pytest.mark.parametrize("endpoint", ("zero", "one"))
def test_alpha_divergence_endpoint_limits_match_kl(endpoint, dtype):
    alpha = 1.0e-6 if endpoint == "zero" else 1.0 - 1.0e-6
    student_hidden = torch.eye(2, dtype=dtype, requires_grad=True)
    weight = torch.tensor(
        [[0.8, -0.6], [-0.4, 1.1], [0.3, 0.7]],
        dtype=dtype,
        requires_grad=True,
    )
    labels = torch.tensor([0, 1])
    teacher_logits = torch.tensor(
        [[0.2, -0.9, 0.5], [-0.7, 0.4, -0.1]],
        dtype=dtype,
    )

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        kl_weight=1.0,
        kl_divergence="alpha_divergence",
        kl_alpha=alpha,
        reduction="sum",
        backend="torch",
    )
    out.loss.backward()

    hidden_ref = student_hidden.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    log_p_s = torch.log_softmax(hidden_ref @ weight_ref.t(), dim=-1)
    log_p_t = torch.log_softmax(teacher_logits, dim=-1)
    if endpoint == "zero":
        reference = (log_p_t.exp() * (log_p_t - log_p_s)).sum()
    else:
        reference = (log_p_s.exp() * (log_p_s - log_p_t)).sum()
    reference.backward()

    torch.testing.assert_close(out.kl, reference.detach(), rtol=1e-4, atol=1e-10)
    torch.testing.assert_close(
        student_hidden.grad, hidden_ref.grad, rtol=1e-4, atol=1e-10
    )
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-4, atol=1e-10)


# ── Overflow and proxy cases ─────────────────────────────────────────────────
def test_precomputed_hidden_weight_overflow_is_not_a_dead_teacher_row():
    big = 1.0e20
    student_hidden = torch.tensor([[1.0, -1.0]], requires_grad=True)
    weight = torch.tensor([[0.25, 0.5], [-0.5, 0.25]], requires_grad=True)
    labels = torch.zeros(1, dtype=torch.long)
    teacher_hidden = torch.full((1, 2), big)
    teacher_weight = torch.full((2, 2), -big)

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        ),
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_divergence="forward",
        backend="torch",
    )

    assert torch.isnan(out.kl)
    assert torch.isnan(out.loss)
    out.loss.backward()
    assert torch.isnan(student_hidden.grad).any()
    assert torch.isnan(weight.grad).any()


@pytest.mark.parametrize(
    "variant,kwargs",
    (
        ("alpha_divergence", {"kl_alpha": 0.4}),
        ("renyi", {"kl_alpha": 0.9}),
        ("renyi_reverse", {"kl_alpha": 1.5}),
        ("abkd", {"kl_alpha": 0.7, "kl_beta": 0.9}),
    ),
)
def test_stabilized_proxy_variants_reject_higher_order_autograd(variant, kwargs):
    generator = torch.Generator().manual_seed(29)
    student_hidden = torch.randn(
        3, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    weight = torch.randn(
        5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    labels = torch.tensor([0, 2, 4])
    teacher_logits = torch.randn(labels.shape[0], weight.shape[0], dtype=torch.float64)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=1.0,
        kl_weight=1.0,
        kl_divergence=variant,
        backend="torch",
        **kwargs,
    )

    with pytest.raises(RuntimeError, match="first-order autograd only"):
        torch.autograd.grad(out.loss, student_hidden, create_graph=True)


def test_dkd_target_endpoint_keeps_tckd():
    student_hidden = torch.tensor([[1.0, 0.0]], requires_grad=True)
    weight = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [-0.25, 0.0]],
        requires_grad=True,
    )
    labels = torch.zeros(1, dtype=torch.long)
    teacher_logits = torch.tensor([[0.0, float("-inf"), float("-inf")]])
    log_p_y = torch.log_softmax(student_hidden.detach() @ weight.detach().t(), dim=-1)[
        0, 0
    ]

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_temperature=1.0,
        kl_divergence="dkd",
        kl_alpha=1.0,
        kl_beta=8.0,
        reduction="sum",
        backend="torch",
    )

    assert torch.allclose(out.kl, -log_p_y)
    _assert_all_finite(out, student_hidden, weight)


def test_dkd_generated_nonfinite_student_is_visible():
    big = 1.0e20
    student_hidden = torch.tensor([[big, 1.0]], requires_grad=True)
    weight = torch.tensor(
        [[0.0, 0.0], [-big, 0.0], [-big, 0.0]],
        requires_grad=True,
    )
    labels = torch.zeros(1, dtype=torch.long)
    teacher_logits = torch.tensor([[0.0, 0.0, float("-inf")]])

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_temperature=1.0,
        kl_divergence="dkd",
        kl_alpha=1.0,
        kl_beta=4.0,
        reduction="sum",
        backend="torch",
    )

    assert torch.isnan(out.kl)
    out.loss.backward()
    assert torch.isnan(student_hidden.grad).any()
    assert torch.isnan(weight.grad).any()


def test_dkd_ignored_label_is_zero():
    student_hidden = torch.tensor([[1.0, 0.0]], requires_grad=True)
    weight = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [-0.25, 0.0]],
        requires_grad=True,
    )
    labels = torch.full((1,), -100, dtype=torch.long)
    teacher_logits = torch.tensor([[0.0, float("-inf"), float("-inf")]])

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_temperature=1.0,
        kl_divergence="dkd",
        reduction="sum",
        backend="torch",
    )

    assert torch.equal(out.kl, torch.zeros_like(out.kl))
    _assert_all_finite(out, student_hidden, weight)


def test_triangular_both_zero_columns_stay_finite():
    student_hidden, weight, labels, teacher_logits = _precomputed_masked_inputs(seed=11)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        kl_weight=0.7,
        kl_temperature=1.0,
        teacher_ce_weight=0.0,
        kl_divergence="triangular",
        backend="torch",
    )

    _assert_all_finite(out, student_hidden, weight)


# ── Entropy gate endpoints ───────────────────────────────────────────────────
@pytest.mark.parametrize("kl_temperature", [1.0, 1.7])
def test_entropy_adaptive_saturated_gate_stays_finite(kl_temperature):
    generator = torch.Generator().manual_seed(1)
    student_hidden = (0.3 * torch.randn(_BT, _H, generator=generator)).requires_grad_(
        True
    )
    weight = (0.3 * torch.randn(_V, _H, generator=generator)).requires_grad_(True)
    labels = torch.randint(0, _V, (_BT,), generator=generator)
    teacher_logits = torch.zeros(_BT, _V)  # near-uniform -> normalized entropy ~1
    teacher_logits[:, _DEAD[0]] = float("-inf")  # teacher zero; student keeps mass

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        kl_weight=0.7,
        kl_temperature=kl_temperature,
        teacher_ce_weight=0.0,
        kl_divergence="entropy_adaptive",
        entropy_gate=ock.EntropyGate(scale=0.02),  # sharp -> gate saturates to 1.0
        backend="torch",
    )
    _assert_all_finite(out, student_hidden, weight)


@pytest.mark.parametrize("kl_temperature", [1.0, 1.7])
def test_entropy_adaptive_endpoint_gate_equals_forward(kl_temperature):
    student_hidden, weight, labels, teacher_logits = _precomputed_masked_inputs()
    teacher_logits = teacher_logits.clone()
    teacher_logits[:, _DEAD[1]] = torch.randn(_BT)  # keep one dead col, revive other

    def run(variant, gate=None):
        sh = student_hidden.detach().clone().requires_grad_(True)
        w = weight.detach().clone().requires_grad_(True)
        out = ock.distillation_loss(
            sh,
            w,
            labels,
            ock.PrecomputedTeacher(logits=teacher_logits),
            kl_weight=0.7,
            kl_temperature=kl_temperature,
            teacher_ce_weight=0.0,
            kl_divergence=variant,
            entropy_gate=gate,
            backend="torch",
        )
        return out

    forward = run("forward")
    gate_forward = run(
        "entropy_adaptive", ock.EntropyGate(alpha_min=1.0, alpha_max=1.0)
    )
    assert torch.isfinite(gate_forward.kl)
    assert torch.allclose(gate_forward.kl, forward.kl)
    gate_forward.loss.backward()

    # The reverse-only gate keeps a genuine infinite divergence.
    gate_reverse = run(
        "entropy_adaptive", ock.EntropyGate(alpha_min=0.0, alpha_max=0.0)
    )
    reverse = run("reverse")
    assert torch.equal(torch.isinf(gate_reverse.kl), torch.isinf(reverse.kl))
    if torch.isfinite(reverse.kl):
        assert torch.allclose(gate_reverse.kl, reverse.kl)


# ── Dead teacher rows ────────────────────────────────────────────────────────
@pytest.mark.parametrize("variant", DETACHED_KL_VARIANTS)
@pytest.mark.parametrize("kl_temperature", [1.0, 1.7])
@pytest.mark.parametrize("teacher_ce_weight", [0.0, 0.5])
def test_fully_dead_teacher_row_contributes_nothing(
    variant, kl_temperature, teacher_ce_weight
):
    generator = torch.Generator().manual_seed(3)
    student_hidden = (0.3 * torch.randn(_BT, _H, generator=generator)).requires_grad_(
        True
    )
    weight = (0.3 * torch.randn(_V, _H, generator=generator)).requires_grad_(True)
    labels = torch.randint(0, _V, (_BT,), generator=generator)
    teacher_logits = torch.randn(_BT, _V, generator=generator)
    teacher_logits[1, :] = float("-inf")  # row 1 fully dead

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        kl_weight=0.7,
        kl_temperature=kl_temperature,
        teacher_ce_weight=teacher_ce_weight,
        kl_divergence=variant,
        reduction="sum",
        **variant_kwargs(variant),
        backend="torch",
    )
    assert torch.isfinite(out.loss), f"{variant} loss not finite: {out.loss}"
    assert torch.isfinite(out.kl), f"{variant} kl not finite: {out.kl}"
    out.loss.backward()
    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_dead_teacher_row_leaves_live_rows_unchanged():
    generator = torch.Generator().manual_seed(3)
    student_hidden = (0.3 * torch.randn(_BT, _H, generator=generator)).requires_grad_(
        True
    )
    weight = (0.3 * torch.randn(_V, _H, generator=generator)).requires_grad_(True)
    labels = torch.randint(0, _V, (_BT,), generator=generator)
    teacher_logits = torch.randn(_BT, _V, generator=generator)
    teacher_logits[1, :] = float("-inf")

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        kl_weight=1.0,
        kl_temperature=1.0,
        teacher_ce_weight=0.0,
        kl_divergence="reverse",
        reduction="sum",
        backend="torch",
    )

    import torch.nn.functional as F

    logits_s = student_hidden.detach() @ weight.detach().t()
    log_p_s = F.log_softmax(logits_s, dim=-1)
    log_p_t = F.log_softmax(teacher_logits, dim=-1)
    p_s = log_p_s.exp()
    per_row = _zero_preserving_mul(p_s, log_p_s - log_p_t).sum(dim=-1)
    live = torch.tensor([0, 2, 3])
    expected = per_row[live].sum()
    assert torch.allclose(out.kl, expected)


# ── Finite-input parity ──────────────────────────────────────────────────────
def _finite_inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    student_hidden = torch.randn(
        _BT, _H, dtype=torch.float64, generator=generator, requires_grad=True
    )
    teacher_hidden = torch.randn(
        _BT, _H, dtype=torch.float64, generator=generator, requires_grad=True
    )
    weight = torch.randn(
        _V, _H, dtype=torch.float64, generator=generator, requires_grad=True
    )
    teacher_weight = torch.randn(
        _V, _H, dtype=torch.float64, generator=generator, requires_grad=True
    )
    labels = torch.randint(0, _V, (_BT,), generator=generator)
    labels[1] = -100
    return student_hidden, teacher_hidden, weight, teacher_weight, labels


@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
def test_finite_inputs_unaffected_by_guard(variant):
    (
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
    ) = _finite_inputs(seed=7)
    extra = variant_kwargs(variant)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_weight=0.7,
        kl_temperature=1.7,
        teacher_ce_weight=0.0,
        kl_divergence=variant,
        reduction="sum",
        backend="torch",
        **extra,
    )
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.kl)
    # A non-degenerate KL term: the guard did not collapse the value to zero.
    assert out.kl.abs().item() > 0.0
    out.loss.backward()
    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_finite_inputs_bit_identical_to_naive_reference():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = _finite_inputs(
        seed=13
    )
    temperature = 1.7
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.SeparateTeacher(teacher_hidden, teacher_weight),
        kl_weight=1.0,
        kl_temperature=temperature,
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_divergence="forward",
        reduction="sum",
        backend="torch",
    )

    import torch.nn.functional as F

    logits_s = student_hidden @ weight.t()
    logits_t = (teacher_hidden @ teacher_weight.t()).detach()
    mask = labels != -100
    log_p_s = F.log_softmax(logits_s / temperature, dim=-1)
    log_p_t = F.log_softmax(logits_t / temperature, dim=-1)
    p_t = log_p_t.exp()
    kl_row = (p_t * (log_p_t - log_p_s)).sum(dim=-1) * (temperature * temperature)
    kl_row = kl_row.masked_fill(~mask, 0.0)
    expected_kl = kl_row.sum()

    assert torch.allclose(out.kl, expected_kl)
    assert torch.allclose(out.loss, expected_kl)
