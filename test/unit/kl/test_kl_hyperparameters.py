"""KL hyperparameter domains, defaults, and plan-identity contracts."""

from __future__ import annotations

import pytest
import torch
from utils.variants import ALL_KL_VARIANTS

from orda_kernel import ChunkSize, EntropyGate
from orda_kernel._planning.models import (
    ALPHA_KL_VARIANTS,
    BETA_KL_VARIANTS,
    DEFAULT_KL_ALPHA,
    DEFAULT_KL_BETA,
    KL_HYPERPARAM_SPECS,
    VALID_KL_VARIANTS,
)
from orda_kernel._planning.policy import (
    build_execution_plan,
    resolve_kl_hyperparams,
)

_EXPECTED_HYPERPARAM_SPECS = {
    "alpha_divergence": {"kl_alpha": (0.0, 1.0, 0.5, True, True, None)},
    "skew_forward": {"kl_beta": (0.0, 1.0, 0.5, True, True, None)},
    "skew_js": {"kl_beta": (0.0, 1.0, 0.5, True, True, None)},
    "skew_reverse": {"kl_beta": (0.0, 1.0, 0.5, True, True, None)},
    "renyi": {"kl_alpha": (0.0, 2.0, 0.5, True, False, (0.99, 1.01))},
    "renyi_reverse": {"kl_alpha": (0.0, 2.0, 0.5, True, False, (0.99, 1.01))},
    "dkd": {
        "kl_alpha": (0.0, 16.0, 1.0, True, False, None),
        "kl_beta": (0.0, 16.0, 1.0, True, False, None),
    },
    "abkd": {
        "kl_alpha": (0.05, 2.0, 0.5, False, False, None),
        "kl_beta": (0.05, 2.0, 0.5, False, False, None),
    },
}


# ── Resolution helpers ───────────────────────────────────────────────────────
def _resolve(variant, *, kl_weight=1.0, kl_alpha=None, kl_beta=None):
    return resolve_kl_hyperparams(
        variant,
        kl_weight=kl_weight,
        kl_alpha=kl_alpha,
        kl_beta=kl_beta,
        entropy_gate=None,
    )


def _plan(variant, *, kl_weight=1.0, kl_alpha=None, kl_beta=None):
    return build_execution_plan(
        backend="torch",
        teacher_mode="separate",
        compute_dtype=torch.float32,
        bt=4,
        vocab_size=8,
        student_ce_weight=1.0,
        teacher_ce_weight=0.0,
        kl_weight=kl_weight,
        kl_temperature=1.0,
        kl_variant=variant,
        ignore_index=-100,
        reduction="mean",
        chunk_size=ChunkSize.nrows(4),
        num_chunks=None,
        max_chunks=None,
        max_fused_size=16,
        autotune=False,
        grad_weight_accumulation="auto",
        kl_alpha=kl_alpha,
        kl_beta=kl_beta,
    )


# ── Hyperparameter spec map ──────────────────────────────────────────────────
def test_hyperparameter_specs_are_exact():
    observed = {
        variant: {
            name: (
                spec.lo,
                spec.hi,
                spec.default,
                spec.open_lo,
                spec.open_hi,
                spec.exclude,
            )
            for name, spec in specs.items()
        }
        for variant, specs in KL_HYPERPARAM_SPECS.items()
    }

    assert observed == _EXPECTED_HYPERPARAM_SPECS


def test_alpha_and_beta_variant_sets_are_derived_from_specs():
    expected_alpha = frozenset(
        variant for variant, specs in KL_HYPERPARAM_SPECS.items() if "kl_alpha" in specs
    )
    expected_beta = frozenset(
        variant for variant, specs in KL_HYPERPARAM_SPECS.items() if "kl_beta" in specs
    )

    assert ALPHA_KL_VARIANTS == expected_alpha
    assert BETA_KL_VARIANTS == expected_beta


@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
def test_defaults_are_resolved_on_the_plan(variant):
    specs = _EXPECTED_HYPERPARAM_SPECS.get(variant, {})
    expected_alpha = specs.get("kl_alpha", (None, None, DEFAULT_KL_ALPHA))[2]
    expected_beta = specs.get("kl_beta", (None, None, DEFAULT_KL_BETA))[2]
    expected_gate = EntropyGate() if variant == "entropy_adaptive" else None
    alpha, beta, gate = _resolve(variant)
    plan = _plan(variant)

    assert (alpha, beta, gate) == (expected_alpha, expected_beta, expected_gate)
    assert (plan.kl_alpha, plan.kl_beta, plan.entropy_gate) == (
        expected_alpha,
        expected_beta,
        expected_gate,
    )


# ── Domains and canonicalization ─────────────────────────────────────────────
@pytest.mark.parametrize(
    ("variant", "kl_alpha", "kl_beta"),
    (
        ("alpha_divergence", 0.3, None),
        ("skew_forward", None, 0.3),
        ("skew_js", None, 0.3),
        ("skew_reverse", None, 0.3),
        ("renyi", 2.0, None),
        ("renyi_reverse", 0.25, None),
        ("dkd", 16.0, 16.0),
        ("abkd", 0.05, 2.0),
        ("triangular", None, None),
    ),
)
def test_domain_boundaries_accept_valid_values(variant, kl_alpha, kl_beta):
    alpha, beta, _ = _resolve(variant, kl_alpha=kl_alpha, kl_beta=kl_beta)

    assert alpha == (DEFAULT_KL_ALPHA if kl_alpha is None else kl_alpha)
    assert beta == (DEFAULT_KL_BETA if kl_beta is None else kl_beta)


@pytest.mark.parametrize(
    ("variant", "name", "value"),
    (
        ("alpha_divergence", "kl_alpha", 0.0),
        ("alpha_divergence", "kl_alpha", 1.0),
        ("skew_forward", "kl_beta", 0.0),
        ("skew_js", "kl_beta", 1.0),
        ("skew_reverse", "kl_beta", 0.0),
        ("skew_reverse", "kl_beta", 1.0),
        ("dkd", "kl_alpha", 0.0),
        ("dkd", "kl_beta", 16.1),
        ("abkd", "kl_alpha", 0.049),
        ("abkd", "kl_beta", 2.1),
    ),
)
def test_domains_reject_invalid_values(variant, name, value):
    kwargs = {"kl_alpha": None, "kl_beta": None}
    kwargs[name] = value

    with pytest.raises(ValueError, match=name):
        _resolve(variant, **kwargs)


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(True, id="bool"),
        pytest.param("0.5", id="numeric-string"),
        pytest.param("bad", id="non-numeric-string"),
        pytest.param(object(), id="object"),
    ),
)
@pytest.mark.parametrize(
    ("variant", "name", "kl_weight"),
    (
        ("dkd", "kl_alpha", 1.0),
        ("triangular", "kl_beta", 1.0),
        ("dkd", "kl_alpha", 0.0),
    ),
    ids=("active", "unused", "inactive"),
)
def test_hyperparameters_reject_non_real_values(
    variant,
    name,
    kl_weight,
    value,
):
    kwargs = {"kl_alpha": None, "kl_beta": None}
    kwargs[name] = value

    with pytest.raises(ValueError, match=rf"{name} must be a real number"):
        _resolve(variant, kl_weight=kl_weight, **kwargs)


@pytest.mark.parametrize("variant", ("renyi", "renyi_reverse"))
@pytest.mark.parametrize("order", (0.99, 1.01, 2.0))
def test_renyi_accepts_documented_orders(variant, order):
    alpha, beta, _ = _resolve(variant, kl_alpha=order)

    assert alpha == order
    assert beta == DEFAULT_KL_BETA


@pytest.mark.parametrize("variant", ("renyi", "renyi_reverse"))
@pytest.mark.parametrize("order", (0.0, 0.995, 1.0, 1.005, 2.5))
def test_renyi_rejects_invalid_or_ill_conditioned_orders(variant, order):
    with pytest.raises(ValueError, match="kl_alpha|abs\\(r - 1\\)"):
        _resolve(variant, kl_alpha=order)


def test_unused_explicit_values_use_canonical_defaults():
    alpha, beta, _ = _resolve("triangular", kl_alpha=0.7, kl_beta=0.3)
    assert (alpha, beta) == (DEFAULT_KL_ALPHA, DEFAULT_KL_BETA)

    with pytest.raises(ValueError, match="kl_alpha"):
        _resolve("triangular", kl_alpha=1.5)


def test_inactive_kl_uses_canonical_hyperparameters():
    alpha, beta, gate = _resolve("dkd", kl_weight=0.0, kl_alpha=0.7, kl_beta=0.3)
    assert (alpha, beta, gate) == (DEFAULT_KL_ALPHA, DEFAULT_KL_BETA, None)

    with pytest.raises(ValueError, match="kl_alpha"):
        _resolve("dkd", kl_weight=0.0, kl_alpha=16.1)


@pytest.mark.parametrize(
    ("variant", "name", "value"),
    (
        ("renyi", "kl_alpha", 1.5),
        ("dkd", "kl_beta", 16.0),
        ("abkd", "kl_alpha", 0.05),
    ),
)
def test_inactive_values_use_selected_variant_domain_before_canonicalization(
    variant, name, value
):
    kwargs = {"kl_alpha": None, "kl_beta": None}
    kwargs[name] = value
    alpha, beta, gate = _resolve(variant, kl_weight=0.0, **kwargs)

    assert (alpha, beta, gate) == (DEFAULT_KL_ALPHA, DEFAULT_KL_BETA, None)


def test_plan_equality_uses_defaults_and_canonical_unused_values():
    assert _plan("dkd") == _plan("dkd", kl_alpha=1.0, kl_beta=1.0)
    assert _plan("triangular") == _plan("triangular", kl_alpha=0.7, kl_beta=0.3)
    assert _plan("dkd", kl_weight=0.0) == _plan(
        "dkd",
        kl_weight=0.0,
        kl_alpha=0.7,
        kl_beta=0.3,
    )


# ── Registry completeness ────────────────────────────────────────────────────
def test_variant_registry_and_nonparameterized_defaults_are_complete():
    assert VALID_KL_VARIANTS == frozenset(ALL_KL_VARIANTS)
    assert set(KL_HYPERPARAM_SPECS) <= set(ALL_KL_VARIANTS)

    for variant in set(ALL_KL_VARIANTS) - set(KL_HYPERPARAM_SPECS):
        alpha, beta, _ = _resolve(variant)
        assert (alpha, beta) == (DEFAULT_KL_ALPHA, DEFAULT_KL_BETA)
