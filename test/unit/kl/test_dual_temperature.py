"""Dual-temperature Python, planning, and Torch contracts."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
import torch.nn.functional as F
from utils.reference import reference_distillation_loss
from utils.variants import DETACHED_KL_VARIANTS, variant_kwargs

import orda_kernel as ock
import orda_kernel._planning.policy as planning_policy
from orda_kernel import ops
from orda_kernel._planning.models import PRECOMPUTED_TOKENWISE
from orda_kernel._planning.policy import build_execution_plan

_OMITTED = object()
_T_S = 1.7
_T_T = 0.8
_KL_WEIGHT = 0.6
_ENTROPY_GATE = ock.EntropyGate(
    alpha_min=0.2,
    alpha_max=0.8,
    tau=0.4,
    scale=0.2,
)
_DETACHED_CASES = tuple(
    (
        variant,
        {"entropy_gate": _ENTROPY_GATE}
        if variant == "entropy_adaptive"
        else variant_kwargs(variant),
    )
    for variant in DETACHED_KL_VARIANTS
)


# ── Reference inputs ─────────────────────────────────────────────────────────
def _base_inputs():
    generator = torch.Generator().manual_seed(211)
    student_hidden = 0.3 * torch.randn(3, 4, dtype=torch.float64, generator=generator)
    weight = 0.25 * torch.randn(7, 4, dtype=torch.float64, generator=generator)
    teacher_logits = 0.35 * torch.randn(3, 7, dtype=torch.float64, generator=generator)
    labels = torch.tensor([0, 3, 6])
    return student_hidden, weight, teacher_logits, labels


def _run_public(
    variant: str,
    extra: dict,
    *,
    teacher_temperature: object = _OMITTED,
):
    student_hidden, weight, teacher_logits, labels = _base_inputs()
    student_hidden.requires_grad_(True)
    weight.requires_grad_(True)
    kwargs = dict(
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=_KL_WEIGHT,
        kl_temperature=_T_S,
        kl_divergence=variant,
        reduction="sum",
        backend="torch",
        **extra,
    )
    if teacher_temperature is not _OMITTED:
        kwargs["kl_teacher_temperature"] = teacher_temperature
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        **kwargs,
    )
    out.loss.backward()
    return out, student_hidden.grad, weight.grad


@pytest.mark.parametrize(
    ("variant", "extra"),
    _DETACHED_CASES,
    ids=[variant for variant, _ in _DETACHED_CASES],
)
def test_torch_asymmetric_detached_variants_match_fp64_oracle(variant, extra):
    student_hidden, weight, teacher_logits, labels = _base_inputs()
    student_hidden.requires_grad_(True)
    weight.requires_grad_(True)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=_KL_WEIGHT,
        kl_temperature=_T_S,
        kl_teacher_temperature=_T_T,
        kl_divergence=variant,
        reduction="sum",
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
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=_KL_WEIGHT,
        kl_temperature=_T_S,
        kl_teacher_temperature=_T_T,
        kl_divergence=variant,
        reduction="sum",
        **extra,
    )
    expected_loss, _, _, expected_kl, expected_hidden, _, expected_weight, _ = expected

    torch.testing.assert_close(out.kl, expected_kl.detach(), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(out.loss, expected_loss, rtol=1e-10, atol=1e-12)
    out.loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(
        student_hidden.grad, expected_hidden.grad, rtol=1e-9, atol=1e-11
    )
    torch.testing.assert_close(weight.grad, expected_weight.grad, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize(
    ("variant", "extra"),
    (
        ("forward", {}),
        ("dkd", {"kl_alpha": 1.0, "kl_beta": 4.0}),
        ("triangular", {}),
    ),
)
def test_asymmetric_torch_gradcheck(variant, extra):
    student_hidden, weight, teacher_logits, labels = _base_inputs()
    student_hidden.requires_grad_(True)
    weight.requires_grad_(True)

    def objective(hidden, projection):
        return ock.distillation_loss(
            hidden,
            projection,
            labels,
            ock.PrecomputedTeacher(logits=teacher_logits),
            student_ce_weight=0.0,
            teacher_ce_weight=0.0,
            kl_weight=_KL_WEIGHT,
            kl_temperature=_T_S,
            kl_teacher_temperature=_T_T,
            kl_divergence=variant,
            reduction="sum",
            backend="torch",
            **extra,
        ).loss

    assert torch.autograd.gradcheck(
        objective,
        (student_hidden, weight),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
        fast_mode=True,
    )


# ── Public compatibility ─────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("variant", "extra"),
    (
        ("forward", {}),
        ("entropy_adaptive", {"entropy_gate": _ENTROPY_GATE}),
        ("dkd", {"kl_alpha": 1.0, "kl_beta": 4.0}),
    ),
)
def test_omitted_none_and_explicit_equal_teacher_temperature_match(variant, extra):
    omitted = _run_public(variant, extra)
    inherited = _run_public(variant, extra, teacher_temperature=None)
    explicit = _run_public(variant, extra, teacher_temperature=_T_S)

    for candidate in (inherited, explicit):
        for actual, expected in zip(candidate[0], omitted[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(candidate[1], omitted[1], rtol=0, atol=0)
        torch.testing.assert_close(candidate[2], omitted[2], rtol=0, atol=0)


@pytest.mark.parametrize(
    "entrypoint",
    (
        ock.distillation_loss,
        ock.DistillationLoss,
        ock.KernelConfig,
        ops.tied_distillation_loss,
        ops.separate_distillation_loss,
        ops.precomputed_distillation_loss,
        build_execution_plan,
    ),
)
def test_dual_temperature_is_threaded_through_python_entrypoints(entrypoint):
    parameter = inspect.signature(entrypoint).parameters["kl_teacher_temperature"]
    assert parameter.default is None


def test_dual_temperature_does_not_extend_precomputed_builders_or_aligned_span():
    for entrypoint in (
        ock.aligned_span_distillation_loss,
        ock.mix_teacher_logits,
        ock.generalized_jensen_shannon,
    ):
        assert "kl_teacher_temperature" not in inspect.signature(entrypoint).parameters


@pytest.mark.parametrize("value", (0.0, -1.0, math.nan, math.inf, -math.inf))
def test_invalid_teacher_temperature_is_rejected(value):
    student_hidden, weight, teacher_logits, labels = _base_inputs()
    with pytest.raises(
        ValueError,
        match=r"kl_teacher_temperature.*(finite|> 0|representable)",
    ):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.PrecomputedTeacher(logits=teacher_logits),
            kl_weight=1.0,
            kl_temperature=1.3,
            kl_teacher_temperature=value,
            backend="torch",
        )


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(1.0e-40, id="reciprocal-overflow"),
        pytest.param(1.0e38, id="reciprocal-subnormal"),
    ),
)
def test_unrepresentable_teacher_temperature_reciprocal_is_rejected(value):
    student_hidden, weight, teacher_logits, labels = _base_inputs()
    with pytest.raises(
        ValueError,
        match=r"kl_teacher_temperature.*(representable|finite).*float32",
    ):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.PrecomputedTeacher(logits=teacher_logits),
            kl_weight=1.0,
            kl_temperature=1.3,
            kl_teacher_temperature=value,
            backend="torch",
        )


def test_kernel_config_teacher_temperature_override_reaches_torch():
    student_hidden, weight, teacher_logits, labels = _base_inputs()
    common = dict(
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_temperature=1.4,
        kl_divergence="forward",
        reduction="sum",
        backend="torch",
    )
    configured = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        kl_teacher_temperature=2.0,
        config=ock.KernelConfig(kl_teacher_temperature=0.7),
        **common,
    )
    explicit = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        kl_teacher_temperature=0.7,
        **common,
    )

    for actual, expected in zip(configured, explicit, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_module_threads_teacher_temperature_to_functional_api():
    student_hidden, weight, teacher_logits, labels = _base_inputs()
    criterion = ock.DistillationLoss(
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=0.4,
        kl_temperature=1.6,
        kl_teacher_temperature=0.75,
        kl_divergence="reverse",
        reduction="sum",
        backend="torch",
    )
    out_module = criterion(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
    )
    out_function = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=0.4,
        kl_temperature=1.6,
        kl_teacher_temperature=0.75,
        kl_divergence="reverse",
        reduction="sum",
        backend="torch",
    )

    assert criterion.kl_teacher_temperature == 0.75
    for actual, expected in zip(out_module, out_function, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ce_terms_use_raw_logits_when_teacher_temperature_changes():
    generator = torch.Generator().manual_seed(223)
    student_hidden = torch.randn(3, 4, dtype=torch.float64, generator=generator)
    teacher_hidden = torch.randn(3, 4, dtype=torch.float64, generator=generator)
    weight = torch.randn(7, 4, dtype=torch.float64, generator=generator)
    labels = torch.tensor([1, 4, 6])
    common = dict(
        student_ce_weight=1.0,
        teacher_ce_weight=0.5,
        kl_weight=0.4,
        kl_temperature=1.3,
        kl_divergence="forward",
        reduction="sum",
        backend="torch",
    )
    cold = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.TiedTeacher(teacher_hidden),
        kl_teacher_temperature=0.6,
        **common,
    )
    hot = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.TiedTeacher(teacher_hidden),
        kl_teacher_temperature=2.2,
        **common,
    )
    expected_student = F.cross_entropy(
        student_hidden @ weight.t(), labels, reduction="sum"
    )
    expected_teacher = F.cross_entropy(
        teacher_hidden @ weight.t(), labels, reduction="sum"
    )

    torch.testing.assert_close(cold.student_ce, hot.student_ce, rtol=0, atol=0)
    torch.testing.assert_close(cold.teacher_ce, hot.teacher_ce, rtol=0, atol=0)
    torch.testing.assert_close(
        cold.student_ce, expected_student, rtol=1e-15, atol=1e-15
    )
    torch.testing.assert_close(
        cold.teacher_ce, expected_teacher, rtol=1e-15, atol=1e-15
    )


@pytest.mark.parametrize("mode", ("tied", "separate"))
def test_asymmetric_detached_variant_keeps_teacher_gradient_detached(mode):
    generator = torch.Generator().manual_seed(227)
    student_hidden = torch.randn(
        3, 4, dtype=torch.float64, generator=generator, requires_grad=True
    )
    weight = torch.randn(
        7, 4, dtype=torch.float64, generator=generator, requires_grad=True
    )
    teacher_hidden = torch.randn(
        3, 4, dtype=torch.float64, generator=generator, requires_grad=True
    )
    labels = torch.tensor([0, 2, 5])
    if mode == "tied":
        teacher = ock.TiedTeacher(teacher_hidden, mode="frozen")
        teacher_weight = None
    else:
        teacher_weight = torch.randn(
            7, 4, dtype=torch.float64, generator=generator, requires_grad=True
        )
        teacher = ock.SeparateTeacher(teacher_hidden, teacher_weight)

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        teacher,
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_temperature=_T_S,
        kl_teacher_temperature=_T_T,
        kl_divergence="forward",
        backend="torch",
    )
    out.loss.backward()

    assert student_hidden.grad is not None
    assert weight.grad is not None
    assert teacher_hidden.grad is None
    if teacher_weight is not None:
        assert teacher_weight.grad is None


# ── Planning and backend boundary ────────────────────────────────────────────
def _plan(
    *,
    backend: str = "torch",
    teacher_mode: str = "precomputed",
    teacher_ce_weight: float = 0.0,
    variant: str = "forward",
    kl_weight: float = 1.0,
    student_temperature: float = _T_S,
    teacher_temperature: float | None = None,
):
    return build_execution_plan(
        backend=backend,
        teacher_mode=teacher_mode,
        compute_dtype=torch.float32,
        bt=3,
        vocab_size=7,
        student_ce_weight=1.0,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
        kl_temperature=student_temperature,
        kl_teacher_temperature=teacher_temperature,
        kl_variant=variant,
        ignore_index=-100,
        reduction="mean",
        chunk_size=ock.ChunkSize.nrows(3),
        num_chunks=None,
        max_chunks=None,
        max_fused_size=8,
        autotune=False,
        grad_weight_accumulation="auto",
    )


def test_plan_resolves_none_to_student_temperature():
    inherited = _plan(teacher_temperature=None)
    explicit = _plan(teacher_temperature=_T_S)

    assert inherited == explicit
    assert inherited.kl_temperature == _T_S
    assert inherited.kl_teacher_temperature == _T_S


def test_plan_resolves_temperatures_once(monkeypatch):
    original = planning_policy.resolve_kl_temperatures
    calls = 0

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(planning_policy, "resolve_kl_temperatures", counted)

    _plan()

    assert calls == 1


def test_plan_rejects_unrepresentable_student_temperature():
    with pytest.raises(ValueError, match=r"kl_temperature.*normal float32"):
        _plan(student_temperature=1.0e155)


def test_inactive_kl_canonicalizes_both_temperatures_to_one():
    plan = _plan(
        backend="triton",
        kl_weight=0.0,
        student_temperature=2.5,
        teacher_temperature=0.5,
    )

    assert plan.backend == "triton"
    assert plan.kl_temperature == 1.0
    assert plan.kl_teacher_temperature == 1.0
    assert plan.specialization.t_is_one is True


@pytest.mark.parametrize("backend", ("triton", "auto"))
@pytest.mark.parametrize(
    "variant",
    [variant for variant, _ in _DETACHED_CASES],
)
def test_phase5_asymmetric_detached_plans_keep_triton(backend, variant):
    plan = _plan(
        backend=backend,
        variant=variant,
        teacher_temperature=_T_T,
    )

    assert plan.backend == "triton"
    assert plan.kl_temperature == _T_S
    assert plan.kl_teacher_temperature == _T_T
    assert plan.specialization.kl_variant == variant
    assert plan.specialization.t_is_one is False


@pytest.mark.parametrize(
    "variant",
    [variant for variant, _ in _DETACHED_CASES],
)
def test_asymmetric_temperature_reuses_nonunit_specialization(variant):
    equal = _plan(
        backend="triton",
        variant=variant,
        student_temperature=2.0,
        teacher_temperature=2.0,
    )
    asymmetric = _plan(
        backend="triton",
        variant=variant,
        student_temperature=2.0,
        teacher_temperature=1.0,
    )

    assert asymmetric.specialization == equal.specialization


@pytest.mark.parametrize(
    ("student_temperature", "teacher_temperature", "expected"),
    (
        (1.0, None, True),
        (1.0, 1.0, True),
        (1.0, 2.0, False),
        (2.0, 1.0, False),
        (2.0, 2.0, False),
    ),
)
def test_t_is_one_means_both_resolved_temperatures_are_one(
    student_temperature,
    teacher_temperature,
    expected,
):
    plan = _plan(
        backend="triton",
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
    )

    assert plan.specialization.t_is_one is expected


def test_teacher_temperature_does_not_require_a_representable_square():
    teacher_temperature = 1.0e20
    plan = _plan(
        backend="triton",
        student_temperature=1.0,
        teacher_temperature=teacher_temperature,
    )

    assert plan.kl_teacher_temperature == teacher_temperature
    assert plan.specialization.t_is_one is False


def test_equal_temperature_triton_planning_does_not_regress():
    inherited = _plan(backend="triton", teacher_temperature=None)
    explicit = _plan(backend="triton", teacher_temperature=_T_S)

    assert inherited == explicit
    assert inherited.backend == "triton"
    assert inherited.kernel_path == PRECOMPUTED_TOKENWISE
    assert inherited.specialization.t_is_one is False


# ── Coupled restriction ──────────────────────────────────────────────────────
@pytest.mark.parametrize("backend", ("torch", "triton", "auto"))
@pytest.mark.parametrize("variant", ("mutual", "cojeffreys"))
def test_coupled_variants_reject_asymmetric_temperature_planning(backend, variant):
    with pytest.raises(ValueError, match=r"equal|same|T_s.*T_t|temperature"):
        _plan(
            backend=backend,
            teacher_mode="tied",
            variant=variant,
            teacher_temperature=_T_T,
        )


@pytest.mark.parametrize("variant", ("mutual", "cojeffreys"))
def test_coupled_variants_reject_asymmetric_temperatures(variant):
    student_hidden, weight, _, labels = _base_inputs()
    teacher_hidden = student_hidden.detach().clone().requires_grad_(True)

    with pytest.raises(ValueError, match=r"equal|same|T_s.*T_t|temperature"):
        ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.TiedTeacher(teacher_hidden, mode="cotrained"),
            student_ce_weight=0.0,
            teacher_ce_weight=0.0,
            kl_weight=1.0,
            kl_temperature=_T_S,
            kl_teacher_temperature=_T_T,
            kl_divergence=variant,
            backend="torch",
        )


@pytest.mark.parametrize("variant", ("mutual", "cojeffreys"))
def test_coupled_variants_keep_equal_temperature_behavior(variant):
    student_hidden, weight, _, labels = _base_inputs()
    student_hidden.requires_grad_(True)
    weight.requires_grad_(True)
    teacher_hidden = student_hidden.detach().clone().requires_grad_(True)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.TiedTeacher(teacher_hidden, mode="cotrained"),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_temperature=_T_S,
        kl_teacher_temperature=_T_S,
        kl_divergence=variant,
        backend="torch",
    )

    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert student_hidden.grad is not None
    assert teacher_hidden.grad is not None
    assert weight.grad is not None


@pytest.mark.parametrize("variant", ("mutual", "cojeffreys"))
def test_coupled_variants_keep_equal_temperature_triton_planning(variant):
    plan = _plan(
        backend="triton",
        teacher_mode="tied",
        variant=variant,
        student_temperature=2.0,
        teacher_temperature=2.0,
    )

    assert plan.backend == "triton"
    assert plan.specialization.kl_variant == variant
    assert plan.specialization.t_is_one is False
