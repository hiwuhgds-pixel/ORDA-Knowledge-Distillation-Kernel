from __future__ import annotations

import inspect
import threading
import typing
import warnings

import pytest
import torch
import torch.nn.functional as F
from utils.factories import make_separate_inputs, make_tied_inputs
from utils.variants import ALL_KL_VARIANTS, DETACHED_KL_VARIANTS

import orda_kernel
from orda_kernel import api as public_api
from orda_kernel._execution import dispatcher as execution_dispatcher
from orda_kernel._planning import policy as planning_policy


def _valid_cpu_inputs():
    return make_tied_inputs(
        torch,
        BT=4,
        H=8,
        V=16,
        dtype=torch.float32,
        seed=17,
        ignore_index=None,
        scale=1.0,
    )


def _valid_separate_inputs():
    return make_separate_inputs(
        torch,
        BT=3,
        Hs=4,
        Ht=6,
        V=11,
        dtype=torch.float32,
        seed=19,
        ignore_index=None,
        teacher_requires_grad=True,
        teacher_weight_requires_grad=True,
        scale=1.0,
    )


def test_public_availability_flag_is_boolean():
    assert isinstance(orda_kernel.is_available(), bool)


def test_public_exports_are_expected_surface():
    expected = {
        "AlignedLikelihoodConfig",
        "AlignedSpanPairs",
        "ChunkSize",
        "CrossVocabTeacher",
        "DistillationLoss",
        "DistillationLossOutput",
        "EntropyGate",
        "KernelConfig",
        "PrecomputedSpanTeacher",
        "PrecomputedTeacher",
        "SeparateTeacher",
        "TiedTeacher",
        "aligned_span_distillation_loss",
        "begin_autotune_run",
        "build_exact_token_map",
        "build_precomputed_span_teacher",
        "distillation_loss",
        "generalized_jensen_shannon",
        "is_available",
        "mix_teacher_logits",
        "project_teacher_logits",
        "reset_autotune_cache",
    }
    assert set(orda_kernel.__all__) == expected
    for name in expected:
        assert hasattr(orda_kernel, name)
    assert isinstance(orda_kernel.__version__, str)
    assert orda_kernel.__version__

    reflected_types = (
        orda_kernel.ChunkSize,
        orda_kernel.DistillationLossOutput,
        orda_kernel.KernelConfig,
        orda_kernel.PrecomputedTeacher,
        orda_kernel.SeparateTeacher,
        orda_kernel.TiedTeacher,
    )
    assert orda_kernel.DistillationLoss.__module__ == "orda_kernel.api"
    assert orda_kernel.distillation_loss.__module__ == "orda_kernel.api"
    for public_type in reflected_types:
        assert typing.get_type_hints(public_type)
        assert f"class {public_type.__name__}" in inspect.getsource(public_type)

    from orda_kernel import ops

    expected_ops = {
        "aligned_span_distillation_loss",
        "precomputed_distillation_loss",
        "separate_distillation_loss",
        "tied_distillation_loss",
    }
    assert set(ops.__all__) == expected_ops
    for name in expected_ops:
        assert hasattr(ops, name)


def test_tied_teacher_torch_backend_returns_named_output_and_backprops():
    hs, ht, weight, labels = _valid_cpu_inputs()
    out = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.TiedTeacher(ht),
        kl_weight=0.2,
        kl_temperature=1.5,
        backend="torch",
    )

    assert isinstance(out, orda_kernel.DistillationLossOutput)
    loss, student_ce, teacher_ce, kl = out
    assert loss is out.loss
    assert student_ce is out.student_ce
    assert teacher_ce is out.teacher_ce
    assert kl is out.kl
    out.loss.backward()
    assert hs.grad is not None
    assert ht.grad is not None
    assert weight.grad is not None


def test_default_auto_backend_on_cpu_matches_explicit_torch_backend():
    hs, ht, weight, labels = _valid_cpu_inputs()
    teacher = orda_kernel.TiedTeacher(ht)

    out_auto = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        kl_weight=0.2,
        kl_temperature=1.5,
    )
    out_torch = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        kl_weight=0.2,
        kl_temperature=1.5,
        backend="torch",
    )

    assert torch.allclose(out_auto.loss, out_torch.loss)
    assert torch.allclose(out_auto.student_ce, out_torch.student_ce)
    assert torch.allclose(out_auto.teacher_ce, out_torch.teacher_ce)
    assert torch.allclose(out_auto.kl, out_torch.kl)


def test_separate_teacher_default_teacher_ce_weight_is_zero():
    hs = torch.randn(3, 4, requires_grad=True)
    ht = torch.randn(3, 6, requires_grad=True)
    weight = torch.randn(11, 4, requires_grad=True)
    teacher_weight = torch.randn(11, 6, requires_grad=True)
    labels = torch.randint(0, 11, (3,))

    out = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.SeparateTeacher(ht, teacher_weight),
        kl_weight=0.3,
        backend="torch",
    )
    out.loss.backward()

    assert torch.equal(out.teacher_ce, torch.zeros_like(out.teacher_ce))
    assert teacher_weight.grad is None


def test_precomputed_teacher_accepts_hidden_weight_cache_in_torch_backend():
    hs = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(13, 4, requires_grad=True)
    teacher_hidden = torch.randn(5, 6)
    teacher_weight = torch.randn(13, 6)
    labels = torch.randint(0, 13, (5,))

    out = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.PrecomputedTeacher(
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        ),
        kl_weight=0.25,
        kl_temperature=1.7,
        backend="torch",
    )
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert torch.equal(out.teacher_ce, torch.zeros_like(out.teacher_ce))
    assert hs.grad is not None
    assert weight.grad is not None


def test_precomputed_teacher_rejects_ambiguous_or_grad_cache():
    logits = torch.randn(4, 16)
    teacher_hidden = torch.randn(4, 8)
    teacher_weight = torch.randn(16, 8)
    with pytest.raises(ValueError, match="logits or"):
        orda_kernel.PrecomputedTeacher(
            logits=logits,
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        )
    with pytest.raises(ValueError, match="logits or"):
        orda_kernel.PrecomputedTeacher(
            logits=logits,
            teacher_hidden=teacher_hidden,
        )
    with pytest.raises(ValueError, match="logits or"):
        orda_kernel.PrecomputedTeacher(
            logits=logits,
            teacher_weight=teacher_weight,
        )
    with pytest.raises(ValueError, match="must provide"):
        orda_kernel.PrecomputedTeacher()
    with pytest.raises(ValueError, match="both be provided"):
        orda_kernel.PrecomputedTeacher(teacher_hidden=teacher_hidden)
    with pytest.raises(ValueError, match="both be provided"):
        orda_kernel.PrecomputedTeacher(teacher_weight=teacher_weight)

    hs, _ht, weight, labels = _valid_cpu_inputs()
    bad_logits = torch.randn(labels.shape[0], weight.shape[0], requires_grad=True)
    with pytest.raises(ValueError, match="must not require gradients"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.PrecomputedTeacher(logits=bad_logits),
            backend="torch",
        )

    bad_weight = teacher_weight.detach().clone().requires_grad_(True)
    with pytest.raises(ValueError, match="must not require gradients"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.PrecomputedTeacher(
                teacher_hidden=teacher_hidden.detach(),
                teacher_weight=bad_weight,
            ),
            backend="torch",
        )

    bad_hidden = teacher_hidden.detach().clone().requires_grad_(True)
    with pytest.raises(ValueError, match="must not require gradients"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.PrecomputedTeacher(
                teacher_hidden=bad_hidden,
                teacher_weight=teacher_weight,
            ),
            backend="torch",
        )


@pytest.mark.parametrize("mutated", ["logits", "teacher_hidden", "teacher_weight"])
def test_precomputed_teacher_revalidates_mutated_cache(mutated):
    student_hidden, _teacher_hidden, weight, labels = _valid_cpu_inputs()
    logits = torch.randn(labels.shape[0], weight.shape[0])
    cached_hidden = torch.randn(labels.shape[0], 6)
    cached_weight = torch.randn(weight.shape[0], 6)

    if mutated == "logits":
        teacher = orda_kernel.PrecomputedTeacher(logits=logits)
        logits.requires_grad_(True)
    else:
        teacher = orda_kernel.PrecomputedTeacher(
            teacher_hidden=cached_hidden,
            teacher_weight=cached_weight,
        )
        getattr(teacher, mutated).requires_grad_(True)

    with pytest.raises(ValueError, match="must not require gradients"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            teacher,
            kl_weight=0.25,
            backend="torch",
        )


def test_teacher_descriptors_preserve_the_provided_tensors():
    tied_hidden = torch.randn(3, 4)
    separate_hidden = torch.randn(3, 6)
    separate_weight = torch.randn(11, 6)
    logits = torch.randn(3, 11)

    tied = orda_kernel.TiedTeacher(tied_hidden)
    separate = orda_kernel.SeparateTeacher(separate_hidden, separate_weight)
    precomputed = orda_kernel.PrecomputedTeacher(logits=logits)

    assert tied.hidden is tied_hidden
    assert separate.hidden is separate_hidden
    assert separate.weight is separate_weight
    assert precomputed.logits is logits
    assert precomputed.teacher_hidden is None
    assert precomputed.teacher_weight is None


@pytest.mark.parametrize(
    ("descriptor", "kwargs"),
    [
        (orda_kernel.TiedTeacher, {"hidden": object()}),
        (
            orda_kernel.SeparateTeacher,
            {"hidden": object(), "weight": torch.randn(11, 6)},
        ),
        (
            orda_kernel.SeparateTeacher,
            {"hidden": torch.randn(3, 6), "weight": object()},
        ),
        (orda_kernel.PrecomputedTeacher, {"logits": object()}),
    ],
)
def test_teacher_descriptors_reject_non_tensor_values(descriptor, kwargs):
    with pytest.raises(TypeError, match="torch.Tensor"):
        descriptor(**kwargs)


# ── Teacher shape contracts ──────────────────────────────────────────────────
def test_tied_and_separate_teacher_shapes_are_validated_before_execution():
    student_hidden = torch.randn(4, 4, requires_grad=True)
    weight = torch.randn(11, 4, requires_grad=True)
    labels = torch.randint(0, 11, (4,))

    with pytest.raises(ValueError, match="shape|row|batch"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(torch.randn(3, 4)),
            backend="torch",
        )
    with pytest.raises(ValueError, match="shape|hidden|dimension"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(torch.randn(4, 5)),
            backend="torch",
        )
    with pytest.raises(ValueError, match="shape|row|batch"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.SeparateTeacher(
                torch.randn(3, 6),
                torch.randn(11, 6),
            ),
            backend="torch",
        )
    with pytest.raises(ValueError, match="shape|vocab"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.SeparateTeacher(
                torch.randn(4, 6),
                torch.randn(10, 6),
            ),
            backend="torch",
        )
    with pytest.raises(ValueError, match="shape|hidden|dimension"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.SeparateTeacher(
                torch.randn(4, 6),
                torch.randn(11, 7),
            ),
            backend="torch",
        )


def test_kernel_config_profile_and_validation_contract():
    debug = public_api._resolve_profile("debug")
    assert debug.grad_weight_accumulation == "fp32"
    balanced = public_api._resolve_profile("balanced")
    assert balanced.grad_weight_accumulation == "auto"
    assert balanced.max_fused_size == 16384
    requested_chunk_size = orda_kernel.ChunkSize.nrows(7)
    merged = public_api._resolve_config(
        "debug", orda_kernel.KernelConfig(chunk_size=requested_chunk_size)
    )
    assert merged.grad_weight_accumulation == "fp32"
    assert merged.chunk_size == requested_chunk_size
    assert merged.chunk_size.value == 7
    override = public_api._resolve_config(
        "debug",
        orda_kernel.KernelConfig(
            grad_weight_accumulation="compute",
            chunk_size=orda_kernel.ChunkSize.nrows(7),
        ),
    )
    assert override.grad_weight_accumulation == "compute"

    fast = public_api._resolve_profile("fast")
    assert fast.autotune == "Max"
    assert fast.max_fused_size == 32768
    assert fast.refresh_autotune is False
    assert fast.chunk_size == orda_kernel.ChunkSize.fast()
    assert debug.max_fused_size == 32768

    assert public_api._resolve_config("fast", None).chunk_size.is_fast
    assert public_api._resolve_config("balanced", None).chunk_size.is_dynamic
    assert public_api._resolve_config("debug", None).chunk_size.is_dynamic
    chunk_override = public_api._resolve_config(
        "fast",
        orda_kernel.KernelConfig(chunk_size=orda_kernel.ChunkSize.nrows(2048)),
    )
    assert chunk_override.chunk_size.value == 2048
    # num_chunks outranks the profile's fast partition instead of conflicting.
    assert (
        public_api._resolve_config(
            "fast",
            orda_kernel.KernelConfig(num_chunks=8),
        ).num_chunks
        == 8
    )
    refreshed = public_api._resolve_config(
        "balanced",
        orda_kernel.KernelConfig(refresh_autotune=True),
    )
    assert refreshed.refresh_autotune is True
    with pytest.raises(ValueError, match="requires autotune"):
        public_api._resolve_config(
            "debug",
            orda_kernel.KernelConfig(refresh_autotune=True),
        )
    with pytest.raises(ValueError, match="autotune"):
        orda_kernel.KernelConfig(autotune="max")
    with pytest.raises(ValueError, match="refresh_autotune"):
        orda_kernel.KernelConfig(refresh_autotune=1)
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(ValueError, match="max_chunks"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            config=orda_kernel.KernelConfig(max_chunks=0),
        )
    with pytest.raises(ValueError, match="num_chunks"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            config=orda_kernel.KernelConfig(num_chunks=0),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            config=orda_kernel.KernelConfig(
                chunk_size=orda_kernel.ChunkSize.nrows(2),
                num_chunks=2,
            ),
        )
    with pytest.raises(ValueError, match="power of two"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            config=orda_kernel.KernelConfig(max_fused_size=100),
        )
    with pytest.raises(ValueError, match=r"max_fused_size must be <= 32768"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            config=orda_kernel.KernelConfig(max_fused_size=65536),
        )


def test_chunk_size_nominal_contract():
    dynamic = orda_kernel.ChunkSize.dynamic()
    fixed = orda_kernel.ChunkSize.nrows(512)

    assert dynamic.is_dynamic is True
    assert repr(dynamic) == "ChunkSize.dynamic()"
    with pytest.raises(TypeError, match="no nrows value"):
        _ = dynamic.value

    assert fixed.is_dynamic is False
    assert fixed.value == 512
    assert repr(fixed) == "ChunkSize.nrows(512)"
    assert orda_kernel.KernelConfig().chunk_size is None


@pytest.mark.parametrize("nrows", [0, -1, -2])
def test_chunk_size_nrows_rejects_nonpositive_values(nrows):
    with pytest.raises(ValueError, match=">= 1"):
        orda_kernel.ChunkSize.nrows(nrows)


@pytest.mark.parametrize("nrows", [True, 3.14, [1, 2, 3]])
def test_chunk_size_nrows_rejects_noninteger_values(nrows):
    with pytest.raises(TypeError, match="requires an int"):
        orda_kernel.ChunkSize.nrows(nrows)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw_value",
    ["auto", "dynamic", -2, -1, 0, 512, True, 3.14, [1, 2, 3]],
)
def test_kernel_config_rejects_raw_chunk_size_values(raw_value):
    with pytest.raises(TypeError, match="ChunkSize"):
        orda_kernel.KernelConfig(chunk_size=raw_value)  # type: ignore[arg-type]


def test_kernel_config_chunk_size_none_inherits_the_profile():
    assert orda_kernel.KernelConfig(chunk_size=None).chunk_size is None


# ── Numeric and config strictness ────────────────────────────────────────────
@pytest.mark.parametrize(
    "argument",
    [
        "student_ce_weight",
        "teacher_ce_weight",
        "kl_weight",
        "kl_temperature",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), id="negative-inf"),
    ],
)
def test_nonfinite_loss_scalars_are_rejected(argument, value):
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(ValueError, match=argument):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            backend="torch",
            **{argument: value},
        )


@pytest.mark.parametrize(
    "argument,value",
    [
        ("student_ce_weight", -0.5),
        ("teacher_ce_weight", -0.5),
        ("kl_weight", -0.5),
        ("kl_temperature", 0.0),
        ("kl_temperature", -0.5),
    ],
)
def test_loss_scalar_ranges_are_rejected(argument, value):
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(ValueError, match=argument):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            backend="torch",
            **{argument: value},
        )


@pytest.mark.parametrize("temperature", [5e-324, 1e155])
def test_kl_temperature_requires_safe_float32_derived_scalars(temperature):
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(ValueError, match="kl_temperature.*float32"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            kl_weight=0.25,
            kl_temperature=temperature,
            backend="torch",
        )


@pytest.mark.parametrize("value", [True, "1.0"])
@pytest.mark.parametrize(
    "argument",
    ["student_ce_weight", "teacher_ce_weight", "kl_weight", "kl_temperature"],
)
def test_loss_scalars_reject_bool_and_numeric_string(argument, value):
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(ValueError, match=argument):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            backend="torch",
            **{argument: value},
        )


@pytest.mark.parametrize(
    "field",
    ["num_chunks", "max_chunks", "max_fused_size"],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool"),
        pytest.param(1.0, id="integral-float"),
        pytest.param(1.5, id="fractional-float"),
    ],
)
def test_integer_kernel_config_fields_reject_bool_and_float(field, value):
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(ValueError, match=field):
        config = orda_kernel.KernelConfig(**{field: value})
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            backend="torch",
            config=config,
        )


# ── Input contract matrix ────────────────────────────────────────────────────
def test_zero_vocab_is_rejected_before_backend_dispatch():
    student_hidden = torch.randn(4, 8, requires_grad=True)
    teacher_hidden = torch.randn(4, 8, requires_grad=True)
    weight = torch.empty(0, 8, requires_grad=True)
    labels = torch.full((4,), -100, dtype=torch.long)

    with pytest.raises(ValueError, match="vocab|weight|non-empty|positive"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(teacher_hidden),
            backend="torch",
        )


def test_empty_batch_returns_differentiable_zero_output():
    student_hidden = torch.empty(0, 8, requires_grad=True)
    teacher_hidden = torch.empty(0, 8, requires_grad=True)
    weight = torch.randn(16, 8, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)

    out = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.TiedTeacher(teacher_hidden),
        kl_weight=0.25,
        backend="torch",
    )

    assert out.loss.requires_grad
    for component in out:
        assert torch.equal(component, torch.zeros_like(component))
    out.loss.backward()
    assert student_hidden.grad is not None
    assert teacher_hidden.grad is not None
    assert weight.grad is not None
    assert torch.all(student_hidden.grad == 0)
    assert torch.all(teacher_hidden.grad == 0)
    assert torch.all(weight.grad == 0)


def test_empty_teacher_only_keeps_student_branch_out_of_graph():
    student_hidden = torch.empty(0, 3, requires_grad=True)
    teacher_hidden = torch.empty(0, 4, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    teacher_weight = torch.randn(5, 4, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)

    out = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.SeparateTeacher(teacher_hidden, teacher_weight),
        student_ce_weight=0.0,
        teacher_ce_weight=1.0,
        kl_weight=0.0,
        backend="torch",
    )
    out.loss.backward()

    assert student_hidden.grad is None
    assert weight.grad is None
    assert teacher_hidden.grad is not None
    assert teacher_weight.grad is not None
    assert torch.equal(teacher_hidden.grad, torch.zeros_like(teacher_hidden))
    assert torch.equal(teacher_weight.grad, torch.zeros_like(teacher_weight))


def test_empty_tied_kl_only_keeps_teacher_hidden_out_of_graph():
    student_hidden = torch.empty(0, 3, requires_grad=True)
    teacher_hidden = torch.empty(0, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)

    out = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.TiedTeacher(teacher_hidden),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        backend="torch",
    )
    out.loss.backward()

    assert student_hidden.grad is not None
    assert weight.grad is not None
    assert teacher_hidden.grad is None
    assert torch.equal(student_hidden.grad, torch.zeros_like(student_hidden))
    assert torch.equal(weight.grad, torch.zeros_like(weight))


def test_empty_cotrained_kl_routes_both_branches():
    student_hidden = torch.empty(0, 3, requires_grad=True)
    teacher_hidden = torch.empty(0, 4, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    teacher_weight = torch.randn(5, 4, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)

    out = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.SeparateTeacher(teacher_hidden, teacher_weight),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=1.0,
        kl_divergence="mutual",
        backend="torch",
    )
    out.loss.backward()

    for tensor in (student_hidden, weight, teacher_hidden, teacher_weight):
        assert tensor.grad is not None
        assert torch.equal(tensor.grad, torch.zeros_like(tensor))


def test_empty_inactive_objective_has_no_gradient_graph():
    student_hidden = torch.empty(0, 3, requires_grad=True)
    teacher_hidden = torch.empty(0, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)

    out = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.TiedTeacher(teacher_hidden),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=0.0,
        backend="torch",
    )

    assert not out.loss.requires_grad
    assert student_hidden.grad is None
    assert teacher_hidden.grad is None
    assert weight.grad is None


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
# ── Dtype contracts ──────────────────────────────────────────────────────────
def test_torch_backend_accepts_supported_floating_dtypes(dtype):
    student_hidden = (torch.randn(4, 8, dtype=dtype) * 0.1).requires_grad_(True)
    teacher_hidden = (torch.randn(4, 8, dtype=dtype) * 0.1).requires_grad_(True)
    weight = (torch.randn(16, 8, dtype=dtype) * 0.1).requires_grad_(True)
    labels = torch.randint(0, 16, (4,))

    out = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.TiedTeacher(teacher_hidden),
        kl_weight=0.2,
        kl_temperature=1.4,
        backend="torch",
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert student_hidden.grad is not None
    assert teacher_hidden.grad is not None
    assert weight.grad is not None


def test_mismatched_floating_dtypes_are_rejected_before_execution():
    student_hidden = torch.randn(4, 8, dtype=torch.float32)
    teacher_hidden = torch.randn(4, 8, dtype=torch.float32)
    weight = torch.randn(16, 8, dtype=torch.float64)
    labels = torch.randint(0, 16, (4,))

    with pytest.raises(ValueError, match="dtype"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(teacher_hidden),
            backend="torch",
        )


def test_inactive_teacher_path_does_not_constrain_model_dtype():
    student_hidden = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    teacher_hidden = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 16, (4,))

    out = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.TiedTeacher(teacher_hidden),
        teacher_ce_weight=0.0,
        kl_weight=0.0,
        backend="torch",
    )
    out.loss.backward()

    assert teacher_hidden.grad is None
    assert student_hidden.grad is not None
    assert weight.grad is not None


def test_precomputed_logits_use_the_execution_compute_dtype():
    student_hidden = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 16, (4,))
    teacher_logits = torch.randn(4, 16, dtype=torch.float64)

    mixed = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.PrecomputedTeacher(logits=teacher_logits),
        kl_weight=0.5,
        backend="torch",
    )
    normalized = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.PrecomputedTeacher(logits=teacher_logits.float()),
        kl_weight=0.5,
        backend="torch",
    )

    assert torch.allclose(mixed.loss, normalized.loss, atol=1e-5)
    assert torch.allclose(mixed.kl, normalized.kl, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_nonfloating_model_tensors_are_rejected_before_execution(dtype):
    student_hidden = torch.ones(4, 8, dtype=dtype)
    teacher_hidden = torch.ones(4, 8, dtype=dtype)
    weight = torch.ones(16, 8, dtype=dtype)
    labels = torch.randint(0, 16, (4,))

    with pytest.raises(ValueError, match="floating|dtype"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(teacher_hidden),
            backend="torch",
        )


@pytest.mark.parametrize("profile", ["balanced", "fast", "debug"])
@pytest.mark.parametrize("backend", ["auto", "torch", "triton"])
@pytest.mark.parametrize(
    "bad_label",
    [
        pytest.param(-1, id="negative-non-ignore"),
        pytest.param(16, id="equal-to-vocab"),
    ],
)
def test_invalid_labels_are_rejected_before_backend_dispatch(
    profile,
    backend,
    bad_label,
):
    hs, ht, weight, labels = _valid_cpu_inputs()
    labels = labels.clone()
    labels[1] = bad_label

    with pytest.raises(ValueError, match="label|target|range|vocab"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            profile=profile,
            backend=backend,
        )


@pytest.mark.parametrize("ignore_index", [True, 1.9, "-100"])
def test_ignore_index_requires_an_integral_non_boolean_value(ignore_index):
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(ValueError, match="ignore_index.*integer"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            ignore_index=ignore_index,
            backend="torch",
        )


# ── Host label validation contract ───────────────────────────────────────────
def _zero_torch_output(*, student_hidden, **_):
    zero = student_hidden.sum() * 0.0
    return public_api.DistillationLossOutput(
        zero,
        zero.detach(),
        zero.detach(),
        zero.detach(),
    )


def test_default_host_label_validation_runs_once_before_torch_execution(monkeypatch):
    hs, ht, weight, labels = _valid_cpu_inputs()
    calls = []
    original_validate = public_api._validate_label_range
    captured_plans = []
    original_build = public_api.build_execution_plan

    def count_validation(*args, **kwargs):
        calls.append((args, kwargs))
        return original_validate(*args, **kwargs)

    def capture_plan(**kwargs):
        plan = original_build(**kwargs)
        captured_plans.append(plan)
        return plan

    monkeypatch.setattr(public_api, "_validate_label_range", count_validation)
    monkeypatch.setattr(public_api, "build_execution_plan", capture_plan)
    monkeypatch.setattr(public_api, "_torch_reference", _zero_torch_output)

    public_api.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.TiedTeacher(ht),
        backend="torch",
    )

    assert len(calls) == 1
    assert captured_plans[0].backend == "torch"
    assert captured_plans[0].assume_valid_labels is False


def test_trusted_labels_skip_host_scan_and_propagate_to_plan(monkeypatch):
    hs, ht, weight, labels = _valid_cpu_inputs()
    validation_flags = []
    plans = []
    original_validate = public_api._validate_label_range
    original_build = public_api.build_execution_plan

    class UnscannableLabels:
        def __eq__(self, _other):
            raise AssertionError("trusted labels must not be scanned")

    original_validate(
        typing.cast(torch.Tensor, UnscannableLabels()),
        16,
        -100,
        assume_valid_labels=True,
    )

    def capture_validation(*args, **kwargs):
        validation_flags.append(kwargs["assume_valid_labels"])
        return original_validate(*args, **kwargs)

    def capture_plan(**kwargs):
        plan = original_build(**kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(public_api, "_validate_label_range", capture_validation)
    monkeypatch.setattr(public_api, "build_execution_plan", capture_plan)
    monkeypatch.setattr(public_api, "_torch_reference", _zero_torch_output)

    public_api.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.TiedTeacher(ht),
        backend="torch",
        config=orda_kernel.KernelConfig(assume_valid_labels=True),
    )

    assert validation_flags == [True]
    assert plans[0].assume_valid_labels is True


def test_host_label_validation_runs_once_across_triton_plan_handoff(monkeypatch):
    hs, ht, weight, labels = _valid_cpu_inputs()
    calls = []
    captured_plans = []
    original_build = public_api.build_execution_plan
    original_validate = public_api._validate_label_range

    def count_validation(*args, **kwargs):
        calls.append((args, kwargs))
        return original_validate(*args, **kwargs)

    def capture_plan(**kwargs):
        plan = original_build(**kwargs)
        captured_plans.append(plan)
        return plan

    def fake_kernel(student_hidden, *_args, **_kwargs):
        zero = student_hidden.sum() * 0.0
        return zero, zero.detach(), zero.detach(), zero.detach()

    monkeypatch.setattr(public_api, "_validate_label_range", count_validation)
    monkeypatch.setattr(public_api, "build_execution_plan", capture_plan)
    monkeypatch.setattr(public_api, "_supports_triton", lambda *_: True)
    monkeypatch.setattr(public_api, "_validate_triton_model_dtype", lambda *_: None)
    monkeypatch.setattr(
        execution_dispatcher,
        "_validate_label_range",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prevalidated execution must not host-validate twice")
        ),
    )
    monkeypatch.setattr(
        execution_dispatcher, "_resolve_plan_executor", lambda _: fake_kernel
    )

    public_api.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.TiedTeacher(ht),
        backend="triton",
        config=orda_kernel.KernelConfig(chunk_size=orda_kernel.ChunkSize.nrows(4)),
    )

    assert len(calls) == 1
    assert captured_plans[0].backend == "triton"
    assert captured_plans[0].assume_valid_labels is False


def test_label_validation_uses_the_trusted_flag():
    assert (
        "assume_valid_labels" in inspect.signature(orda_kernel.KernelConfig).parameters
    )
    assert (
        "assume_valid_labels"
        in inspect.signature(planning_policy.build_execution_plan).parameters
    )


def test_bool_labels_are_rejected_before_dispatch():
    hs, ht, weight, _labels = _valid_cpu_inputs()
    labels = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError, match="integer class-index|torch.long"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            backend="torch",
        )


def _manual_tied_reference(
    hs, ht, weight, labels, *, kl_weight, kl_temperature, reduction
):
    logits_s = hs @ weight.t()
    logits_t = ht @ weight.t()
    mask = labels != -100
    denom = max(int(mask.sum().item()), 1)
    ce_s_all = F.cross_entropy(logits_s, labels, ignore_index=-100, reduction="none")
    ce_t_all = F.cross_entropy(logits_t, labels, ignore_index=-100, reduction="none")
    t = float(kl_temperature)
    kl_all = F.kl_div(
        F.log_softmax(logits_s / t, dim=-1),
        F.softmax(logits_t.detach() / t, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (t * t)
    kl_all = kl_all.masked_fill(~mask, 0.0)
    if reduction == "mean":
        ce_s = ce_s_all.sum() / denom
        ce_t = ce_t_all.sum() / denom
        kl = kl_all.sum() / denom
    else:
        ce_s = ce_s_all.sum()
        ce_t = ce_t_all.sum()
        kl = kl_all.sum()
    return ce_s + ce_t + kl_weight * kl, ce_s, ce_t, kl


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_torch_backend_reduction_contract_with_kl(reduction):
    hs, ht, weight, labels = _valid_cpu_inputs()
    labels = labels.clone()
    labels[1] = -100
    kl_weight = 0.35
    kl_temperature = 1.7

    out = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.TiedTeacher(ht),
        kl_weight=kl_weight,
        kl_temperature=kl_temperature,
        reduction=reduction,
        backend="torch",
    )
    expected = _manual_tied_reference(
        hs,
        ht,
        weight,
        labels,
        kl_weight=kl_weight,
        kl_temperature=kl_temperature,
        reduction=reduction,
    )

    assert torch.allclose(out.loss, expected[0])
    assert torch.allclose(out.student_ce, expected[1])
    assert torch.allclose(out.teacher_ce, expected[2])
    assert torch.allclose(out.kl, expected[3])


def test_reported_components_are_detached_but_loss_backprops():
    hs, ht, weight, labels = _valid_cpu_inputs()
    out = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.TiedTeacher(ht),
        kl_weight=0.25,
        backend="torch",
    )

    assert out.loss.requires_grad
    assert not out.student_ce.requires_grad
    assert not out.teacher_ce.requires_grad
    assert not out.kl.requires_grad
    out.loss.backward()
    assert hs.grad is not None


def test_zero_kl_weight_skips_and_reports_zero_kl_component():
    hs, ht, weight, labels = _valid_cpu_inputs()
    out = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.TiedTeacher(ht),
        kl_weight=0.0,
        backend="torch",
    )
    assert torch.equal(out.kl, torch.zeros_like(out.kl))


@pytest.mark.parametrize("variant", ALL_KL_VARIANTS)
def test_separate_teacher_torch_backend_accepts_all_kl_variants(variant: str):
    hs, ht, weight, teacher_weight, labels = _valid_separate_inputs()
    out = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.SeparateTeacher(ht, teacher_weight),
        kl_weight=0.3,
        kl_divergence=variant,
        backend="torch",
    )
    assert torch.isfinite(out.loss)


# ── Cross-device rejection ───────────────────────────────────────────────────
@pytest.mark.parametrize("bad_operand", ["weight", "labels", "teacher_hidden"])
def test_cross_device_operands_rejected_before_dispatch(bad_operand):
    hs, ht, weight, labels = _valid_cpu_inputs()
    if bad_operand == "weight":
        weight = weight.to("meta")
    elif bad_operand == "labels":
        labels = labels.to("meta")
    else:
        ht = ht.to("meta")

    with pytest.raises(ValueError, match="same device as student_hidden"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            kl_weight=0.2,
        )


def test_separate_teacher_weight_cross_device_rejected_before_dispatch():
    hs = torch.randn(3, 4, requires_grad=True)
    ht = torch.randn(3, 6, requires_grad=True)
    weight = torch.randn(11, 4, requires_grad=True)
    teacher_weight = torch.randn(11, 6).to("meta")
    labels = torch.randint(0, 11, (3,))

    with pytest.raises(ValueError, match="same device as student_hidden"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.SeparateTeacher(ht, teacher_weight),
            kl_weight=0.3,
            backend="torch",
        )


@pytest.mark.parametrize("bad_cache", ["logits", "teacher_hidden", "teacher_weight"])
def test_precomputed_cache_cross_device_rejected_before_dispatch(bad_cache):
    hs = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(11, 4, requires_grad=True)
    labels = torch.randint(0, 11, (3,))
    if bad_cache == "logits":
        teacher = orda_kernel.PrecomputedTeacher(logits=torch.randn(3, 11).to("meta"))
    else:
        teacher_hidden = torch.randn(3, 6)
        teacher_weight = torch.randn(11, 6)
        if bad_cache == "teacher_hidden":
            teacher_hidden = teacher_hidden.to("meta")
        else:
            teacher_weight = teacher_weight.to("meta")
        teacher = orda_kernel.PrecomputedTeacher(
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        )

    with pytest.raises(ValueError, match="same device as student_hidden"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            teacher,
            kl_weight=0.3,
            backend="torch",
        )


# ── Jeffreys inert/live contract ─────────────────────────────────────────────
def test_jeffreys_with_ce_teacher_allowed_when_kl_inert():
    hs, ht, weight, teacher_weight, labels = _valid_separate_inputs()

    out = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.SeparateTeacher(ht, teacher_weight),
        teacher_ce_weight=0.5,
        kl_weight=0.0,
        kl_divergence="jeffreys",
        backend="torch",
    )
    assert torch.equal(out.kl, torch.zeros_like(out.kl))
    out.loss.backward()
    assert ht.grad is not None
    assert teacher_weight.grad is not None
    assert ht.grad.abs().max() > 0
    assert teacher_weight.grad.abs().max() > 0


@pytest.mark.parametrize("backend", ("torch", "auto"))
def test_jeffreys_with_ce_teacher_runs_when_torch_backend_is_selected(backend):
    hs, ht, weight, teacher_weight, labels = _valid_separate_inputs()

    out = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.SeparateTeacher(ht, teacher_weight),
        teacher_ce_weight=0.5,
        kl_weight=0.3,
        kl_divergence="jeffreys",
        backend=backend,
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert ht.grad is not None
    assert teacher_weight.grad is not None


@pytest.mark.parametrize("variant", DETACHED_KL_VARIANTS)
@pytest.mark.parametrize("backend", ("triton", "auto"))
def test_live_detached_variants_with_trained_teacher_keep_a_triton_plan(
    monkeypatch,
    variant,
    backend,
):
    hs, ht, weight, teacher_weight, labels = _valid_separate_inputs()
    teacher = orda_kernel.SeparateTeacher(ht, teacher_weight)
    captured_plans = []
    original_build = public_api.build_execution_plan

    class _PlanCaptured(Exception):
        pass

    def capture_plan(**kwargs):
        captured_plans.append(original_build(**kwargs))
        raise _PlanCaptured

    monkeypatch.setattr(public_api, "_supports_triton", lambda *_: True)
    monkeypatch.setattr(public_api, "_validate_triton_model_dtype", lambda *_: None)
    monkeypatch.setattr(public_api, "build_execution_plan", capture_plan)

    with pytest.raises(_PlanCaptured):
        public_api.distillation_loss(
            hs,
            weight,
            labels,
            teacher,
            teacher_ce_weight=0.5,
            kl_weight=0.3,
            kl_divergence=variant,
            backend=backend,
        )
    plan = captured_plans[-1]
    assert plan.backend == "triton"
    assert plan.kernel_path == "separate_trained"
    assert plan.specialization.kl_variant == variant


def test_live_jeffreys_trained_teacher_reaches_label_validation(monkeypatch):
    hs, ht, weight, teacher_weight, _ = _valid_separate_inputs()
    labels = torch.tensor([0, -1, 2])

    monkeypatch.setattr(public_api, "_supports_triton", lambda *_: True)
    monkeypatch.setattr(public_api, "_validate_triton_model_dtype", lambda *_: None)

    with pytest.raises(ValueError, match="labels"):
        public_api.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.SeparateTeacher(ht, teacher_weight),
            teacher_ce_weight=0.5,
            kl_weight=0.3,
            kl_divergence="jeffreys",
            backend="triton",
            profile="debug",
        )


def test_inert_jeffreys_with_teacher_ce_keeps_a_triton_plan(monkeypatch):
    hs = torch.empty(0, 4, requires_grad=True)
    ht = torch.empty(0, 6, requires_grad=True)
    weight = torch.empty(11, 4, requires_grad=True)
    teacher_weight = torch.empty(11, 6, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)
    captured_plans = []
    original_build = public_api.build_execution_plan

    def capture_plan(**kwargs):
        plan = original_build(**kwargs)
        captured_plans.append(plan)
        return plan

    monkeypatch.setattr(public_api, "_supports_triton", lambda *_: True)
    monkeypatch.setattr(public_api, "_validate_triton_model_dtype", lambda *_: None)
    monkeypatch.setattr(public_api, "build_execution_plan", capture_plan)

    public_api.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.SeparateTeacher(ht, teacher_weight),
        teacher_ce_weight=0.5,
        kl_weight=0.0,
        kl_divergence="jeffreys",
        backend="triton",
    )

    plan = captured_plans[0]
    assert plan.backend == "triton"
    assert plan.kernel_path == "separate_trained"
    assert not plan.specialization.compute_kl
    assert plan.specialization.kl_variant == "forward"


def test_explicit_triton_backend_fails_fast_on_cpu():
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(RuntimeError, match="backend='triton'"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            backend="triton",
        )


def test_module_wrapper_matches_functional_torch_backend():
    hs, ht, weight, labels = _valid_cpu_inputs()
    teacher = orda_kernel.TiedTeacher(ht)
    gradient_materialization_scale = torch.tensor(8192.0, dtype=torch.float32)
    module = orda_kernel.DistillationLoss(
        kl_weight=0.25,
        kl_temperature=1.3,
        backend="torch",
    )

    out_module = module(
        hs,
        weight,
        labels,
        teacher,
        gradient_materialization_scale=gradient_materialization_scale,
    )
    out_function = orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        kl_weight=0.25,
        kl_temperature=1.3,
        backend="torch",
        gradient_materialization_scale=gradient_materialization_scale,
    )

    assert torch.allclose(out_module.loss, out_function.loss)
    assert torch.allclose(out_module.student_ce, out_function.student_ce)
    assert torch.allclose(out_module.teacher_ce, out_function.teacher_ce)
    assert torch.allclose(out_module.kl, out_function.kl)


@pytest.mark.parametrize(
    "gradient_materialization_scale,error,match",
    [
        (8192.0, TypeError, "torch.Tensor or None"),
        (torch.ones(2), ValueError, "exactly one element"),
        (torch.ones((), dtype=torch.float64), ValueError, "torch.float32"),
        (torch.ones((), requires_grad=True), ValueError, "must not require"),
        (torch.empty((), device="meta"), ValueError, "same device"),
    ],
)
def test_gradient_materialization_scale_has_a_metadata_only_contract(
    gradient_materialization_scale,
    error,
    match,
):
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(error, match=match):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            backend="torch",
            gradient_materialization_scale=gradient_materialization_scale,
        )


def test_grad_scaler_supplies_gradient_materialization_scale():
    class GradScalerStub:
        def __init__(self):
            self.scale_calls = 0

        def is_enabled(self):
            return True

        def scale(self, value):
            self.scale_calls += 1
            return value * 8192.0

    hs, ht, weight, labels = _valid_cpu_inputs()
    scaler = GradScalerStub()
    orda_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_kernel.TiedTeacher(ht),
        backend="torch",
        grad_scaler=scaler,
    )

    assert scaler.scale_calls == 1


def test_grad_scaler_rejects_explicit_materialization_scale():
    hs, ht, weight, labels = _valid_cpu_inputs()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with pytest.raises(ValueError, match="cannot be used together"):
        orda_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_kernel.TiedTeacher(ht),
            backend="torch",
            grad_scaler=scaler,
            gradient_materialization_scale=torch.tensor(8192.0),
        )


# ── Execution handoff ────────────────────────────────────────────────────────
@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_zero_objective_skips_backend_execution(monkeypatch, backend):
    student_hidden, teacher_hidden, weight, labels = _valid_cpu_inputs()
    student_hidden.requires_grad_()
    teacher_hidden.requires_grad_()
    weight.requires_grad_()

    def fail(*_args, **_kwargs):
        raise AssertionError("zero objective must not enter a backend executor")

    monkeypatch.setattr(public_api, "_torch_reference", fail)
    monkeypatch.setattr(execution_dispatcher, "_resolve_plan_executor", fail)
    monkeypatch.setattr(public_api, "_supports_triton", lambda *_: True)
    monkeypatch.setattr(public_api, "_validate_triton_model_dtype", lambda *_: None)

    output = orda_kernel.distillation_loss(
        student_hidden,
        weight,
        labels,
        orda_kernel.TiedTeacher(teacher_hidden),
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=0.0,
        backend=backend,
    )
    output.loss.backward()

    assert all(torch.equal(value, torch.zeros_like(value)) for value in output)
    assert torch.count_nonzero(student_hidden.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0
    assert teacher_hidden.grad is None


def test_public_call_validates_and_builds_one_plan_at_small_and_large_bt(monkeypatch):
    original_build = public_api.build_execution_plan
    plan_calls = []
    passed_plans = []
    validator_names = (
        "_validate_public_args",
        "_validate_teacher",
        "_validate_model_dtypes",
        "_validate_variant",
        "_validate_label_range",
    )
    validator_calls = dict.fromkeys(validator_names, 0)

    for validator_name in validator_names:
        original_validator = getattr(public_api, validator_name)

        def count_validation(
            *args,
            _name=validator_name,
            _validator=original_validator,
            **kwargs,
        ):
            validator_calls[_name] += 1
            return _validator(*args, **kwargs)

        monkeypatch.setattr(public_api, validator_name, count_validation)

    def count_plan(**kwargs):
        plan = original_build(**kwargs)
        plan_calls.append(plan)
        return plan

    def fake_torch_reference(*, student_hidden, execution_plan, **_):
        passed_plans.append(execution_plan)
        zero = student_hidden.sum() * 0.0
        return public_api.DistillationLossOutput(
            zero,
            zero.detach(),
            zero.detach(),
            zero.detach(),
        )

    monkeypatch.setattr(public_api, "build_execution_plan", count_plan)
    monkeypatch.setattr(public_api, "_torch_reference", fake_torch_reference)

    for bt in (4, 8192):
        student_hidden = torch.zeros(bt, 2)
        teacher_hidden = torch.zeros(bt, 2)
        weight = torch.zeros(4, 2)
        labels = torch.zeros(bt, dtype=torch.long)
        public_api.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(teacher_hidden),
            backend="torch",
        )

    # Public control flow remains constant as BT grows; the backend owns data work.
    assert validator_calls == dict.fromkeys(validator_names, 2)
    assert len(plan_calls) == len(passed_plans) == 2
    assert plan_calls == passed_plans
    assert all(plan.backend == "torch" for plan in plan_calls)


def test_public_triton_passes_plan_and_attempt_to_executor(monkeypatch):
    student_hidden, teacher_hidden, weight, labels = _valid_cpu_inputs()
    received = []

    def fail_executor(*args):
        received.append((args[-2], args[-1]))
        raise RuntimeError("executor failed")

    monkeypatch.setattr(public_api, "_supports_triton", lambda *_: True)
    monkeypatch.setattr(public_api, "_validate_triton_model_dtype", lambda *_: None)
    monkeypatch.setattr(
        execution_dispatcher, "_resolve_plan_executor", lambda _: fail_executor
    )
    monkeypatch.setattr(
        execution_dispatcher,
        "_validate_label_range",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public API must pass prevalidated labels")
        ),
    )

    with pytest.raises(RuntimeError, match="executor failed"):
        public_api.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(teacher_hidden),
            backend="triton",
        )

    assert len(received) == 1
    execution_plan, attempt = received[0]
    assert execution_plan.teacher_mode == "tied"
    assert attempt.num_chunks >= 1


# ── Auto-fallback warnings ───────────────────────────────────────────────────
def test_auto_backend_torch_fallback_warns_once(monkeypatch):
    monkeypatch.setattr(public_api, "_WARNED_AUTO_TORCH_FALLBACK", False)
    student_hidden, teacher_hidden, weight, labels = _valid_cpu_inputs()

    with pytest.warns(RuntimeWarning, match="fell back to the torch reference"):
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(teacher_hidden),
            backend="auto",
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(teacher_hidden),
            backend="auto",
        )


def test_auto_backend_fallback_warning_is_thread_safe(monkeypatch):
    monkeypatch.setattr(public_api, "_WARNED_AUTO_TORCH_FALLBACK", False)
    worker_count = 8
    barrier = threading.Barrier(worker_count)
    warning_calls = []
    errors = []
    result_lock = threading.Lock()

    def capture_warning(*args, **kwargs):
        with result_lock:
            warning_calls.append((args, kwargs))

    def worker():
        try:
            barrier.wait()
            public_api._warn_auto_torch_fallback("test")
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    monkeypatch.setattr(public_api.warnings, "warn", capture_warning)
    threads = [threading.Thread(target=worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(warning_calls) == 1


def test_explicit_torch_backend_does_not_warn(monkeypatch):
    monkeypatch.setattr(public_api, "_WARNED_AUTO_TORCH_FALLBACK", False)
    student_hidden, teacher_hidden, weight, labels = _valid_cpu_inputs()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        orda_kernel.distillation_loss(
            student_hidden,
            weight,
            labels,
            orda_kernel.TiedTeacher(teacher_hidden),
            backend="torch",
        )
