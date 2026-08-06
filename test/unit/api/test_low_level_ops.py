"""Strict public contracts for the three direct low-level ops wrappers."""

from __future__ import annotations

import inspect
import math
from unittest.mock import MagicMock

import pytest
import torch
from utils.variants import COUPLED_KL_VARIANTS

from orda_kernel import ChunkSize, ops
from orda_kernel.ops import precomputed_teacher, separate_teacher, tied_teacher
from orda_kernel.ops._common import (
    normalize_gradient_materialization_scale,
    prepare_gradient_materialization_factor,
)
from orda_kernel.ops._kernels.precomputed.tokenwise import PrecomputedCEFunction
from orda_kernel.ops._kernels.separate.cotrained import SeparateCotrainedCEFunction
from orda_kernel.ops._kernels.separate.frozen import SeparateFrozenCEFunction
from orda_kernel.ops._kernels.separate.trained import SeparateTrainedCEFunction
from orda_kernel.ops._kernels.tied.cotrained import TiedCotrainedCEFunction
from orda_kernel.ops._kernels.tied.frozen import TiedFrozenCEFunction
from orda_kernel.ops._kernels.tied.trained import TiedTrainedCEFunction

# ── Shared call helpers ──────────────────────────────────────────────────────
_MODULES = {
    "tied": tied_teacher,
    "separate": separate_teacher,
    "precomputed": precomputed_teacher,
}


@pytest.mark.parametrize(
    "module,export",
    [
        (tied_teacher, "tied_distillation_loss"),
        (separate_teacher, "separate_distillation_loss"),
        (precomputed_teacher, "precomputed_distillation_loss"),
    ],
)
def test_direct_modules_publish_only_the_low_level_operation(module, export):
    assert module.__all__ == [export]


def _cpu_operands(mode: str) -> dict[str, object]:
    student_hidden = torch.randn(4, 5, requires_grad=True)
    weight = torch.randn(11, 5, requires_grad=True)
    labels = torch.randint(0, 11, (4,))
    operands: dict[str, object] = {
        "student_hidden": student_hidden,
        "weight": weight,
        "labels": labels,
    }
    if mode == "tied":
        operands["teacher_hidden"] = torch.randn(4, 5, requires_grad=True)
    elif mode == "separate":
        operands["teacher_hidden"] = torch.randn(4, 7, requires_grad=True)
        operands["teacher_weight"] = torch.randn(11, 7, requires_grad=True)
    else:
        operands["logits_teacher"] = torch.randn(4, 11)
        operands["teacher_hidden"] = None
        operands["teacher_weight"] = None
    return operands


def _invoke(mode: str, operands: dict[str, object], **kwargs):
    if mode == "tied":
        return ops.tied_distillation_loss(
            operands["student_hidden"],
            operands.get("teacher_hidden"),
            operands["weight"],
            operands["labels"],
            **kwargs,
        )
    if mode == "separate":
        return ops.separate_distillation_loss(
            operands["student_hidden"],
            operands.get("teacher_hidden"),
            operands["weight"],
            operands.get("teacher_weight"),
            operands["labels"],
            **kwargs,
        )
    return ops.precomputed_distillation_loss(
        operands["student_hidden"],
        operands["weight"],
        logits_teacher=operands.get("logits_teacher"),
        labels=operands["labels"],
        teacher_hidden=operands.get("teacher_hidden"),
        teacher_weight=operands.get("teacher_weight"),
        **kwargs,
    )


def _enable_wrapper_validation(monkeypatch, mode: str) -> None:
    monkeypatch.setattr(_MODULES[mode], "HAS_TRITON", True)


def _block_function_apply(monkeypatch, mode: str) -> None:
    def unexpected_apply(*_args):
        raise AssertionError(
            "wrapper reached Function.apply before rejecting invalid input"
        )

    if mode == "tied":
        owners = (
            TiedFrozenCEFunction,
            TiedTrainedCEFunction,
            TiedCotrainedCEFunction,
        )
    elif mode == "separate":
        owners = (
            SeparateFrozenCEFunction,
            SeparateTrainedCEFunction,
            SeparateCotrainedCEFunction,
        )
    else:
        owners = (PrecomputedCEFunction,)
    for owner in owners:
        monkeypatch.setattr(owner, "apply", unexpected_apply)


@pytest.mark.parametrize(
    "reduction,expected_factor,expected_denominator",
    [("mean", 4.0, 2), ("sum", 8.0, None)],
)
def test_gradient_materialization_factor_folds_token_reduction_once(
    reduction,
    expected_factor,
    expected_denominator,
):
    labels = torch.tensor([0, -100, 1, -100])
    scale = torch.tensor(8.0)
    factor, denominator, effective_scale = prepare_gradient_materialization_factor(
        labels,
        -100,
        reduction,
        scale,
    )

    assert factor is not None
    assert effective_scale is not None
    assert torch.equal(effective_scale, scale)
    assert float(factor) == expected_factor
    if expected_denominator is None:
        assert denominator is None
    else:
        assert denominator is not None
        assert int(denominator) == expected_denominator


@pytest.mark.parametrize(
    ("dtype", "label_count"),
    (
        (torch.float16, 2),
        (torch.bfloat16, 512),
    ),
)
def test_gradient_materialization_factor_neutralizes_storage_underflow(
    dtype,
    label_count,
):
    labels = torch.zeros(label_count, dtype=torch.long)
    factor, denominator, effective_scale = prepare_gradient_materialization_factor(
        labels,
        -100,
        "mean",
        torch.tensor(torch.finfo(torch.float32).tiny),
        dtype,
    )

    assert factor is not None
    assert denominator is not None
    assert effective_scale is not None
    assert float(factor) == 1.0
    assert int(denominator) == label_count
    assert float(effective_scale) == label_count


def test_invalid_materialization_scale_uses_a_neutral_mean_factor():
    labels = torch.tensor([0, 1])
    factor, denominator, effective_scale = prepare_gradient_materialization_factor(
        labels,
        -100,
        "mean",
        torch.tensor(float("nan")),
        torch.float16,
    )

    assert factor is not None
    assert denominator is not None
    assert effective_scale is not None
    assert float(factor) == 1.0
    assert int(denominator) == 2
    assert float(effective_scale) == 2.0


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        torch.finfo(torch.float32).tiny / 2.0,
        torch.finfo(torch.float32).max,
    ],
)
def test_unsafe_gradient_materialization_scale_normalizes_to_one(value):
    effective = normalize_gradient_materialization_scale(torch.tensor(value))

    assert effective is not None
    assert torch.equal(effective, torch.ones((), dtype=torch.float32))


@pytest.mark.parametrize(
    "value",
    [
        torch.finfo(torch.float32).tiny,
        1.0,
        8192.0,
        1.0 / torch.finfo(torch.float32).tiny,
    ],
)
def test_safe_gradient_materialization_scale_is_preserved(value):
    scale = torch.tensor(value)
    effective = normalize_gradient_materialization_scale(scale)

    assert effective is not None
    assert torch.equal(effective, scale)


def test_materialization_scale_uses_the_gradient_storage_dtype_boundary():
    scale = torch.tensor(torch.finfo(torch.float32).tiny)

    effective = normalize_gradient_materialization_scale(scale, torch.float16)

    assert effective is not None
    assert torch.equal(effective, torch.ones((), dtype=torch.float32))


def test_scaled_cached_backward_compensates_materialization_scale():
    class Context:
        compute_grad = True
        gradient_materialization_scaled = True
        weight_dtype = torch.float32
        saved_tensors = (
            torch.full((2, 2), 4.0),
            torch.full((2, 2), 4.0),
            torch.tensor(8.0),
        )

    gradients = TiedFrozenCEFunction.backward(Context(), torch.tensor(8.0))

    assert len(gradients) == TiedFrozenCEFunction.INPUT_COUNT
    assert torch.equal(gradients[0] / 8.0, torch.full((2, 2), 0.5))
    assert torch.equal(gradients[2] / 8.0, torch.full((2, 2), 0.5))
    assert gradients[4] is None


# ── Pre-launch validation stubs ──────────────────────────────────────────────
def _TensorStub(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: object = torch.float32,
    requires_grad: bool = False,
):
    # A spec-backed mock preserves the production isinstance(Tensor) contract
    # without allocating CUDA storage in CPU-only validation tests.
    tensor = MagicMock(spec=torch.Tensor)
    tensor.shape = shape
    tensor.device = device
    tensor.dtype = dtype
    tensor.layout = torch.strided
    tensor.requires_grad = requires_grad
    tensor.ndim = len(shape)
    tensor.is_floating_point.return_value = dtype.is_floating_point
    tensor.numel.return_value = math.prod(shape)
    tensor.amax.return_value = torch.tensor(0.0)
    valid_mask = _MaskStub(valid=True)
    tensor.__eq__.return_value = valid_mask
    tensor.__ge__.return_value = valid_mask
    tensor.__lt__.return_value = valid_mask
    return tensor


class _MaskStub:
    def __init__(self, *, valid: bool):
        self._valid = valid

    def __and__(self, _other):
        return self

    def __or__(self, _other):
        return self

    def __invert__(self):
        return self

    def all(self) -> bool:
        return self._valid


class _BadValuesStub:
    def __getitem__(self, _item):
        return self

    def tolist(self) -> list[int]:
        return [-1]


def _InvalidLabelsStub(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: object = torch.float32,
    requires_grad: bool = False,
):
    tensor = _TensorStub(shape, device, dtype, requires_grad)
    invalid_mask = _MaskStub(valid=False)
    tensor.__eq__.return_value = invalid_mask
    tensor.__ge__.return_value = invalid_mask
    tensor.__lt__.return_value = invalid_mask
    tensor.__getitem__.return_value = _BadValuesStub()
    return tensor


def _cuda_stub_operands(mode: str) -> dict[str, object]:
    cuda = torch.device("cuda")
    operands: dict[str, object] = {
        "student_hidden": _TensorStub((4, 5), cuda),
        "weight": _TensorStub((11, 5), cuda),
        "labels": _TensorStub((4,), cuda, dtype=torch.long),
    }
    if mode == "tied":
        operands["teacher_hidden"] = _TensorStub((4, 5), cuda)
    elif mode == "separate":
        operands["teacher_hidden"] = _TensorStub((4, 7), cuda)
        operands["teacher_weight"] = _TensorStub((11, 7), cuda)
    else:
        operands["logits_teacher"] = _TensorStub((4, 11), cuda)
        operands["teacher_hidden"] = None
        operands["teacher_weight"] = None
    return operands


def _prelaunch_validation_operands(monkeypatch, mode: str) -> dict[str, object]:
    _enable_wrapper_validation(monkeypatch, mode)
    _block_function_apply(monkeypatch, mode)
    return _cuda_stub_operands(mode)


# ── Imports and fail-fast behavior ───────────────────────────────────────────
def test_ops_public_imports_are_callable():
    expected = {
        "aligned_span_distillation_loss",
        "precomputed_distillation_loss",
        "separate_distillation_loss",
        "tied_distillation_loss",
    }
    assert set(ops.__all__) == expected
    for name in expected:
        assert callable(getattr(ops, name))


@pytest.mark.parametrize("mode", _MODULES)
def test_direct_ops_fail_fast_when_triton_is_unavailable(monkeypatch, mode):
    monkeypatch.setattr(_MODULES[mode], "HAS_TRITON", False)
    with pytest.raises(RuntimeError, match="requires Triton"):
        _invoke(mode, _cpu_operands(mode))


@pytest.mark.parametrize("mode", _MODULES)
def test_direct_ops_reject_cpu_tensors_before_function_apply(monkeypatch, mode):
    _enable_wrapper_validation(monkeypatch, mode)
    with pytest.raises(ValueError, match="CUDA/HIP"):
        _invoke(mode, _cpu_operands(mode))


# ── Shape and dtype contracts ────────────────────────────────────────────────
@pytest.mark.parametrize("mode", _MODULES)
def test_student_hidden_must_be_2d(monkeypatch, mode):
    _enable_wrapper_validation(monkeypatch, mode)
    operands = _cpu_operands(mode)
    operands["student_hidden"] = torch.randn(2, 2, 5)
    with pytest.raises(ValueError, match="student_hidden.*2D"):
        _invoke(mode, operands)


@pytest.mark.parametrize("mode", _MODULES)
def test_weight_shape_is_validated(monkeypatch, mode):
    _enable_wrapper_validation(monkeypatch, mode)
    operands = _cpu_operands(mode)
    operands["weight"] = torch.randn(11, 6)
    with pytest.raises(ValueError, match="weight.*shape"):
        _invoke(mode, operands)


@pytest.mark.parametrize("mode", _MODULES)
def test_model_operands_require_strided_layout(monkeypatch, mode):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    operands["student_hidden"].layout = torch.sparse_coo

    with pytest.raises(ValueError, match="student_hidden.*torch.strided"):
        _invoke(mode, operands)


@pytest.mark.parametrize("mode", _MODULES)
def test_labels_shape_and_dtype_are_validated(monkeypatch, mode):
    _enable_wrapper_validation(monkeypatch, mode)
    bad_shape = _cpu_operands(mode)
    bad_shape["labels"] = torch.zeros(4, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="labels.*shape"):
        _invoke(mode, bad_shape)

    bad_dtype = _cpu_operands(mode)
    bad_dtype["labels"] = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(ValueError, match="torch.long"):
        _invoke(mode, bad_dtype)


def test_tied_teacher_hidden_shape_is_validated(monkeypatch):
    _enable_wrapper_validation(monkeypatch, "tied")
    operands = _cpu_operands("tied")
    operands["teacher_hidden"] = torch.randn(4, 7)
    with pytest.raises(ValueError, match="teacher_hidden.*does not match"):
        _invoke("tied", operands)


def test_separate_teacher_shapes_are_validated(monkeypatch):
    _enable_wrapper_validation(monkeypatch, "separate")
    bad_hidden = _cpu_operands("separate")
    bad_hidden["teacher_hidden"] = torch.randn(3, 7)
    with pytest.raises(ValueError, match="teacher_hidden.*shape"):
        _invoke("separate", bad_hidden)

    bad_weight = _cpu_operands("separate")
    bad_weight["teacher_weight"] = torch.randn(10, 7)
    with pytest.raises(ValueError, match="teacher_weight.*shape"):
        _invoke("separate", bad_weight)


@pytest.mark.parametrize(
    "mode,missing",
    [
        ("tied", "teacher_hidden"),
        ("separate", "teacher_hidden"),
        ("separate", "teacher_weight"),
    ],
)
def test_required_teacher_operands_are_not_optional(monkeypatch, mode, missing):
    _enable_wrapper_validation(monkeypatch, mode)
    operands = _cpu_operands(mode)
    operands[missing] = None
    with pytest.raises(ValueError, match=missing):
        _invoke(mode, operands)


@pytest.mark.parametrize(
    "mode,bad_operand",
    [
        ("tied", "teacher_hidden"),
        ("tied", "weight"),
        ("tied", "labels"),
        ("separate", "teacher_hidden"),
        ("separate", "teacher_weight"),
        ("separate", "weight"),
        ("separate", "labels"),
        ("precomputed", "logits_teacher"),
        ("precomputed", "weight"),
        ("precomputed", "labels"),
    ],
)
def test_device_mismatch_is_rejected_before_apply(monkeypatch, mode, bad_operand):
    _enable_wrapper_validation(monkeypatch, mode)
    operands = _cuda_stub_operands(mode)
    original = operands[bad_operand]
    operands[bad_operand] = _TensorStub(
        original.shape,
        torch.device("cpu"),
        dtype=original.dtype,
        requires_grad=original.requires_grad,
    )
    with pytest.raises(ValueError, match="same device|all tensors"):
        _invoke(mode, operands)


@pytest.mark.parametrize("mode", _MODULES)
def test_invalid_label_range_is_rejected_before_apply(monkeypatch, mode):
    _enable_wrapper_validation(monkeypatch, mode)
    operands = _cuda_stub_operands(mode)
    operands["labels"] = _InvalidLabelsStub(
        (4,),
        torch.device("cuda"),
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="labels.*outside"):
        _invoke(mode, operands)


def test_direct_label_validation_uses_only_the_trusted_flag():
    for function in (
        ops.tied_distillation_loss,
        ops.separate_distillation_loss,
        ops.precomputed_distillation_loss,
    ):
        parameters = inspect.signature(function).parameters
        names = list(parameters)
        assert parameters["assume_valid_labels"].default is False
        assert parameters["assume_valid_labels"].kind is inspect.Parameter.KEYWORD_ONLY
        assert names.index("assume_valid_labels") == names.index("autotune") + 1


@pytest.mark.parametrize("mode", _MODULES)
def test_direct_trusted_label_flag_reaches_execution(
    monkeypatch,
    mode,
):
    _enable_wrapper_validation(monkeypatch, mode)
    operands = _cuda_stub_operands(mode)
    operands["labels"] = _InvalidLabelsStub(
        (4,),
        torch.device("cuda"),
        dtype=torch.long,
    )
    marker = (object(), object(), object(), object())
    monkeypatch.setattr(
        _MODULES[mode], "execute_plan", lambda *_args, **_kwargs: marker
    )

    assert _invoke(mode, operands, assume_valid_labels=True) is marker


@pytest.mark.parametrize("mode", _MODULES)
def test_direct_host_label_validation_runs_once_before_execution(monkeypatch, mode):
    _enable_wrapper_validation(monkeypatch, mode)
    module = _MODULES[mode]
    original_validate = module._validate_label_range
    calls = []
    marker = (object(), object(), object(), object())

    def count_validation(*args, **kwargs):
        calls.append((args, kwargs))
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(module, "_validate_label_range", count_validation)
    monkeypatch.setattr(module, "execute_plan", lambda *_args, **_kwargs: marker)

    assert _invoke(mode, _cuda_stub_operands(mode)) is marker
    assert len(calls) == 1


@pytest.mark.parametrize("mode", _MODULES)
def test_direct_dynamic_ops_use_shared_execution_engine(monkeypatch, mode):
    _enable_wrapper_validation(monkeypatch, mode)
    module = _MODULES[mode]
    marker = (object(), object(), object(), object())
    captured = []

    def fake_execute(plan, *args, **kwargs):
        captured.append((plan, args, kwargs))
        return marker

    monkeypatch.setattr(module, "execute_plan", fake_execute)

    assert _invoke(mode, _cuda_stub_operands(mode)) is marker
    assert len(captured) == 1
    plan, _args, kwargs = captured[0]
    assert plan.chunk_policy.dynamic is True
    assert kwargs["_labels_prevalidated"] is True


# ── Scalar and config strictness ─────────────────────────────────────────────
@pytest.mark.parametrize("mode", _MODULES)
@pytest.mark.parametrize(
    "keyword,value",
    [
        ("student_ce_weight", -0.5),
        ("teacher_ce_weight", -0.5),
        ("kl_weight", -0.5),
        ("kl_temperature", 0.0),
    ],
)
def test_invalid_scalar_ranges_are_rejected(monkeypatch, mode, keyword, value):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match=keyword):
        _invoke(
            mode,
            operands,
            **{keyword: value},
        )


@pytest.mark.parametrize("mode", _MODULES)
@pytest.mark.parametrize(
    "keyword",
    ["student_ce_weight", "teacher_ce_weight", "kl_weight", "kl_temperature"],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), id="negative-inf"),
    ],
)
def test_nonfinite_scalars_are_rejected(monkeypatch, mode, keyword, value):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match=keyword):
        _invoke(
            mode,
            operands,
            **{keyword: value},
        )


@pytest.mark.parametrize("mode", _MODULES)
@pytest.mark.parametrize("value", [5e-324, 1e155])
def test_unrepresentable_kl_temperature_is_rejected(monkeypatch, mode, value):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match="kl_temperature.*float32"):
        _invoke(mode, operands, kl_temperature=value)


@pytest.mark.parametrize("value", [True, "1.0"])
def test_direct_ops_reject_bool_and_numeric_string_scalars(monkeypatch, value):
    mode = "tied"
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match="student_ce_weight"):
        _invoke(
            mode,
            operands,
            student_ce_weight=value,
        )


@pytest.mark.parametrize("mode", _MODULES)
@pytest.mark.parametrize(
    "keyword",
    ["autotune", "grad_weight_accumulation"],
)
def test_direct_execution_flags_reject_non_string_values(monkeypatch, mode, keyword):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match=keyword):
        _invoke(mode, operands, **{keyword: 1})


@pytest.mark.parametrize("mode", _MODULES)
@pytest.mark.parametrize("ignore_index", [True, 1.9, "-100"])
def test_direct_ignore_index_requires_an_integer(monkeypatch, mode, ignore_index):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match="ignore_index.*integer"):
        _invoke(mode, operands, ignore_index=ignore_index)


@pytest.mark.parametrize("mode", _MODULES)
def test_reduction_is_strict(monkeypatch, mode):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match="reduction"):
        _invoke(
            mode,
            operands,
            reduction="batchmean",
        )


@pytest.mark.parametrize("mode", _MODULES)
@pytest.mark.parametrize("value", [0, 1.0, True])
def test_max_chunks_is_strict_positive_integer(monkeypatch, mode, value):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match="max_chunks"):
        _invoke(
            mode,
            operands,
            max_chunks=value,
        )


@pytest.mark.parametrize("mode", _MODULES)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(100, id="not-power-of-two"),
        pytest.param(128.0, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_max_fused_size_is_strict_power_of_two_integer(monkeypatch, mode, value):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match="max_fused_size"):
        _invoke(
            mode,
            operands,
            max_fused_size=value,
        )


@pytest.mark.parametrize("mode", _MODULES)
@pytest.mark.parametrize(
    "value",
    [None, "auto", "dynamic", -2, -1, 0, 2, 1.0, 1.5, True],
)
def test_chunk_size_rejects_raw_values(
    monkeypatch,
    mode,
    value,
):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(TypeError, match="ChunkSize"):
        _invoke(
            mode,
            operands,
            chunk_size=value,
        )


@pytest.mark.parametrize("mode", _MODULES)
@pytest.mark.parametrize("value", [0, 1.0, 1.5, True])
def test_num_chunks_is_strict_positive_integer(monkeypatch, mode, value):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match="num_chunks"):
        _invoke(
            mode,
            operands,
            num_chunks=value,
        )


@pytest.mark.parametrize("mode", _MODULES)
def test_chunk_size_and_num_chunks_are_mutually_exclusive(monkeypatch, mode):
    operands = _prelaunch_validation_operands(monkeypatch, mode)
    with pytest.raises(ValueError, match="chunk_size.*num_chunks|mutually exclusive"):
        _invoke(
            mode,
            operands,
            chunk_size=ChunkSize.nrows(2),
            num_chunks=2,
        )


# ── Variant and routing contracts ────────────────────────────────────────────
@pytest.mark.parametrize("variant", COUPLED_KL_VARIANTS)
def test_precomputed_op_rejects_coupled_variants(monkeypatch, variant):
    monkeypatch.setattr(_MODULES["precomputed"], "HAS_TRITON", False)
    with pytest.raises(ValueError, match=variant):
        _invoke(
            "precomputed",
            _cpu_operands("precomputed"),
            kl_weight=0.3,
            kl_divergence=variant,
        )


@pytest.mark.parametrize(
    "variant,kl_weight,teacher_ce_weight,expected_path",
    [
        ("forward", 0.0, 0.0, "tied_frozen"),
        ("forward", 0.3, 0.0, "tied_frozen"),
        ("jeffreys", 0.3, 0.0, "tied_frozen"),
        ("mutual", 0.0, 0.0, "tied_frozen"),
        ("cojeffreys", 0.0, 0.0, "tied_frozen"),
        ("forward", 0.0, 0.5, "tied_trained"),
        ("forward", 0.3, 0.5, "tied_trained"),
        ("jeffreys", 0.3, 0.5, "tied_trained"),
        ("mutual", 0.0, 0.5, "tied_trained"),
        ("cojeffreys", 0.0, 0.5, "tied_trained"),
        ("mutual", 0.3, 0.0, "tied_cotrained"),
        ("cojeffreys", 0.3, 0.5, "tied_cotrained"),
    ],
)
def test_tied_wrapper_routes_each_gradient_policy(
    monkeypatch,
    variant,
    kl_weight,
    teacher_ce_weight,
    expected_path,
):
    _enable_wrapper_validation(monkeypatch, "tied")
    monkeypatch.setattr(
        TiedFrozenCEFunction,
        "apply",
        lambda *_args: "tied_frozen",
    )
    monkeypatch.setattr(
        TiedTrainedCEFunction,
        "apply",
        lambda *_args: "tied_trained",
    )
    monkeypatch.setattr(
        TiedCotrainedCEFunction,
        "apply",
        lambda *_args: "tied_cotrained",
    )

    result = _invoke(
        "tied",
        _cuda_stub_operands("tied"),
        kl_weight=kl_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_divergence=variant,
    )
    assert result == expected_path


def test_separate_rejects_unknown_variant(monkeypatch):
    monkeypatch.setattr(separate_teacher, "HAS_TRITON", False)
    with pytest.raises(ValueError, match="kl_divergence"):
        _invoke(
            "separate",
            _cpu_operands("separate"),
            kl_weight=0.3,
            kl_divergence="hellinger",
        )


@pytest.mark.parametrize(
    "variant,kl_weight,teacher_ce_weight,expected_path",
    [
        ("forward", 0.0, 0.0, "frozen"),
        ("forward", 0.3, 0.0, "frozen"),
        ("jeffreys", 0.3, 0.0, "frozen"),
        ("mutual", 0.0, 0.0, "frozen"),
        ("cojeffreys", 0.0, 0.0, "frozen"),
        ("forward", 0.3, 0.5, "trained"),
        ("jeffreys", 0.0, 0.5, "trained"),
        ("jeffreys", 0.3, 0.5, "trained"),
        ("reverse", 0.3, 0.5, "trained"),
        ("entropy_adaptive", 0.3, 0.5, "trained"),
        ("mutual", 0.3, 0.0, "cotrained"),
        ("cojeffreys", 0.3, 0.0, "cotrained"),
        ("cojeffreys", 0.3, 0.5, "cotrained"),
    ],
)
def test_separate_wrapper_routes_each_gradient_policy(
    monkeypatch,
    variant,
    kl_weight,
    teacher_ce_weight,
    expected_path,
):
    _enable_wrapper_validation(monkeypatch, "separate")
    monkeypatch.setattr(
        SeparateFrozenCEFunction,
        "apply",
        lambda *_args: "frozen",
    )
    monkeypatch.setattr(
        SeparateTrainedCEFunction,
        "apply",
        lambda *_args: "trained",
    )
    monkeypatch.setattr(
        SeparateCotrainedCEFunction,
        "apply",
        lambda *_args: "cotrained",
    )

    result = _invoke(
        "separate",
        _cuda_stub_operands("separate"),
        kl_weight=kl_weight,
        teacher_ce_weight=teacher_ce_weight,
        kl_divergence=variant,
    )
    assert result == expected_path


# ── Tuple order and defaults ─────────────────────────────────────────────────
@pytest.mark.parametrize(
    "mode,apply_owner,expected_default",
    [
        ("tied", TiedTrainedCEFunction, 1.0),
        ("separate", SeparateFrozenCEFunction, 0.0),
        ("precomputed", PrecomputedCEFunction, 0.0),
    ],
)
def test_direct_output_tuple_order_and_teacher_ce_default(
    monkeypatch,
    mode,
    apply_owner,
    expected_default,
):
    _enable_wrapper_validation(monkeypatch, mode)
    outputs = tuple(torch.tensor(float(index)) for index in range(4))
    captured: list[tuple[object, ...]] = []
    gradient_materialization_scale = _TensorStub((), torch.device("cuda"))

    def fake_apply(*args):
        captured.append(args)
        return outputs

    monkeypatch.setattr(apply_owner, "apply", fake_apply)
    result = _invoke(
        mode,
        _cuda_stub_operands(mode),
        gradient_materialization_scale=gradient_materialization_scale,
    )

    assert isinstance(result, tuple)
    assert len(result) == 4
    assert result is outputs
    assert all(
        actual is expected for actual, expected in zip(result, outputs, strict=True)
    )
    execution_plan = captured[0][-2]
    attempt = captured[0][-1]
    assert captured[0][-3] is gradient_materialization_scale
    assert execution_plan.teacher_ce_weight == expected_default
    assert attempt.num_chunks >= 1


# ── Precomputed representation contracts ─────────────────────────────────────
def test_precomputed_accepts_hidden_weight_representation_before_apply(monkeypatch):
    _enable_wrapper_validation(monkeypatch, "precomputed")
    operands = _cuda_stub_operands("precomputed")
    operands["logits_teacher"] = None
    operands["teacher_hidden"] = _TensorStub((4, 7), torch.device("cuda"))
    operands["teacher_weight"] = _TensorStub((11, 7), torch.device("cuda"))
    marker = (object(), object(), object(), object())
    monkeypatch.setattr(
        PrecomputedCEFunction,
        "apply",
        lambda *_args: marker,
    )
    assert (
        _invoke(
            "precomputed",
            operands,
            assume_valid_teacher_cache=True,
        )
        is marker
    )


def test_precomputed_representation_shapes_are_validated(monkeypatch):
    _enable_wrapper_validation(monkeypatch, "precomputed")
    bad_logits = _cpu_operands("precomputed")
    bad_logits["logits_teacher"] = torch.randn(4, 10)
    with pytest.raises(ValueError, match="logits_teacher.*shape"):
        _invoke("precomputed", bad_logits)

    bad_hidden = _cpu_operands("precomputed")
    bad_hidden["logits_teacher"] = None
    bad_hidden["teacher_hidden"] = torch.randn(3, 7)
    bad_hidden["teacher_weight"] = torch.randn(11, 7)
    with pytest.raises(ValueError, match="teacher_hidden.*shape"):
        _invoke("precomputed", bad_hidden)

    bad_weight = _cpu_operands("precomputed")
    bad_weight["logits_teacher"] = None
    bad_weight["teacher_hidden"] = torch.randn(4, 7)
    bad_weight["teacher_weight"] = torch.randn(10, 7)
    with pytest.raises(ValueError, match="teacher_weight.*shape"):
        _invoke("precomputed", bad_weight)


def test_precomputed_rejects_ambiguous_representations(monkeypatch):
    operands = _prelaunch_validation_operands(monkeypatch, "precomputed")
    operands["teacher_hidden"] = _TensorStub((4, 7), torch.device("cuda"))
    operands["teacher_weight"] = _TensorStub((11, 7), torch.device("cuda"))
    with pytest.raises(
        ValueError,
        match="not both|ambiguous|either|exactly one|logits.*hidden",
    ):
        _invoke("precomputed", operands)


@pytest.mark.parametrize("orphan", ["teacher_hidden", "teacher_weight"])
def test_precomputed_rejects_orphan_hidden_or_weight(monkeypatch, orphan):
    _enable_wrapper_validation(monkeypatch, "precomputed")
    operands = _cpu_operands("precomputed")
    operands["logits_teacher"] = None
    operands["teacher_hidden"] = None
    operands["teacher_weight"] = None
    if orphan == "teacher_hidden":
        operands["teacher_hidden"] = torch.randn(4, 7)
    else:
        operands["teacher_weight"] = torch.randn(11, 7)
    with pytest.raises(ValueError, match="requires logits_teacher|teacher_hidden"):
        _invoke("precomputed", operands)


@pytest.mark.parametrize("representation", ["logits", "hidden", "weight"])
def test_precomputed_rejects_teacher_tensors_requiring_grad(
    monkeypatch,
    representation,
):
    _enable_wrapper_validation(monkeypatch, "precomputed")
    operands = _cpu_operands("precomputed")
    if representation == "logits":
        operands["logits_teacher"] = torch.randn(4, 11, requires_grad=True)
    else:
        operands["logits_teacher"] = None
        operands["teacher_hidden"] = torch.randn(
            4,
            7,
            requires_grad=representation == "hidden",
        )
        operands["teacher_weight"] = torch.randn(
            11,
            7,
            requires_grad=representation == "weight",
        )
    with pytest.raises(ValueError, match="without gradients|must not require"):
        _invoke("precomputed", operands)


def test_precomputed_rejects_nonzero_teacher_ce(monkeypatch):
    _enable_wrapper_validation(monkeypatch, "precomputed")
    with pytest.raises(ValueError, match="teacher_ce_weight=0.0"):
        _invoke(
            "precomputed",
            _cpu_operands("precomputed"),
            teacher_ce_weight=0.5,
        )
