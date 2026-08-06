"""Low-level precomputed cached-logit dtype contracts."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

import orda_kernel as ock
from orda_kernel import _runtime as runtime
from orda_kernel.ops import precomputed_teacher
from orda_kernel.ops._kernels.precomputed import tokenwise as precomputed_kernels


# ── CUDA-like validation stubs ───────────────────────────────────────────────
class _ValidMask:
    def __and__(self, _other):
        return self

    def __or__(self, _other):
        return self

    def __invert__(self):
        return self

    def all(self) -> bool:
        return True


def _cuda_tensor_stub(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    requires_grad: bool = False,
) -> MagicMock:
    tensor = MagicMock(spec=torch.Tensor)
    tensor.shape = shape
    tensor.ndim = len(shape)
    tensor.dtype = dtype
    tensor.device = torch.device("cuda")
    tensor.layout = torch.strided
    tensor.requires_grad = requires_grad
    tensor.is_floating_point.return_value = dtype.is_floating_point
    tensor.numel.return_value = math.prod(shape)
    tensor.amax.return_value = torch.tensor(0.0)
    valid_mask = _ValidMask()
    tensor.__eq__.return_value = valid_mask
    tensor.__ge__.return_value = valid_mask
    tensor.__lt__.return_value = valid_mask
    return tensor


# ── Public direct wrapper ────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cached_dtype",
    [torch.float16, torch.bfloat16, torch.float64],
)
def test_direct_precomputed_plan_dtype_ignores_cached_logits_dtype(
    monkeypatch,
    cached_dtype: torch.dtype,
):
    """Cached logits are source data, not inputs to the student GEMM policy."""
    student_hidden = _cuda_tensor_stub((5, 3), torch.float32)
    weight = _cuda_tensor_stub((7, 3), torch.float32)
    labels = _cuda_tensor_stub((5,), torch.long)
    logits_teacher = _cuda_tensor_stub((5, 7), cached_dtype)
    captured: dict[str, object] = {}
    marker = (object(), object(), object(), object())

    monkeypatch.setattr(precomputed_teacher, "HAS_TRITON", True)

    def build_plan(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(precomputed_teacher, "build_execution_plan", build_plan)
    monkeypatch.setattr(
        precomputed_teacher,
        "execute_plan",
        lambda *_args, **_kwargs: marker,
    )

    result = precomputed_teacher.precomputed_distillation_loss(
        student_hidden,
        weight,
        logits_teacher=logits_teacher,
        labels=labels,
        kl_weight=0.5,
    )

    assert result is marker
    assert captured["compute_dtype"] is torch.float32


# ── Unplanned Function fallback ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "cached_dtype",
    [torch.float16, torch.bfloat16, torch.float64],
)
def test_unplanned_precomputed_logits_use_student_model_dtype(
    cached_dtype: torch.dtype,
):
    student_hidden = torch.randn(5, 3, dtype=torch.float32)
    weight = torch.randn(7, 3, dtype=torch.float32)
    logits_teacher = torch.randn(5, 7, dtype=cached_dtype)

    compute_dtype = precomputed_kernels._resolve_precomputed_compute_dtype(
        student_hidden=student_hidden,
        weight=weight,
        logits_teacher=logits_teacher,
        teacher_hidden=None,
        teacher_weight=None,
        compute_kl=True,
    )

    assert compute_dtype is torch.float32


def test_unplanned_precomputed_logits_follow_the_model_autocast_policy(monkeypatch):
    student_hidden = torch.randn(5, 3, dtype=torch.float32)
    weight = torch.randn(7, 3, dtype=torch.float32)
    logits_teacher = torch.randn(5, 7, dtype=torch.float64)
    monkeypatch.setattr(
        runtime,
        "_active_autocast_dtype",
        lambda _device: torch.bfloat16,
    )

    compute_dtype = precomputed_kernels._resolve_precomputed_compute_dtype(
        student_hidden=student_hidden,
        weight=weight,
        logits_teacher=logits_teacher,
        teacher_hidden=None,
        teacher_weight=None,
        compute_kl=True,
    )

    assert compute_dtype is torch.bfloat16


def test_cached_logits_conversion_reuses_a_chunk_bounded_scratch():
    source = torch.arange(35, dtype=torch.float64).reshape(5, 7)
    get_chunk = precomputed_kernels._cached_logits_chunk_getter(
        source,
        torch.float32,
        chunk_size=2,
    )

    chunks = []
    pointers = []
    for start, end in ((0, 2), (2, 4), (4, 5)):
        chunk = get_chunk(start, end)
        chunks.append(chunk.clone())
        pointers.append(chunk.data_ptr())
        assert chunk.dtype is torch.float32
        assert chunk.shape == (end - start, source.shape[1])

    assert pointers == [pointers[0]] * len(pointers)
    for expected, actual in zip(
        (source[0:2], source[2:4], source[4:5]),
        chunks,
        strict=True,
    ):
        torch.testing.assert_close(
            torch.softmax(actual, dim=-1),
            torch.softmax(expected, dim=-1).float(),
        )


def test_cached_logits_are_centered_before_narrowing_cast():
    source = torch.tensor(
        [[1.0e39, 0.0], [float("-inf"), float("-inf")]],
        dtype=torch.float64,
    )
    get_chunk = precomputed_kernels._cached_logits_chunk_getter(
        source,
        torch.float32,
        chunk_size=2,
    )

    chunk = get_chunk(0, 2)

    assert chunk[0, 0].item() == 0.0
    assert chunk[0, 1].item() == torch.finfo(torch.float32).min
    assert torch.isneginf(chunk[1]).all()
    assert not torch.isnan(chunk).any()


# ── Explicit full-cache preparation ──────────────────────────────────────────
def test_prepare_logits_reuses_an_already_prepared_cache():
    teacher = ock.PrecomputedTeacher(logits=torch.randn(5, 7))

    assert teacher.prepare_logits(torch.float32) is teacher


def test_prepare_logits_matches_runtime_cast_and_preserves_metadata():
    source = torch.tensor(
        [[1.0e39, 0.0], [float("-inf"), float("-inf")]],
        dtype=torch.float64,
    )
    teacher = ock.PrecomputedTeacher(
        logits=source,
        logits_temperature=2.0,
    )

    prepared = teacher.prepare_logits(torch.float32)

    assert prepared is not teacher
    assert prepared.logits is not None
    assert prepared.logits.dtype is torch.float32
    assert prepared.logits.is_contiguous()
    assert prepared.logits_temperature == 2.0
    torch.testing.assert_close(
        prepared.logits,
        runtime.cast_logits_for_compute(source, torch.float32),
    )


def test_prepare_logits_owns_an_independent_validated_tensor():
    source = torch.randn(5, 7, dtype=torch.float64)
    teacher = ock.PrecomputedTeacher(logits=source)
    prepared = teacher.prepare_logits(torch.float32)
    assert prepared.logits is not None

    source.add_(1.0)

    with pytest.raises(ValueError, match="modified after validation"):
        teacher.prepare_logits(torch.float32)
    prepared._validate_cache_unchanged()

    prepared.logits.add_(1.0)
    with pytest.raises(ValueError, match="modified after validation"):
        prepared._validate_cache_unchanged()


def test_prepare_logits_materializes_noncontiguous_matching_dtype():
    source = torch.randn(5, 14)[:, ::2]
    assert not source.is_contiguous()
    teacher = ock.PrecomputedTeacher(logits=source)

    prepared = teacher.prepare_logits(torch.float32)

    assert prepared is not teacher
    assert prepared.logits is not None and prepared.logits.is_contiguous()
    torch.testing.assert_close(prepared.logits, source)


def test_prepare_logits_keeps_mutation_checks_when_called_in_inference_mode():
    teacher = ock.PrecomputedTeacher(logits=torch.randn(5, 7, dtype=torch.float64))

    with torch.inference_mode():
        prepared = teacher.prepare_logits(torch.float32)

    assert prepared.logits is not None
    assert not prepared.logits.is_inference()
    prepared._validate_cache_unchanged()


def test_prepare_logits_restores_version_checks_for_a_trusted_inference_cache():
    with torch.inference_mode():
        source = torch.randn(5, 7, dtype=torch.float64)
    teacher = ock.PrecomputedTeacher(
        logits=source,
        trust_cache_immutability=True,
    )

    with torch.inference_mode():
        prepared = teacher.prepare_logits(torch.float32)

    assert prepared.logits is not None
    assert prepared.trust_cache_immutability
    assert not prepared.logits.is_inference()
    prepared.logits.add_(1.0)
    with pytest.raises(ValueError, match="modified after validation"):
        prepared._validate_cache_unchanged()


def test_prepared_cache_matches_runtime_conversion_in_public_loss():
    student_hidden = torch.randn(5, 3)
    weight = torch.randn(7, 3)
    labels = torch.tensor([0, 1, -100, 3, 4])
    teacher = ock.PrecomputedTeacher(logits=torch.randn(5, 7, dtype=torch.float64))
    prepared = teacher.prepare_logits(torch.float32)

    def run(candidate: ock.PrecomputedTeacher):
        return ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            candidate,
            student_ce_weight=0.0,
            kl_weight=1.0,
            reduction="sum",
            backend="torch",
        )

    expected = run(teacher)
    actual = run(prepared)

    for expected_component, actual_component in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_component, expected_component)


@pytest.mark.parametrize(
    ("dtype", "error", "match"),
    [
        ("float32", TypeError, "must be a torch.dtype"),
        (torch.int64, ValueError, "must be one of"),
    ],
)
def test_prepare_logits_rejects_invalid_compute_dtype(dtype, error, match):
    teacher = ock.PrecomputedTeacher(logits=torch.randn(5, 7))

    with pytest.raises(error, match=match):
        teacher.prepare_logits(dtype)


def test_prepare_logits_rejects_hidden_weight_representation():
    teacher = ock.PrecomputedTeacher(
        teacher_hidden=torch.randn(5, 3),
        teacher_weight=torch.randn(7, 3),
    )

    with pytest.raises(ValueError, match="requires the logits representation"):
        teacher.prepare_logits(torch.float32)


def test_unplanned_hidden_weight_representation_keeps_model_dtype_strict():
    student_hidden = torch.randn(5, 3, dtype=torch.float32)
    weight = torch.randn(7, 3, dtype=torch.float32)
    teacher_hidden = torch.randn(5, 4, dtype=torch.float32)
    teacher_weight = torch.randn(7, 4, dtype=torch.float16)

    with pytest.raises(ValueError, match="same dtype"):
        precomputed_kernels._resolve_precomputed_compute_dtype(
            student_hidden=student_hidden,
            weight=weight,
            logits_teacher=None,
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
            compute_kl=True,
        )
