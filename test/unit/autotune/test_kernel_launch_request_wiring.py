"""CPU regression tests for Triton launch-request construction.

These stop immediately after a kernel wrapper constructs its
``FullBatchLaunchRequest``.  They therefore cover the execution-plan wiring
without requiring a CUDA/HIP device or launching Triton.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch

from orda_kernel import ChunkSize
from orda_kernel._autotune import (
    FullBatchLaunchPlan,
    FullBatchLaunchShape,
    LaunchConfig,
)
from orda_kernel._planning.models import TEACHER_GRAD_NONE, ChunkAttempt
from orda_kernel._planning.policy import (
    build_aligned_span_execution_plan,
    build_execution_plan,
)
from orda_kernel.ops._common import grad_accum_dtype_key
from orda_kernel.ops._kernels.precomputed import aligned_span, tokenwise
from orda_kernel.ops._kernels.separate import cotrained as separate_cotrained
from orda_kernel.ops._kernels.separate import frozen as separate_frozen
from orda_kernel.ops._kernels.separate import trained as separate_trained
from orda_kernel.ops._kernels.tied import cotrained as tied_cotrained
from orda_kernel.ops._kernels.tied import frozen as tied_frozen
from orda_kernel.ops._kernels.tied import trained as tied_trained

_ATTEMPT = ChunkAttempt(chunk_size=2, num_chunks=1)
_ACTIVE_ACCUMULATOR_KEY = grad_accum_dtype_key(torch.float32)


class _RequestCaptured(Exception):
    """Internal sentinel raised before a CUDA/HIP launch can happen."""


def _capture_request(
    monkeypatch: pytest.MonkeyPatch,
    module,
    invoke: Callable[[], object],
):
    captured = []

    def capture(request, *, benchmark_factory):
        captured.append(request)
        raise _RequestCaptured

    monkeypatch.setattr(module, "select_full_batch_launch_plan", capture)
    with pytest.raises(_RequestCaptured):
        invoke()
    assert len(captured) == 1
    return captured[0]


def _execution_plan(
    *,
    teacher_mode: str,
    path: str,
    kl_variant: str | None = None,
    kl_alpha: float | None = None,
    autotune=False,
):
    options = {
        "frozen": {
            "teacher_ce_weight": 0.0,
            "kl_weight": 0.0,
            "kl_variant": "forward",
        },
        "trained": {
            "teacher_ce_weight": 1.0,
            "kl_weight": 0.0,
            "kl_variant": "forward",
        },
        "cotrained": {
            "teacher_ce_weight": 1.0,
            "kl_weight": 0.2,
            "kl_variant": "mutual",
        },
    }[path]
    if kl_variant is not None:
        options.update(kl_weight=0.2, kl_variant=kl_variant, kl_alpha=kl_alpha)
    return build_execution_plan(
        backend="triton",
        teacher_mode=teacher_mode,
        compute_dtype=torch.bfloat16,
        bt=2,
        vocab_size=3,
        student_ce_weight=1.0,
        kl_temperature=1.0,
        ignore_index=-100,
        reduction="mean",
        chunk_size=ChunkSize.dynamic(),
        num_chunks=1,
        max_chunks=1,
        max_fused_size=16,
        autotune=autotune,
        grad_weight_accumulation="auto",
        **options,
    )


def _tied_invoke(
    function,
    plan,
    *,
    weight_requires_grad: bool,
    student_hidden_requires_grad: bool = False,
    teacher_hidden_requires_grad: bool = False,
    gradient_materialization_scale=None,
    attempt: ChunkAttempt = _ATTEMPT,
):
    student_hidden = torch.randn(2, 2, dtype=torch.bfloat16).requires_grad_(
        student_hidden_requires_grad
    )
    teacher_hidden = torch.randn(2, 2, dtype=torch.bfloat16).requires_grad_(
        teacher_hidden_requires_grad
    )
    weight = torch.randn(3, 2, dtype=torch.float16).requires_grad_(weight_requires_grad)
    labels = torch.tensor([0, 1])
    return lambda: function.apply(
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        gradient_materialization_scale,
        plan,
        attempt,
    )


def _precomputed_invoke(
    plan,
    *,
    weight_requires_grad: bool,
    student_hidden_requires_grad: bool = False,
    gradient_materialization_scale=None,
    attempt: ChunkAttempt = _ATTEMPT,
):
    student_hidden = torch.randn(2, 2, dtype=torch.bfloat16).requires_grad_(
        student_hidden_requires_grad
    )
    weight = torch.randn(3, 2, dtype=torch.float16).requires_grad_(weight_requires_grad)
    logits_teacher = torch.randn(2, 3, dtype=torch.bfloat16)
    labels = torch.tensor([0, 1])
    return lambda: tokenwise.PrecomputedCEFunction.apply(
        student_hidden,
        weight,
        logits_teacher,
        labels,
        None,
        None,
        gradient_materialization_scale,
        plan,
        attempt,
    )


def _separate_invoke(
    function,
    plan,
    *,
    weight_requires_grad: bool,
    teacher_weight_requires_grad: bool,
    student_hidden_requires_grad: bool = False,
    teacher_hidden_requires_grad: bool = False,
    gradient_materialization_scale=None,
    attempt: ChunkAttempt = _ATTEMPT,
):
    student_hidden = torch.randn(2, 2, dtype=torch.bfloat16).requires_grad_(
        student_hidden_requires_grad
    )
    teacher_hidden = torch.randn(2, 2, dtype=torch.bfloat16).requires_grad_(
        teacher_hidden_requires_grad
    )
    weight = torch.randn(3, 2, dtype=torch.float16).requires_grad_(weight_requires_grad)
    teacher_weight = torch.randn(3, 2, dtype=torch.float16).requires_grad_(
        teacher_weight_requires_grad
    )
    labels = torch.tensor([0, 1])
    return lambda: function.apply(
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        gradient_materialization_scale,
        plan,
        attempt,
    )


def _student_cases():
    return (
        pytest.param(
            tied_frozen,
            _tied_invoke,
            tied_frozen.TiedFrozenCEFunction,
            _execution_plan(teacher_mode="tied", path="frozen"),
            id="tied-frozen",
        ),
        pytest.param(
            tied_trained,
            _tied_invoke,
            tied_trained.TiedTrainedCEFunction,
            _execution_plan(teacher_mode="tied", path="trained"),
            id="tied-trained",
        ),
        pytest.param(
            tied_cotrained,
            _tied_invoke,
            tied_cotrained.TiedCotrainedCEFunction,
            _execution_plan(teacher_mode="tied", path="cotrained"),
            id="tied-cotrained",
        ),
        pytest.param(
            tokenwise,
            _precomputed_invoke,
            None,
            _execution_plan(teacher_mode="precomputed", path="frozen"),
            id="precomputed",
        ),
        pytest.param(
            separate_frozen,
            _separate_invoke,
            separate_frozen.SeparateFrozenCEFunction,
            _execution_plan(teacher_mode="separate", path="frozen"),
            id="separate-frozen",
        ),
        pytest.param(
            separate_trained,
            _separate_invoke,
            separate_trained.SeparateTrainedCEFunction,
            _execution_plan(teacher_mode="separate", path="trained"),
            id="separate-trained",
        ),
        pytest.param(
            separate_cotrained,
            _separate_invoke,
            separate_cotrained.SeparateCotrainedCEFunction,
            _execution_plan(teacher_mode="separate", path="cotrained"),
            id="separate-cotrained",
        ),
    )


def _student_invoke(
    invoker,
    function,
    plan,
    *,
    weight_requires_grad: bool,
    student_hidden_requires_grad: bool = False,
    teacher_hidden_requires_grad: bool = False,
    teacher_weight_requires_grad: bool = False,
    gradient_materialization_scale=None,
    attempt: ChunkAttempt = _ATTEMPT,
):
    common = {
        "weight_requires_grad": weight_requires_grad,
        "student_hidden_requires_grad": student_hidden_requires_grad,
        "gradient_materialization_scale": gradient_materialization_scale,
        "attempt": attempt,
    }
    if invoker is _precomputed_invoke:
        return invoker(plan, **common)
    if invoker is _separate_invoke:
        return invoker(
            function,
            plan,
            teacher_hidden_requires_grad=teacher_hidden_requires_grad,
            teacher_weight_requires_grad=teacher_weight_requires_grad,
            **common,
        )
    return invoker(
        function,
        plan,
        teacher_hidden_requires_grad=teacher_hidden_requires_grad,
        **common,
    )


def _capture_runtime_and_benchmark_calls(
    monkeypatch: pytest.MonkeyPatch,
    module,
    wrapper_name: str,
    invoke: Callable[[], object],
):
    wrapper_calls = []
    accumulation_calls = []
    benchmark_wrapper_calls = None
    benchmark_accumulation_calls = None
    captured_request = None
    real_accumulate = module.accumulate_weight_gradient_

    def capture_wrapper(*_args, **kwargs):
        wrapper_calls.append(dict(kwargs))

    def capture_accumulation(
        destination,
        dlogits,
        hidden,
        *,
        row_active,
        overwrite,
        dlogits_zero_for_inactive=False,
    ):
        accumulation_calls.append(
            (
                tuple(destination.shape),
                destination.dtype,
                tuple(dlogits.shape),
                dlogits.dtype,
                tuple(hidden.shape),
                hidden.dtype,
                None if row_active is None else tuple(row_active.tolist()),
                overwrite,
                dlogits_zero_for_inactive,
            )
        )
        return real_accumulate(
            destination,
            dlogits,
            hidden,
            row_active=row_active,
            overwrite=overwrite,
            dlogits_zero_for_inactive=dlogits_zero_for_inactive,
        )

    def capture_selection(request, *, benchmark_factory):
        nonlocal benchmark_accumulation_calls
        nonlocal benchmark_wrapper_calls
        nonlocal captured_request
        captured_request = request
        task = benchmark_factory()
        if task.prepare_batch is not None:
            task.prepare_batch(1)
        task.run(16, 4)
        benchmark_wrapper_calls = tuple(wrapper_calls)
        benchmark_accumulation_calls = tuple(accumulation_calls)
        wrapper_calls.clear()
        accumulation_calls.clear()
        shape = FullBatchLaunchShape.from_attempt(
            bt=request.bt,
            chunk_rows=request.chunk_rows,
            chunk_dynamic=request.chunk_dynamic,
        )
        return FullBatchLaunchPlan(shape, LaunchConfig(16, 4))

    monkeypatch.setattr(module, wrapper_name, capture_wrapper)
    monkeypatch.setattr(module, "accumulate_weight_gradient_", capture_accumulation)
    monkeypatch.setattr(module, "select_full_batch_launch_plan", capture_selection)
    invoke()
    assert captured_request is not None
    assert benchmark_wrapper_calls is not None
    assert benchmark_accumulation_calls is not None
    return (
        captured_request,
        benchmark_wrapper_calls,
        tuple(wrapper_calls),
        benchmark_accumulation_calls,
        tuple(accumulation_calls),
    )


@pytest.mark.parametrize("weight_requires_grad", [False, True])
@pytest.mark.parametrize(("module", "invoker", "function", "plan"), _student_cases())
def test_tokenwise_launch_request_tracks_student_weight_gradient_activity(
    monkeypatch,
    module,
    invoker,
    function,
    plan,
    weight_requires_grad,
):
    request = _capture_request(
        monkeypatch,
        module,
        _student_invoke(
            invoker,
            function,
            plan,
            weight_requires_grad=weight_requires_grad,
        ),
    )

    assert request.grad_weight_accum_dtype == (
        _ACTIVE_ACCUMULATOR_KEY if weight_requires_grad else None
    )


@pytest.mark.parametrize("module, invoker, function, plan", _student_cases())
def test_benchmark_and_runtime_use_the_same_wrapper_launch_arguments(
    monkeypatch,
    module,
    invoker,
    function,
    plan,
):
    tuned_plan = replace(plan, autotune=True)
    teacher_grad_active = tuned_plan.teacher_grad_policy != TEACHER_GRAD_NONE
    wrapper_name = next(
        name
        for name in vars(module)
        if name.startswith("_") and name.endswith("_fwdbwd")
    )
    invoke = _student_invoke(
        invoker,
        function,
        tuned_plan,
        weight_requires_grad=True,
        student_hidden_requires_grad=True,
        teacher_hidden_requires_grad=teacher_grad_active,
        teacher_weight_requires_grad=(
            invoker is _separate_invoke and teacher_grad_active
        ),
        attempt=ChunkAttempt(chunk_size=1, num_chunks=2),
    )
    (
        request,
        benchmark_calls,
        runtime_calls,
        benchmark_accumulations,
        runtime_accumulations,
    ) = _capture_runtime_and_benchmark_calls(
        monkeypatch,
        module,
        wrapper_name,
        invoke,
    )

    assert len(benchmark_calls) == len(runtime_calls) == 2
    for benchmark_kwargs, runtime_kwargs in zip(
        benchmark_calls,
        runtime_calls,
        strict=True,
    ):
        assert benchmark_kwargs.keys() == runtime_kwargs.keys()
        for key in benchmark_kwargs:
            benchmark_value = benchmark_kwargs[key]
            runtime_value = runtime_kwargs[key]
            if isinstance(benchmark_value, torch.Tensor):
                assert isinstance(runtime_value, torch.Tensor), key
                assert benchmark_value.shape == runtime_value.shape, key
                assert benchmark_value.dtype == runtime_value.dtype, key
                assert benchmark_value.device == runtime_value.device, key
            else:
                assert benchmark_value == runtime_value, key
    assert benchmark_accumulations == runtime_accumulations
    assert request.compute_kl is tuned_plan.specialization.compute_kl
    assert request.compute_student_ce is (tuned_plan.student_ce_weight != 0.0)
    assert request.kl_variant == tuned_plan.specialization.kl_variant
    if module is tokenwise:
        assert request.pipeline_key == (
            "logits",
            "allow_teacher_neg_inf_support",
            "torch.bfloat16",
            "contiguous",
        )
        assert all(
            call["allow_teacher_neg_inf_support"] is True
            for call in (*benchmark_calls, *runtime_calls)
        )
    assert request.grad_student_hidden is True
    assert request.grad_teacher_hidden is teacher_grad_active
    assert request.grad_weight_accum_dtype == _ACTIVE_ACCUMULATOR_KEY
    assert request.grad_teacher_weight_accum_dtype == (
        _ACTIVE_ACCUMULATOR_KEY
        if invoker is _separate_invoke and teacher_grad_active
        else None
    )
    assert request.gradient_materialization_scaled is False


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize(("module", "invoker", "function", "plan"), _student_cases())
def test_tokenwise_launch_request_tracks_gradient_materialization(
    monkeypatch,
    module,
    invoker,
    function,
    plan,
    scaled,
):
    scale = torch.tensor(8192.0) if scaled else None
    request = _capture_request(
        monkeypatch,
        module,
        _student_invoke(
            invoker,
            function,
            plan,
            weight_requires_grad=True,
            gradient_materialization_scale=scale,
        ),
    )

    assert request.gradient_materialization_scaled is scaled


@pytest.mark.parametrize(
    ("module", "function", "plan"),
    (
        pytest.param(
            separate_trained,
            separate_trained.SeparateTrainedCEFunction,
            _execution_plan(teacher_mode="separate", path="trained"),
            id="separate-trained",
        ),
        pytest.param(
            separate_cotrained,
            separate_cotrained.SeparateCotrainedCEFunction,
            _execution_plan(teacher_mode="separate", path="cotrained"),
            id="separate-cotrained",
        ),
    ),
)
@pytest.mark.parametrize("teacher_weight_requires_grad", [False, True])
def test_separate_launch_request_tracks_teacher_weight_gradient_independently(
    monkeypatch,
    module,
    function,
    plan,
    teacher_weight_requires_grad,
):
    request = _capture_request(
        monkeypatch,
        module,
        _separate_invoke(
            function,
            plan,
            weight_requires_grad=False,
            teacher_weight_requires_grad=teacher_weight_requires_grad,
        ),
    )

    assert request.grad_weight_accum_dtype is None
    assert request.grad_teacher_weight_accum_dtype == (
        _ACTIVE_ACCUMULATOR_KEY if teacher_weight_requires_grad else None
    )


@pytest.mark.parametrize(
    ("kl_alpha", "expected"),
    ((0.875, True), (1.125, True), (1.5, False)),
)
def test_tokenwise_launch_request_tracks_renyi_near_one(
    monkeypatch,
    kl_alpha,
    expected,
):
    plan = _execution_plan(
        teacher_mode="precomputed",
        path="frozen",
        kl_variant="renyi",
        kl_alpha=kl_alpha,
    )
    request = _capture_request(
        monkeypatch,
        tokenwise,
        _precomputed_invoke(plan, weight_requires_grad=True),
    )

    assert request.renyi_near_one is expected


@pytest.mark.parametrize("weight_requires_grad", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_aligned_span_launch_request_resolves_and_keys_weight_gradient(
    monkeypatch,
    weight_requires_grad,
    scaled,
):
    plan = build_aligned_span_execution_plan(
        backend="triton",
        teacher_type="precomputed_span",
        compute_dtype=torch.bfloat16,
        bt=2,
        vocab_size=3,
        num_spans=1,
        student_ce_weight=1.0,
        span_kl_weight=1.0,
        ignore_index=-100,
        reduction="mean",
        objective="binary_kl",
        chunk_size=ChunkSize.dynamic(),
        num_chunks=1,
        max_chunks=1,
        max_fused_size=16,
        autotune=False,
        grad_weight_accumulation="auto",
    )
    student_hidden = torch.randn(2, 2, dtype=torch.bfloat16)
    weight = torch.randn(3, 2, dtype=torch.float16).requires_grad_(weight_requires_grad)
    labels = torch.tensor([0, 1])
    teacher_span_log_probs = torch.zeros(1)
    student_ranges = torch.tensor([[0, 2]], dtype=torch.int64)
    scale = torch.tensor(torch.finfo(torch.float32).tiny) if scaled else None
    prepared_scales = []
    original_prepare = aligned_span.prepare_gradient_materialization_factor

    def observe_prepare(*args):
        result = original_prepare(*args)
        prepared_scales.append(result[2])
        return result

    monkeypatch.setattr(
        aligned_span, "prepare_gradient_materialization_factor", observe_prepare
    )
    request = _capture_request(
        monkeypatch,
        aligned_span,
        lambda: aligned_span.PrecomputedAlignedSpanFunction.apply(
            student_hidden,
            weight,
            labels,
            teacher_span_log_probs,
            student_ranges,
            scale,
            plan,
            _ATTEMPT,
        ),
    )

    assert len(prepared_scales) == 1
    if scaled:
        torch.testing.assert_close(prepared_scales[0], torch.tensor(2.0))
    else:
        assert prepared_scales[0] is None
    assert request.grad_weight_accum_dtype == (
        _ACTIVE_ACCUMULATOR_KEY if weight_requires_grad else None
    )
    assert request.gradient_materialization_scaled is scaled


# ── Timed benchmark specialization ───────────────────────────────────────────
def test_scaled_tokenwise_benchmark_receives_materialization_factor(monkeypatch):
    observed = []

    def capture_factor(
        logits_student,
        _logits_teacher,
        _labels,
        student_loss,
        kl_loss,
        **kwargs,
    ):
        observed.append(kwargs["gradient_materialization_factor"])
        logits_student.zero_()
        student_loss.zero_()
        kl_loss.zero_()

    monkeypatch.setattr(tokenwise, "_precomputed_student_fwdbwd", capture_factor)
    scale = torch.tensor(8.0)
    task = tokenwise._make_precomputed_benchmark_task(
        student_hidden=torch.randn(2, 2),
        weight=torch.randn(3, 2),
        logits_teacher=torch.randn(2, 3),
        teacher_hidden=None,
        teacher_weight=None,
        labels=torch.tensor([0, 1]),
        compute_dtype=torch.float32,
        compute_kl=False,
        chunk_size=2,
        num_chunks=1,
        need_grad_student_hidden=False,
        need_grad_weight=False,
        grad_weight_accum_dtype=torch.float32,
        student_ce_weight=1.0,
        kl_weight=0.0,
        kl_temperature=1.0,
        kl_teacher_temperature=1.0,
        ignore_index=-100,
        reduction="mean",
        max_fused_size=16,
        kl_divergence="forward",
        allow_teacher_neg_inf_support=True,
        gradient_materialization_scale=scale,
    )

    task.run(16, 4)

    assert len(observed) == 1
    assert observed[0] is not None
    torch.testing.assert_close(observed[0], torch.ones(()))


def test_scaled_aligned_benchmark_receives_materialization_scale(monkeypatch):
    observed = []

    def capture_scale(**kwargs):
        observed.append(kwargs["gradient_materialization_scale"])
        zero = torch.zeros(())
        return zero, zero, zero, None, None

    monkeypatch.setattr(aligned_span, "_run_aligned_pipeline", capture_scale)
    plan = build_aligned_span_execution_plan(
        backend="triton",
        teacher_type="precomputed_span",
        compute_dtype=torch.float32,
        bt=2,
        vocab_size=3,
        num_spans=1,
        student_ce_weight=1.0,
        span_kl_weight=1.0,
        ignore_index=-100,
        reduction="mean",
        objective="binary_kl",
        chunk_size=ChunkSize.dynamic(),
        num_chunks=1,
        max_chunks=1,
        max_fused_size=16,
        autotune=True,
        grad_weight_accumulation="auto",
    )
    scale = torch.tensor(8.0)
    task = aligned_span._make_aligned_benchmark_task(
        student_hidden=torch.randn(2, 2),
        weight=torch.randn(3, 2),
        labels=torch.tensor([0, 1]),
        teacher_span_log_probs=torch.zeros(1),
        student_ranges=torch.tensor([[0, 2]], dtype=torch.int64),
        plan=plan,
        chunk_size=2,
        num_chunks=1,
        need_grad_student_hidden=False,
        need_grad_weight=False,
        gradient_materialization_scale=scale,
    )

    task.run(16, 4)

    assert len(observed) == 1
    assert observed[0] is not None
    torch.testing.assert_close(observed[0], torch.ones(()))
