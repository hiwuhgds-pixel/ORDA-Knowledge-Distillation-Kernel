from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from orda_kernel import (
    AlignedLikelihoodConfig,
    AlignedSpanPairs,
    ChunkSize,
    PrecomputedSpanTeacher,
    aligned_span_distillation_loss,
    build_precomputed_span_teacher,
)
from orda_kernel._autotune import LaunchConfig
from orda_kernel._execution import aligned_span as aligned_execution
from orda_kernel._execution import dispatcher
from orda_kernel._planning.policy import build_aligned_span_execution_plan
from orda_kernel.ops._kernels.precomputed import aligned_span as aligned_kernel
from orda_kernel.ops.precomputed_inputs._aligned_span_math import (
    MAX_SPAN_LOG_PROB,
    binary_span_objective,
    expand_span_values,
    span_log_probs_from_nll,
)


# ── Pipeline stage contracts ─────────────────────────────────────────────────
def _fake_stages(monkeypatch):
    stats_logits: list[torch.Tensor] = []
    grad_logits: list[torch.Tensor] = []

    def stats(logits, labels, nll_out, lse_out, *, ignore_index, **_kwargs):
        stats_logits.append(logits)
        logits32 = logits.float()
        lse = torch.logsumexp(logits32, dim=-1)
        valid = labels != ignore_index
        safe_labels = torch.where(valid, labels, labels.new_zeros(()))
        label_logits = logits32.gather(1, safe_labels[:, None]).squeeze(1)
        nll_out.copy_(torch.where(valid, lse - label_logits, lse.new_zeros(())))
        lse_out.copy_(torch.where(valid, lse, lse.new_zeros(())))

    def grad(logits, labels, lse, coefficients, *, ignore_index, **kwargs):
        grad_logits.append(logits)
        valid = labels != ignore_index
        safe_labels = torch.where(valid, labels, labels.new_zeros(()))
        probability = torch.exp(logits.float() - lse[:, None])
        target = torch.zeros_like(probability)
        target.scatter_(1, safe_labels[:, None], 1.0)
        dlogits = coefficients[:, None] * (probability - target)
        gradient_materialization_scale = kwargs.get("gradient_materialization_scale")
        if gradient_materialization_scale is not None:
            dlogits = dlogits * gradient_materialization_scale
        logits.copy_(torch.where(valid[:, None], dlogits, dlogits.new_zeros(())))

    monkeypatch.setattr(aligned_kernel, "_aligned_span_stats_fwdbwd", stats)
    monkeypatch.setattr(aligned_kernel, "_aligned_span_grad_fwdbwd", grad)
    return stats_logits, grad_logits


@pytest.mark.parametrize("objective", ["binary_kl", "binary_ce"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("num_chunks", [1, 3])
def test_two_pass_pipeline_matches_torch_reference(
    monkeypatch,
    objective,
    reduction,
    num_chunks,
):
    stats_logits, grad_logits = _fake_stages(monkeypatch)
    generator = torch.Generator().manual_seed(71)
    hidden = torch.randn(7, 4, generator=generator, requires_grad=True)
    weight = torch.randn(9, 4, generator=generator, requires_grad=True)
    labels = torch.tensor([0, 2, -100, 4, 1, 8, 3])
    ranges = torch.tensor([[0, 2], [2, 5], [4, 7]], dtype=torch.int64)
    teacher_log_probs = torch.tensor([-0.7, -1.1, -0.4], dtype=torch.float64)
    config = AlignedLikelihoodConfig(
        student_ce_weight=0.7,
        span_kl_weight=0.4,
        reduction=reduction,
        objective=objective,
    )
    plan = build_aligned_span_execution_plan(
        backend="triton",
        teacher_type="precomputed_span",
        compute_dtype=torch.float32,
        bt=hidden.shape[0],
        vocab_size=weight.shape[0],
        num_spans=ranges.shape[0],
        student_ce_weight=config.student_ce_weight,
        span_kl_weight=config.span_kl_weight,
        ignore_index=config.ignore_index,
        reduction=config.reduction,
        objective=config.objective,
        chunk_size=ChunkSize.dynamic(),
        num_chunks=num_chunks,
        max_chunks=None,
        max_fused_size=16,
        autotune=False,
        grad_weight_accumulation="auto",
    )
    chunk_size = (hidden.shape[0] + num_chunks - 1) // num_chunks
    actual = aligned_kernel._run_aligned_pipeline(
        student_hidden=hidden,
        weight=weight,
        labels=labels,
        teacher_span_log_probs=teacher_log_probs,
        student_ranges=ranges,
        plan=plan,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        launch_config=LaunchConfig(16, 4),
        need_grad_student_hidden=True,
        need_grad_weight=True,
    )

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    pairs = AlignedSpanPairs(ranges, ranges.clone())
    reference = aligned_span_distillation_loss(
        hidden_ref,
        weight_ref,
        labels,
        PrecomputedSpanTeacher(teacher_log_probs, teacher_ranges=pairs.teacher_ranges),
        pairs,
        config,
        backend="torch",
    )
    reference.loss.backward()

    loss, student_ce, span_div, grad_hidden, grad_weight = actual
    assert torch.allclose(loss, reference.loss, atol=2e-6, rtol=2e-6)
    assert torch.allclose(student_ce, reference.student_ce, atol=2e-6, rtol=2e-6)
    assert torch.allclose(span_div, reference.kl, atol=2e-6, rtol=2e-6)
    assert grad_hidden is not None and hidden_ref.grad is not None
    assert grad_weight is not None and weight_ref.grad is not None
    assert torch.allclose(grad_hidden, hidden_ref.grad, atol=3e-6, rtol=3e-6)
    assert torch.allclose(grad_weight, weight_ref.grad, atol=3e-6, rtol=3e-6)

    if num_chunks == 1:
        assert grad_logits[0] is stats_logits[0]
    else:
        assert all(
            grad_item is not stats_item
            for grad_item in grad_logits
            for stats_item in stats_logits
        )


def test_stats_stage_remains_live_when_student_ce_is_disabled(monkeypatch):
    _fake_stages(monkeypatch)
    hidden = torch.randn(4, 3)
    weight = torch.randn(6, 3)
    labels = torch.tensor([0, 1, 2, 3])
    ranges = torch.tensor([[0, 4]], dtype=torch.int64)
    plan = build_aligned_span_execution_plan(
        backend="triton",
        teacher_type="precomputed_span",
        compute_dtype=torch.float32,
        bt=4,
        vocab_size=6,
        num_spans=1,
        student_ce_weight=0.0,
        span_kl_weight=1.0,
        ignore_index=-100,
        reduction="mean",
        objective="binary_kl",
        chunk_size=ChunkSize.dynamic(),
        num_chunks=1,
        max_chunks=None,
        max_fused_size=16,
        autotune=False,
        grad_weight_accumulation="auto",
    )
    loss, student_ce, span_div, _, _ = aligned_kernel._run_aligned_pipeline(
        student_hidden=hidden,
        weight=weight,
        labels=labels,
        teacher_span_log_probs=torch.tensor([-0.8]),
        student_ranges=ranges,
        plan=plan,
        chunk_size=4,
        num_chunks=1,
        launch_config=LaunchConfig(16, 4),
        need_grad_student_hidden=False,
        need_grad_weight=False,
    )
    assert float(student_ce) == 0.0
    assert torch.isfinite(loss)
    assert float(span_div) != 0.0


def test_aligned_pipeline_scales_only_cached_gradients(monkeypatch):
    _fake_stages(monkeypatch)
    hidden = torch.randn(4, 3)
    weight = torch.randn(6, 3)
    labels = torch.tensor([0, 1, -100, 3])
    ranges = torch.tensor([[0, 4]], dtype=torch.int64)
    teacher = torch.tensor([-0.8])
    plan = _aligned_executor_plan(num_chunks=1)
    common = dict(
        student_hidden=hidden,
        weight=weight,
        labels=labels,
        teacher_span_log_probs=teacher,
        student_ranges=ranges,
        plan=plan,
        chunk_size=4,
        num_chunks=1,
        launch_config=LaunchConfig(16, 4),
        need_grad_student_hidden=True,
        need_grad_weight=True,
    )
    unscaled = aligned_kernel._run_aligned_pipeline(**common)
    materialization_scale = torch.tensor(128.0)
    scaled = aligned_kernel._run_aligned_pipeline(
        **common,
        gradient_materialization_scale=materialization_scale,
    )

    for actual, expected in zip(scaled[:3], unscaled[:3], strict=True):
        assert torch.equal(actual, expected)
    assert scaled[3] is not None and unscaled[3] is not None
    assert scaled[4] is not None and unscaled[4] is not None
    assert torch.allclose(scaled[3], unscaled[3] * materialization_scale)
    assert torch.allclose(scaled[4], unscaled[4] * materialization_scale)


def test_ce_only_pipeline_skips_span_reduction_workspace(monkeypatch):
    _fake_stages(monkeypatch)
    monkeypatch.setattr(
        aligned_kernel,
        "span_log_probs_from_nll",
        lambda *_args, **_kwargs: pytest.fail("span reduction must remain inactive"),
    )
    hidden = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(6, 3, requires_grad=True)
    labels = torch.tensor([0, 1, -100, 3])
    ranges = torch.tensor([[0, 4]], dtype=torch.int64)
    plan = build_aligned_span_execution_plan(
        backend="triton",
        teacher_type="precomputed_span",
        compute_dtype=torch.float32,
        bt=4,
        vocab_size=6,
        num_spans=1,
        student_ce_weight=0.7,
        span_kl_weight=0.0,
        ignore_index=-100,
        reduction="mean",
        objective="binary_kl",
        chunk_size=ChunkSize.dynamic(),
        num_chunks=1,
        max_chunks=None,
        max_fused_size=16,
        autotune=False,
        grad_weight_accumulation="auto",
    )
    loss, student_ce, span_div, grad_hidden, grad_weight = (
        aligned_kernel._run_aligned_pipeline(
            student_hidden=hidden,
            weight=weight,
            labels=labels,
            teacher_span_log_probs=torch.tensor([-0.8]),
            student_ranges=ranges,
            plan=plan,
            chunk_size=4,
            num_chunks=1,
            launch_config=LaunchConfig(16, 4),
            need_grad_student_hidden=True,
            need_grad_weight=True,
        )
    )

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    expected_ce = F.cross_entropy(
        hidden_ref @ weight_ref.t(),
        labels,
        ignore_index=-100,
    )
    (0.7 * expected_ce).backward()
    assert torch.allclose(student_ce, expected_ce)
    assert torch.allclose(loss, 0.7 * expected_ce)
    assert float(span_div) == 0.0
    assert torch.allclose(grad_hidden, hidden_ref.grad, atol=2e-6, rtol=2e-6)
    assert torch.allclose(grad_weight, weight_ref.grad, atol=2e-6, rtol=2e-6)


def test_binary_objectives_share_aligned_autotune_pipeline_identity():
    plans = [
        build_aligned_span_execution_plan(
            backend="triton",
            teacher_type="precomputed_span",
            compute_dtype=torch.float32,
            bt=4,
            vocab_size=6,
            num_spans=1,
            student_ce_weight=1.0,
            span_kl_weight=1.0,
            ignore_index=-100,
            reduction="mean",
            objective=objective,
            chunk_size=ChunkSize.dynamic(),
            num_chunks=1,
            max_chunks=None,
            max_fused_size=16,
            autotune=False,
            grad_weight_accumulation="auto",
        )
        for objective in ("binary_kl", "binary_ce")
    ]

    assert aligned_kernel._aligned_pipeline_key(plans[0]) == (
        "aligned_span",
        "span_active",
    )
    assert aligned_kernel._aligned_pipeline_key(plans[0]) == (
        aligned_kernel._aligned_pipeline_key(plans[1])
    )


# ── Executor recovery contracts ──────────────────────────────────────────────
def _aligned_executor_plan(*, num_chunks: int | None):
    return build_aligned_span_execution_plan(
        backend="triton",
        teacher_type="precomputed_span",
        compute_dtype=torch.float32,
        bt=4,
        vocab_size=6,
        num_spans=1,
        student_ce_weight=1.0,
        span_kl_weight=1.0,
        ignore_index=-100,
        reduction="mean",
        objective="binary_kl",
        chunk_size=ChunkSize.dynamic(),
        num_chunks=num_chunks,
        max_chunks=4,
        max_fused_size=16,
        autotune=False,
        grad_weight_accumulation="auto",
    )


def _aligned_executor_inputs():
    return (
        torch.randn(4, 3),
        torch.randn(6, 3),
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([-0.5]),
        torch.tensor([[0, 4]], dtype=torch.int64),
    )


def test_aligned_executor_retries_only_chunk_reducible_oom(monkeypatch):
    aligned_execution.clear_aligned_span_chunk_cache()
    attempts = []
    marker = tuple(torch.tensor(float(index)) for index in range(4))

    def fake_apply(_plan, attempt, *_args):
        attempts.append(attempt.num_chunks)
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
        return marker

    monkeypatch.setattr(aligned_execution, "_apply", fake_apply)
    result = aligned_execution.execute_aligned_span_plan(
        _aligned_executor_plan(num_chunks=None),
        *_aligned_executor_inputs(),
        _labels_prevalidated=True,
    )

    assert result == marker
    assert attempts == [1, 2]
    aligned_execution.clear_aligned_span_chunk_cache()


@pytest.mark.parametrize("num_chunks", [None, 1])
def test_aligned_executor_does_not_retry_fixed_workspace_oom(
    monkeypatch,
    num_chunks,
):
    aligned_execution.clear_aligned_span_chunk_cache()
    calls = 0

    def fake_apply(_plan, _attempt, *_args):
        nonlocal calls
        calls += 1
        raise aligned_execution.NonChunkableWorkspaceOOM(
            "synthetic fixed workspace OOM"
        )

    monkeypatch.setattr(aligned_execution, "_apply", fake_apply)
    monkeypatch.setattr(dispatcher, "recover_after_oom", lambda _device: None)
    with pytest.raises(
        RuntimeError,
        match=r"non-chunkable workspace.*splitting into more chunks cannot",
    ):
        aligned_execution.execute_aligned_span_plan(
            _aligned_executor_plan(num_chunks=num_chunks),
            *_aligned_executor_inputs(),
            _labels_prevalidated=True,
        )

    assert calls == 1
    aligned_execution.clear_aligned_span_chunk_cache()


def test_fake_stage_probability_matches_softmax(monkeypatch):
    _fake_stages(monkeypatch)
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    labels = torch.tensor([0])
    nll = torch.empty(1)
    lse = torch.empty(1)
    aligned_kernel._aligned_span_stats_fwdbwd(
        logits,
        labels,
        nll,
        lse,
        ignore_index=-100,
    )
    assert torch.allclose(torch.exp(logits - lse[:, None]), F.softmax(logits, dim=-1))


def test_span_cache_builder_matches_direct_teacher_likelihood():
    generator = torch.Generator().manual_seed(91)
    hidden = torch.randn(
        6, 3, generator=generator, dtype=torch.float64
    ).requires_grad_()
    weight = torch.randn(
        8, 3, generator=generator, dtype=torch.float64
    ).requires_grad_()
    labels = torch.tensor([0, 1, -100, 3, 4, 5])
    ranges = torch.tensor([[0, 2], [2, 5], [5, 6]], dtype=torch.int32)
    cache = build_precomputed_span_teacher(hidden, weight, labels, ranges)

    with torch.no_grad():
        active = labels != -100
        logits = torch.where(active[:, None], hidden, hidden.new_zeros(())) @ weight.t()
        safe_labels = torch.where(active, labels, labels.new_zeros(()))
        token_ll = (
            F.log_softmax(logits, dim=-1).gather(1, safe_labels[:, None]).squeeze(1)
        )
        token_ll = torch.where(active, token_ll, token_ll.new_zeros(()))
        expected = span_log_probs_from_nll(-token_ll, ranges)

    assert not cache.span_log_probs.requires_grad
    assert torch.allclose(cache.span_log_probs, expected)

    with pytest.raises(ValueError, match=r"no valid teacher tokens.*\[0\]"):
        build_precomputed_span_teacher(
            hidden,
            weight,
            torch.tensor([-100, -100, 2, 3, 4, 5]),
            torch.tensor([[0, 2], [2, 6]], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match=r"non-finite logits.*\[0\]"):
        build_precomputed_span_teacher(
            torch.tensor([[1.0e20]]),
            torch.tensor([[1.0e20]]),
            torch.tensor([0]),
            torch.tensor([[0, 1]], dtype=torch.int32),
        )


def test_span_cache_builder_preserves_float64_logits_beyond_float32_range():
    hidden = torch.tensor([[1.0e20], [1.0e20]], dtype=torch.float64)
    weight = torch.tensor([[1.0e20], [-1.0e20]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    ranges = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)

    cache = build_precomputed_span_teacher(hidden, weight, labels, ranges)

    logits = hidden @ weight.t()
    expected = F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    torch.testing.assert_close(cache.span_log_probs, expected)


def test_span_reduction_preserves_zero_support_without_poisoning_later_spans():
    ranges = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    result = span_log_probs_from_nll(torch.tensor([float("inf"), 1.25]), ranges)
    assert torch.isneginf(result[0])
    assert float(result[1]) == pytest.approx(-1.25)


# ── Span math contracts ──────────────────────────────────────────────────────
def test_difference_array_handles_overlap_nesting_and_duplicates():
    ranges = torch.tensor([[0, 4], [2, 6], [2, 6], [3, 5]], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0, -0.5, 3.0], dtype=torch.float64)
    actual = expand_span_values(values, ranges, n_rows=7)
    expected = torch.zeros(7, dtype=torch.float64)
    for value, (start, end) in zip(values, ranges.tolist(), strict=True):
        expected[start:end] += value

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("objective", ["binary_kl", "binary_ce"])
def test_span_clip_is_straight_through_and_keeps_mean_denominator(objective):
    raw = torch.tensor(
        [MAX_SPAN_LOG_PROB, MAX_SPAN_LOG_PROB + 1e-8],
        dtype=torch.float64,
        requires_grad=True,
    )
    terms = binary_span_objective(
        raw,
        torch.tensor([-1.0, -1.0], dtype=torch.float64),
        torch.tensor([True, True]),
        objective=objective,
        reduction="mean",
    )

    assert torch.equal(
        terms.student_log_probs,
        torch.full_like(raw, MAX_SPAN_LOG_PROB),
    )
    assert terms.gradient[0] != 0.0
    assert terms.gradient[1] == terms.gradient[0]
    assert terms.reduction_scale == 0.5
    terms.reduced_loss.backward()
    assert raw.grad is not None
    assert torch.allclose(
        raw.grad,
        terms.gradient.detach() * terms.reduction_scale,
    )


@pytest.mark.parametrize("objective", ["binary_kl", "binary_ce"])
def test_zero_student_span_support_does_not_create_nan(objective):
    raw = torch.tensor([float("-inf")], dtype=torch.float64, requires_grad=True)
    terms = binary_span_objective(
        raw,
        torch.tensor([-1.0], dtype=torch.float64),
        torch.tensor([True]),
        objective=objective,
        reduction="mean",
    )

    assert torch.isneginf(terms.student_log_probs).all()
    assert not torch.isnan(terms.reduced_loss)
    assert torch.isfinite(terms.gradient).all()
    terms.reduced_loss.backward()
    assert torch.isfinite(raw.grad).all()
