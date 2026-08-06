"""CUDA contracts for in-place logits-gradient kernel outputs."""

# ruff: noqa: E402

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pytest
from utils.env import pytest_skip_if_no_cuda_kernel

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_kernel.ops._kernels.precomputed.aligned_span import (
    _aligned_span_grad_fwdbwd,
    _aligned_span_stats_fwdbwd,
)
from orda_kernel.ops._kernels.precomputed.tokenwise import (
    _precomputed_student_fwdbwd as _precomputed,
)
from orda_kernel.ops._kernels.separate.cotrained import (
    _separate_cotrained_fwdbwd as _separate_cotrained,
)
from orda_kernel.ops._kernels.separate.frozen import (
    _separate_frozen_fwdbwd as _separate_frozen,
)
from orda_kernel.ops._kernels.separate.trained import (
    _separate_trained_fwdbwd as _separate_trained,
)
from orda_kernel.ops._kernels.tied.cotrained import (
    _tied_cotrained_fwdbwd as _tied_cotrained,
)
from orda_kernel.ops._kernels.tied.frozen import (
    _tied_frozen_fwdbwd as _tied_frozen,
)
from orda_kernel.ops._kernels.tied.trained import (
    _tied_trained_fwdbwd as _tied_trained,
)

_IGNORE_INDEX = -100
_ROWS = 4
_VOCAB = 129


class _TokenwiseCase(NamedTuple):
    name: str
    kernel: Callable[..., None]
    layout: str
    writes_teacher: bool
    kl_weight: float
    divergence: str


_SEPARATE_COTRAINED = _TokenwiseCase(
    "separate-cotrained-mutual",
    _separate_cotrained,
    "split",
    True,
    0.35,
    "mutual",
)
_PRECOMPUTED_TOKENWISE = _TokenwiseCase(
    "precomputed-tokenwise-forward",
    _precomputed,
    "split",
    False,
    0.35,
    "forward",
)
_TOKENWISE_CASES = (
    _TokenwiseCase(
        "separate-frozen-ce",
        _separate_frozen,
        "split",
        False,
        0.0,
        "forward",
    ),
    _TokenwiseCase(
        "separate-frozen-forward",
        _separate_frozen,
        "split",
        False,
        0.35,
        "forward",
    ),
    _TokenwiseCase(
        "separate-trained-forward",
        _separate_trained,
        "split",
        True,
        0.35,
        "forward",
    ),
    _SEPARATE_COTRAINED,
    _TokenwiseCase(
        "tied-frozen-forward",
        _tied_frozen,
        "tied",
        False,
        0.35,
        "forward",
    ),
    _TokenwiseCase(
        "tied-trained-forward",
        _tied_trained,
        "tied",
        True,
        0.35,
        "forward",
    ),
    _TokenwiseCase(
        "tied-cotrained-mutual",
        _tied_cotrained,
        "tied",
        True,
        0.35,
        "mutual",
    ),
    _PRECOMPUTED_TOKENWISE,
)


# ── Contract launch adapters ─────────────────────────────────────────────────
def _mixed_labels():
    return torch.tensor(
        [0, _IGNORE_INDEX, 2, _IGNORE_INDEX],
        dtype=torch.long,
        device="cuda",
    )


def _make_logits():
    values = torch.arange(_ROWS * _VOCAB, dtype=torch.float32, device="cuda")
    student = ((values.remainder(37) - 18.0) / 9.0).reshape(_ROWS, _VOCAB)
    teacher = ((values.mul(7).remainder(41) - 20.0) / 8.0).reshape(
        _ROWS,
        _VOCAB,
    )
    return student.contiguous(), teacher.contiguous()


def _loss_buffers(count: int):
    return tuple(
        torch.empty(_ROWS, dtype=torch.float32, device="cuda") for _ in range(count)
    )


def _run_tokenwise_contract(
    case: _TokenwiseCase,
    labels,
    *,
    gradient_materialization_factor=None,
):
    student, teacher = _make_logits()
    student_loss, teacher_loss, kl_loss = _loss_buffers(3)
    kwargs = {
        "kl_weight": case.kl_weight,
        "kl_temperature": 1.0,
        "kl_divergence": case.divergence,
        "ignore_index": _IGNORE_INDEX,
        "student_scale": 0.8,
        "max_fused_size": 256,
        "gradient_materialization_factor": gradient_materialization_factor,
    }
    if case.writes_teacher:
        kwargs["teacher_scale"] = 0.6

    if case.layout == "split":
        if case.writes_teacher:
            case.kernel(
                student,
                teacher,
                labels,
                student_loss,
                teacher_loss,
                kl_loss,
                **kwargs,
            )
        else:
            case.kernel(
                student,
                teacher,
                labels,
                student_loss,
                kl_loss,
                **kwargs,
            )
        return (
            student,
            teacher if case.writes_teacher else None,
            student_loss,
            teacher_loss if case.writes_teacher else None,
            kl_loss,
        )

    if case.layout != "tied":
        raise AssertionError(f"unknown contract layout {case.layout!r}")
    stacked = torch.cat((student, teacher))
    if case.writes_teacher:
        case.kernel(
            stacked,
            labels,
            student_loss,
            kl_loss,
            teacher_loss=teacher_loss,
            **kwargs,
        )
    else:
        case.kernel(
            stacked,
            labels,
            student_loss,
            kl_loss,
            **kwargs,
        )
    return (
        stacked[:_ROWS],
        stacked[_ROWS:] if case.writes_teacher else None,
        student_loss,
        teacher_loss if case.writes_teacher else None,
        kl_loss,
    )


def _assert_raw_gradient_contract(student, teacher, labels, *, require_active=True):
    ignored = labels == _IGNORE_INDEX
    for gradient in (student, teacher):
        if gradient is None:
            continue
        assert torch.equal(gradient[ignored], torch.zeros_like(gradient[ignored]))
        if require_active:
            active_gradient = gradient[~ignored]
            assert torch.isfinite(active_gradient).all()
            assert torch.count_nonzero(active_gradient) > 0


def _assert_loss_contract(losses, labels, *, require_active=True):
    ignored = labels == _IGNORE_INDEX
    for loss in losses:
        if loss is None:
            continue
        assert torch.equal(loss[ignored], torch.zeros_like(loss[ignored]))
        if require_active:
            assert torch.isfinite(loss[~ignored]).all()


# ── Raw-buffer contracts ─────────────────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.parametrize("case", _TOKENWISE_CASES, ids=lambda case: case.name)
def test_tokenwise_kernels_zero_inactive_dlogits(case):
    labels = _mixed_labels()
    student, teacher, student_loss, teacher_loss, kl_loss = _run_tokenwise_contract(
        case,
        labels,
    )

    _assert_raw_gradient_contract(student, teacher, labels)
    _assert_loss_contract((student_loss, teacher_loss, kl_loss), labels)


@pytest.mark.gpu
@pytest.mark.parametrize("all_ignored", (False, True), ids=("mixed", "all-ignored"))
def test_aligned_span_kernel_zeroes_inactive_dlogits(all_ignored):
    labels = (
        torch.full(
            (_ROWS,),
            _IGNORE_INDEX,
            dtype=torch.long,
            device="cuda",
        )
        if all_ignored
        else _mixed_labels()
    )
    student, _ = _make_logits()
    lse = torch.logsumexp(student, dim=-1)
    row_coefficients = torch.tensor(
        [0.75, 1.0, -0.5, 1.0],
        dtype=torch.float32,
        device="cuda",
    )

    _aligned_span_grad_fwdbwd(
        student,
        labels,
        lse,
        row_coefficients,
        ignore_index=_IGNORE_INDEX,
        max_fused_size=256,
    )

    _assert_raw_gradient_contract(
        student,
        None,
        labels,
        require_active=not all_ignored,
    )


@pytest.mark.gpu
def test_aligned_stats_preserves_logits_and_recompute_matches_reuse():
    torch.manual_seed(929)
    hidden = torch.randn(5, 4, device="cuda", dtype=torch.float16)
    weight = torch.randn(13, 4, device="cuda", dtype=torch.float16)
    labels = torch.tensor([0, 2, -100, 5, 7], device="cuda")
    logits = hidden @ weight.t()
    original = logits.clone()
    nll = torch.empty(5, device="cuda", dtype=torch.float32)
    lse = torch.empty_like(nll)
    _aligned_span_stats_fwdbwd(
        logits,
        labels,
        nll,
        lse,
        block_size=32,
        num_warps=4,
    )
    assert torch.equal(logits, original)
    ignored = labels == _IGNORE_INDEX
    assert torch.equal(nll[ignored], torch.zeros_like(nll[ignored]))
    assert torch.equal(lse[ignored], torch.zeros_like(lse[ignored]))

    coefficients = torch.tensor(
        [0.5, -0.25, 0.0, 0.75, -0.5],
        device="cuda",
        dtype=torch.float32,
    )
    reused = logits
    recomputed = hidden @ weight.t()
    _aligned_span_grad_fwdbwd(
        reused,
        labels,
        lse,
        coefficients,
        block_size=32,
        num_warps=4,
    )
    _aligned_span_grad_fwdbwd(
        recomputed,
        labels,
        lse,
        coefficients,
        block_size=32,
        num_warps=4,
    )
    assert torch.equal(reused, recomputed)
    assert torch.count_nonzero(reused[2]) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("case", _TOKENWISE_CASES, ids=lambda case: case.name)
def test_tokenwise_kernels_zero_all_ignored_outputs(case):
    labels = torch.full(
        (_ROWS,),
        _IGNORE_INDEX,
        dtype=torch.long,
        device="cuda",
    )
    student, teacher, student_loss, teacher_loss, kl_loss = _run_tokenwise_contract(
        case,
        labels,
    )

    _assert_raw_gradient_contract(
        student,
        teacher,
        labels,
        require_active=False,
    )
    _assert_loss_contract(
        (student_loss, teacher_loss, kl_loss),
        labels,
        require_active=False,
    )


@pytest.mark.gpu
def test_materialization_factor_preserves_zero_inactive_dlogits():
    labels = _mixed_labels()
    unscaled, _, *_ = _run_tokenwise_contract(
        _PRECOMPUTED_TOKENWISE,
        labels,
    )
    factor = torch.tensor(7.0, dtype=torch.float32, device="cuda")
    scaled, _, *_ = _run_tokenwise_contract(
        _PRECOMPUTED_TOKENWISE,
        labels,
        gradient_materialization_factor=factor,
    )

    _assert_raw_gradient_contract(scaled, None, labels)
    active = labels != _IGNORE_INDEX
    torch.testing.assert_close(
        scaled[active],
        unscaled[active] * factor,
        rtol=1e-5,
        atol=1e-6,
    )
