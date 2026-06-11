"""Validation matrix for teacher_mode + args combinations."""
from __future__ import annotations

import pytest

from tests.utils.env import pytest_skip_if_no_cuda_kernel

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_ce_kernel.ops.cross_entropy import distill_cross_entropy


def _make_simple(BT=8, H=16, V=64):
    hs = torch.randn(BT, H, device="cuda", dtype=torch.float16, requires_grad=True)
    ht = torch.randn(BT, H, device="cuda", dtype=torch.float16)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.randint(0, V, (BT,), device="cuda")
    return hs, ht, weight, target, BT, V


@pytest.mark.gpu
def test_tied_rejects_weight_teacher():
    hs, ht, w, target, BT, V = _make_simple()
    w_teacher = torch.randn(V, w.shape[1], device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="weight_teacher"):
        distill_cross_entropy(hs, ht, w, target, teacher_mode="tied", weight_teacher=w_teacher)


@pytest.mark.gpu
def test_tied_rejects_logits_teacher():
    hs, ht, w, target, BT, V = _make_simple()
    logits_t = torch.randn(BT, V, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="logits_teacher"):
        distill_cross_entropy(hs, ht, w, target, teacher_mode="tied", logits_teacher=logits_t)


@pytest.mark.gpu
def test_separate_requires_weight_teacher():
    hs, ht, w, target, BT, V = _make_simple()
    with pytest.raises(ValueError, match="weight_teacher"):
        distill_cross_entropy(hs, ht, w, target, teacher_mode="separate")


@pytest.mark.gpu
def test_separate_rejects_logits_teacher():
    hs, ht, w, target, BT, V = _make_simple()
    w_teacher = torch.randn(V, w.shape[1], device="cuda", dtype=torch.float16)
    logits_t = torch.randn(BT, V, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="logits_teacher"):
        distill_cross_entropy(
            hs, ht, w, target,
            teacher_mode="separate", weight_teacher=w_teacher, logits_teacher=logits_t,
        )


@pytest.mark.gpu
def test_precomputed_requires_logits_teacher():
    hs, ht, w, target, BT, V = _make_simple()
    with pytest.raises(ValueError, match="logits_teacher"):
        distill_cross_entropy(hs, None, w, target, teacher_mode="precomputed")


@pytest.mark.gpu
def test_precomputed_rejects_weight_teacher():
    hs, ht, w, target, BT, V = _make_simple()
    w_teacher = torch.randn(V, w.shape[1], device="cuda", dtype=torch.float16)
    logits_t = torch.randn(BT, V, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="weight_teacher"):
        distill_cross_entropy(
            hs, None, w, target,
            teacher_mode="precomputed", logits_teacher=logits_t, weight_teacher=w_teacher,
        )


@pytest.mark.gpu
def test_invalid_mode_string_raises():
    hs, ht, w, target, _, _ = _make_simple()
    with pytest.raises(ValueError, match="teacher_mode"):
        distill_cross_entropy(hs, ht, w, target, teacher_mode="bogus")


@pytest.mark.gpu
def test_negative_teacher_loss_weight_raises():
    hs, ht, w, target, _, _ = _make_simple()
    with pytest.raises(ValueError, match="teacher_loss_weight"):
        distill_cross_entropy(hs, ht, w, target, teacher_loss_weight=-0.1)


