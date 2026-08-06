"""CUDA/Triton parity across teacher modes and KL variants."""

from __future__ import annotations

import pytest
import torch
from utils.variants import (
    ALL_KL_VARIANTS,
    DETACHED_KL_VARIANTS,
    variant_kwargs,
)

import orda_kernel as ock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and ock.is_available()),
        reason="requires CUDA + Triton",
    ),
]

_MODE_VARIANTS = (
    *(("tied", variant) for variant in ALL_KL_VARIANTS),
    *(("separate", variant) for variant in ALL_KL_VARIANTS),
    *(("separate-trained", variant) for variant in DETACHED_KL_VARIANTS),
    *(("precomputed-logits", variant) for variant in DETACHED_KL_VARIANTS),
    *(("precomputed-hidden", variant) for variant in DETACHED_KL_VARIANTS),
)

# Variants whose zero-support sentinel decision the Triton kernel once derived
# from a per-block running normalization (the P1 block-independence bug).
_REVERSE_SUPPORT_VARIANTS = ("reverse", "jeffreys", "entropy_adaptive")


def _make_teacher(mode, teacher_hidden, teacher_weight, teacher_logits):
    if mode == "tied":
        return ock.TiedTeacher(teacher_hidden)
    if mode.startswith("separate"):
        return ock.SeparateTeacher(teacher_hidden, teacher_weight)
    if mode == "precomputed-hidden":
        return ock.PrecomputedTeacher(
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        )
    return ock.PrecomputedTeacher(logits=teacher_logits)


# ── GPU kernel/reference parity ──────────────────────────────────────────────
@pytest.mark.parametrize("mode,variant", _MODE_VARIANTS)
@pytest.mark.parametrize("kl_temperature", [1.0, 2.0])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_kernel_matches_reference(mode, variant, kl_temperature, reduction):
    device = "cuda"
    bt, hidden_size, vocab = 64, 32, 512
    torch.manual_seed(0)
    teacher_hidden_size = hidden_size if mode == "tied" else 24
    precomputed = mode.startswith("precomputed")
    if mode == "tied":
        teacher_ce_weight = 1.0
    elif mode == "separate-trained":
        teacher_ce_weight = 0.5
    elif mode == "separate" and variant == "forward":
        teacher_ce_weight = 0.5
    else:
        teacher_ce_weight = 0.0

    student_hidden = torch.randn(
        bt,
        hidden_size,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_hidden = torch.randn(
        bt,
        teacher_hidden_size,
        device=device,
        dtype=torch.float32,
        requires_grad=not precomputed,
    )
    weight = torch.randn(
        vocab,
        hidden_size,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_weight = torch.randn(
        vocab,
        teacher_hidden_size,
        device=device,
        dtype=torch.float32,
        requires_grad=mode.startswith("separate"),
    )
    labels = torch.randint(0, vocab, (bt,), device=device)
    labels[3] = -100
    teacher_logits = (
        (teacher_hidden @ teacher_weight.t()).detach()
        if mode == "precomputed-logits"
        else None
    )

    hs_ref = student_hidden.detach().clone().requires_grad_(True)
    ht_ref = (
        teacher_hidden.detach().clone().requires_grad_(teacher_hidden.requires_grad)
    )
    w_ref = weight.detach().clone().requires_grad_(True)
    tw_ref = (
        teacher_weight.detach().clone().requires_grad_(teacher_weight.requires_grad)
    )
    logits_ref = None if teacher_logits is None else teacher_logits.detach().clone()

    common = {
        "kl_weight": 0.4,
        "kl_temperature": kl_temperature,
        "teacher_ce_weight": teacher_ce_weight,
        "kl_divergence": variant,
        "reduction": reduction,
        **variant_kwargs(variant),
    }
    out_kernel = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        _make_teacher(mode, teacher_hidden, teacher_weight, teacher_logits),
        backend="triton",
        **common,
    )
    out_reference = ock.distillation_loss(
        hs_ref,
        w_ref,
        labels,
        _make_teacher(mode, ht_ref, tw_ref, logits_ref),
        backend="torch",
        **common,
    )

    for actual, expected in zip(out_kernel, out_reference, strict=True):
        assert torch.allclose(actual, expected, rtol=2e-3, atol=2e-3)

    out_kernel.loss.backward()
    out_reference.loss.backward()
    assert torch.allclose(student_hidden.grad, hs_ref.grad, rtol=2e-3, atol=2e-3)
    assert torch.allclose(weight.grad, w_ref.grad, rtol=2e-3, atol=2e-3)
    if teacher_hidden.requires_grad:
        assert (teacher_hidden.grad is not None) == (ht_ref.grad is not None)
        if teacher_hidden.grad is not None:
            assert torch.allclose(
                teacher_hidden.grad,
                ht_ref.grad,
                rtol=2e-3,
                atol=2e-3,
            )
    if teacher_weight.requires_grad:
        assert (teacher_weight.grad is not None) == (tw_ref.grad is not None)
        if teacher_weight.grad is not None:
            assert torch.allclose(
                teacher_weight.grad,
                tw_ref.grad,
                rtol=2e-3,
                atol=2e-3,
            )


def _partial_support_precomputed_inputs(vocab, dead_columns, *, seed):
    device = "cuda"
    bt, hidden_size = 16, 32
    torch.manual_seed(seed)
    student_hidden = torch.randn(
        bt, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(
        vocab, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    teacher_logits = torch.randn(bt, vocab, device=device, dtype=torch.float32)
    teacher_logits[:, dead_columns] = float("-inf")
    labels = torch.randint(0, vocab, (bt,), device=device)
    labels[2] = -100
    return student_hidden, weight, teacher_logits, labels


# ── Rényi partial-support parity ─────────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.parametrize("variant", ("renyi", "renyi_reverse"))
@pytest.mark.parametrize("order", (0.5, 1.5))
def test_renyi_partial_support_matches_reference(variant, order):
    from utils.reference import reference_distillation_loss

    vocab = 96
    student_hidden, weight, teacher_logits, labels = (
        _partial_support_precomputed_inputs(vocab, [5, 40, 77], seed=17)
    )
    common = dict(
        student_ce_weight=0.0,
        kl_weight=0.4,
        kl_temperature=1.3,
        kl_divergence=variant,
        kl_alpha=order,
        reduction="mean",
    )
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        backend="triton",
        **common,
    )
    reference = reference_distillation_loss(
        torch,
        student_hidden.detach().cpu(),
        weight.detach().cpu(),
        labels.detach().cpu(),
        teacher_mode="precomputed",
        teacher_logits=teacher_logits.detach().cpu(),
        **common,
    )

    assert torch.allclose(out.kl.cpu(), reference[3].float(), rtol=3e-3, atol=3e-3)
    out.loss.backward()
    assert torch.isfinite(student_hidden.grad).all()


# ── Block-independent reverse support ────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.parametrize("variant", _REVERSE_SUPPORT_VARIANTS)
@pytest.mark.parametrize("max_fused_size", (32, 128, 512))
def test_reverse_support_decision_is_block_independent(variant, max_fused_size):
    from orda_kernel import ChunkSize, KernelConfig

    vocab = 300
    student_hidden, weight, teacher_logits, labels = (
        _partial_support_precomputed_inputs(vocab, [7, 150, 260], seed=23)
    )
    extra = {}
    if variant == "entropy_adaptive":
        extra["entropy_gate"] = ock.EntropyGate(alpha_min=0.5, alpha_max=0.5)
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=0.0,
        kl_weight=0.5,
        kl_divergence=variant,
        backend="triton",
        config=KernelConfig(
            chunk_size=ChunkSize.nrows(8),
            max_fused_size=max_fused_size,
            autotune=False,
        ),
        **extra,
    )
    assert torch.isposinf(out.kl)
    assert torch.isposinf(out.loss)
    out.loss.backward()
    assert torch.equal(student_hidden.grad, torch.zeros_like(student_hidden.grad))
    assert torch.equal(weight.grad, torch.zeros_like(weight.grad))


# ── Alpha-divergence endpoints ───────────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.parametrize("alpha", (1.0e-3, 0.5, 1.0 - 1.0e-3))
def test_alpha_divergence_endpoints_match_reference(alpha):
    from utils.reference import reference_distillation_loss

    vocab = 128
    device = "cuda"
    bt, hidden_size = 16, 32
    torch.manual_seed(31)
    student_hidden = torch.randn(
        bt, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(
        vocab, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    teacher_logits = torch.randn(bt, vocab, device=device, dtype=torch.float32)
    labels = torch.randint(0, vocab, (bt,), device=device)
    common = dict(
        student_ce_weight=0.0,
        kl_weight=0.4,
        kl_divergence="alpha_divergence",
        kl_alpha=alpha,
        reduction="mean",
    )
    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        backend="triton",
        **common,
    )
    reference = reference_distillation_loss(
        torch,
        student_hidden.detach().cpu(),
        weight.detach().cpu(),
        labels.detach().cpu(),
        teacher_mode="precomputed",
        teacher_logits=teacher_logits.detach().cpu(),
        **common,
    )

    assert out.kl >= 0.0
    assert torch.allclose(out.kl.cpu(), reference[3].float(), rtol=5e-3, atol=5e-3)
    out.loss.backward()
    assert torch.isfinite(student_hidden.grad).all()


@pytest.mark.parametrize(
    ("variant", "alpha"),
    (
        ("alpha_divergence", 1.0e-3),
        ("renyi", 0.99),
        ("renyi_reverse", 0.01),
        ("renyi_reverse", 1.01),
    ),
)
def test_log_space_escort_extreme_ratio_matches_torch(variant, alpha):
    def run(backend):
        student_hidden = torch.tensor([[1.0]], device="cuda", requires_grad=True)
        weight = torch.tensor(
            [[0.0], [-100.0]],
            device="cuda",
            requires_grad=True,
        )
        labels = torch.zeros(1, device="cuda", dtype=torch.long)
        teacher_logits = torch.tensor([[-100.0, 0.0]], device="cuda")
        out = ock.distillation_loss(
            student_hidden,
            weight,
            labels,
            ock.PrecomputedTeacher(logits=teacher_logits),
            student_ce_weight=0.0,
            kl_weight=1.0,
            kl_divergence=variant,
            kl_alpha=alpha,
            reduction="sum",
            backend=backend,
        )
        out.loss.backward()
        return out, student_hidden.grad, weight.grad

    actual_out, actual_hidden_grad, actual_weight_grad = run("triton")
    expected_out, expected_hidden_grad, expected_weight_grad = run("torch")
    pairs = (
        (actual_out.loss, expected_out.loss),
        (actual_out.kl, expected_out.kl),
        (actual_hidden_grad, expected_hidden_grad),
        (actual_weight_grad, expected_weight_grad),
    )
    for actual_tensor, expected_tensor in pairs:
        assert torch.isfinite(actual_tensor).all()
        assert torch.isfinite(expected_tensor).all()
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=3e-3,
            atol=3e-3,
        )


# ── Frozen graph ownership ───────────────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ("tied", "separate"))
def test_frozen_teacher_does_not_leak_graph_edge(mode):
    device = "cuda"
    bt, hidden_size, vocab = 16, 32, 128
    torch.manual_seed(37)
    student_hidden = torch.randn(
        bt, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(
        vocab, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    teacher_hidden = torch.randn(
        bt, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    teacher_weight = torch.randn(
        vocab, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    labels = torch.randint(0, vocab, (bt,), device=device)
    if mode == "tied":
        teacher = ock.TiedTeacher(teacher_hidden, mode="frozen")
    else:
        teacher = ock.SeparateTeacher(teacher_hidden, teacher_weight)

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        teacher,
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=0.5,
        kl_divergence="reverse",
        backend="triton",
    )
    out.loss.backward()

    assert teacher_hidden.grad is None
    if mode == "separate":
        assert teacher_weight.grad is None
    assert student_hidden.grad is not None
    assert weight.grad is not None


# ── Fully-dead teacher row sentinel ──────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.parametrize("variant", ("forward", "reverse", "renyi"))
def test_all_dead_teacher_row_contributes_zero_and_keeps_student_ce(variant):
    device = "cuda"
    bt, hidden_size, vocab = 8, 16, 64
    torch.manual_seed(41)
    student_hidden = torch.randn(
        bt, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(
        vocab, hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    teacher_logits = torch.randn(bt, vocab, device=device, dtype=torch.float32)
    teacher_logits[3, :] = float("-inf")
    labels = torch.randint(0, vocab, (bt,), device=device)
    extra = {"kl_alpha": 0.5} if variant == "renyi" else {}

    out = ock.distillation_loss(
        student_hidden,
        weight,
        labels,
        ock.PrecomputedTeacher(logits=teacher_logits),
        student_ce_weight=1.0,
        kl_weight=0.5,
        kl_divergence=variant,
        reduction="mean",
        backend="triton",
        **extra,
    )

    assert torch.isfinite(out.kl)
    out.loss.backward()
    assert torch.isfinite(student_hidden.grad).all()
    assert student_hidden.grad[3].abs().sum() > 0.0
