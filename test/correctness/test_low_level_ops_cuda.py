"""CUDA correctness tests for the public direct low-level ops."""

from __future__ import annotations

import pytest
from utils.assertions import (
    assert_grad_close,
    assert_loss_components_match,
    assert_scalar_close,
)
from utils.env import pytest_skip_if_no_cuda_kernel
from utils.factories import (
    make_precomputed_hidden_inputs,
    make_precomputed_logits_inputs,
    make_separate_inputs,
    make_tied_inputs,
)
from utils.reference import reference_distillation_loss
from utils.variants import (
    ALL_KL_VARIANTS,
    COUPLED_KL_VARIANTS,
    DETACHED_KL_VARIANTS,
)

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_kernel import ChunkSize
from orda_kernel._runtime import supports_bfloat16
from orda_kernel.ops import (
    precomputed_distillation_loss,
    separate_distillation_loss,
    tied_distillation_loss,
)
from orda_kernel.ops._kernels.precomputed import tokenwise as precomputed_kernels

# ── Direct mode matrix ───────────────────────────────────────────────────────
_DIRECT_MODE_VARIANTS = (
    *(("tied", variant, None) for variant in DETACHED_KL_VARIANTS),
    ("tied", "forward", 0.0),
    *(("tied", variant, 0.0) for variant in COUPLED_KL_VARIANTS),
    *(("separate", variant, 0.0) for variant in ALL_KL_VARIANTS),
    *(("separate", variant, 0.5) for variant in DETACHED_KL_VARIANTS),
    *(("precomputed_logits", variant, 0.0) for variant in DETACHED_KL_VARIANTS),
    *(("precomputed_hidden_weight", variant, 0.0) for variant in DETACHED_KL_VARIANTS),
)
_TEMPERATURE_PAIRS = ((1.0, 1.0), (2.0, 2.0), (1.0, 2.0), (2.0, 1.0))
_DIRECT_PARITY_CASES = tuple(
    (mode, variant, teacher_ce_weight, student_temperature, teacher_temperature)
    for mode, variant, teacher_ce_weight in _DIRECT_MODE_VARIANTS
    for student_temperature, teacher_temperature in _TEMPERATURE_PAIRS
    if variant not in COUPLED_KL_VARIANTS or student_temperature == teacher_temperature
)
_DTYPES = (
    ("float16", "bfloat16", "float32")
    if supports_bfloat16(torch.device("cuda"))
    else ("float16", "float32")
)


def _resolve_dtype(dtype_name: str):
    return getattr(torch, dtype_name)


def _make_direct_inputs(mode: str, *, dtype, seed: int):
    common = dict(BT=12, V=129, dtype=dtype, seed=seed, device="cuda")
    if mode == "tied":
        hs, ht, weight, labels = make_tied_inputs(torch, H=16, **common)
        return hs, weight, labels, ht, None, None
    if mode == "separate":
        hs, ht, weight, teacher_weight, labels = make_separate_inputs(
            torch,
            Hs=16,
            Ht=20,
            teacher_requires_grad=True,
            teacher_weight_requires_grad=True,
            **common,
        )
        return hs, weight, labels, ht, teacher_weight, None
    if mode == "precomputed_logits":
        hs, weight, teacher_logits, labels = make_precomputed_logits_inputs(
            torch,
            H=16,
            **common,
        )
        return hs, weight, labels, None, None, teacher_logits
    hs, weight, teacher_hidden, teacher_weight, labels = make_precomputed_hidden_inputs(
        torch,
        Hs=16,
        Ht=20,
        **common,
    )
    return hs, weight, labels, teacher_hidden, teacher_weight, None


def _call_direct(
    mode,
    student_hidden,
    weight,
    labels,
    teacher_hidden,
    teacher_weight,
    teacher_logits,
    *,
    variant,
    teacher_ce_weight,
    student_temperature,
    teacher_temperature,
    reduction,
    gradient_materialization_scale,
):
    common = dict(
        student_ce_weight=0.8,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=student_temperature,
        kl_teacher_temperature=teacher_temperature,
        kl_divergence=variant,
        reduction=reduction,
        chunk_size=ChunkSize.nrows(5),
        max_fused_size=256,
        autotune=False,
        gradient_materialization_scale=gradient_materialization_scale,
    )
    if mode == "tied":
        return tied_distillation_loss(
            student_hidden,
            teacher_hidden,
            weight,
            labels,
            **common,
        )
    if mode == "separate":
        return separate_distillation_loss(
            student_hidden,
            teacher_hidden,
            weight,
            teacher_weight,
            labels,
            **common,
        )
    return precomputed_distillation_loss(
        student_hidden,
        weight,
        logits_teacher=teacher_logits,
        labels=labels,
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        **common,
    )


def _run_reference(
    mode,
    student_hidden,
    weight,
    labels,
    teacher_hidden,
    teacher_weight,
    teacher_logits,
    *,
    variant,
    teacher_ce_weight,
    student_temperature,
    teacher_temperature,
    reduction,
):
    teacher_mode = (
        "precomputed"
        if mode in ("precomputed_logits", "precomputed_hidden_weight")
        else mode
    )
    return reference_distillation_loss(
        torch,
        student_hidden.detach().cpu(),
        weight.detach().cpu(),
        labels.detach().cpu(),
        teacher_mode=teacher_mode,
        teacher_hidden=(
            None if teacher_hidden is None else teacher_hidden.detach().cpu()
        ),
        teacher_weight=(
            None if teacher_weight is None else teacher_weight.detach().cpu()
        ),
        teacher_logits=(
            None if teacher_logits is None else teacher_logits.detach().cpu()
        ),
        student_ce_weight=0.8,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=student_temperature,
        kl_teacher_temperature=teacher_temperature,
        kl_divergence=variant,
        reduction=reduction,
    )


def _tolerances(dtype):
    if dtype is torch.bfloat16:
        return 3e-2, 1e-1
    if dtype is torch.float16:
        return 8e-3, 6e-2
    return 3e-3, 3e-2


# ── Value and gradient parity ────────────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.parametrize(
    (
        "mode",
        "variant",
        "teacher_ce_weight",
        "student_temperature",
        "teacher_temperature",
    ),
    _DIRECT_PARITY_CASES,
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_direct_ops_match_independent_reference(
    mode,
    variant,
    teacher_ce_weight,
    student_temperature,
    teacher_temperature,
    reduction,
    dtype_name,
):
    dtype = _resolve_dtype(dtype_name)
    seed = 700 + _DIRECT_MODE_VARIANTS.index((mode, variant, teacher_ce_weight))
    (
        student_hidden,
        weight,
        labels,
        teacher_hidden,
        teacher_weight,
        teacher_logits,
    ) = _make_direct_inputs(mode, dtype=dtype, seed=seed)
    grad_scale = 8192.0 if dtype is torch.float16 else 1.0
    gradient_materialization_scale = (
        torch.full((), grad_scale, dtype=torch.float32, device="cuda")
        if dtype is torch.float16
        else None
    )
    out = _call_direct(
        mode,
        student_hidden,
        weight,
        labels,
        teacher_hidden,
        teacher_weight,
        teacher_logits,
        variant=variant,
        teacher_ce_weight=teacher_ce_weight,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        reduction=reduction,
        gradient_materialization_scale=gradient_materialization_scale,
    )
    reference = _run_reference(
        mode,
        student_hidden,
        weight,
        labels,
        teacher_hidden,
        teacher_weight,
        teacher_logits,
        variant=variant,
        teacher_ce_weight=teacher_ce_weight,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        reduction=reduction,
    )

    assert isinstance(out, tuple)
    assert len(out) == 4
    assert out[0].requires_grad
    assert not out[1].requires_grad
    assert not out[2].requires_grad
    assert not out[3].requires_grad
    value_tol, grad_tol = _tolerances(dtype)
    assert_scalar_close(
        torch,
        out[0],
        reference[0],
        atol=value_tol,
        rtol=value_tol,
        name="loss",
    )
    assert_loss_components_match(
        torch,
        out,
        reference,
        atol=value_tol,
        rtol=value_tol,
    )

    (out[0] * grad_scale).backward()
    if grad_scale != 1.0:
        for tensor in (student_hidden, weight, teacher_hidden, teacher_weight):
            if tensor is not None and tensor.grad is not None:
                tensor.grad.div_(grad_scale)
    reference[0].backward()
    assert_grad_close(
        torch,
        student_hidden.grad,
        reference[4].grad,
        atol=grad_tol,
        rtol=grad_tol,
        name="student_hidden",
    )
    assert_grad_close(
        torch,
        weight.grad,
        reference[6].grad,
        atol=grad_tol,
        rtol=grad_tol,
        name="weight",
    )

    if mode == "tied":
        teacher_should_have_grad = teacher_ce_weight != 0.0 or variant in (
            "mutual",
            "cojeffreys",
        )
        assert (teacher_hidden.grad is not None) == teacher_should_have_grad
        assert (reference[5].grad is not None) == teacher_should_have_grad
        if teacher_should_have_grad:
            assert_grad_close(
                torch,
                teacher_hidden.grad,
                reference[5].grad,
                atol=grad_tol,
                rtol=grad_tol,
                name="teacher_hidden",
            )
    elif mode == "separate":
        teacher_should_have_grad = teacher_ce_weight != 0.0 or variant in (
            "mutual",
            "cojeffreys",
        )
        assert (teacher_hidden.grad is not None) == teacher_should_have_grad
        assert (teacher_weight.grad is not None) == teacher_should_have_grad
        if teacher_should_have_grad:
            assert_grad_close(
                torch,
                teacher_hidden.grad,
                reference[5].grad,
                atol=grad_tol,
                rtol=grad_tol,
                name="teacher_hidden",
            )
            assert_grad_close(
                torch,
                teacher_weight.grad,
                reference[7].grad,
                atol=grad_tol,
                rtol=grad_tol,
                name="teacher_weight",
            )


# ── Gradient materialization scale safety ────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.parametrize(
    "scale_value",
    (float("nan"), torch.finfo(torch.float32).tiny),
)
def test_unsafe_gradient_materialization_scale_is_neutral_for_tokenwise_cuda(
    scale_value,
):
    def run(scale):
        inputs = _make_direct_inputs("separate", dtype=torch.float16, seed=769)
        output = _call_direct(
            "separate",
            *inputs,
            variant="forward",
            teacher_ce_weight=0.0,
            student_temperature=1.4,
            teacher_temperature=1.4,
            reduction="mean",
            gradient_materialization_scale=scale,
        )
        output[0].backward()
        return output, inputs[0].grad, inputs[1].grad

    unsafe_scale = torch.full((), scale_value, device="cuda")
    scaled = run(unsafe_scale)
    unscaled = run(None)

    for actual, expected in zip(scaled[0], unscaled[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(scaled[1], unscaled[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(scaled[2], unscaled[2], atol=2e-2, rtol=2e-2)


# ── Tied zero-teacher-CE routing ─────────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["forward", "jeffreys"])
def test_tied_direct_teacher_ce_zero_has_no_nonzero_teacher_gradient(variant):
    student_hidden, teacher_hidden, weight, labels = make_tied_inputs(
        torch,
        BT=12,
        H=16,
        V=129,
        dtype=torch.float32,
        seed=811,
        device="cuda",
    )
    out = tied_distillation_loss(
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        teacher_ce_weight=0.0,
        kl_weight=0.35,
        kl_temperature=1.4,
        kl_divergence=variant,
        chunk_size=ChunkSize.nrows(5),
        max_fused_size=256,
    )
    out[0].backward()

    assert teacher_hidden.grad is None


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["tied", "separate"])
@pytest.mark.parametrize("assume_valid_labels", [False, True])
def test_ignored_overflowed_teacher_logits_do_not_poison_teacher_gradients(
    mode, assume_valid_labels
):
    bt, hidden_size, vocab = 2, 16, 129
    student_hidden = torch.zeros(
        bt,
        hidden_size,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_hidden = torch.zeros(
        bt,
        hidden_size,
        device="cuda",
        dtype=torch.float32,
    )
    teacher_hidden[0].fill_(1e20)
    teacher_hidden.requires_grad_()
    labels = torch.tensor([-100, 0], device="cuda", dtype=torch.long)

    if mode == "tied":
        teacher_weight = torch.full(
            (vocab, hidden_size),
            1e20,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        out = tied_distillation_loss(
            student_hidden,
            teacher_hidden,
            teacher_weight,
            labels,
            student_ce_weight=0.0,
            teacher_ce_weight=1.0,
            kl_weight=0.0,
            chunk_size=ChunkSize.nrows(bt),
            max_fused_size=256,
            autotune=False,
            assume_valid_labels=assume_valid_labels,
        )
    else:
        student_weight = torch.zeros(
            vocab,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher_weight = torch.full(
            (vocab, hidden_size),
            1e20,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        out = separate_distillation_loss(
            student_hidden,
            teacher_hidden,
            student_weight,
            teacher_weight,
            labels,
            student_ce_weight=0.0,
            teacher_ce_weight=1.0,
            kl_weight=0.0,
            chunk_size=ChunkSize.nrows(bt),
            max_fused_size=256,
            autotune=False,
            assume_valid_labels=assume_valid_labels,
        )

    assert torch.isfinite(out[0])
    out[0].backward()
    assert teacher_hidden.grad is not None
    assert teacher_weight.grad is not None
    assert torch.equal(teacher_hidden.grad[0], torch.zeros_like(teacher_hidden.grad[0]))
    assert torch.isfinite(teacher_hidden.grad).all()
    assert torch.isfinite(teacher_weight.grad).all()


# ── Inactive weighted branches ───────────────────────────────────────────────
@pytest.mark.gpu
def test_tied_inactive_student_branch_does_not_propagate_nan_through_shared_weight():
    bt, hidden_size, vocab = 4, 16, 129
    student_hidden = torch.full(
        (bt, hidden_size), float("nan"), device="cuda", requires_grad=True
    )
    teacher_hidden = torch.randn(bt, hidden_size, device="cuda", requires_grad=True)
    weight = torch.randn(vocab, hidden_size, device="cuda", requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.long)

    out = tied_distillation_loss(
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        student_ce_weight=0.0,
        teacher_ce_weight=1.0,
        kl_weight=0.0,
        chunk_size=ChunkSize.nrows(bt),
        max_fused_size=256,
        autotune=False,
    )
    assert torch.isfinite(out[0])
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    out[0].backward()

    # Matches the torch backend: an inactive student branch owns no gradient
    # edge, so backward returns None rather than a zero tensor.
    assert student_hidden.grad is None
    assert torch.isfinite(teacher_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()


@pytest.mark.gpu
def test_frozen_inactive_teacher_branch_does_not_propagate_nan():
    bt, student_hidden_size, teacher_hidden_size, vocab = 4, 16, 20, 129
    student_hidden = torch.randn(
        bt, student_hidden_size, device="cuda", requires_grad=True
    )
    teacher_hidden = torch.full(
        (bt, teacher_hidden_size), float("nan"), device="cuda", requires_grad=True
    )
    weight = torch.randn(vocab, student_hidden_size, device="cuda", requires_grad=True)
    teacher_weight = torch.full(
        (vocab, teacher_hidden_size),
        float("nan"),
        device="cuda",
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.long)

    out = separate_distillation_loss(
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        student_ce_weight=1.0,
        teacher_ce_weight=0.0,
        kl_weight=0.0,
        chunk_size=ChunkSize.nrows(bt),
        max_fused_size=256,
        autotune=False,
    )
    assert torch.isfinite(out[0])
    assert torch.equal(out[2], torch.zeros_like(out[2]))
    assert torch.equal(out[3], torch.zeros_like(out[3]))
    out[0].backward()

    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()
    assert teacher_hidden.grad is None
    assert teacher_weight.grad is None


@pytest.mark.gpu
def test_precomputed_inactive_kl_does_not_propagate_nan_logits():
    bt, hidden_size, vocab = 4, 16, 129
    student_hidden = torch.randn(bt, hidden_size, device="cuda", requires_grad=True)
    weight = torch.randn(vocab, hidden_size, device="cuda", requires_grad=True)
    teacher_logits = torch.full((bt, vocab), float("nan"), device="cuda")
    labels = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.long)

    out = precomputed_distillation_loss(
        student_hidden,
        weight,
        logits_teacher=teacher_logits,
        labels=labels,
        student_ce_weight=1.0,
        kl_weight=0.0,
        chunk_size=ChunkSize.nrows(bt),
        max_fused_size=256,
        autotune=False,
        assume_valid_teacher_cache=True,
    )
    assert torch.isfinite(out[0])
    assert torch.equal(out[3], torch.zeros_like(out[3]))
    out[0].backward()

    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(weight.grad).all()


# ── Ignored-row weight-gradient masking ──────────────────────────────────────
@pytest.mark.gpu
def test_ignored_nonfinite_hidden_rows_do_not_poison_weight_gradient():
    torch.manual_seed(929)
    base_hidden = torch.randn(4, 16, device="cuda")
    base_weight = torch.randn(129, 16, device="cuda")
    teacher_hidden = torch.randn(4, 20, device="cuda")
    teacher_weight = torch.randn(129, 20, device="cuda")
    labels = torch.tensor([-100, 1, -100, 3], device="cuda")

    def run(*, poison_ignored_rows: bool):
        hidden = base_hidden.detach().clone()
        hidden[0].fill_(float("nan") if poison_ignored_rows else 0.0)
        hidden[2].fill_(float("inf") if poison_ignored_rows else 0.0)
        hidden.requires_grad_()
        weight = base_weight.detach().clone().requires_grad_()
        output = separate_distillation_loss(
            hidden,
            teacher_hidden,
            weight,
            teacher_weight,
            labels,
            student_ce_weight=0.8,
            teacher_ce_weight=0.0,
            kl_weight=0.35,
            kl_divergence="forward",
            chunk_size=ChunkSize.nrows(2),
            max_fused_size=256,
            autotune=False,
        )
        output[0].backward()
        return output, hidden.grad, weight.grad

    poisoned = run(poison_ignored_rows=True)
    clean = run(poison_ignored_rows=False)

    for actual, expected in zip(poisoned[0], clean[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    assert torch.isfinite(poisoned[1]).all()
    ignored_gradient = poisoned[1][labels == -100]
    assert torch.equal(ignored_gradient, torch.zeros_like(ignored_gradient))
    torch.testing.assert_close(poisoned[1], clean[1], atol=3e-3, rtol=3e-3)
    assert torch.isfinite(poisoned[2]).all()
    torch.testing.assert_close(poisoned[2], clean[2], atol=3e-3, rtol=3e-3)


# ── Precomputed cached-logit dtype normalization ─────────────────────────────
_CACHED_LOGIT_DTYPES = (
    (torch.float16, torch.bfloat16, torch.float64)
    if supports_bfloat16(torch.device("cuda"))
    else (torch.float16, torch.float64)
)


@pytest.mark.gpu
@pytest.mark.parametrize("cached_dtype", _CACHED_LOGIT_DTYPES)
@pytest.mark.parametrize("kl_divergence", ["forward", "jeffreys"])
def test_precomputed_cached_logits_are_chunk_cast_to_model_compute_dtype(
    monkeypatch,
    cached_dtype,
    kl_divergence,
):
    bt, hidden_size, vocab = 5, 16, 129
    torch.manual_seed(937)
    base_student_hidden = torch.randn(bt, hidden_size, device="cuda")
    base_weight = torch.randn(vocab, hidden_size, device="cuda")
    cached_logits = torch.randn(bt, vocab, device="cuda").to(cached_dtype)
    labels = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.long)
    observed_chunks = []
    original = precomputed_kernels._precomputed_student_fwdbwd

    def observe_student_fwdbwd(
        logits_student_chunk, logits_teacher_chunk, *args, **kwargs
    ):
        observed_chunks.append(
            (
                logits_teacher_chunk.shape,
                logits_teacher_chunk.dtype,
                logits_teacher_chunk.data_ptr(),
            )
        )
        return original(logits_student_chunk, logits_teacher_chunk, *args, **kwargs)

    monkeypatch.setattr(
        precomputed_kernels,
        "_precomputed_student_fwdbwd",
        observe_student_fwdbwd,
    )

    def run(teacher_logits):
        student_hidden = base_student_hidden.detach().clone().requires_grad_()
        weight = base_weight.detach().clone().requires_grad_()
        output = precomputed_distillation_loss(
            student_hidden,
            weight,
            logits_teacher=teacher_logits,
            labels=labels,
            student_ce_weight=0.7,
            kl_weight=0.3,
            kl_temperature=1.4,
            kl_divergence=kl_divergence,
            reduction="mean",
            chunk_size=ChunkSize.nrows(2),
            max_fused_size=256,
            autotune=False,
        )
        output[0].backward()
        return output, student_hidden.grad, weight.grad

    mixed_output, mixed_student_grad, mixed_weight_grad = run(cached_logits)
    assert [shape[0] for shape, _dtype, _pointer in observed_chunks] == [2, 2, 1]
    assert {dtype for _shape, dtype, _pointer in observed_chunks} == {torch.float32}
    assert len({pointer for _shape, _dtype, pointer in observed_chunks}) == 1

    monkeypatch.setattr(
        precomputed_kernels,
        "_precomputed_student_fwdbwd",
        original,
    )
    normalized_output, normalized_student_grad, normalized_weight_grad = run(
        cached_logits.float()
    )

    for mixed, normalized in zip(mixed_output, normalized_output, strict=True):
        torch.testing.assert_close(mixed, normalized, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(
        mixed_student_grad,
        normalized_student_grad,
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        mixed_weight_grad,
        normalized_weight_grad,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.gpu
def test_all_zero_tied_weights_produce_finite_zero_grads_with_nan_models():
    bt, hidden_size, vocab = 4, 16, 129
    student_hidden = torch.full(
        (bt, hidden_size), float("nan"), device="cuda", requires_grad=True
    )
    teacher_hidden = torch.full(
        (bt, hidden_size), float("nan"), device="cuda", requires_grad=True
    )
    weight = torch.full(
        (vocab, hidden_size), float("nan"), device="cuda", requires_grad=True
    )
    labels = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.long)

    out = tied_distillation_loss(
        student_hidden,
        teacher_hidden,
        weight,
        labels,
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=0.0,
        chunk_size=ChunkSize.nrows(bt),
        max_fused_size=256,
        autotune=False,
    )
    assert all(torch.equal(component, torch.zeros_like(component)) for component in out)
    out[0].backward()

    assert torch.equal(student_hidden.grad, torch.zeros_like(student_hidden.grad))
    assert torch.equal(weight.grad, torch.zeros_like(weight.grad))
    assert teacher_hidden.grad is None


@pytest.mark.gpu
def test_cotrained_zero_ce_components_remain_finite_when_kl_is_live():
    student_hidden, teacher_hidden, weight, teacher_weight, labels = (
        make_separate_inputs(
            torch,
            BT=4,
            Hs=16,
            Ht=20,
            V=129,
            dtype=torch.float32,
            seed=919,
            device="cuda",
            teacher_requires_grad=True,
            teacher_weight_requires_grad=True,
        )
    )
    out = separate_distillation_loss(
        student_hidden,
        teacher_hidden,
        weight,
        teacher_weight,
        labels,
        student_ce_weight=0.0,
        teacher_ce_weight=0.0,
        kl_weight=0.35,
        kl_divergence="mutual",
        chunk_size=ChunkSize.nrows(4),
        max_fused_size=256,
        autotune=False,
    )
    assert torch.isfinite(out[0])
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.equal(out[2], torch.zeros_like(out[2]))
    out[0].backward()

    for gradient in (
        student_hidden.grad,
        teacher_hidden.grad,
        weight.grad,
        teacher_weight.grad,
    ):
        assert gradient is not None and torch.isfinite(gradient).all()
