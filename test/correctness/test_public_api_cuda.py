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

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_kernel import (
    AlignedLikelihoodConfig,
    AlignedSpanPairs,
    ChunkSize,
    KernelConfig,
    PrecomputedSpanTeacher,
    PrecomputedTeacher,
    SeparateTeacher,
    TiedTeacher,
    aligned_span_distillation_loss,
    build_precomputed_span_teacher,
    distillation_loss,
)


@pytest.mark.parametrize("objective", ["binary_kl", "binary_ce"])
@pytest.mark.parametrize("num_chunks", [1, 3])
@pytest.mark.parametrize("student_ce_weight", [0.0, 0.7])
def test_aligned_span_triton_matches_torch_reference(
    objective,
    num_chunks,
    student_ce_weight,
):
    torch.manual_seed(919)
    hidden = (
        torch.randn(7, 5, device="cuda", dtype=torch.float16) * 0.25
    ).requires_grad_(True)
    weight = (
        torch.randn(17, 5, device="cuda", dtype=torch.float16) * 0.25
    ).requires_grad_(True)
    labels = torch.tensor([0, 3, -100, 5, 8, 2, 11], device="cuda")
    ranges = torch.tensor([[0, 2], [2, 5], [4, 7]], dtype=torch.int64)
    pairs = AlignedSpanPairs(ranges, ranges.clone())
    teacher = PrecomputedSpanTeacher(
        torch.tensor([-0.8, -1.2, -0.5], dtype=torch.float64),
        teacher_ranges=pairs.teacher_ranges,
    )
    config = AlignedLikelihoodConfig(
        student_ce_weight=student_ce_weight,
        span_kl_weight=0.6,
        reduction="mean",
        objective=objective,
    )

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    grad_scale = 8192.0
    gradient_materialization_scale = torch.full(
        (),
        grad_scale,
        dtype=torch.float32,
        device="cuda",
    )
    actual = aligned_span_distillation_loss(
        hidden,
        weight,
        labels,
        teacher,
        pairs,
        config,
        backend="triton",
        kernel_config=KernelConfig(
            num_chunks=num_chunks,
            autotune=False,
            max_fused_size=32,
        ),
        gradient_materialization_scale=gradient_materialization_scale,
    )
    expected = aligned_span_distillation_loss(
        hidden_ref,
        weight_ref,
        labels,
        teacher,
        pairs,
        config,
        backend="torch",
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert torch.allclose(actual_value, expected_value, atol=3e-3, rtol=3e-3)
    (actual.loss * grad_scale).backward()
    hidden.grad.div_(grad_scale)
    weight.grad.div_(grad_scale)
    expected.loss.backward()
    assert torch.allclose(hidden.grad, hidden_ref.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(weight.grad, weight_ref.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "scale_value",
    (float("inf"), torch.finfo(torch.float32).tiny),
)
def test_unsafe_gradient_materialization_scale_is_neutral_for_aligned_span_cuda(
    scale_value,
):
    torch.manual_seed(923)
    base_hidden = torch.randn(5, 4, device="cuda", dtype=torch.float16) * 0.25
    base_weight = torch.randn(13, 4, device="cuda", dtype=torch.float16) * 0.25
    labels = torch.tensor([0, -100, 3, 5, 7], device="cuda")
    ranges = torch.tensor([[0, 2], [2, 5]], dtype=torch.int64)
    pairs = AlignedSpanPairs(ranges, ranges.clone())
    teacher = PrecomputedSpanTeacher(
        torch.tensor([-0.7, -1.1], dtype=torch.float64),
        teacher_ranges=pairs.teacher_ranges,
    )
    config = AlignedLikelihoodConfig(
        student_ce_weight=0.7,
        span_kl_weight=0.6,
        reduction="mean",
        objective="binary_kl",
    )

    def run(scale):
        hidden = base_hidden.detach().clone().requires_grad_()
        weight = base_weight.detach().clone().requires_grad_()
        output = aligned_span_distillation_loss(
            hidden,
            weight,
            labels,
            teacher,
            pairs,
            config,
            backend="triton",
            kernel_config=KernelConfig(
                num_chunks=2,
                autotune=False,
                max_fused_size=32,
            ),
            gradient_materialization_scale=scale,
        )
        output.loss.backward()
        return output, hidden.grad, weight.grad

    unsafe_scale = torch.full((), scale_value, device="cuda")
    scaled = run(unsafe_scale)
    unscaled = run(None)

    for actual, expected in zip(scaled[0], unscaled[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(scaled[1], unscaled[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(scaled[2], unscaled[2], atol=2e-2, rtol=2e-2)


def test_precomputed_span_builder_accepts_cpu_ranges_for_cuda_teacher():
    torch.manual_seed(937)
    hidden = torch.randn(6, 4, device="cuda", dtype=torch.float16)
    weight = torch.randn(11, 4, device="cuda", dtype=torch.float16)
    labels = torch.tensor([0, 1, 2, -100, 4, 5], device="cuda")
    ranges = torch.tensor([[0, 3], [3, 6]], dtype=torch.int64)

    teacher = build_precomputed_span_teacher(hidden, weight, labels, ranges)

    assert teacher.span_log_probs.device.type == "cuda"
    assert teacher.span_log_probs.dtype == torch.float64
    assert not teacher.span_log_probs.requires_grad
    assert torch.isfinite(teacher.span_log_probs).all()


def _noncontiguous_randn(shape, *, dtype, device, requires_grad=False, seed=0):
    torch.manual_seed(seed)
    base = torch.randn(shape[1], shape[0], dtype=dtype, device=device)
    tensor = base.t()
    assert tensor.shape == shape
    assert not tensor.is_contiguous()
    return tensor.requires_grad_(requires_grad)


def _run_reference(
    hs,
    weight,
    labels,
    ref_kwargs,
    *,
    kl_weight=0.35,
    kl_temperature=1.4,
    reduction="mean",
):
    ref = reference_distillation_loss(
        torch,
        hs.cpu(),
        weight.cpu(),
        labels.cpu(),
        kl_weight=kl_weight,
        kl_temperature=kl_temperature,
        reduction=reduction,
        **{
            key: value.cpu() if hasattr(value, "cpu") else value
            for key, value in ref_kwargs.items()
        },
    )
    ref[0].backward()
    return ref


def _assert_common_grads(hs, weight, ref):
    assert hs.grad is not None
    assert weight.grad is not None
    assert_grad_close(
        torch, hs.grad, ref[4].grad, atol=5e-2, rtol=5e-2, name="student_hidden"
    )
    assert_grad_close(
        torch, weight.grad, ref[6].grad, atol=5e-2, rtol=5e-2, name="weight"
    )


def _make_mode_inputs(
    teacher_mode: str,
    *,
    dtype,
    seed,
    V=257,
    cached_logits_dtype=torch.float32,
):
    teacher_ce_weight = None
    teacher_weight_for_grad = None
    if teacher_mode == "tied":
        hs, ht, weight, labels = make_tied_inputs(
            torch,
            BT=16,
            H=32,
            V=V,
            dtype=dtype,
            seed=seed,
            device="cuda",
        )
        teacher = TiedTeacher(ht)
        ref_kwargs = dict(teacher_mode="tied", teacher_hidden=ht)
    elif teacher_mode in ("separate-frozen", "separate-trained"):
        hs, ht, weight, teacher_weight, labels = make_separate_inputs(
            torch,
            BT=16,
            Hs=32,
            Ht=48,
            V=V,
            dtype=dtype,
            seed=seed,
            device="cuda",
            teacher_requires_grad=teacher_mode == "separate-trained",
            teacher_weight_requires_grad=teacher_mode == "separate-trained",
        )
        if teacher_mode == "separate-trained":
            teacher_ce_weight = 1.0
            teacher_weight_for_grad = teacher_weight
        teacher = SeparateTeacher(ht, teacher_weight)
        ref_kwargs = dict(
            teacher_mode="separate",
            teacher_hidden=ht,
            teacher_weight=teacher_weight,
            teacher_ce_weight=teacher_ce_weight,
        )
    elif teacher_mode == "precomputed-logits":
        hs, weight, teacher_logits, labels = make_precomputed_logits_inputs(
            torch,
            BT=16,
            H=32,
            V=V,
            dtype=dtype,
            seed=seed,
            device="cuda",
        )
        if cached_logits_dtype is not None:
            teacher_logits = teacher_logits.to(cached_logits_dtype)
        teacher = PrecomputedTeacher(logits=teacher_logits)
        ref_kwargs = dict(teacher_mode="precomputed", teacher_logits=teacher_logits)
    else:
        hs, weight, teacher_hidden, teacher_weight, labels = (
            make_precomputed_hidden_inputs(
                torch,
                BT=16,
                Hs=32,
                Ht=48,
                V=V,
                dtype=dtype,
                seed=seed,
                device="cuda",
            )
        )
        teacher = PrecomputedTeacher(
            teacher_hidden=teacher_hidden, teacher_weight=teacher_weight
        )
        ref_kwargs = dict(
            teacher_mode="precomputed",
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        )
    return (
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight,
        teacher_weight_for_grad,
        ref_kwargs,
    )


# ── Inactive student-CE backend parity ───────────────────────────────────────
_INACTIVE_STUDENT_CE_PATHS = (
    "tied",
    "separate-frozen",
    "separate-trained",
    "separate-cotrained",
    "precomputed",
)


def _run_inactive_student_ce_path(path: str, backend: str):
    seed = 1_500 + _INACTIVE_STUDENT_CE_PATHS.index(path)
    teacher_ce_weight = 0.0
    kl_weight = 0.35
    kl_divergence = "forward"
    teacher_tensors = ()

    if path == "tied":
        hs, ht, weight, labels = make_tied_inputs(
            torch, BT=6, H=16, V=129, dtype=torch.float32, seed=seed, device="cuda"
        )
        teacher = TiedTeacher(ht)
        teacher_ce_weight = 1.0
        kl_weight = 0.0
        teacher_tensors = (ht,)
    elif path == "precomputed":
        hs, weight, logits, labels = make_precomputed_logits_inputs(
            torch, BT=6, H=16, V=129, dtype=torch.float32, seed=seed, device="cuda"
        )
        teacher = PrecomputedTeacher(logits=logits)
    else:
        teacher_live = path != "separate-frozen"
        hs, ht, weight, teacher_weight, labels = make_separate_inputs(
            torch,
            BT=6,
            Hs=16,
            Ht=20,
            V=129,
            dtype=torch.float32,
            seed=seed,
            device="cuda",
            teacher_requires_grad=teacher_live,
            teacher_weight_requires_grad=teacher_live,
        )
        teacher = SeparateTeacher(ht, teacher_weight)
        teacher_tensors = (ht, teacher_weight)
        if path == "separate-trained":
            teacher_ce_weight = 1.0
            kl_weight = 0.0
        elif path == "separate-cotrained":
            kl_divergence = "mutual"

    output = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        student_ce_weight=0.0,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=kl_weight,
        kl_temperature=1.4,
        kl_divergence=kl_divergence,
        profile="debug",
        backend=backend,
        config=KernelConfig(
            chunk_size=ChunkSize.nrows(3),
            max_fused_size=256,
            autotune=False,
        ),
    )
    output.loss.backward()
    gradients = (hs.grad, weight.grad, *(tensor.grad for tensor in teacher_tensors))
    return output, gradients


def _assert_optional_grad_matches(actual, expected, *, name: str):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert_grad_close(
        torch,
        actual,
        expected,
        atol=5e-3,
        rtol=5e-2,
        name=name,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("path", _INACTIVE_STUDENT_CE_PATHS)
def test_inactive_student_ce_triton_matches_torch_backend(path: str):
    triton_output, triton_gradients = _run_inactive_student_ce_path(path, "triton")
    torch_output, torch_gradients = _run_inactive_student_ce_path(path, "torch")

    assert_loss_components_match(
        torch,
        triton_output,
        torch_output,
        atol=5e-3,
        rtol=5e-2,
    )
    assert_scalar_close(
        torch,
        triton_output.loss,
        torch_output.loss,
        atol=5e-3,
        rtol=5e-2,
        name="loss",
    )
    assert torch.equal(
        triton_output.student_ce,
        torch.zeros_like(triton_output.student_ce),
    )
    assert len(triton_gradients) == len(torch_gradients)
    for index, (actual, expected) in enumerate(
        zip(triton_gradients, torch_gradients, strict=True)
    ):
        _assert_optional_grad_matches(actual, expected, name=f"gradient_{index}")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "teacher_mode",
    [
        "tied",
        "separate-frozen",
        "separate-trained",
        "precomputed-logits",
        "precomputed-hidden",
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_public_triton_backend_matches_fp64_reference(
    teacher_mode: str, reduction: str
):
    (
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight,
        teacher_weight_for_grad,
        ref_kwargs,
    ) = _make_mode_inputs(
        teacher_mode,
        dtype=torch.float16,
        seed=201,
        cached_logits_dtype=None,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=1.4,
        reduction=reduction,
        backend="triton",
        config=KernelConfig(chunk_size=ChunkSize.nrows(7)),
    )
    out.loss.backward()

    ref = _run_reference(hs, weight, labels, ref_kwargs, reduction=reduction)

    assert_loss_components_match(torch, out, ref, atol=5e-3, rtol=5e-3)
    assert_scalar_close(torch, out.loss, ref[0], atol=5e-3, rtol=5e-3, name="loss")
    _assert_common_grads(hs, weight, ref)
    if teacher_mode == "tied":
        assert_grad_close(
            torch,
            teacher.hidden.grad,
            ref[5].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_hidden",
        )
    if teacher_mode == "separate-trained":
        assert_grad_close(
            torch,
            teacher.hidden.grad,
            ref[5].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_hidden",
        )
        assert teacher_weight_for_grad is not None
        assert_grad_close(
            torch,
            teacher_weight_for_grad.grad,
            ref[7].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_weight",
        )


# ── Prepared teacher cache ───────────────────────────────────────────────────
@pytest.mark.gpu
def test_prepared_logits_cache_matches_raw_cache_on_cuda():
    torch.manual_seed(307)
    base_hidden = torch.randn(5, 16, device="cuda", dtype=torch.float16)
    base_weight = torch.randn(129, 16, device="cuda", dtype=torch.float16)
    labels = torch.tensor([0, 3, -100, 7, 11], device="cuda")
    raw_teacher = PrecomputedTeacher(
        logits=torch.randn(5, 129, device="cuda", dtype=torch.float64)
    )
    prepared_teacher = raw_teacher.prepare_logits(torch.float16)

    def run(teacher):
        hidden = base_hidden.detach().clone().requires_grad_()
        weight = base_weight.detach().clone().requires_grad_()
        output = distillation_loss(
            hidden,
            weight,
            labels,
            teacher,
            student_ce_weight=0.7,
            kl_weight=0.35,
            kl_temperature=1.4,
            reduction="mean",
            backend="triton",
            config=KernelConfig(
                chunk_size=ChunkSize.nrows(2),
                max_fused_size=256,
                autotune=False,
            ),
        )
        output.loss.backward()
        return output, hidden.grad, weight.grad

    raw = run(raw_teacher)
    prepared = run(prepared_teacher)
    for actual, expected in zip(raw[0], prepared[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(raw[1], prepared[1], atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(raw[2], prepared[2], atol=3e-2, rtol=3e-2)


# ── Multi-device launch guard ────────────────────────────────────────────────
@pytest.mark.gpu
def test_triton_launch_uses_operand_device_when_current_device_differs():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA/HIP devices")

    original_device = torch.cuda.current_device()
    ambient_device = 0
    operand_device = torch.device("cuda", 1)
    try:
        torch.cuda.set_device(ambient_device)
        student_hidden = torch.randn(
            4, 8, device=operand_device, dtype=torch.float16, requires_grad=True
        )
        teacher_hidden = torch.randn(
            4, 8, device=operand_device, dtype=torch.float16, requires_grad=True
        )
        weight = torch.randn(
            17, 8, device=operand_device, dtype=torch.float16, requires_grad=True
        )
        labels = torch.randint(0, 17, (4,), device=operand_device)

        output = distillation_loss(
            student_hidden,
            weight,
            labels,
            TiedTeacher(teacher_hidden),
            teacher_ce_weight=1.0,
            kl_weight=0.35,
            backend="triton",
            config=KernelConfig(max_fused_size=64, autotune=False),
        )
        output.loss.backward()

        assert output.loss.device == operand_device
        assert torch.isfinite(output.loss)
        assert torch.cuda.current_device() == ambient_device
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "teacher_mode",
    [
        "tied",
        "separate-frozen",
        "separate-trained",
        "precomputed-logits",
        "precomputed-hidden",
    ],
)
def test_public_triton_backend_skips_kl_when_weight_is_zero(teacher_mode: str):
    (
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight,
        teacher_weight_for_grad,
        ref_kwargs,
    ) = _make_mode_inputs(
        teacher_mode,
        dtype=torch.float16,
        seed=401,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.0,
        kl_temperature=1.4,
        backend="triton",
        config=KernelConfig(chunk_size=ChunkSize.nrows(7)),
    )
    out.loss.backward()

    ref = _run_reference(
        hs,
        weight,
        labels,
        ref_kwargs,
        kl_weight=0.0,
        kl_temperature=1.4,
    )

    assert_scalar_close(torch, out.loss, ref[0], atol=5e-3, rtol=5e-3, name="loss")
    assert_scalar_close(
        torch, out.student_ce, ref[1], atol=5e-3, rtol=5e-3, name="student_ce"
    )
    assert_scalar_close(
        torch, out.teacher_ce, ref[2], atol=5e-3, rtol=5e-3, name="teacher_ce"
    )
    assert_scalar_close(
        torch, out.kl, torch.zeros_like(out.kl), atol=0.0, rtol=0.0, name="kl"
    )
    _assert_common_grads(hs, weight, ref)
    if teacher_mode == "tied":
        assert_grad_close(
            torch,
            teacher.hidden.grad,
            ref[5].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_hidden",
        )
    if teacher_mode == "separate-trained":
        assert_grad_close(
            torch,
            teacher.hidden.grad,
            ref[5].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_hidden",
        )
        assert teacher_weight_for_grad is not None
        assert_grad_close(
            torch,
            teacher_weight_for_grad.grad,
            ref[7].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_weight",
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "teacher_mode",
    [
        "tied",
        "separate-frozen",
        "separate-trained",
        "precomputed-logits",
        "precomputed-hidden",
    ],
)
@pytest.mark.parametrize("max_fused_size", [128, 512])
def test_public_triton_backend_temperature_one_matches_reference(
    teacher_mode: str, max_fused_size: int
):
    (
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight,
        teacher_weight_for_grad,
        ref_kwargs,
    ) = _make_mode_inputs(
        teacher_mode,
        dtype=torch.float16,
        seed=501 + max_fused_size,
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=1.0,
        backend="triton",
        config=KernelConfig(
            chunk_size=ChunkSize.nrows(7),
            max_fused_size=max_fused_size,
        ),
    )
    out.loss.backward()

    ref = _run_reference(
        hs,
        weight,
        labels,
        ref_kwargs,
        kl_weight=0.35,
        kl_temperature=1.0,
    )

    assert_loss_components_match(torch, out, ref, atol=5e-3, rtol=5e-3)
    assert_scalar_close(torch, out.loss, ref[0], atol=5e-3, rtol=5e-3, name="loss")
    _assert_common_grads(hs, weight, ref)
    if teacher_mode == "tied":
        assert_grad_close(
            torch,
            teacher.hidden.grad,
            ref[5].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_hidden",
        )
    if teacher_mode == "separate-trained":
        assert_grad_close(
            torch,
            teacher.hidden.grad,
            ref[5].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_hidden",
        )
        assert teacher_weight_for_grad is not None
        assert_grad_close(
            torch,
            teacher_weight_for_grad.grad,
            ref[7].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_weight",
        )


@pytest.mark.gpu
def test_tied_triton_backend_teacher_ce_zero_keeps_teacher_grad_none():
    hs, ht, weight, labels = make_tied_inputs(
        torch,
        BT=16,
        H=32,
        V=257,
        dtype=torch.float16,
        seed=601,
        device="cuda",
    )

    out = distillation_loss(
        hs,
        weight,
        labels,
        TiedTeacher(ht),
        teacher_ce_weight=0.0,
        kl_weight=0.35,
        kl_temperature=1.4,
        backend="triton",
        config=KernelConfig(chunk_size=ChunkSize.nrows(7)),
    )
    out.loss.backward()

    assert ht.grad is None


@pytest.mark.gpu
@pytest.mark.parametrize("kl_divergence", ["jeffreys", "mutual", "cojeffreys"])
def test_separate_triton_backend_kl_variants_match_reference(kl_divergence: str):
    hs, ht, weight, teacher_weight, labels = make_separate_inputs(
        torch,
        BT=16,
        Hs=32,
        Ht=48,
        V=257,
        dtype=torch.float16,
        seed=701,
        device="cuda",
        teacher_requires_grad=True,
        teacher_weight_requires_grad=True,
    )
    out = distillation_loss(
        hs,
        weight,
        labels,
        SeparateTeacher(ht, teacher_weight),
        teacher_ce_weight=0.0,
        kl_weight=0.35,
        kl_temperature=1.4,
        kl_divergence=kl_divergence,
        backend="triton",
        config=KernelConfig(chunk_size=ChunkSize.nrows(7)),
    )
    out.loss.backward()

    ref = reference_distillation_loss(
        torch,
        hs.cpu(),
        weight.cpu(),
        labels.cpu(),
        teacher_mode="separate",
        teacher_hidden=ht.cpu(),
        teacher_weight=teacher_weight.cpu(),
        teacher_ce_weight=0.0,
        kl_weight=0.35,
        kl_temperature=1.4,
        kl_divergence=kl_divergence,
    )
    ref[0].backward()

    assert_loss_components_match(torch, out, ref, atol=5e-3, rtol=5e-3)
    _assert_common_grads(hs, weight, ref)
    if kl_divergence in ("mutual", "cojeffreys"):
        assert_grad_close(
            torch,
            ht.grad,
            ref[5].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_hidden",
        )
        assert_grad_close(
            torch,
            teacher_weight.grad,
            ref[7].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_weight",
        )
    else:
        assert ht.grad is None
        assert teacher_weight.grad is None


@pytest.mark.gpu
@pytest.mark.parametrize(
    "teacher_mode",
    [
        "tied",
        "separate-frozen",
        "separate-trained",
        "precomputed-logits",
        "precomputed-hidden",
    ],
)
def test_public_triton_backend_preserves_noncontiguous_input_gradients(
    teacher_mode: str,
):
    BT, Hs, Ht, V = 16, 32, 48, 257
    labels = torch.randint(0, V, (BT,), device="cuda")
    labels[::4] = -100
    teacher_ce_weight = None
    teacher_weight_for_grad = None

    hs = _noncontiguous_randn(
        (BT, Hs), dtype=torch.float16, device="cuda", requires_grad=True, seed=301
    )
    weight = _noncontiguous_randn(
        (V, Hs), dtype=torch.float16, device="cuda", requires_grad=True, seed=302
    )

    if teacher_mode == "tied":
        ht = _noncontiguous_randn(
            (BT, Hs), dtype=torch.float16, device="cuda", requires_grad=True, seed=303
        )
        teacher = TiedTeacher(ht)
        ref_kwargs = dict(teacher_mode="tied", teacher_hidden=ht)
    elif teacher_mode in ("separate-frozen", "separate-trained"):
        ht = _noncontiguous_randn(
            (BT, Ht),
            dtype=torch.float16,
            device="cuda",
            requires_grad=teacher_mode == "separate-trained",
            seed=304,
        )
        teacher_weight = _noncontiguous_randn(
            (V, Ht),
            dtype=torch.float16,
            device="cuda",
            requires_grad=teacher_mode == "separate-trained",
            seed=305,
        )
        if teacher_mode == "separate-trained":
            teacher_ce_weight = 1.0
            teacher_weight_for_grad = teacher_weight
        teacher = SeparateTeacher(ht, teacher_weight)
        ref_kwargs = dict(
            teacher_mode="separate",
            teacher_hidden=ht,
            teacher_weight=teacher_weight,
            teacher_ce_weight=teacher_ce_weight,
        )
    elif teacher_mode == "precomputed-logits":
        teacher_logits = _noncontiguous_randn(
            (BT, V),
            dtype=torch.float16,
            device="cuda",
            requires_grad=False,
            seed=306,
        )
        teacher = PrecomputedTeacher(logits=teacher_logits)
        ref_kwargs = dict(teacher_mode="precomputed", teacher_logits=teacher_logits)
    else:
        teacher_hidden = _noncontiguous_randn(
            (BT, Ht),
            dtype=torch.float16,
            device="cuda",
            requires_grad=False,
            seed=307,
        )
        teacher_weight = _noncontiguous_randn(
            (V, Ht),
            dtype=torch.float16,
            device="cuda",
            requires_grad=False,
            seed=308,
        )
        teacher = PrecomputedTeacher(
            teacher_hidden=teacher_hidden, teacher_weight=teacher_weight
        )
        ref_kwargs = dict(
            teacher_mode="precomputed",
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
        )

    out = distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        teacher_ce_weight=teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=1.4,
        backend="triton",
        config=KernelConfig(chunk_size=ChunkSize.nrows(7)),
    )
    out.loss.backward()

    ref = _run_reference(hs, weight, labels, ref_kwargs)

    _assert_common_grads(hs, weight, ref)
    if teacher_mode == "tied":
        assert ht.grad is not None
        assert_grad_close(
            torch, ht.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden"
        )
    if teacher_mode == "separate-trained":
        assert ht.grad is not None
        assert teacher_weight_for_grad is not None
        assert teacher_weight_for_grad.grad is not None
        assert_grad_close(
            torch, ht.grad, ref[5].grad, atol=5e-2, rtol=5e-2, name="teacher_hidden"
        )
        assert_grad_close(
            torch,
            teacher_weight_for_grad.grad,
            ref[7].grad,
            atol=5e-2,
            rtol=5e-2,
            name="teacher_weight",
        )
