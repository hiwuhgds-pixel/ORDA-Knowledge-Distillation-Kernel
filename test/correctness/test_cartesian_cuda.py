"""Full sharded Cartesian CUDA matrix, executed as grouped per-item loops."""

from __future__ import annotations

import traceback

import pytest
from utils.assertions import (
    assert_grad_close,
    assert_loss_components_match,
    assert_scalar_close,
)
from utils.env import pytest_skip_if_no_cuda_kernel
from utils.matrices import (
    REJECTION_CONTRACTS,
    ExpectedOutcome,
    cartesian_selection_from_environment,
    classify_case,
    effective_teacher_ce_weight,
    group_cartesian_cases,
    representation_mode,
    selected_cartesian_cases_for_shard,
    shard_spec_from_environment,
)
from utils.reference import reference_distillation_loss

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch

from orda_kernel import (
    ChunkSize,
    KernelConfig,
    PrecomputedTeacher,
    SeparateTeacher,
    TiedTeacher,
    distillation_loss,
)
from orda_kernel._runtime import supports_bfloat16
from orda_kernel.ops import (
    precomputed_distillation_loss,
    separate_distillation_loss,
    tied_distillation_loss,
)

# ── Deterministic shard parameters ───────────────────────────────────────────
_SHARD_SPEC = shard_spec_from_environment()
_SELECTION = cartesian_selection_from_environment()
_SELECTED_CASES = selected_cartesian_cases_for_shard(_SELECTION, _SHARD_SPEC)
assert all(
    case.ordinal % _SHARD_SPEC.count == _SHARD_SPEC.index for case in _SELECTED_CASES
)
_NATIVE_BF16 = supports_bfloat16(torch.device("cuda"))

_RUNNABLE_CASE_COUNT = sum(
    case.dtype != "bfloat16" or _NATIVE_BF16 for case in _SELECTED_CASES
)
_CASE_GROUPS = group_cartesian_cases(
    case for case in _SELECTED_CASES if case.dtype != "bfloat16" or _NATIVE_BF16
)
del _SELECTED_CASES
assert sum(len(members) for members in _CASE_GROUPS.values()) == _RUNNABLE_CASE_COUNT
_GROUPED_CASES = tuple(case for members in _CASE_GROUPS.values() for case in members)
del _CASE_GROUPS
_ITEM_COUNT = min(200, len(_GROUPED_CASES))
_GROUP_PARAMS = tuple(
    pytest.param(
        _GROUPED_CASES[
            index * len(_GROUPED_CASES) // _ITEM_COUNT : (index + 1)
            * len(_GROUPED_CASES)
            // _ITEM_COUNT
        ],
        id=f"batch-{index + 1:03d}",
    )
    for index in range(_ITEM_COUNT)
)
if not _GROUP_PARAMS:
    _GROUP_PARAMS = (
        pytest.param(
            (),
            id="no-matching-cases",
            marks=pytest.mark.skip(
                reason="Cartesian selection matched no runnable case"
            ),
        ),
    )


# ── Tensor and label construction ────────────────────────────────────────────
def _randn_2d(shape, *, dtype, noncontiguous, requires_grad):
    if noncontiguous:
        tensor = (torch.randn(shape[1], shape[0], device="cuda", dtype=dtype) * 0.1).t()
        assert not tensor.is_contiguous()
    else:
        tensor = torch.randn(*shape, device="cuda", dtype=dtype) * 0.1
    return tensor.requires_grad_(requires_grad)


def _make_labels(pattern: str, *, bt: int, vocab: int):
    labels = torch.randint(0, vocab, (bt,), device="cuda")
    if pattern == "with_ignore_index":
        labels[1] = -100
    elif pattern == "all_ignored":
        labels.fill_(-100)
    elif pattern == "invalid_negative":
        labels[1] = -1
    elif pattern == "invalid_high":
        labels[1] = vocab
    return labels


def _make_case_inputs(case, dtype):
    bt, student_dim, teacher_dim, vocab = 6, 8, 6, 17
    torch.manual_seed(10_000 + case.ordinal)
    noncontiguous_hidden = case.input_layout == "noncontiguous_hidden"
    noncontiguous_weight = case.input_layout == "noncontiguous_weight"
    student_hidden = _randn_2d(
        (bt, student_dim),
        dtype=dtype,
        noncontiguous=noncontiguous_hidden,
        requires_grad=True,
    )
    weight = _randn_2d(
        (vocab, student_dim),
        dtype=dtype,
        noncontiguous=noncontiguous_weight,
        requires_grad=True,
    )
    labels = _make_labels(case.label_pattern, bt=bt, vocab=vocab)
    inputs = {
        "student_hidden": student_hidden,
        "weight": weight,
        "labels": labels,
        "teacher_hidden": None,
        "teacher_weight": None,
        "teacher_logits": None,
    }

    if case.teacher_representation == "tied":
        inputs["teacher_hidden"] = _randn_2d(
            (bt, student_dim),
            dtype=dtype,
            noncontiguous=noncontiguous_hidden,
            requires_grad=True,
        )
    elif case.teacher_representation == "separate":
        inputs["teacher_hidden"] = _randn_2d(
            (bt, teacher_dim),
            dtype=dtype,
            noncontiguous=noncontiguous_hidden,
            requires_grad=True,
        )
        inputs["teacher_weight"] = _randn_2d(
            (vocab, teacher_dim),
            dtype=dtype,
            noncontiguous=noncontiguous_weight,
            requires_grad=True,
        )
    elif case.teacher_representation == "precomputed_logits":
        inputs["teacher_logits"] = _randn_2d(
            (bt, vocab),
            dtype=dtype,
            noncontiguous=False,
            requires_grad=False,
        )
    else:
        inputs["teacher_hidden"] = _randn_2d(
            (bt, teacher_dim),
            dtype=dtype,
            noncontiguous=noncontiguous_hidden,
            requires_grad=False,
        )
        inputs["teacher_weight"] = _randn_2d(
            (vocab, teacher_dim),
            dtype=dtype,
            noncontiguous=noncontiguous_weight,
            requires_grad=False,
        )
    return inputs


# ── API and chunk-policy invocation ──────────────────────────────────────────
def _high_level_config(case) -> KernelConfig:
    common = dict(max_fused_size=64)
    if case.chunk_policy == "fixed_chunk_size":
        return KernelConfig(
            chunk_size=ChunkSize.nrows(3),
            autotune=False,
            **common,
        )
    if case.chunk_policy == "fixed_num_chunks":
        return KernelConfig(num_chunks=2, autotune=False, **common)
    if case.chunk_policy == "dynamic":
        return KernelConfig(
            chunk_size=ChunkSize.dynamic(),
            autotune=False,
            **common,
        )
    if case.chunk_policy == "autotune_off":
        return KernelConfig(autotune=False, **common)
    raise AssertionError(f"unknown Cartesian chunk policy: {case.chunk_policy}")


def _direct_chunk_kwargs(case) -> dict[str, object]:
    common: dict[str, object] = {
        "max_fused_size": 64,
    }
    if case.chunk_policy == "fixed_chunk_size":
        common.update(chunk_size=ChunkSize.nrows(3), autotune=False)
    elif case.chunk_policy == "fixed_num_chunks":
        common.update(num_chunks=2, autotune=False)
    elif case.chunk_policy == "dynamic":
        common.update(chunk_size=ChunkSize.dynamic(), autotune=False)
    elif case.chunk_policy == "autotune_off":
        common.update(autotune=False)
    else:
        raise AssertionError(f"unknown Cartesian chunk policy: {case.chunk_policy}")
    return common


def _teacher_descriptor(case, inputs):
    if case.teacher_representation == "tied":
        return TiedTeacher(inputs["teacher_hidden"])
    if case.teacher_representation == "separate":
        return SeparateTeacher(
            inputs["teacher_hidden"],
            inputs["teacher_weight"],
        )
    if case.teacher_representation == "precomputed_logits":
        return PrecomputedTeacher(logits=inputs["teacher_logits"])
    return PrecomputedTeacher(
        teacher_hidden=inputs["teacher_hidden"],
        teacher_weight=inputs["teacher_weight"],
    )


def _common_loss_kwargs(case) -> dict[str, object]:
    return {
        "student_ce_weight": case.student_ce_weight,
        "teacher_ce_weight": case.teacher_ce_weight,
        "kl_weight": case.kl_weight,
        "kl_temperature": case.kl_temperature,
        "kl_divergence": case.kl_variant,
        "reduction": case.reduction,
    }


def _invoke_case(case, inputs):
    common = _common_loss_kwargs(case)
    if case.api_level == "high_level":
        return distillation_loss(
            inputs["student_hidden"],
            inputs["weight"],
            inputs["labels"],
            _teacher_descriptor(case, inputs),
            profile="debug",
            backend="triton",
            config=_high_level_config(case),
            **common,
        )

    direct = _direct_chunk_kwargs(case)
    direct.update(common)
    if case.api_level == "direct_tied":
        teacher_hidden = (
            inputs["teacher_logits"]
            if case.teacher_representation == "precomputed_logits"
            else inputs["teacher_hidden"]
        )
        return tied_distillation_loss(
            inputs["student_hidden"],
            teacher_hidden,
            inputs["weight"],
            inputs["labels"],
            **direct,
        )
    if case.api_level == "direct_separate":
        if case.teacher_representation == "separate":
            teacher_hidden = inputs["teacher_hidden"]
            teacher_weight = inputs["teacher_weight"]
        else:
            teacher_hidden = (
                inputs["teacher_logits"]
                if case.teacher_representation == "precomputed_logits"
                else inputs["teacher_hidden"]
            )
            teacher_weight = None
        return separate_distillation_loss(
            inputs["student_hidden"],
            teacher_hidden,
            inputs["weight"],
            teacher_weight,
            inputs["labels"],
            **direct,
        )

    if case.teacher_representation == "precomputed_logits":
        logits_teacher = inputs["teacher_logits"]
        teacher_hidden = None
        teacher_weight = None
    else:
        logits_teacher = None
        teacher_hidden = inputs["teacher_hidden"]
        teacher_weight = (
            inputs["teacher_weight"] if case.teacher_representation != "tied" else None
        )
    return precomputed_distillation_loss(
        inputs["student_hidden"],
        inputs["weight"],
        logits_teacher=logits_teacher,
        labels=inputs["labels"],
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        **direct,
    )


# ── Independent oracle and assertions ────────────────────────────────────────
def _manual_reference(case, inputs):
    return reference_distillation_loss(
        torch,
        inputs["student_hidden"].detach().cpu(),
        inputs["weight"].detach().cpu(),
        inputs["labels"].detach().cpu(),
        teacher_mode=representation_mode(case),
        teacher_hidden=(
            None
            if inputs["teacher_hidden"] is None
            else inputs["teacher_hidden"].detach().cpu()
        ),
        teacher_weight=(
            None
            if inputs["teacher_weight"] is None
            else inputs["teacher_weight"].detach().cpu()
        ),
        teacher_logits=(
            None
            if inputs["teacher_logits"] is None
            else inputs["teacher_logits"].detach().cpu()
        ),
        student_ce_weight=case.student_ce_weight,
        teacher_ce_weight=case.teacher_ce_weight,
        kl_weight=case.kl_weight,
        kl_temperature=case.kl_temperature,
        kl_divergence=case.kl_variant,
        reduction=case.reduction,
    )


def _dtype_for_case(case):
    dtype = getattr(torch, case.dtype)
    if dtype is torch.bfloat16 and not supports_bfloat16(torch.device("cuda")):
        pytest.skip("CUDA/HIP device does not support native bfloat16")
    return dtype


def _tolerances(dtype):
    if dtype is torch.bfloat16:
        return 4e-2, 1.2e-1
    if dtype is torch.float16:
        return 1e-2, 7e-2
    return 4e-3, 4e-2


def _assert_optional_teacher_grad(actual, expected, *, dtype, name, strict_none):
    if expected is None:
        if strict_none:
            assert actual is None
        elif actual is not None:
            assert torch.equal(actual, torch.zeros_like(actual))
        return
    assert actual is not None
    _, grad_tol = _tolerances(dtype)
    assert_grad_close(
        torch,
        actual,
        expected,
        atol=grad_tol,
        rtol=grad_tol,
        name=name,
    )


def _assert_optional_grad(actual, expected, *, dtype, name):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    _, grad_tol = _tolerances(dtype)
    assert_grad_close(
        torch,
        actual,
        expected,
        atol=grad_tol,
        rtol=grad_tol,
        name=name,
    )


def _assert_success(case, inputs, out, reference, dtype):
    assert isinstance(out, tuple)
    assert len(out) == 4
    assert out[0].requires_grad
    assert not out[1].requires_grad
    assert not out[2].requires_grad
    assert not out[3].requires_grad

    value_tol, _ = _tolerances(dtype)
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

    out[0].backward()
    reference[0].backward()
    _assert_optional_grad(
        inputs["student_hidden"].grad,
        reference[4].grad,
        dtype=dtype,
        name="student_hidden",
    )
    _assert_optional_grad(
        inputs["weight"].grad,
        reference[6].grad,
        dtype=dtype,
        name="weight",
    )

    mode = representation_mode(case)
    if mode == "tied":
        detached_teacher_target = (
            effective_teacher_ce_weight(case) == 0.0 and case.kl_weight != 0.0
        )
        _assert_optional_teacher_grad(
            inputs["teacher_hidden"].grad,
            reference[5].grad,
            dtype=dtype,
            name="teacher_hidden",
            strict_none=not detached_teacher_target,
        )
    elif mode == "separate":
        teacher_should_have_grad = effective_teacher_ce_weight(case) != 0.0 or (
            case.kl_weight != 0.0 and case.kl_variant in ("mutual", "cojeffreys")
        )
        assert (reference[5].grad is not None) == teacher_should_have_grad
        assert (reference[7].grad is not None) == teacher_should_have_grad
        _assert_optional_teacher_grad(
            inputs["teacher_hidden"].grad,
            reference[5].grad,
            dtype=dtype,
            name="teacher_hidden",
            strict_none=True,
        )
        _assert_optional_teacher_grad(
            inputs["teacher_weight"].grad,
            reference[7].grad,
            dtype=dtype,
            name="teacher_weight",
            strict_none=True,
        )


# ── Full Cartesian execution ─────────────────────────────────────────────────
_MAX_RECORDED_FAILURES = 100
_LISTED_FAILURES = 20


def _run_single_case(case):
    dtype = _dtype_for_case(case)
    inputs = _make_case_inputs(case, dtype)
    outcome = classify_case(case)

    if outcome is not ExpectedOutcome.SUCCESS:
        rejection = REJECTION_CONTRACTS[outcome]
        with pytest.raises(
            rejection.exception_type,
            match=rejection.message_pattern,
        ):
            _invoke_case(case, inputs)
        return

    out = _invoke_case(case, inputs)
    reference = _manual_reference(case, inputs)
    _assert_success(case, inputs, out, reference, dtype)


def _failure_summary(exc):
    summary = f"{type(exc).__name__}: {exc}".splitlines()[0]
    if len(summary) > 200:
        summary = f"{summary[:200]}..."
    return summary


def _format_group_failures(failures, first_traceback, executed, total):
    listed = failures[:_LISTED_FAILURES]
    lines = [
        f"{len(failures)}/{total} cases failed (executed {executed}/{total})",
        "",
        "first failure traceback:",
        first_traceback.rstrip(),
        "",
    ]
    lines.extend(f"{case.case_id} - {summary}" for case, summary in listed)
    if len(failures) > len(listed):
        lines.append(f"... and {len(failures) - len(listed)} more failures")
    ordinal_args = " ".join(f"--ordinal {case.ordinal}" for case, _ in listed)
    lines.append("")
    lines.append(
        f"rerun without shard variables: python test/run_test_gpu.py {ordinal_args}"
    )
    return "\n".join(lines)


@pytest.mark.gpu
@pytest.mark.parametrize("group_cases", _GROUP_PARAMS)
def test_full_cartesian_group(group_cases):
    failures = []
    first_traceback = None
    executed = 0
    for case in group_cases:
        executed += 1
        try:
            _run_single_case(case)
        except (Exception, pytest.fail.Exception) as exc:
            if not failures:
                first_traceback = traceback.format_exc()
            failures.append((case, _failure_summary(exc)))
            if len(failures) >= _MAX_RECORDED_FAILURES:
                break
    if failures:
        message = _format_group_failures(
            failures, first_traceback, executed, len(group_cases)
        )
        pytest.fail(message, pytrace=False)
