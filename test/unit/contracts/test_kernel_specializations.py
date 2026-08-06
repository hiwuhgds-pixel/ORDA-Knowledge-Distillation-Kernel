from __future__ import annotations

import ast
import math
from itertools import product
from pathlib import Path

import pytest

from orda_kernel import EntropyGate
from orda_kernel._planning.capabilities import (
    KERNEL_CAPABILITIES,
    UnsupportedTritonPolicyError,
)
from orda_kernel._planning.models import (
    PRECOMPUTED,
    PRECOMPUTED_TOKENWISE,
    SEPARATE,
    SEPARATE_COTRAINED,
    SEPARATE_FROZEN,
    SEPARATE_TRAINED,
    STUDENT_GRAD_KL_VARIANTS,
    TEACHER_KL_GRAD_VARIANTS,
    TIED,
    TIED_COTRAINED,
    TIED_FROZEN,
    TIED_TRAINED,
    VALID_KL_VARIANTS,
)
from orda_kernel._planning.policy import resolve_execution_policy
from orda_kernel.ops._kernels._tokenwise_launch import (
    resolve_tokenwise_kl_launch,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "orda_kernel"
_KERNEL_ROOT = _PACKAGE_ROOT / "ops" / "_kernels"
_CANONICAL_KERNEL = _KERNEL_ROOT / "ce_kl_fwdbwd.py"
_TRITON_LAUNCH_META = frozenset({"num_warps"})
_STYLE = {
    "forward": 0,
    "jeffreys": 1,
    "mutual": 2,
    "cojeffreys": 3,
    "reverse": 4,
    "alpha_divergence": 5,
    "skew_forward": 6,
    "skew_js": 7,
    "entropy_adaptive": 8,
    "skew_reverse": 9,
    "renyi": 10,
    "renyi_reverse": 11,
    "dkd": 12,
    "abkd": 13,
    "triangular": 14,
}
_MODE_FILES = (
    "tied/frozen.py",
    "tied/trained.py",
    "tied/cotrained.py",
    "precomputed/tokenwise.py",
    "precomputed/aligned_span.py",
    "separate/frozen.py",
    "separate/trained.py",
    "separate/cotrained.py",
)


# ── Launch specialization ────────────────────────────────────────────────────
def test_tokenwise_launch_specialization_resolves_runtime_scalars():
    launch = resolve_tokenwise_kl_launch(
        compute_kl=True,
        kl_temperature=1.0,
        kl_teacher_temperature=2.0,
        kl_divergence="entropy_adaptive",
        kl_alpha=0.5,
        vocab_size=16,
        entropy_gate=EntropyGate(
            alpha_min=0.2,
            alpha_max=0.8,
            tau=0.4,
            scale=0.25,
        ),
    )

    assert launch.teacher_temperature == 2.0
    assert launch.both_t_one is False
    assert launch.kl_style == _STYLE["entropy_adaptive"]
    assert launch.renyi_near_one is False
    assert launch.gate_alpha_min == 0.2
    assert launch.gate_alpha_span == pytest.approx(0.6)
    assert launch.gate_tau == 0.4
    assert launch.gate_inv_scale == 4.0
    assert launch.inv_log_vocab == pytest.approx(1.0 / math.log(16))

    inactive = resolve_tokenwise_kl_launch(
        compute_kl=False,
        kl_temperature=2.0,
        kl_teacher_temperature=None,
        kl_divergence="entropy_adaptive",
        kl_alpha=0.5,
        vocab_size=16,
        entropy_gate=None,
    )
    assert inactive.both_t_one is True
    assert inactive.kl_style == _STYLE["forward"]
    assert inactive.renyi_near_one is False
    assert inactive.inv_log_vocab == 0.0


@pytest.mark.parametrize(
    ("compute_kl", "variant", "alpha", "expected"),
    (
        (True, "renyi", 0.875, True),
        (True, "renyi_reverse", 1.125, True),
        (True, "renyi", 0.5, False),
        (True, "forward", 1.0, False),
        (False, "renyi", 1.01, False),
    ),
)
def test_tokenwise_launch_specializes_renyi_near_one(
    compute_kl,
    variant,
    alpha,
    expected,
):
    launch = resolve_tokenwise_kl_launch(
        compute_kl=compute_kl,
        kl_temperature=1.0,
        kl_teacher_temperature=None,
        kl_divergence=variant,
        kl_alpha=alpha,
        vocab_size=16,
        entropy_gate=None,
    )

    assert launch.renyi_near_one is expected


# ── Reachable policies ───────────────────────────────────────────────────────
def _reachable_policies():
    for mode, teacher_ce, kl_weight, temperature, variant in product(
        (TIED, SEPARATE, PRECOMPUTED),
        (0.0, 1.0),
        (0.0, 1.0),
        (1.0, 2.0),
        sorted(VALID_KL_VARIANTS),
    ):
        try:
            yield resolve_execution_policy(
                mode,
                teacher_ce_weight=teacher_ce,
                kl_weight=kl_weight,
                kl_temperature=temperature,
                kl_variant=variant,
            )
        except (ValueError, UnsupportedTritonPolicyError):
            continue


def _legacy_key(policy, *, allow_teacher_neg_inf_support=False):
    path = policy.kernel_path
    spec = policy.specialization
    return (
        path,
        spec.compute_kl,
        spec.t_is_one,
        _STYLE[spec.kl_variant],
        allow_teacher_neg_inf_support,
    )


def _canonical_key(
    policy,
    *,
    allow_teacher_neg_inf_support=False,
    renyi_near_one=False,
):
    path = policy.kernel_path
    spec = policy.specialization
    return (
        path in (TIED_FROZEN, TIED_TRAINED, TIED_COTRAINED),
        path
        in (
            TIED_TRAINED,
            TIED_COTRAINED,
            SEPARATE_TRAINED,
            SEPARATE_COTRAINED,
        ),
        spec.compute_kl,
        path in (SEPARATE_COTRAINED, TIED_COTRAINED),
        spec.t_is_one,
        _STYLE[spec.kl_variant],
        0,
        allow_teacher_neg_inf_support,
        renyi_near_one,
    )


# ── Compile-variant gate ─────────────────────────────────────────────────────
def test_canonical_kernel_does_not_expand_semantic_compile_variants():
    policies = tuple(_reachable_policies())
    legacy = {_legacy_key(policy) for policy in policies}
    canonical = {_canonical_key(policy) for policy in policies}
    for policy in policies:
        spec = policy.specialization
        if spec.compute_kl and spec.kl_variant in ("renyi", "renyi_reverse"):
            canonical.add(_canonical_key(policy, renyi_near_one=True))
        if policy.kernel_path == PRECOMPUTED_TOKENWISE:
            legacy.add(_legacy_key(policy, allow_teacher_neg_inf_support=True))
            canonical.add(_canonical_key(policy, allow_teacher_neg_inf_support=True))
            if spec.compute_kl and spec.kl_variant in ("renyi", "renyi_reverse"):
                canonical.add(
                    _canonical_key(
                        policy,
                        allow_teacher_neg_inf_support=True,
                        renyi_near_one=True,
                    )
                )

    aligned_stages = {
        (False, False, False, False, True, _STYLE["forward"], stage, False, False)
        for stage in (1, 2)
    }
    canonical.update(aligned_stages)

    # Raw precomputed logits alone enable -inf support; generated teacher logits
    # keep the strict finite-GEMM program for every reachable KL specialization.
    assert len(legacy) == 170
    assert len(canonical) == 165
    assert len(canonical) <= len(legacy)


def test_kernel_capabilities_remain_product_gated():
    assert STUDENT_GRAD_KL_VARIANTS == frozenset(
        {
            "forward",
            "jeffreys",
            "reverse",
            "alpha_divergence",
            "skew_forward",
            "skew_js",
            "entropy_adaptive",
            "skew_reverse",
            "renyi",
            "renyi_reverse",
            "dkd",
            "abkd",
            "triangular",
        }
    )
    assert TEACHER_KL_GRAD_VARIANTS == frozenset({"mutual", "cojeffreys"})
    assert KERNEL_CAPABILITIES[TIED_FROZEN].kl_variants == STUDENT_GRAD_KL_VARIANTS
    assert KERNEL_CAPABILITIES[TIED_TRAINED].kl_variants == STUDENT_GRAD_KL_VARIANTS
    assert KERNEL_CAPABILITIES[TIED_COTRAINED].kl_variants == frozenset(
        {"mutual", "cojeffreys"}
    )
    assert (
        KERNEL_CAPABILITIES[PRECOMPUTED_TOKENWISE].kl_variants
        == STUDENT_GRAD_KL_VARIANTS
    )
    assert KERNEL_CAPABILITIES[SEPARATE_FROZEN].kl_variants == STUDENT_GRAD_KL_VARIANTS
    assert KERNEL_CAPABILITIES[SEPARATE_TRAINED].kl_variants == STUDENT_GRAD_KL_VARIANTS
    assert KERNEL_CAPABILITIES[SEPARATE_COTRAINED].kl_variants == frozenset(
        {"mutual", "cojeffreys"}
    )


def test_mode_wrappers_lower_only_the_expected_constexpr_axes():
    expected = {
        "tied/frozen.py": {
            "TIED_LAYOUT": "True",
            "COMPUTE_TEACHER_CE": "False",
            "COMPUTE_KL": "compute_kl",
            "TEACHER_KL_GRAD": "False",
            "BOTH_T_ONE": "both_t_one",
            "KERNEL_STAGE": "KERNEL_STAGE_TOKENWISE",
            "ALLOW_TEACHER_NEG_INF_SUPPORT": "False",
            "RENYI_NEAR_ONE": "launch.renyi_near_one",
            "SCALE_GRADIENT_MATERIALIZATION": (
                "gradient_materialization_factor is not None"
            ),
        },
        "tied/trained.py": {
            "TIED_LAYOUT": "True",
            "COMPUTE_TEACHER_CE": "True",
            "COMPUTE_KL": "compute_kl",
            "TEACHER_KL_GRAD": "False",
            "BOTH_T_ONE": "both_t_one",
            "KERNEL_STAGE": "KERNEL_STAGE_TOKENWISE",
            "ALLOW_TEACHER_NEG_INF_SUPPORT": "False",
            "RENYI_NEAR_ONE": "launch.renyi_near_one",
            "SCALE_GRADIENT_MATERIALIZATION": (
                "gradient_materialization_factor is not None"
            ),
        },
        "tied/cotrained.py": {
            "TIED_LAYOUT": "True",
            "COMPUTE_TEACHER_CE": "compute_teacher_ce",
            "COMPUTE_KL": "compute_kl",
            "TEACHER_KL_GRAD": "True",
            "BOTH_T_ONE": "both_t_one",
            "KERNEL_STAGE": "KERNEL_STAGE_TOKENWISE",
            "ALLOW_TEACHER_NEG_INF_SUPPORT": "False",
            "RENYI_NEAR_ONE": "launch.renyi_near_one",
            "SCALE_GRADIENT_MATERIALIZATION": (
                "gradient_materialization_factor is not None"
            ),
        },
        "precomputed/tokenwise.py": {
            "TIED_LAYOUT": "False",
            "COMPUTE_TEACHER_CE": "False",
            "COMPUTE_KL": "compute_kl",
            "TEACHER_KL_GRAD": "False",
            "BOTH_T_ONE": "both_t_one",
            "KERNEL_STAGE": "KERNEL_STAGE_TOKENWISE",
            "ALLOW_TEACHER_NEG_INF_SUPPORT": "allow_teacher_neg_inf_support",
            "RENYI_NEAR_ONE": "launch.renyi_near_one",
            "SCALE_GRADIENT_MATERIALIZATION": (
                "gradient_materialization_factor is not None"
            ),
        },
        "precomputed/aligned_span.py": (
            {
                "TIED_LAYOUT": "False",
                "COMPUTE_TEACHER_CE": "False",
                "COMPUTE_KL": "False",
                "TEACHER_KL_GRAD": "False",
                "BOTH_T_ONE": "True",
                "KERNEL_STAGE": "KERNEL_STAGE_ALIGNED_STATS",
                "ALLOW_TEACHER_NEG_INF_SUPPORT": "False",
                "RENYI_NEAR_ONE": "False",
                "SCALE_GRADIENT_MATERIALIZATION": "False",
            },
            {
                "TIED_LAYOUT": "False",
                "COMPUTE_TEACHER_CE": "False",
                "COMPUTE_KL": "False",
                "TEACHER_KL_GRAD": "False",
                "BOTH_T_ONE": "True",
                "KERNEL_STAGE": "KERNEL_STAGE_ALIGNED_GRAD",
                "ALLOW_TEACHER_NEG_INF_SUPPORT": "False",
                "RENYI_NEAR_ONE": "False",
                "SCALE_GRADIENT_MATERIALIZATION": (
                    "gradient_materialization_scale is not None"
                ),
            },
        ),
        "separate/frozen.py": {
            "TIED_LAYOUT": "False",
            "COMPUTE_TEACHER_CE": "False",
            "COMPUTE_KL": "compute_kl",
            "TEACHER_KL_GRAD": "False",
            "BOTH_T_ONE": "both_t_one",
            "KERNEL_STAGE": "KERNEL_STAGE_TOKENWISE",
            "ALLOW_TEACHER_NEG_INF_SUPPORT": "False",
            "RENYI_NEAR_ONE": "launch.renyi_near_one",
            "SCALE_GRADIENT_MATERIALIZATION": (
                "gradient_materialization_factor is not None"
            ),
        },
        "separate/trained.py": {
            "TIED_LAYOUT": "False",
            "COMPUTE_TEACHER_CE": "True",
            "COMPUTE_KL": "compute_kl",
            "TEACHER_KL_GRAD": "False",
            "BOTH_T_ONE": "both_t_one",
            "KERNEL_STAGE": "KERNEL_STAGE_TOKENWISE",
            "ALLOW_TEACHER_NEG_INF_SUPPORT": "False",
            "RENYI_NEAR_ONE": "launch.renyi_near_one",
            "SCALE_GRADIENT_MATERIALIZATION": (
                "gradient_materialization_factor is not None"
            ),
        },
        "separate/cotrained.py": {
            "TIED_LAYOUT": "False",
            "COMPUTE_TEACHER_CE": "compute_teacher_ce",
            "COMPUTE_KL": "compute_kl",
            "TEACHER_KL_GRAD": "True",
            "BOTH_T_ONE": "both_t_one",
            "KERNEL_STAGE": "KERNEL_STAGE_TOKENWISE",
            "ALLOW_TEACHER_NEG_INF_SUPPORT": "False",
            "RENYI_NEAR_ONE": "launch.renyi_near_one",
            "SCALE_GRADIENT_MATERIALIZATION": (
                "gradient_materialization_factor is not None"
            ),
        },
    }

    for relative_path, expected_axes in expected.items():
        source = (_KERNEL_ROOT / relative_path).read_text(encoding="utf-8")
        module = ast.parse(source)
        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ce_kl_fwdbwd_kernel"
        ]
        expected_calls = (
            expected_axes if isinstance(expected_axes, tuple) else (expected_axes,)
        )
        assert len(calls) == len(expected_calls)
        actual_calls = [
            {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in call.keywords
                if keyword.arg in expected_call
            }
            for call, expected_call in zip(calls, expected_calls, strict=True)
        ]
        assert actual_calls == list(expected_calls)


def test_every_launch_satisfies_the_canonical_kernel_signature():
    module = ast.parse(_CANONICAL_KERNEL.read_text(encoding="utf-8"))
    kernels = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name == "ce_kl_fwdbwd_kernel"
        and any(
            ast.unparse(decorator) == "triton.jit" for decorator in node.decorator_list
        )
    ]
    assert len(kernels) == 1

    signature = kernels[0].args
    positional = [argument.arg for argument in signature.args]
    required_count = len(positional) - len(signature.defaults)
    required = set(positional[:required_count])
    parameters = set(positional)
    failures = []

    for relative_path in _MODE_FILES:
        wrapper = ast.parse((_KERNEL_ROOT / relative_path).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ce_kl_fwdbwd_kernel"
        ]
        expected_count = 2 if relative_path == "precomputed/aligned_span.py" else 1
        assert len(calls) == expected_count
        for call in calls:
            assert not any(isinstance(argument, ast.Starred) for argument in call.args)
            assert all(keyword.arg is not None for keyword in call.keywords)

            positional_names = set(positional[: len(call.args)])
            keyword_names = {keyword.arg for keyword in call.keywords}
            provided = positional_names | keyword_names
            missing = sorted(required - provided)
            duplicate = sorted(positional_names & keyword_names)
            unknown = sorted(keyword_names - parameters - _TRITON_LAUNCH_META)
            if len(call.args) > len(positional) or missing or duplicate or unknown:
                failures.append(
                    {
                        "path": relative_path,
                        "missing": missing,
                        "duplicate": duplicate,
                        "unknown": unknown,
                        "extra_positional": max(0, len(call.args) - len(positional)),
                    }
                )

    assert failures == []


def test_renyi_control_variate_sweep_is_compile_time_gated():
    source = _CANONICAL_KERNEL.read_text(encoding="utf-8")
    module = ast.parse(source)
    gates = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "RENYI_NEAR_ONE"
    ]

    assert len(gates) == 1
    gated_source = ast.unparse(gates[0])
    assert "for start in range(0, n_cols, BLOCK_SIZE)" in gated_source
    assert "renyi_log_a = renyi_log_a_lse" in gated_source
    assert "RENYI_STABLE_SWEEP" not in source
    assert "tl.abs(kl_alpha - 1.0) <= 0.125" not in source


def test_aligned_stages_reuse_the_canonical_pointer_slots():
    source = (_KERNEL_ROOT / "precomputed/aligned_span.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    expected = {
        "_aligned_span_stats_fwdbwd": [
            "logits_chunk",
            "logits_chunk.stride(0)",
            "logits_chunk",
            "logits_chunk.stride(0)",
            "labels_chunk",
            "labels_chunk.stride(0)",
            "nll_out",
            "lse_out",
            "nll_out",
        ],
        "_aligned_span_grad_fwdbwd": [
            "logits_chunk",
            "logits_chunk.stride(0)",
            "logits_chunk",
            "logits_chunk.stride(0)",
            "labels_chunk",
            "labels_chunk.stride(0)",
            "row_coefficients",
            "lse",
            "row_coefficients",
        ],
    }

    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in expected:
            continue
        calls = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Subscript)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "ce_kl_fwdbwd_kernel"
        ]
        assert len(calls) == 1
        assert [ast.unparse(value) for value in calls[0].args[:9]] == expected[
            node.name
        ]


def test_aligned_grad_stage_pins_zero_coefficients_before_probability_math():
    source = (_KERNEL_ROOT / "_kl_math.py").read_text(encoding="utf-8")

    assert "active = coefficient != 0.0" in source
    assert "safe_lse = tl.where(active, lse, 0.0)" in source
    assert "tl.where(active, gradient, 0.0)" in source
    assert (
        source.count("tl.load(\n                gradient_materialization_factor_ptr")
        == 1
    )
    assert "gradient = gradient * gradient_materialization_factor" in source


def test_materialization_scale_is_applied_only_at_gradient_stores():
    source = _CANONICAL_KERNEL.read_text(encoding="utf-8")
    teacher_scale = source.index(
        "teacher_grad = teacher_grad * gradient_materialization_factor"
    )
    teacher_store = source.index("tl.store(row_t + cols, teacher_grad", teacher_scale)
    student_scale = source.index(
        "grad_s = grad_s * gradient_materialization_factor",
        teacher_store,
    )
    student_store = source.index("tl.store(row_s + cols, grad_s", student_scale)

    assert teacher_scale < teacher_store < student_scale < student_store
    assert "tl.store(K_ptr + i, kl_row * kl_loss_scale)" in source


def test_tied_teacher_loss_alias_is_not_written_by_the_zero_fill_gate():
    source = _CANONICAL_KERNEL.read_text(encoding="utf-8")

    assert "elif TEACHER_KL_GRAD:" in source
    assert "elif TIED_LAYOUT or TEACHER_KL_GRAD:" not in source


def test_teacher_row_dead_has_a_default_for_empty_teacher_paths():
    source = _CANONICAL_KERNEL.read_text(encoding="utf-8")
    row_scalars = source.index("# ── Row scalars and CE")
    capability_branch = source.index("if ALLOW_TEACHER_NEG_INF_SUPPORT:", row_scalars)

    assert source.index("teacher_row_dead = False", row_scalars) < capability_branch


def test_aligned_stats_stage_only_writes_row_statistics():
    module = ast.parse(_CANONICAL_KERNEL.read_text(encoding="utf-8"))
    stage = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "KERNEL_STAGE == _KERNEL_STAGE_ALIGNED_STATS_C"
    )
    stores = [
        ast.unparse(call.args[0])
        for call in ast.walk(stage)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "tl.store"
    ]

    assert stores == ["L_ptr + i", "T_L_ptr + i"]
    assert any(isinstance(node, ast.Return) for node in stage.body)


def test_each_mode_builds_one_launch_plan_without_revalidating_label_dtype():
    for relative_path in _MODE_FILES:
        source = (_KERNEL_ROOT / relative_path).read_text(encoding="utf-8")
        assert source.count("select_full_batch_launch_plan(") == 1
        assert "labels.dtype" not in source


def test_each_mode_claims_cached_gradients_before_reading_saved_tensors():
    for relative_path in _MODE_FILES:
        source = (_KERNEL_ROOT / relative_path).read_text(encoding="utf-8")
        backward = source.index("    def backward(")
        claim = source.index("claim_cached_gradients(", backward)
        saved_tensors = source.index("ctx.saved_tensors", backward)
        assert claim < saved_tensors


def test_inactive_student_paths_zero_the_existing_weight_accumulator():
    for relative_path in (
        "precomputed/tokenwise.py",
        "separate/frozen.py",
        "tied/frozen.py",
    ):
        module = ast.parse((_KERNEL_ROOT / relative_path).read_text(encoding="utf-8"))
        inactive_branch = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "need_grad_weight and (not student_path_active)"
        )
        assert ast.unparse(inactive_branch.body[0]) == "grad_weight.zero_()"

        grad_weight_assignments = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "grad_weight"
                for target in node.targets
            )
        ]
        assert not any(
            isinstance(call, ast.Call) and ast.unparse(call.func) == "torch.zeros"
            for assignment in grad_weight_assignments
            for call in ast.walk(assignment.value)
        )
