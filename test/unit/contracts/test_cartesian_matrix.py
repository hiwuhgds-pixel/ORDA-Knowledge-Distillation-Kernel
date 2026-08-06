"""Integrity contracts for the exhaustive Cartesian correctness matrix."""

from __future__ import annotations

import pytest
import run_test_cpu
import run_test_gpu
import utils.matrices as matrices
from utils.matrices import (
    API_LEVELS,
    CARTESIAN_ROUTE_CASE_COUNT,
    CARTESIAN_ROUTES,
    CHUNK_POLICIES,
    DTYPES,
    EXPECTED_OUTCOME_COUNTS,
    KL_VARIANTS,
    MATRIX_STRIDES,
    TEACHER_REPRESENTATIONS,
    TOTAL_CARTESIAN_CASES,
    CartesianSelection,
    ExpectedOutcome,
    ShardSpec,
    cartesian_case_from_ordinal,
    cartesian_group_key,
    cartesian_ordinal_for_case,
    cartesian_outcome_counts,
    cartesian_route_ordinal_range,
    classify_case,
    group_cartesian_cases,
    iter_cartesian_shard,
    iter_selected_cartesian_cases,
    parse_cartesian_route,
    selected_cartesian_cases_for_shard,
    shard_case_count,
)


# ── Cartesian matrix integrity ───────────────────────────────────────────────
def test_cartesian_matrix_analytical_counts_cover_the_full_product():
    counts = cartesian_outcome_counts()
    assert sum(counts.values()) == TOTAL_CARTESIAN_CASES == 2_073_600
    assert counts == EXPECTED_OUTCOME_COUNTS
    assert set(counts) == set(ExpectedOutcome)
    assert len(CARTESIAN_ROUTES) == 4
    assert CARTESIAN_ROUTE_CASE_COUNT == 518_400
    assert len(CARTESIAN_ROUTES) * CARTESIAN_ROUTE_CASE_COUNT == 2_073_600


def test_cartesian_matrix_uses_the_dynamic_chunk_policy_name():
    assert "dynamic" in CHUNK_POLICIES
    assert "dynamic_auto" not in CHUNK_POLICIES
    assert "autotune_on" not in CHUNK_POLICIES


def test_cartesian_shards_partition_the_full_space_without_sampling():
    shard_count = 7
    declared_counts = [
        shard_case_count(ShardSpec(shard_count, index)) for index in range(shard_count)
    ]

    assert sum(declared_counts) == TOTAL_CARTESIAN_CASES
    assert max(declared_counts) - min(declared_counts) <= 1

    selected = ShardSpec(shard_count, 3)
    observed = 0
    for case in iter_cartesian_shard(selected):
        assert case.ordinal % selected.count == selected.index
        observed += 1
    assert observed == shard_case_count(selected)


def test_cartesian_routes_are_a_disjoint_union_of_global_ordinals():
    expected_start = 0
    for route in CARTESIAN_ROUTES:
        ordinal_range = cartesian_route_ordinal_range(route)
        assert ordinal_range.start == expected_start
        assert ordinal_range.step == 1
        assert len(ordinal_range) == CARTESIAN_ROUTE_CASE_COUNT
        expected_start = ordinal_range.stop
    assert expected_start == TOTAL_CARTESIAN_CASES


def test_cartesian_ordinal_roundtrip_preserves_original_product_order():
    ordinals = (
        0,
        1,
        4,
        MATRIX_STRIDES[1] - 1,
        MATRIX_STRIDES[1],
        MATRIX_STRIDES[0],
        TOTAL_CARTESIAN_CASES - 1,
    )
    cases = tuple(cartesian_case_from_ordinal(ordinal) for ordinal in ordinals)

    assert all(cartesian_ordinal_for_case(case) == case.ordinal for case in cases)
    assert cases[0].teacher_representation == "tied"
    assert cases[0].kl_variant == "forward"
    assert cases[1].api_level == "direct_tied"
    assert cases[2].label_pattern == "with_ignore_index"
    assert cases[4].kl_variant == "jeffreys"
    assert cases[5].teacher_representation == "separate"
    assert cases[-1].teacher_representation == "precomputed_hidden_weight"
    assert cases[-1].kl_variant == "triangular"
    assert cases[-1].api_level == "direct_precomputed"


def test_route_filters_do_not_call_the_full_matrix_generator(monkeypatch):
    def fail_if_materialized():
        raise AssertionError("full Cartesian generator must not be called")

    monkeypatch.setattr(matrices, "iter_cartesian_cases", fail_if_materialized)
    selection = CartesianSelection(
        routes=(parse_cartesian_route("tied"),),
        kl_variants=("forward",),
        dtypes=("float32",),
        api_levels=("high_level",),
        chunk_policies=("dynamic",),
        label_patterns=("invalid_negative",),
        outcomes=(ExpectedOutcome.REJECT_INVALID_LABEL,),
    )
    cases = tuple(iter_selected_cartesian_cases(selection))

    assert len(cases) == 144
    assert list(cases) == sorted(cases, key=lambda case: case.ordinal)
    assert all(case.teacher_representation == "tied" for case in cases)
    assert all(case.kl_variant == "forward" for case in cases)
    assert all(case.dtype == "float32" for case in cases)
    assert all(case.api_level == "high_level" for case in cases)
    assert all(case.chunk_policy == "dynamic" for case in cases)
    assert all(case.label_pattern == "invalid_negative" for case in cases)
    assert all(
        classify_case(case) is ExpectedOutcome.REJECT_INVALID_LABEL for case in cases
    )


def test_exact_ordinal_filter_decodes_directly_and_intersects_route_filters():
    selection = CartesianSelection(
        routes=(parse_cartesian_route("tied"),),
        kl_variants=("forward",),
        ordinals=(TOTAL_CARTESIAN_CASES - 1, 0, MATRIX_STRIDES[1], 0),
    )

    assert tuple(iter_selected_cartesian_cases(selection)) == (
        cartesian_case_from_ordinal(0),
    )


def test_cartesian_selection_errors_only_when_globally_empty():
    globally_empty = CartesianSelection(
        routes=(parse_cartesian_route("tied"),),
        kl_variants=("forward",),
        ordinals=(TOTAL_CARTESIAN_CASES - 1,),
    )
    with pytest.raises(ValueError, match="matched no cases before sharding"):
        selected_cartesian_cases_for_shard(globally_empty, ShardSpec(1, 0))

    one_case = CartesianSelection(ordinals=(0,))
    assert selected_cartesian_cases_for_shard(one_case, ShardSpec(2, 1)) == ()


def test_cartesian_groups_keep_the_existing_key_on_a_filtered_route():
    selection = CartesianSelection(
        routes=(parse_cartesian_route("tied"),),
        kl_variants=("forward",),
        dtypes=("float32",),
        api_levels=("high_level",),
        chunk_policies=("dynamic",),
        label_patterns=("all_valid",),
    )
    cases = tuple(iter_selected_cartesian_cases(selection))
    groups = group_cartesian_cases(cases)
    expected_group_count = (
        len(TEACHER_REPRESENTATIONS) * len(KL_VARIANTS) * len(DTYPES) * len(API_LEVELS)
    )

    assert expected_group_count == 720
    assert TOTAL_CARTESIAN_CASES // expected_group_count == 2_880
    assert len(groups) == 1
    assert sum(len(members) for members in groups.values()) == len(cases) == 144
    assert all(
        cartesian_group_key(case) == key
        for key, members in groups.items()
        for case in members
    )


def test_cartesian_runner_accepts_an_interactive_numeric_selection(
    monkeypatch,
    capsys,
):
    calls = []

    monkeypatch.setattr("builtins.input", lambda _prompt: "7")
    monkeypatch.setattr(
        run_test_gpu,
        "run_logged",
        lambda command, **kwargs: calls.append((command, kwargs)) or 0,
    )

    assert run_test_gpu._main(["-q"]) == 0
    assert len(calls) == 1
    assert calls[0][1]["env"]["ORDA_CARTESIAN_ROUTES"] == "separate"
    output = capsys.readouterr().out
    assert "TEST: GPU suite" in output
    assert "── test 1/1 · cartesian route 1/1 · separate " in output
    assert "=== Cartesian route" not in output


def test_cartesian_runner_reports_filtered_interactive_case_count(
    monkeypatch,
    capsys,
):
    calls = []
    monkeypatch.setattr("builtins.input", lambda _prompt: "6-9")
    monkeypatch.setattr(
        run_test_gpu,
        "run_logged",
        lambda command, **kwargs: calls.append((command, kwargs)) or 0,
    )

    assert run_test_gpu._main(["--kl", "forward", "--dtype", "float32", "-q"]) == 0
    assert len(calls) == len(CARTESIAN_ROUTES)
    assert [kwargs["env"]["ORDA_CARTESIAN_ROUTES"] for _, kwargs in calls] == [
        route.route_id for route in CARTESIAN_ROUTES
    ]


def test_cartesian_runner_all_uses_one_sequential_subprocess_per_route(
    monkeypatch,
    capsys,
):
    calls = []

    def record_run(command, **kwargs):
        calls.append((command, kwargs))
        return 0

    monkeypatch.setattr(run_test_gpu, "run_logged", record_run)

    assert (
        run_test_gpu._main(
            [
                *(
                    item
                    for route in CARTESIAN_ROUTES
                    for item in ("--test", f"cartesian-{route.route_id}")
                ),
                "--kl",
                "forward",
                "--dtype",
                "float32",
                "-q",
            ]
        )
        == 0
    )
    assert len(calls) == len(CARTESIAN_ROUTES) == 4
    assert [kwargs["env"]["ORDA_CARTESIAN_ROUTES"] for _, kwargs in calls] == [
        route.route_id for route in CARTESIAN_ROUTES
    ]
    assert all(
        kwargs["env"]["ORDA_CARTESIAN_DTYPES"] == "float32" for _, kwargs in calls
    )
    assert all(
        kwargs["env"]["ORDA_CARTESIAN_KL_VARIANTS"] == "forward" for _, kwargs in calls
    )
    assert all(command[-1] == "-q" for command, _ in calls)
    output = capsys.readouterr().out
    assert (
        " routes : tied, separate, precomputed_logits, precomputed_hidden_weight"
        in output
    )
    assert output.count("· cartesian route ") == len(CARTESIAN_ROUTES)
    assert "=== Cartesian route" not in output


def test_gpu_runner_selects_one_direct_suite(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(
        run_test_gpu,
        "run_logged",
        lambda command, **kwargs: calls.append((command, kwargs)) or 0,
    )

    assert run_test_gpu._main(["--test", "public-api", "-q"]) == 0
    assert len(calls) == 1
    assert calls[0][0][-2:] == [
        str(run_test_gpu.GPU_TESTS["public-api"]),
        "-q",
    ]
    output = capsys.readouterr().out
    assert " tests  : public-api" in output
    assert "── test 1/1 · public-api " in output


def test_gpu_runner_route_shorthand_selects_cartesian_route(monkeypatch):
    calls = []

    monkeypatch.setattr(
        run_test_gpu,
        "run_logged",
        lambda command, **kwargs: calls.append((command, kwargs)) or 0,
    )

    assert run_test_gpu._main(["--route", "tied", "--dtype", "float32", "-q"]) == 0
    assert len(calls) == 1
    assert calls[0][1]["env"]["ORDA_CARTESIAN_ROUTES"] == "tied"


def test_gpu_runner_all_runs_every_gpu_test(monkeypatch):
    calls = []

    monkeypatch.setattr(
        run_test_gpu,
        "run_logged",
        lambda command, **kwargs: calls.append((command, kwargs)) or 0,
    )

    assert run_test_gpu._main(["--all", "--dtype", "float32", "-q"]) == 0
    direct_count = len(run_test_gpu.GPU_TESTS) - len(CARTESIAN_ROUTES)
    assert len(calls) == direct_count + len(CARTESIAN_ROUTES)
    assert [
        kwargs["env"]["ORDA_CARTESIAN_ROUTES"] for _, kwargs in calls[direct_count:]
    ] == [route.route_id for route in CARTESIAN_ROUTES]


def test_cpu_runner_collects_only_cpu_paths(monkeypatch):
    calls = []

    monkeypatch.setattr(
        run_test_cpu,
        "run_logged",
        lambda command, *, cwd, env=None: calls.append((command, env)) or 0,
    )

    assert run_test_cpu._main(["-q"]) == 0
    command, _ = calls[0]
    assert str(run_test_cpu.TEST_ROOT / "unit") in command
    assert (
        str(run_test_cpu.TEST_ROOT / "correctness" / "test_torch_backend_cpu.py")
        in command
    )
    assert (
        str(run_test_cpu.TEST_ROOT / "nightly" / "test_autotune_multiprocess.py")
        in command
    )
    assert not any("_cuda.py" in argument for argument in command)
    assert str(run_test_cpu.TEST_ROOT / "run_test_gpu.py") not in command
