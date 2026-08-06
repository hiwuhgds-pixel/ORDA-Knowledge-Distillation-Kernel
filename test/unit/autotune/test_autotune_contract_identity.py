from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from utils.autotune import identity

from orda_kernel._autotune import _contract as contract
from orda_kernel._autotune.models import (
    AUTOTUNE_SCHEMA_VERSION,
    AUTOTUNE_TIMED_CONTRACT_ID,
)


# ── Timed contract guard ─────────────────────────────────────────────────────
def test_registered_fallback_matches_the_current_timed_contract_surface():
    # This is the CI guard for source, editable, and wheel fallback identities.
    assert (
        contract.current_source_contract_digest()
        == contract.AUTOTUNE_TIMED_CONTRACT_FALLBACK_DIGEST
    )
    assert (
        contract.AUTOTUNE_TIMED_CONTRACT_DIGEST
        == contract.AUTOTUNE_TIMED_CONTRACT_FALLBACK_DIGEST
    )
    assert AUTOTUNE_TIMED_CONTRACT_ID == contract.contract_id_for_source_digest(
        contract.AUTOTUNE_TIMED_CONTRACT_FALLBACK_DIGEST
    )


def test_source_signature_changes_contract_id_but_comments_and_unrelated_files_do_not():
    sources = {
        path: "def run(config):\n    return config\n"
        for path in contract.AUTOTUNE_TIMED_CONTRACT_SOURCES
    }
    baseline_digest = contract.source_contract_digest(sources)
    baseline_id = contract.contract_id_for_source_digest(baseline_digest)

    comment_only = dict(sources)
    first_path = contract.AUTOTUNE_TIMED_CONTRACT_SOURCES[0]
    comment_only[first_path] = "# documentation only\n" + comment_only[first_path]
    assert contract.source_contract_digest(comment_only) == baseline_digest

    with_unrelated_file = {**sources, "unrelated_module.py": "value = 1\n"}
    assert contract.source_contract_digest(with_unrelated_file) == baseline_digest

    with_imports = dict(sources)
    with_imports[first_path] = (
        "import alpha\nimport beta\n\ndef run(config):\n    return config\n"
    )
    reordered_imports = dict(with_imports)
    reordered_imports[first_path] = (
        "import beta\nimport alpha\n\ndef run(config):\n    return config\n"
    )
    assert contract.source_contract_digest(
        reordered_imports
    ) != contract.source_contract_digest(with_imports)

    shadowing_a = dict(sources)
    shadowing_a[first_path] = (
        "from alpha import launch as selected\nfrom beta import launch as selected\n"
    )
    shadowing_b = dict(sources)
    shadowing_b[first_path] = (
        "from beta import launch as selected\nfrom alpha import launch as selected\n"
    )
    assert contract.source_contract_digest(
        shadowing_a
    ) != contract.source_contract_digest(shadowing_b)

    changed_signature = dict(sources)
    changed_signature[first_path] = (
        "def run(config, *, num_warps):\n    return config\n"
    )
    changed_id = contract.contract_id_for_source_digest(
        contract.source_contract_digest(changed_signature)
    )
    assert changed_id != baseline_id


@pytest.mark.parametrize(
    "error_type",
    [
        OSError,
        UnicodeError,
        SyntaxError,
        LookupError,
        ImportError,
        TypeError,
        AttributeError,
    ],
)
def test_source_unavailable_uses_the_checked_in_wheel_fallback(error_type):
    def unavailable_source() -> str:
        raise error_type("source resources unavailable")

    assert (
        contract.resolve_source_contract_digest(unavailable_source)
        == contract.AUTOTUNE_TIMED_CONTRACT_FALLBACK_DIGEST
    )


@pytest.mark.parametrize("digest", ["", "not-a-digest", "f" * 63, "g" * 64])
def test_source_loader_rejects_invalid_digest(digest):
    with pytest.raises(ValueError, match="SHA-256"):
        contract.resolve_source_contract_digest(lambda: digest)


def test_normalized_dump_is_interpreter_stable_across_python_versions():
    source = "def run(config):\n    return config\n\nclass K:\n    pass\n"
    normalized = contract._normalized_module(source, "sample.py")
    assert normalized == (
        "Module(body=[FunctionDef(name='run', args=arguments(posonlyargs=[], "
        "args=[arg(arg='config')], kwonlyargs=[], kw_defaults=[], defaults=[]), "
        "body=[Return(value=Name(id='config', ctx=Load()))], decorator_list=[]), "
        "ClassDef(name='K', bases=[], keywords=[], body=[Pass()], "
        "decorator_list=[])], type_ignores=[])"
    )


def test_timed_contract_includes_shared_helpers_and_split_tied_paths():
    for path in (
        "_execution/aligned_span.py",
        "_runtime.py",
        "_torch_backend.py",
        "ops/_common.py",
        "ops/precomputed_inputs/_aligned_span_math.py",
        "ops/_kernels/_kl_math.py",
        "ops/_kernels/_tokenwise_launch.py",
        "ops/_kernels/precomputed/aligned_span.py",
        "ops/_kernels/tied/frozen.py",
        "ops/_kernels/tied/trained.py",
        "ops/_kernels/tied/cotrained.py",
    ):
        assert path in contract.AUTOTUNE_TIMED_CONTRACT_SOURCES
    assert "ops/_kernels/tied/tied.py" not in contract.AUTOTUNE_TIMED_CONTRACT_SOURCES


# ── Cache identity ───────────────────────────────────────────────────────────
def test_timed_contract_id_separates_hot_and_persistent_workload_identity():
    runtime_key, workload = identity(torch)
    alternate_contract_id = contract.contract_id_for_source_digest("e" * 64)

    alternate_runtime_key = runtime_key._replace(
        timed_contract_id=alternate_contract_id
    )
    alternate_workload = replace(workload, timed_contract_id=alternate_contract_id)
    payload = json.loads(workload.disk_key())

    assert runtime_key.timed_contract_id == AUTOTUNE_TIMED_CONTRACT_ID
    assert alternate_runtime_key != runtime_key
    assert alternate_workload.disk_key() != workload.disk_key()
    assert payload["schema_version"] == AUTOTUNE_SCHEMA_VERSION
    assert payload["timed_contract_id"] == AUTOTUNE_TIMED_CONTRACT_ID
    assert "kernel_revision" not in payload
