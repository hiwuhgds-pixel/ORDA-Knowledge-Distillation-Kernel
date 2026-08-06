from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_ROOT = REPOSITORY_ROOT / ".github/workflows"
SCRIPTS_ROOT = REPOSITORY_ROOT / ".github/scripts"
PR_WORKFLOW = WORKFLOWS_ROOT / "pr.yml"
RELEASE_WORKFLOW = WORKFLOWS_ROOT / "release.yml"
EXPECTED_WORKFLOWS = frozenset({"pr.yml", "release.yml"})
RETIRED_CI_PATHS = (
    ".github/workflows/benchmark.yml",
    ".github/workflows/minimum-stack.yml",
    ".github/workflows/nightly.yml",
    ".github/scripts/accuracy_thresholds.json",
    ".github/scripts/check_benchmark_results.py",
    ".github/scripts/verify_gpu_runtime.py",
    "scripts/update_accuracy_thresholds.py",
)
FORBIDDEN_WORKFLOW_REFERENCES = (
    "self-hosted",
    ".github/workflows/benchmark.yml",
    ".github/workflows/minimum-stack.yml",
    ".github/workflows/nightly.yml",
    ".github/scripts/check_benchmark_results.py",
    ".github/scripts/verify_gpu_runtime.py",
    ".[gpu]",
    "-m gpu",
)
LOCAL_PYTHON_TARGET = re.compile(
    r"\bpython\s+(?P<target>(?:\.github|scripts|test)/[^\s]+\.py)"
)
JOB_DECLARATION = re.compile(r"^  (?P<name>[a-z][a-z0-9-]+):\n(?=    )", re.MULTILINE)


def _load_wheel_verifier():
    path = SCRIPTS_ROOT / "verify_wheel.py"
    spec = importlib.util.spec_from_file_location("_wheel_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow_jobs(workflow: str) -> set[str]:
    _, separator, jobs = workflow.partition("\njobs:\n")
    assert separator, "workflow must define jobs"
    return {match["name"] for match in JOB_DECLARATION.finditer(jobs)}


wheel_verifier = _load_wheel_verifier()


# ── Workflow contracts ───────────────────────────────────────────────────────
def test_personal_ci_keeps_only_pr_and_release_workflows():
    assert {path.name for path in WORKFLOWS_ROOT.glob("*.yml")} == EXPECTED_WORKFLOWS
    for relative_path in RETIRED_CI_PATHS:
        assert not (REPOSITORY_ROOT / relative_path).exists()


def test_remaining_workflows_do_not_reference_retired_gpu_or_benchmark_ci():
    for workflow_path in (PR_WORKFLOW, RELEASE_WORKFLOW):
        workflow = workflow_path.read_text(encoding="utf-8")
        for reference in FORBIDDEN_WORKFLOW_REFERENCES:
            assert reference not in workflow, (
                f"{workflow_path.name} retains retired CI reference: {reference}"
            )


def test_workflow_local_python_targets_exist():
    for workflow_path in (PR_WORKFLOW, RELEASE_WORKFLOW):
        workflow = workflow_path.read_text(encoding="utf-8")
        for match in LOCAL_PYTHON_TARGET.finditer(workflow):
            target = match["target"]
            assert (REPOSITORY_ROOT / target).is_file(), (
                f"{workflow_path.name} invokes missing local Python target: {target}"
            )


def test_pr_workflow_checks_quality_and_cpu_endpoints_only():
    workflow = PR_WORKFLOW.read_text(encoding="utf-8")
    assert _workflow_jobs(workflow) == {"quality", "cpu-full"}
    assert "Check autotune contract fallback digest" in workflow
    assert "--no-build-isolation" not in workflow
    assert 'python-version: "3.10"' in workflow
    assert 'torch-spec: "torch==2.3.0"' in workflow
    assert 'python-version: "3.13"' in workflow
    assert 'torch-spec: "torch>=2.3.0"' in workflow
    assert workflow.count("- python-version:") == 2
    assert workflow.count("torch-spec:") == 2
    assert (
        '-m "not gpu" test/unit test/correctness/test_torch_backend_cpu.py' in workflow
    )


def test_release_workflow_checks_installed_wheel_before_publish():
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert _workflow_jobs(workflow) == {
        "build-wheel",
        "release-contracts",
        "publish-pypi",
        "publish-github-release",
    }
    assert "needs: build-wheel" in workflow
    assert "needs: [build-wheel, release-contracts]" in workflow
    assert "--no-build-isolation" not in workflow
    assert "--no-deps --force-reinstall dist/*.whl" in workflow
    assert "assert not p.is_relative_to(source), p" in workflow
    assert workflow.count("python -m pytest") == 1
    assert "gh release create" in workflow


# ── Wheel verifier contracts ─────────────────────────────────────────────────
def test_wheel_verifier_rejects_release_tag_version_mismatch():
    wheel_verifier._require_expected_version("0.1.0", None)
    wheel_verifier._require_expected_version("0.1.0", "0.1.0")
    with pytest.raises(ValueError, match="does not match release tag"):
        wheel_verifier._require_expected_version("0.1.0", "0.2.0")
    with pytest.raises(ValueError, match="must be non-empty"):
        wheel_verifier._require_expected_version("0.1.0", "  ")
