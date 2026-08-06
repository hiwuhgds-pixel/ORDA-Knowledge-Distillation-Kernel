from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = _REPO_ROOT / "src" / "orda_kernel"


def test_pep561_marker_is_declared_and_present():
    assert (_PACKAGE_ROOT / "py.typed").is_file()
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in pyproject
    assert 'orda_kernel = ["py.typed"]' in pyproject


def test_only_one_launchable_triton_kernel_exists():
    kernel_root = _PACKAGE_ROOT / "ops" / "_kernels"
    launched_functions = []
    for path in kernel_root.rglob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name)
            ):
                launched_functions.append(node.func.value.id)

    assert set(launched_functions) == {"ce_kl_fwdbwd_kernel"}


def test_canonical_kernel_handles_ignore_index_before_safe_label_guard():
    source = (_PACKAGE_ROOT / "ops" / "_kernels" / "ce_kl_fwdbwd.py").read_text(
        encoding="utf-8"
    )
    ignored_branch = source.index("if ignored:")
    ignored_return = source.index("            return", ignored_branch)
    safe_label = source.index("y_safe = tl.where(~in_range, 0, y)")

    assert ignored_branch < ignored_return < safe_label
    assert "row_s + y_safe" in source
    assert "row_t + y_safe" in source


def test_multichunk_aligned_span_releases_statistics_logits_before_gradient_pass():
    source = (
        _PACKAGE_ROOT / "ops" / "_kernels" / "precomputed" / "aligned_span.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    deletes = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Delete)
        and any(
            isinstance(target, ast.Name) and target.id == "logits_chunk"
            for target in node.targets
        )
    ]
    assert len(deletes) == 1
    assert source.index("del logits_chunk") < source.index(
        "grad_logits_chunk =",
        source.index("del logits_chunk"),
    )
