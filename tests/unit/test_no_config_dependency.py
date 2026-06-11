from __future__ import annotations

import ast
from pathlib import Path


CORE_FILES = [
    Path("src/orda_ce_kernel/api.py"),
    Path("src/orda_ce_kernel/ops/cross_entropy.py"),
    Path("src/orda_ce_kernel/ops/kl_kernel.py"),
    Path("src/orda_ce_kernel/ops/kernels.py"),
    Path("src/orda_ce_kernel/utils/dispatcher.py"),
    Path("src/orda_ce_kernel/utils/resolver.py"),
]


def _imports_for(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_core_runtime_imports_stay_inside_runtime_option_boundary():
    offenders = []
    for path in CORE_FILES:
        for imported in _imports_for(path):
            if imported == "yaml" or imported.endswith(".config") or imported == "orda_ce_kernel.config":
                offenders.append(f"{path}: {imported}")
    assert offenders == []


def test_runtime_options_are_not_file_backed():
    assert not Path("src/orda_ce_kernel/config.py").exists()
    assert not Path("configs/config.yaml").exists()


