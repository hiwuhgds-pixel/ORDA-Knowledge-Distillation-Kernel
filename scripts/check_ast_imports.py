"""Cross-platform import-hygiene + AST check for the test suite.

Run from the repository root::

    python scripts/check_ast_imports.py

Exits non-zero if any file under ``tests/`` fails to parse, or if any Python
file under ``tests/`` mutates the import path with ``sys.path`` /
``sitecustomize`` / ``PYTHONPATH``.

The check ignores Markdown / doc strings / comments — only Python code is
parsed and inspected. Use this script in CI instead of bash heredocs or
PowerShell here-strings so the same command works on Linux, macOS, and
Windows.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


FORBIDDEN_PATTERN = re.compile(r"\b(sys\.path|sitecustomize|PYTHONPATH)\b")


def check_ast(repo_root: Path, extra_paths: list[Path]) -> list[tuple[Path, str]]:
    failures: list[tuple[Path, str]] = []
    for py_path in list(repo_root.joinpath("tests").rglob("*.py")) + extra_paths:
        if not py_path.exists() or py_path.suffix != ".py":
            continue
        try:
            ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        except SyntaxError as exc:
            failures.append((py_path, f"SyntaxError: {exc}"))
    return failures


def check_import_hygiene(repo_root: Path) -> list[tuple[Path, int, str]]:
    findings: list[tuple[Path, int, str]] = []
    for py_path in repo_root.joinpath("tests").rglob("*.py"):
        try:
            text = py_path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(text, filename=str(py_path))
        except SyntaxError:
            # Already reported by check_ast.
            continue
        # Walk AST attribute/name nodes — comments and docstrings are
        # naturally excluded because they are not part of the AST surface
        # we inspect here.
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                qualified = _resolve_attr(node)
                if qualified and FORBIDDEN_PATTERN.search(qualified):
                    findings.append((py_path, node.lineno, qualified))
            elif isinstance(node, ast.Name) and FORBIDDEN_PATTERN.search(node.id):
                findings.append((py_path, node.lineno, node.id))
    return findings


def _resolve_attr(node: ast.Attribute) -> str | None:
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="AST + import-hygiene gate")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="Repository root (defaults to current working directory).")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    pyproject = repo_root / "pyproject.toml"
    extra_paths = [pyproject] if pyproject.exists() else []

    ast_failures = check_ast(repo_root, extra_paths)
    hygiene_findings = check_import_hygiene(repo_root)

    if ast_failures:
        print("[FAIL] AST parse errors:")
        for path, msg in ast_failures:
            print(f"  {path}: {msg}")
    else:
        print("[OK] AST parse: clean.")

    if hygiene_findings:
        print("[FAIL] Forbidden import-path mutations found:")
        for path, lineno, snippet in hygiene_findings:
            print(f"  {path}:{lineno}: {snippet}")
    else:
        print("[OK] Import hygiene: no sys.path / sitecustomize / PYTHONPATH usage.")

    return 1 if (ast_failures or hygiene_findings) else 0


if __name__ == "__main__":
    sys.exit(main())
