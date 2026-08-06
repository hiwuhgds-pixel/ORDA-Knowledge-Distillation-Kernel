"""Stable identity for the code exercised by the autotune benchmark."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from importlib.resources import files

# ── Timed contract surface ───────────────────────────────────────────────────
# These modules define the timed full-BT pipeline and launch consumption.
AUTOTUNE_TIMED_CONTRACT_FORMAT = "ast-v3"
AUTOTUNE_TIMED_CONTRACT_SOURCES = (
    "_autotune/benchmark.py",
    "_autotune/keys.py",
    "_autotune/models.py",
    "_autotune/tuner.py",
    "_execution/dispatcher.py",
    "_execution/aligned_span.py",
    "_execution/recovery.py",
    "_runtime.py",
    "_torch_backend.py",
    "ops/_common.py",
    "ops/precomputed_inputs/_aligned_span_math.py",
    "ops/_kernels/_kl_math.py",
    "ops/_kernels/_tokenwise_launch.py",
    "ops/_kernels/ce_kl_fwdbwd.py",
    "ops/_kernels/tied/frozen.py",
    "ops/_kernels/tied/trained.py",
    "ops/_kernels/tied/cotrained.py",
    "ops/_kernels/precomputed/tokenwise.py",
    "ops/_kernels/precomputed/aligned_span.py",
    "ops/_kernels/separate/frozen.py",
    "ops/_kernels/separate/trained.py",
    "ops/_kernels/separate/cotrained.py",
)


# ── AST normalization ────────────────────────────────────────────────────────
class _StripDocstrings(ast.NodeTransformer):
    """Remove non-executable documentation before producing a source digest."""

    @staticmethod
    def _strip(node: ast.AST) -> ast.AST:
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)


def _stable_ast_dump(node: ast.AST) -> str:
    if sys.version_info >= (3, 13):
        return ast.dump(
            node,
            annotate_fields=True,
            include_attributes=False,
            show_empty=True,
        )
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _normalized_module(source: str, path: str) -> str:
    module = _StripDocstrings().visit(ast.parse(source, filename=path))
    if not isinstance(module, ast.Module):  # pragma: no cover - ast.parse invariant
        raise TypeError(f"expected a module AST for {path}")
    return _strip_version_variant_fields(_stable_ast_dump(module))


# Strip empty AST fields introduced by newer supported Python versions.
_VERSION_VARIANT_FIELDS = ("type_params",)
_VERSION_VARIANT_ALTERNATION = "|".join(
    re.escape(name) for name in _VERSION_VARIANT_FIELDS
)
_VERSION_VARIANT_RE = re.compile(rf",\s*(?:{_VERSION_VARIANT_ALTERNATION})=\[\]")


def _strip_version_variant_fields(dumped: str) -> str:
    return _VERSION_VARIANT_RE.sub("", dumped)


# ── Digest computation ───────────────────────────────────────────────────────
def source_contract_digest(sources: Mapping[str, str]) -> str:
    """Return a deterministic AST digest for the declared timed surface."""
    missing = [path for path in AUTOTUNE_TIMED_CONTRACT_SOURCES if path not in sources]
    if missing:
        raise ValueError(f"missing autotune timed-contract sources: {missing}")

    normalized = {
        path: _normalized_module(sources[path], path)
        for path in AUTOTUNE_TIMED_CONTRACT_SOURCES
    }
    payload = json.dumps(
        {
            "format": AUTOTUNE_TIMED_CONTRACT_FORMAT,
            "modules": normalized,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_source_contract_digest() -> str:
    """Read the packaged source once at import, never on a cache hot path."""
    package_root = files("orda_kernel")
    sources = {
        path: package_root.joinpath(*path.split("/")).read_text(encoding="utf-8")
        for path in AUTOTUNE_TIMED_CONTRACT_SOURCES
    }
    return source_contract_digest(sources)


def contract_id_for_source_digest(source_digest: str) -> str:
    """Namespace the digest so a fingerprint-format change cannot collide."""
    if not source_digest:
        raise ValueError("autotune timed-contract digest must be non-empty")
    return f"{AUTOTUNE_TIMED_CONTRACT_FORMAT}:{source_digest}"


# Used when packaged source resources are unavailable.
AUTOTUNE_TIMED_CONTRACT_FALLBACK_DIGEST = (
    "be41152deb55eb4d3c5fc26337d238f5cc919a141bf23e5a9491e2be511ea470"
)

_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")

_SOURCE_UNAVAILABLE = (
    OSError,
    UnicodeError,
    SyntaxError,
    LookupError,
    ImportError,
    TypeError,
    AttributeError,
)


# ── Digest resolution ────────────────────────────────────────────────────────
def _require_sha256(digest: str) -> str:
    if not _SHA256_HEX.match(digest):
        raise ValueError(
            f"source contract digest is not a SHA-256 hex string: {digest!r}"
        )
    return digest


def resolve_source_contract_digest(
    source_loader: Callable[[], str] = current_source_contract_digest,
) -> str:
    """Use the packaged fingerprint, with a deterministic wheel fallback."""
    try:
        digest = source_loader()
    except _SOURCE_UNAVAILABLE:
        return _require_sha256(AUTOTUNE_TIMED_CONTRACT_FALLBACK_DIGEST)
    return _require_sha256(digest)


AUTOTUNE_TIMED_CONTRACT_DIGEST = resolve_source_contract_digest()
AUTOTUNE_TIMED_CONTRACT_ID = contract_id_for_source_digest(
    AUTOTUNE_TIMED_CONTRACT_DIGEST
)
