from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from orda_kernel._autotune import _device


@pytest.fixture(autouse=True)
def _clear_triton_version_cache():
    _device._triton_version.cache_clear()
    yield
    _device._triton_version.cache_clear()


def _version_probe(installed: dict[str, str]):
    def probe(distribution: str) -> str:
        try:
            return installed[distribution]
        except KeyError:
            raise PackageNotFoundError(distribution) from None

    return probe


# ── Triton version fingerprint ───────────────────────────────────────────────
def test_triton_version_detects_standard_cuda_distribution(monkeypatch):
    monkeypatch.setattr(_device, "version", _version_probe({"triton": "3.2.0"}))
    assert _device._triton_version() == "3.2.0"


@pytest.mark.parametrize(
    "distribution",
    ["pytorch-triton-rocm", "triton-rocm", "pytorch-triton"],
)
def test_triton_version_detects_rocm_and_nightly_distributions(
    monkeypatch, distribution
):
    # A ROCm/nightly wheel ships Triton under a non-"triton" distribution name; the
    # fingerprint must still resolve so a compiler upgrade invalidates stale winners.
    monkeypatch.setattr(
        _device, "version", _version_probe({distribution: "3.1.0+rocm"})
    )
    assert _device._triton_version() == "3.1.0+rocm"


def test_triton_version_prefers_canonical_name_over_variants(monkeypatch):
    monkeypatch.setattr(
        _device,
        "version",
        _version_probe({"triton": "3.2.0", "pytorch-triton-rocm": "3.1.0+rocm"}),
    )
    assert _device._triton_version() == "3.2.0"


def test_triton_version_is_none_when_no_distribution_present(monkeypatch):
    monkeypatch.setattr(_device, "version", _version_probe({}))
    assert _device._triton_version() is None


# ── Device profiles ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["Tesla T4", "NVIDIA T4", "T4"])
def test_t4_profile_requires_an_explicit_model_token(name):
    assert _device._is_t4_properties(SimpleNamespace(name=name))


def test_t4_profile_does_not_infer_from_compute_capability_and_memory():
    properties = SimpleNamespace(
        name="Quadro RTX 5000",
        major=7,
        minor=5,
        total_memory=16 * 1024**3,
    )
    assert not _device._is_t4_properties(properties)


# ── Launch geometry ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("backend", "warp_size", "expected"),
    [("cuda", 32, 32), ("hip", 64, 16), ("hip", 32, 32)],
)
def test_max_num_warps_respects_device_workgroup_limit(backend, warp_size, expected):
    properties = SimpleNamespace(
        warp_size=warp_size,
        max_threads_per_block=1024,
    )
    assert _device._max_num_warps_from_properties(backend, properties) == expected
