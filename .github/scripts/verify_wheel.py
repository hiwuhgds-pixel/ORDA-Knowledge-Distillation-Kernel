from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import re
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

# ── Wheel contract constants ─────────────────────────────────────────────────
EXPECTED_DISTRIBUTION = "orda-kernel"
REQUIRED_PACKAGE_FILES = {
    "orda_kernel/__init__.py",
    "orda_kernel/api.py",
    "orda_kernel/_api_types.py",
    "orda_kernel/_torch_backend.py",
    "orda_kernel/ops/__init__.py",
    "orda_kernel/ops/_kernels/ce_kl_fwdbwd.py",
    "orda_kernel/py.typed",
}


# ── Archive validation helpers ───────────────────────────────────────────────
def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _require_expected_version(version: str, expected_version: str | None) -> None:
    if expected_version is None:
        return
    expected_version = expected_version.strip()
    if not expected_version:
        raise ValueError("expected release version must be non-empty")
    if version != expected_version:
        raise ValueError(
            "wheel version does not match release tag: "
            f"metadata={version!r}, expected={expected_version!r}"
        )


def _validate_member_names(names: list[str]) -> None:
    if len(names) != len(set(names)):
        raise ValueError("wheel contains duplicate archive members")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"wheel contains unsafe archive member {name!r}")


def _validate_record(archive: ZipFile, record_name: str, file_names: set[str]) -> None:
    rows = csv.reader(archive.read(record_name).decode("utf-8").splitlines())
    recorded: set[str] = set()
    for path, encoded_hash, encoded_size in rows:
        if path in recorded:
            raise ValueError(f"RECORD contains duplicate entry {path!r}")
        recorded.add(path)
        if path not in file_names:
            raise ValueError(f"RECORD references missing member {path!r}")
        payload = archive.read(path)
        if encoded_size and int(encoded_size) != len(payload):
            raise ValueError(f"RECORD size mismatch for {path!r}")
        if encoded_hash:
            algorithm, expected = encoded_hash.split("=", 1)
            digest = hashlib.new(algorithm, payload).digest()
            actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
            if actual != expected:
                raise ValueError(f"RECORD hash mismatch for {path!r}")
    missing = file_names - recorded
    if missing:
        raise ValueError(f"RECORD omits wheel members: {sorted(missing)!r}")


# ── Wheel verification ───────────────────────────────────────────────────────
def verify_wheel(
    wheel: Path, *, expected_version: str | None = None
) -> tuple[str, str]:
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError(f"wheel path must name one .whl file, got {wheel}")
    with ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"{wheel.name} contains a corrupt archive member")
        names = archive.namelist()
        _validate_member_names(names)
        file_names = {info.filename for info in archive.infolist() if not info.is_dir()}
        missing_package_files = REQUIRED_PACKAGE_FILES - file_names
        if missing_package_files:
            raise ValueError(
                f"{wheel.name} is missing package files: {sorted(missing_package_files)!r}"
            )

        dist_info_roots = {
            name.split("/", 1)[0] for name in file_names if ".dist-info/" in name
        }
        if len(dist_info_roots) != 1:
            raise ValueError(
                f"wheel must contain exactly one .dist-info directory, got {sorted(dist_info_roots)!r}"
            )
        dist_info = next(iter(dist_info_roots))
        metadata_name = f"{dist_info}/METADATA"
        wheel_name = f"{dist_info}/WHEEL"
        record_name = f"{dist_info}/RECORD"
        for required in (metadata_name, wheel_name, record_name):
            if required not in file_names:
                raise ValueError(
                    f"wheel is missing required metadata file {required!r}"
                )

        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        distribution = metadata.get("Name", "")
        version = metadata.get("Version", "")
        if _normalized_distribution(distribution) != EXPECTED_DISTRIBUTION:
            raise ValueError(f"unexpected wheel distribution name {distribution!r}")
        if not version.strip():
            raise ValueError("wheel METADATA does not contain a version")
        _require_expected_version(version, expected_version)
        if not metadata.get("Requires-Python", "").strip():
            raise ValueError("wheel METADATA does not contain Requires-Python")

        wheel_metadata = archive.read(wheel_name).decode("utf-8")
        if (
            "Wheel-Version:" not in wheel_metadata
            or "Root-Is-Purelib: true" not in wheel_metadata
        ):
            raise ValueError("WHEEL metadata is incomplete or not a pure-Python wheel")
        _validate_record(archive, record_name, file_names)
    return distribution, version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--expected-version")
    args = parser.parse_args()
    try:
        distribution, version = verify_wheel(
            args.wheel, expected_version=args.expected_version
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"verified wheel {args.wheel.name}: distribution={distribution} "
        f"version={version}"
    )


if __name__ == "__main__":
    main()
