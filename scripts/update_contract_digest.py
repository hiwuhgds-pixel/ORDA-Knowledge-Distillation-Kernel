from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
CONTRACT_FILE = SRC_ROOT / "orda_kernel" / "_autotune" / "_contract.py"


def main() -> int:
    sys.path.insert(0, str(SRC_ROOT))
    from orda_kernel._autotune._contract import (
        AUTOTUNE_TIMED_CONTRACT_FALLBACK_DIGEST,
        current_source_contract_digest,
    )

    stored = AUTOTUNE_TIMED_CONTRACT_FALLBACK_DIGEST
    current = current_source_contract_digest()
    if stored == current:
        print(f"[ok] digest unchanged: {current}")
        return 0

    payload = CONTRACT_FILE.read_bytes()
    stored_bytes = stored.encode("ascii")
    if payload.count(stored_bytes) != 1:
        raise RuntimeError(
            f"stored autotune digest must occur exactly once in {CONTRACT_FILE}"
        )
    CONTRACT_FILE.write_bytes(payload.replace(stored_bytes, current.encode("ascii"), 1))
    print(f"[updated] {stored[:16]}... -> {current}")
    print(f"          file: {CONTRACT_FILE.relative_to(REPO_ROOT)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
