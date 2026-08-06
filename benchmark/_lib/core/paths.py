from __future__ import annotations

from pathlib import Path

LIB_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = LIB_ROOT.parent
REPO_ROOT = BENCHMARK_ROOT.parent
TEST_ROOT = REPO_ROOT / "test"
LOGGING_UTILS_PATH = TEST_ROOT / "logging_utils.py"
LOGS_ROOT = REPO_ROOT / "logs"
