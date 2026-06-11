from __future__ import annotations

import csv
import json

from tests.benchmarks.common import benchmark_row, compile_model, write_artifacts


def test_benchmark_artifact_writers_create_json_and_csv(tmp_path):
    rows = [
        benchmark_row(
            config="1x32",
            method="orda",
            latency_ms=1.25,
            peak_vram_mib=128.0,
            status="ok",
        )
    ]
    metadata = {
        "benchmark": "unit",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "python": "3.12",
        "platform": "test",
        "torch": "test",
        "cuda": "12.x",
        "hip": None,
        "device": {"name": "Fake GPU", "capability": [9, 0]},
        "args": {"steps": 1},
    }
    json_path = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"

    write_artifacts(rows, metadata, output_json=str(json_path), output_csv=str(csv_path))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["benchmark"] == "unit"
    assert payload["rows"][0]["method"] == "orda"

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows[0]["benchmark"] == "unit"
    assert csv_rows[0]["method"] == "orda"
    assert csv_rows[0]["device"] == "Fake GPU"


def test_compile_model_falls_back_to_eager_after_runtime_failure():
    class FakeTorch:
        @staticmethod
        def compile(module):
            class Compiled:
                def __call__(self, value):
                    raise RuntimeError("compile backend failed")

            return Compiled()

    class Eager:
        marker = "eager"

        def __init__(self):
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return value + 1

    eager = Eager()
    wrapped = compile_model(FakeTorch, eager, "unit")

    assert wrapped(1) == 2
    assert wrapped(2) == 3
    assert eager.calls == 2
    assert wrapped.marker == "eager"


