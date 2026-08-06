from __future__ import annotations

import sys

import logging_utils
import pytest
import run_test_cpu
import run_test_gpu


def test_gpu_log_path_matches_benchmark_layout_and_avoids_collisions(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(logging_utils, "active_gpu_log_name", lambda: "Tesla_T4")

    first = logging_utils.test_log_path(tmp_path, "run_test_gpu")
    second = logging_utils.test_log_path(tmp_path, "run_test_gpu")

    assert first == tmp_path / "logs/Tesla_T4/test/run_test_gpu.log"
    assert second == tmp_path / "logs/Tesla_T4/test/run_test_gpu_1.log"


def test_cpu_log_path_stays_in_cpu_directory(tmp_path):
    first = logging_utils.test_log_path(
        tmp_path,
        "run_test_cpu",
        device_name="cpu",
        group=None,
    )
    second = logging_utils.test_log_path(
        tmp_path,
        "run_test_cpu",
        device_name="cpu",
        group=None,
    )

    assert first == tmp_path / "logs/cpu/run_test_cpu.log"
    assert second == tmp_path / "logs/cpu/run_test_cpu_1.log"


def test_test_log_records_stdout_stderr_and_exception_traceback(tmp_path):
    with pytest.raises(RuntimeError, match="three-hour crash"):
        with logging_utils.test_log(
            tmp_path,
            "run_test_gpu",
            device_name="Tesla_T4",
        ) as path:
            print("stdout marker")
            print("stderr marker", file=sys.stderr)
            raise RuntimeError("three-hour crash")

    content = path.read_text(encoding="utf-8")
    assert "stdout marker" in content
    assert "stderr marker" in content
    assert "Traceback (most recent call last)" in content
    assert "RuntimeError: three-hour crash" in content


def test_run_logged_streams_combined_output_and_preserves_exit_code(
    tmp_path,
    capsys,
):
    status = logging_utils.run_logged(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(7)",
        ],
        cwd=tmp_path,
    )

    assert status == 7
    output = capsys.readouterr().out
    assert "out" in output
    assert "err" in output


@pytest.mark.parametrize(
    ("runner", "relative_path"),
    (
        (run_test_cpu, "logs/cpu/run_test_cpu.log"),
        (run_test_gpu, "logs/Tesla_T4/test/run_test_gpu.log"),
    ),
)
def test_runner_main_uses_shared_log_wrapper(
    tmp_path,
    monkeypatch,
    runner,
    relative_path,
):
    monkeypatch.setattr(logging_utils, "active_gpu_log_name", lambda: "Tesla_T4")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_main", lambda argv: 0)

    assert runner.main([]) == 0
    assert (tmp_path / relative_path).is_file()
