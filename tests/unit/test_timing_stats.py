from __future__ import annotations

import math

import pytest

from tests.utils.timing import (
    BenchResult,
    cv_pct,
    percentile,
    stdev,
    summarize_samples,
    trimmed_mean,
)


def test_percentile_matches_known_quantiles():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 50.0) == 3.0
    assert percentile(values, 100.0) == 5.0
    assert percentile(values, 25.0) == pytest.approx(2.0)
    assert percentile(values, 95.0) == pytest.approx(4.8)


def test_percentile_handles_single_and_empty():
    assert percentile([7.0], 50.0) == 7.0
    assert math.isnan(percentile([], 50.0))


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError, match="percentile"):
        percentile([1.0, 2.0], 150.0)


def test_stdev_sample_formula():
    assert stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.138089935)
    assert math.isnan(stdev([1.0]))


def test_cv_pct_returns_nan_for_zero_mean():
    assert math.isnan(cv_pct(0.0, 1.0))
    assert cv_pct(100.0, 5.0) == pytest.approx(5.0)


def test_trimmed_mean_drops_tails_symmetrically():
    # trimmed_mean drops `int(len * ratio)` samples from each end. With 10
    # samples and ratio=0.20 → 2 dropped per side, leaving the six middle 10.0s.
    samples = [1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0]
    assert trimmed_mean(samples, 0.20) == pytest.approx(10.0)


def test_trimmed_mean_zero_ratio_is_plain_mean():
    samples = [1.0, 2.0, 3.0, 4.0]
    assert trimmed_mean(samples, 0.0) == pytest.approx(2.5)


def test_trimmed_mean_handles_empty():
    assert math.isnan(trimmed_mean([], 0.1))


def test_summarize_samples_populates_every_field():
    samples = [10.0, 11.0, 12.0, 13.0, 14.0]
    summary = summarize_samples(samples, trim_ratio=0.0)

    assert summary["latency_ms"] == pytest.approx(12.0)
    assert summary["latency_ms_min"] == 10.0
    assert summary["latency_ms_max"] == 14.0
    assert summary["latency_ms_p50"] == pytest.approx(12.0)
    assert summary["latency_ms_p95"] == pytest.approx(13.8)
    assert summary["latency_ms_std"] == pytest.approx(stdev(samples))
    expected_cv = (summary["latency_ms_std"] / summary["latency_ms"]) * 100.0
    assert summary["cv_pct"] == pytest.approx(expected_cv)


def test_summarize_samples_handles_empty_input():
    summary = summarize_samples([])
    for value in summary.values():
        assert math.isnan(value)


def test_bench_result_ok_property_flags_oom():
    ok_result = BenchResult(latency_ms=1.5, peak_vram_mib=128.0)
    bad_result = BenchResult(latency_ms=float("nan"), peak_vram_mib=float("nan"))
    assert ok_result.ok is True
    assert bad_result.ok is False


