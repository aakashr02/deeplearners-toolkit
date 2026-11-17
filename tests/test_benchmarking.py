"""Tests for numcompute.benchmarking: correctness, timing and contracts."""

import numpy as np
import pytest

from numcompute import benchmarking as bm
from numcompute.preprocessing import StandardScaler


RESULT_KEYS = {"operation", "size", "loop_time", "vector_time", "speedup"}


# ---------------------------------------------------------
# Loop reference implementations agree with NumPy
# ---------------------------------------------------------

def test_loop_mean_matches_numpy():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(bm.loop_mean(data), np.mean(data))


def test_loop_mse_matches_numpy():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 5.0])
    assert np.isclose(bm.loop_mse(a, b), np.mean((a - b) ** 2))


def test_loop_standard_scaler_matches_vectorised():
    X = np.random.default_rng(0).random((20, 3))
    assert np.allclose(
        bm.loop_standard_scaler(X), StandardScaler().fit_transform(X)
    )


def test_loop_standard_scaler_constant_column_matches_vectorised():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    assert np.allclose(
        bm.loop_standard_scaler(X), StandardScaler().fit_transform(X)
    )


# ---------------------------------------------------------
# measure_time
# ---------------------------------------------------------

def test_measure_time_calls_warmup_plus_repeats():
    calls = []
    bm.measure_time(lambda: calls.append(1), repeats=3)
    assert len(calls) == 4  # one warm-up + three timed runs


def test_measure_time_returns_non_negative_float():
    elapsed = bm.measure_time(lambda: sum(range(100)), repeats=2)
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


@pytest.mark.parametrize("repeats", [0, -1])
def test_measure_time_rejects_non_positive_repeats(repeats):
    with pytest.raises(ValueError):
        bm.measure_time(lambda: None, repeats=repeats)


def test_measure_time_rejects_non_integer_repeats():
    with pytest.raises(TypeError):
        bm.measure_time(lambda: None, repeats=2.5)


def test_measure_time_rejects_non_callable():
    with pytest.raises(TypeError):
        bm.measure_time(42)


# ---------------------------------------------------------
# Benchmark workloads (small sizes keep the suite fast)
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "run, operation, expected_size",
    [
        (lambda: bm.benchmark_mean(size=500, repeats=1), "Mean", 500),
        (lambda: bm.benchmark_mse(size=500, repeats=1), "MSE", 500),
        (
            lambda: bm.benchmark_scaler(rows=50, cols=4, repeats=1),
            "StandardScaler",
            200,
        ),
    ],
)
def test_benchmark_result_contract(run, operation, expected_size):
    result = run()

    assert set(result) == RESULT_KEYS
    assert result["operation"] == operation
    assert result["size"] == expected_size
    assert result["loop_time"] > 0
    assert result["vector_time"] > 0
    assert np.isclose(
        result["speedup"], result["loop_time"] / result["vector_time"]
    )


def test_vectorised_mean_is_faster_than_loop():
    # Large enough that vectorisation must win clearly on any machine.
    result = bm.benchmark_mean(size=50_000, repeats=3)
    assert result["speedup"] > 1.0


@pytest.mark.parametrize("size", [0, -5])
def test_benchmark_mean_rejects_invalid_size(size):
    with pytest.raises(ValueError):
        bm.benchmark_mean(size=size)


@pytest.mark.parametrize("size", [0, -5])
def test_benchmark_mse_rejects_invalid_size(size):
    with pytest.raises(ValueError):
        bm.benchmark_mse(size=size)


@pytest.mark.parametrize("rows, cols", [(0, 3), (3, 0)])
def test_benchmark_scaler_rejects_invalid_shape(rows, cols):
    with pytest.raises(ValueError):
        bm.benchmark_scaler(rows=rows, cols=cols)


def test_run_benchmarks_prints_table_and_returns_results(capsys):
    results = bm.run_benchmarks(size=200, rows=20, cols=2, repeats=1)
    output = capsys.readouterr().out

    assert [r["operation"] for r in results] == [
        "Mean", "MSE", "StandardScaler"
    ]
    assert "All correctness checks passed." in output
    assert "median of 1 runs" in output
