"""
benchmarking.py
===============

Reproducible timing of NumCompute's vectorised operations against
equivalent pure-Python loop implementations.

Every ``benchmark_*`` function first checks that the loop and vectorised
results agree (raising AssertionError otherwise), then times both with
``measure_time`` and returns a result dictionary with the keys
``operation``, ``size``, ``loop_time``, ``vector_time`` (seconds) and
``speedup`` (``loop_time / vector_time``).

Run ``python -m numcompute.benchmarking`` to print the results table.
"""

import time
import numpy as np

from numcompute.utils import validate_int
from numcompute.stats import mean
from numcompute.metrics import mse
from numcompute.preprocessing import StandardScaler


# Python loop versions used only for benchmarking

def loop_mean(data):
    """Arithmetic mean of a 1D sequence using a Python loop. O(n)."""
    total = 0.0

    for value in data:
        total += value

    return total / len(data)


def loop_mse(y_true, y_pred):
    """Mean squared error using a Python loop. O(n)."""
    total_error = 0.0

    for actual, predicted in zip(y_true, y_pred):
        difference = actual - predicted
        total_error += difference ** 2

    return total_error / len(y_true)


def loop_standard_scaler(data):
    """
    Standardise each column using Python loops.
    Uses population standard deviation (ddof=0).
    """
    rows, cols = data.shape
    result = np.zeros_like(data, dtype=float)

    for col in range(cols):
        total = 0.0

        for row in range(rows):
            total += data[row, col]

        column_mean = total / rows

        variance_sum = 0.0

        for row in range(rows):
            variance_sum += (data[row, col] - column_mean) ** 2

        column_std = (variance_sum / rows) ** 0.5

        if column_std == 0:
            column_std = 1.0

        for row in range(rows):
            result[row, col] = (
                data[row, col] - column_mean
            ) / column_std

    return result


# Reusable timing function

def measure_time(function, repeats=5):
    """
    Return the median wall-clock time of ``function()`` in seconds.

    The function is called once as an untimed warm-up, then ``repeats``
    timed calls are made with ``time.perf_counter``.

    Parameters
    ----------
    function : callable
        Zero-argument callable to time.
    repeats : int, default=5
        Number of timed runs, at least 1.

    Returns
    -------
    float
        Median execution time in seconds (non-negative).

    Raises
    ------
    TypeError
        If function is not callable or repeats is not an integer.
    ValueError
        If repeats is less than 1.
    """
    if not callable(function):
        raise TypeError("function must be callable")

    repeats = validate_int(repeats, "repeats", minimum=1)

    times = []

    # Warm-up run
    function()

    for _ in range(repeats):
        start = time.perf_counter()
        function()
        end = time.perf_counter()

        times.append(end - start)

    return float(np.median(times))


# Benchmark 1: Mean

def benchmark_mean(size=100000, repeats=5):
    """
    Benchmark ``stats.mean`` against ``loop_mean`` on ``size`` values.

    Returns the result dictionary described in the module docstring.

    Raises
    ------
    TypeError
        If size or repeats is not an integer.
    ValueError
        If size or repeats is less than 1.
    AssertionError
        If the two implementations disagree.
    """
    size = validate_int(size, "size", minimum=1)
    rng = np.random.default_rng(42)
    data = rng.random(size)

    expected = loop_mean(data)
    actual = mean(data)

    if not np.isclose(expected, actual):
        raise AssertionError("Mean implementations disagree")

    loop_time = measure_time(
        lambda: loop_mean(data), repeats
    )

    vector_time = measure_time(
        lambda: mean(data), repeats
    )

    return {
        "operation": "Mean",
        "size": size,
        "loop_time": loop_time,
        "vector_time": vector_time,
        "speedup": loop_time / vector_time
    }


# Benchmark 2: Mean Squared Error

def benchmark_mse(size=100000, repeats=5):
    """
    Benchmark ``metrics.mse`` against ``loop_mse`` on ``size`` pairs.

    Returns the result dictionary described in the module docstring.

    Raises
    ------
    TypeError
        If size or repeats is not an integer.
    ValueError
        If size or repeats is less than 1.
    AssertionError
        If the two implementations disagree.
    """
    size = validate_int(size, "size", minimum=1)

    rng = np.random.default_rng(42)

    y_true = rng.random(size)
    y_pred = rng.random(size)

    expected = loop_mse(y_true, y_pred)
    actual = mse(y_true, y_pred)

    if not np.isclose(expected, actual):
        raise AssertionError("MSE implementations disagree")

    loop_time = measure_time(
        lambda: loop_mse(y_true, y_pred), repeats
    )

    vector_time = measure_time(
        lambda: mse(y_true, y_pred), repeats
    )

    return {
        "operation": "MSE",
        "size": size,
        "loop_time": loop_time,
        "vector_time": vector_time,
        "speedup": loop_time / vector_time
    }


# Benchmark 3: StandardScaler

def benchmark_scaler(rows=10000, cols=5, repeats=5):
    """
    Benchmark ``StandardScaler.fit_transform`` against
    ``loop_standard_scaler`` on a ``(rows, cols)`` matrix.

    ``size`` in the result is ``rows * cols``.

    Raises
    ------
    TypeError
        If rows, cols or repeats is not an integer.
    ValueError
        If rows, cols or repeats is less than 1.
    AssertionError
        If the two implementations disagree.
    """
    rows = validate_int(rows, "rows", minimum=1)
    cols = validate_int(cols, "cols", minimum=1)

    rng = np.random.default_rng(42)
    data = rng.random((rows, cols))

    def vector_scaler():
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    expected = loop_standard_scaler(data)
    actual = vector_scaler()

    if not np.allclose(expected, actual, atol=1e-8):
        raise AssertionError(
            "StandardScaler implementations disagree"
        )

    loop_time = measure_time(
        lambda: loop_standard_scaler(data), repeats
    )

    vector_time = measure_time(
        vector_scaler, repeats
    )

    return {
        "operation": "StandardScaler",
        "size": rows * cols,
        "loop_time": loop_time,
        "vector_time": vector_time,
        "speedup": loop_time / vector_time
    }


# Display all benchmark results

def run_benchmarks(size=100000, rows=10000, cols=5, repeats=5):
    """
    Run all benchmarks, print a results table and return the results.

    Returns
    -------
    list of dict
        One result dictionary per benchmark (Mean, MSE, StandardScaler).
    """
    print("\nNUMCOMPUTE - PERFORMANCE BENCHMARKS")
    print("=" * 82)

    print(
        f"{'Operation':<18}"
        f"{'Elements':>12}"
        f"{'Loop (ms)':>16}"
        f"{'NumPy (ms)':>16}"
        f"{'Speedup':>14}"
    )

    print("-" * 82)

    results = [
        benchmark_mean(size, repeats),
        benchmark_mse(size, repeats),
        benchmark_scaler(rows, cols, repeats)
    ]

    for result in results:
        loop_ms = result["loop_time"] * 1000
        vector_ms = result["vector_time"] * 1000

        print(
            f"{result['operation']:<18}"
            f"{result['size']:>12,}"
            f"{loop_ms:>16.4f}"
            f"{vector_ms:>16.4f}"
            f"{result['speedup']:>13.2f}x"
        )

    print("=" * 82)
    print("Random seed: 42")
    print(f"Timing: median of {repeats} runs, after one warm-up")
    print("All correctness checks passed.")

    return results


if __name__ == "__main__":
    run_benchmarks()
