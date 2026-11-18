"""
benchmarking.py
===============

Reproducible timing of NumCompute's vectorised operations against
equivalent pure-Python loop implementations, plus streaming experiments.

Every ``benchmark_*`` loop-vs-vectorised function first checks that the
loop and vectorised results agree (raising AssertionError otherwise),
then times both with ``measure_time`` and returns a result dictionary with
the keys ``operation``, ``size``, ``loop_time``, ``vector_time`` (seconds)
and ``speedup`` (``loop_time / vector_time``).

* ``run_benchmarks()``: batch operations (mean, MSE, StandardScaler).
* ``run_streaming_benchmarks()``: (a) loop-based vs vectorised streaming
  components (tree retraining via ``partial_fit``, tree prediction,
  running statistics) and (b) how accuracy, retraining time and memory
  change with the tree's ``buffer_size``.

Run ``python -m numcompute.benchmarking`` to print every table together
with environment notes.
"""

import math
import platform
import time

import numpy as np

from numcompute.utils import validate_int
from numcompute.stats import RunningStats, mean
from numcompute.metrics import mse
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import SimpleImputer, StandardScaler
from numcompute.stream import StreamTrainer
from numcompute.tree import DecisionTreeClassifier


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


# ============================================================
# Streaming data
# ============================================================

def make_stream_data(n_samples=6000, n_features=4, drift_at=0.5,
                     missing_rate=0.0, label_noise=0.05, seed=42):
    """
    Synthetic binary classification stream with one concept drift.

    Features are standard normal. Before the drift point the label is
    ``x0 + x1**2 > 0.5``; afterwards it is ``x0 - x2 > 0``, so a model
    trained on old data becomes stale. A fraction ``label_noise`` of labels
    is flipped, and a fraction ``missing_rate`` of feature values is set
    to NaN (labels are never missing).

    Parameters
    ----------
    n_samples : int, default=6000
        Number of rows (at least 1).
    n_features : int, default=4
        Number of features (at least 3, because the rules use x0..x2).
    drift_at : float or None, default=0.5
        Fraction of the stream after which the concept changes; None
        disables drift.
    missing_rate : float, default=0.0
        Fraction of feature values replaced by NaN, in [0, 1).
    label_noise : float, default=0.05
        Fraction of labels flipped, in [0, 0.5).
    seed : int, default=42
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,), int labels 0/1

    Raises
    ------
    ValueError
        If an argument is out of range.
    TypeError
        If n_samples or n_features is not an integer.

    Notes
    -----
    Time complexity: O(n_samples * n_features).
    """
    n_samples = validate_int(n_samples, "n_samples", minimum=1)
    n_features = validate_int(n_features, "n_features", minimum=3)
    if drift_at is not None and not 0.0 <= drift_at <= 1.0:
        raise ValueError("drift_at must be in [0, 1] or None")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    if not 0.0 <= label_noise < 0.5:
        raise ValueError("label_noise must be in [0, 0.5)")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    before = (X[:, 0] + X[:, 1] ** 2 > 0.5)
    after = (X[:, 0] - X[:, 2] > 0)
    switch = n_samples if drift_at is None else int(drift_at * n_samples)
    y = np.where(np.arange(n_samples) < switch, before, after).astype(int)

    flip = rng.random(n_samples) < label_noise
    y[flip] = 1 - y[flip]

    if missing_rate > 0:
        X[rng.random(X.shape) < missing_rate] = np.nan

    return X, y


def _chunked(X, y, chunk_size):
    starts = range(0, len(y), chunk_size)
    return [(X[s:s + chunk_size], y[s:s + chunk_size]) for s in starts]


# ============================================================
# Loop-based reference implementations (streaming components)
# ============================================================

class LoopDecisionTreeClassifier(DecisionTreeClassifier):
    """
    ``DecisionTreeClassifier`` whose split search uses Python loops.

    Only ``_best_split`` differs: instead of sorting all features at once
    and evaluating every threshold with array arithmetic, it sorts each
    feature with ``sorted``, walks the samples one at a time updating
    class counts in a list, and computes each candidate's impurity in a
    Python loop. Candidates are visited in the same order and with the
    same arithmetic as the vectorised version, so both build identical
    trees; only the speed differs. Used for benchmarking only.
    """

    def _best_split(self, X, onehot, parent_counts):
        n, n_features = X.shape
        n_classes = onehot.shape[1]
        labels = onehot.argmax(axis=1).tolist()
        columns = X.T.tolist()
        totals = parent_counts.tolist()
        criterion = self.criterion

        def impurity(counts, size):
            if size == 0:
                return 0.0
            if criterion == "gini":
                squares = 0.0
                for count in counts:
                    p = count / size
                    squares += p * p
                return 1.0 - squares
            entropy = 0.0
            for count in counts:
                if count > 0:
                    p = count / size
                    entropy += p * math.log2(p)
            return -entropy

        # Score every (position, feature) candidate with explicit loops.
        sorted_values, scores = [], []
        for column in columns:
            order = sorted(range(n), key=column.__getitem__)  # stable
            values = [column[i] for i in order]
            left = [0.0] * n_classes
            column_scores = []
            for position in range(n - 1):
                left[labels[order[position]]] += 1.0
                if values[position + 1] > values[position]:
                    n_left = position + 1
                    n_right = n - n_left
                    right = [totals[c] - left[c] for c in range(n_classes)]
                    column_scores.append(
                        (n_left * impurity(left, n_left)
                         + n_right * impurity(right, n_right)) / n
                    )
                else:
                    column_scores.append(math.inf)
            sorted_values.append(values)
            scores.append(column_scores)

        # Same visiting order as argmin over the (position, feature) grid.
        best, best_position, best_column = math.inf, None, None
        for position in range(n - 1):
            for column in range(n_features):
                if scores[column][position] < best:
                    best = scores[column][position]
                    best_position, best_column = position, column

        if best_column is None:
            return None
        if not best < impurity(totals, n) - 1e-12:
            return None

        values = sorted_values[best_column]
        low, high = values[best_position], values[best_position + 1]
        cut = low / 2.0 + high / 2.0
        if not low <= cut < high:
            cut = low

        return best_column, float(cut), X[:, best_column] <= cut


def loop_tree_predict(tree, X):
    """
    Predict with a trained tree, one sample and one node at a time.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        A trained tree.
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Same predictions as ``tree.predict(X)``.

    Notes
    -----
    Time complexity: O(n * depth) Python steps.
    """
    feature = tree.feature_.tolist()
    threshold = tree.threshold_.tolist()
    left = tree.children_left_.tolist()
    right = tree.children_right_.tolist()
    leaf_class = np.argmax(tree.value_, axis=1).tolist()

    out = []
    for row in np.asarray(X, dtype=float).tolist():
        node = 0
        while feature[node] != -1:
            if row[feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        out.append(leaf_class[node])
    return tree.classes_[np.array(out, dtype=np.intp)]


def loop_running_stats(chunks):
    """
    Running mean and population variance per feature, updated one value
    at a time with Welford's algorithm in Python loops.

    Parameters
    ----------
    chunks : iterable of array-like of shape (n_samples, n_features)

    Returns
    -------
    mean, variance : list of float
        One value per feature.

    Notes
    -----
    Time complexity: O(total values) Python steps.
    """
    count, means, m2 = None, None, None
    for chunk in chunks:
        for row in np.asarray(chunk, dtype=float).tolist():
            if count is None:
                count = [0] * len(row)
                means = [0.0] * len(row)
                m2 = [0.0] * len(row)
            for j, value in enumerate(row):
                count[j] += 1
                delta = value - means[j]
                means[j] += delta / count[j]
                m2[j] += delta * (value - means[j])
    return means, [m2[j] / count[j] for j in range(len(m2))]


# ============================================================
# (a) Loop-based vs vectorised under streaming
# ============================================================

def _timing_result(operation, size, loop_time, vector_time):
    return {
        "operation": operation,
        "size": size,
        "loop_time": loop_time,
        "vector_time": vector_time,
        "speedup": loop_time / vector_time,
    }


def benchmark_tree_streaming(n_samples=4000, chunk_size=200,
                             buffer_size=1000, max_depth=6, repeats=3):
    """
    Stream the same chunks into a vectorised and a loop-based tree.

    Both trees receive every chunk through ``partial_fit`` and retrain
    each time ``buffer_size`` samples are buffered. The trees must end up
    identical (same splits and predictions) before timing.

    Returns
    -------
    dict
        Timing result (see module docstring); ``size`` is n_samples.

    Raises
    ------
    ValueError
        If a size argument is less than 1.
    AssertionError
        If the two trees differ.

    Notes
    -----
    Time complexity: n_samples / buffer_size retrains of a tree.
    """
    n_samples = validate_int(n_samples, "n_samples", minimum=1)
    chunk_size = validate_int(chunk_size, "chunk_size", minimum=1)
    X, y = make_stream_data(n_samples, seed=42)
    chunks = _chunked(X, y, chunk_size)

    def stream(tree_class):
        tree = tree_class(max_depth=max_depth, buffer_size=buffer_size)
        for X_chunk, y_chunk in chunks:
            tree.partial_fit(X_chunk, y_chunk)
        return tree

    vector_tree = stream(DecisionTreeClassifier)
    loop_tree = stream(LoopDecisionTreeClassifier)
    if not (np.array_equal(vector_tree.feature_, loop_tree.feature_)
            and np.array_equal(vector_tree.threshold_, loop_tree.threshold_,
                               equal_nan=True)
            and np.array_equal(vector_tree.predict(X), loop_tree.predict(X))):
        raise AssertionError("Loop and vectorised trees differ")

    loop_time = measure_time(lambda: stream(LoopDecisionTreeClassifier),
                             repeats)
    vector_time = measure_time(lambda: stream(DecisionTreeClassifier),
                               repeats)
    return _timing_result("Tree partial_fit", n_samples, loop_time,
                          vector_time)


def benchmark_tree_predict(n_samples=20000, max_depth=8, repeats=5):
    """
    Time tree prediction: one sample at a time vs all samples per level.

    Returns
    -------
    dict
        Timing result; ``size`` is n_samples.

    Raises
    ------
    ValueError
        If n_samples is less than 1.
    AssertionError
        If the predictions differ.

    Notes
    -----
    Time complexity: O(n_samples * depth).
    """
    n_samples = validate_int(n_samples, "n_samples", minimum=1)
    X, y = make_stream_data(max(n_samples, 2), seed=7)
    X, y = X[:n_samples], y[:n_samples]
    tree = DecisionTreeClassifier(max_depth=max_depth).fit(
        *make_stream_data(3000, seed=8)
    )

    if not np.array_equal(loop_tree_predict(tree, X), tree.predict(X)):
        raise AssertionError("Loop and vectorised predictions differ")

    loop_time = measure_time(lambda: loop_tree_predict(tree, X), repeats)
    vector_time = measure_time(lambda: tree.predict(X), repeats)
    return _timing_result("Tree predict", n_samples, loop_time, vector_time)


def benchmark_running_stats(n_samples=20000, n_features=5, chunk_size=500,
                            repeats=5):
    """
    Time running mean/variance over a stream: per-value Welford loop vs
    ``RunningStats.update_stats`` per chunk.

    Returns
    -------
    dict
        Timing result; ``size`` is n_samples * n_features.

    Raises
    ------
    ValueError
        If a size argument is less than 1.
    AssertionError
        If the statistics differ.

    Notes
    -----
    Time complexity: O(n_samples * n_features).
    """
    n_samples = validate_int(n_samples, "n_samples", minimum=1)
    n_features = validate_int(n_features, "n_features", minimum=1)
    chunk_size = validate_int(chunk_size, "chunk_size", minimum=1)
    rng = np.random.default_rng(42)
    data = rng.normal(100.0, 5.0, size=(n_samples, n_features))
    chunks = [data[s:s + chunk_size] for s in range(0, n_samples, chunk_size)]

    def vectorised():
        stats = RunningStats()
        for chunk in chunks:
            stats.update_stats(chunk)
        return stats

    loop_mean_, loop_var = loop_running_stats(chunks)
    stats = vectorised()
    if not (np.allclose(loop_mean_, stats.mean)
            and np.allclose(loop_var, stats.variance())):
        raise AssertionError("Running statistics implementations disagree")

    loop_time = measure_time(lambda: loop_running_stats(chunks), repeats)
    vector_time = measure_time(vectorised, repeats)
    return _timing_result("RunningStats", n_samples * n_features, loop_time,
                          vector_time)


# ============================================================
# (b) Effect of buffer_size
# ============================================================

def benchmark_buffer_sizes(buffer_sizes=(100, 250, 500, 1000, 2000), X=None,
                           y=None, chunk_size=100, max_depth=6,
                           n_samples=10000):
    """
    Stream the same data through trees with different ``buffer_size``.

    Each run uses ``Pipeline([SimpleImputer, StandardScaler,
    DecisionTreeClassifier])`` (the imputer handles streams with NaN)
    inside a ``StreamTrainer`` (test-then-train), so accuracy is always
    measured on chunks the model has not trained on.

    Parameters
    ----------
    buffer_sizes : sequence of int
        Values to compare.
    X, y : array-like or None
        Stream to use. None generates ``make_stream_data(n_samples)``
        without drift and with 3% missing feature values.
    chunk_size : int, default=100
        Rows per chunk.
    max_depth : int, default=6
        Tree depth limit.
    n_samples : int, default=10000
        Size of the generated stream when X is None.

    Returns
    -------
    list of dict
        One per buffer size with ``buffer_size``, ``accuracy`` (cumulative
        test-then-train accuracy), ``n_retrains``,
        ``total_retrain_time``, ``mean_retrain_time`` (seconds),
        ``first_scored_chunk``, ``peak_memory_bytes`` and
        ``chunk_accuracy`` (array, for plotting).

    Raises
    ------
    ValueError
        If buffer_sizes is empty or contains values below 1, or X/y are
        inconsistent.

    Notes
    -----
    Time complexity: for each buffer size, n / buffer_size retrains.
    """
    buffer_sizes = [validate_int(b, "buffer_size", minimum=1)
                    for b in buffer_sizes]
    if not buffer_sizes:
        raise ValueError("buffer_sizes must not be empty")
    chunk_size = validate_int(chunk_size, "chunk_size", minimum=1)

    if X is None:
        X, y = make_stream_data(n_samples, drift_at=None, missing_rate=0.03,
                                seed=42)
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2D with one label per row")

    chunks = _chunked(X, y, chunk_size)
    results = []

    for buffer_size in buffer_sizes:
        pipe = Pipeline([
            ("impute", SimpleImputer("median")),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=max_depth,
                                             buffer_size=buffer_size)),
        ])
        trainer = StreamTrainer(pipe).run(chunks)

        retrained = trainer.get_metric("retrained").astype(bool)
        fit_times = trainer.get_metric("fit_time")
        chunk_acc = trainer.get_metric("chunk_accuracy")
        scored = np.flatnonzero(trainer.get_metric("scored").astype(bool))

        n_retrains = int(retrained.sum())
        total_retrain = float(fit_times[retrained].sum())
        results.append({
            "buffer_size": buffer_size,
            "accuracy": trainer.summary()["cumulative_accuracy"],
            "n_retrains": n_retrains,
            "total_retrain_time": total_retrain,
            "mean_retrain_time": (total_retrain / n_retrains
                                  if n_retrains else float("nan")),
            "first_scored_chunk": int(scored[0]) if scored.size else None,
            "peak_memory_bytes": int(trainer.get_metric("memory_bytes").max()),
            "chunk_accuracy": chunk_acc,
        })

    return results


# ============================================================
# Reporting
# ============================================================

def environment_info():
    """
    Machine and library versions, for reporting benchmark context.

    Returns
    -------
    dict
        ``platform``, ``processor``, ``python`` and ``numpy``.

    Raises
    ------
    None.

    Notes
    -----
    Time complexity: O(1).
    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def run_streaming_benchmarks(repeats=3, buffer_sizes=(100, 250, 500, 1000,
                                                      2000)):
    """
    Run and print both streaming comparisons required by the assignment.

    Returns
    -------
    dict
        ``{"loop_vs_vectorised": [...], "buffer_sizes": [...]}``.

    Raises
    ------
    AssertionError
        If any loop and vectorised implementation disagree.

    Notes
    -----
    Time complexity: the sum of the individual benchmarks; it takes a few
    seconds, dominated by the loop-based tree.
    """
    print("\nNUMCOMPUTE - STREAMING BENCHMARKS")
    print("=" * 82)
    print("(a) Loop-based vs vectorised (streaming components)")
    print(f"{'Operation':<18}{'Elements':>12}{'Loop (ms)':>16}"
          f"{'NumPy (ms)':>16}{'Speedup':>14}")
    print("-" * 82)

    timing = [
        benchmark_tree_streaming(repeats=repeats),
        benchmark_tree_predict(repeats=repeats),
        benchmark_running_stats(repeats=repeats),
    ]
    for result in timing:
        print(f"{result['operation']:<18}{result['size']:>12,}"
              f"{result['loop_time'] * 1000:>16.2f}"
              f"{result['vector_time'] * 1000:>16.2f}"
              f"{result['speedup']:>13.1f}x")

    print("\n(b) Effect of buffer_size (10,000 samples, 100 chunks of 100, "
          "3% missing values)")
    print(f"{'buffer':>8}{'accuracy':>10}{'first scored':>14}{'retrains':>10}"
          f"{'retrain total (ms)':>20}{'per retrain (ms)':>18}"
          f"{'peak memory (KB)':>18}")
    print("-" * 97)

    buffers = benchmark_buffer_sizes(buffer_sizes)
    for r in buffers:
        print(f"{r['buffer_size']:>8}{r['accuracy']:>10.3f}"
              f"{str(r['first_scored_chunk']):>14}{r['n_retrains']:>10}"
              f"{r['total_retrain_time'] * 1000:>20.1f}"
              f"{r['mean_retrain_time'] * 1000:>18.2f}"
              f"{r['peak_memory_bytes'] / 1024:>18.1f}")

    print("=" * 97)
    print("Accuracy is test-then-train: each chunk is scored before training.")
    print(f"Timing: median of {repeats} runs, after one warm-up.")
    return {"loop_vs_vectorised": timing, "buffer_sizes": buffers}


if __name__ == "__main__":
    info = environment_info()
    print("Environment:", ", ".join(f"{k}={v}" for k, v in info.items()))
    run_benchmarks()
    run_streaming_benchmarks()
