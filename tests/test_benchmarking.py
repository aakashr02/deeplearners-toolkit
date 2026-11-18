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


# ---------------------------------------------------------
# Streaming benchmarks: data generator
# ---------------------------------------------------------

def test_make_stream_data_shapes_and_labels():
    X, y = bm.make_stream_data(500, n_features=5, seed=1)
    assert X.shape == (500, 5) and y.shape == (500,)
    assert set(np.unique(y)) <= {0, 1}


def test_make_stream_data_is_reproducible():
    a = bm.make_stream_data(200, seed=3)
    b = bm.make_stream_data(200, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_make_stream_data_missing_values_only_in_features():
    X, y = bm.make_stream_data(4000, missing_rate=0.1, seed=0)
    assert 0.08 < np.isnan(X).mean() < 0.12
    assert not np.isnan(y.astype(float)).any()


def test_make_stream_data_drift_changes_the_rule():
    X, y = bm.make_stream_data(4000, drift_at=0.5, label_noise=0.0, seed=0)
    first, second = slice(0, 2000), slice(2000, None)
    rule_before = (X[:, 0] + X[:, 1] ** 2 > 0.5).astype(int)
    rule_after = (X[:, 0] - X[:, 2] > 0).astype(int)
    assert np.array_equal(y[first], rule_before[first])
    assert np.array_equal(y[second], rule_after[second])


@pytest.mark.parametrize(
    "kwargs",
    [{"n_samples": 0}, {"n_features": 2}, {"drift_at": 1.5},
     {"missing_rate": 1.0}, {"label_noise": 0.5}],
)
def test_make_stream_data_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        bm.make_stream_data(**kwargs)


# ---------------------------------------------------------
# Loop-based references give identical results
# ---------------------------------------------------------

@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_loop_tree_builds_identical_tree(criterion):
    X, y = bm.make_stream_data(600, seed=5)
    fast = bm.DecisionTreeClassifier(criterion=criterion, max_depth=5).fit(X, y)
    slow = bm.LoopDecisionTreeClassifier(criterion=criterion,
                                         max_depth=5).fit(X, y)
    assert np.array_equal(fast.feature_, slow.feature_)
    assert np.array_equal(fast.threshold_, slow.threshold_, equal_nan=True)


def test_loop_tree_identical_under_streaming():
    X, y = bm.make_stream_data(1200, seed=6)
    fast = bm.DecisionTreeClassifier(max_depth=4, buffer_size=300)
    slow = bm.LoopDecisionTreeClassifier(max_depth=4, buffer_size=300)
    for start in range(0, 1200, 100):
        fast.partial_fit(X[start:start + 100], y[start:start + 100])
        slow.partial_fit(X[start:start + 100], y[start:start + 100])
    assert fast.n_retrains_ == slow.n_retrains_ == 4
    assert np.array_equal(fast.predict(X), slow.predict(X))


def test_loop_tree_handles_ties_and_constant_features():
    X = np.array([[1.0, 5.0], [1.0, 5.0], [2.0, 5.0], [2.0, 5.0]])
    y = np.array([0, 1, 1, 1])
    fast = bm.DecisionTreeClassifier().fit(X, y)
    slow = bm.LoopDecisionTreeClassifier().fit(X, y)
    assert np.array_equal(fast.feature_, slow.feature_)


def test_loop_tree_predict_matches_vectorised():
    X, y = bm.make_stream_data(800, seed=2)
    tree = bm.DecisionTreeClassifier(max_depth=6).fit(X, y)
    assert np.array_equal(bm.loop_tree_predict(tree, X), tree.predict(X))


def test_loop_tree_predict_string_labels():
    tree = bm.DecisionTreeClassifier().fit([[0.0], [1.0]], ["no", "yes"])
    assert bm.loop_tree_predict(tree, [[0.2], [0.9]]).tolist() == ["no", "yes"]


def test_loop_running_stats_matches_running_stats():
    rng = np.random.default_rng(0)
    chunks = [rng.normal(50, 3, size=(n, 3)) for n in (5, 1, 40)]
    loop_mean_, loop_var = bm.loop_running_stats(chunks)
    data = np.vstack(chunks)
    assert np.allclose(loop_mean_, data.mean(axis=0))
    assert np.allclose(loop_var, data.var(axis=0))


# ---------------------------------------------------------
# Streaming benchmark functions
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "run, operation",
    [
        (lambda: bm.benchmark_tree_streaming(n_samples=400, chunk_size=50,
                                             buffer_size=100, max_depth=3,
                                             repeats=1), "Tree partial_fit"),
        (lambda: bm.benchmark_tree_predict(n_samples=300, max_depth=3,
                                           repeats=1), "Tree predict"),
        (lambda: bm.benchmark_running_stats(n_samples=300, n_features=2,
                                            chunk_size=50, repeats=1),
         "RunningStats"),
    ],
)
def test_streaming_timing_result_contract(run, operation):
    result = run()
    assert set(result) == RESULT_KEYS
    assert result["operation"] == operation
    assert result["loop_time"] > 0 and result["vector_time"] > 0


def test_vectorised_tree_retraining_is_faster_than_loops():
    result = bm.benchmark_tree_streaming(n_samples=2000, chunk_size=200,
                                         buffer_size=1000, repeats=1)
    assert result["speedup"] > 1.0


def test_buffer_size_benchmark_fields_and_trends():
    results = bm.benchmark_buffer_sizes((100, 400, 1600), n_samples=3200,
                                        chunk_size=100)
    assert [r["buffer_size"] for r in results] == [100, 400, 1600]

    retrains = [r["n_retrains"] for r in results]
    assert retrains == [32, 8, 2]
    first_scored = [r["first_scored_chunk"] for r in results]
    assert first_scored == [1, 4, 16]               # bigger buffer, later start
    assert results[2]["peak_memory_bytes"] > results[0]["peak_memory_bytes"]
    for r in results:
        assert 0.5 < r["accuracy"] <= 1.0
        assert r["chunk_accuracy"].shape == (32,)


def test_buffer_size_benchmark_with_missing_values():
    X, y = bm.make_stream_data(600, missing_rate=0.05, seed=4)
    results = bm.benchmark_buffer_sizes((200,), X=X, y=y, chunk_size=100)
    assert not np.isnan(results[0]["accuracy"])


def test_buffer_larger_than_stream_never_scores():
    results = bm.benchmark_buffer_sizes((10_000,), n_samples=300,
                                        chunk_size=100)
    assert results[0]["n_retrains"] == 0
    assert results[0]["first_scored_chunk"] is None
    assert np.isnan(results[0]["accuracy"])
    assert np.isnan(results[0]["mean_retrain_time"])


@pytest.mark.parametrize(
    "kwargs",
    [{"buffer_sizes": ()}, {"buffer_sizes": (0,)}, {"chunk_size": 0},
     {"X": np.ones((5, 2)), "y": np.ones(4)}],
)
def test_buffer_size_benchmark_rejects_invalid_input(kwargs):
    with pytest.raises(ValueError):
        bm.benchmark_buffer_sizes(**kwargs)


def test_environment_info_keys():
    info = bm.environment_info()
    assert set(info) == {"platform", "processor", "python", "numpy"}
    assert info["numpy"] == np.__version__
