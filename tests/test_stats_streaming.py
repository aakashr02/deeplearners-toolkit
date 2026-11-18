"""Tests for the streaming statistics in numcompute.stats."""

import warnings

import numpy as np
import pytest

from numcompute.io import read_csv_chunks
from numcompute.stats import (
    RunningStats,
    StreamingHistogram,
    StreamingQuantile,
    update_stats,
    variance,
)
from numcompute.utils import NotFittedError


@pytest.fixture(autouse=True)
def _numpy_warnings_are_errors():
    # Empty chunks, all-NaN features and ddof edge cases must be handled
    # explicitly, not by emitting RuntimeWarnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def _feed(stats, chunks):
    for chunk in chunks:
        stats.update_stats(chunk)
    return stats


def _split(X, cuts):
    return np.split(X, cuts)


SPLITS = [
    [],                    # one chunk
    [1],                   # single-row first chunk
    [3, 3, 7],             # includes an empty chunk
    list(range(1, 20)),    # one row per chunk
    [10, 10, 10],          # several empty chunks in a row
]


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(5.0, 3.0, size=(20, 3))
    X[4, 0] = np.nan
    X[11, 2] = np.nan
    return X


# =========================================================
# RunningStats: equivalence with whole-array statistics
# =========================================================

@pytest.mark.parametrize("cuts", SPLITS)
def test_running_stats_matches_batch_for_any_chunking(data, cuts):
    rs = _feed(RunningStats(), _split(data, cuts))

    assert np.allclose(rs.mean, np.nanmean(data, axis=0))
    assert np.allclose(rs.variance(), np.nanvar(data, axis=0))
    assert np.allclose(rs.variance(ddof=1), np.nanvar(data, axis=0, ddof=1))
    assert np.allclose(rs.std(), np.nanstd(data, axis=0))
    assert np.allclose(rs.min, np.nanmin(data, axis=0))
    assert np.allclose(rs.max, np.nanmax(data, axis=0))
    assert np.array_equal(rs.count, [19, 20, 19])
    assert rs.n_samples_seen_ == 20
    assert rs.n_chunks_seen_ == len(cuts) + 1


def test_running_stats_chunking_is_order_independent(data):
    forward = _feed(RunningStats(), _split(data, [5, 12]))
    backward = _feed(RunningStats(), _split(data, [5, 12])[::-1])
    assert np.allclose(forward.mean, backward.mean)
    assert np.allclose(forward.variance(), backward.variance())


def test_running_stats_1d_stream_returns_python_scalars():
    rs = _feed(RunningStats(), ([1.0, 2.0], [3.0], [4.0, 5.0]))
    assert rs.mean == 3.0 and type(rs.mean) is float
    assert rs.variance() == 2.0
    assert rs.count == 5 and type(rs.count) is int
    assert rs.min == 1.0 and rs.max == 5.0


def test_running_stats_numerically_stable_with_large_offset():
    # The naive E[x^2] - E[x]^2 formula loses all precision here.
    rng = np.random.default_rng(0)
    offset_data = 1e9 + rng.normal(0.0, 1.0, 10_000)
    rs = _feed(RunningStats(), np.array_split(offset_data, 37))

    exact = np.var(offset_data - 1e9)
    assert np.isclose(rs.variance(), exact, rtol=1e-6)

    naive = np.mean(offset_data ** 2) - np.mean(offset_data) ** 2
    assert not np.isclose(naive, exact, rtol=1e-2)


def test_running_stats_constant_feature_has_zero_variance():
    rs = _feed(RunningStats(), np.array_split(np.full((50, 2), 0.1), 7))
    assert np.all(rs.variance() == 0.0)
    assert np.all(rs.std(ddof=1) == 0.0)


def test_running_stats_all_equal_large_values():
    rs = _feed(RunningStats(), ([1e12] * 3, [1e12] * 4))
    assert rs.mean == 1e12
    assert rs.variance() == 0.0


# =========================================================
# RunningStats: NaN handling
# =========================================================

def test_running_stats_counts_nan_per_feature(data):
    rs = _feed(RunningStats(), _split(data, [6]))
    assert np.array_equal(rs.n_nan_, [1, 0, 1])


def test_running_stats_all_nan_feature_gives_nan_without_warning():
    X = np.array([[1.0, np.nan], [3.0, np.nan]])
    rs = _feed(RunningStats(), (X[:1], X[1:]))

    assert np.isnan(rs.mean[1]) and rs.mean[0] == 2.0
    assert np.isnan(rs.variance()[1])
    assert np.isnan(rs.min[1]) and np.isnan(rs.max[1])
    assert np.array_equal(rs.count, [2, 0])


def test_running_stats_nan_only_first_chunk_then_data():
    rs = _feed(RunningStats(), ([np.nan, np.nan], [4.0, 6.0]))
    assert rs.mean == 5.0 and rs.count == 2


def test_running_stats_ignore_nan_false_marks_feature_nan():
    X = np.array([[1.0, 2.0], [np.nan, 4.0]])
    rs = RunningStats(ignore_nan=False).update_stats(X)
    assert np.isnan(rs.mean[0])
    assert rs.mean[1] == 3.0


# =========================================================
# RunningStats: ddof, empty input and state handling
# =========================================================

def test_running_stats_ddof_larger_than_count_is_nan():
    rs = RunningStats().update_stats([5.0])
    assert rs.variance(ddof=0) == 0.0
    assert np.isnan(rs.variance(ddof=1))
    assert np.isnan(rs.std(ddof=1))


@pytest.mark.parametrize("ddof", [-1, 1.5, True])
def test_running_stats_rejects_invalid_ddof(ddof):
    rs = RunningStats().update_stats([1.0, 2.0])
    with pytest.raises((TypeError, ValueError)):
        rs.variance(ddof=ddof)


def test_running_stats_before_any_data_raises():
    rs = RunningStats()
    for getter in (lambda: rs.mean, lambda: rs.variance(), lambda: rs.min,
                   lambda: rs.count, rs.result):
        with pytest.raises(NotFittedError):
            getter()


def test_running_stats_only_empty_chunks():
    rs = _feed(RunningStats(), (np.empty((0, 2)), np.empty((0, 2))))
    assert rs.n_features_in_ == 2
    assert np.all(np.isnan(rs.mean))
    assert np.array_equal(rs.count, [0, 0])
    assert rs.n_samples_seen_ == 0 and rs.n_chunks_seen_ == 2


def test_running_stats_empty_chunk_with_no_columns_is_ignored():
    rs = RunningStats().update_stats(np.empty((0, 0)))
    with pytest.raises(NotFittedError):
        rs.mean
    rs.update_stats([[1.0, 2.0]])
    assert np.array_equal(rs.mean, [1.0, 2.0])


def test_running_stats_rejects_feature_count_change():
    rs = RunningStats().update_stats(np.ones((3, 2)))
    with pytest.raises(ValueError, match="features"):
        rs.update_stats(np.ones((3, 3)))
    with pytest.raises(ValueError, match="features"):
        rs.update_stats(np.empty((0, 4)))


def test_running_stats_rejects_infinity_and_bad_types():
    rs = RunningStats()
    with pytest.raises(ValueError, match="infinite"):
        rs.update_stats([1.0, np.inf])
    with pytest.raises(TypeError):
        rs.update_stats(["a", "b"])
    with pytest.raises(TypeError):
        rs.update_stats(np.array([True, False]))
    with pytest.raises(ValueError, match="1D or 2D"):
        rs.update_stats(np.ones((2, 2, 2)))
    with pytest.raises(TypeError):
        RunningStats(ignore_nan=1)


def test_running_stats_failed_update_leaves_state_unchanged():
    rs = RunningStats().update_stats([[1.0, 2.0]])
    with pytest.raises(ValueError):
        rs.update_stats([[1.0, 2.0, 3.0]])
    assert rs.n_samples_seen_ == 1
    assert np.array_equal(rs.mean, [1.0, 2.0])


def test_running_stats_reset():
    rs = RunningStats().update_stats(np.ones((4, 2)))
    rs.reset()
    with pytest.raises(NotFittedError):
        rs.mean
    rs.update_stats(np.ones((2, 5)))  # new feature count allowed after reset
    assert rs.n_features_in_ == 5


def test_running_stats_accepts_integers_and_strided_views():
    base = np.arange(40, dtype=np.int32).reshape(10, 4)
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    rs = RunningStats().update_stats(view)
    assert np.allclose(rs.mean, view.mean(axis=0))


def test_running_stats_does_not_modify_input():
    chunk = np.array([[1.0, np.nan], [3.0, 4.0]])
    original = chunk.copy()
    RunningStats().update_stats(chunk)
    assert np.array_equal(chunk, original, equal_nan=True)


def test_running_stats_result_dict(data):
    result = _feed(RunningStats(), _split(data, [7])).result(ddof=1)
    assert set(result) == {"count", "mean", "variance", "std", "min", "max",
                           "n_samples_seen", "n_chunks_seen"}
    assert np.allclose(result["variance"], np.nanvar(data, axis=0, ddof=1))


# =========================================================
# RunningStats: merge (parallel reduction) and helpers
# =========================================================

def test_running_stats_merge_equals_sequential(data):
    left = _feed(RunningStats(), _split(data[:8], [3]))
    right = _feed(RunningStats(), _split(data[8:], [5]))
    merged = left.merge(right)
    sequential = _feed(RunningStats(), _split(data, [3, 8, 13]))

    assert np.allclose(merged.mean, sequential.mean)
    assert np.allclose(merged.variance(), sequential.variance())
    assert np.array_equal(merged.count, sequential.count)
    assert merged.n_samples_seen_ == 20


def test_running_stats_merge_with_empty_sides():
    filled = RunningStats().update_stats([1.0, 3.0])
    assert RunningStats().merge(filled).mean == 2.0
    assert filled.merge(RunningStats()).mean == 2.0


def test_running_stats_merge_rejects_mismatches():
    a = RunningStats().update_stats(np.ones((2, 2)))
    with pytest.raises(ValueError, match="features"):
        a.merge(RunningStats().update_stats(np.ones((2, 3))))
    with pytest.raises(ValueError, match="ignore_nan"):
        a.merge(RunningStats(ignore_nan=False))
    with pytest.raises(TypeError):
        a.merge("not stats")


def test_running_stats_from_csv_chunks(tmp_path):
    X = np.arange(1.0, 61.0).reshape(30, 2)
    path = tmp_path / "stream.csv"
    np.savetxt(path, X, delimiter=",")

    rs = RunningStats.from_chunks(read_csv_chunks(path, chunk_size=7))
    assert np.allclose(rs.mean, X.mean(axis=0))
    assert np.allclose(rs.std(), X.std(axis=0))


def test_partial_fit_alias():
    rs = RunningStats()
    rs.partial_fit([1.0, 2.0])
    assert rs.mean == 1.5


# =========================================================
# StreamingHistogram
# =========================================================

@pytest.mark.parametrize("cuts", SPLITS)
def test_histogram_matches_numpy_histogram(cuts):
    values = np.random.default_rng(1).uniform(-2, 2, 20)
    h = _feed(StreamingHistogram(bins=8, range=(-2, 2)), _split(values, cuts))
    counts, edges = h.result()
    expected_counts, expected_edges = np.histogram(values, bins=8, range=(-2, 2))

    assert np.array_equal(counts, expected_counts)
    assert np.allclose(edges, expected_edges)


def test_histogram_right_edge_and_out_of_range():
    h = StreamingHistogram(bins=2, range=(0, 1))
    h.update_stats([0.0, 0.5, 1.0, -0.1, 1.1, 2.0, np.nan])

    assert np.array_equal(h.counts_, [1, 2])     # 1.0 joins the last bin
    assert h.n_below_ == 1 and h.n_above_ == 2 and h.n_nan_ == 1


def test_histogram_sliding_window_evicts_old_chunks():
    h = StreamingHistogram(bins=2, range=(0, 2), window=2)
    h.update_stats([0.5, 0.5, 0.5])   # evicted later
    h.update_stats([1.5])
    h.update_stats([0.5, 1.5])

    assert np.array_equal(h.counts_, [1, 2])
    assert h.n_chunks_seen_ == 3


def test_histogram_window_of_one_shows_latest_chunk():
    h = StreamingHistogram(bins=[0, 1, 2], window=1)
    for chunk in ([0.5], [1.5, 1.5], [0.2]):
        h.update_stats(chunk)
    assert np.array_equal(h.counts_, [1, 0])


def test_histogram_per_feature_counts():
    X = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.6]])
    h = StreamingHistogram(bins=2, range=(0, 1)).update_stats(X)
    assert h.counts_.shape == (2, 2)
    assert np.array_equal(h.counts_, [[2, 1], [0, 3]])


def test_histogram_explicit_uneven_edges():
    h = StreamingHistogram(bins=[0.0, 1.0, 10.0]).update_stats([0.5, 2.0, 9.0])
    assert np.array_equal(h.counts_, [1, 2])


def test_histogram_empty_chunks_and_reset():
    h = StreamingHistogram(bins=2, range=(0, 1))
    h.update_stats(np.array([]))
    assert np.array_equal(h.counts_, [0, 0])

    h.update_stats([0.2, 0.7]).reset()
    with pytest.raises(NotFittedError):
        h.counts_
    assert h.bin_edges_.size == 3  # edges survive reset


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bins": 5},                          # int bins without range
        {"bins": 5, "range": (1, 1)},
        {"bins": 5, "range": (0, np.inf)},
        {"bins": 0, "range": (0, 1)},
        {"bins": [0, 1, 1]},                  # not strictly increasing
        {"bins": [1]},
        {"bins": 2, "range": (0, 1), "window": 0},
    ],
)
def test_histogram_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        StreamingHistogram(**kwargs)


def test_histogram_rejects_feature_change():
    h = StreamingHistogram(bins=2, range=(0, 1)).update_stats([[0.1, 0.2]])
    with pytest.raises(ValueError, match="features"):
        h.update_stats([0.1])


# =========================================================
# StreamingQuantile
# =========================================================

@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
)
def test_quantile_matches_numpy_across_chunks(data, interpolation):
    q = [0.0, 0.1, 0.5, 0.9, 1.0]
    sq = _feed(StreamingQuantile(q, interpolation=interpolation),
               _split(data, [1, 9, 9]))

    expected = np.nanquantile(data, q, axis=0, method=interpolation)
    assert sq.result().shape == (5, 3)
    assert np.allclose(sq.result(), expected)


def test_quantile_scalar_q_on_1d_stream_is_float():
    sq = _feed(StreamingQuantile(0.5), ([3.0, 1.0], [2.0]))
    assert sq.result() == 2.0 and type(sq.result()) is float


def test_quantile_window_uses_recent_chunks_only():
    sq = StreamingQuantile(0.5, window=2)
    for chunk in ([100.0, 100.0], [1.0, 2.0], [3.0]):
        sq.update_stats(chunk)
    assert sq.result() == 2.0
    assert sq.n_values_stored_ == 3


def test_quantile_unbounded_stream_consolidates_correctly():
    values = np.random.default_rng(3).normal(size=1000)
    sq = _feed(StreamingQuantile([0.25, 0.75]), np.array_split(values, 200))
    assert np.allclose(sq.result(), np.quantile(values, [0.25, 0.75]))
    assert sq.n_values_stored_ == 1000


def test_quantile_all_nan_feature_and_empty_window():
    sq = StreamingQuantile(0.5).update_stats([[1.0, np.nan], [3.0, np.nan]])
    result = sq.result()
    assert result[0] == 2.0 and np.isnan(result[1])

    only_empty = StreamingQuantile(0.5).update_stats(np.empty((0, 2)))
    assert np.all(np.isnan(only_empty.result()))


def test_quantile_copies_chunk():
    chunk = np.array([1.0, 2.0, 3.0])
    sq = StreamingQuantile(0.5).update_stats(chunk)
    chunk[:] = 100.0
    assert sq.result() == 2.0


def test_quantile_before_data_and_reset():
    sq = StreamingQuantile(0.5)
    with pytest.raises(NotFittedError):
        sq.result()
    sq.update_stats([1.0]).reset()
    with pytest.raises(NotFittedError):
        sq.result()


@pytest.mark.parametrize(
    "kwargs",
    [{"q": 1.5}, {"q": -0.1}, {"q": np.nan},
     {"q": 0.5, "interpolation": "cubic"}, {"q": 0.5, "window": 0},
     {"q": 0.5, "max_centroids": 1},
     {"q": 0.5, "max_centroids": 10, "window": 2}],
)
def test_quantile_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        StreamingQuantile(**kwargs)


# =========================================================
# StreamingQuantile: bounded-memory sketch
# =========================================================

@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
)
def test_sketch_is_exact_before_compression(data, interpolation):
    q = [0.1, 0.5, 0.9]
    sq = _feed(StreamingQuantile(q, interpolation=interpolation,
                                 max_centroids=50), _split(data, [4, 4, 13]))

    assert sq.is_exact_
    assert np.allclose(
        sq.result(), np.nanquantile(data, q, axis=0, method=interpolation)
    )


def test_sketch_memory_is_bounded():
    values = np.random.default_rng(5).normal(size=(50_000, 2))
    sq = StreamingQuantile(0.5, max_centroids=64)

    peak = 0
    for chunk in np.array_split(values, 250):
        sq.update_stats(chunk)
        peak = max(peak, sq.n_values_stored_)

    assert not sq.is_exact_
    # At most 2 * K raw values plus one chunk per feature before compressing.
    assert peak <= 2 * (2 * 64 + 200)
    assert sq.n_values_stored_ <= 2 * 64


@pytest.mark.parametrize("distribution", ["normal", "uniform", "exponential"])
def test_sketch_quantiles_are_accurate_after_compression(distribution):
    rng = np.random.default_rng(9)
    values = getattr(rng, distribution)(size=100_000)
    q = [0.01, 0.25, 0.5, 0.75, 0.99]
    sq = _feed(StreamingQuantile(q, max_centroids=256),
               np.array_split(values, 100))

    expected = np.quantile(values, q)
    spread = values.max() - values.min()
    assert np.max(np.abs(sq.result() - expected)) < 0.01 * spread


def test_sketch_extreme_quantiles_are_exact_min_max():
    values = np.random.default_rng(2).normal(size=10_000)
    sq = _feed(StreamingQuantile([0.0, 1.0], max_centroids=32),
               np.array_split(values, 20))
    assert not sq.is_exact_
    assert np.allclose(sq.result(), [values.min(), values.max()])


def test_sketch_result_is_monotone_in_q():
    values = np.random.default_rng(4).exponential(size=20_000)
    q = np.linspace(0, 1, 41)
    sq = _feed(StreamingQuantile(q, max_centroids=32),
               np.array_split(values, 40))
    assert np.all(np.diff(sq.result()) >= 0)


def test_sketch_handles_nan_constant_and_all_nan_features():
    X = np.column_stack([
        np.random.default_rng(6).normal(size=3000),
        np.full(3000, 7.0),
        np.full(3000, np.nan),
    ])
    X[::5, 0] = np.nan
    sq = _feed(StreamingQuantile([0.2, 0.8], max_centroids=16),
               np.array_split(X, 30))
    result = sq.result()

    assert result.shape == (2, 3)
    assert np.allclose(result[:, 1], 7.0)            # constant feature
    assert np.all(np.isnan(result[:, 2]))            # all-NaN feature
    assert np.allclose(result[:, 0],
                       np.nanquantile(X[:, 0], [0.2, 0.8]), atol=0.1)


def test_sketch_ignores_empty_chunks_and_resets():
    sq = StreamingQuantile(0.5, max_centroids=8)
    sq.update_stats(np.array([]))
    assert np.isnan(sq.result())

    _feed(sq, np.array_split(np.arange(100.0), 10))
    assert not sq.is_exact_
    sq.reset()
    assert sq.is_exact_ and sq.n_values_stored_ == 0
    with pytest.raises(NotFittedError):
        sq.result()


def test_sketch_scalar_output_on_1d_stream():
    sq = _feed(StreamingQuantile(0.5, max_centroids=4),
               ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]))
    assert type(sq.result()) is float
    assert abs(sq.result() - 5.0) < 1.0


# =========================================================
# update_stats function and one-shot variance
# =========================================================

def test_update_stats_function_creates_and_updates_state():
    state = None
    for chunk in ([1.0, 2.0], [3.0, 4.0]):
        state = update_stats(chunk, state)
    assert isinstance(state, RunningStats)
    assert state.mean == 2.5


def test_update_stats_function_accepts_other_states():
    h = StreamingHistogram(bins=2, range=(0, 1))
    assert update_stats([0.1, 0.9], h) is h
    assert np.array_equal(h.counts_, [1, 1])


def test_update_stats_function_rejects_bad_state():
    with pytest.raises(TypeError):
        update_stats([1.0], state={"mean": 0})


def test_one_shot_variance():
    data = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    assert np.allclose(variance(data, axis=0), np.nanvar(data, axis=0))
    assert np.isclose(variance([1, 2, 3, 4], ddof=1), np.var([1, 2, 3, 4], ddof=1))
    with pytest.raises(ValueError):
        variance([])
