"""Tests for partial_fit in the scalers, SimpleImputer and OneHotEncoder."""

import numpy as np
import pytest

from numcompute.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


def _stream(component, X, n_chunks):
    for chunk in np.array_split(X, n_chunks):
        component.partial_fit(chunk)
    return component


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(10.0, 3.0, size=(200, 4))


@pytest.fixture
def X_missing():
    data = np.random.default_rng(1).normal(size=(300, 3)).round(1)
    data[np.random.default_rng(2).random(data.shape) < 0.15] = np.nan
    return data


# ---------------------------------------------------------
# StandardScaler.partial_fit
# ---------------------------------------------------------

@pytest.mark.parametrize("n_chunks", [1, 2, 7, 50])
def test_standard_scaler_partial_fit_matches_fit(X, n_chunks):
    streamed = _stream(StandardScaler(), X, n_chunks)
    batch = StandardScaler().fit(X)

    assert np.allclose(streamed.mean_, batch.mean_)
    assert np.allclose(streamed.std_, batch.std_)
    assert np.allclose(streamed.transform(X), batch.transform(X))
    assert streamed.n_samples_seen_ == 200


def test_standard_scaler_statistics_match_numpy(X):
    scaler = _stream(StandardScaler(), X, 5)
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    assert np.allclose(scaler.var_, X.var(axis=0))


def test_standard_scaler_large_offset_stream():
    data = 1e8 + np.random.default_rng(3).normal(size=(5000, 2))
    scaler = _stream(StandardScaler(), data, 25)
    assert np.allclose(scaler.std_, (data - 1e8).std(axis=0), rtol=1e-6)


def test_standard_scaler_transform_uses_stats_so_far():
    scaler = StandardScaler().partial_fit(np.array([[0.0], [2.0]]))
    assert np.allclose(scaler.transform([[3.0]]), [[2.0]])

    scaler.partial_fit(np.array([[4.0], [6.0]]))          # mean 3
    assert np.allclose(scaler.mean_, [3.0])
    assert np.allclose(scaler.transform([[3.0]]), [[0.0]])


def test_fit_resets_previous_stream(X):
    scaler = _stream(StandardScaler(), X + 100.0, 4)
    scaler.fit(X)
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    assert scaler.n_samples_seen_ == 200


def test_scaler_reset(X):
    scaler = StandardScaler().partial_fit(X)
    scaler.reset()
    assert not hasattr(scaler, "mean_")
    scaler.partial_fit(np.ones((3, 2)))
    assert scaler.n_features_in_ == 2


def test_partial_fit_rejects_feature_change(X):
    scaler = StandardScaler().partial_fit(X)
    with pytest.raises(ValueError, match="features"):
        scaler.partial_fit(np.ones((5, 3)))


# ---------------------------------------------------------
# MinMaxScaler.partial_fit
# ---------------------------------------------------------

@pytest.mark.parametrize("n_chunks", [1, 3, 40])
def test_minmax_partial_fit_matches_fit(X, n_chunks):
    streamed = _stream(MinMaxScaler(), X, n_chunks)
    batch = MinMaxScaler().fit(X)

    assert np.array_equal(streamed.data_min_, X.min(axis=0))
    assert np.array_equal(streamed.data_max_, X.max(axis=0))
    assert np.allclose(streamed.transform(X), batch.transform(X))


def test_minmax_range_expands_with_stream():
    scaler = MinMaxScaler().partial_fit(np.array([[0.0], [10.0]]))
    assert np.allclose(scaler.transform([[20.0]]), [[2.0]])

    scaler.partial_fit(np.array([[20.0]]))
    assert np.allclose(scaler.transform([[20.0]]), [[1.0]])


def test_minmax_clip():
    scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
    scaler.partial_fit(np.array([[0.0], [10.0]]))
    assert np.allclose(scaler.transform([[-5.0], [5.0], [50.0]]),
                       [[-1.0], [0.0], [1.0]])


# ---------------------------------------------------------
# SimpleImputer
# ---------------------------------------------------------

@pytest.mark.parametrize("n_chunks", [1, 4, 30])
def test_imputer_mean_matches_nanmean(X_missing, n_chunks):
    imputer = _stream(SimpleImputer("mean"), X_missing, n_chunks)
    assert np.allclose(imputer.statistics_, np.nanmean(X_missing, axis=0))


def test_imputer_median_matches_nanmedian(X_missing):
    imputer = _stream(SimpleImputer("median"), X_missing, 6)
    assert np.allclose(imputer.statistics_, np.nanmedian(X_missing, axis=0))


def test_imputer_most_frequent():
    X = np.array([[1.0, 5.0], [2.0, np.nan], [2.0, 5.0], [3.0, 7.0],
                  [np.nan, 7.0], [3.0, 7.0]])
    imputer = _stream(SimpleImputer("most_frequent"), X, 3)
    # Column 0: 2 and 3 tie (twice each) -> smallest wins.
    assert np.array_equal(imputer.statistics_, [2.0, 7.0])


def test_imputer_constant():
    imputer = SimpleImputer("constant", fill_value=-1.0)
    out = imputer.fit_transform(np.array([[1.0, np.nan], [np.nan, 4.0]]))
    assert np.array_equal(out, [[1.0, -1.0], [-1.0, 4.0]])


def test_imputer_transform_fills_every_nan(X_missing):
    imputer = _stream(SimpleImputer("mean"), X_missing, 5)
    out = imputer.transform(X_missing)

    assert not np.isnan(out).any()
    observed = ~np.isnan(X_missing)
    assert np.array_equal(out[observed], X_missing[observed])


def test_imputer_estimates_update_on_the_fly():
    imputer = SimpleImputer("mean").partial_fit(np.array([[1.0], [3.0]]))
    assert imputer.statistics_[0] == 2.0
    imputer.partial_fit(np.array([[np.nan], [8.0]]))
    assert imputer.statistics_[0] == 4.0
    assert imputer.n_missing_seen_[0] == 1


def test_imputer_does_not_modify_input(X_missing):
    original = X_missing.copy()
    SimpleImputer().fit_transform(X_missing)
    assert np.array_equal(X_missing, original, equal_nan=True)


@pytest.mark.parametrize(
    "kwargs", [{"strategy": "mode"}, {"strategy": "constant"}]
)
def test_imputer_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        SimpleImputer(**kwargs)


# ---------------------------------------------------------
# OneHotEncoder.partial_fit (incremental categories)
# ---------------------------------------------------------

def test_onehot_expands_categories_incrementally():
    encoder = OneHotEncoder()
    encoder.partial_fit(np.array([["b"], ["a"]]))
    assert encoder.categories_[0].tolist() == ["a", "b"]
    assert encoder.n_features_out_ == 2

    encoder.partial_fit(np.array([["c"], ["a"]]))
    assert encoder.categories_[0].tolist() == ["a", "b", "c"]
    assert encoder.n_features_out_ == 3


def test_onehot_existing_columns_do_not_move():
    encoder = OneHotEncoder().partial_fit(np.array([[5], [9]]))
    before = encoder.transform(np.array([[5], [9]]))

    encoder.partial_fit(np.array([[1]]))         # sorts before 5 and 9
    after = encoder.transform(np.array([[5], [9], [1]]))

    assert np.array_equal(after[:2, :2], before)
    assert np.array_equal(after[2], [0, 0, 1])


def test_onehot_multi_column_stream():
    encoder = OneHotEncoder()
    encoder.partial_fit(np.array([["x", 1], ["y", 1]], dtype=object))
    encoder.partial_fit(np.array([["x", 2]], dtype=object))

    out = encoder.transform(np.array([["y", 2]], dtype=object))
    assert out.shape == (1, 4)
    assert np.array_equal(out, [[0, 1, 0, 1]])


def test_onehot_fit_matches_sorted_unique():
    X = np.array([[3], [1], [2], [1]])
    assert OneHotEncoder().fit(X).categories_[0].tolist() == [1, 2, 3]
