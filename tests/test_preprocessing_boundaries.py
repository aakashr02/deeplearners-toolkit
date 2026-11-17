"""Specification-based boundary tests for numcompute.preprocessing."""

import numpy as np
import pytest

from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from numcompute.utils import NotFittedError


# ---------------------------------------------------------
# Constant (zero-variance / zero-range) features
# ---------------------------------------------------------

def test_standard_scaler_constant_column_maps_to_zero():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    X_scaled = StandardScaler().fit_transform(X)

    assert np.all(np.isfinite(X_scaled))
    assert np.allclose(X_scaled[:, 1], 0.0)
    # The non-constant column is still standardised normally.
    assert np.isclose(X_scaled[:, 0].mean(), 0.0)
    assert np.isclose(X_scaled[:, 0].std(), 1.0)


def test_standard_scaler_constant_column_attributes():
    scaler = StandardScaler().fit(np.array([[2.0, 1.0], [2.0, 3.0]]))

    assert scaler.std_[0] == 0.0       # true statistic is kept
    assert scaler.scale_[0] == 1.0     # divisor actually used is safe
    assert scaler.n_features_in_ == 2


def test_standard_scaler_single_sample_is_finite():
    X_scaled = StandardScaler().fit_transform(np.array([[4.0, -2.0]]))
    assert np.array_equal(X_scaled, [[0.0, 0.0]])


def test_minmax_scaler_constant_column_maps_to_range_min():
    X = np.array([[1.0, 7.0], [3.0, 7.0]])
    X_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

    assert np.all(np.isfinite(X_scaled))
    assert np.allclose(X_scaled[:, 1], -1.0)
    assert np.allclose(X_scaled[:, 0], [-1.0, 1.0])


def test_minmax_scaler_constant_column_unseen_value_is_finite():
    scaler = MinMaxScaler().fit(np.array([[2.0], [2.0]]))
    assert np.all(np.isfinite(scaler.transform(np.array([[10.0]]))))


# ---------------------------------------------------------
# Fitted feature mismatches must not broadcast silently
# ---------------------------------------------------------

@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler])
def test_scaler_rejects_extra_features(make):
    scaler = make().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError, match="features"):
        scaler.transform(np.array([[1.0, 2.0, 3.0]]))


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler])
def test_scaler_rejects_single_column_that_would_broadcast(make):
    # Shape (n, 1) previously broadcast silently against 2 fitted features.
    scaler = make().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError, match="features"):
        scaler.transform(np.array([[1.0], [2.0]]))


def test_onehot_rejects_feature_count_mismatch():
    encoder = OneHotEncoder().fit(np.array([[1, 2], [3, 4]]))

    with pytest.raises(ValueError, match="features"):
        encoder.transform(np.array([[1]]))


# ---------------------------------------------------------
# Use before fit
# ---------------------------------------------------------

@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, OneHotEncoder])
def test_transform_before_fit_raises(make):
    with pytest.raises(NotFittedError, match="not fitted"):
        make().transform(np.array([[1.0, 2.0]]))


def test_not_fitted_error_is_runtime_and_attribute_error():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.array([[1.0]]))
    with pytest.raises(AttributeError):
        MinMaxScaler().transform(np.array([[1.0]]))


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, OneHotEncoder])
def test_rejects_1d_input(make):
    with pytest.raises(ValueError, match="2 dimensions"):
        make().fit(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, OneHotEncoder])
def test_rejects_empty_input(make):
    with pytest.raises(ValueError, match="at least one sample"):
        make().fit(np.empty((0, 2)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler])
def test_scaler_rejects_nan_and_inf(make, bad):
    with pytest.raises(ValueError):
        make().fit(np.array([[1.0], [bad]]))


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler])
def test_scaler_transform_rejects_nan(make):
    scaler = make().fit(np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError):
        scaler.transform(np.array([[np.nan]]))


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler])
def test_scaler_rejects_non_numeric(make):
    with pytest.raises(TypeError):
        make().fit(np.array([["a"], ["b"]]))


@pytest.mark.parametrize("feature_range", [(1, 1), (2, 0), (0, np.inf), (1,)])
def test_minmax_scaler_rejects_invalid_range(feature_range):
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=feature_range)


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler])
def test_scaler_does_not_modify_input(make):
    X = np.array([[1.0, 2.0], [3.0, 8.0]])
    original = X.copy()
    make().fit_transform(X)
    assert np.array_equal(X, original)


def test_scaler_accepts_integer_input_and_returns_float():
    X_scaled = StandardScaler().fit_transform(np.array([[1, 2], [3, 4]]))
    assert X_scaled.dtype == np.float64


def test_standard_scaler_transforms_new_data_with_training_stats():
    scaler = StandardScaler().fit(np.array([[0.0], [2.0]]))  # mean 1, std 1
    assert np.allclose(scaler.transform(np.array([[3.0]])), [[2.0]])


def test_minmax_scaler_does_not_clip_out_of_range_values():
    scaler = MinMaxScaler().fit(np.array([[0.0], [10.0]]))
    assert np.allclose(scaler.transform(np.array([[20.0]])), [[2.0]])


def test_fit_returns_self():
    for make in (StandardScaler, MinMaxScaler, OneHotEncoder):
        component = make()
        assert component.fit(np.array([[1.0], [2.0]])) is component


# ---------------------------------------------------------
# OneHotEncoder categories
# ---------------------------------------------------------

def test_onehot_string_categories():
    X = np.array([["red"], ["blue"], ["red"]])
    encoder = OneHotEncoder().fit(X)

    assert encoder.categories_[0].tolist() == ["blue", "red"]
    assert np.array_equal(encoder.transform(X), [[0, 1], [1, 0], [0, 1]])
    assert encoder.n_features_out_ == 2


def test_onehot_unknown_category_raises_by_default():
    encoder = OneHotEncoder().fit(np.array([[1], [2]]))

    with pytest.raises(ValueError, match="unknown categories"):
        encoder.transform(np.array([[3]]))


def test_onehot_unknown_category_ignore_gives_zero_row():
    encoder = OneHotEncoder(handle_unknown="ignore").fit(np.array([[1], [2]]))
    assert np.array_equal(encoder.transform(np.array([[3]])), [[0, 0]])


def test_onehot_rejects_invalid_handle_unknown():
    with pytest.raises(ValueError):
        OneHotEncoder(handle_unknown="skip")
