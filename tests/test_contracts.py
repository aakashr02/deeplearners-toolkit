"""Cross-module contract tests: shared helpers, pipeline, metrics, package."""

import numpy as np
import pytest

import numcompute
from numcompute.metrics import accuracy, f1, mse, precision, recall
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import MinMaxScaler, StandardScaler
from numcompute.utils import (
    NotFittedError,
    check_is_fitted,
    validate_2d,
    validate_bool,
    validate_int,
)


# ---------------------------------------------------------
# Shared validation helpers (utils)
# ---------------------------------------------------------

def test_validate_bool_accepts_numpy_bool():
    assert validate_bool(np.bool_(True), "flag") is True


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_validate_bool_rejects_non_bool(value):
    with pytest.raises(TypeError):
        validate_bool(value, "flag")


def test_validate_int_bounds():
    assert validate_int(np.int64(3), "k", minimum=0, maximum=5) == 3
    with pytest.raises(ValueError):
        validate_int(-1, "k", minimum=0)
    with pytest.raises(ValueError):
        validate_int(6, "k", maximum=5)


@pytest.mark.parametrize("value", [True, 2.0, "3"])
def test_validate_int_rejects_non_int(value):
    with pytest.raises(TypeError):
        validate_int(value, "k")


def test_check_is_fitted():
    class Thing:
        pass

    thing = Thing()
    with pytest.raises(NotFittedError, match="Thing"):
        check_is_fitted(thing, "mean_")

    thing.mean_ = 0
    check_is_fitted(thing, ["mean_"])  # does not raise


def test_validate_2d_feature_count():
    validate_2d(np.ones((2, 3)), n_features=3)
    with pytest.raises(ValueError, match="fitted with 3 features"):
        validate_2d(np.ones((2, 2)), n_features=3)


def test_validate_2d_non_numeric_mode_accepts_strings():
    arr = validate_2d(np.array([["a"], ["b"]]), numeric=False)
    assert arr.shape == (2, 1)


# ---------------------------------------------------------
# Pipeline contracts
# ---------------------------------------------------------

class _MeanEstimator:
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X).mean(axis=1)


def test_pipeline_propagates_feature_mismatch():
    pipeline = Pipeline([("scale", StandardScaler())])
    pipeline.fit_transform(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError, match="features"):
        pipeline.transform(np.array([[1.0, 2.0, 3.0]]))


def test_pipeline_not_fitted_error_type():
    with pytest.raises(NotFittedError):
        Pipeline([("scale", StandardScaler())]).transform(np.ones((1, 1)))


def test_pipeline_constant_feature_is_finite_end_to_end():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    pipeline = Pipeline([("std", StandardScaler()), ("mm", MinMaxScaler())])
    assert np.all(np.isfinite(pipeline.fit_transform(X)))


def test_pipeline_estimator_requires_y():
    pipeline = Pipeline([("scale", StandardScaler()), ("model", _MeanEstimator())])
    with pytest.raises(ValueError, match="y is required"):
        pipeline.fit(np.array([[1.0], [2.0]]))


def test_pipeline_fit_transformer_only_without_y():
    pipeline = Pipeline([("scale", StandardScaler())])
    assert pipeline.fit(np.array([[1.0], [2.0]])) is pipeline


def test_pipeline_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("a", StandardScaler()), ("a", MinMaxScaler())])


@pytest.mark.parametrize("steps", [[StandardScaler()], [(1, StandardScaler())]])
def test_pipeline_rejects_malformed_steps(steps):
    with pytest.raises(TypeError):
        Pipeline(steps)


def test_pipeline_rejects_component_without_transform():
    class NoTransform:
        def fit(self, X):
            return self

    pipeline = Pipeline([("bad", NoTransform()), ("scale", StandardScaler())])
    with pytest.raises(AttributeError, match="bad"):
        pipeline.fit(np.array([[1.0]]))


def test_pipeline_named_steps():
    scaler = StandardScaler()
    assert Pipeline([("scale", scaler)]).named_steps["scale"] is scaler


# ---------------------------------------------------------
# Metrics return a consistent Python float
# ---------------------------------------------------------

@pytest.mark.parametrize("metric", [accuracy, precision, recall, f1, mse])
def test_metrics_return_python_float(metric):
    result = metric(np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1]))
    assert type(result) is float


@pytest.mark.parametrize("metric", [accuracy, precision, recall, f1, mse])
def test_metrics_reject_shape_mismatch_and_empty(metric):
    with pytest.raises(ValueError):
        metric([1, 0], [1, 0, 1])
    with pytest.raises(ValueError):
        metric([], [])


# ---------------------------------------------------------
# Package-level API
# ---------------------------------------------------------

def test_package_exports_public_api():
    for name in numcompute.__all__:
        assert hasattr(numcompute, name), name


def test_readme_quickstart_imports_work():
    from numcompute import binary_search, stable_sort

    sorted_values = stable_sort(np.array([5, 2, 8, 1, 3]))
    assert binary_search(sorted_values, 3) == (2, True)
