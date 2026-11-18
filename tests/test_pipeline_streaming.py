"""Tests for Pipeline.partial_fit and predict_proba (streaming pipelines)."""

import warnings

import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
from numcompute.tree import DecisionTreeClassifier
from numcompute.utils import NotFittedError


@pytest.fixture(autouse=True)
def _numpy_warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


@pytest.fixture
def stream():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 2.0, size=(1200, 3))
    y = (X[:, 0] + X[:, 1] > 10).astype(int)
    chunks = np.array_split(np.arange(1200), 12)
    return X, y, chunks


def _tree_pipeline(buffer_size=300, **tree_kwargs):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size,
                                         **tree_kwargs)),
    ])


# ---------------------------------------------------------
# Main path
# ---------------------------------------------------------

def test_streaming_pipeline_learns(stream):
    X, y, chunks = stream
    pipe = _tree_pipeline()
    for idx in chunks:
        pipe.partial_fit(X[idx], y[idx])

    assert pipe.named_steps["model"].n_retrains_ == 4
    assert np.mean(pipe.predict(X) == y) > 0.9


def test_partial_fit_equals_manual_chaining(stream):
    X, y, chunks = stream
    pipe = _tree_pipeline()
    scaler = StandardScaler()
    tree = DecisionTreeClassifier(max_depth=5, buffer_size=300)

    for idx in chunks:
        pipe.partial_fit(X[idx], y[idx])
        scaler.partial_fit(X[idx])
        tree.partial_fit(scaler.transform(X[idx]), y[idx])

    assert np.allclose(pipe.named_steps["scale"].mean_, scaler.mean_)
    assert np.array_equal(pipe.predict(X), tree.predict(scaler.transform(X)))


def test_each_chunk_is_transformed_with_stats_including_it():
    pipe = Pipeline([("scale", StandardScaler())])
    pipe.partial_fit(np.array([[0.0], [2.0]]))
    pipe.partial_fit(np.array([[10.0], [12.0]]))
    assert np.isclose(pipe.named_steps["scale"].mean_[0], 6.0)


def test_multi_step_streaming_with_missing_values(stream):
    X, y, chunks = stream
    X = X.copy()
    X[np.random.default_rng(1).random(X.shape) < 0.1] = np.nan

    pipe = Pipeline([
        ("impute", SimpleImputer("median")),
        ("scale", MinMaxScaler(clip=True)),
        ("model", DecisionTreeClassifier(max_depth=5, buffer_size=400)),
    ])
    for idx in chunks:
        pipe.partial_fit(X[idx], y[idx])

    predictions = pipe.predict(X)
    assert predictions.shape == y.shape
    assert np.mean(predictions == y) > 0.8


def test_predict_proba_passes_through(stream):
    X, y, chunks = stream
    pipe = _tree_pipeline()
    for idx in chunks[:3]:
        pipe.partial_fit(X[idx], y[idx])

    proba = pipe.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_transformer_only_pipeline_streams_without_y():
    pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    pipe.partial_fit(np.array([[1.0, np.nan], [3.0, 4.0]]))
    out = pipe.transform(np.array([[np.nan, 4.0]]))
    assert out.shape == (1, 2) and not np.isnan(out).any()


def test_batch_fit_still_works_with_tree(stream):
    X, y, _ = stream
    pipe = _tree_pipeline()
    pipe.fit(X, y)
    assert np.mean(pipe.predict(X) == y) > 0.9


# ---------------------------------------------------------
# Edge cases
# ---------------------------------------------------------

def test_predict_before_any_partial_fit_raises():
    with pytest.raises(NotFittedError):
        _tree_pipeline().predict(np.ones((1, 3)))


def test_predict_before_tree_retrains_raises_clear_error(stream):
    X, y, chunks = stream
    pipe = _tree_pipeline(buffer_size=10_000)
    pipe.partial_fit(X[chunks[0]], y[chunks[0]])
    with pytest.raises(NotFittedError, match="buffered"):
        pipe.predict(X)


def test_empty_chunk_is_a_no_op(stream):
    X, y, chunks = stream
    pipe = _tree_pipeline()
    pipe.partial_fit(np.empty((0, 3)), np.array([]))
    with pytest.raises(NotFittedError):
        pipe.predict(X)

    pipe.partial_fit(X[chunks[0]], y[chunks[0]])
    pipe.partial_fit(np.empty((0, 3)), np.array([]))
    assert pipe.named_steps["scale"].n_samples_seen_ == chunks[0].size


def test_estimator_pipeline_requires_y(stream):
    X, _, _ = stream
    with pytest.raises(ValueError, match="y_chunk"):
        _tree_pipeline().partial_fit(X[:10])


def test_step_without_partial_fit_is_rejected():
    class BatchOnly:
        def fit(self, X):
            return self

        def transform(self, X):
            return X

    pipe = Pipeline([("batch", BatchOnly()), ("scale", StandardScaler())])
    with pytest.raises(AttributeError, match="batch"):
        pipe.partial_fit(np.ones((2, 2)))


def test_no_step_is_updated_when_validation_fails():
    class BatchOnly:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X

    scaler = StandardScaler()
    pipe = Pipeline([("scale", scaler), ("model", BatchOnly())])
    with pytest.raises(AttributeError):
        pipe.partial_fit(np.ones((2, 2)), np.ones(2))
    assert not hasattr(scaler, "mean_")


def test_empty_pipeline_partial_fit_raises():
    with pytest.raises(ValueError, match="at least one"):
        Pipeline([]).partial_fit(np.ones((2, 2)))


def test_feature_count_change_between_chunks(stream):
    X, y, chunks = stream
    pipe = _tree_pipeline()
    pipe.partial_fit(X[chunks[0]], y[chunks[0]])
    with pytest.raises(ValueError, match="features"):
        pipe.partial_fit(X[chunks[1], :2], y[chunks[1]])


def test_growing_one_hot_width_before_tree_is_reported():
    pipe = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore")),
        ("model", DecisionTreeClassifier(buffer_size=1000)),
    ])
    pipe.partial_fit(np.array([["a"], ["b"]]), np.array([0, 1]))
    # A new category widens the encoded output; the tree's buffer has the
    # old width, so the pipeline must fail loudly rather than misalign.
    with pytest.raises(ValueError, match="features"):
        pipe.partial_fit(np.array([["c"]]), np.array([1]))


def test_predict_proba_requires_capable_final_step():
    pipe = Pipeline([("scale", StandardScaler())])
    pipe.partial_fit(np.ones((2, 1)))
    with pytest.raises(AttributeError, match="predict_proba"):
        pipe.predict_proba(np.ones((1, 1)))


def test_predict_proba_before_fit_raises():
    with pytest.raises(NotFittedError):
        _tree_pipeline().predict_proba(np.ones((1, 3)))
