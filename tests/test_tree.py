"""Tests for DecisionTreeClassifier: training, prediction, streaming."""

import warnings

import numpy as np
import pytest

from numcompute.tree import DecisionTreeClassifier
from numcompute.utils import NotFittedError


@pytest.fixture(autouse=True)
def _numpy_warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = ((X[:, 0] + X[:, 1] ** 2) > 0.5).astype(int)
    return X, y


def _reference_best_split(X, y, criterion):
    """Exhaustive loop-based search used to check the vectorised one."""
    classes = np.unique(y)

    def impurity(labels):
        if labels.size == 0:
            return 0.0
        p = np.array([np.mean(labels == c) for c in classes])
        if criterion == "gini":
            return 1.0 - np.sum(p ** 2)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    best = (np.inf, None, None)
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for low, high in zip(values[:-1], values[1:]):
            cut = (low + high) / 2
            left = X[:, f] <= cut
            score = (left.sum() * impurity(y[left])
                     + (~left).sum() * impurity(y[~left])) / y.size
            if score < best[0] - 1e-12:
                best = (score, f, cut)
    return best


# =========================================================
# Training and prediction
# =========================================================

@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_root_split_matches_exhaustive_search(criterion):
    rng = np.random.default_rng(1)
    X = rng.integers(0, 6, size=(80, 3)).astype(float)
    y = rng.integers(0, 3, size=80)

    tree = DecisionTreeClassifier(criterion=criterion, max_depth=1).fit(X, y)
    _, feature, cut = _reference_best_split(X, y, criterion)

    assert tree.feature_[0] == feature
    assert np.isclose(tree.threshold_[0], cut)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_learns_separable_data_perfectly(blobs, criterion):
    X, y = blobs
    tree = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    assert tree.score(X, y) == 1.0


def test_generalises_on_held_out_data(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=6).fit(X[:450], y[:450])
    assert tree.score(X[450:], y[450:]) > 0.85


@pytest.mark.parametrize("max_depth", [0, 1, 2, 4])
def test_max_depth_is_respected(blobs, max_depth):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=max_depth).fit(X, y)
    assert tree.get_depth() <= max_depth
    assert tree.get_n_leaves() <= 2 ** max_depth


def test_max_depth_zero_predicts_majority_class():
    X = np.arange(10.0).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    tree = DecisionTreeClassifier(max_depth=0).fit(X, y)
    assert tree.node_count_ == 1
    assert np.all(tree.predict(X) == 1)


def test_min_samples_split_stops_growth():
    X = np.arange(8.0).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    tree = DecisionTreeClassifier(min_samples_split=9).fit(X, y)
    assert tree.node_count_ == 1


def test_simple_threshold_is_midpoint():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    tree = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
    assert tree.threshold_[0] == 2.5
    assert np.array_equal(tree.predict([[2.4], [2.6]]), [0, 1])


def test_predict_proba_rows_sum_to_one(blobs):
    X, y = blobs
    proba = DecisionTreeClassifier(max_depth=3).fit(X, y).predict_proba(X)
    assert proba.shape == (600, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_multiclass_and_string_labels():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [20.0], [21.0]])
    y = np.array(["low", "low", "low", "mid", "mid", "high", "high"])
    tree = DecisionTreeClassifier().fit(X, y)

    assert tree.classes_.tolist() == ["high", "low", "mid"]
    assert tree.predict([[0.5], [10.5], [25.0]]).tolist() == [
        "low", "mid", "high"
    ]


def test_apply_returns_leaves(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    leaves = tree.apply(X)
    assert np.all(tree.feature_[leaves] == -1)


def test_input_is_not_modified(blobs):
    X, y = blobs
    X_copy, y_copy = X.copy(), y.copy()
    DecisionTreeClassifier(max_depth=3).fit(X, y).predict(X)
    assert np.array_equal(X, X_copy) and np.array_equal(y, y_copy)


def test_non_contiguous_input(blobs):
    X, y = blobs
    view = np.asfortranarray(X)[:, ::2]
    tree = DecisionTreeClassifier(max_depth=3).fit(view, y)
    assert tree.score(view, y) == tree.score(view.copy(), y)


# =========================================================
# Edge cases: degenerate data
# =========================================================

def test_single_class_gives_single_leaf():
    tree = DecisionTreeClassifier().fit(np.random.rand(20, 3), np.ones(20))
    assert tree.node_count_ == 1
    assert np.all(tree.predict(np.random.rand(5, 3)) == 1)
    assert np.array_equal(tree.predict_proba([[0.1, 0.2, 0.3]]), [[1.0]])


def test_constant_features_cannot_be_split():
    X = np.ones((10, 3))
    y = np.array([0, 1] * 5)
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.node_count_ == 1


def test_duplicate_rows_with_conflicting_labels():
    X = np.array([[1.0], [1.0], [1.0], [2.0]])
    y = np.array([0, 1, 1, 0])
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.predict([[1.0]])[0] == 1          # majority of the tie group
    assert tree.predict([[2.0]])[0] == 0


def test_class_tie_in_leaf_goes_to_smallest_label():
    tree = DecisionTreeClassifier(max_depth=0).fit([[0.0], [1.0]], [5, 3])
    assert tree.predict([[0.5]])[0] == 3


def test_single_sample():
    tree = DecisionTreeClassifier().fit([[3.0, 4.0]], [7])
    assert tree.predict([[0.0, 0.0]])[0] == 7


def test_adjacent_floats_threshold_separates_them():
    low = 1.0
    high = np.nextafter(low, 2.0)                 # no float strictly between
    tree = DecisionTreeClassifier().fit([[low], [high]], [0, 1])
    assert np.array_equal(tree.predict([[low], [high]]), [0, 1])


def test_huge_values_do_not_overflow_threshold():
    X = np.array([[-1e308], [1e308]])
    tree = DecisionTreeClassifier().fit(X, [0, 1])
    assert np.isfinite(tree.threshold_[0])
    assert np.array_equal(tree.predict(X), [0, 1])


def test_deep_tree_does_not_hit_recursion_limit():
    X = np.arange(1500.0).reshape(-1, 1)
    y = np.arange(1500) % 2                       # needs ~1500 leaves
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.score(X, y) == 1.0


# =========================================================
# max_features and reproducibility
# =========================================================

@pytest.mark.parametrize("max_features", [1, 2, 0.5, "sqrt", "log2", None])
def test_max_features_options_train(blobs, max_features):
    X, y = blobs
    tree = DecisionTreeClassifier(max_features=max_features, random_state=0)
    assert tree.fit(X, y).score(X, y) > 0.9


def test_max_features_larger_than_n_features_raises(blobs):
    X, y = blobs
    with pytest.raises(ValueError, match="exceeds"):
        DecisionTreeClassifier(max_features=10).fit(X, y)


def test_random_state_makes_trees_reproducible(blobs):
    X, y = blobs
    a = DecisionTreeClassifier(max_features=1, random_state=3).fit(X, y)
    b = DecisionTreeClassifier(max_features=1, random_state=3).fit(X, y)
    assert np.array_equal(a.feature_, b.feature_)
    assert np.array_equal(a.threshold_, b.threshold_, equal_nan=True)


def test_max_features_one_feature_dataset():
    X = np.arange(6.0).reshape(-1, 1)
    tree = DecisionTreeClassifier(max_features="log2").fit(X, [0, 0, 0, 1, 1, 1])
    assert tree.score(X, [0, 0, 0, 1, 1, 1]) == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"criterion": "mse"},
        {"max_depth": -1},
        {"min_samples_split": 1},
        {"buffer_size": 0},
        {"max_features": 0},
        {"max_features": 1.5},
        {"max_features": 0.0},
        {"max_features": "all"},
        {"max_features": True},
    ],
)
def test_invalid_hyperparameters(kwargs):
    with pytest.raises((ValueError, TypeError)):
        DecisionTreeClassifier(**kwargs)


# =========================================================
# Input validation
# =========================================================

def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError, match="no trained tree"):
        DecisionTreeClassifier().predict([[1.0]])


@pytest.mark.parametrize(
    "X, y, error",
    [
        ([[1.0], [np.nan]], [0, 1], ValueError),
        ([[1.0], [np.inf]], [0, 1], ValueError),
        ([1.0, 2.0], [0, 1], ValueError),               # 1D X
        ([[1.0], [2.0]], [0], ValueError),              # length mismatch
        ([[1.0], [2.0]], [[0], [1]], ValueError),       # 2D y
        ([[1.0], [2.0]], [0.0, np.nan], ValueError),    # NaN label
        (np.empty((0, 2)), [], ValueError),             # empty fit
        ([["a"], ["b"]], [0, 1], TypeError),
    ],
)
def test_fit_rejects_invalid_input(X, y, error):
    with pytest.raises(error):
        DecisionTreeClassifier().fit(X, y)


def test_predict_rejects_wrong_feature_count(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        tree.predict(X[:, :3])


def test_predict_on_empty_input(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert tree.predict(np.empty((0, 4))).shape == (0,)


# =========================================================
# Streaming: buffered retraining
# =========================================================

def test_no_tree_until_buffer_is_full(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit(X[:60], y[:60])

    assert tree.n_buffered_ == 60 and tree.n_retrains_ == 0
    with pytest.raises(NotFittedError, match="60"):
        tree.predict(X)


def test_retrains_when_buffer_reaches_size_and_clears_it(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit(X[:60], y[:60])
    tree.partial_fit(X[60:100], y[60:100])        # exactly 100

    assert tree.n_retrains_ == 1
    assert tree.n_buffered_ == 0
    assert tree.predict(X).shape == (600,)


def test_retrains_on_buffered_data_only():
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    assert tree.predict([[0.5]])[0] == 0

    # The concept flips; the next retrain must not remember old data.
    tree.partial_fit([[0.0], [1.0], [2.0], [3.0]], [1, 1, 0, 0])
    assert tree.predict([[0.5]])[0] == 1


def test_predict_uses_last_tree_between_retrains(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=200)
    tree.partial_fit(X[:200], y[:200])
    before = tree.predict(X)

    tree.partial_fit(X[200:350], 1 - y[200:350])  # buffered, no retrain
    assert tree.n_retrains_ == 1
    assert np.array_equal(tree.predict(X), before)


def test_chunk_larger_than_buffer_triggers_one_retrain(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=50)
    tree.partial_fit(X[:300], y[:300])
    assert tree.n_retrains_ == 1
    assert tree.n_buffered_ == 0
    assert tree.value_[0].sum() == 300            # trained on the whole chunk


def test_buffer_size_one_retrains_every_chunk(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=1)
    for i in range(5):
        tree.partial_fit(X[i:i + 1], y[i:i + 1])
    assert tree.n_retrains_ == 5


def test_streamed_tree_matches_batch_tree_on_same_buffer(blobs):
    X, y = blobs
    streamed = DecisionTreeClassifier(max_depth=4, buffer_size=300)
    for chunk in np.array_split(np.arange(300), 7):
        streamed.partial_fit(X[chunk], y[chunk])
    batch = DecisionTreeClassifier(max_depth=4).fit(X[:300], y[:300])

    assert np.array_equal(streamed.feature_, batch.feature_)
    assert np.array_equal(streamed.predict(X), batch.predict(X))


def test_classes_stay_stable_when_buffer_lacks_a_class():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0.0], [1.0]], [0, 1])
    tree.partial_fit([[5.0], [6.0]], [1, 1])      # buffer has only class 1

    assert tree.classes_.tolist() == [0, 1]
    proba = tree.predict_proba([[5.5]])
    assert proba.shape == (1, 2)
    assert np.array_equal(proba, [[0.0, 1.0]])


def test_new_class_appears_mid_stream():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0.0], [1.0]], [0, 1])
    tree.partial_fit([[2.0], [3.0]], [2, 2])
    assert tree.classes_.tolist() == [0, 1, 2]
    assert tree.predict([[2.5]])[0] == 2


def test_empty_chunk_is_a_no_op(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=10)
    tree.partial_fit(np.empty((0, 4)), np.array([]))
    assert tree.n_buffered_ == 0
    tree.partial_fit(X[:10], y[:10])
    assert tree.n_retrains_ == 1


def test_partial_fit_rejects_feature_change_before_first_retrain(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=1000)
    tree.partial_fit(X[:10], y[:10])
    with pytest.raises(ValueError, match="features"):
        tree.partial_fit(X[:10, :3], y[:10])


def test_partial_fit_rejects_feature_change_after_retrain(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=10)
    tree.partial_fit(X[:10], y[:10])
    with pytest.raises(ValueError, match="features"):
        tree.partial_fit(X[:5, :2], y[:5])


def test_partial_fit_rejects_mismatched_lengths(blobs):
    X, y = blobs
    with pytest.raises(ValueError, match="samples"):
        DecisionTreeClassifier().partial_fit(X[:10], y[:9])


def test_fit_clears_streaming_buffer(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit(X[:80], 1 - y[:80])          # buffered
    tree.fit(X[:100], y[:100])
    assert tree.n_buffered_ == 0 and tree.n_retrains_ == 0

    tree.partial_fit(X[100:200], y[100:200])      # retrain on these only
    assert tree.value_[0].sum() == 100


def test_last_fit_time_is_recorded(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.last_fit_time_ > 0
