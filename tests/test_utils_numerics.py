"""Tests for activations, logsumexp, distances and batching in utils."""

import warnings

import numpy as np
import pytest

from numcompute.utils import (
    batch_iterator,
    cosine_distance,
    euclidean_distance,
    leaky_relu,
    log_softmax,
    logsumexp,
    manhattan_distance,
    pairwise_distances,
    relu,
    sigmoid,
    softmax,
    tanh,
)


@pytest.fixture(autouse=True)
def _fail_on_numpy_warnings():
    # Overflow/underflow must be handled, not merely produce warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


# ---------------------------------------------------------
# Activations
# ---------------------------------------------------------

def test_sigmoid_known_values():
    assert np.allclose(sigmoid([0.0, np.log(3.0)]), [0.5, 0.75])


def test_sigmoid_extreme_inputs_do_not_overflow():
    out = sigmoid([-1000.0, 1000.0, -np.inf, np.inf])
    assert np.array_equal(out, [0.0, 1.0, 0.0, 1.0])


def test_sigmoid_symmetry_and_nan():
    x = np.linspace(-30, 30, 13)
    assert np.allclose(sigmoid(x) + sigmoid(-x), 1.0)
    assert np.isnan(sigmoid([np.nan]))[0]


def test_sigmoid_preserves_shape_and_accepts_ints():
    out = sigmoid(np.zeros((2, 3), dtype=int))
    assert out.shape == (2, 3)
    assert out.dtype == np.float64


def test_relu_and_leaky_relu():
    x = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(relu(x), [0.0, 0.0, 3.0])
    assert np.allclose(leaky_relu(x, alpha=0.1), [-0.2, 0.0, 3.0])


def test_leaky_relu_rejects_negative_alpha():
    with pytest.raises(ValueError):
        leaky_relu([1.0], alpha=-0.1)


def test_tanh_bounds():
    assert np.array_equal(tanh([-1000.0, 0.0, 1000.0]), [-1.0, 0.0, 1.0])


@pytest.mark.parametrize("func", [sigmoid, relu, tanh, softmax, logsumexp])
def test_activations_reject_non_numeric(func):
    with pytest.raises(TypeError):
        func(["a", "b"])


# ---------------------------------------------------------
# logsumexp / softmax
# ---------------------------------------------------------

def test_logsumexp_matches_naive_on_safe_input():
    x = np.array([0.5, -1.0, 2.0])
    assert np.isclose(logsumexp(x), np.log(np.sum(np.exp(x))))


def test_logsumexp_large_values_do_not_overflow():
    assert np.isclose(logsumexp([1000.0, 1000.0]), 1000.0 + np.log(2.0))
    assert np.isclose(logsumexp([-1000.0, -1000.0]), -1000.0 + np.log(2.0))


def test_logsumexp_infinities_and_nan():
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert logsumexp([np.inf, 1.0]) == np.inf
    assert np.isnan(logsumexp([np.nan, 1.0]))


def test_logsumexp_axis_and_keepdims():
    x = np.arange(6.0).reshape(2, 3)
    expected = np.log(np.exp(x).sum(axis=1))
    assert np.allclose(logsumexp(x, axis=1), expected)
    assert logsumexp(x, axis=1, keepdims=True).shape == (2, 1)
    assert isinstance(logsumexp(x), float)


def test_logsumexp_rejects_empty():
    with pytest.raises(ValueError):
        logsumexp([])


def test_softmax_rows_sum_to_one_with_huge_logits():
    logits = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]])
    probs = softmax(logits, axis=1)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[0], softmax([0.0, 1.0, 2.0]))


def test_softmax_is_shift_invariant_and_handles_neg_inf():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(softmax(x), softmax(x + 500.0))
    assert np.allclose(softmax([0.0, -np.inf]), [1.0, 0.0])


def test_softmax_all_equal_is_uniform():
    assert np.allclose(softmax(np.full(4, 7.0)), 0.25)


def test_log_softmax_is_stable():
    out = log_softmax([0.0, -2000.0])
    assert np.isfinite(out).all()
    assert np.allclose(out, [0.0, -2000.0])
    assert np.allclose(np.exp(log_softmax([1.0, 2.0])), softmax([1.0, 2.0]))


def test_softmax_on_non_contiguous_view():
    base = np.arange(12.0).reshape(3, 4)
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert np.allclose(softmax(view, axis=1), softmax(view.copy(), axis=1))


# ---------------------------------------------------------
# Distances
# ---------------------------------------------------------

def test_vector_distances_known_values():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0
    assert manhattan_distance([0, 0], [3, -4]) == 7.0
    assert np.isclose(cosine_distance([1, 0], [0, 1]), 1.0)
    assert np.isclose(cosine_distance([1, 1], [2, 2]), 0.0)
    assert np.isclose(cosine_distance([1, 0], [-1, 0]), 2.0)


def test_distance_broadcasts_vector_against_rows():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    assert np.allclose(euclidean_distance(X, [0.0, 0.0]), [0.0, 5.0, 10.0])


def test_distance_identical_points_is_zero():
    x = np.array([1.5, -2.0, 3.0])
    assert euclidean_distance(x, x) == 0.0
    assert manhattan_distance(x, x) == 0.0


def test_distance_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="last dimension"):
        euclidean_distance([1, 2], [1, 2, 3])


def test_cosine_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero"):
        cosine_distance([0, 0], [1, 1])


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_pairwise_distances_match_vector_functions(metric):
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(4, 3)), rng.normal(size=(5, 3))
    func = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "cosine": cosine_distance,
    }[metric]

    D = pairwise_distances(X, Y, metric=metric)
    expected = np.array([[func(x, y) for y in Y] for x in X])

    assert D.shape == (4, 5)
    assert np.allclose(D, expected)


def test_pairwise_euclidean_self_distance_is_exactly_non_negative():
    X = np.random.default_rng(1).normal(size=(6, 4)) * 1e8
    D = pairwise_distances(X)
    assert np.all(D >= 0)
    assert np.allclose(np.diag(D), 0.0, atol=1e-2 * 1e8)
    assert np.allclose(D, D.T)


def test_pairwise_rejects_bad_input():
    with pytest.raises(ValueError, match="metric"):
        pairwise_distances(np.ones((2, 2)), metric="chebyshev")
    with pytest.raises(ValueError, match="features"):
        pairwise_distances(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        pairwise_distances(np.ones(3))


# ---------------------------------------------------------
# Batching
# ---------------------------------------------------------

def test_batch_iterator_sizes_and_alignment():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    batches = list(batch_iterator(X, y, batch_size=2))

    assert [len(b[1]) for b in batches] == [2, 2, 1]
    for X_batch, y_batch in batches:
        assert np.array_equal(X_batch[:, 0] // 2, y_batch)


def test_batch_iterator_single_array_and_drop_last():
    batches = list(batch_iterator(np.arange(5), batch_size=2, drop_last=True))
    assert [b.tolist() for b in batches] == [[0, 1], [2, 3]]


def test_batch_iterator_shuffle_is_reproducible_permutation():
    a = np.concatenate(list(batch_iterator(np.arange(10), batch_size=3,
                                           shuffle=True, seed=5)))
    b = np.concatenate(list(batch_iterator(np.arange(10), batch_size=3,
                                           shuffle=True, seed=5)))
    assert np.array_equal(a, b)
    assert np.array_equal(np.sort(a), np.arange(10))
    assert not np.array_equal(a, np.arange(10))


def test_batch_iterator_batch_larger_than_data():
    assert [b.tolist() for b in batch_iterator([1, 2], batch_size=10)] == [[1, 2]]


def test_batch_iterator_rejects_bad_arguments():
    with pytest.raises(ValueError, match="same length"):
        list(batch_iterator(np.arange(3), np.arange(4), batch_size=2))
    with pytest.raises(ValueError):
        list(batch_iterator(np.arange(3), batch_size=0))
    with pytest.raises(TypeError):
        list(batch_iterator(np.arange(3), batch_size=1.5))
    with pytest.raises(ValueError):
        list(batch_iterator(batch_size=1))
