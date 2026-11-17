"""Contract tests for stats.py validation and the batched optim functions."""

import warnings

import numpy as np
import pytest

from numcompute.optim import (
    grad,
    gradient,
    gradient_batched,
    jacobian,
    jacobian_batched,
)
from numcompute.stats import (
    histogram,
    maximum,
    mean,
    median,
    minimum,
    quantile,
    std,
)

REDUCTIONS = [mean, median, std, minimum, maximum]


# ---------------------------------------------------------
# stats: shared input contract
# ---------------------------------------------------------

@pytest.mark.parametrize("func", REDUCTIONS)
def test_stats_reject_empty(func):
    with pytest.raises(ValueError, match="empty"):
        func([])


@pytest.mark.parametrize("func", REDUCTIONS)
def test_stats_reject_non_numeric(func):
    with pytest.raises(TypeError):
        func(["a", "b"])


@pytest.mark.parametrize("func", REDUCTIONS)
def test_stats_reject_out_of_bounds_axis(func):
    with pytest.raises(ValueError, match="out of bounds"):
        func(np.ones((2, 3)), axis=2)


@pytest.mark.parametrize("func", REDUCTIONS)
def test_stats_reject_non_integer_axis(func):
    with pytest.raises(TypeError):
        func(np.ones((2, 3)), axis=1.0)


@pytest.mark.parametrize("func", REDUCTIONS)
def test_stats_scalar_result_is_python_float(func):
    assert type(func([1, 2, 3])) is float


@pytest.mark.parametrize("func", REDUCTIONS)
def test_stats_negative_axis_and_output_shape(func):
    data = np.arange(24.0).reshape(2, 3, 4)
    assert func(data, axis=-1).shape == (2, 3)
    assert np.allclose(func(data, axis=-1), func(data, axis=2))


@pytest.mark.parametrize("func", REDUCTIONS)
def test_stats_all_nan_slice_gives_nan_without_warning(func):
    data = np.array([[np.nan, np.nan], [1.0, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = func(data, axis=1)
    assert np.isnan(result[0]) and np.isfinite(result[1])


@pytest.mark.parametrize("func", REDUCTIONS)
def test_stats_nan_propagates_when_not_ignored(func):
    assert np.isnan(func([1.0, np.nan, 3.0], ignore_nan=False))


@pytest.mark.parametrize("func", REDUCTIONS)
def test_stats_all_equal_values(func):
    expected = 0.0 if func is std else 4.0
    assert func([4, 4, 4, 4]) == expected


def test_stats_non_contiguous_view():
    view = np.arange(20.0).reshape(4, 5)[:, ::2]
    assert np.allclose(mean(view, axis=0), view.copy().mean(axis=0))


def test_std_ddof():
    data = [1.0, 2.0, 3.0, 4.0]
    assert np.isclose(std(data, ddof=1), np.std(data, ddof=1))
    with pytest.raises(ValueError):
        std(data, ddof=-1)


def test_histogram_ignore_nan_false_raises():
    with pytest.raises(ValueError, match="NaN"):
        histogram([1.0, np.nan, 2.0], ignore_nan=False)


def test_histogram_all_nan_raises():
    with pytest.raises(ValueError, match="no non-NaN"):
        histogram([np.nan, np.nan])


def test_histogram_flattens_2d_and_counts_all_values():
    counts, edges = histogram(np.arange(6.0).reshape(2, 3), bins=3)
    assert counts.sum() == 6 and edges.shape == (4,)


@pytest.mark.parametrize("q", [-0.1, 1.5, np.nan])
def test_quantile_rejects_invalid_q(q):
    with pytest.raises(ValueError):
        quantile([1, 2, 3], q)


def test_quantile_array_q_shape():
    data = np.arange(12.0).reshape(3, 4)
    assert quantile(data, [0.25, 0.75], axis=0).shape == (2, 4)


# ---------------------------------------------------------
# optim: batched variants share gradient()/jacobian() validation
# ---------------------------------------------------------

def _batched_sum_of_squares(X):
    return np.sum(X ** 2, axis=tuple(range(1, X.ndim)))


@pytest.mark.parametrize("method", ["central", "forward"])
def test_gradient_batched_matches_gradient(method):
    x = np.array([[1.0, -2.0], [0.5, 3.0]])
    expected = gradient(lambda v: np.sum(v ** 2), x, method=method)
    result = gradient_batched(_batched_sum_of_squares, x, method=method)
    assert result.shape == x.shape
    assert np.allclose(result, expected, atol=1e-4)


@pytest.mark.parametrize("method", ["central", "forward"])
def test_jacobian_batched_matches_jacobian(method):
    def F(v):
        return np.array([v[0] * v[1], v[0] + v[1]])

    def F_batched(X):
        return np.stack([X[:, 0] * X[:, 1], X[:, 0] + X[:, 1]], axis=1)

    x = np.array([2.0, 3.0])
    assert np.allclose(
        jacobian_batched(F_batched, x, method=method),
        jacobian(F, x, method=method),
        atol=1e-4,
    )


@pytest.mark.parametrize("func", [gradient_batched, jacobian_batched])
def test_batched_validate_like_unbatched(func):
    with pytest.raises(TypeError):
        func("not callable", [1.0])
    with pytest.raises(ValueError):
        func(_batched_sum_of_squares, [1.0], h=0)
    with pytest.raises(ValueError):
        func(_batched_sum_of_squares, [np.nan])
    with pytest.raises(ValueError):
        func(_batched_sum_of_squares, [])
    with pytest.raises(ValueError):
        func(_batched_sum_of_squares, [1.0], method="backward")


@pytest.mark.parametrize("func", [gradient_batched, jacobian_batched])
def test_batched_reject_non_finite_output(func):
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            func(lambda X: 1.0 / (X[:, 0] * 0.0), [1.0])


def test_gradient_batched_calls_f_twice():
    calls = []

    def f(X):
        calls.append(1)
        return _batched_sum_of_squares(X)

    gradient_batched(f, np.ones(50))
    assert len(calls) == 2


def test_grad_alias():
    assert grad is gradient
