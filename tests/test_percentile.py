"""Tests for rank.percentile (interpolation API) and the rank() alias."""

import warnings

import numpy as np
import pytest

from numcompute.rank import percentile, rank, rankdata

METHODS = ["linear", "lower", "higher", "midpoint", "nearest"]


@pytest.mark.parametrize("method", METHODS)
def test_percentile_matches_numpy_random(method):
    rng = np.random.default_rng(0)
    for trial in range(60):
        shape = tuple(rng.integers(1, 6, rng.integers(1, 4)))
        data = rng.integers(0, 5, shape).astype(float)
        axis = None if trial % 2 else int(rng.integers(len(shape)))
        q = rng.uniform(0, 100, 3)

        assert np.allclose(
            percentile(data, q, method, axis=axis),
            np.percentile(data, q, axis=axis, method=method),
        )


@pytest.mark.parametrize(
    "method, expected",
    [("linear", 2.5), ("lower", 2.0), ("higher", 3.0), ("midpoint", 2.5)],
)
def test_percentile_interpolation_known_values(method, expected):
    assert percentile([4, 1, 3, 2], 50, interpolation=method) == expected


def test_percentile_extreme_q():
    data = [5, 1, 9, 3]
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 9.0


def test_percentile_single_value_and_all_equal():
    assert percentile([7], 30) == 7.0
    for method in METHODS:
        assert percentile([2, 2, 2], 40, interpolation=method) == 2.0


def test_percentile_scalar_returns_float():
    assert type(percentile([1, 2, 3], 50)) is float


def test_percentile_output_shapes():
    data = np.arange(24.0).reshape(2, 3, 4)
    assert percentile(data, 50, axis=1).shape == (2, 4)
    assert percentile(data, [10, 90], axis=1).shape == (2, 2, 4)
    assert percentile(data, [10, 90]).shape == (2,)


def test_percentile_negative_axis():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(percentile(data, 50, axis=-1), [2.0, 5.0])


def test_percentile_nan_policies():
    data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])

    with pytest.raises(ValueError, match="NaN"):
        percentile(data, 50, axis=1)

    assert np.allclose(
        percentile(data, 50, axis=1, nan_policy="omit"), [2.0, 5.0]
    )
    propagated = percentile(data, 50, axis=1, nan_policy="propagate")
    assert np.isnan(propagated[0]) and propagated[1] == 5.0


def test_percentile_omit_matches_nanpercentile():
    data = np.array([[1.0, np.nan, 3.0, 8.0], [np.nan, 5.0, 6.0, 2.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in METHODS:
            assert np.allclose(
                percentile(data, [25, 60], method, axis=1, nan_policy="omit"),
                np.nanpercentile(data, [25, 60], axis=1, method=method),
            )


def test_percentile_all_nan_slice_omit_gives_nan():
    result = percentile(np.array([[np.nan, np.nan], [1.0, 3.0]]), 50,
                        axis=1, nan_policy="omit")
    assert np.isnan(result[0]) and result[1] == 2.0


def test_percentile_handles_infinity():
    assert percentile([np.inf, np.inf], 50) == np.inf
    assert percentile([1.0, np.inf], 0) == 1.0


def test_percentile_non_contiguous_view():
    base = np.arange(20.0).reshape(4, 5)
    view = base[:, ::2]
    assert np.allclose(percentile(view, 50, axis=0),
                       np.percentile(view, 50, axis=0))


def test_percentile_does_not_modify_input():
    data = np.array([3.0, 1.0, 2.0])
    percentile(data, 50)
    assert np.array_equal(data, [3.0, 1.0, 2.0])


@pytest.mark.parametrize("q", [-1, 101, np.nan, [10, 200]])
def test_percentile_rejects_invalid_q(q):
    with pytest.raises(ValueError):
        percentile([1, 2, 3], q)


def test_percentile_rejects_bad_arguments():
    with pytest.raises(ValueError, match="interpolation"):
        percentile([1, 2], 50, interpolation="cubic")
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(TypeError):
        percentile(["a"], 50)
    with pytest.raises(ValueError):
        percentile([[1, 2]], 50, axis=2)


def test_rank_alias_matches_rankdata():
    data = [40, 10, 30, 10, 50]
    for method in ["average", "dense", "ordinal"]:
        assert np.array_equal(rank(data, method=method),
                              rankdata(data, method=method))
