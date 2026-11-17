"""Tests for the vectorised tie-group assignment in rankdata()."""

import numpy as np
import pytest

from numcompute.rank import rankdata


def _reference_rank(values, method, ascending):
    """Straightforward O(n^2) reference implementation."""
    values = np.asarray(values, dtype=float)
    keys = values if ascending else -values
    distinct = np.unique(keys)
    ranks = np.empty(values.size)

    for i in range(values.size):
        less = np.sum(keys < keys[i])
        equal = np.sum(keys == keys[i])

        if method == "average":
            ranks[i] = less + (equal + 1) / 2
        elif method == "min":
            ranks[i] = less + 1
        elif method == "max":
            ranks[i] = less + equal
        elif method == "dense":
            ranks[i] = np.searchsorted(distinct, keys[i]) + 1
        else:  # ordinal: ties ordered by original position
            ranks[i] = less + np.sum(keys[:i] == keys[i]) + 1

    return ranks


@pytest.mark.parametrize("method", ["average", "dense", "ordinal", "min", "max"])
@pytest.mark.parametrize("ascending", [True, False])
def test_rankdata_matches_reference_with_many_ties(method, ascending):
    rng = np.random.default_rng(11)
    for _ in range(50):
        values = rng.integers(0, 4, int(rng.integers(1, 30)))
        assert np.array_equal(
            rankdata(values, method=method, ascending=ascending),
            _reference_rank(values, method, ascending),
        )


def test_rankdata_all_tied():
    assert np.array_equal(rankdata([7, 7, 7]), [2.0, 2.0, 2.0])
    assert np.array_equal(rankdata([7, 7, 7], method="ordinal"), [1, 2, 3])
    assert np.array_equal(
        rankdata([7, 7, 7], method="ordinal", ascending=False), [1, 2, 3]
    )


def test_rankdata_empty_input():
    assert rankdata([]).shape == (0,)


def test_rankdata_rejects_non_boolean_ascending():
    with pytest.raises(TypeError):
        rankdata([1, 2], ascending=1)


def test_rankdata_has_no_python_loop_over_groups():
    # 100k distinct values -> 100k groups; a per-group Python loop would
    # take seconds. The vectorised version finishes far below that.
    import time

    values = np.random.default_rng(0).permutation(100_000)
    start = time.perf_counter()
    ranks = rankdata(values, method="dense")
    elapsed = time.perf_counter() - start

    assert np.array_equal(ranks, values + 1)
    assert elapsed < 1.0
