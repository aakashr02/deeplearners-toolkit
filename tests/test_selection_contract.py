"""Contract tests for top_k() and the explicit quickselect."""

import numpy as np
import pytest

from numcompute.sort_search import quick_select, top_k


def _stable_rank_order(values, largest):
    """Reference ordering: by value, ties broken by the lower index."""
    if not largest:
        return np.argsort(values, kind="stable")
    reversed_order = np.argsort(values[::-1], kind="stable")
    return (values.size - 1 - reversed_order)[::-1]


@pytest.mark.parametrize("largest", [True, False])
def test_selection_matches_stable_sort_reference(largest):
    rng = np.random.default_rng(7)
    for trial in range(200):
        n = int(rng.integers(1, 25))
        values = rng.integers(-3, 4, n) if trial % 2 else rng.normal(size=n)
        reference = _stable_rank_order(values, largest)

        for k in range(1, n + 1):
            assert np.array_equal(top_k(values, k, largest), reference[:k])
            assert quick_select(values, k - 1, largest) == reference[k - 1]


def test_top_k_returns_best_first_order():
    values = np.array([10, 4, 7, 2, 9])
    assert np.array_equal(top_k(values, 3, return_indices=False), [10, 9, 7])
    assert np.array_equal(
        top_k(values, 3, largest=False, return_indices=False), [2, 4, 7]
    )


def test_selection_ties_break_by_lowest_index():
    values = np.array([5, 1, 5, 5, 0])
    assert np.array_equal(top_k(values, 2), [0, 2])
    assert quick_select(values, 0) == 0
    assert quick_select(values, 2) == 3


def test_quick_select_agrees_with_top_k_last_element():
    values = np.array([3, 3, 1, 2, 3, 0])
    for k in range(values.size):
        for largest in (True, False):
            assert quick_select(values, k, largest) == top_k(
                values, k + 1, largest
            )[-1]


def test_quick_select_returns_python_int_index():
    assert type(quick_select(np.array([3, 1, 2]), 0)) is int


def test_top_k_value_dtype_matches_input():
    values = np.array([3, 1, 2], dtype=np.int32)
    assert top_k(values, 2, return_indices=False).dtype == np.int32


def test_quick_select_does_not_delegate_to_numpy_partition(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("np.partition/argpartition must not be used")

    monkeypatch.setattr(np, "partition", forbidden)
    monkeypatch.setattr(np, "argpartition", forbidden)

    values = np.arange(100)[::-1]
    assert quick_select(values, 10, return_index=False) == 89
    assert quick_select(values, 10, largest=False) == 89  # index of 10


def test_top_k_uses_argpartition_as_specified(monkeypatch):
    calls = []
    original = np.argpartition

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "argpartition", spy)
    top_k(np.array([4, 1, 3]), 2)
    assert calls


def test_topk_alias():
    from numcompute.sort_search import topk
    assert topk is top_k


def test_selection_on_non_contiguous_view():
    base = np.array([9, 0, 1, 0, 7, 0, 3, 0])
    view = base[::2]                       # [9, 1, 7, 3], strided
    assert not view.flags["C_CONTIGUOUS"]
    assert np.array_equal(top_k(view, 2, return_indices=False), [9, 7])
    assert quick_select(view, 1, largest=False, return_index=False) == 3


def test_selection_handles_infinity():
    values = np.array([1.0, np.inf, -np.inf, 3.0])
    assert quick_select(values, 0, return_index=False) == np.inf
    assert quick_select(
        values, 0, largest=False, return_index=False
    ) == -np.inf


def test_selection_extreme_integers_do_not_overflow():
    info = np.iinfo(np.int64)
    values = np.array([info.min, 0, info.max])
    assert np.array_equal(
        top_k(values, 3, return_indices=False), [info.max, 0, info.min]
    )


@pytest.mark.parametrize("func, k", [(top_k, 1), (quick_select, 0)])
def test_selection_rejects_nan(func, k):
    with pytest.raises(ValueError, match="NaN"):
        func(np.array([1.0, np.nan, 3.0]), k)


@pytest.mark.parametrize("func, k", [(top_k, True), (quick_select, False)])
def test_selection_rejects_boolean_k(func, k):
    with pytest.raises(TypeError):
        func(np.array([1, 2, 3]), k)


@pytest.mark.parametrize("func, k", [(top_k, 1), (quick_select, 0)])
def test_selection_rejects_list_input(func, k):
    with pytest.raises(TypeError):
        func([1, 2, 3], k)


def test_quick_select_large_input_matches_sort():
    values = np.random.default_rng(1).normal(size=10_000)
    assert quick_select(
        values, 123, largest=False, return_index=False
    ) == np.sort(values)[123]
