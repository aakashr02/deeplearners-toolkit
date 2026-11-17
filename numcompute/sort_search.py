"""
sort_search.py
==============

Sorting, searching and selection utilities for NumCompute.

Selection contract (``top_k`` and ``quick_select``)
--------------------------------------------------
* Input must be a one-dimensional NumPy array without NaN values.
  Positive and negative infinity are allowed.
* ``largest=True`` ranks from the largest value; ``largest=False`` from
  the smallest.
* Ties are broken deterministically by the lower original index.
* Indices are returned as integer positions into the original array;
  values are returned with the input dtype.
* ``top_k`` (alias ``topk``) returns its k results in rank order
  (best first), and ``quick_select(values, r)`` equals
  ``top_k(values, r + 1)[-1]``.
* ``top_k`` uses ``np.argpartition`` as the specification requires;
  ``quick_select`` is an explicit quickselect for educational purposes.
* The input array is never modified.
"""

import numpy as np

from numcompute.utils import validate_bool, validate_int


# ============================================================
# 1. VALIDATION HELPERS
# ============================================================

def _validate_ndarray(values, name):
    """Require a NumPy array (array-likes are deliberately rejected)."""
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")


def _contains_nan(values):
    """Return whether an array contains any NaN values."""
    if np.issubdtype(values.dtype, np.inexact):
        return bool(np.any(np.isnan(values)))
    return False


def _validate_selection_input(values):
    """Shared checks for top_k() and quick_select()."""
    _validate_ndarray(values, "values")

    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")

    if _contains_nan(values):
        raise ValueError("values must not contain NaN")


# ============================================================
# 2. SORTING AND SEARCHING
# ============================================================

def stable_sort(
    values: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Return a stable sorted copy of an array.

    Parameters
    ----------
    values : np.ndarray
        Array of any shape to sort.
    axis : int, optional
        Axis along which to sort. Defaults to -1.

    Returns
    -------
    np.ndarray
        Sorted copy with the same shape and dtype as ``values``. Equal
        elements keep their original relative order.

    Raises
    ------
    TypeError
        If values is not a NumPy array.
    ValueError
        If axis is not an integer.
    numpy.exceptions.AxisError
        If axis is out of bounds.

    Notes
    -----
    Time complexity: O(n log n) (NumPy timsort/radix sort).
    Space complexity: O(n) for the copy.
    """
    _validate_ndarray(values, "values")

    if isinstance(axis, (bool, np.bool_)) or not isinstance(
        axis, (int, np.integer)
    ):
        raise ValueError("axis must be an integer")

    return np.sort(values, axis=axis, kind="stable")


def multi_key_sort(
    data: np.ndarray,
    ascending: bool = True
) -> np.ndarray:
    """Return indices that sort the rows of a 2D array by multiple columns.

    Columns are used as sorting keys, with the first column as the
    primary key. NaNs follow NumPy's lexsort behaviour (sorted last).

    Parameters
    ----------
    data : np.ndarray of shape (n_rows, n_keys)
        Two-dimensional array to sort, with at least one column.
    ascending : bool, optional
        If True, sort in ascending order; otherwise, descending.

    Returns
    -------
    np.ndarray of shape (n_rows,)
        Integer row indices; ``data[indices]`` is sorted.

    Raises
    ------
    TypeError
        If data is not a NumPy array or ascending is not Boolean.
    ValueError
        If data is not 2D or has no columns.

    Notes
    -----
    Time complexity: O(n_keys * n_rows log n_rows).
    Space complexity: O(n_rows).
    """
    _validate_ndarray(data, "data")

    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array")

    if data.shape[1] == 0:
        raise ValueError("data must contain at least one column")

    ascending = validate_bool(ascending, "ascending")

    # lexsort uses the last key as primary, so reverse the column order.
    indices = np.lexsort(data.T[::-1])

    if not ascending:
        indices = indices[::-1]

    return indices


def binary_search(
    sorted_array: np.ndarray,
    x: object,
) -> tuple:
    """Return the insertion index and whether x exists in a sorted array.

    The input must be sorted in ascending order. Duplicate matches
    return the index of the first occurrence.

    NaN search targets are rejected. Existing NaNs in sorted_array
    follow NumPy's searchsorted behaviour.

    Parameters
    ----------
    sorted_array : np.ndarray of shape (n,)
        One-dimensional array sorted in ascending order.
    x : object
        Scalar value to search for.

    Returns
    -------
    tuple[int, bool]
        ``(index, found)`` where ``index`` is the leftmost position at
        which x could be inserted while keeping the array sorted.

    Raises
    ------
    TypeError
        If sorted_array is not a NumPy array.
    ValueError
        If sorted_array is not 1D or x is NaN.

    Notes
    -----
    Time complexity: O(log n). Space complexity: O(1).
    """
    _validate_ndarray(sorted_array, "sorted_array")

    if sorted_array.ndim != 1:
        raise ValueError("sorted_array must be a one-dimensional array")

    try:
        if np.isscalar(x) and np.issubdtype(
            np.asarray(x).dtype, np.inexact
        ) and np.isnan(x):
            raise ValueError("x must not be NaN")
    except TypeError:
        pass

    index = int(np.searchsorted(sorted_array, x, side="left"))
    exists = index < sorted_array.size and bool(sorted_array[index] == x)

    return index, exists


# ============================================================
# 3. SELECTION (QUICKSELECT)
# ============================================================

def _quickselect(values, k, largest, rng=None):
    """
    Return the original index of the element at 0-based rank ``k``.

    Ranks follow a stable ordering: by value (descending when
    ``largest`` is True, ascending otherwise), then by original index.

    Explicit iterative quickselect (Hoare's FIND) with a random pivot and
    a three-way partition into ``better``, ``equal`` and ``worse`` than
    the pivot. Each round keeps only the partition containing rank ``k``,
    so the candidate set shrinks until the answer is found. The partition
    step uses vectorised NumPy comparisons, but the selection logic
    (pivot choice, partition, narrowing to one side) is implemented here
    rather than delegated to ``np.partition``.

    Boolean masking preserves order, so the candidate index array stays
    sorted; the answer inside the ``equal`` group is therefore at offset
    ``k - n_better``, which reproduces the stable tie order.

    Expected time O(n), worst case O(n^2); auxiliary space O(n).
    """
    if rng is None:
        # Fixed seed: results and timings are reproducible run to run.
        rng = np.random.default_rng(0)

    candidates = np.arange(values.size)
    current = values

    while True:
        pivot = current[rng.integers(current.size)]

        if largest:
            better, worse = current > pivot, current < pivot
        else:
            better, worse = current < pivot, current > pivot

        n_better = int(np.count_nonzero(better))
        n_equal = current.size - n_better - int(np.count_nonzero(worse))

        if k < n_better:
            candidates, current = candidates[better], current[better]
        elif k < n_better + n_equal:
            equal = ~(better | worse)
            return int(candidates[equal][k - n_better])
        else:
            k -= n_better + n_equal
            candidates, current = candidates[worse], current[worse]


def _order_ranked(indices, selected_values, largest):
    """
    Order selected items best-first, breaking ties by lower index.

    ``indices`` must be in increasing order (as produced by masking).
    """
    if not largest:
        # Stable ascending sort keeps increasing index order among ties.
        return indices[np.argsort(selected_values, kind="stable")]

    # Descending with ascending-index ties: stably sort the reversed
    # array, then reverse the result. Avoids negating values, which
    # would overflow for the most negative integer.
    reversed_order = np.argsort(selected_values[::-1], kind="stable")
    order = (selected_values.size - 1 - reversed_order)[::-1]
    return indices[order]


def top_k(values: np.ndarray,
          k: int,
          largest: bool = True,
          return_indices: bool = True):
    """Return the k largest or smallest elements, or their indices.

    Uses ``np.argpartition`` (introselect, O(n)) to find the value at the
    k-th rank, collects every element strictly better than it plus the
    lowest-index ties needed to make k, then sorts only those k items.
    ``quick_select`` provides the same selection with an explicit,
    educational quickselect.

    Parameters
    ----------
    values : np.ndarray of shape (n,)
        One-dimensional array without NaN values.
    k : int
        Number of values to select, ``1 <= k <= n``.
    largest : bool, optional
        If True, select the largest values; otherwise, the smallest.
    return_indices : bool, optional
        If True, return indices into ``values``; otherwise, the values.

    Returns
    -------
    np.ndarray of shape (k,)
        Selected indices (integer dtype) or values (input dtype), ordered
        best-first: descending for ``largest=True``, ascending otherwise.
        Ties are broken by the lower original index.

    Raises
    ------
    TypeError
        If values is not an ndarray, k is not an integer, or a Boolean
        argument is invalid.
    ValueError
        If values is not 1D, contains NaNs, or k is out of range.

    Notes
    -----
    Time complexity: O(n + k log k).
    Space complexity: O(n).

    ``topk`` is an alias with the same signature.

    Examples
    --------
    >>> top_k(np.array([10, 4, 7, 2, 9]), k=3, return_indices=False)
    array([10,  9,  7])
    """
    _validate_selection_input(values)
    k = validate_int(k, "k")
    largest = validate_bool(largest, "largest")
    return_indices = validate_bool(return_indices, "return_indices")

    if k < 1 or k > values.size:
        raise ValueError(f"k must satisfy 1 <= k <= {values.size}")

    # Value at the boundary of the top-k set. argpartition places the
    # element of ascending rank `kth` at position `kth`.
    kth = values.size - k if largest else k - 1
    threshold = values[np.argpartition(values, kth)[kth]]

    if largest:
        strictly_better = np.flatnonzero(values > threshold)
    else:
        strictly_better = np.flatnonzero(values < threshold)

    # Fill the remaining slots with the lowest-index ties.
    ties = np.flatnonzero(values == threshold)[: k - strictly_better.size]
    selected = np.sort(np.concatenate((strictly_better, ties)))

    ordered = _order_ranked(selected, values[selected], largest)

    if return_indices:
        return ordered

    return values[ordered]


def quick_select(values: np.ndarray,
                 k: int,
                 largest: bool = True,
                 return_index: bool = True):
    """Select the element at a zero-based rank using quickselect.

    By default, k=0 selects the largest value. If largest=False,
    k=0 selects the smallest value. The array is never fully sorted.

    Parameters
    ----------
    values : np.ndarray of shape (n,)
        One-dimensional array without NaN values. Infinities are allowed.
    k : int
        Zero-based rank, ``0 <= k < n``.
    largest : bool, optional
        If True, rank from largest to smallest; otherwise, smallest
        to largest.
    return_index : bool, optional
        If True, return the selected element's original index;
        otherwise, its value.

    Returns
    -------
    int or numpy scalar
        Python ``int`` index, or the selected value with the input dtype.
        Duplicates are ranked by original index (stable order), so
        ``quick_select(values, k) == top_k(values, k + 1)[-1]``.

    Raises
    ------
    TypeError
        If values is not an ndarray, k is not an integer, or a Boolean
        argument is invalid.
    ValueError
        If values is not 1D, contains NaNs, or k is out of range.

    Notes
    -----
    Time complexity: expected O(n), worst case O(n^2) (random pivot
    with a fixed seed makes the worst case vanishingly unlikely).
    Space complexity: O(n) for the candidate index array.
    """
    _validate_selection_input(values)
    k = validate_int(k, "k")
    largest = validate_bool(largest, "largest")
    return_index = validate_bool(return_index, "return_index")

    if k < 0 or k >= values.size:
        raise ValueError(f"k must satisfy 0 <= k < {values.size}")

    index = _quickselect(values, k, largest)

    if return_index:
        return index

    return values[index]


# Name used in the assignment specification.
topk = top_k


__all__ = [
    "stable_sort",
    "multi_key_sort",
    "binary_search",
    "top_k",
    "topk",
    "quick_select",
]
