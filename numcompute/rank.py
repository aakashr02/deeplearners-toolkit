
"""
rank.py
=======

Ranking and percentile utilities for the DeepLearners Toolkit.

Dependencies:
    Python standard library
    NumPy

Public functions:
    rankdata()
    percentile_rank()

Supported ranking methods:
    average  - Tied values receive the average rank.
    dense    - Tied values share a rank; no gaps between groups.
    ordinal  - Every observation receives a unique rank.
    min      - Tied values receive the smallest rank in their group.
    max      - Tied values receive the largest rank in their group.

Conventions:
    ascending=True  -> smallest value gets rank 1.
    ascending=False -> largest value gets rank 1.
    Ranks are mapped back to the original input order.

NaN policies:
    raise     -> reject NaN values.
    omit      -> exclude NaNs and leave NaN at their original positions.
    propagate -> if a ranking slice contains NaN, its ranks are all NaN.

This module does not use pandas, SciPy or scikit-learn.
"""

import numpy as np

from numcompute.utils import as_numeric_array, validate_bool


# ============================================================
# 1. VALIDATION HELPERS
# ============================================================

_RANK_METHODS = {"average", "dense", "ordinal", "min", "max"}
_NAN_POLICIES = {"raise", "omit", "propagate"}
_PERCENTILE_METHODS = {"rank", "weak", "strict", "mean"}


def _as_real_array(values, name="values"):
    """
    Convert input to a NumPy array and validate its numeric type.

    Boolean, string, complex and nonnumeric arrays are rejected
    (TypeError). Integer inputs remain integers to preserve sorting
    precision. Delegates to the shared ``utils.as_numeric_array``.
    """
    return as_numeric_array(values, name=name, allow_complex=False)


def _validate_method(method):
    """Reject unsupported ranking method names."""
    if method not in _RANK_METHODS:
        choices = ", ".join(sorted(_RANK_METHODS))
        raise ValueError(
            f"Invalid method {method!r}. Choose from: {choices}"
        )


def _validate_nan_policy(nan_policy):
    """Validate the policy used when input contains NaNs."""
    if nan_policy not in _NAN_POLICIES:
        choices = ", ".join(sorted(_NAN_POLICIES))
        raise ValueError(
            f"Invalid nan_policy {nan_policy!r}. "
            f"Choose from: {choices}"
        )


def _validate_axis(axis, ndim):
    """Validate an axis and convert negative axes to positive ones."""
    if isinstance(axis, (bool, np.bool_)) or not isinstance(
        axis, (int, np.integer)
    ):
        raise TypeError("axis must be an integer or None")

    if ndim == 0:
        raise ValueError("axis cannot be specified for scalar input")

    axis = int(axis)

    if not -ndim <= axis < ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for {ndim} dimensions"
        )

    return axis % ndim


# ============================================================
# 2. CORE RANKING ALGORITHM
# ============================================================

def _rank_1d_valid(values, method, ascending):
    """
    Rank a one-dimensional array that contains no NaNs.

    This internal function assumes that method and ascending
    have already been validated.

    Fully vectorised: one stable sort, then every tie group's rank is
    computed with array arithmetic (no Python loop over groups).

    Time complexity: O(n log n) for the sort, O(n) for the rest.
    Space complexity: O(n).
    """
    n = len(values)
    ranks = np.empty(n, dtype=float)

    if n == 0:
        return ranks

    # Stable ascending order: ties keep their original relative order.
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]

    # Mark the first element of every distinct-value group, then label
    # each sorted position with its 0-based group id.
    # Example: [10, 10, 20, 30, 30] -> group_id [0, 0, 1, 2, 2].
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    np.not_equal(sorted_values[1:], sorted_values[:-1], out=is_start[1:])
    group_id = np.cumsum(is_start) - 1

    starts = np.flatnonzero(is_start)      # 0-based first position
    ends = np.r_[starts[1:], n]            # exclusive end position
    n_groups = starts.size

    if ascending:
        # Group g occupies 1-based positions starts[g]+1 .. ends[g].
        first = starts + 1
        last = ends
        dense = np.arange(1, n_groups + 1)
    else:
        # Reversing whole groups (not elements) keeps ties in original
        # order. Group g then occupies n-ends[g]+1 .. n-starts[g].
        first = n - ends + 1
        last = n - starts
        dense = n_groups - np.arange(n_groups)

    if method == "ordinal":
        # Offset of each element inside its tie group, in original order.
        offset = np.arange(n) - starts[group_id]
        per_position = first[group_id] + offset
    elif method == "average":
        per_position = ((first + last) / 2.0)[group_id]
    elif method == "dense":
        per_position = dense[group_id]
    elif method == "min":
        per_position = first[group_id]
    else:  # method == "max"
        per_position = last[group_id]

    # Map ranks from sorted positions back to the original positions.
    ranks[order] = per_position
    return ranks


def _rank_1d(values, method, ascending, nan_policy):
    """
    Apply NaN handling before ranking a one-dimensional array.

    Kept separate from rankdata() so each axis can use the
    same implementation.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=float)

    if n == 0:
        return result

    nan_mask = np.isnan(values)

    if not np.any(nan_mask):
        return _rank_1d_valid(values, method, ascending)

    if nan_policy == "raise":
        raise ValueError(
            "Input contains NaN values. Use nan_policy='omit' "
            "or nan_policy='propagate' to choose another policy."
        )

    if nan_policy == "propagate":
        return result

    # Omit NaNs from ranking while retaining their original positions.
    valid_values = values[~nan_mask]
    result[~nan_mask] = _rank_1d_valid(
        valid_values, method, ascending
    )

    return result


# ============================================================
# 3. PUBLIC RANKING FUNCTION
# ============================================================

def rankdata(
    values,
    method="average",
    ascending=True,
    axis=None,
    nan_policy="raise",
):
    """
    Calculate ranks while preserving the original input order.

    Parameters
    ----------
    values : array-like
        Real numeric values to rank.

    method : str, default="average"
        "average", "dense", "ordinal", "min", or "max".

    ascending : bool, default=True
        True means the smallest value gets rank 1.
        False means the largest value gets rank 1.

    axis : int or None, default=None
        None flattens the array before ranking.
        An integer ranks independently along that axis.

    nan_policy : str, default="raise"
        "raise": reject NaNs.
        "omit": leave NaN ranks at missing positions.
        "propagate": return all NaN ranks for an affected slice.

    Returns
    -------
    np.ndarray
        Floating-point ranks in the original input shape when
        axis is specified, or a flattened array when axis=None.

    Raises
    ------
    TypeError
        If values are not real numeric, ascending is not Boolean, or
        axis is not an integer.
    ValueError
        If method or nan_policy is unknown, axis is out of bounds, or
        values contain NaN under nan_policy="raise".

    Notes
    -----
    Infinities are permitted and ranked numerically.
    Ordinal ties preserve original observation order.

    Time complexity: O(n log n) for n values (one stable sort, then
    vectorised tie-group assignment). With an axis, O(N log k) for N
    total elements and ranked-axis length k.
    Space complexity: O(n).

    Examples
    --------
    >>> rankdata([40, 10, 30, 10, 50])
    array([4. , 1.5, 3. , 1.5, 5. ])

    >>> rankdata([40, 10, 30, 10, 50], method="dense")
    array([3., 1., 2., 1., 4.])

    >>> rankdata([10, 20, 20, 40], ascending=False)
    array([4. , 2.5, 2.5, 1. ])

    >>> rankdata([[30, 10], [20, 40]], axis=1)
    array([[2., 1.], [1., 2.]])
    """
    arr = _as_real_array(values)
    _validate_method(method)
    _validate_nan_policy(nan_policy)

    ascending = validate_bool(ascending, "ascending")

    # Convert only floating-point arrays to float. Keep integers
    # in their original dtype so large integer values remain distinct.
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(float, copy=False)

    if axis is None:
        return _rank_1d(
            arr.ravel(), method, bool(ascending), nan_policy
        )

    axis = _validate_axis(axis, arr.ndim)

    # Move the selected axis to the end so each slice is handled
    # by the same one-dimensional ranking implementation.
    moved = np.moveaxis(arr, axis, -1)
    result = np.empty(moved.shape, dtype=float)

    for index in np.ndindex(moved.shape[:-1]):
        result[index] = _rank_1d(
            moved[index], method, bool(ascending), nan_policy
        )

    # Restore the original axis order and shape.
    return np.moveaxis(result, -1, axis)


# ============================================================
# 4. PERCENTILE-RANK FUNCTION
# ============================================================

def percentile_rank(
    values,
    score,
    kind="rank",
    nan_policy="raise",
):
    """
    Calculate the percentile rank of one or more scores.

    The percentile rank describes relative standing. It is not
    the same as calculating a percentile threshold.

    Parameters
    ----------
    values : array-like
        One-dimensional reference dataset.

    score : scalar or array-like
        Score or scores to evaluate.

    kind : str, default="rank"
        "weak":
            100 * count(values <= score) / n

        "strict":
            100 * count(values < score) / n

        "mean":
            Average of the weak and strict percentages.

        "rank":
            Average percentage rank of matching observations.
            If a score is absent, use its strict percentile rank.

    nan_policy : str, default="raise"
        "raise": reject NaNs in the reference data.
        "omit": exclude NaNs from the reference data.
        "propagate": return NaN results if reference data contain NaN.

    Returns
    -------
    float or np.ndarray
        A float for scalar input, otherwise an array matching
        the shape of score.

    Raises
    ------
    ValueError
        If the reference dataset is empty, contains NaNs under
        the "raise" policy, or becomes empty after omission.

    Notes
    -----
    The reference data is sorted once and each score is located with
    binary search.

    Time complexity: O(n log n + q log n) for n values and q scores.
    Space complexity: O(n + q).
    """
    arr = _as_real_array(values, name="values")
    scores = _as_real_array(score, name="score")

    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")

    if kind not in _PERCENTILE_METHODS:
        choices = ", ".join(sorted(_PERCENTILE_METHODS))
        raise ValueError(
            f"Invalid kind {kind!r}. Choose from: {choices}"
        )

    _validate_nan_policy(nan_policy)

    scalar_score = scores.ndim == 0
    original_score_shape = scores.shape
    scores = scores.reshape(-1)

    # Handle missing reference values according to the policy.
    reference_has_nan = np.any(np.isnan(arr))

    if reference_has_nan:
        if nan_policy == "raise":
            raise ValueError("values contains NaN values")

        if nan_policy == "propagate":
            result = np.full(scores.shape, np.nan, dtype=float)
            return float(result[0]) if scalar_score else result.reshape(
                original_score_shape
            )

        arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        raise ValueError(
            "Cannot calculate percentile rank of an empty dataset"
        )

    # Sorting once is sufficient for every score.
    sorted_values = np.sort(arr, kind="stable")

    # NaN query scores produce NaN outputs.
    result = np.full(scores.shape, np.nan, dtype=float)
    valid_scores = ~np.isnan(scores)

    if np.any(valid_scores):
        query = scores[valid_scores]

        # left = number of observations strictly less than score.
        # right = number of observations less than or equal to score.
        left = np.searchsorted(sorted_values, query, side="left")
        right = np.searchsorted(sorted_values, query, side="right")

        n = sorted_values.size

        if kind == "weak":
            calculated = right / n * 100.0

        elif kind == "strict":
            calculated = left / n * 100.0

        elif kind == "mean":
            calculated = (left + right) / (2.0 * n) * 100.0

        else:  # kind == "rank"
            # If matches exist, average their one-based positions.
            # If absent, use the strict percentile rank.
            has_match = (right > left).astype(float)
            calculated = (
                left + right + has_match
            ) / (2.0 * n) * 100.0

        result[valid_scores] = calculated

    if scalar_score:
        return float(result[0])

    return result.reshape(original_score_shape)


def rank(data, method="average", ascending=True, axis=None,
         nan_policy="raise"):
    """
    Rank data with tie handling (the specification's name for rankdata).

    Identical to ``rankdata``; see its docstring for parameters and
    return shapes.

    Raises
    ------
    TypeError, ValueError
        As for ``rankdata``.

    Notes
    -----
    Time complexity: O(n log n). Space complexity: O(n).

    Examples
    --------
    >>> rank([10, 20, 20, 30], method="dense")
    array([1., 2., 2., 3.])
    """
    return rankdata(
        data,
        method=method,
        ascending=ascending,
        axis=axis,
        nan_policy=nan_policy,
    )


# ============================================================
# 5. PERCENTILE WITH INTERPOLATION
# ============================================================

_INTERPOLATIONS = {"linear", "lower", "higher", "midpoint", "nearest"}


def percentile(data, q, interpolation="linear", axis=None,
               nan_policy="raise"):
    """
    Compute the q-th percentile(s) of the data by sorting.

    For a sorted slice ``s`` with ``n`` valid values, the fractional
    position of percentile ``q`` is ``pos = q / 100 * (n - 1)``, with
    ``i = floor(pos)``, ``j = ceil(pos)`` and ``frac = pos - i``:

    * ``"linear"``   -> ``s[i] + frac * (s[j] - s[i])``
    * ``"lower"``    -> ``s[i]``
    * ``"higher"``   -> ``s[j]``
    * ``"midpoint"`` -> ``(s[i] + s[j]) / 2``
    * ``"nearest"``  -> ``s[round(pos)]`` (ties round to even index)

    These match NumPy's ``np.percentile(..., method=...)`` definitions.

    Parameters
    ----------
    data : array-like
        Real numeric data of any shape.
    q : float or array-like of float
        Percentile(s) in the closed range [0, 100].
    interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}
        How to choose a value when ``pos`` falls between two data points.
    axis : int or None, default=None
        Axis along which to compute. None flattens the data first.
    nan_policy : {"raise", "omit", "propagate"}, default="raise"
        "raise" rejects NaN; "omit" ignores NaN within each slice (a slice
        with no valid values gives NaN); "propagate" gives NaN for any
        slice that contains NaN.

    Returns
    -------
    float or np.ndarray
        For scalar ``q``: a float when ``axis`` is None, otherwise an
        array of ``data.shape`` with ``axis`` removed. For array ``q``:
        an array of shape ``q.shape + reduced_shape``.

    Raises
    ------
    TypeError
        If data or q is not real numeric data, or axis is not an integer.
    ValueError
        If data is empty, q is outside [0, 100] or NaN, the interpolation
        or nan_policy is unknown, data contains NaN under "raise", or axis
        is out of bounds.

    Notes
    -----
    Fully vectorised: one sort along the axis (NaN sorts last), then all
    percentiles of all slices are gathered with ``take_along_axis``, even
    when slices have different numbers of valid values.

    Time complexity: O(N log k + |q| * N / k), where N is the total size
    and k is the length of the reduced axis.
    Space complexity: O(N) for the sorted copy.

    Examples
    --------
    >>> percentile([1, 2, 3, 4], 50)
    2.5
    >>> percentile([1, 2, 3, 4], 50, interpolation="lower")
    2.0
    >>> percentile([[1, 2], [3, 4]], [0, 100], axis=0)
    array([[1., 2.],
           [3., 4.]])
    """
    arr = _as_real_array(data, name="data").astype(float, copy=False)
    q_arr = _as_real_array(q, name="q").astype(float, copy=False)

    if interpolation not in _INTERPOLATIONS:
        choices = ", ".join(sorted(_INTERPOLATIONS))
        raise ValueError(
            f"Invalid interpolation {interpolation!r}. Choose from: {choices}"
        )

    _validate_nan_policy(nan_policy)

    if arr.size == 0:
        raise ValueError("Cannot compute a percentile of an empty array")

    if np.any(np.isnan(q_arr)) or np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100 inclusive")

    nan_mask = np.isnan(arr)
    if nan_policy == "raise" and np.any(nan_mask):
        raise ValueError(
            "data contains NaN values. Use nan_policy='omit' or "
            "nan_policy='propagate' to choose another policy."
        )

    if axis is None:
        arr, nan_mask = arr.ravel(), nan_mask.ravel()
        axis = 0
    else:
        axis = _validate_axis(axis, arr.ndim)

    # Work on the last axis; NaN is sorted to the end of every slice.
    sorted_arr = np.sort(np.moveaxis(arr, axis, -1), axis=-1)
    slice_has_nan = np.any(np.moveaxis(nan_mask, axis, -1), axis=-1)
    n_valid = np.sum(~np.isnan(sorted_arr), axis=-1)          # (...,)

    # Broadcast: q over leading dims, slices over trailing dims.
    q_shape = q_arr.shape
    q_b = q_arr.reshape(q_shape + (1,) * n_valid.ndim)

    position = q_b / 100.0 * np.maximum(n_valid - 1, 0)       # q x (...)
    lower_index = np.floor(position).astype(np.intp)
    upper_index = np.ceil(position).astype(np.intp)
    fraction = position - lower_index

    target_shape = position.shape
    expanded = np.broadcast_to(sorted_arr, q_shape + sorted_arr.shape)

    def gather(index):
        return np.take_along_axis(expanded, index[..., None], axis=-1)[..., 0]

    if interpolation == "lower":
        result = gather(lower_index)
    elif interpolation == "higher":
        result = gather(upper_index)
    elif interpolation == "nearest":
        result = gather(np.around(position).astype(np.intp))
    else:
        low, high = gather(lower_index), gather(upper_index)
        if interpolation == "midpoint":
            result = (low + high) / 2.0
        else:  # linear; written to stay exact when low == high == inf
            with np.errstate(invalid="ignore"):
                result = np.where(
                    (fraction == 0) | (low == high),
                    low,
                    low + fraction * (high - low),
                )

    result = np.array(np.broadcast_to(result, target_shape), dtype=float)

    invalid = n_valid == 0
    if nan_policy == "propagate":
        invalid = invalid | slice_has_nan
    result[..., invalid] = np.nan

    return float(result) if result.ndim == 0 else result


# ============================================================
# 6. PUBLIC API
# ============================================================

__all__ = ["rankdata", "rank", "percentile_rank", "percentile"]
