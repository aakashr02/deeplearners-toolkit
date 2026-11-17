"""
stats.py
========

Descriptive statistics for NumCompute.

Shared contract
---------------
* ``data`` is any real numeric array-like (converted to float64).
  Strings, Booleans and complex values raise ``TypeError``; empty input
  raises ``ValueError``.
* ``axis=None`` reduces over all elements and returns a Python float.
  An integer ``axis`` (negative allowed) returns an array of
  ``data.shape`` with that axis removed.
* ``ignore_nan=True`` (default) skips NaN values. A slice that is
  entirely NaN gives NaN without emitting a RuntimeWarning.
  ``ignore_nan=False`` lets NaN propagate to the result.
* The input is never modified.
"""

import warnings

import numpy as np

from numcompute.utils import validate_array, validate_bool, validate_int


# ============================================================
# 1. SHARED HELPERS
# ============================================================

def _prepare(data, axis, ignore_nan):
    """Validate data, axis and ignore_nan; return a float64 array."""
    arr = validate_array(data, name="data", allow_empty=False)
    arr = arr.astype(float, copy=False)
    validate_bool(ignore_nan, "ignore_nan")

    if axis is not None:
        axis = validate_int(axis, "axis")
        if arr.ndim == 0 or not -arr.ndim <= axis < arr.ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for array of "
                f"dimension {arr.ndim}"
            )

    return arr


def _reduce(nan_func, func, arr, ignore_nan, **kwargs):
    """Apply a NumPy reduction, silencing all-NaN slice warnings."""
    if ignore_nan:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = nan_func(arr, **kwargs)
    else:
        result = func(arr, **kwargs)

    return float(result) if np.ndim(result) == 0 else result


# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================

def mean(data, axis=None, ignore_nan=True):
    """
    Calculate the arithmetic mean.

    Parameters
    ----------
    data : array-like
        Real numeric input of any shape, non-empty.
    axis : int or None, optional
        Axis along which the mean is calculated.
        If None, calculate over the entire array.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.

    Returns
    -------
    float or np.ndarray
        A float when ``axis`` is None, otherwise an array of
        ``data.shape`` with ``axis`` removed.

    Raises
    ------
    TypeError
        If data is not real numeric, axis is not an integer, or
        ignore_nan is not Boolean.
    ValueError
        If data is empty or axis is out of bounds.

    Notes
    -----
    Time complexity: O(n).
    Space complexity: O(n) worst case for the float64 conversion
    (O(1) extra when the input is already float64).
    """
    arr = _prepare(data, axis, ignore_nan)
    return _reduce(np.nanmean, np.mean, arr, ignore_nan, axis=axis)


def median(data, axis=None, ignore_nan=True):
    """
    Calculate the median.

    For an even number of values, the mean of the two middle values.

    Parameters
    ----------
    data : array-like
        Real numeric input of any shape, non-empty.
    axis : int or None, optional
        Axis along which the median is calculated.
        If None, calculate over the entire array.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.

    Returns
    -------
    float or np.ndarray
        A float when ``axis`` is None, otherwise an array of
        ``data.shape`` with ``axis`` removed.

    Raises
    ------
    TypeError
        If data is not real numeric, axis is not an integer, or
        ignore_nan is not Boolean.
    ValueError
        If data is empty or axis is out of bounds.

    Notes
    -----
    Time complexity: O(n) average (selection via partitioning).
    Space complexity: O(n) for the partitioned copy.
    """
    arr = _prepare(data, axis, ignore_nan)
    return _reduce(np.nanmedian, np.median, arr, ignore_nan, axis=axis)


def std(data, axis=None, ignore_nan=True, ddof=0):
    """
    Calculate the standard deviation.

    Parameters
    ----------
    data : array-like
        Real numeric input of any shape, non-empty.
    axis : int or None, optional
        Axis along which the standard deviation is calculated.
        If None, calculate over the entire array.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.
    ddof : int, default=0
        Delta degrees of freedom: the divisor is ``N - ddof``.
        0 gives the population and 1 the sample standard deviation.
        If ``N - ddof <= 0`` the result is NaN.

    Returns
    -------
    float or np.ndarray
        A float when ``axis`` is None, otherwise an array of
        ``data.shape`` with ``axis`` removed.

    Raises
    ------
    TypeError
        If data is not real numeric, axis or ddof is not an integer,
        or ignore_nan is not Boolean.
    ValueError
        If data is empty, axis is out of bounds, or ddof is negative.

    Notes
    -----
    Uses NumPy's two-pass algorithm (mean first, then squared
    deviations), which is numerically stable.

    Time complexity: O(n). Space complexity: O(n) for temporaries.
    """
    arr = _prepare(data, axis, ignore_nan)
    ddof = validate_int(ddof, "ddof", minimum=0)
    return _reduce(np.nanstd, np.std, arr, ignore_nan, axis=axis, ddof=ddof)


def minimum(data, axis=None, ignore_nan=True):
    """
    Calculate the minimum.

    Parameters
    ----------
    data : array-like
        Real numeric input of any shape, non-empty.
    axis : int or None, optional
        Axis along which the minimum is calculated.
        If None, calculate over the entire array.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.

    Returns
    -------
    float or np.ndarray
        A float when ``axis`` is None, otherwise an array of
        ``data.shape`` with ``axis`` removed.

    Raises
    ------
    TypeError
        If data is not real numeric, axis is not an integer, or
        ignore_nan is not Boolean.
    ValueError
        If data is empty or axis is out of bounds.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(1) beyond conversion.
    """
    arr = _prepare(data, axis, ignore_nan)
    return _reduce(np.nanmin, np.min, arr, ignore_nan, axis=axis)


def maximum(data, axis=None, ignore_nan=True):
    """
    Calculate the maximum.

    Parameters
    ----------
    data : array-like
        Real numeric input of any shape, non-empty.
    axis : int or None, optional
        Axis along which the maximum is calculated.
        If None, calculate over the entire array.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.

    Returns
    -------
    float or np.ndarray
        A float when ``axis`` is None, otherwise an array of
        ``data.shape`` with ``axis`` removed.

    Raises
    ------
    TypeError
        If data is not real numeric, axis is not an integer, or
        ignore_nan is not Boolean.
    ValueError
        If data is empty or axis is out of bounds.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(1) beyond conversion.
    """
    arr = _prepare(data, axis, ignore_nan)
    return _reduce(np.nanmax, np.max, arr, ignore_nan, axis=axis)


def histogram(data, bins=10, range=None, ignore_nan=True):
    """
    Compute the histogram of all values (the input is flattened).

    Parameters
    ----------
    data : array-like
        Real numeric input of any shape, non-empty.
    bins : int or sequence of scalars, default=10
        Number of equal-width bins, or monotonically increasing bin edges.
    range : tuple (float, float) or None, optional
        Lower and upper range of the bins. None uses the data min/max.
    ignore_nan : bool, default=True
        If True, NaN values are dropped before counting. If False, any
        NaN raises ValueError, because NaN cannot be placed in a bin.

    Returns
    -------
    hist : np.ndarray of shape (n_bins,)
        Integer counts per bin. The last bin includes its right edge.
    bin_edges : np.ndarray of shape (n_bins + 1,)
        Bin edges.

    Raises
    ------
    TypeError
        If data is not real numeric or ignore_nan is not Boolean.
    ValueError
        If data is empty, contains NaN while ``ignore_nan=False``, has no
        non-NaN values, or bins/range are invalid.

    Notes
    -----
    Time complexity: O(n + b) for b bins (O(n log b) for explicit edges).
    Space complexity: O(n + b).
    """
    arr = _prepare(data, None, ignore_nan).ravel()
    nan_mask = np.isnan(arr)

    if np.any(nan_mask):
        if not ignore_nan:
            raise ValueError(
                "data contains NaN values; use ignore_nan=True to drop them"
            )
        arr = arr[~nan_mask]

    if arr.size == 0:
        raise ValueError("data contains no non-NaN values")

    return np.histogram(arr, bins=bins, range=range)


def quantile(data, q, axis=None, ignore_nan=True):
    """
    Calculate quantile(s) with linear interpolation.

    For percentiles on the 0-100 scale and other interpolation methods,
    see ``numcompute.rank.percentile``.

    Parameters
    ----------
    data : array-like
        Real numeric input of any shape, non-empty.
    q : float or array-like
        Quantile(s) to compute, between 0 and 1 inclusive.
    axis : int or None, optional
        Axis along which the quantiles are computed.
        If None, compute over the entire array.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.

    Returns
    -------
    float or np.ndarray
        A float for scalar ``q`` with ``axis=None``. Otherwise an array of
        shape ``q.shape + reduced_shape``, where ``reduced_shape`` is
        ``data.shape`` with ``axis`` removed.

    Raises
    ------
    TypeError
        If data or q is not real numeric, axis is not an integer, or
        ignore_nan is not Boolean.
    ValueError
        If data is empty, axis is out of bounds, or q is outside [0, 1]
        or NaN.

    Notes
    -----
    Time complexity: O(n) average per quantile (selection-based).
    Space complexity: O(n) for the partitioned copy.
    """
    arr = _prepare(data, axis, ignore_nan)
    q_arr = validate_array(q, name="q").astype(float, copy=False)

    if np.any(np.isnan(q_arr)) or np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1 inclusive")

    return _reduce(
        np.nanquantile, np.quantile, arr, ignore_nan, q=q_arr, axis=axis
    )


__all__ = [
    "mean",
    "median",
    "std",
    "minimum",
    "maximum",
    "histogram",
    "quantile",
]
