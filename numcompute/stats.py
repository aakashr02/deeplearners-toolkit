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

Streaming statistics
--------------------
``RunningStats``, ``StreamingHistogram`` and ``StreamingQuantile``
consume data one chunk at a time through ``update_stats(X_chunk)``, so
statistics can be maintained over streams that never fit in memory.
A chunk is 1D (``n_samples``; a single feature) or 2D
(``n_samples, n_features``); statistics are per feature. Every object
supports ``reset()`` and records ``n_features_in_`` on the first
non-empty chunk; later chunks with a different feature count raise
``ValueError``.
"""

import warnings
from collections import deque

import numpy as np

from numcompute.rank import percentile
from numcompute.utils import (
    NotFittedError,
    validate_array,
    validate_bool,
    validate_int,
)


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


def variance(data, axis=None, ignore_nan=True, ddof=0):
    """
    Calculate the variance.

    Parameters
    ----------
    data : array-like
        Real numeric input of any shape, non-empty.
    axis : int or None, optional
        Axis along which the variance is calculated.
        If None, calculate over the entire array.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.
    ddof : int, default=0
        Delta degrees of freedom: the divisor is ``N - ddof``.
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
    Time complexity: O(n). Space complexity: O(n) for temporaries.
    """
    arr = _prepare(data, axis, ignore_nan)
    ddof = validate_int(ddof, "ddof", minimum=0)
    return _reduce(np.nanvar, np.var, arr, ignore_nan, axis=axis, ddof=ddof)


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


# ============================================================
# 3. STREAMING STATISTICS
# ============================================================

def _as_chunk(X_chunk, allow_nan=True):
    """
    Validate one chunk and return it as a float64 (n_samples, n_features)
    array. A 1D chunk is treated as one feature.
    """
    arr = validate_array(
        X_chunk,
        name="X_chunk",
        allow_nan=allow_nan,
        allow_inf=False,
    ).astype(float, copy=False)

    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(
            f"X_chunk must be 1D or 2D; received {arr.ndim} dimensions"
        )

    return arr


class _StreamingBase:
    """Shared chunk validation, feature tracking and output shaping."""

    def _reset_features(self):
        self.n_features_in_ = None
        self._input_1d = None
        self.n_samples_seen_ = 0
        self.n_chunks_seen_ = 0

    def _validate_chunk(self, X_chunk, allow_nan=True):
        is_1d = np.ndim(X_chunk) == 1
        X = _as_chunk(X_chunk, allow_nan=allow_nan)

        if self.n_features_in_ is None:
            # An empty 2D chunk with zero columns carries no information.
            if X.shape[0] == 0 and X.shape[1] == 0:
                return X
            self.n_features_in_ = X.shape[1]
            self._input_1d = is_1d
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_chunk has {X.shape[1]} features, but previous chunks "
                f"had {self.n_features_in_} features"
            )

        return X

    def _require_data(self):
        if self.n_features_in_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} has not seen any data yet. "
                "Call update_stats() with at least one chunk first."
            )

    def _shape_output(self, values):
        """Return a float for 1D streams, an array per feature otherwise."""
        values = np.asarray(values, dtype=float)
        if self._input_1d and values.shape[-1] == 1:
            values = values[..., 0]
        return float(values) if values.ndim == 0 else values


class RunningStats(_StreamingBase):
    """
    Exact running count, mean, variance, minimum and maximum per feature.

    Chunks are combined with the parallel form of Welford's algorithm
    (Chan, Golub and LeVeque). Each chunk's mean and sum of squared
    deviations ``M2`` are computed with vectorised NumPy, then merged::

        delta = mean_b - mean_a
        n     = n_a + n_b
        mean  = mean_a + delta * n_b / n
        M2    = M2_a + M2_b + delta**2 * n_a * n_b / n

    This never forms ``sum(x**2)``, so it avoids the catastrophic
    cancellation of the naive ``E[x^2] - E[x]^2`` formula, and the result
    is independent of how the stream is split into chunks.

    Parameters
    ----------
    ignore_nan : bool, default=True
        If True, NaN values are skipped and counted per feature in
        ``n_nan_``. If False, a feature that has seen any NaN reports NaN
        for every statistic.

    Attributes
    ----------
    n_features_in_ : int or None
        Number of features, fixed by the first non-empty chunk.
    n_samples_seen_ : int
        Total number of rows seen (including rows with NaN).
    n_chunks_seen_ : int
        Number of ``update_stats`` calls.
    count_ : np.ndarray of shape (n_features,)
        Number of non-NaN values per feature.
    n_nan_ : np.ndarray of shape (n_features,)
        Number of NaN values per feature.

    Examples
    --------
    >>> rs = RunningStats()
    >>> for chunk in ([1.0, 2.0], [3.0], [4.0, 5.0]):
    ...     rs.update_stats(chunk)
    >>> rs.mean, rs.variance()
    (3.0, 2.0)
    """

    def __init__(self, ignore_nan=True):
        self.ignore_nan = validate_bool(ignore_nan, "ignore_nan")
        self.reset()

    def reset(self):
        """
        Discard all accumulated state.

        Returns
        -------
        RunningStats
            ``self``, so calls can be chained.

        Raises
        ------
        None.

        Notes
        -----
        Time complexity: O(1). Space complexity: O(1).
        """
        self._reset_features()
        self.count_ = None
        self.n_nan_ = None
        self._mean = None
        self._m2 = None
        self._min = None
        self._max = None
        return self

    def _init_state(self, n_features):
        self.count_ = np.zeros(n_features, dtype=np.int64)
        self.n_nan_ = np.zeros(n_features, dtype=np.int64)
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)

    def update_stats(self, X_chunk):
        """
        Incorporate one chunk of data.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples,) or (n_samples, n_features)
            Real, finite-or-NaN values. Empty chunks are allowed and
            leave the statistics unchanged.

        Returns
        -------
        RunningStats
            ``self``.

        Raises
        ------
        TypeError
            If the chunk is not real numeric data.
        ValueError
            If the chunk is not 1D/2D, contains infinite values, or its
            feature count differs from previous chunks.

        Notes
        -----
        Time complexity: O(n_samples * n_features) per chunk.
        Space complexity: O(n_features) state, independent of stream
        length.
        """
        X = self._validate_chunk(X_chunk)
        self.n_chunks_seen_ += 1

        if self.n_features_in_ is None:
            return self

        if self.count_ is None:
            self._init_state(self.n_features_in_)

        self.n_samples_seen_ += X.shape[0]
        if X.shape[0] == 0:
            return self

        valid = ~np.isnan(X)
        n_b = valid.sum(axis=0)
        self.n_nan_ += X.shape[0] - n_b

        has_data = n_b > 0
        if not np.any(has_data):
            return self

        # Per-feature chunk statistics over non-NaN values only.
        X0 = np.where(valid, X, 0.0)
        safe_n_b = np.maximum(n_b, 1)
        mean_b = X0.sum(axis=0) / safe_n_b
        dev = np.where(valid, X - mean_b, 0.0)
        m2_b = np.sum(dev * dev, axis=0)

        n_a = self.count_
        n = n_a + n_b
        safe_n = np.maximum(n, 1)
        delta = mean_b - self._mean

        self._mean = np.where(
            has_data, self._mean + delta * n_b / safe_n, self._mean
        )
        self._m2 = np.where(
            has_data,
            self._m2 + m2_b + delta * delta * n_a * n_b / safe_n,
            self._m2,
        )
        self.count_ = n

        self._min = np.minimum(
            self._min, np.min(np.where(valid, X, np.inf), axis=0)
        )
        self._max = np.maximum(
            self._max, np.max(np.where(valid, X, -np.inf), axis=0)
        )
        return self

    # Alias used by streaming components (scalers, pipelines).
    partial_fit = update_stats

    def merge(self, other):
        """
        Combine another RunningStats into this one (parallel reduction).

        After merging, ``self`` holds the statistics of both streams, as
        if all of ``other``'s chunks had been passed to ``self``.

        Parameters
        ----------
        other : RunningStats
            Statistics over a disjoint part of the data.

        Returns
        -------
        RunningStats
            ``self``.

        Raises
        ------
        TypeError
            If other is not a RunningStats.
        ValueError
            If the feature counts or ignore_nan settings differ.

        Notes
        -----
        Time complexity: O(n_features). Space complexity: O(n_features).
        """
        if not isinstance(other, RunningStats):
            raise TypeError("other must be a RunningStats instance")

        if other.ignore_nan != self.ignore_nan:
            raise ValueError("cannot merge stats with different ignore_nan")

        if other.count_ is None:
            self.n_chunks_seen_ += other.n_chunks_seen_
            self.n_samples_seen_ += other.n_samples_seen_
            return self

        if self.count_ is None:
            self.n_features_in_ = other.n_features_in_
            self._input_1d = other._input_1d
            self._init_state(other.n_features_in_)
        elif other.n_features_in_ != self.n_features_in_:
            raise ValueError(
                f"cannot merge stats with {other.n_features_in_} features "
                f"into stats with {self.n_features_in_} features"
            )

        n_a, n_b = self.count_, other.count_
        n = n_a + n_b
        safe_n = np.maximum(n, 1)
        delta = other._mean - self._mean

        self._mean = np.where(
            n > 0, self._mean + delta * n_b / safe_n, self._mean
        )
        self._m2 = self._m2 + other._m2 + delta * delta * n_a * n_b / safe_n
        self.count_ = n
        self.n_nan_ = self.n_nan_ + other.n_nan_
        self._min = np.minimum(self._min, other._min)
        self._max = np.maximum(self._max, other._max)
        self.n_samples_seen_ += other.n_samples_seen_
        self.n_chunks_seen_ += other.n_chunks_seen_
        return self

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    def _finalise(self, values):
        """NaN where a feature has no data (or saw NaN, if not ignoring)."""
        self._require_data()
        if self.count_ is None:  # only empty chunks seen
            values = np.full(self.n_features_in_, np.nan)
        undefined = self.count_ == 0 if self.count_ is not None else True
        if not self.ignore_nan and self.n_nan_ is not None:
            undefined = undefined | (self.n_nan_ > 0)
        values = np.where(undefined, np.nan, values)
        return self._shape_output(values)

    @property
    def mean(self):
        """
        Running mean: a float for a 1D stream, else one value per feature.

        NaN for features with no non-NaN values.

        Raises
        ------
        NotFittedError
            If no chunk has been seen.
        """
        return self._finalise(self._mean)

    def variance(self, ddof=0):
        """
        Running variance ``M2 / (count - ddof)``.

        Parameters
        ----------
        ddof : int, default=0
            Delta degrees of freedom (1 gives the sample variance).
            Features with ``count <= ddof`` give NaN.

        Returns
        -------
        float or np.ndarray of shape (n_features,)

        Raises
        ------
        NotFittedError
            If no chunk has been seen.
        TypeError, ValueError
            If ddof is not a non-negative integer.

        Notes
        -----
        Time complexity: O(n_features).
        """
        ddof = validate_int(ddof, "ddof", minimum=0)
        self._require_data()
        if self.count_ is None:
            return self._finalise(None)

        denom = self.count_ - ddof
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(denom > 0, self._m2 / np.maximum(denom, 1), np.nan)

        # A constant feature (min == max) has variance exactly 0. Without
        # this, rounding in each chunk's mean (e.g. 0.1 is not exact in
        # binary) leaves M2 ~ 1e-34, which would make a downstream scaler
        # divide by ~1e-17 instead of detecting the constant column.
        constant = (self._min == self._max) & (denom > 0)
        var = np.where(constant, 0.0, np.maximum(var, 0.0))
        return self._finalise(var)

    def std(self, ddof=0):
        """
        Running standard deviation ``sqrt(variance(ddof))``.

        Parameters and returns are as for ``variance``.

        Raises
        ------
        NotFittedError
            If no chunk has been seen.
        TypeError, ValueError
            If ddof is not a non-negative integer.

        Notes
        -----
        Time complexity: O(n_features).
        """
        var = self.variance(ddof)
        return float(np.sqrt(var)) if np.ndim(var) == 0 else np.sqrt(var)

    @property
    def min(self):
        """Running minimum per feature (NaN where no data)."""
        self._require_data()
        return self._finalise(self._min if self._min is not None else None)

    @property
    def max(self):
        """Running maximum per feature (NaN where no data)."""
        self._require_data()
        return self._finalise(self._max if self._max is not None else None)

    @property
    def count(self):
        """Number of non-NaN values: an int for a 1D stream, else per feature."""
        self._require_data()
        counts = (
            self.count_ if self.count_ is not None
            else np.zeros(self.n_features_in_, dtype=np.int64)
        )
        if self._input_1d:
            return int(counts[0])
        return counts.copy()

    def result(self, ddof=0):
        """
        All statistics as a dictionary.

        Returns
        -------
        dict
            Keys ``count``, ``mean``, ``variance``, ``std``, ``min``,
            ``max``, ``n_samples_seen``, ``n_chunks_seen``.

        Raises
        ------
        NotFittedError
            If no chunk has been seen.

        Notes
        -----
        Time complexity: O(n_features).
        """
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance(ddof),
            "std": self.std(ddof),
            "min": self.min,
            "max": self.max,
            "n_samples_seen": self.n_samples_seen_,
            "n_chunks_seen": self.n_chunks_seen_,
        }

    @classmethod
    def from_chunks(cls, chunks, ignore_nan=True):
        """
        Build a RunningStats from an iterable of chunks.

        Parameters
        ----------
        chunks : iterable of array-like
            E.g. ``numcompute.io.read_csv_chunks(path)``.
        ignore_nan : bool, default=True

        Returns
        -------
        RunningStats
        """
        stats = cls(ignore_nan=ignore_nan)
        for chunk in chunks:
            stats.update_stats(chunk)
        return stats


class StreamingHistogram(_StreamingBase):
    """
    Histogram over a stream with fixed bin edges, optionally windowed.

    Bin edges must be fixed up front (a streaming histogram cannot move
    its edges without revisiting old data). Counts are per feature.

    Parameters
    ----------
    bins : int or sequence of float, default=10
        Number of equal-width bins (requires ``range``) or explicit,
        strictly increasing bin edges.
    range : tuple (float, float) or None
        Lower and upper edge when ``bins`` is an int.
    window : int or None, default=None
        If given, counts cover only the most recent ``window`` chunks.
        None accumulates over the whole stream.

    Attributes
    ----------
    bin_edges_ : np.ndarray of shape (n_bins + 1,)
        Bin edges. The last bin includes its right edge.
    n_below_, n_above_ : np.ndarray of shape (n_features,)
        Values outside the edges (within the current window), which are
        not counted in any bin.
    n_nan_ : np.ndarray of shape (n_features,)
        NaN values skipped (within the current window).

    Raises
    ------
    ValueError
        If bins/range do not define at least one strictly increasing,
        finite bin, or window < 1.
    """

    def __init__(self, bins=10, range=None, window=None):
        if np.ndim(bins) == 0:
            n_bins = validate_int(bins, "bins", minimum=1)
            if range is None:
                raise ValueError(
                    "range=(low, high) is required when bins is an integer"
                )
            low, high = float(range[0]), float(range[1])
            if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
                raise ValueError("range must be finite with low < high")
            edges = np.linspace(low, high, n_bins + 1)
        else:
            edges = validate_array(
                bins, name="bins", ndim=1, allow_nan=False, allow_inf=False
            ).astype(float)
            if edges.size < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError(
                    "bin edges must contain at least two strictly "
                    "increasing values"
                )

        if window is not None:
            window = validate_int(window, "window", minimum=1)

        self.bin_edges_ = edges
        self.window = window
        self.reset()

    @property
    def n_bins(self):
        return self.bin_edges_.size - 1

    def reset(self):
        """
        Discard all counts (bin edges are kept).

        Returns
        -------
        StreamingHistogram
            ``self``.

        Raises
        ------
        None.

        Notes
        -----
        Time complexity: O(1). Space complexity: O(1).
        """
        self._reset_features()
        self._chunks = deque(maxlen=self.window)
        self._total = None
        return self

    def update_stats(self, X_chunk):
        """
        Add one chunk's values to the histogram.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples,) or (n_samples, n_features)
            Real values; NaN is skipped and counted in ``n_nan_``.

        Returns
        -------
        StreamingHistogram
            ``self``.

        Raises
        ------
        TypeError
            If the chunk is not real numeric data.
        ValueError
            If the chunk is not 1D/2D, contains infinite values, or its
            feature count differs from previous chunks.

        Notes
        -----
        Fully vectorised: each value's bin is found with one
        ``searchsorted`` and counted with one ``bincount`` per chunk.

        Time complexity: O(n_samples * n_features * log n_bins).
        Space complexity: O(window * n_features * n_bins) with a window,
        otherwise O(n_features * n_bins).
        """
        X = self._validate_chunk(X_chunk)
        self.n_chunks_seen_ += 1

        if self.n_features_in_ is None:
            return self

        self.n_samples_seen_ += X.shape[0]
        n_features, n_bins = self.n_features_in_, self.n_bins
        edges = self.bin_edges_

        nan = np.isnan(X)
        below = (X < edges[0]) & ~nan
        above = (X > edges[-1]) & ~nan
        inside = ~(nan | below | above)

        # Bin index per value; the right-most edge belongs to the last bin.
        idx = np.searchsorted(edges, np.where(inside, X, edges[0]),
                              side="right") - 1
        idx = np.clip(idx, 0, n_bins - 1)

        feature = np.broadcast_to(np.arange(n_features), X.shape)
        flat = (feature * n_bins + idx)[inside]
        counts = np.bincount(flat, minlength=n_features * n_bins)
        counts = counts.reshape(n_features, n_bins)

        record = np.column_stack(
            (counts, below.sum(axis=0), above.sum(axis=0), nan.sum(axis=0))
        )

        if self.window is None:
            self._total = record if self._total is None else self._total + record
        else:
            self._chunks.append(record)  # deque drops the oldest chunk
            self._total = None

        return self

    partial_fit = update_stats

    def _current(self):
        self._require_data()
        if self.window is None:
            if self._total is None:
                return np.zeros((self.n_features_in_, self.n_bins + 3),
                                dtype=np.int64)
            return self._total
        if not self._chunks:
            return np.zeros((self.n_features_in_, self.n_bins + 3),
                            dtype=np.int64)
        return np.sum(self._chunks, axis=0)

    def _shape_counts(self, values):
        return values[0] if self._input_1d else values

    @property
    def counts_(self):
        """
        Bin counts: shape (n_bins,) for a 1D stream, else
        (n_features, n_bins).

        Raises
        ------
        NotFittedError
            If no chunk has been seen.
        """
        return self._shape_counts(self._current()[:, : self.n_bins])

    @property
    def n_below_(self):
        return self._shape_counts(self._current()[:, self.n_bins])

    @property
    def n_above_(self):
        return self._shape_counts(self._current()[:, self.n_bins + 1])

    @property
    def n_nan_(self):
        return self._shape_counts(self._current()[:, self.n_bins + 2])

    def result(self):
        """
        Return ``(counts, bin_edges)``, matching ``np.histogram``'s order.

        Raises
        ------
        NotFittedError
            If no chunk has been seen.

        Notes
        -----
        Time complexity: O(n_features * n_bins), plus O(window) chunk
        records summed when a window is used.
        """
        return self.counts_, self.bin_edges_.copy()


class StreamingQuantile(_StreamingBase):
    """
    Quantiles over a stream: exact, windowed, or bounded-memory.

    Three modes, chosen by ``window`` and ``max_centroids``:

    * **Exact** (default): every value is kept, so memory grows with the
      stream (O(total values)). Results equal ``np.nanquantile``.
    * **Sliding window** (``window=W``): only the last W chunks are kept,
      giving exact quantiles of recent data in bounded memory.
    * **Bounded sketch** (``max_centroids=K``): quantiles of the whole
      stream in O(K) memory per feature. Raw values are kept until more
      than ``2 * K`` accumulate for a feature; they are then merged into K
      equal-mass weighted centroids. Results are exact until the first
      compression and approximate afterwards (min and max stay exact).

    Exact results are computed with ``numcompute.rank.percentile``, so all
    of its interpolation methods are available. Once a sketch has been
    compressed, quantiles are interpolated linearly between centroid
    midpoints and ``interpolation`` no longer applies.

    Parameters
    ----------
    q : float or sequence of float
        Quantile(s) in [0, 1].
    interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}
        Interpolation between data points (default "linear").
    window : int or None, default=None
        Number of most recent chunks to keep. None keeps all data.
    max_centroids : int or None, default=None
        Sketch size K (at least 2) for bounded memory. None disables the
        sketch. Cannot be combined with ``window``, because a merged sketch
        cannot forget an old chunk.

    Attributes
    ----------
    is_exact_ : bool
        False once the sketch has compressed values into centroids.

    Raises
    ------
    ValueError
        If q is outside [0, 1], the interpolation is unknown, window < 1,
        max_centroids < 2, or both window and max_centroids are given.

    Notes
    -----
    The sketch update, compression and query are vectorised across all
    features: per-feature rows are padded with ``(value=inf, weight=0)``
    entries, sorted with one ``argsort``, and centroids are formed with
    ``bincount`` over equal-mass buckets of the cumulative weight.
    """

    _INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")

    def __init__(self, q=0.5, interpolation="linear", window=None,
                 max_centroids=None):
        q_arr = validate_array(q, name="q", allow_nan=False).astype(float)
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("q must be between 0 and 1 inclusive")
        if interpolation not in self._INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {self._INTERPOLATIONS}"
            )
        if window is not None:
            window = validate_int(window, "window", minimum=1)
        if max_centroids is not None:
            max_centroids = validate_int(
                max_centroids, "max_centroids", minimum=2
            )
            if window is not None:
                raise ValueError(
                    "window and max_centroids cannot be combined, because "
                    "a merged sketch cannot evict old chunks"
                )

        self.q = q_arr
        self.interpolation = interpolation
        self.window = window
        self.max_centroids = max_centroids
        self.reset()

    def reset(self):
        """
        Discard all stored values.

        Returns
        -------
        StreamingQuantile
            ``self``.

        Raises
        ------
        None.

        Notes
        -----
        Time complexity: O(1). Space complexity: O(1).
        """
        self._reset_features()
        self._chunks = deque(maxlen=self.window)
        self._vals = None      # sketch values  (n_features, m), pad = inf
        self._wts = None       # sketch weights (n_features, m), pad = 0
        self._min = None
        self._max = None
        self.is_exact_ = True
        return self

    def update_stats(self, X_chunk):
        """
        Add one chunk of values.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples,) or (n_samples, n_features)
            Real values; NaN is ignored when computing quantiles.

        Returns
        -------
        StreamingQuantile
            ``self``.

        Raises
        ------
        TypeError
            If the chunk is not real numeric data.
        ValueError
            If the chunk is not 1D/2D, contains infinite values, or its
            feature count differs from previous chunks.

        Notes
        -----
        Time complexity: O(1) amortised in exact/window mode (the chunk is
        copied once); O((K + n) log(K + n)) per chunk with a sketch.
        Space complexity: O(values kept), or O(K * n_features) with a
        sketch.
        """
        X = self._validate_chunk(X_chunk)
        self.n_chunks_seen_ += 1

        if self.n_features_in_ is None:
            return self

        self.n_samples_seen_ += X.shape[0]

        if self.max_centroids is not None:
            self._update_sketch(X)
            return self

        # Copy so later changes to the caller's array cannot leak in.
        self._chunks.append(np.array(X, dtype=float))

        # Without a window, consolidate periodically to keep the list short.
        if self.window is None and len(self._chunks) > 64:
            merged = np.concatenate(list(self._chunks), axis=0)
            self._chunks.clear()
            self._chunks.append(merged)

        return self

    partial_fit = update_stats

    # --------------------------------------------------------
    # Bounded-memory sketch
    # --------------------------------------------------------

    @staticmethod
    def _sort_trim(V, W):
        """Sort each row by value (padding last) and drop unused columns."""
        order = np.argsort(V, axis=1, kind="stable")
        V = np.take_along_axis(V, order, axis=1)
        W = np.take_along_axis(W, order, axis=1)
        m = max(int((W > 0).sum(axis=1).max(initial=0)), 1)
        return V[:, :m], W[:, :m]

    @classmethod
    def _compress(cls, V, W, k):
        """Merge each row into at most k equal-mass weighted centroids."""
        n_rows = V.shape[0]
        total = W.sum(axis=1, keepdims=True)
        midpoint = np.cumsum(W, axis=1) - 0.5 * W
        bucket = np.minimum(
            (midpoint / np.where(total > 0, total, 1.0) * k).astype(int),
            k - 1,
        )
        flat = (bucket + k * np.arange(n_rows)[:, None]).ravel()
        weighted = (np.where(W > 0, V, 0.0) * W).ravel()
        new_w = np.bincount(flat, weights=W.ravel(),
                            minlength=n_rows * k).reshape(n_rows, k)
        new_v = np.bincount(flat, weights=weighted,
                            minlength=n_rows * k).reshape(n_rows, k)
        keep = new_w > 0
        new_v = np.where(keep, new_v / np.where(keep, new_w, 1.0), np.inf)
        return cls._sort_trim(new_v, new_w)

    def _update_sketch(self, X):
        n_features = self.n_features_in_
        if self._vals is None:
            self._vals = np.empty((n_features, 0))
            self._wts = np.empty((n_features, 0))
            self._min = np.full(n_features, np.inf)
            self._max = np.full(n_features, -np.inf)

        if X.shape[0] == 0:
            return

        valid = ~np.isnan(X)
        self._min = np.minimum(
            self._min, np.min(np.where(valid, X, np.inf), axis=0)
        )
        self._max = np.maximum(
            self._max, np.max(np.where(valid, X, -np.inf), axis=0)
        )

        new_v = np.where(valid.T, X.T, np.inf)        # (features, samples)
        V, W = self._sort_trim(
            np.concatenate([self._vals, new_v], axis=1),
            np.concatenate([self._wts, valid.T.astype(float)], axis=1),
        )
        if V.shape[1] > 2 * self.max_centroids:
            V, W = self._compress(V, W, self.max_centroids)
            self.is_exact_ = False
        self._vals, self._wts = V, W

    def _sketch_quantiles(self):
        """Linear interpolation over centroid midpoints, anchored at min/max."""
        V, W = self._vals, self._wts
        n_features = V.shape[0]
        qf = self.q.ravel()
        total = W.sum(axis=1)
        has_data = total > 0

        # Features with no values have min=+inf and max=-inf; use 0 as a
        # placeholder so the interpolation never computes inf - inf. Their
        # results are replaced with NaN below.
        low = np.where(has_data, self._min, 0.0)[:, None]
        top = np.where(has_data, self._max, 0.0)[:, None]

        # Cumulative-weight position of each centroid, plus both ends.
        xp = np.concatenate([np.zeros((n_features, 1)),
                             np.cumsum(W, axis=1) - 0.5 * W,
                             total[:, None]], axis=1)
        fp = np.concatenate([low, np.where(W > 0, V, top), top], axis=1)

        target = qf[None, :] * total[:, None]               # (F, Q)
        i = (xp[:, None, :] <= target[:, :, None]).sum(axis=2) - 1
        i = np.clip(i, 0, xp.shape[1] - 2)
        x0 = np.take_along_axis(xp, i, axis=1)
        x1 = np.take_along_axis(xp, i + 1, axis=1)
        f0 = np.take_along_axis(fp, i, axis=1)
        f1 = np.take_along_axis(fp, i + 1, axis=1)
        gap = x1 - x0
        frac = np.where(gap > 0,
                        (target - x0) / np.where(gap > 0, gap, 1.0), 0.0)
        out = f0 + (f1 - f0) * frac
        out = np.where(has_data[:, None], out, np.nan)      # (F, Q)
        return out.T.reshape(self.q.shape + (n_features,))

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    @property
    def n_values_stored_(self):
        """Number of values (or centroids) currently held in memory."""
        if self.max_centroids is not None:
            return 0 if self._wts is None else int((self._wts > 0).sum())
        return int(sum(chunk.size for chunk in self._chunks))

    def result(self):
        """
        Quantiles of the stored values.

        Returns
        -------
        float or np.ndarray
            For scalar ``q`` on a 1D stream, a float. Otherwise an array
            of shape ``q.shape + (n_features,)``, with the feature axis
            dropped for 1D streams. Features with no non-NaN values give
            NaN.

        Raises
        ------
        NotFittedError
            If no chunk has been seen.

        Notes
        -----
        Time complexity: O(N log N) for N stored values; O(K * |q|) per
        feature for a compressed sketch.
        """
        self._require_data()

        if self.max_centroids is not None:
            if self._vals is None or self._vals.shape[1] == 0:
                values = np.full(self.q.shape + (self.n_features_in_,), np.nan)
            elif not self.is_exact_:
                values = self._sketch_quantiles()
            else:
                # Still raw values: padding becomes NaN and is omitted.
                data = np.where(self._wts > 0, self._vals, np.nan).T
                values = percentile(data, self.q * 100.0,
                                    interpolation=self.interpolation,
                                    axis=0, nan_policy="omit")
            return self._shape_output(values)

        data = (
            np.concatenate(list(self._chunks), axis=0)
            if self._chunks
            else np.empty((0, self.n_features_in_))
        )

        q_shape = self.q.shape
        if data.shape[0] == 0:
            values = np.full(q_shape + (self.n_features_in_,), np.nan)
        else:
            values = percentile(
                data,
                self.q * 100.0,
                interpolation=self.interpolation,
                axis=0,
                nan_policy="omit",
            )
        return self._shape_output(values)


def update_stats(X_chunk, state=None):
    """
    Functional streaming API: update (or create) running statistics.

    Parameters
    ----------
    X_chunk : array-like of shape (n_samples,) or (n_samples, n_features)
        The next chunk of data.
    state : RunningStats, StreamingHistogram, StreamingQuantile or None
        Accumulated state. None creates a new ``RunningStats``.

    Returns
    -------
    RunningStats, StreamingHistogram or StreamingQuantile
        The updated state (the same object when one is passed in).

    Raises
    ------
    TypeError
        If state is not a streaming statistics object.
    ValueError
        As raised by the state's ``update_stats``.

    Examples
    --------
    >>> state = None
    >>> for chunk in ([1.0, 2.0], [3.0, 4.0]):
    ...     state = update_stats(chunk, state)
    >>> state.mean
    2.5

    Notes
    -----
    Time and space complexity are those of ``state.update_stats``.
    """
    if state is None:
        state = RunningStats()
    elif not isinstance(state, _StreamingBase):
        raise TypeError(
            "state must be a RunningStats, StreamingHistogram or "
            "StreamingQuantile instance, or None"
        )
    return state.update_stats(X_chunk)


__all__ = [
    "mean",
    "median",
    "std",
    "variance",
    "minimum",
    "maximum",
    "histogram",
    "quantile",
    "RunningStats",
    "StreamingHistogram",
    "StreamingQuantile",
    "update_stats",
]
