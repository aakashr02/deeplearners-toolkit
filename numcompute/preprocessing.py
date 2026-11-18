"""
preprocessing.py
================

Feature scaling, imputation and categorical encoding for NumCompute,
with batch and streaming (chunk-by-chunk) fitting.

Every transformer follows the same contract:

* ``partial_fit(X_chunk)`` folds one chunk of shape
  ``(n_samples, n_features)`` into the learned state and returns ``self``.
  The first chunk fixes ``n_features_in_``.
* ``fit(X)`` forgets everything and then calls ``partial_fit(X)``, so a
  single batch and a stream share one code path.
* ``transform(X)`` uses the state learned so far and never changes it.
  It requires at least one ``partial_fit``/``fit`` (otherwise
  ``NotFittedError``) and exactly ``n_features_in_`` columns (otherwise
  ``ValueError``). The input is never modified.
* ``fit_transform(X)`` is equivalent to ``fit(X).transform(X)``.
* ``reset()`` discards all learned state.

``y`` is accepted and ignored by ``fit``/``partial_fit`` so transformers
can sit in a ``Pipeline`` next to an estimator.

The scalers reject NaN and infinite values, because they would silently
corrupt the learned statistics; place a ``SimpleImputer`` before them to
handle missing values. Constant features are scaled with a divisor of 1
instead of 0, so they never produce NaN or inf.
"""

import numpy as np

from numcompute.stats import RunningStats, StreamingQuantile
from numcompute.utils import check_is_fitted, validate_2d


class _Transformer:
    """Shared validation, reset and ``fit``/``fit_transform`` logic."""

    # Attributes created by partial_fit(); used by check_is_fitted().
    _fitted_attributes = ("n_features_in_",)
    # Every attribute reset() must remove.
    _state_attributes = ("n_features_in_", "n_samples_seen_")
    _numeric = True
    _allow_nan = False

    def reset(self):
        """
        Discard everything learned so far (constructor settings are kept).

        Returns
        -------
        self

        Raises
        ------
        None.

        Notes
        -----
        Time complexity: O(1).
        """
        for attr in self._state_attributes:
            self.__dict__.pop(attr, None)
        self._reset_state()
        return self

    def _reset_state(self):
        """Hook for subclasses with extra internal state."""

    def _is_fitted(self):
        return hasattr(self, "n_features_in_")

    def _validate_partial_fit_input(self, X):
        # Empty chunks are allowed (streams can produce them); callers
        # return early after the shape check without changing any state.
        return validate_2d(
            X,
            numeric=self._numeric,
            allow_nan=self._allow_nan,
            n_features=getattr(self, "n_features_in_", None),
            allow_empty=True,
        )

    def _validate_transform_input(self, X):
        check_is_fitted(self, self._fitted_attributes)
        return validate_2d(
            X,
            numeric=self._numeric,
            allow_nan=self._allow_nan,
            n_features=self.n_features_in_,
            allow_empty=True,
        )

    def _record_chunk(self, X):
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = getattr(self, "n_samples_seen_", 0) + X.shape[0]

    def fit(self, X, y=None):
        """
        Forget previous state and learn from ``X`` (batch use).

        Equivalent to ``reset().partial_fit(X)``, except that ``X`` must
        contain at least one sample: an empty chunk is a harmless no-op in
        a stream, but an empty batch passed to ``fit`` is a caller error.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X has no samples, plus everything ``partial_fit`` raises.

        Notes
        -----
        Time and space complexity are those of one ``partial_fit`` call.
        """
        X_checked = validate_2d(X, numeric=self._numeric,
                                allow_nan=self._allow_nan)
        return self.reset().partial_fit(X_checked)

    def fit_transform(self, X, y=None):
        """
        Fit to ``X`` and return the transformed ``X``.

        Equivalent to ``fit(X).transform(X)``; see those methods for
        shapes, exceptions and complexity.
        """
        return self.fit(X).transform(X)


def _safe_scale(scale):
    """Replace zero scale factors with 1 so constant features map safely."""
    scale = np.asarray(scale, dtype=float).copy()
    scale[scale == 0.0] = 1.0
    return scale


# ============================================================
# StandardScaler
# ============================================================

class StandardScaler(_Transformer):
    """
    Standardise features by removing the mean and scaling to unit variance.

    ``z = (x - mean_) / scale_`` where ``scale_`` is the population
    standard deviation (ddof=0), with zero replaced by 1.

    The running mean and variance are maintained by
    ``numcompute.stats.RunningStats`` (parallel Welford updates), so
    ``partial_fit`` over any chunking gives the same statistics as ``fit``
    on all of the data, and stays accurate for data with a large offset.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean of all data seen.
    var_ : np.ndarray of shape (n_features,)
        Per-feature population variance.
    std_ : np.ndarray of shape (n_features,)
        Per-feature population standard deviation (may contain 0).
    scale_ : np.ndarray of shape (n_features,)
        Divisor used by ``transform``: ``std_`` with zeros replaced by 1,
        so a constant feature is transformed to 0 rather than NaN.
    n_features_in_ : int
        Number of features seen during fitting.
    n_samples_seen_ : int
        Total number of rows seen.
    """

    _fitted_attributes = ("mean_", "scale_", "n_features_in_")
    _state_attributes = (
        "mean_", "var_", "std_", "scale_", "n_features_in_",
        "n_samples_seen_",
    )

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self._stats = RunningStats()

    def partial_fit(self, X, y=None):
        """
        Update the running mean and standard deviation with one chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Real, finite data. An empty chunk is a no-op. After the first
            chunk, ``n_features`` must equal ``n_features_in_``.
        y : ignored

        Returns
        -------
        StandardScaler
            The updated scaler (``self``).

        Raises
        ------
        TypeError
            If X is not real numeric data.
        ValueError
            If X is not 2D, contains NaN/inf, or its column
            count differs from earlier chunks.

        Notes
        -----
        Time complexity: O(n_samples * n_features) per chunk.
        Space complexity: O(n_features), independent of stream length.
        """
        X = self._validate_partial_fit_input(X)
        if X.shape[0] == 0:
            return self
        self._stats.update_stats(X)
        self._record_chunk(X)

        self.mean_ = self._stats.mean
        self.var_ = self._stats.variance()
        self.std_ = np.sqrt(self.var_)
        self.scale_ = _safe_scale(self.std_)
        return self

    def transform(self, X):
        """
        Standardise ``X`` using the mean and scale learned so far.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)
            Real, finite data to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features_in_)
            Standardised float64 data. Constant training features map to 0.

        Raises
        ------
        NotFittedError
            If neither ``fit`` nor ``partial_fit`` has been called.
        TypeError
            If X is not real numeric data.
        ValueError
            If X is not 2D, contains NaN/inf, or its column count differs
            from ``n_features_in_``.

        Notes
        -----
        Time complexity: O(n_samples * n_features).
        Space complexity: O(n_samples * n_features) for the output.
        """
        X = self._validate_transform_input(X)
        return (X - self.mean_) / self.scale_


# ============================================================
# MinMaxScaler
# ============================================================

class MinMaxScaler(_Transformer):
    """
    Scale features linearly to ``feature_range`` (default ``(0, 1)``).

    ``x_scaled = (x - data_min_) / scale_ * (max - min) + min`` where
    ``scale_`` is ``data_max_ - data_min_`` with zero replaced by 1.
    ``partial_fit`` keeps an exact running minimum and maximum.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired output range. ``min`` must be strictly less than ``max``.
    clip : bool, default=False
        If True, ``transform`` clips outputs to ``feature_range``. When
        streaming, later data often lies outside the range seen so far,
        which otherwise maps outside ``feature_range``.

    Attributes
    ----------
    data_min_, data_max_ : np.ndarray of shape (n_features,)
        Per-feature minimum and maximum of all data seen.
    data_range_ : np.ndarray of shape (n_features,)
        ``data_max_ - data_min_`` (may contain 0).
    scale_ : np.ndarray of shape (n_features,)
        Divisor used by ``transform``: ``data_range_`` with zeros replaced
        by 1, so a constant feature maps to ``feature_range[0]``.
    n_features_in_ : int
        Number of features seen during fitting.
    n_samples_seen_ : int
        Total number of rows seen.

    Raises
    ------
    ValueError
        If ``feature_range`` is not two finite numbers with min < max.
    TypeError
        If ``clip`` is not Boolean.
    """

    _fitted_attributes = ("data_min_", "scale_", "n_features_in_")
    _state_attributes = (
        "data_min_", "data_max_", "data_range_", "scale_",
        "n_features_in_", "n_samples_seen_",
    )

    def __init__(self, feature_range=(0, 1), clip=False):
        try:
            min_val, max_val = feature_range
            min_val, max_val = float(min_val), float(max_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "feature_range must be a pair of numbers (min, max)"
            ) from exc

        if not (np.isfinite(min_val) and np.isfinite(max_val)):
            raise ValueError("feature_range values must be finite")

        if min_val >= max_val:
            raise ValueError(
                "feature_range minimum must be smaller than maximum"
            )

        if not isinstance(clip, (bool, np.bool_)):
            raise TypeError("clip must be a boolean")

        self.feature_range = (min_val, max_val)
        self.min_val, self.max_val = min_val, max_val
        self.clip = bool(clip)

    def partial_fit(self, X, y=None):
        """
        Update the running minimum and maximum with one chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Real, finite data. An empty chunk is a no-op.
        y : ignored

        Returns
        -------
        MinMaxScaler
            The updated scaler (``self``).

        Raises
        ------
        TypeError
            If X is not real numeric data.
        ValueError
            If X is not 2D, contains NaN/inf, or its column
            count differs from earlier chunks.

        Notes
        -----
        Time complexity: O(n_samples * n_features) per chunk.
        Space complexity: O(n_features).
        """
        X = self._validate_partial_fit_input(X)
        if X.shape[0] == 0:
            return self
        chunk_min, chunk_max = X.min(axis=0), X.max(axis=0)

        if self._is_fitted():
            chunk_min = np.minimum(self.data_min_, chunk_min)
            chunk_max = np.maximum(self.data_max_, chunk_max)

        self._record_chunk(X)
        self.data_min_ = chunk_min
        self.data_max_ = chunk_max
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = _safe_scale(self.data_range_)
        return self

    def transform(self, X):
        """
        Scale ``X`` to ``feature_range`` using the min and range so far.

        Values outside the range seen so far are scaled linearly and fall
        outside ``feature_range`` unless ``clip=True``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)
            Real, finite data to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features_in_)
            Scaled float64 data.

        Raises
        ------
        NotFittedError
            If neither ``fit`` nor ``partial_fit`` has been called.
        TypeError
            If X is not real numeric data.
        ValueError
            If X is not 2D, contains NaN/inf, or its column count differs
            from ``n_features_in_``.

        Notes
        -----
        Time complexity: O(n_samples * n_features).
        Space complexity: O(n_samples * n_features) for the output.
        """
        X = self._validate_transform_input(X)
        scaled = (X - self.data_min_) / self.scale_
        out = scaled * (self.max_val - self.min_val) + self.min_val

        if self.clip:
            out = np.clip(out, self.min_val, self.max_val)

        return out


# ============================================================
# SimpleImputer
# ============================================================

class SimpleImputer(_Transformer):
    """
    Replace missing values (NaN) with per-feature statistics that are
    updated on the fly.

    Parameters
    ----------
    strategy : {"mean", "median", "most_frequent", "constant"}, default="mean"
        * "mean": running mean (``RunningStats``, exact).
        * "median": running median (``StreamingQuantile``); exact until
          ``2 * max_centroids`` values per feature, then a bounded-memory
          approximation.
        * "most_frequent": running value counts; ties go to the smallest
          value.
        * "constant": always ``fill_value``.
    fill_value : float or None, default=None
        Replacement used by the "constant" strategy (required for it). For
        the other strategies it is the fallback for a feature that has had
        no observed (non-NaN) value yet; None means a fallback of 0.0.
    max_centroids : int, default=512
        Memory bound for the "median" strategy.

    Attributes
    ----------
    statistics_ : np.ndarray of shape (n_features,)
        Current replacement value for each feature. Always finite: a
        feature with no observed values yet uses the fallback, and switches
        to its real statistic as soon as values arrive.
    n_observed_seen_ : np.ndarray of shape (n_features,)
        Number of non-NaN values seen per feature.
    n_missing_seen_ : np.ndarray of shape (n_features,)
        Number of NaN values seen per feature during fitting.
    n_features_in_ : int
        Number of features seen during fitting.
    n_samples_seen_ : int
        Total number of rows seen.

    Raises
    ------
    ValueError
        If the strategy is unknown, ``fill_value`` is given but is not a
        finite real number, or "constant" is used without a ``fill_value``.
    """

    _STRATEGIES = ("mean", "median", "most_frequent", "constant")
    _fitted_attributes = ("statistics_", "n_features_in_")
    _state_attributes = (
        "statistics_", "n_missing_seen_", "n_observed_seen_",
        "n_features_in_", "n_samples_seen_",
    )
    _allow_nan = True

    def __init__(self, strategy="mean", fill_value=None, max_centroids=512):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self._STRATEGIES}; "
                f"received {strategy!r}"
            )

        if fill_value is not None:
            is_real = isinstance(fill_value, (int, float, np.integer,
                                              np.floating))
            if (isinstance(fill_value, (bool, np.bool_)) or not is_real
                    or not np.isfinite(fill_value)):
                raise ValueError(
                    f"fill_value must be a finite real number; "
                    f"received {fill_value!r}"
                )
        elif strategy == "constant":
            raise ValueError("strategy='constant' requires a fill_value")

        self.strategy = strategy
        self.fill_value = fill_value
        self.max_centroids = max_centroids
        self._reset_state()

    def _reset_state(self):
        if self.strategy == "mean":
            self._stats = RunningStats()
        elif self.strategy == "median":
            self._stats = StreamingQuantile(
                0.5, max_centroids=self.max_centroids
            )
        else:
            self._stats = None
        # most_frequent: per-feature sorted unique values and their counts.
        self._values = None
        self._counts = None

    def _update_counts(self, X):
        """Merge this chunk's value counts into the running counts."""
        if self._values is None:
            self._values = [np.empty(0) for _ in range(X.shape[1])]
            self._counts = [np.empty(0, dtype=np.int64)
                            for _ in range(X.shape[1])]

        # One loop over features; counting within a feature is vectorised.
        for j in range(X.shape[1]):
            column = X[:, j]
            column = column[~np.isnan(column)]
            values = np.concatenate((self._values[j], column))
            weights = np.concatenate(
                (self._counts[j], np.ones(column.size, dtype=np.int64))
            )
            unique, inverse = np.unique(values, return_inverse=True)
            self._values[j] = unique
            self._counts[j] = np.bincount(
                inverse, weights=weights, minlength=unique.size
            ).astype(np.int64)

    def _most_frequent(self):
        out = np.full(len(self._values), np.nan)
        for j, (values, counts) in enumerate(zip(self._values, self._counts)):
            if counts.size:
                out[j] = values[np.argmax(counts)]  # first max = smallest
        return out

    def partial_fit(self, X, y=None):
        """
        Update the replacement statistics with one chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Real data; NaN marks a missing value and is excluded from the
            statistics.
        y : ignored

        Returns
        -------
        SimpleImputer
            The updated imputer (``self``).

        Raises
        ------
        TypeError
            If X is not real numeric data.
        ValueError
            If X is not 2D, contains inf, or its column
            count differs from earlier chunks.

        Notes
        -----
        Time complexity per chunk: O(n * d) for "mean"/"constant";
        O((K + n) log(K + n) * d) for "median"; O((u + n) log(u + n) * d)
        for "most_frequent", with u the number of distinct values so far.
        Space complexity: O(d), O(K * d) and O(u * d) respectively.
        """
        X = self._validate_partial_fit_input(X)
        if X.shape[0] == 0:
            return self
        missing = np.isnan(X).sum(axis=0)

        if self.strategy == "mean":
            statistics = self._stats.update_stats(X).mean
        elif self.strategy == "median":
            statistics = self._stats.update_stats(X).result()
        elif self.strategy == "most_frequent":
            self._update_counts(X)
            statistics = self._most_frequent()
        else:
            statistics = np.full(X.shape[1], float(self.fill_value))

        self._record_chunk(X)
        self.n_missing_seen_ = getattr(self, "n_missing_seen_", 0) + missing
        self.n_observed_seen_ = (
            getattr(self, "n_observed_seen_", 0) + (X.shape[0] - missing)
        )

        # A feature with no observed value yet has a NaN statistic; use the
        # fallback so transform() never leaves NaN behind.
        statistics = np.atleast_1d(np.asarray(statistics, dtype=float))
        fallback = 0.0 if self.fill_value is None else float(self.fill_value)
        self.statistics_ = np.where(np.isnan(statistics), fallback, statistics)
        return self

    def transform(self, X):
        """
        Replace NaN values with the current per-feature statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)
            Real data, NaN marking missing values.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features_in_)
            float64 copy of X with NaN replaced.

        Raises
        ------
        NotFittedError
            If neither ``fit`` nor ``partial_fit`` has been called.
        TypeError
            If X is not real numeric data.
        ValueError
            If X is not 2D, contains inf, or its column count differs from
            ``n_features_in_``.

        Notes
        -----
        Time complexity: O(n_samples * n_features).
        Space complexity: O(n_samples * n_features) for the output.
        """
        X = self._validate_transform_input(X)
        return np.where(np.isnan(X), self.statistics_, X)


# ============================================================
# OneHotEncoder
# ============================================================

def _nan_mask(column):
    """True where a float column is NaN (other dtypes have no NaN)."""
    if column.dtype.kind == "f":
        return np.isnan(column)
    return np.zeros(column.shape, dtype=bool)


def _drop_nan(column):
    return column[~_nan_mask(column)]


def _dtype_family(dtype):
    """Coarse dtype family, so integers and strings never mix silently."""
    if dtype.kind in "biuf":
        return "numeric"
    if dtype.kind in "US":
        return "string"
    return "object"


def _check_same_family(categories, values, column_index):
    """Reject a column whose values change type between chunks."""
    if categories.size == 0 or values.size == 0:
        return
    known, received = _dtype_family(categories.dtype), _dtype_family(values.dtype)
    if known != received:
        raise TypeError(
            f"Column {column_index} has {received} values, but its "
            f"categories so far are {known} (type {categories.dtype}); "
            "convert the column to one consistent type"
        )

class OneHotEncoder(_Transformer):
    """
    Encode categorical columns as one-hot indicator columns, learning the
    categories incrementally.

    Categories are **append-only**: the first chunk's categories are
    sorted (``np.unique``), and categories first seen in a later
    ``partial_fit`` are appended, sorted among themselves, after the
    existing ones. Earlier output columns therefore never move; the
    output only grows. ``fit`` on a single array gives the sorted order.

    Parameters
    ----------
    handle_unknown : {"error", "ignore"}, default="error"
        Behaviour when ``transform`` sees a category that has not been
        learned. "error" raises ValueError; "ignore" encodes that value
        as all zeros for its column.

    Attributes
    ----------
    categories_ : list of np.ndarray
        Learned categories for each input column, in output order.
    n_features_in_ : int
        Number of input columns.
    n_features_out_ : int
        Current total number of output indicator columns.
    n_samples_seen_ : int
        Total number of rows seen.

    Notes
    -----
    NaN in a float column is treated as missing: it never becomes a
    category and encodes as an all-zero block (not as "unknown").
    A column whose values change type between chunks (e.g. integers, then
    strings) raises TypeError rather than silently converting categories.

    Because the output width can grow during streaming, a model trained on
    earlier encoded chunks expects the earlier width. One Python loop runs
    over columns (each has its own categories); the work within a column
    is vectorised over samples.
    """

    _fitted_attributes = ("categories_", "n_features_in_")
    _state_attributes = (
        "categories_", "n_features_in_", "n_features_out_",
        "n_samples_seen_",
    )
    _numeric = False

    def __init__(self, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self.handle_unknown = handle_unknown

    def partial_fit(self, X, y=None):
        """
        Learn new categories from one chunk (append-only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Categorical data of any dtype (numbers or strings).
        y : ignored

        Returns
        -------
        OneHotEncoder
            The updated encoder (``self``).

        Raises
        ------
        ValueError
            If X is not 2D or its column count differs from earlier
            chunks.

        Notes
        -----
        Time complexity: O(n_features * (n + C) log(n + C)) per chunk,
        for C categories learned so far.
        Space complexity: O(C).
        """
        X = self._validate_partial_fit_input(X)
        if X.shape[0] == 0:
            return self
        previous = getattr(self, "categories_", None)

        categories = []
        for j, column in enumerate(X.T):
            seen = np.unique(_drop_nan(column))
            if previous is None:
                categories.append(seen)
                continue

            _check_same_family(previous[j], seen, j)
            fresh = seen[~np.isin(seen, previous[j])]
            categories.append(np.concatenate((previous[j], fresh)))

        self._record_chunk(X)
        self.categories_ = categories
        self.n_features_out_ = int(sum(c.size for c in categories))
        return self

    def transform(self, X):
        """
        Encode ``X`` with the categories learned so far.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)
            Categorical data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features_out_)
            Integer (0/1) indicator matrix.

        Raises
        ------
        NotFittedError
            If neither ``fit`` nor ``partial_fit`` has been called.
        ValueError
            If X is not 2D, its column count differs from
            ``n_features_in_``, or it contains an unknown category while
            ``handle_unknown="error"``.

        Notes
        -----
        Time complexity: O(n_samples * C).
        Space complexity: O(n_samples * C) for the output.
        """
        X = self._validate_transform_input(X)
        outputs = []

        for column_index, (col, cats) in enumerate(
            zip(X.T, self.categories_)
        ):
            _check_same_family(cats, col, column_index)
            onehot = (col[:, None] == cats[None, :]).astype(int)

            if self.handle_unknown == "error":
                # NaN means "missing" (an all-zero row), not an unknown value.
                unknown = ~onehot.any(axis=1) & ~_nan_mask(col)
                if np.any(unknown):
                    raise ValueError(
                        f"Found unknown categories "
                        f"{np.unique(col[unknown]).tolist()} in column "
                        f"{column_index} during transform"
                    )

            outputs.append(onehot)

        if not outputs:
            return np.zeros((X.shape[0], 0), dtype=int)

        return np.hstack(outputs)


__all__ = ["StandardScaler", "MinMaxScaler", "SimpleImputer", "OneHotEncoder"]
