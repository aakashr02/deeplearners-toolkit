"""
preprocessing.py
================

Feature scaling and categorical encoding for NumCompute.

Every transformer follows the same contract:

* ``fit(X)`` learns per-feature state from a 2D array of shape
  ``(n_samples, n_features)``, records ``n_features_in_`` and returns
  ``self``.
* ``transform(X)`` requires a prior ``fit`` (otherwise
  ``NotFittedError``) and an ``X`` with exactly ``n_features_in_``
  columns (otherwise ``ValueError``). The input is never modified.
* ``fit_transform(X)`` is equivalent to ``fit(X).transform(X)``.

Numeric scalers reject NaN and infinite values, because they would
silently corrupt the learned statistics. Constant features are scaled
with a divisor of 1 instead of 0, so they never produce NaN or inf.
"""

import numpy as np

from numcompute.utils import check_is_fitted, validate_2d


class _Transformer:
    """Shared ``fit_transform`` and input checks for all transformers."""

    # Attributes created by fit(); used by check_is_fitted().
    _fitted_attributes = ("n_features_in_",)
    _numeric = True

    def _validate_fit_input(self, X):
        return validate_2d(X, numeric=self._numeric)

    def _validate_transform_input(self, X):
        check_is_fitted(self, self._fitted_attributes)
        return validate_2d(
            X,
            numeric=self._numeric,
            n_features=self.n_features_in_,
            allow_empty=True,
        )

    def fit_transform(self, X):
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


class StandardScaler(_Transformer):
    """
    Standardise features by removing the mean and scaling to unit variance.

    ``z = (x - mean_) / scale_`` where ``scale_`` is the population
    standard deviation (ddof=0), with zero replaced by 1.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean of the training data.
    std_ : np.ndarray of shape (n_features,)
        Per-feature population standard deviation (may contain 0).
    scale_ : np.ndarray of shape (n_features,)
        Divisor used by ``transform``: ``std_`` with zeros replaced by 1,
        so a constant feature is transformed to 0 rather than NaN.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    _fitted_attributes = ("mean_", "scale_", "n_features_in_")

    def fit(self, X):
        """
        Compute the per-feature mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Real, finite training data with at least one sample.

        Returns
        -------
        StandardScaler
            The fitted scaler (``self``).

        Raises
        ------
        TypeError
            If X is not real numeric data.
        ValueError
            If X is not 2D, has no samples, or contains NaN/inf.

        Notes
        -----
        Time complexity: O(n_samples * n_features).
        Space complexity: O(n_features) for the stored statistics.
        """
        X = self._validate_fit_input(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.scale_ = _safe_scale(self.std_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Standardise ``X`` using the fitted mean and scale.

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
            If ``fit`` has not been called.
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


class MinMaxScaler(_Transformer):
    """
    Scale features linearly to ``feature_range`` (default ``(0, 1)``).

    ``x_scaled = (x - data_min_) / scale_ * (max - min) + min`` where
    ``scale_`` is ``data_max_ - data_min_`` with zero replaced by 1.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired output range. ``min`` must be strictly less than ``max``.

    Attributes
    ----------
    data_min_, data_max_ : np.ndarray of shape (n_features,)
        Per-feature minimum and maximum of the training data.
    data_range_ : np.ndarray of shape (n_features,)
        ``data_max_ - data_min_`` (may contain 0).
    scale_ : np.ndarray of shape (n_features,)
        Divisor used by ``transform``: ``data_range_`` with zeros replaced
        by 1, so a constant feature maps to ``feature_range[0]``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Raises
    ------
    ValueError
        If ``feature_range`` is not two finite numbers with min < max.
    """

    _fitted_attributes = ("data_min_", "scale_", "n_features_in_")

    def __init__(self, feature_range=(0, 1)):
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

        self.feature_range = (min_val, max_val)
        self.min_val, self.max_val = min_val, max_val

    def fit(self, X):
        """
        Compute the per-feature minimum and maximum.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Real, finite training data with at least one sample.

        Returns
        -------
        MinMaxScaler
            The fitted scaler (``self``).

        Raises
        ------
        TypeError
            If X is not real numeric data.
        ValueError
            If X is not 2D, has no samples, or contains NaN/inf.

        Notes
        -----
        Time complexity: O(n_samples * n_features).
        Space complexity: O(n_features).
        """
        X = self._validate_fit_input(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = _safe_scale(self.data_range_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Scale ``X`` to ``feature_range`` using the fitted min and range.

        Values outside the training range are scaled linearly and are not
        clipped, so they may fall outside ``feature_range``.

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
            If ``fit`` has not been called.
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
        return scaled * (self.max_val - self.min_val) + self.min_val


class OneHotEncoder(_Transformer):
    """
    Encode categorical columns as one-hot indicator columns.

    Categories are learned per column with ``np.unique`` (sorted order).
    Output columns are grouped by input column, in category order.

    Parameters
    ----------
    handle_unknown : {"error", "ignore"}, default="error"
        Behaviour when ``transform`` sees a category absent during
        ``fit``. "error" raises ValueError; "ignore" encodes that value
        as all zeros for its column.

    Attributes
    ----------
    categories_ : list of np.ndarray
        Sorted unique categories for each input column.
    n_features_in_ : int
        Number of input columns seen during ``fit``.
    n_features_out_ : int
        Total number of output indicator columns.
    """

    _fitted_attributes = ("categories_", "n_features_in_")
    _numeric = False

    def __init__(self, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self.handle_unknown = handle_unknown

    def fit(self, X):
        """
        Learn the categories of every column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Categorical data of any dtype (numbers or strings).

        Returns
        -------
        OneHotEncoder
            The fitted encoder (``self``).

        Raises
        ------
        ValueError
            If X is not 2D or has no samples.

        Notes
        -----
        Time complexity: O(n_features * n_samples log n_samples).
        Space complexity: O(C), where C is the total number of categories.
        """
        X = self._validate_fit_input(X)
        self.categories_ = [np.unique(col) for col in X.T]
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = int(sum(c.size for c in self.categories_))
        return self

    def transform(self, X):
        """
        Encode ``X`` with the fitted categories.

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
            If ``fit`` has not been called.
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
            onehot = (col[:, None] == cats[None, :]).astype(int)

            if self.handle_unknown == "error":
                unknown = ~onehot.any(axis=1)
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


__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder"]
