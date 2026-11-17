
"""
utils.py
========

Shared utility functions for the DeepLearners Toolkit.

Purpose
-------
Provide reusable helpers for numerical array validation,
shape checking, finite-value validation, safe division and
array metadata.

Dependencies
------------
Python standard library and NumPy only.

Design principles
-----------------
1. Keep functions small and reusable.
2. Raise clear exceptions for invalid inputs.
3. Do not silently change numerical data.
4. Return predictable NumPy arrays or Python values.
5. Avoid duplicating domain-specific logic from other modules.
"""

import numpy as np


# ============================================================
# 1. NUMERIC ARRAY CONVERSION
# ============================================================

def as_numeric_array(
    values,
    name="values",
    dtype=None,
    allow_complex=False,
):
    """Convert supported input into a validated NumPy numeric array."""
    try:
        # Validate the original input before any dtype conversion.
        arr = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be convertible to a numeric array"
        ) from exc

    # Reject strings, objects, Boolean arrays and nonnumeric types.
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")

    if (
        not allow_complex
        and np.issubdtype(arr.dtype, np.complexfloating)
    ):
        raise TypeError(
            f"{name} must contain real values, not complex values"
        )

    # Convert only after validating the original input.
    if dtype is not None:
        try:
            arr = arr.astype(dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"{name} cannot be converted to dtype {dtype}"
            ) from exc

        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values")

        if (
            not allow_complex
            and np.issubdtype(arr.dtype, np.complexfloating)
        ):
            raise TypeError(
                f"{name} must contain real values, not complex values"
            )

    return arr


# ============================================================
# 2. GENERAL ARRAY VALIDATION
# ============================================================

def validate_array(
    values,
    name="values",
    ndim=None,
    allow_empty=True,
    allow_nan=True,
    allow_inf=True,
    allow_complex=False,
):
    """
    Validate an array's numeric type, shape and value constraints.

    Parameters
    ----------
    values : array-like
        Input data.

    name : str
        Name used in error messages.

    ndim : int or None
        Required number of dimensions. None means any dimension.

    allow_empty : bool
        Whether arrays with zero elements are permitted.

    allow_nan : bool
        Whether NaN values are permitted.

    allow_inf : bool
        Whether positive and negative infinity are permitted.

    allow_complex : bool
        Whether complex-valued numeric data are permitted.

    Returns
    -------
    np.ndarray
        The validated array, without unnecessary dtype conversion.

    Raises
    ------
    TypeError
        If the array type is invalid.
    ValueError
        If dimensionality or value constraints are violated.
    """
    if ndim is not None:
        if (
            isinstance(ndim, (bool, np.bool_))
            or not isinstance(ndim, (int, np.integer))
            or ndim < 0
        ):
            raise ValueError("ndim must be a nonnegative integer")

    arr = as_numeric_array(
        values,
        name=name,
        allow_complex=allow_complex,
    )

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions; "
            f"received {arr.ndim}"
        )

    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must not be empty")

    # NaN and infinity checks apply to floating-point arrays.
    # Complex arrays are also checked when explicitly allowed.
    if np.issubdtype(arr.dtype, np.inexact):
        if not allow_nan and np.any(np.isnan(arr)):
            raise ValueError(f"{name} must not contain NaN values")

        if not allow_inf and np.any(np.isinf(arr)):
            raise ValueError(
                f"{name} must not contain infinite values"
            )

    return arr


# ============================================================
# 3. FINITE-VALUE VALIDATION
# ============================================================

def check_finite(values, name="values"):
    """
    Require all array values to be finite.

    Finite means that no element is NaN, positive infinity or
    negative infinity.

    Returns
    -------
    np.ndarray
        The validated array.
    """
    arr = as_numeric_array(
        values,
        name=name,
        allow_complex=False,
    )

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return arr


# ============================================================
# 4. SHAPE VALIDATION
# ============================================================

def validate_matching_shapes(*arrays, names=None):
    """
    Validate that two or more arrays have exactly matching shapes.

    Useful when comparing actual and predicted values, checking
    corresponding numerical arrays, or validating elementwise
    calculations.

    Parameters
    ----------
    *arrays
        Two or more array-like objects.

    names : sequence of str or None
        Optional names for clearer error messages.

    Returns
    -------
    tuple of np.ndarray
        Validated arrays in their original order.

    Raises
    ------
    ValueError
        If fewer than two arrays are provided or shapes differ.
    TypeError
        If any input is not numeric.

    Examples
    --------
    >>> a, b = validate_matching_shapes([1, 2], [3, 4])
    >>> a.shape == b.shape
    True
    """
    if len(arrays) < 2:
        raise ValueError(
            "At least two arrays are required for shape comparison"
        )

    if names is None:
        names = [f"array_{i + 1}" for i in range(len(arrays))]
    elif len(names) != len(arrays):
        raise ValueError(
            "names must contain one name per input array"
        )

    converted = tuple(
        as_numeric_array(arr, name=name)
        for arr, name in zip(arrays, names)
    )

    expected_shape = converted[0].shape

    for name, arr in zip(names[1:], converted[1:]):
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {arr.shape}; "
                f"expected {expected_shape}"
            )

    return converted


# ============================================================
# 5. SAFE DIVISION
# ============================================================

def safe_divide(numerator, denominator, zero_policy="raise"):
    """
    Divide numerical values with explicit zero-denominator handling.

    Parameters
    ----------
    numerator : array-like
        Numerator values.

    denominator : array-like
        Denominator values. NumPy broadcasting is supported.

    zero_policy : str, default="raise"
        "raise":
            Raise ZeroDivisionError if any denominator is zero.

        "nan":
            Return NaN wherever the denominator is zero.

        "zero":
            Return zero wherever the denominator is zero.

    Returns
    -------
    np.ndarray
        The division result. The output shape follows NumPy's
        broadcasting rules.

    Raises
    ------
    ValueError
        If zero_policy is unknown or the shapes do not broadcast.
    TypeError
        If either input is not real numeric.
    ZeroDivisionError
        If zero_policy="raise" and any denominator is zero.

    Notes
    -----
    Nonzero divisions use normal NumPy arithmetic. This function
    does not silently replace NaNs or infinities produced by
    other arithmetic operations.

    Time complexity: O(n) for n broadcast elements.
    Space complexity: O(n) for the float64 output.
    """
    if zero_policy not in {"raise", "nan", "zero"}:
        raise ValueError(
            "zero_policy must be 'raise', 'nan' or 'zero'"
        )

    num = as_numeric_array(numerator, name="numerator")
    den = as_numeric_array(denominator, name="denominator")

    try:
        num, den = np.broadcast_arrays(num, den)
    except ValueError as exc:
        raise ValueError(
            "numerator and denominator shapes are not broadcastable"
        ) from exc

    zero_mask = den == 0

    if zero_policy == "raise" and np.any(zero_mask):
        raise ZeroDivisionError(
            "Division by zero encountered in denominator"
        )

    # Use floating-point output so NaN can be represented.
    # Complex inputs are rejected by the general array validator.
    try:
        output = np.full(num.shape, np.nan, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Inputs must support real numerical division"
        ) from exc

    nonzero_mask = ~zero_mask

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(
            num,
            den,
            out=output,
            where=nonzero_mask,
            casting="unsafe",
        )

    if zero_policy == "zero":
        output[zero_mask] = 0.0

    return output


# ============================================================
# 6. ARRAY INFORMATION
# ============================================================

def get_array_info(values, name="values"):
    """
    Return basic metadata about a numeric array.

    This is useful for debugging, logging and inspecting the
    numerical data used by the toolkit.

    Returns
    -------
    dict
        Contains name, shape, ndim, size, dtype and nbytes.
    """
    arr = as_numeric_array(values, name=name, allow_complex=True)

    return {
        "name": name,
        "shape": arr.shape,
        "ndim": arr.ndim,
        "size": arr.size,
        "dtype": str(arr.dtype),
        "nbytes": arr.nbytes,
    }


# ============================================================
# 7. SCALAR ARGUMENT VALIDATION
# ============================================================

def validate_bool(value, name):
    """
    Require a Boolean flag.

    Parameters
    ----------
    value : object
        Value to check. Python ``bool`` and ``np.bool_`` are accepted.
    name : str
        Argument name used in error messages.

    Returns
    -------
    bool
        The value converted to a Python ``bool``.

    Raises
    ------
    TypeError
        If value is not a Boolean. Integers such as 0 and 1 are
        rejected so that flags cannot be confused with counts.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def validate_int(value, name, minimum=None, maximum=None):
    """
    Require an integer within optional inclusive bounds.

    Parameters
    ----------
    value : object
        Value to check. Python ``int`` and NumPy integers are accepted;
        ``bool`` is rejected.
    name : str
        Argument name used in error messages.
    minimum, maximum : int or None
        Inclusive lower and upper bounds. None means unbounded.

    Returns
    -------
    int
        The value converted to a Python ``int``.

    Raises
    ------
    TypeError
        If value is not an integer.
    ValueError
        If value lies outside ``[minimum, maximum]``.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")

    value = int(value)

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {value}")

    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}; received {value}")

    return value


# ============================================================
# 8. FEATURE-MATRIX AND FITTED-STATE VALIDATION
# ============================================================

class NotFittedError(RuntimeError, AttributeError):
    """
    Raised when a component is used before ``fit`` has been called.

    Subclasses both RuntimeError and AttributeError so callers can catch
    it with either convention.
    """


def check_is_fitted(component, attributes):
    """
    Require that a component has been fitted.

    Parameters
    ----------
    component : object
        Transformer or estimator instance.
    attributes : str or sequence of str
        Attribute names that ``fit`` creates. All must exist.

    Raises
    ------
    NotFittedError
        If any of the attributes is missing.
    """
    if isinstance(attributes, str):
        attributes = (attributes,)

    missing = [attr for attr in attributes if not hasattr(component, attr)]

    if missing:
        raise NotFittedError(
            f"This {type(component).__name__} instance is not fitted yet. "
            "Call fit() before using this method."
        )


def validate_2d(
    X,
    name="X",
    n_features=None,
    numeric=True,
    allow_nan=False,
    allow_inf=False,
    allow_empty=False,
):
    """
    Validate a two-dimensional feature matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input matrix.
    name : str, default="X"
        Name used in error messages.
    n_features : int or None, default=None
        Required number of columns, usually the ``n_features_in_``
        recorded during fitting. None skips the check.
    numeric : bool, default=True
        If True, require real numeric data and return a float array.
        If False, any dtype (e.g. string categories) is accepted.
    allow_nan, allow_inf : bool, default=False
        Whether NaN or infinite values are permitted (numeric only).
    allow_empty : bool, default=False
        Whether zero rows are permitted.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Float64 array when ``numeric`` is True, otherwise the input
        converted with ``np.asarray``.

    Raises
    ------
    TypeError
        If ``numeric`` is True and X is not real numeric data.
    ValueError
        If X is not 2D, is empty, contains disallowed values, or its
        column count differs from ``n_features``.

    Notes
    -----
    Time complexity: O(n_samples * n_features) for the value checks.
    """
    if numeric:
        arr = validate_array(
            X,
            name=name,
            ndim=2,
            allow_nan=allow_nan,
            allow_inf=allow_inf,
        ).astype(float, copy=False)
    else:
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must have 2 dimensions; received {arr.ndim}"
            )

    if not allow_empty and arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")

    if n_features is not None and arr.shape[1] != n_features:
        raise ValueError(
            f"{name} has {arr.shape[1]} features, but this component "
            f"was fitted with {n_features} features"
        )

    return arr


# ============================================================
# 9. ACTIVATION FUNCTIONS (numerically stable)
# ============================================================
#
# All activations are elementwise unless an axis is given, accept any
# real array-like (including non-contiguous views), return float64
# arrays of the same shape as the input, and propagate NaN.

def _as_float(x, name="x", allow_empty=True):
    return validate_array(
        x, name=name, allow_empty=allow_empty
    ).astype(float, copy=False)


def sigmoid(x):
    """
    Logistic sigmoid ``1 / (1 + exp(-x))``, computed without overflow.

    Uses ``1 / (1 + exp(-x))`` for ``x >= 0`` and ``exp(x) / (1 + exp(x))``
    for ``x < 0`` so that ``exp`` is only ever called on non-positive
    numbers. ``sigmoid(+inf) = 1`` and ``sigmoid(-inf) = 0``.

    Parameters
    ----------
    x : array-like of any shape
        Real input values.

    Returns
    -------
    np.ndarray
        float64 array with the shape of ``x``, values in [0, 1].

    Raises
    ------
    TypeError
        If x is not real numeric data.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(n).
    """
    x = _as_float(x)
    out = np.empty_like(x)
    positive = x >= 0
    negative = ~positive  # includes NaN, which then propagates

    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1.0 + exp_x)
    return out


def relu(x):
    """
    Rectified linear unit ``max(x, 0)``.

    Parameters
    ----------
    x : array-like of any shape
        Real input values.

    Returns
    -------
    np.ndarray
        float64 array with the shape of ``x``. NaN propagates.

    Raises
    ------
    TypeError
        If x is not real numeric data.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(n).
    """
    return np.maximum(_as_float(x), 0.0)


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU: ``x`` for ``x >= 0`` and ``alpha * x`` otherwise.

    Parameters
    ----------
    x : array-like of any shape
        Real input values.
    alpha : float, default=0.01
        Non-negative slope for negative inputs.

    Returns
    -------
    np.ndarray
        float64 array with the shape of ``x``.

    Raises
    ------
    TypeError
        If x is not real numeric data.
    ValueError
        If alpha is negative or not finite.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(n).
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite, non-negative number")
    x = _as_float(x)
    return np.where(x >= 0, x, alpha * x)


def tanh(x):
    """
    Hyperbolic tangent (NumPy's implementation is already stable).

    Parameters
    ----------
    x : array-like of any shape
        Real input values.

    Returns
    -------
    np.ndarray
        float64 array with the shape of ``x``, values in [-1, 1].

    Raises
    ------
    TypeError
        If x is not real numeric data.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(n).
    """
    return np.tanh(_as_float(x))


def logsumexp(x, axis=None, keepdims=False):
    """
    Compute ``log(sum(exp(x)))`` without overflow or underflow.

    Uses the max-shift identity
    ``logsumexp(x) = m + log(sum(exp(x - m)))`` with ``m = max(x)``,
    so the largest exponent evaluated is ``exp(0) = 1``.

    Parameters
    ----------
    x : array-like
        Real, non-empty input.
    axis : int, tuple of int or None, default=None
        Axis or axes to reduce. None reduces over all elements.
    keepdims : bool, default=False
        Keep reduced axes with length 1 (useful for broadcasting).

    Returns
    -------
    float or np.ndarray
        Reduced result. A Python float when ``axis`` is None and
        ``keepdims`` is False, otherwise an array with the reduced shape.

    Special values
    --------------
    All ``-inf`` along the axis gives ``-inf``; any ``+inf`` gives
    ``+inf``; any NaN gives NaN.

    Raises
    ------
    TypeError
        If x is not real numeric data.
    ValueError
        If x is empty.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(n) for the shifted copy.

    Examples
    --------
    >>> logsumexp([1000.0, 1000.0])   # naive version overflows to inf
    1000.6931471805599
    """
    x = _as_float(x, allow_empty=False)

    m = np.max(x, axis=axis, keepdims=True)
    # Shifting by an infinite max would create inf - inf = NaN.
    m_safe = np.where(np.isfinite(m), m, 0.0)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        summed = np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)
        out = np.log(summed) + m_safe

    if not keepdims:
        out = np.squeeze(out, axis=axis) if axis is not None else out.reshape(())

    return float(out) if out.ndim == 0 else out


def softmax(x, axis=-1):
    """
    Softmax ``exp(x) / sum(exp(x))`` along an axis, using a max shift.

    Parameters
    ----------
    x : array-like
        Real, non-empty input, e.g. logits of shape (n_samples, n_classes).
    axis : int, default=-1
        Axis along which probabilities sum to 1.

    Returns
    -------
    np.ndarray
        float64 array with the shape of ``x``; entries in [0, 1] and each
        slice along ``axis`` sums to 1. ``-inf`` logits get probability 0.

    Raises
    ------
    TypeError
        If x is not real numeric data.
    ValueError
        If x is empty.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(n).
    """
    x = _as_float(x, allow_empty=False)
    with np.errstate(invalid="ignore"):
        return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def log_softmax(x, axis=-1):
    """
    Logarithm of softmax, ``x - logsumexp(x)``, computed stably.

    More accurate than ``np.log(softmax(x))``, which underflows to
    ``-inf`` for very negative logits.

    Parameters
    ----------
    x : array-like
        Real, non-empty input.
    axis : int, default=-1
        Axis along which to normalise.

    Returns
    -------
    np.ndarray
        float64 array with the shape of ``x``; all entries <= 0.

    Raises
    ------
    TypeError
        If x is not real numeric data.
    ValueError
        If x is empty.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(n).
    """
    x = _as_float(x, allow_empty=False)
    with np.errstate(invalid="ignore"):
        return x - logsumexp(x, axis=axis, keepdims=True)


# ============================================================
# 10. DISTANCES
# ============================================================

_DISTANCE_METRICS = ("euclidean", "manhattan", "cosine")


def _paired_vectors(a, b):
    a = _as_float(a, name="a", allow_empty=False)
    b = _as_float(b, name="b", allow_empty=False)

    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("a and b must have at least one dimension")

    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"a and b must have the same last dimension; "
            f"received {a.shape[-1]} and {b.shape[-1]}"
        )

    try:
        np.broadcast_shapes(a.shape, b.shape)
    except ValueError as exc:
        raise ValueError(
            f"a and b shapes {a.shape} and {b.shape} are not broadcastable"
        ) from exc

    return a, b


def euclidean_distance(a, b):
    """
    Euclidean (L2) distance between vectors along the last axis.

    Parameters
    ----------
    a, b : array-like of shape (..., d)
        Real vectors. Leading dimensions broadcast, so a single vector
        can be compared with every row of a matrix.

    Returns
    -------
    float or np.ndarray of shape (...)
        A Python float for two 1D vectors, otherwise one distance per
        broadcast row.

    Raises
    ------
    TypeError
        If inputs are not real numeric data.
    ValueError
        If inputs are empty or scalar, the last dimensions differ, or the
        leading dimensions do not broadcast.

    Notes
    -----
    Time complexity: O(n * d). Space complexity: O(n * d).
    """
    a, b = _paired_vectors(a, b)
    out = np.sqrt(np.sum((a - b) ** 2, axis=-1))
    return float(out) if out.ndim == 0 else out


def manhattan_distance(a, b):
    """
    Manhattan (L1) distance between vectors along the last axis.

    Parameters and broadcasting are as for ``euclidean_distance``.

    Returns
    -------
    float or np.ndarray of shape (...)
        Sum of absolute differences per broadcast row.

    Raises
    ------
    TypeError, ValueError
        As for ``euclidean_distance``.

    Notes
    -----
    Time complexity: O(n * d). Space complexity: O(n * d).
    """
    a, b = _paired_vectors(a, b)
    out = np.sum(np.abs(a - b), axis=-1)
    return float(out) if out.ndim == 0 else out


def cosine_distance(a, b):
    """
    Cosine distance ``1 - (a . b) / (||a|| ||b||)`` along the last axis.

    Parameters and broadcasting are as for ``euclidean_distance``.

    Returns
    -------
    float or np.ndarray of shape (...)
        Values in [0, 2] (clipped to absorb rounding error).

    Raises
    ------
    TypeError
        If inputs are not real numeric data.
    ValueError
        As for ``euclidean_distance``, or if any vector has zero norm
        (cosine distance is undefined for the zero vector).

    Notes
    -----
    Time complexity: O(n * d). Space complexity: O(n * d).
    """
    a, b = _paired_vectors(a, b)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)

    if np.any(norm_a == 0) or np.any(norm_b == 0):
        raise ValueError("cosine distance is undefined for zero vectors")

    similarity = np.sum(a * b, axis=-1) / (norm_a * norm_b)
    out = 1.0 - np.clip(similarity, -1.0, 1.0)
    return float(out) if out.ndim == 0 else out


def pairwise_distances(X, Y=None, metric="euclidean"):
    """
    Distance between every row of X and every row of Y.

    Parameters
    ----------
    X : array-like of shape (n, d)
        First set of row vectors.
    Y : array-like of shape (m, d) or None, default=None
        Second set of row vectors. None means ``Y = X``.
    metric : {"euclidean", "manhattan", "cosine"}, default="euclidean"
        Distance to compute.

    Returns
    -------
    np.ndarray of shape (n, m)
        ``D[i, j]`` is the distance between ``X[i]`` and ``Y[j]``.

    Raises
    ------
    TypeError
        If inputs are not real numeric data.
    ValueError
        If inputs are not 2D, are empty, have different column counts,
        contain NaN/inf, the metric is unknown, or (cosine) a row has
        zero norm.

    Notes
    -----
    Euclidean uses the expansion ``||x||^2 + ||y||^2 - 2 x.y`` (one matrix
    product, no (n, m, d) temporary) and clips tiny negative rounding
    errors to 0 before the square root. Manhattan broadcasts to an
    (n, m, d) temporary.

    Time complexity: O(n * m * d).
    Space complexity: O(n * m) for euclidean/cosine, O(n * m * d) for
    manhattan.
    """
    if metric not in _DISTANCE_METRICS:
        raise ValueError(
            f"metric must be one of {', '.join(_DISTANCE_METRICS)}; "
            f"received {metric!r}"
        )

    X = validate_2d(X, name="X")
    Y = X if Y is None else validate_2d(Y, name="Y", n_features=X.shape[1])

    if metric == "manhattan":
        return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=-1)

    if metric == "cosine":
        norm_x = np.linalg.norm(X, axis=1)
        norm_y = np.linalg.norm(Y, axis=1)
        if np.any(norm_x == 0) or np.any(norm_y == 0):
            raise ValueError("cosine distance is undefined for zero vectors")
        similarity = (X @ Y.T) / np.outer(norm_x, norm_y)
        return 1.0 - np.clip(similarity, -1.0, 1.0)

    squared = (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(Y ** 2, axis=1)[None, :]
        - 2.0 * (X @ Y.T)
    )
    return np.sqrt(np.maximum(squared, 0.0))


# ============================================================
# 11. BATCHING
# ============================================================

def batch_iterator(*arrays, batch_size, shuffle=False, seed=None,
                   drop_last=False):
    """
    Yield aligned mini-batches from one or more arrays.

    Parameters
    ----------
    *arrays : array-like
        One or more arrays with the same length along axis 0
        (e.g. ``X`` of shape (n, d) and ``y`` of shape (n,)).
    batch_size : int
        Number of samples per batch, at least 1.
    shuffle : bool, default=False
        Shuffle sample order once before batching.
    seed : int or None, default=None
        Seed for the shuffle, for reproducibility.
    drop_last : bool, default=False
        Drop the final batch if it has fewer than ``batch_size`` samples.

    Yields
    ------
    np.ndarray or tuple of np.ndarray
        A single array when one array is given, otherwise a tuple of
        batches in argument order. Each has ``batch_size`` rows (the last
        may be smaller unless ``drop_last`` is True).

    Raises
    ------
    ValueError
        If no arrays are given, any is 0-dimensional, or their lengths
        differ.
    TypeError
        If batch_size is not an integer or a flag is not Boolean.

    Notes
    -----
    Without shuffling, batches are views (no copy).

    Time complexity: O(n) over all batches.
    Space complexity: O(1) per batch without shuffling (views); O(n)
    for the permutation plus O(batch_size) per copied batch when
    shuffling.
    """
    if not arrays:
        raise ValueError("at least one array is required")

    batch_size = validate_int(batch_size, "batch_size", minimum=1)
    shuffle = validate_bool(shuffle, "shuffle")
    drop_last = validate_bool(drop_last, "drop_last")

    arrays = [np.asarray(a) for a in arrays]

    if any(a.ndim == 0 for a in arrays):
        raise ValueError("arrays must have at least one dimension")

    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise ValueError("all arrays must have the same length along axis 0")

    order = (
        np.random.default_rng(seed).permutation(n) if shuffle else None
    )
    stop = n - n % batch_size if drop_last else n

    for start in range(0, stop, batch_size):
        if order is None:
            batch = tuple(a[start:start + batch_size] for a in arrays)
        else:
            index = order[start:start + batch_size]
            batch = tuple(a[index] for a in arrays)

        yield batch[0] if len(batch) == 1 else batch


# ============================================================
# 12. PUBLIC API
# ============================================================

__all__ = [
    "as_numeric_array",
    "validate_array",
    "check_finite",
    "validate_matching_shapes",
    "safe_divide",
    "get_array_info",
    "validate_bool",
    "validate_int",
    "NotFittedError",
    "check_is_fitted",
    "validate_2d",
    "sigmoid",
    "relu",
    "leaky_relu",
    "tanh",
    "softmax",
    "log_softmax",
    "logsumexp",
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "pairwise_distances",
    "batch_iterator",
]
