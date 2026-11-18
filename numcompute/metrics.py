"""
metrics.py
==========

Classification and regression metrics for NumCompute.

* One-shot functions (``accuracy``, ``precision``, ``recall``, ``f1``,
  ``confusion_matrix``, ``mse``, ``roc_curve``, ``auc``,
  ``roc_auc_score``) evaluate complete arrays.
* Streaming metric classes (``StreamingAccuracy``, ``StreamingPrecision``,
  ``StreamingRecall``, ``StreamingF1``, ``StreamingMSE``,
  ``StreamingConfusionMatrix``, ``StreamingAUC``) accumulate over chunks
  with ``update(y_true_chunk, y_pred_chunk)``, report with ``result()``
  and start over with ``reset()``. Passing ``window=W`` turns any of them
  into a rolling-window metric over the most recent W chunks.

Streaming metrics store only sufficient statistics per chunk (counts or
sums), never the raw labels, so their memory does not grow with the
stream (except ``StreamingAUC``, which keeps one count per distinct score).
"""

from collections import deque

import numpy as np

from numcompute.utils import validate_int


def _validate_pair(y_true, y_pred, numeric=False):
    """
    Shared input contract for every metric.

    Both inputs are converted with ``np.asarray`` (float64 when
    ``numeric`` is True), must have identical shapes and must not be
    empty. Labels may be of any comparable dtype when ``numeric`` is
    False.

    Raises
    ------
    ValueError
        If shapes differ or the inputs are empty.
    """
    dtype = float if numeric else None
    y_true = np.asarray(y_true, dtype=dtype)
    y_pred = np.asarray(y_pred, dtype=dtype)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")

    return y_true, y_pred


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Proportion of correct predictions.

    Raises
    ------
    ValueError
        If the shapes of y_true and y_pred do not match.
        or contain NaN values.
        
    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(1)
    """
    y_true, y_pred = _validate_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute the confusion matrix for classification.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    labels : array-like or None, optional
        Class labels to include. If None, labels are
        determined automatically from y_true and y_pred.

    Returns
    -------
    np.ndarray
        Confusion matrix where rows represent true labels
        and columns represent predicted labels.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes
        or are empty.

    Notes
    -----
    Time complexity: O(n * c), where c is the number of classes.
    Space complexity: O(c^2).
    """
    y_true, y_pred = _validate_pair(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true.ravel(), y_pred.ravel())))
    else:
        labels = np.asarray(labels)

    y_true, y_pred = y_true.ravel(), y_pred.ravel()

    # Vectorised counting for each true/predicted label combination
    true_matches = y_true[:, None] == labels
    pred_matches = y_pred[:, None] == labels

    matrix = true_matches.astype(int).T @ pred_matches.astype(int)

    return matrix

def precision(y_true, y_pred, positive_label=1):
    """
    Calculate precision for binary classification.

    Precision = TP / (TP + FP)

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : int or str, default=1
        Label considered as the positive class.

    Returns
    -------
    float
        Precision score. Returns 0.0 if there are
        no predicted positive samples.

    Raises
    ------
    ValueError
        If input shapes differ or inputs are empty.

    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(n)
    """
    y_true, y_pred = _validate_pair(y_true, y_pred)

    true_positive = np.sum(
        (y_true == positive_label) & (y_pred == positive_label)
    )

    false_positive = np.sum(
        (y_true != positive_label) & (y_pred == positive_label)
    )

    denominator = true_positive + false_positive

    if denominator == 0:
        return 0.0

    return float(true_positive / denominator)


def recall(y_true, y_pred, positive_label=1):
    """
    Calculate recall for binary classification.

    Recall = TP / (TP + FN)

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : int or str, default=1
        Label considered as the positive class.

    Returns
    -------
    float
        Recall score. Returns 0.0 if there are
        no actual positive samples.

    Raises
    ------
    ValueError
        If input shapes differ or inputs are empty.

    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(n)
    """
    y_true, y_pred = _validate_pair(y_true, y_pred)

    true_positive = np.sum(
        (y_true == positive_label) & (y_pred == positive_label)
    )

    false_negative = np.sum(
        (y_true == positive_label) & (y_pred != positive_label)
    )

    denominator = true_positive + false_negative

    if denominator == 0:
        return 0.0

    return float(true_positive / denominator)


def f1(y_true, y_pred, positive_label=1):
    """
    Calculate the F1 score for binary classification.

    F1 is the harmonic mean of precision and recall.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : int or str, default=1
        Label considered as the positive class.

    Returns
    -------
    float
        F1 score. Returns 0.0 when precision and recall are both 0.

    Raises
    ------
    ValueError
        If input shapes differ or inputs are empty.

    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(n)
    """
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)

    if p + r == 0:
        return 0.0

    return float(2 * p * r / (p + r))

def mse(y_true, y_pred):
    """
    Calculate Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        True numerical values.
    y_pred : array-like
        Predicted numerical values.

    Returns
    -------
    float
        Mean squared error.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes
        or contain no values.

    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(n)
    """
    y_true, y_pred = _validate_pair(y_true, y_pred, numeric=True)
    return float(np.mean((y_true - y_pred) ** 2))


# ============================================================
# ROC curve and AUC
# ============================================================

def _score_counts(y_true, y_score, positive_label):
    """
    Distinct scores (ascending) with the number of positive and negative
    samples at each. Ties share one entry, which is what makes tied scores
    count as half-correct in the AUC.
    """
    y_true, y_score = _validate_pair(y_true, y_score)
    y_score = np.asarray(y_score, dtype=float).ravel()
    is_positive = (y_true.ravel() == positive_label).astype(float)

    scores, inverse = np.unique(y_score, return_inverse=True)
    positives = np.bincount(inverse, weights=is_positive,
                            minlength=scores.size)
    totals = np.bincount(inverse, minlength=scores.size).astype(float)
    return scores, positives, totals - positives


def _roc_from_counts(scores, positives, negatives):
    """ROC points from per-score counts, sweeping thresholds high to low."""
    tps = np.cumsum(positives[::-1])
    fps = np.cumsum(negatives[::-1])
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    thresholds = np.r_[np.inf, scores[::-1]]
    return fpr, tpr, thresholds


def roc_curve(y_true, y_score, positive_label=1):
    """
    Receiver operating characteristic curve for binary classification.

    Every distinct score is used as a threshold (predict positive when
    ``score >= threshold``), swept from high to low.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_score : array-like of shape (n_samples,)
        Scores or probabilities; higher means more likely positive.
    positive_label : int or str, default=1
        Label of the positive class. Every other label is negative.

    Returns
    -------
    fpr, tpr : np.ndarray of shape (n_thresholds + 1,)
        False- and true-positive rates, starting at (0, 0) and ending at
        (1, 1).
    thresholds : np.ndarray of shape (n_thresholds + 1,)
        Decreasing thresholds; the first is +inf.

    Raises
    ------
    ValueError
        If shapes differ or the inputs are empty.

    Notes
    -----
    Time complexity: O(n log n). Space complexity: O(n).
    """
    return _roc_from_counts(*_score_counts(y_true, y_score, positive_label))


def auc(x, y):
    """
    Area under a curve by the trapezoidal rule.

    Parameters
    ----------
    x : array-like of shape (n_points,)
        Monotonic x coordinates (e.g. ``fpr``).
    y : array-like of shape (n_points,)
        y coordinates (e.g. ``tpr``).

    Returns
    -------
    float
        Area under the curve.

    Raises
    ------
    ValueError
        If x and y differ in shape or have fewer than two points.

    Notes
    -----
    Time complexity: O(n). Space complexity: O(n).
    """
    x, y = _validate_pair(x, y, numeric=True)
    if x.size < 2:
        raise ValueError("at least two points are required to compute AUC")
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def roc_auc_score(y_true, y_score, positive_label=1):
    """
    Area under the ROC curve.

    Equals the probability that a random positive sample is scored higher
    than a random negative one, with ties counted as one half.

    Parameters
    ----------
    y_true, y_score, positive_label
        As for ``roc_curve``.

    Returns
    -------
    float
        AUC in [0, 1].

    Raises
    ------
    ValueError
        If shapes differ or the inputs are empty.

    Notes
    -----
    Time complexity: O(n log n). Space complexity: O(n).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, positive_label)
    return auc(fpr, tpr)


# ============================================================
# Streaming metrics
# ============================================================

class _StreamingMetric:
    """
    Base class: accumulate per-chunk sufficient statistics.

    Subclasses implement ``_chunk_stats(y_true, y_pred)`` (a NumPy array
    of counts or sums for one chunk) and ``_compute(stats)`` (the metric
    from summed statistics). The base class handles the cumulative total
    and the rolling window.

    Parameters
    ----------
    window : int or None, default=None
        If given, ``result()`` covers only the most recent ``window``
        chunks (rolling-window metric). None accumulates over the stream.
    """

    _numeric = False

    def __init__(self, window=None):
        if window is not None:
            window = validate_int(window, "window", minimum=1)
        self.window = window
        self.reset()

    def reset(self):
        """
        Discard all accumulated statistics.

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
        self._chunks = deque(maxlen=self.window)
        self._total = None
        self.n_updates_ = 0
        self.n_samples_seen_ = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """
        Add one chunk of true and predicted values.

        Parameters
        ----------
        y_true_chunk : array-like of shape (n_samples,)
            True labels (or values, for MSE).
        y_pred_chunk : array-like of shape (n_samples,)
            Predicted labels (or scores, for AUC).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the shapes differ or the chunk is empty.

        Notes
        -----
        Time complexity: O(n_samples) for counting metrics.
        Space complexity: O(1) per chunk kept (O(window) in total).
        """
        y_true, y_pred = _validate_pair(
            y_true_chunk, y_pred_chunk, numeric=self._numeric
        )
        stats = self._chunk_stats(y_true, y_pred)

        if self.window is None:
            self._total = stats if self._total is None else self._total + stats
        else:
            self._chunks.append(stats)   # deque drops the oldest chunk

        self.n_updates_ += 1
        self.n_samples_seen_ += y_true.size
        return self

    def _current(self):
        if self.window is None:
            return self._total
        return np.sum(self._chunks, axis=0)

    def result(self):
        """
        The metric over all chunks so far (or over the current window).

        Returns
        -------
        float or np.ndarray
            The metric value.

        Notes
        -----
        Time complexity: O(window) with a window, otherwise O(1).
        """
        return self._compute(self._current())


class StreamingAccuracy(_StreamingMetric):
    """
    Accuracy accumulated over chunks: total correct / total seen.

    Parameters
    ----------
    window : int or None, default=None
        Rolling window size in chunks.

    Examples
    --------
    >>> m = StreamingAccuracy()
    >>> m.update([1, 0], [1, 1]).update([1, 1], [1, 1]).result()
    0.75
    """

    def _chunk_stats(self, y_true, y_pred):
        return np.array([np.sum(y_true == y_pred), y_true.size], dtype=float)

    def _compute(self, stats):
        return float(stats[0] / stats[1])


class _BinaryCountMetric(_StreamingMetric):
    """Accumulates TP, FP and FN for one positive label."""

    def __init__(self, positive_label=1, window=None):
        self.positive_label = positive_label
        super().__init__(window=window)

    def _chunk_stats(self, y_true, y_pred):
        true_pos = y_true == self.positive_label
        pred_pos = y_pred == self.positive_label
        return np.array([
            np.sum(true_pos & pred_pos),
            np.sum(~true_pos & pred_pos),
            np.sum(true_pos & ~pred_pos),
        ], dtype=float)

    @staticmethod
    def _ratio(numerator, denominator):
        return 0.0 if denominator == 0 else float(numerator / denominator)


class StreamingPrecision(_BinaryCountMetric):
    """
    Precision ``TP / (TP + FP)`` accumulated over chunks.

    Parameters
    ----------
    positive_label : int or str, default=1
        Label considered positive.
    window : int or None, default=None
        Rolling window size in chunks.

    Notes
    -----
    Returns 0.0 when nothing has been predicted positive, matching
    ``precision``.
    """

    def _compute(self, stats):
        tp, fp, _ = stats
        return self._ratio(tp, tp + fp)


class StreamingRecall(_BinaryCountMetric):
    """
    Recall ``TP / (TP + FN)`` accumulated over chunks.

    Parameters
    ----------
    positive_label : int or str, default=1
        Label considered positive.
    window : int or None, default=None
        Rolling window size in chunks.

    Notes
    -----
    Returns 0.0 when no positive sample has been seen, matching ``recall``.
    """

    def _compute(self, stats):
        tp, _, fn = stats
        return self._ratio(tp, tp + fn)


class StreamingF1(_BinaryCountMetric):
    """
    F1 score ``2TP / (2TP + FP + FN)`` accumulated over chunks.

    Computed from the summed counts, so it equals ``f1`` on all data seen
    (it is not an average of per-chunk F1 scores).

    Parameters
    ----------
    positive_label : int or str, default=1
        Label considered positive.
    window : int or None, default=None
        Rolling window size in chunks.
    """

    def _compute(self, stats):
        tp, fp, fn = stats
        return self._ratio(2 * tp, 2 * tp + fp + fn)


class StreamingMSE(_StreamingMetric):
    """
    Mean squared error accumulated over chunks.

    Parameters
    ----------
    window : int or None, default=None
        Rolling window size in chunks.
    """

    _numeric = True

    def _chunk_stats(self, y_true, y_pred):
        return np.array([np.sum((y_true - y_pred) ** 2), y_true.size])

    def _compute(self, stats):
        return float(stats[0] / stats[1])


class StreamingConfusionMatrix(_StreamingMetric):
    """
    Confusion matrix accumulated over chunks, with a growing label set.

    Labels are discovered as they appear and kept sorted in ``labels_``;
    rows are true labels and columns predicted labels, as in
    ``confusion_matrix``.

    Parameters
    ----------
    labels : array-like or None, default=None
        Fixed label set. None learns labels from the data.
    window : int or None, default=None
        Rolling window size in chunks.

    Attributes
    ----------
    labels_ : np.ndarray
        Sorted labels covered by ``result()``.
    """

    def __init__(self, labels=None, window=None):
        self._fixed_labels = None if labels is None else np.unique(labels)
        super().__init__(window=window)

    def reset(self):
        super().reset()
        self.labels_ = (
            np.array([]) if self._fixed_labels is None
            else self._fixed_labels.copy()
        )
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true, y_pred = _validate_pair(y_true_chunk, y_pred_chunk)
        y_true, y_pred = y_true.ravel(), y_pred.ravel()

        if self._fixed_labels is None:
            seen = np.unique(np.concatenate((y_true, y_pred)))
            self.labels_ = (
                seen if self.labels_.size == 0
                else np.union1d(self.labels_, seen)
            )

        chunk = (self.labels_, confusion_matrix(y_true, y_pred, self.labels_))

        if self.window is None:
            self._total = chunk if self._total is None else (
                self.labels_, self._align(*self._total) + chunk[1]
            )
        else:
            self._chunks.append(chunk)

        self.n_updates_ += 1
        self.n_samples_seen_ += y_true.size
        return self

    def _align(self, labels, matrix):
        """Embed a matrix over ``labels`` into the current ``labels_``."""
        if labels.size == self.labels_.size:
            return matrix
        idx = np.searchsorted(self.labels_, labels)
        out = np.zeros((self.labels_.size, self.labels_.size), dtype=int)
        out[np.ix_(idx, idx)] = matrix
        return out

    def result(self):
        """
        The accumulated confusion matrix.

        Returns
        -------
        np.ndarray of shape (n_labels, n_labels)
            Integer counts over ``labels_``.

        Notes
        -----
        Time complexity: O(window * n_labels^2) with a window, otherwise
        O(n_labels^2).
        """
        if self.window is None:
            return self._align(*self._total)
        return sum(self._align(*chunk) for chunk in self._chunks)


class StreamingAUC(_StreamingMetric):
    """
    ROC AUC accumulated over chunks (exact, with ties counted as 1/2).

    Keeps the number of positive and negative samples at each distinct
    score, so memory is O(distinct scores) rather than O(samples). This is
    small for models with few output values, such as a decision tree's
    leaf probabilities.

    Parameters
    ----------
    positive_label : int or str, default=1
        Label considered positive.
    window : int or None, default=None
        Rolling window size in chunks.

    Notes
    -----
    ``update(y_true_chunk, y_score_chunk)`` takes scores or probabilities
    for the positive class as the second argument.
    """

    def __init__(self, positive_label=1, window=None):
        self.positive_label = positive_label
        super().__init__(window=window)

    @staticmethod
    def _merge(a, b):
        scores = np.concatenate((a[0], b[0]))
        unique, inverse = np.unique(scores, return_inverse=True)
        pos = np.bincount(inverse, weights=np.concatenate((a[1], b[1])),
                          minlength=unique.size)
        neg = np.bincount(inverse, weights=np.concatenate((a[2], b[2])),
                          minlength=unique.size)
        return unique, pos, neg

    def update(self, y_true_chunk, y_score_chunk):
        chunk = _score_counts(y_true_chunk, y_score_chunk, self.positive_label)

        if self.window is None:
            self._total = chunk if self._total is None else self._merge(
                self._total, chunk
            )
        else:
            self._chunks.append(chunk)

        self.n_updates_ += 1
        self.n_samples_seen_ += int(chunk[1].sum() + chunk[2].sum())
        return self

    def _current(self):
        if self.window is None:
            return self._total
        merged = self._chunks[0]
        for chunk in list(self._chunks)[1:]:
            merged = self._merge(merged, chunk)
        return merged

    def roc_curve(self):
        """
        ROC curve of everything accumulated (or of the current window).

        Returns
        -------
        fpr, tpr, thresholds : np.ndarray
            As for ``roc_curve``.

        Notes
        -----
        Time complexity: O(u) for u distinct scores.
        """
        return _roc_from_counts(*self._current())

    def result(self):
        """
        AUC of everything accumulated (or of the current window).

        Returns
        -------
        float

        Notes
        -----
        Time complexity: O(u) for u distinct scores.
        """
        fpr, tpr, _ = self.roc_curve()
        return auc(fpr, tpr)


__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "mse",
    "roc_curve",
    "auc",
    "roc_auc_score",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingMSE",
    "StreamingConfusionMatrix",
    "StreamingAUC",
]
