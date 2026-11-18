"""
stream.py
=========

``StreamTrainer``: incremental training, evaluation and logging over a
stream of chunks.

The trainer wraps any model with ``partial_fit(X, y)`` and ``predict(X)``
(a ``Pipeline`` or a ``DecisionTreeClassifier``) and evaluates it
*prequentially* (test-then-train): each chunk is first scored by the
model trained on all earlier chunks, then used for training. Every chunk
therefore acts as unseen test data, without a separate hold-out set.

For each chunk the trainer logs:

* per-chunk metrics (``chunk_<metric>``), plus cumulative
  (``cumulative_<metric>``) and, if ``window`` is set, rolling
  (``rolling_<metric>``) versions from ``numcompute.metrics``;
* the memory footprint of the model state (bytes held in NumPy arrays,
  including any training buffer), and optionally the peak memory traced
  during training;
* training and prediction time, whether the model retrained, and how many
  samples are waiting in its buffer.

Example
-------
>>> trainer = StreamTrainer(pipe, metrics=("accuracy", "f1"))   # doctest: +SKIP
>>> for X_chunk, y_chunk in stream:                              # doctest: +SKIP
...     trainer.process_chunk(X_chunk, y_chunk)
>>> trainer.get_metric("cumulative_accuracy")                    # doctest: +SKIP
"""

import time
import tracemalloc
from collections import deque

import numpy as np

from numcompute.metrics import (
    StreamingAccuracy,
    StreamingAUC,
    StreamingF1,
    StreamingMSE,
    StreamingPrecision,
    StreamingRecall,
    accuracy,
    f1,
    mse,
    precision,
    recall,
    roc_auc_score,
)
from numcompute.utils import NotFittedError, validate_int


# name -> (one-shot function, streaming class, needs positive_label)
_METRICS = {
    "accuracy": (accuracy, StreamingAccuracy, False),
    "precision": (precision, StreamingPrecision, True),
    "recall": (recall, StreamingRecall, True),
    "f1": (f1, StreamingF1, True),
    "mse": (mse, StreamingMSE, False),
    "auc": (roc_auc_score, StreamingAUC, True),
}


# ============================================================
# Memory footprint
# ============================================================

def estimate_memory(obj):
    """
    Total bytes held in NumPy arrays reachable from an object.

    Walks instance attributes, lists, tuples, sets, deques and dicts
    (following each object only once), and sums ``ndarray.nbytes``. This
    measures the data a model keeps (statistics, tree arrays, training
    buffers), which is what grows or shrinks during streaming.

    Parameters
    ----------
    obj : object
        A model, pipeline or any container.

    Returns
    -------
    int
        Bytes in reachable arrays. Python object overhead is not counted.

    Raises
    ------
    None.

    Notes
    -----
    Views of the same base array may be counted more than once.

    Time complexity: O(number of reachable objects).
    Space complexity: O(number of reachable objects) for the visited set.
    """
    seen = set()
    stack = [obj]
    total = 0

    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))

        if isinstance(item, np.ndarray):
            total += item.nbytes
            if item.dtype == object:
                stack.extend(item.ravel().tolist())
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset, deque)):
            stack.extend(item)
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            stack.extend(vars(item).values())

    return int(total)


# ============================================================
# StreamTrainer
# ============================================================

class StreamTrainer:
    """
    Manage incremental training, prequential evaluation and logging.

    Parameters
    ----------
    model : object
        Anything with ``partial_fit(X, y)`` and ``predict(X)``, e.g. a
        ``Pipeline`` ending in a ``DecisionTreeClassifier``. The "auc"
        metric additionally needs ``predict_proba``.
    metrics : sequence of str, default=("accuracy",)
        Metrics to log, from "accuracy", "precision", "recall", "f1",
        "mse" and "auc". "accuracy" is always included, because the
        cumulative accuracy is part of every log.
    positive_label : int or str, default=1
        Positive class for "precision", "recall", "f1" and "auc".
    window : int or None, default=None
        If given, also log rolling metrics over the last ``window`` scored
        chunks.
    track_peak_memory : bool, default=False
        If True, record the peak memory allocated during each training
        step with ``tracemalloc`` (slower).

    Attributes
    ----------
    history_ : list of dict
        One log entry per processed chunk (see ``process_chunk``).
    n_chunks_seen_ : int
        Number of chunks processed.
    n_samples_seen_ : int
        Number of samples processed.

    Raises
    ------
    TypeError
        If the model lacks ``partial_fit`` or ``predict``, or an argument
        has the wrong type.
    ValueError
        If a metric name is unknown, "auc" is requested for a model
        without ``predict_proba``, or window < 1.
    """

    def __init__(self, model, metrics=("accuracy",), positive_label=1,
                 window=None, track_peak_memory=False):
        for method in ("partial_fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"model must implement {method}()")

        if isinstance(metrics, str):
            metrics = (metrics,)
        metrics = tuple(dict.fromkeys(("accuracy",) + tuple(metrics)))
        unknown = [m for m in metrics if m not in _METRICS]
        if unknown:
            raise ValueError(
                f"Unknown metric(s) {unknown}; choose from "
                f"{sorted(_METRICS)}"
            )
        if "auc" in metrics and not callable(
            getattr(model, "predict_proba", None)
        ):
            raise ValueError("the 'auc' metric requires model.predict_proba")

        if window is not None:
            window = validate_int(window, "window", minimum=1)
        if not isinstance(track_peak_memory, (bool, np.bool_)):
            raise TypeError("track_peak_memory must be a boolean")

        self.model = model
        self.metrics = metrics
        self.positive_label = positive_label
        self.window = window
        self.track_peak_memory = bool(track_peak_memory)
        self.reset()

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    def _make_metric(self, name, window):
        _, cls, needs_label = _METRICS[name]
        if needs_label:
            return cls(positive_label=self.positive_label, window=window)
        return cls(window=window)

    def reset(self):
        """
        Clear the log and all metric accumulators (the model is untouched).

        Returns
        -------
        StreamTrainer
            ``self``.

        Raises
        ------
        None.

        Notes
        -----
        Time complexity: O(number of metrics).
        """
        self._cumulative = {m: self._make_metric(m, None) for m in self.metrics}
        self._rolling = (
            {m: self._make_metric(m, self.window) for m in self.metrics}
            if self.window is not None else {}
        )
        self.history_ = []
        self.n_chunks_seen_ = 0
        self.n_samples_seen_ = 0
        return self

    # --------------------------------------------------------
    # Chunk operations
    # --------------------------------------------------------

    def _positive_scores(self, X):
        """Probability of the positive class (0 if it is not yet known)."""
        proba = self.model.predict_proba(X)
        classes = getattr(self.model, "classes_", None)
        if classes is None and hasattr(self.model, "steps"):
            classes = getattr(self.model.steps[-1][1], "classes_", None)
        if classes is None:
            raise ValueError("cannot locate classes_ to compute AUC")

        matches = np.flatnonzero(np.asarray(classes) == self.positive_label)
        if matches.size == 0:
            return np.zeros(proba.shape[0])
        return proba[:, matches[0]]

    @staticmethod
    def _one_shot(function, *args, **kwargs):
        """Evaluate a metric on one chunk; NaN where it is undefined."""
        try:
            return float(function(*args, **kwargs))
        except ValueError:
            return float("nan")    # e.g. AUC on a single-class chunk

    def score_chunk(self, X, y):
        """
        Evaluate the current model on a chunk without training on it.

        Updates the cumulative (and rolling) metrics and returns this
        chunk's values. If the model cannot predict yet (e.g. a tree whose
        buffer has not filled), every metric is NaN and the accumulators
        are left unchanged.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.
        y : array-like of shape (n_samples,)
            True labels (or values, for "mse").

        Returns
        -------
        dict
            ``scored`` (bool), ``predict_time`` (seconds), and for each
            metric ``chunk_<m>``, ``cumulative_<m>`` and, with a window,
            ``rolling_<m>``.

        Raises
        ------
        ValueError, TypeError
            As raised by the model for malformed input.

        Notes
        -----
        Time complexity: one ``predict`` (and ``predict_proba`` for "auc")
        plus O(n) per metric.
        """
        y = np.asarray(y)
        result = {"scored": False, "predict_time": float("nan")}

        if len(y) > 0:
            start = time.perf_counter()
            try:
                y_pred = self.model.predict(X)
                scores = (self._positive_scores(X)
                          if "auc" in self.metrics else None)
            except NotFittedError:
                y_pred = None
            result["predict_time"] = time.perf_counter() - start

            if y_pred is not None:
                result["scored"] = True
                for name in self.metrics:
                    function, _, needs_label = _METRICS[name]
                    prediction = scores if name == "auc" else y_pred
                    kwargs = ({"positive_label": self.positive_label}
                              if needs_label else {})
                    result[f"chunk_{name}"] = self._one_shot(
                        function, y, prediction, **kwargs
                    )
                    self._cumulative[name].update(y, prediction)
                    if self.window is not None:
                        self._rolling[name].update(y, prediction)

        for name in self.metrics:
            result.setdefault(f"chunk_{name}", float("nan"))
            result[f"cumulative_{name}"] = self._current(self._cumulative, name)
            if self.window is not None:
                result[f"rolling_{name}"] = self._current(self._rolling, name)

        return result

    @staticmethod
    def _current(accumulators, name):
        try:
            return float(accumulators[name].result())
        except NotFittedError:
            return float("nan")

    def fit_chunk(self, X, y):
        """
        Train the model on one chunk with ``partial_fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.
        y : array-like of shape (n_samples,)
            Labels.

        Returns
        -------
        dict
            ``fit_time`` (seconds), ``retrained`` (bool, if the model
            exposes ``n_retrains_``), ``n_buffered`` (samples waiting in
            the model's buffer, or None), ``memory_bytes`` (model state
            after training) and, if tracking is on, ``peak_memory_bytes``.

        Raises
        ------
        ValueError, TypeError
            As raised by the model's ``partial_fit``.

        Notes
        -----
        Time complexity: one ``partial_fit`` plus O(model state) for the
        memory estimate.
        """
        estimator = self._final_estimator()
        before = getattr(estimator, "n_retrains_", None)

        if self.track_peak_memory:
            tracemalloc.start()
        start = time.perf_counter()
        try:
            self.model.partial_fit(X, y)
        finally:
            fit_time = time.perf_counter() - start
            if self.track_peak_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

        after = getattr(estimator, "n_retrains_", None)
        result = {
            "fit_time": fit_time,
            "retrained": (after is not None and before is not None
                          and after > before),
            "n_buffered": getattr(estimator, "n_buffered_", None),
            "memory_bytes": estimate_memory(self.model),
        }
        if self.track_peak_memory:
            result["peak_memory_bytes"] = int(peak)
        return result

    def _final_estimator(self):
        if hasattr(self.model, "steps") and self.model.steps:
            return self.model.steps[-1][1]
        return self.model

    def process_chunk(self, X, y):
        """
        Test-then-train on one chunk and append a log entry.

        The chunk is scored with the model as trained on earlier chunks,
        then passed to ``partial_fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.
        y : array-like of shape (n_samples,)
            Labels.

        Returns
        -------
        dict
            The log entry: ``chunk`` (0-based index), ``n_samples``, plus
            every key from ``score_chunk`` and ``fit_chunk``.

        Raises
        ------
        ValueError
            If X and y have different lengths, plus anything the model
            raises.

        Notes
        -----
        Time complexity: ``score_chunk`` plus ``fit_chunk``.
        """
        n = len(y)
        if np.shape(X)[0] != n:
            raise ValueError(
                f"X has {np.shape(X)[0]} samples but y has {n}"
            )

        entry = {"chunk": self.n_chunks_seen_, "n_samples": int(n)}
        entry.update(self.score_chunk(X, y))
        entry.update(self.fit_chunk(X, y))

        self.history_.append(entry)
        self.n_chunks_seen_ += 1
        self.n_samples_seen_ += int(n)
        return entry

    partial_fit = process_chunk

    def run(self, chunks):
        """
        Process every ``(X_chunk, y_chunk)`` pair from an iterable.

        Parameters
        ----------
        chunks : iterable of (array-like, array-like)

        Returns
        -------
        StreamTrainer
            ``self``.

        Raises
        ------
        ValueError
            If an item is not an (X, y) pair, plus anything raised by
            ``process_chunk``.

        Notes
        -----
        Time complexity: the sum of ``process_chunk`` costs.
        """
        for item in chunks:
            try:
                X_chunk, y_chunk = item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "each item of chunks must be an (X_chunk, y_chunk) pair"
                ) from exc
            self.process_chunk(X_chunk, y_chunk)
        return self

    # --------------------------------------------------------
    # Reading the log
    # --------------------------------------------------------

    def get_metric(self, name):
        """
        One logged value per chunk, as an array (ready for plotting).

        Parameters
        ----------
        name : str
            A log key, e.g. "cumulative_accuracy", "chunk_f1",
            "memory_bytes" or "fit_time".

        Returns
        -------
        np.ndarray of shape (n_chunks,)
            float64 values; NaN where the value is undefined or missing.

        Raises
        ------
        KeyError
            If no log entry has that key.

        Notes
        -----
        Time complexity: O(n_chunks).
        """
        if self.history_ and not any(name in e for e in self.history_):
            keys = sorted(self.history_[0])
            raise KeyError(f"Unknown log key {name!r}; available: {keys}")

        values = [e.get(name) for e in self.history_]
        return np.array(
            [np.nan if v is None else float(v) for v in values], dtype=float
        )

    def summary(self):
        """
        Final cumulative metrics and totals for the whole stream.

        Returns
        -------
        dict
            ``n_chunks``, ``n_samples``, ``cumulative_<metric>`` for each
            metric, ``total_fit_time``, ``n_retrains`` and
            ``final_memory_bytes``.

        Raises
        ------
        None.

        Notes
        -----
        Time complexity: O(n_chunks).
        """
        out = {"n_chunks": self.n_chunks_seen_,
               "n_samples": self.n_samples_seen_}
        for name in self.metrics:
            out[f"cumulative_{name}"] = self._current(self._cumulative, name)
        out["total_fit_time"] = float(
            np.nansum(self.get_metric("fit_time"))
        ) if self.history_ else 0.0
        out["n_retrains"] = int(sum(e.get("retrained", False)
                                    for e in self.history_))
        out["final_memory_bytes"] = (
            self.history_[-1]["memory_bytes"] if self.history_ else 0
        )
        return out


__all__ = ["StreamTrainer", "estimate_memory"]
