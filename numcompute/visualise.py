"""
visualise.py
============

Reusable matplotlib plots for streaming experiments.

Required plots
--------------
* ``plot_metric_over_time(metric_values, title, ylabel)``: a metric
  (e.g. accuracy) across chunks.
* ``compare_models(metric1, metric2, labels)``: two models on the same
  streaming metric (e.g. trees with different ``buffer_size``).
* ``plot_predictions_vs_ground_truth(y_true, y_pred)``: predictions
  against true labels on the latest chunk.

``plot_stream_log`` plots several keys of a ``StreamTrainer`` log at once.

Every function

* returns the ``matplotlib.figure.Figure`` it drew on;
* draws on ``ax`` if one is given (so plots can be combined in subplots),
  otherwise creates a new figure;
* saves to ``save_path`` if given (format from the extension, e.g. .png,
  .pdf, .svg) and calls ``plt.show()`` if ``show=True``. In a notebook the
  returned figure is displayed inline automatically.

NaN values (e.g. chunks scored before the first retrain) appear as gaps.
matplotlib is imported only when a plot is drawn, so the rest of
NumCompute works without it.
"""

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "numcompute.visualise requires matplotlib; install it with "
            "'pip install matplotlib' or 'pip install numcompute[viz]'"
        ) from exc
    return plt


def _as_series(values, name):
    """Validate a 1D numeric series; NaN is allowed (drawn as a gap)."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; received {arr.ndim} dimensions")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} must not contain infinite values")
    return arr


def _figure(ax, figsize):
    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _integer_chunk_axis(ax):
    """Chunks are whole numbers: never show ticks such as 2.5."""
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _finish(fig, save_path, show, dpi):
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        _pyplot().show()
    return fig


def plot_metric_over_time(metric_values, title="Metric over time",
                          ylabel="Value", xlabel="Chunk", label=None,
                          ax=None, save_path=None, show=False, dpi=100,
                          figsize=(8, 4.5)):
    """
    Plot a metric across chunks.

    Parameters
    ----------
    metric_values : array-like of shape (n_chunks,)
        One value per chunk, e.g. ``trainer.get_metric("cumulative_accuracy")``.
        NaN values are drawn as gaps.
    title, ylabel, xlabel : str
        Axis title and labels.
    label : str or None
        Legend entry; a legend is shown only when given.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. None creates a new figure.
    save_path : str or path-like or None
        Save the figure here (format from the extension).
    show : bool, default=False
        Call ``plt.show()`` after drawing.
    dpi : int, default=100
        Resolution used when saving.
    figsize : tuple, default=(8, 4.5)
        Size of a newly created figure, in inches.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If metric_values is not numeric.
    ValueError
        If metric_values is not 1D, is empty, or contains infinity.
    ImportError
        If matplotlib is not installed.

    Notes
    -----
    Time complexity: O(n_chunks).
    """
    values = _as_series(metric_values, "metric_values")
    fig, ax = _figure(ax, figsize)

    ax.plot(np.arange(values.size), values, marker="o", markersize=3,
            label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _integer_chunk_axis(ax)
    if label is not None:
        ax.legend()

    return _finish(fig, save_path, show, dpi)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),
                   title="Model comparison", ylabel="Value", xlabel="Chunk",
                   ax=None, save_path=None, show=False, dpi=100,
                   figsize=(8, 4.5)):
    """
    Compare two models on a streaming metric, chunk by chunk.

    Parameters
    ----------
    metric1, metric2 : array-like of shape (n_chunks,)
        Per-chunk values for each model. The lengths may differ (e.g. two
        streams of different length); NaN values are drawn as gaps.
    labels : sequence of two str, default=("Model 1", "Model 2")
        Legend entries, e.g. ``("buffer_size=200", "buffer_size=800")``.
    title, ylabel, xlabel : str
        Axis title and labels.
    ax, save_path, show, dpi, figsize
        As for ``plot_metric_over_time``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If a metric is not numeric.
    ValueError
        If a metric is not 1D, is empty or contains infinity, or labels
        does not contain exactly two entries.
    ImportError
        If matplotlib is not installed.

    Notes
    -----
    Time complexity: O(n_chunks).
    """
    first = _as_series(metric1, "metric1")
    second = _as_series(metric2, "metric2")
    labels = list(labels) if not isinstance(labels, str) else [labels]
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two entries")

    fig, ax = _figure(ax, figsize)
    ax.plot(np.arange(first.size), first, marker="o", markersize=3,
            label=str(labels[0]))
    ax.plot(np.arange(second.size), second, marker="s", markersize=3,
            linestyle="--", label=str(labels[1]))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _integer_chunk_axis(ax)
    ax.legend()

    return _finish(fig, save_path, show, dpi)


def plot_predictions_vs_ground_truth(y_true, y_pred,
                                     title="Predictions vs ground truth",
                                     save_path=None, show=False, dpi=100,
                                     figsize=(11, 4.5), max_points=500):
    """
    Visualise predictions against true labels on one chunk.

    Two panels:

    * left: each sample's true label (circles) and prediction (crosses),
      with mistakes highlighted in red;
    * right: the confusion matrix with counts (rows = true labels,
      columns = predicted labels).

    Labels may be numbers or strings; the y axis lists the label names.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        True and predicted labels, e.g. for the latest chunk.
    title : str
        Figure title (the chunk accuracy is appended).
    save_path, show, dpi, figsize
        As for ``plot_metric_over_time``.
    max_points : int, default=500
        At most this many samples are drawn in the left panel (the first
        ones); the confusion matrix always uses every sample.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the inputs are not 1D, differ in length, or are empty, or
        max_points < 1.
    ImportError
        If matplotlib is not installed.

    Notes
    -----
    Time complexity: O(n + k^2) for k distinct labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length; received "
            f"{y_true.size} and {y_pred.size}"
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    max_points = _validate_max_points(max_points)

    labels = np.unique(np.concatenate((y_true, y_pred)))
    true_idx = np.searchsorted(labels, y_true)
    pred_idx = np.searchsorted(labels, y_pred)
    correct = true_idx == pred_idx

    counts = np.zeros((labels.size, labels.size), dtype=int)
    np.add.at(counts, (true_idx, pred_idx), 1)

    plt = _pyplot()
    fig, (ax_points, ax_matrix) = plt.subplots(1, 2, figsize=figsize)

    shown = np.arange(min(y_true.size, max_points))
    ax_points.scatter(shown, true_idx[shown], marker="o", facecolors="none",
                      edgecolors="tab:blue", label="ground truth")
    ax_points.scatter(shown[correct[shown]], pred_idx[shown][correct[shown]],
                      marker="x", color="tab:green", label="correct")
    wrong = shown[~correct[shown]]
    ax_points.scatter(wrong, pred_idx[wrong], marker="x", color="tab:red",
                      label="wrong")
    ax_points.set_yticks(np.arange(labels.size))
    ax_points.set_yticklabels([str(lab) for lab in labels])
    ax_points.set_xlabel("Sample in chunk")
    ax_points.set_ylabel("Label")
    ax_points.set_title("Per-sample predictions")
    ax_points.legend(loc="best", fontsize="small")
    ax_points.grid(True, alpha=0.3)

    image = ax_matrix.imshow(counts, cmap="Blues")
    ax_matrix.set_xticks(np.arange(labels.size))
    ax_matrix.set_yticks(np.arange(labels.size))
    ax_matrix.set_xticklabels([str(lab) for lab in labels])
    ax_matrix.set_yticklabels([str(lab) for lab in labels])
    ax_matrix.set_xlabel("Predicted label")
    ax_matrix.set_ylabel("True label")
    ax_matrix.set_title("Confusion matrix")
    threshold = counts.max() / 2.0 if counts.max() > 0 else 0.5
    for i in range(labels.size):
        for j in range(labels.size):
            ax_matrix.text(j, i, str(counts[i, j]), ha="center", va="center",
                           color="white" if counts[i, j] > threshold
                           else "black")
    fig.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title} (accuracy {correct.mean():.3f})")
    fig.tight_layout()
    return _finish(fig, save_path, show, dpi)


def _validate_max_points(max_points):
    if isinstance(max_points, (bool, np.bool_)) or not isinstance(
        max_points, (int, np.integer)
    ) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    return int(max_points)


def plot_stream_log(history, keys=("chunk_accuracy", "cumulative_accuracy"),
                    title="Streaming metrics", ylabel="Value",
                    ax=None, save_path=None, show=False, dpi=100,
                    figsize=(8, 4.5)):
    """
    Plot several keys of a ``StreamTrainer`` log on one set of axes.

    Parameters
    ----------
    history : StreamTrainer or list of dict
        A trainer, or its ``history_`` list (one dict per chunk).
    keys : sequence of str
        Log keys to plot, e.g. ``("chunk_accuracy", "rolling_accuracy")``.
    title, ylabel : str
        Axis title and y label.
    ax, save_path, show, dpi, figsize
        As for ``plot_metric_over_time``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the history is empty or keys is empty.
    KeyError
        If a key appears in no log entry.
    ImportError
        If matplotlib is not installed.

    Notes
    -----
    Time complexity: O(n_chunks * len(keys)).
    """
    entries = getattr(history, "history_", history)
    if not entries:
        raise ValueError("history is empty; process at least one chunk")
    if isinstance(keys, str):
        keys = (keys,)
    if not keys:
        raise ValueError("keys must not be empty")

    fig, ax = _figure(ax, figsize)
    for key in keys:
        if not any(key in entry for entry in entries):
            raise KeyError(
                f"Unknown log key {key!r}; available: {sorted(entries[0])}"
            )
        values = [entry.get(key) for entry in entries]
        series = np.array([np.nan if v is None else float(v) for v in values])
        ax.plot(np.arange(series.size), series, marker="o", markersize=3,
                label=key)

    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _integer_chunk_axis(ax)
    ax.legend()
    return _finish(fig, save_path, show, dpi)


__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
    "plot_stream_log",
]
