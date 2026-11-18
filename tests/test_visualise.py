"""Tests for numcompute.visualise (run headless with the Agg backend)."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from numcompute.stream import StreamTrainer  # noqa: E402
from numcompute.tree import DecisionTreeClassifier  # noqa: E402
from numcompute.visualise import (  # noqa: E402
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
    plot_stream_log,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def trainer():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 2))
    y = (X[:, 0] > 0).astype(int)
    chunks = [(X[i], y[i]) for i in np.array_split(np.arange(800), 8)]
    return StreamTrainer(DecisionTreeClassifier(buffer_size=200),
                         window=2).run(chunks)


# ---------------------------------------------------------
# plot_metric_over_time
# ---------------------------------------------------------

def test_metric_plot_returns_figure_with_one_line():
    fig = plot_metric_over_time([0.5, 0.7, 0.8], title="Accuracy",
                                ylabel="Accuracy")
    ax = fig.axes[0]
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(ax.lines) == 1
    assert ax.get_title() == "Accuracy" and ax.get_ylabel() == "Accuracy"
    assert np.array_equal(ax.lines[0].get_ydata(), [0.5, 0.7, 0.8])


def test_metric_plot_draws_nan_as_gap():
    fig = plot_metric_over_time([np.nan, np.nan, 0.8, 0.9])
    assert np.isnan(fig.axes[0].lines[0].get_ydata()[0])


def test_metric_plot_uses_integer_chunk_ticks():
    fig = plot_metric_over_time(np.linspace(0, 1, 7))
    fig.canvas.draw()
    ticks = fig.axes[0].get_xticks()
    assert np.allclose(ticks, np.round(ticks))


def test_metric_plot_draws_on_given_axes():
    fig, (left, right) = plt.subplots(1, 2)
    returned = plot_metric_over_time([1, 2, 3], ax=right, label="loss")
    assert returned is fig
    assert len(right.lines) == 1 and len(left.lines) == 0
    assert right.get_legend() is not None


@pytest.mark.parametrize("extension", ["png", "svg", "pdf"])
def test_metric_plot_saves_to_file(tmp_path, extension):
    path = tmp_path / f"metric.{extension}"
    plot_metric_over_time([0.1, 0.2], save_path=path)
    assert path.exists() and path.stat().st_size > 0


def test_show_calls_pyplot_show(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(1))
    plot_metric_over_time([1.0, 2.0], show=True)
    assert calls == [1]


@pytest.mark.parametrize(
    "values, error",
    [
        ([], ValueError),
        ([[1.0, 2.0]], ValueError),
        ([1.0, np.inf], ValueError),
        (["a", "b"], TypeError),
    ],
)
def test_metric_plot_rejects_invalid_values(values, error):
    with pytest.raises(error):
        plot_metric_over_time(values)


# ---------------------------------------------------------
# compare_models
# ---------------------------------------------------------

def test_compare_models_two_labelled_lines():
    fig = compare_models([0.6, 0.7], [0.5, 0.9],
                         labels=("buffer_size=200", "buffer_size=800"))
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "buffer_size=200", "buffer_size=800"
    ]


def test_compare_models_different_lengths():
    fig = compare_models([0.5, 0.6, 0.7], [0.4])
    assert len(fig.axes[0].lines[1].get_ydata()) == 1


@pytest.mark.parametrize("labels", [("only one",), ("a", "b", "c"), "ab"])
def test_compare_models_requires_two_labels(labels):
    with pytest.raises(ValueError, match="two"):
        compare_models([1.0], [2.0], labels=labels)


def test_compare_models_rejects_empty_series():
    with pytest.raises(ValueError, match="metric2"):
        compare_models([1.0], [])


# ---------------------------------------------------------
# plot_predictions_vs_ground_truth
# ---------------------------------------------------------

def _matrix_texts(fig):
    return [t.get_text() for t in fig.axes[1].texts]


def test_predictions_plot_confusion_counts():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    fig = plot_predictions_vs_ground_truth(y_true, y_pred)
    assert _matrix_texts(fig) == ["1", "1", "1", "2"]
    assert "accuracy 0.600" in fig._suptitle.get_text()


def test_predictions_plot_string_labels():
    fig = plot_predictions_vs_ground_truth(["cat", "dog"], ["cat", "bird"])
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["bird", "cat", "dog"]


def test_predictions_plot_all_correct_single_label():
    fig = plot_predictions_vs_ground_truth([1, 1, 1], [1, 1, 1])
    assert _matrix_texts(fig) == ["3"]


def test_predictions_plot_limits_points_but_not_matrix():
    y = np.arange(1000) % 2
    fig = plot_predictions_vs_ground_truth(y, y, max_points=50)
    truth_points = fig.axes[0].collections[0].get_offsets()
    assert len(truth_points) == 50
    assert _matrix_texts(fig) == ["500", "0", "0", "500"]


def test_predictions_plot_saves(tmp_path):
    path = tmp_path / "pred.png"
    plot_predictions_vs_ground_truth([0, 1], [0, 1], save_path=path)
    assert path.exists()


@pytest.mark.parametrize(
    "y_true, y_pred, kwargs",
    [
        ([], [], {}),
        ([0, 1], [0], {}),
        ([[0, 1]], [[0, 1]], {}),
        ([0, 1], [0, 1], {"max_points": 0}),
    ],
)
def test_predictions_plot_rejects_invalid_input(y_true, y_pred, kwargs):
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth(y_true, y_pred, **kwargs)


# ---------------------------------------------------------
# plot_stream_log (StreamTrainer integration)
# ---------------------------------------------------------

def test_stream_log_from_trainer(trainer):
    fig = plot_stream_log(trainer, keys=("chunk_accuracy", "rolling_accuracy",
                                         "cumulative_accuracy"))
    ax = fig.axes[0]
    assert len(ax.lines) == 3
    assert len(ax.lines[0].get_ydata()) == trainer.n_chunks_seen_


def test_stream_log_from_history_list(trainer):
    fig = plot_stream_log(trainer.history_, keys="memory_bytes")
    assert len(fig.axes[0].lines) == 1


def test_trainer_metrics_feed_required_plots(trainer):
    plot_metric_over_time(trainer.get_metric("cumulative_accuracy"))
    compare_models(trainer.get_metric("chunk_accuracy"),
                   trainer.get_metric("rolling_accuracy"),
                   labels=("chunk", "rolling"))


def test_stream_log_errors(trainer):
    with pytest.raises(ValueError, match="empty"):
        plot_stream_log([])
    with pytest.raises(ValueError, match="keys"):
        plot_stream_log(trainer, keys=())
    with pytest.raises(KeyError, match="available"):
        plot_stream_log(trainer, keys=("chunk_r2",))
