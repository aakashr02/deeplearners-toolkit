"""Tests for ROC/AUC and the streaming metric classes."""

import numpy as np
import pytest

from numcompute.metrics import (
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingMSE,
    StreamingPrecision,
    StreamingRecall,
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_auc_score,
    roc_curve,
)


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    return rng.integers(0, 3, 400), rng.integers(0, 3, 400)


def _chunks(n, n_chunks):
    return np.array_split(np.arange(n), n_chunks)


def _brute_force_auc(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true != 1]
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return greater + 0.5 * ties


# ---------------------------------------------------------
# One-shot ROC / AUC
# ---------------------------------------------------------

def test_roc_curve_perfect_separation():
    fpr, tpr, thresholds = roc_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert np.array_equal(fpr, [0, 0, 0, 0.5, 1])
    assert np.array_equal(tpr, [0, 0.5, 1, 1, 1])
    assert thresholds[0] == np.inf
    assert np.all(np.diff(thresholds) < 0)


@pytest.mark.parametrize(
    "y_true, y_score, expected",
    [
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),
        ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.0),
        ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.5),   # all tied
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.75),
    ],
)
def test_roc_auc_score_known_values(y_true, y_score, expected):
    assert np.isclose(roc_auc_score(y_true, y_score), expected)


def test_roc_auc_score_matches_brute_force_with_ties():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 300)
    y_score = rng.integers(0, 10, 300) / 10.0           # many ties
    assert np.isclose(roc_auc_score(y_true, y_score),
                      _brute_force_auc(y_true, y_score))


def test_roc_auc_custom_positive_label():
    y_true = np.array(["neg", "pos", "neg", "pos"])
    assert roc_auc_score(y_true, [0.1, 0.9, 0.2, 0.8],
                         positive_label="pos") == 1.0


def test_auc_trapezoid():
    assert auc([0, 1], [0, 1]) == 0.5
    assert auc([0, 0.5, 1], [0, 1, 1]) == 0.75


# ---------------------------------------------------------
# Streaming metrics equal the one-shot metric on all data
# ---------------------------------------------------------

@pytest.mark.parametrize("n_chunks", [1, 3, 17])
@pytest.mark.parametrize(
    "metric_class, function",
    [
        (StreamingAccuracy, accuracy),
        (StreamingPrecision, precision),
        (StreamingRecall, recall),
        (StreamingF1, f1),
    ],
)
def test_streaming_metric_matches_batch(labels, metric_class, function,
                                        n_chunks):
    y_true, y_pred = labels
    metric = metric_class()
    for idx in _chunks(y_true.size, n_chunks):
        metric.update(y_true[idx], y_pred[idx])

    assert np.isclose(metric.result(), function(y_true, y_pred))
    assert metric.n_samples_seen_ == y_true.size
    assert metric.n_updates_ == n_chunks


def test_streaming_f1_is_not_mean_of_chunk_f1():
    chunks = [([1, 1, 0], [1, 0, 0]), ([1, 0, 0, 0], [1, 1, 1, 0])]
    metric = StreamingF1()
    for y_true, y_pred in chunks:
        metric.update(y_true, y_pred)

    y_true = np.concatenate([c[0] for c in chunks])
    y_pred = np.concatenate([c[1] for c in chunks])
    assert np.isclose(metric.result(), f1(y_true, y_pred))


def test_streaming_precision_positive_label():
    metric = StreamingPrecision(positive_label="spam")
    metric.update(["spam", "ham"], ["spam", "spam"])
    assert metric.result() == 0.5


def test_streaming_mse_matches_batch():
    rng = np.random.default_rng(2)
    y_true, y_pred = rng.normal(size=100), rng.normal(size=100)
    metric = StreamingMSE()
    for idx in _chunks(100, 6):
        metric.update(y_true[idx], y_pred[idx])
    assert np.isclose(metric.result(), mse(y_true, y_pred))


def test_streaming_confusion_matrix_matches_batch(labels):
    y_true, y_pred = labels
    metric = StreamingConfusionMatrix()
    for idx in _chunks(y_true.size, 9):
        metric.update(y_true[idx], y_pred[idx])

    assert np.array_equal(metric.result(), confusion_matrix(y_true, y_pred))
    assert np.array_equal(metric.labels_, [0, 1, 2])


def test_streaming_confusion_matrix_grows_label_set():
    metric = StreamingConfusionMatrix()
    metric.update([0, 1], [0, 1])
    metric.update([2, 2], [2, 0])

    assert np.array_equal(metric.labels_, [0, 1, 2])
    assert np.array_equal(metric.result(), [[1, 0, 0], [0, 1, 0], [1, 0, 1]])


def test_streaming_confusion_matrix_fixed_labels():
    metric = StreamingConfusionMatrix(labels=[0, 1, 2])
    metric.update([0, 0], [0, 1])
    assert metric.result().shape == (3, 3)


@pytest.mark.parametrize("n_chunks", [1, 5, 40])
def test_streaming_auc_matches_batch(n_chunks):
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 400)
    y_score = rng.integers(0, 20, 400) / 20.0
    metric = StreamingAUC()
    for idx in _chunks(400, n_chunks):
        metric.update(y_true[idx], y_score[idx])

    assert np.isclose(metric.result(), roc_auc_score(y_true, y_score))
    assert np.isclose(metric.result(), _brute_force_auc(y_true, y_score))


def test_streaming_auc_memory_is_per_distinct_score():
    rng = np.random.default_rng(4)
    metric = StreamingAUC()
    for _ in range(50):
        metric.update(rng.integers(0, 2, 100), rng.integers(0, 5, 100) / 4)
    assert metric._total[0].size == 5          # 5000 samples, 5 scores


def test_streaming_auc_roc_curve():
    metric = StreamingAUC().update([0, 1], [0.2, 0.7]).update([0, 1], [0.4, 0.9])
    fpr, tpr, _ = metric.roc_curve()
    assert fpr[0] == 0 and tpr[-1] == 1
    assert metric.result() == 1.0


# ---------------------------------------------------------
# Rolling windows and reset
# ---------------------------------------------------------

def test_rolling_accuracy_uses_last_chunks():
    metric = StreamingAccuracy(window=2)
    metric.update([1, 1], [0, 0])     # 0% - falls out of the window
    metric.update([1, 1], [1, 1])     # 100%
    metric.update([1, 1], [1, 0])     # 50%
    assert metric.result() == 0.75
    assert metric.n_updates_ == 3


def test_rolling_f1_window():
    metric = StreamingF1(window=1)
    metric.update([1, 1], [0, 0])
    metric.update([1, 1], [1, 1])
    assert metric.result() == 1.0


def test_rolling_confusion_matrix_window():
    metric = StreamingConfusionMatrix(window=1)
    metric.update([0, 0], [0, 0])
    metric.update([1, 1], [1, 0])
    assert np.array_equal(metric.result(), [[0, 0], [1, 1]])


def test_rolling_auc_window():
    metric = StreamingAUC(window=1)
    metric.update([0, 1], [0.9, 0.1])     # AUC 0, then evicted
    metric.update([0, 1], [0.1, 0.9])     # AUC 1
    assert metric.result() == 1.0


def test_reset_starts_over(labels):
    y_true, y_pred = labels
    metric = StreamingAccuracy().update(y_true, y_pred)
    metric.reset()
    metric.update([1, 0], [1, 0])
    assert metric.result() == 1.0
    assert metric.n_samples_seen_ == 2


def test_update_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        StreamingAccuracy().update([1, 0], [1])


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        StreamingAccuracy(window=0)
