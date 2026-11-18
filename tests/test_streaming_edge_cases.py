"""Edge cases for streaming preprocessing and streaming metrics."""

import warnings

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
    auc,
    roc_auc_score,
    roc_curve,
)
from numcompute.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
from numcompute.utils import NotFittedError

SCALERS = [StandardScaler, MinMaxScaler]
TRANSFORMERS = SCALERS + [SimpleImputer, OneHotEncoder]
STREAMING_METRICS = [
    StreamingAccuracy, StreamingPrecision, StreamingRecall, StreamingF1,
    StreamingMSE, StreamingConfusionMatrix, StreamingAUC,
]


@pytest.fixture(autouse=True)
def _numpy_warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


# =========================================================
# Preprocessing: empty chunks and state
# =========================================================

@pytest.mark.parametrize("make", TRANSFORMERS)
def test_empty_chunk_is_a_no_op(make):
    component = make().partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    before = component.transform(np.array([[1.0, 2.0]]))

    component.partial_fit(np.empty((0, 2)))

    assert component.n_samples_seen_ == 2
    assert np.array_equal(component.transform(np.array([[1.0, 2.0]])), before)


@pytest.mark.parametrize("make", TRANSFORMERS)
def test_empty_first_chunk_then_data(make):
    component = make().partial_fit(np.empty((0, 2)))
    component.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert component.n_samples_seen_ == 2


@pytest.mark.parametrize("make", TRANSFORMERS)
def test_empty_chunk_with_wrong_width_is_rejected(make):
    component = make().partial_fit(np.ones((2, 2)))
    with pytest.raises(ValueError, match="features"):
        component.partial_fit(np.empty((0, 3)))


@pytest.mark.parametrize("make", TRANSFORMERS)
def test_empty_batch_fit_is_rejected_but_empty_chunk_is_not(make):
    with pytest.raises(ValueError, match="at least one sample"):
        make().fit(np.empty((0, 2)))
    make().partial_fit(np.empty((0, 2)))          # no error


@pytest.mark.parametrize("make", TRANSFORMERS)
def test_transform_before_partial_fit_raises(make):
    with pytest.raises(NotFittedError):
        make().transform(np.ones((1, 2)))


@pytest.mark.parametrize("make", TRANSFORMERS)
def test_transform_after_only_empty_chunks_raises(make):
    component = make().partial_fit(np.empty((0, 2)))
    with pytest.raises(NotFittedError):
        component.transform(np.ones((1, 2)))


@pytest.mark.parametrize("make", TRANSFORMERS)
def test_reset_then_new_width(make):
    component = make().partial_fit(np.ones((2, 2)))
    component.reset()
    component.partial_fit(np.ones((2, 3)))
    assert component.n_features_in_ == 3


@pytest.mark.parametrize("make", SCALERS)
def test_scalers_reject_1d_chunks(make):
    with pytest.raises(ValueError, match="2 dimensions"):
        make().partial_fit(np.array([1.0, 2.0]))


# =========================================================
# Preprocessing: zero-variance and constant chunks
# =========================================================

def test_standard_scaler_column_constant_across_all_chunks():
    scaler = StandardScaler()
    for _ in range(5):
        scaler.partial_fit(np.array([[0.1, 1.0], [0.1, 2.0]]))

    assert scaler.std_[0] == 0.0
    assert scaler.scale_[0] == 1.0
    assert np.allclose(scaler.transform([[0.1, 1.5]])[0, 0], 0.0)


def test_standard_scaler_single_row_chunks():
    data = np.array([[1.0], [2.0], [3.0], [4.0]])
    scaler = StandardScaler()
    scaler.partial_fit(data[:1])
    assert scaler.std_[0] == 0.0                     # one sample so far
    for row in data[1:]:
        scaler.partial_fit(row[None, :])
    assert np.isclose(scaler.std_[0], data.std())


def test_minmax_constant_then_varying():
    scaler = MinMaxScaler().partial_fit(np.full((3, 1), 5.0))
    assert np.array_equal(scaler.transform([[5.0]]), [[0.0]])
    scaler.partial_fit(np.array([[15.0]]))
    assert np.allclose(scaler.transform([[10.0]]), [[0.5]])


# =========================================================
# SimpleImputer edge cases
# =========================================================

@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_imputer_all_nan_column_never_leaves_nan(strategy):
    X = np.array([[1.0, np.nan], [3.0, np.nan]])
    imputer = SimpleImputer(strategy).partial_fit(X)
    out = imputer.transform(X)

    assert not np.isnan(out).any()
    assert np.all(np.isfinite(imputer.statistics_))


def test_imputer_all_nan_column_uses_fill_value_fallback():
    imputer = SimpleImputer("mean", fill_value=-7.0)
    imputer.partial_fit(np.array([[1.0, np.nan]]))
    assert np.array_equal(imputer.transform([[np.nan, np.nan]]),
                          [[1.0, -7.0]])


def test_imputer_all_nan_column_recovers_when_data_arrives():
    imputer = SimpleImputer("mean").partial_fit(np.array([[np.nan], [np.nan]]))
    imputer.partial_fit(np.array([[4.0], [6.0]]))
    assert imputer.statistics_[0] == 5.0


def test_imputer_chunk_with_no_missing_values():
    imputer = SimpleImputer("median").partial_fit(np.array([[1.0], [2.0]]))
    assert np.array_equal(imputer.transform([[5.0]]), [[5.0]])


def test_imputer_rejects_infinity():
    with pytest.raises(ValueError, match="infinite"):
        SimpleImputer().partial_fit(np.array([[1.0], [np.inf]]))


@pytest.mark.parametrize("fill_value", ["a", np.nan, np.inf, True])
def test_imputer_constant_rejects_invalid_fill_value(fill_value):
    with pytest.raises(ValueError, match="fill_value"):
        SimpleImputer("constant", fill_value=fill_value)


def test_imputer_most_frequent_negative_and_float_values():
    X = np.array([[-1.5], [-1.5], [2.25], [np.nan]])
    assert SimpleImputer("most_frequent").fit(X).statistics_[0] == -1.5


# =========================================================
# OneHotEncoder edge cases
# =========================================================

def test_onehot_nan_is_not_a_category():
    X = np.array([[1.0], [np.nan], [2.0]])
    encoder = OneHotEncoder().partial_fit(X)

    assert encoder.categories_[0].tolist() == [1.0, 2.0]
    assert np.array_equal(encoder.transform(X), [[1, 0], [0, 0], [0, 1]])


def test_onehot_nan_only_chunk_learns_nothing():
    encoder = OneHotEncoder().partial_fit(np.array([[np.nan]]))
    assert encoder.n_features_out_ == 0
    encoder.partial_fit(np.array([[3.0]]))
    assert encoder.categories_[0].tolist() == [3.0]


def test_onehot_rejects_type_change_between_chunks():
    encoder = OneHotEncoder().partial_fit(np.array([[1], [2]]))
    with pytest.raises(TypeError, match="type"):
        encoder.partial_fit(np.array([["a"]]))


def test_onehot_rejects_type_change_in_transform():
    encoder = OneHotEncoder().partial_fit(np.array([["a"], ["b"]]))
    with pytest.raises(TypeError, match="type"):
        encoder.transform(np.array([[1]]))


def test_onehot_longer_strings_in_later_chunk():
    encoder = OneHotEncoder().partial_fit(np.array([["a"]]))
    encoder.partial_fit(np.array([["longer-category"]]))
    assert encoder.categories_[0].tolist() == ["a", "longer-category"]
    assert np.array_equal(encoder.transform(np.array([["longer-category"]])),
                          [[0, 1]])


def test_onehot_unknown_ignore_after_stream():
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.partial_fit(np.array([["a"]]))
    assert np.array_equal(encoder.transform(np.array([["z"]])), [[0]])


# =========================================================
# Metrics: before any update, empty chunks
# =========================================================

@pytest.mark.parametrize("metric_class", STREAMING_METRICS)
def test_result_before_update_raises_clear_error(metric_class):
    with pytest.raises(NotFittedError, match="update"):
        metric_class().result()


@pytest.mark.parametrize("metric_class", STREAMING_METRICS)
def test_result_after_reset_raises_clear_error(metric_class):
    metric = metric_class().update([1, 0], [1, 0])
    metric.reset()
    with pytest.raises(NotFittedError):
        metric.result()


@pytest.mark.parametrize("metric_class", STREAMING_METRICS)
def test_empty_chunk_does_not_change_result(metric_class):
    metric = metric_class().update([1, 0, 1], [1, 0, 0])
    before = metric.result()
    metric.update([], [])
    assert np.array_equal(metric.result(), before, equal_nan=True)
    assert metric.n_samples_seen_ == 3


def test_accuracy_with_only_empty_chunks_is_nan():
    assert np.isnan(StreamingAccuracy().update([], []).result())


def test_mse_with_only_empty_chunks_is_nan():
    assert np.isnan(StreamingMSE().update([], []).result())


@pytest.mark.parametrize("metric_class", STREAMING_METRICS)
def test_update_rejects_mismatched_shapes(metric_class):
    with pytest.raises(ValueError, match="shape"):
        metric_class().update([1, 0, 1], [1, 0])


# =========================================================
# Metrics: zero denominators
# =========================================================

def test_precision_with_no_predicted_positives():
    assert StreamingPrecision().update([1, 1], [0, 0]).result() == 0.0


def test_recall_with_no_actual_positives():
    assert StreamingRecall().update([0, 0], [1, 0]).result() == 0.0


def test_f1_with_no_positives_anywhere():
    assert StreamingF1().update([0, 0], [0, 0]).result() == 0.0


# =========================================================
# Metrics: rolling windows
# =========================================================

def test_window_larger_than_number_of_chunks():
    metric = StreamingAccuracy(window=10)
    metric.update([1, 1], [1, 0]).update([0], [0])
    assert np.isclose(metric.result(), 2 / 3)


def test_window_with_chunks_of_different_sizes():
    metric = StreamingAccuracy(window=2)
    metric.update([1] * 100, [0] * 100)          # evicted
    metric.update([1] * 3, [1] * 3)
    metric.update([1], [0])
    assert metric.result() == 0.75


def test_rolling_confusion_matrix_label_seen_only_in_evicted_chunk():
    metric = StreamingConfusionMatrix(window=1)
    metric.update([2, 2], [2, 2])                # label 2 then evicted
    metric.update([0, 1], [0, 1])
    result = metric.result()
    assert result.shape == (3, 3)
    assert result[2].sum() == 0 and np.trace(result) == 2


# =========================================================
# Confusion matrix edge cases
# =========================================================

def test_confusion_matrix_fixed_labels_reject_unknown_label():
    metric = StreamingConfusionMatrix(labels=[0, 1])
    with pytest.raises(ValueError, match="label"):
        metric.update([0, 2], [0, 1])


def test_confusion_matrix_string_labels():
    metric = StreamingConfusionMatrix()
    metric.update(["cat", "dog"], ["cat", "cat"])
    metric.update(["bird"], ["bird"])
    assert metric.labels_.tolist() == ["bird", "cat", "dog"]
    assert np.array_equal(metric.result(),
                          [[1, 0, 0], [0, 1, 0], [0, 1, 0]])


def test_confusion_matrix_single_label():
    assert np.array_equal(
        StreamingConfusionMatrix().update([1, 1], [1, 1]).result(), [[2]]
    )


# =========================================================
# ROC / AUC edge cases
# =========================================================

@pytest.mark.parametrize("y_true", [[1, 1, 1], [0, 0, 0]])
def test_roc_auc_score_single_class_raises(y_true):
    with pytest.raises(ValueError, match="both"):
        roc_auc_score(y_true, [0.1, 0.5, 0.9])


@pytest.mark.parametrize("y_true", [[1, 1], [0, 0]])
def test_streaming_auc_single_class_so_far_is_nan(y_true):
    metric = StreamingAUC().update(y_true, [0.2, 0.8])
    assert np.isnan(metric.result())


def test_streaming_auc_becomes_defined_when_second_class_arrives():
    metric = StreamingAUC().update([1, 1], [0.9, 0.8])
    metric.update([0], [0.1])
    assert metric.result() == 1.0


def test_streaming_auc_window_with_single_class_is_nan():
    metric = StreamingAUC(window=1)
    metric.update([0, 1], [0.1, 0.9])
    metric.update([1], [0.5])
    assert np.isnan(metric.result())


def test_roc_rejects_nan_scores():
    with pytest.raises(ValueError, match="NaN"):
        roc_curve([0, 1], [0.5, np.nan])
    with pytest.raises(ValueError, match="NaN"):
        StreamingAUC().update([0, 1], [np.nan, 0.5])


def test_roc_rejects_non_numeric_scores():
    with pytest.raises(TypeError):
        roc_curve([0, 1], ["low", "high"])


def test_roc_accepts_infinite_scores():
    assert roc_auc_score([0, 1], [-np.inf, np.inf]) == 1.0


def test_all_scores_tied_gives_half():
    assert StreamingAUC().update([0, 1, 0, 1], [0.3] * 4).result() == 0.5


def test_auc_rejects_non_monotonic_x():
    with pytest.raises(ValueError, match="monotonic"):
        auc([0.0, 0.5, 0.2, 1.0], [0.0, 0.5, 0.6, 1.0])


def test_auc_accepts_decreasing_x():
    assert auc([1.0, 0.0], [1.0, 0.0]) == 0.5
