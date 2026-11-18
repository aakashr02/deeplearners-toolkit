"""Tests for StreamTrainer and estimate_memory."""

import warnings
from collections import deque

import numpy as np
import pytest

from numcompute.metrics import accuracy, f1
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.stream import StreamTrainer, estimate_memory
from numcompute.tree import DecisionTreeClassifier


@pytest.fixture(autouse=True)
def _numpy_warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


@pytest.fixture
def chunks():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return [(X[idx], y[idx]) for idx in np.array_split(np.arange(2000), 10)]


def _pipeline(buffer_size=400):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size)),
    ])


class AlwaysOne:
    """A model that can predict before any training (no buffer)."""

    def __init__(self):
        self.seen = 0

    def partial_fit(self, X, y):
        self.seen += len(y)
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)


# =========================================================
# Prequential (test-then-train) evaluation
# =========================================================

def test_chunk_is_scored_before_it_is_trained_on(chunks):
    trainer = StreamTrainer(_pipeline(buffer_size=200))
    trainer.process_chunk(*chunks[0])

    first = trainer.history_[0]
    assert first["scored"] is False                  # nothing trained yet
    assert np.isnan(first["chunk_accuracy"])
    assert first["retrained"] is True                # trained afterwards

    second = trainer.process_chunk(*chunks[1])
    assert second["scored"] is True


def test_cumulative_accuracy_matches_manual_test_then_train(chunks):
    trainer = StreamTrainer(_pipeline()).run(chunks)

    manual = _pipeline()
    y_all, pred_all = [], []
    for X, y in chunks:
        try:
            pred_all.append(manual.predict(X))
            y_all.append(y)
        except Exception:
            pass
        manual.partial_fit(X, y)

    expected = accuracy(np.concatenate(y_all), np.concatenate(pred_all))
    assert np.isclose(trainer.summary()["cumulative_accuracy"], expected)
    assert np.isclose(trainer.get_metric("cumulative_accuracy")[-1], expected)


def test_chunk_metrics_match_one_shot_functions(chunks):
    trainer = StreamTrainer(AlwaysOne(), metrics=("accuracy", "f1"))
    for X, y in chunks[:3]:
        entry = trainer.process_chunk(X, y)
        assert np.isclose(entry["chunk_accuracy"], accuracy(y, np.ones_like(y)))
        assert np.isclose(entry["chunk_f1"], f1(y, np.ones_like(y)))


def test_model_that_predicts_immediately_is_scored_from_chunk_zero(chunks):
    trainer = StreamTrainer(AlwaysOne()).run(chunks)
    assert all(entry["scored"] for entry in trainer.history_)


def test_rolling_metrics_cover_last_window(chunks):
    trainer = StreamTrainer(AlwaysOne(), window=2).run(chunks)
    chunk_acc = trainer.get_metric("chunk_accuracy")
    rolling = trainer.get_metric("rolling_accuracy")

    sizes = np.array([len(y) for _, y in chunks], dtype=float)
    expected = ((chunk_acc[-1] * sizes[-1] + chunk_acc[-2] * sizes[-2])
                / (sizes[-1] + sizes[-2]))
    assert np.isclose(rolling[-1], expected)


def test_works_with_a_bare_tree(chunks):
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=300)).run(chunks)
    # 10 chunks of 200: the buffer passes 300 on every second chunk.
    assert trainer.summary()["n_retrains"] == 5
    assert trainer.summary()["cumulative_accuracy"] > 0.85


# =========================================================
# Logged fields
# =========================================================

def test_log_entry_fields(chunks):
    trainer = StreamTrainer(_pipeline(), metrics=("accuracy", "f1", "auc"),
                            window=3)
    entry = trainer.process_chunk(*chunks[0])
    for key in ("chunk", "n_samples", "scored", "predict_time", "fit_time",
                "retrained", "n_buffered", "memory_bytes",
                "chunk_accuracy", "cumulative_accuracy", "rolling_accuracy",
                "chunk_f1", "cumulative_auc"):
        assert key in entry


def test_retrain_flag_and_buffer_count(chunks):
    trainer = StreamTrainer(_pipeline(buffer_size=400)).run(chunks[:4])
    assert trainer.get_metric("retrained").tolist() == [0, 1, 0, 1]
    assert trainer.get_metric("n_buffered").tolist() == [200, 0, 200, 0]


def test_memory_grows_with_buffer_and_drops_after_retrain(chunks):
    trainer = StreamTrainer(_pipeline(buffer_size=600)).run(chunks[:4])
    memory = trainer.get_metric("memory_bytes")
    assert memory[1] > memory[0]          # buffer holds 400 rows vs 200
    assert memory[2] < memory[1]          # retrain cleared the buffer


def test_larger_buffer_uses_more_memory(chunks):
    small = StreamTrainer(_pipeline(buffer_size=200)).run(chunks)
    large = StreamTrainer(_pipeline(buffer_size=1000)).run(chunks)
    assert (large.get_metric("memory_bytes").max()
            > small.get_metric("memory_bytes").max())


def test_peak_memory_tracking(chunks):
    trainer = StreamTrainer(_pipeline(), track_peak_memory=True)
    entry = trainer.process_chunk(*chunks[0])
    assert entry["peak_memory_bytes"] > 0


def test_auc_logging(chunks):
    trainer = StreamTrainer(_pipeline(), metrics=("auc",)).run(chunks)
    auc_values = trainer.get_metric("chunk_auc")
    assert np.all(np.isnan(auc_values[:2]))         # before the first retrain
    assert np.all(auc_values[2:] > 0.8)


def test_auc_on_single_class_chunk_is_nan_without_warning(chunks):
    trainer = StreamTrainer(_pipeline(buffer_size=200), metrics=("auc",))
    trainer.process_chunk(*chunks[0])
    X, _ = chunks[1]
    entry = trainer.process_chunk(X, np.ones(len(X), dtype=int))
    assert np.isnan(entry["chunk_auc"])


def test_auc_when_positive_class_not_yet_seen_by_model():
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2),
                            metrics=("auc",), positive_label=1)
    trainer.process_chunk(np.array([[0.0], [1.0]]), np.array([0, 0]))
    entry = trainer.process_chunk(np.array([[0.5], [2.0]]), np.array([0, 1]))
    assert entry["scored"] and entry["chunk_auc"] == 0.5


def test_mse_metric_for_regression_style_model():
    class Mean:
        def partial_fit(self, X, y):
            self.mean = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), getattr(self, "mean", 0.0))

    trainer = StreamTrainer(Mean(), metrics=("mse",))
    trainer.process_chunk(np.zeros((2, 1)), np.array([1.0, 3.0]))
    entry = trainer.process_chunk(np.zeros((2, 1)), np.array([2.0, 4.0]))
    # Predicts 2.0 (mean of the first chunk): ((2-2)^2 + (4-2)^2) / 2 = 2.
    assert np.isclose(entry["chunk_mse"], 2.0)


# =========================================================
# Edge cases
# =========================================================

def test_empty_chunk_is_logged_but_not_scored(chunks):
    trainer = StreamTrainer(AlwaysOne())
    entry = trainer.process_chunk(np.empty((0, 3)), np.array([]))
    assert entry["n_samples"] == 0 and entry["scored"] is False
    assert np.isnan(entry["cumulative_accuracy"])

    trainer.process_chunk(*chunks[0])
    assert trainer.n_chunks_seen_ == 2


def test_mismatched_chunk_lengths_raise(chunks):
    X, y = chunks[0]
    with pytest.raises(ValueError, match="samples"):
        StreamTrainer(AlwaysOne()).process_chunk(X, y[:-1])


def test_run_rejects_items_that_are_not_pairs():
    X, y = np.ones((2, 2)), np.ones(2)
    with pytest.raises(ValueError, match="pair"):
        StreamTrainer(AlwaysOne()).run([(X, y, "extra")])
    with pytest.raises(ValueError, match="pair"):
        StreamTrainer(AlwaysOne()).run([42])


def test_accuracy_is_always_logged():
    trainer = StreamTrainer(AlwaysOne(), metrics="f1")
    assert trainer.metrics == ("accuracy", "f1")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"metrics": ("accuracy", "r2")}, ValueError),
        ({"window": 0}, ValueError),
        ({"track_peak_memory": 1}, TypeError),
    ],
)
def test_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        StreamTrainer(AlwaysOne(), **kwargs)


def test_model_without_streaming_interface_is_rejected():
    class BatchOnly:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X

    with pytest.raises(TypeError, match="partial_fit"):
        StreamTrainer(BatchOnly())


def test_auc_requires_predict_proba():
    with pytest.raises(ValueError, match="predict_proba"):
        StreamTrainer(AlwaysOne(), metrics=("auc",))


def test_get_metric_unknown_key(chunks):
    trainer = StreamTrainer(AlwaysOne()).run(chunks[:1])
    with pytest.raises(KeyError, match="available"):
        trainer.get_metric("chunk_r2")


def test_get_metric_and_summary_before_any_chunk():
    trainer = StreamTrainer(AlwaysOne())
    assert trainer.get_metric("chunk_accuracy").size == 0
    summary = trainer.summary()
    assert summary["n_chunks"] == 0 and np.isnan(summary["cumulative_accuracy"])


def test_reset_clears_log_but_keeps_model(chunks):
    model = AlwaysOne()
    trainer = StreamTrainer(model).run(chunks[:3])
    trainer.reset()
    assert trainer.history_ == [] and trainer.n_samples_seen_ == 0
    assert model.seen == sum(len(y) for _, y in chunks[:3])


def test_model_errors_propagate(chunks):
    trainer = StreamTrainer(_pipeline())
    trainer.process_chunk(*chunks[0])
    X, y = chunks[1]
    with pytest.raises(ValueError, match="features"):
        trainer.process_chunk(X[:, :2], y)


def test_partial_fit_alias(chunks):
    trainer = StreamTrainer(AlwaysOne())
    trainer.partial_fit(*chunks[0])
    assert trainer.n_chunks_seen_ == 1


# =========================================================
# estimate_memory
# =========================================================

def test_estimate_memory_counts_arrays_in_containers():
    class Holder:
        pass

    holder = Holder()
    holder.a = np.zeros(10)                       # 80 bytes
    holder.items = [np.zeros(5), {"x": np.zeros(2)}]  # 40 + 16
    holder.queue = deque([np.zeros(1)])           # 8
    holder.name = "not counted"
    assert estimate_memory(holder) == 80 + 40 + 16 + 8


def test_estimate_memory_handles_cycles_and_shared_arrays():
    class Node:
        pass

    a, b = Node(), Node()
    shared = np.zeros(4)
    a.other, b.other = b, a
    a.data, b.data = shared, shared
    assert estimate_memory(a) == shared.nbytes


def test_estimate_memory_object_arrays():
    arr = np.empty(2, dtype=object)
    arr[0] = np.zeros(3)
    arr[1] = "text"
    assert estimate_memory(arr) == arr.nbytes + 24
