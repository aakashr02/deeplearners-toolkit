# NumCompute

**A lightweight numerical computing toolkit built with Python and NumPy.**

NumCompute provides utilities for numerical differentiation, sorting and searching, statistical analysis, ranking, classification and regression metrics, data preprocessing, CSV loading, and pipeline composition.

The package is designed to offer useful numerical and data-processing operations through a simple API, without relying on libraries such as SciPy, pandas, or scikit-learn.

## Features

- **Sorting and searching:** Stable sorting, multi-key sorting, binary search, top-\(k\) selection, and quick selection.
- **Numerical differentiation:** Finite-difference gradient and Jacobian estimation, including batched variants.
- **Statistics:** Mean, median, standard deviation, variance, minimum, maximum, histogram, and quantile calculations, plus streaming (chunk-by-chunk) running statistics, histograms and quantiles.
- **Ranking:** Multiple ranking methods, configurable ordering, axis support, and NaN handling.
- **Evaluation metrics:** Accuracy, confusion matrix, precision, recall, F1 score, and mean squared error.
- **Preprocessing:** Standard scaling, min-max scaling, and one-hot encoding.
- **Pipelines:** Sequential preprocessing and optional final estimator integration, with streaming `partial_fit`.
- **Decision tree:** `DecisionTreeClassifier` (Gini or entropy) with buffered streaming retraining.
- **Stream training:** `StreamTrainer` runs test-then-train evaluation over chunks and logs per-chunk, cumulative and rolling metrics, memory footprint and timing.
- **Visualisation:** matplotlib plots of metrics over time, model comparisons and predictions against ground truth.
- **Input/output:** CSV loading into NumPy arrays, including chunked streaming and skip/fill handling of missing values.
- **Utilities:** Numerically stable activations (sigmoid, softmax, log-softmax, logsumexp), distance functions and mini-batching.
- **Percentiles:** `percentile()` with linear, lower, higher, midpoint and nearest interpolation.

## Installation

### Prerequisites

- Python 3.9 or later
- Git
- NumPy

### Install directly from Git

```bash
python -m pip install "git+https://github.com/Sammie1875/deeplearners-toolkit.git"
```


### Install from a local clone

```bash
git clone https://github.com/Sammie1875/deeplearners-toolkit.git
cd deeplearners-toolkit
python -m pip install .
```

Plotting (`visualise.py`) needs matplotlib, which is an optional extra; the rest of the package needs only NumPy:

```bash
python -m pip install ".[viz]"     # matplotlib for plots
python -m pip install ".[test]"    # pytest + matplotlib to run the tests
```

For development, install in editable mode:

```bash
python -m pip install -e .
```

## Quick start

### Sorting and searching

```python
import numpy as np
from numcompute import stable_sort, binary_search

values = np.array([5, 2, 8, 1, 3])

sorted_values = stable_sort(values)
position, found = binary_search(sorted_values, 3)

print(sorted_values)
# [1 2 3 5 8]

print(position, found)
# 2 True
```

### Statistical calculations

```python
import numpy as np
from numcompute import mean, median, std

values = np.array([10, 20, 30, 40, 50])

print(mean(values))    # 30.0
print(median(values))  # 30.0
print(std(values))     # Population standard deviation
```

### Numerical differentiation

```python
import numpy as np
from numcompute import gradient

def f(x):
    return np.sum(x ** 2)

x = np.array([2.0, 3.0])

g = gradient(f, x, method="central")
print(g)
# Approximately [4. 6.]
```

### Classification metrics

```python
import numpy as np
from numcompute import accuracy, precision, recall, f1

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0])

print(accuracy(y_true, y_pred))  # 0.8
print(precision(y_true, y_pred)) # 1.0
print(recall(y_true, y_pred))    # 2/3
print(f1(y_true, y_pred))        # 0.8
```

### Data preprocessing

```python
import numpy as np
from numcompute import StandardScaler

X = np.array([
    [10.0, 100.0],
    [20.0, 200.0],
    [30.0, 300.0],
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

### Pipeline composition

```python
from numcompute import Pipeline, StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
])

X_transformed = pipe.fit_transform(X)
```

The pipeline supports preprocessing components that implement `fit()` and `transform()`. Its optional estimator support depends on the estimator implementing the expected `fit()` and `predict()` interface.

## API overview

### 1. `io.py` — Input/output

| Function | Description |
|---|---|
| `read_csv(path, delimiter=",", missing_value=np.nan, dtype=float, missing="fill", skip_header=0)` | Loads a CSV file into a 2D array using NumPy's `genfromtxt`. |
| `read_csv_chunks(path, chunk_size=1000, ...)` | Streams the file lazily as 2D chunks of at most `chunk_size` rows; same options as `read_csv`. |

`missing="fill"` replaces empty fields with `missing_value`, and `missing="skip"` drops rows that contain them. Stacking all chunks with `np.vstack` reproduces `read_csv` exactly.

```python
from numcompute import read_csv_chunks

total = 0.0
for chunk in read_csv_chunks("large.csv", chunk_size=10_000, missing="skip"):
    total += chunk.sum()          # only one chunk is in memory at a time
```

### 2. `sort_search.py` — Sorting and searching

| Function | Description |
|---|---|
| `stable_sort(values, axis=-1)` | Sorts array elements along a specified axis. |
| `multi_key_sort(data, ascending=True)` | Returns indices that order rows using multiple keys. |
| `binary_search(sorted_array, x)` | Returns the insertion index and whether the target exists. |
| `top_k(values, k, largest=True, return_indices=True)` (alias `topk`) | Selects the \(k\) largest or smallest elements or their indices using `np.argpartition`, ordered best-first. |
| `quick_select(values, k, largest=True, return_index=True)` | Selects the element at zero-based rank \(k\) with an explicit, educational quickselect (random pivot, three-way partition); never calls `np.partition`. |

**Selection contract** (shared by `top_k` and `quick_select`):

- Input must be a 1D `np.ndarray` without NaN (`ValueError` otherwise); infinities are allowed.
- Ties are broken by the lower original index, so results are deterministic.
- `top_k` returns a length-\(k\) array, best-first (descending for `largest=True`); `quick_select` returns a Python `int` index or a value of the input dtype.
- `quick_select(values, r) == top_k(values, r + 1)[-1]` for every valid rank.
- Non-integer or Boolean `k` raises `TypeError`; out-of-range `k` raises `ValueError`.

### 3. `optim.py` — Numerical differentiation

| Function | Description |
|---|---|
| `gradient(f, x, h=1e-5, method="central")` (alias `grad`) | Estimates the gradient of a scalar-valued function. |
| `jacobian(F, x, h=1e-5, method="central")` | Estimates the Jacobian of a vector-valued function. |
| `gradient_batched(f, x, h=1e-5, method="central")` | Estimates gradients using batched function evaluations. |
| `jacobian_batched(F, x, h=1e-5, method="central")` | Estimates Jacobians using batched function evaluations. |

Supported finite-difference methods are central and forward differences.

### 4. `stats.py` — Descriptive statistics

| Function | Description |
|---|---|
| `mean(data, axis=None, ignore_nan=True)` | Calculates arithmetic means. |
| `median(data, axis=None, ignore_nan=True)` | Calculates medians. |
| `std(data, axis=None, ignore_nan=True, ddof=0)` | Calculates standard deviations. |
| `minimum(data, axis=None, ignore_nan=True)` | Calculates minimum values. |
| `maximum(data, axis=None, ignore_nan=True)` | Calculates maximum values. |
| `histogram(data, bins=10, range=None, ignore_nan=True)` | Computes histogram counts and bin edges. |
| `quantile(data, q, axis=None, ignore_nan=True)` | Computes quantiles. |
| `variance(data, axis=None, ignore_nan=True, ddof=0)` | Calculates variances. |

#### Streaming statistics

| Class / function | Description |
|---|---|
| `RunningStats(ignore_nan=True)` | Exact running count, mean, variance, std, min and max per feature, using the parallel Welford algorithm. Supports `merge()` for combining partial results. |
| `StreamingHistogram(bins=10, range=None, window=None)` | Fixed-edge histogram that accumulates per chunk; `window=W` keeps only the last W chunks. Also tracks out-of-range and NaN counts. |
| `StreamingQuantile(q=0.5, interpolation="linear", window=None, max_centroids=None)` | Exact quantiles over all data, over the last W chunks (`window`), or over the whole stream in bounded memory (`max_centroids=K`: exact until 2K values, then K weighted centroids; about 0.8% max error with K=256 on 100k values). |
| `update_stats(X_chunk, state=None)` | Functional form; creates a `RunningStats` when `state` is None. |

Every streaming object has `update_stats(X_chunk)` (alias `partial_fit`), `reset()` and `result()`. A chunk is 1D (one feature, so results are scalars) or 2D `(n_samples, n_features)`. The feature count is fixed by the first non-empty chunk. Empty chunks are allowed, and infinite values raise `ValueError`.

```python
from numcompute import RunningStats, read_csv_chunks

stats = RunningStats.from_chunks(read_csv_chunks("large.csv", chunk_size=10_000))
stats.mean, stats.std(ddof=1)
```

Chunks are merged as `mean = mean_a + delta * n_b / n` and `M2 = M2_a + M2_b + delta^2 * n_a * n_b / n`. These never form `sum(x^2)`, so the variance stays accurate even for data such as `1e9 + noise`, where the naive `E[x^2] - E[x]^2` formula fails. The result does not depend on how the stream is split into chunks.

### 5. `rank.py` — Ranking and percentile utilities

| Function | Description |
|---|---|
| `rankdata(values, method="average", ascending=True, axis=None, nan_policy="raise")` (alias `rank`) | Computes observation ranks. |
| `percentile(data, q, interpolation="linear", axis=None, nan_policy="raise")` | Computes the q-th percentile(s), \(q \in [0, 100]\), with `linear`, `lower`, `higher`, `midpoint` or `nearest` interpolation. |
| `percentile_rank(values, score, kind="rank", nan_policy="raise")` | Calculates the percentile rank of one or more scores. |

`percentile` sorts each slice once and gathers every requested percentile in one vectorised step. Its results match `np.percentile(..., method=...)`, and with `nan_policy="omit"` they match `np.nanpercentile`. A scalar `q` returns a float when `axis=None`. An array `q` returns shape `q.shape + reduced_shape`.

```python
from numcompute import percentile

percentile([1, 2, 3, 4], 50)                          # 2.5
percentile([1, 2, 3, 4], 50, interpolation="lower")   # 2.0
percentile([1, 2, 3, 4], 50, interpolation="higher")  # 3.0
```

Supported ranking methods are `average`, `dense`, `ordinal`, `min`, and `max`. NaN policies are `raise`, `omit`, and `propagate`.

Supported percentile-rank kinds are `rank`, `weak`, `strict`, and `mean`.

### 6. `metrics.py` — Evaluation metrics

| Function | Description |
|---|---|
| `accuracy(y_true, y_pred)` | Calculates the proportion of correct predictions. |
| `confusion_matrix(y_true, y_pred, labels=None)` | Counts true versus predicted class combinations. |
| `precision(y_true, y_pred, positive_label=1)` | Calculates positive predictive value. |
| `recall(y_true, y_pred, positive_label=1)` | Calculates the proportion of actual positives identified. |
| `f1(y_true, y_pred, positive_label=1)` | Calculates the harmonic mean of precision and recall. |
| `mse(y_true, y_pred)` | Calculates mean squared error for numerical predictions. |

| `roc_curve(y_true, y_score, positive_label=1)` | ROC points (`fpr`, `tpr`, `thresholds`) using every distinct score as a threshold. |
| `auc(x, y)` | Trapezoidal area under a curve. |
| `roc_auc_score(y_true, y_score, positive_label=1)` | ROC AUC; tied scores count as one half. |

Precision, recall, and F1 use a specified positive class and are intended for binary classification.

#### Streaming metrics

`StreamingAccuracy`, `StreamingPrecision`, `StreamingRecall`, `StreamingF1`, `StreamingMSE`, `StreamingConfusionMatrix` and `StreamingAUC` all provide `update(y_true_chunk, y_pred_chunk)`, `result()` and `reset()`.
- Pass `window=W` to any of them for a rolling metric over the last W chunks.
- They store only per-chunk counts or sums, so results equal the one-shot function on all data seen. F1 is computed from summed counts, not by averaging per-chunk F1 scores.
- `result()` before any `update` raises `NotFittedError`. Empty chunks are accepted. Accuracy and MSE over zero samples are NaN; precision, recall and F1 with a zero denominator are 0.0, as in the one-shot functions.
- `StreamingAUC.result()` is NaN while only one class has been seen, and the one-shot `roc_curve`/`roc_auc_score` raise `ValueError` in that case. NaN scores are rejected; infinite scores are allowed.
- The confusion matrix grows its label set as new labels appear.
- `StreamingAUC` keeps positive and negative counts per distinct score, which is exact and small for tree leaf probabilities.

```python
from numcompute import StreamingAccuracy

acc, recent = StreamingAccuracy(), StreamingAccuracy(window=5)
for y_chunk, pred_chunk in stream:
    acc.update(y_chunk, pred_chunk)
    recent.update(y_chunk, pred_chunk)
print(acc.result(), recent.result())
```

### 7. `preprocessing.py` — Data preprocessing

| Class | Methods | Description |
|---|---|---|
| `StandardScaler` | `fit`, `partial_fit`, `transform`, `fit_transform`, `reset` | Centers features and scales by standard deviation; running mean and variance via `RunningStats` (Welford). |
| `MinMaxScaler(feature_range=(0, 1), clip=False)` | same | Scales features to a specified range; running min and max. `clip=True` keeps streamed values inside the range. |
| `SimpleImputer(strategy="mean", fill_value=None, max_centroids=512)` | same | Replaces NaN with a per-feature `mean`, `median`, `most_frequent` or `constant` value, updated on the fly. |
| `OneHotEncoder(handle_unknown="error")` | same | Encodes categorical columns as binary features. Categories expand incrementally and are append-only, so existing output columns never move. |

```python
from numcompute import SimpleImputer, StandardScaler, read_csv_chunks

imputer, scaler = SimpleImputer("median"), StandardScaler()
for chunk in read_csv_chunks("stream.csv", chunk_size=500):
    filled = imputer.partial_fit(chunk).transform(chunk)
    scaled = scaler.partial_fit(filled).transform(filled)
```

Fit preprocessing components on training data and reuse the fitted components to transform new data.

**Transformer contract** (shared validation lives in `utils.py`):

- `partial_fit` on an empty chunk does nothing, but `fit` on an empty batch raises `ValueError`.
- `SimpleImputer` never leaves NaN: a column with no observed values yet uses `fill_value` (default 0.0) until real values arrive.
- `OneHotEncoder` treats NaN as missing (an all-zero block), and raises `TypeError` if a column changes value type between chunks.

- `X` must be 2D with shape `(n_samples, n_features)` and at least one sample. Scalers also require real, finite numbers (`TypeError` / `ValueError`).
- `fit` records `n_features_in_` and returns `self`. `transform` raises `ValueError` if `X` has a different number of columns, instead of silently broadcasting.
- Calling `transform` before `fit` raises `numcompute.NotFittedError`, which subclasses both `RuntimeError` and `AttributeError`.
- Constant features are safe: the zero standard deviation or range is replaced by 1 in `scale_`. A constant column becomes 0 with `StandardScaler` and `feature_range[0]` with `MinMaxScaler`, never NaN.
- `OneHotEncoder` raises `ValueError` for categories not seen during `fit`, unless `handle_unknown="ignore"` (encoded as all zeros).

### 8. `utils.py` — Activations, distances, batching and validation

| Function | Description |
|---|---|
| `sigmoid(x)` | Logistic function; branch-wise form never overflows (`sigmoid(±1000)` gives exactly 1 / 0). |
| `relu(x)`, `leaky_relu(x, alpha=0.01)`, `tanh(x)` | Elementwise activations, same shape as input. |
| `softmax(x, axis=-1)`, `log_softmax(x, axis=-1)` | Max-shifted, so logits of 1000+ are safe; `-inf` logits get probability 0. |
| `logsumexp(x, axis=None, keepdims=False)` | `m + log(sum(exp(x - m)))`; handles `±inf` and NaN explicitly. |
| `euclidean_distance(a, b)`, `manhattan_distance(a, b)`, `cosine_distance(a, b)` | Distances along the last axis, with broadcasting over leading axes. |
| `pairwise_distances(X, Y=None, metric="euclidean")` | `(n, m)` distance matrix; Euclidean uses one matrix product and clips rounding error. |
| `batch_iterator(*arrays, batch_size, shuffle=False, seed=None, drop_last=False)` | Yields aligned mini-batches (views when not shuffled). |
| `validate_2d`, `check_is_fitted`, `validate_int`, `validate_bool`, `NotFittedError` | Shared validation used by every module. |

All activations accept non-contiguous views, return float64, and raise `TypeError` for non-numeric input.

### 9. `pipeline.py` — Pipeline composition

| Class or method | Description |
|---|---|
| `Pipeline(steps)` | Creates an ordered sequence of named components. |
| `fit_transform(X)` | Fits preprocessing steps and returns transformed data. |
| `fit(X, y)` | Fits preprocessing steps and an optional final estimator. |
| `transform(X)` | Applies fitted preprocessing steps. |
| `predict(X)` | Transforms input and predicts using the final estimator. |
| `predict_proba(X)` | Transforms input and returns the final estimator's class probabilities. |
| `partial_fit(X_chunk, y_chunk=None)` | Streams one chunk. Each transformer runs `partial_fit` and then `transform`; the final estimator runs `partial_fit(X, y)`. Empty chunks are no-ops. |

```python
from numcompute import DecisionTreeClassifier, Pipeline, StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])
for X_chunk, y_chunk in stream:
    pipe.partial_fit(X_chunk, y_chunk)
pipe.predict(X_new)
```

Every step must implement `partial_fit` for streaming; this is checked before any step is updated. A `OneHotEncoder` that gains categories mid-stream widens its output, so a tree after it raises a `ValueError` about the feature count; fix the categories up front or use `handle_unknown="ignore"` with a warm-up chunk.

### 10. `tree.py` — Decision tree with buffered retraining

| Method / attribute | Description |
|---|---|
| `DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=2, max_features=None, buffer_size=500, random_state=None)` | Depth-limited classification tree. `max_features` accepts an int, a float fraction, `"sqrt"`, `"log2"` or None. |
| `fit(X, y)` | Trains from scratch; also clears the streaming buffer. |
| `partial_fit(X_chunk, y_chunk)` | Appends the chunk to a buffer. When the buffer holds at least `buffer_size` samples, the tree is retrained on the buffered data and the buffer is cleared. |
| `predict(X)`, `predict_proba(X)`, `score(X, y)`, `apply(X)` | Use the most recently trained tree. Before the first retrain they raise `NotFittedError`, stating how many samples are buffered. |
| `get_depth()`, `get_n_leaves()`, `n_retrains_`, `n_buffered_`, `last_fit_time_` | Introspection and streaming counters. |

- **Split search, vectorised per node:** all candidate features are sorted at once, cumulative one-hot class counts give left and right impurity for every threshold, and one `argmin` picks the best split.
- **Thresholds:** midpoints computed as `low/2 + high/2`, so they never overflow; if adjacent floats round onto the upper value, the lower value is used instead.
- **Growth and prediction:** the tree grows with an explicit stack (no recursion limit), and prediction moves all samples one level per step.
- **Stable classes:** `classes_` covers every label seen by `partial_fit`, so `predict_proba` columns stay stable even when a buffer lacks a class.
- **Input checks:** NaN and infinity in X are rejected; impute missing values first.

### 11. `stream.py` — Streaming training and logging

| Class / function | Description |
|---|---|
| `StreamTrainer(model, metrics=("accuracy",), positive_label=1, window=None, track_peak_memory=False)` | Wraps any model with `partial_fit` and `predict` (a `Pipeline` or a tree). `metrics` can include `accuracy`, `precision`, `recall`, `f1`, `mse` and `auc`; accuracy is always logged. |
| `score_chunk(X, y)` | Scores the current model on a chunk and updates the cumulative and rolling metrics. Metrics are NaN while the model cannot predict yet. |
| `fit_chunk(X, y)` | `partial_fit`, recording fit time, whether the model retrained, the samples still buffered, and the memory footprint. |
| `process_chunk(X, y)` (alias `partial_fit`) | Test-then-train: score the chunk, then train on it, and append a log entry to `history_`. |
| `run(chunks)` | Processes every `(X_chunk, y_chunk)` pair. |
| `get_metric(key)`, `summary()` | One logged value per chunk as an array, ready to plot; final totals. |
| `estimate_memory(obj)` | Bytes held in NumPy arrays reachable from a model, buffers included. |

Each chunk is scored by the model trained on earlier chunks before it is used for training. This prequential evaluation makes every chunk unseen test data, without a separate hold-out set.

```python
from numcompute import StreamTrainer

trainer = StreamTrainer(pipe, metrics=("accuracy", "f1", "auc"), window=5)
for X_chunk, y_chunk in stream:
    trainer.process_chunk(X_chunk, y_chunk)

trainer.get_metric("cumulative_accuracy")   # one value per chunk
trainer.get_metric("memory_bytes")          # grows with the buffer, drops after a retrain
trainer.summary()
```

### 12. `visualise.py` — Plots (matplotlib)

| Function | Description |
|---|---|
| `plot_metric_over_time(metric_values, title, ylabel)` | A metric across chunks. |
| `compare_models(metric1, metric2, labels)` | Two models on the same metric, e.g. trees with different `buffer_size`. |
| `plot_predictions_vs_ground_truth(y_true, y_pred)` | Per-sample predictions with mistakes highlighted, plus a confusion matrix, for the latest chunk. |
| `plot_stream_log(trainer_or_history, keys)` | Several `StreamTrainer` log keys on one set of axes. |

Every function returns the matplotlib `Figure`.
- `ax=` draws onto existing subplots.
- `save_path=` saves the figure, with the format taken from the extension (.png, .pdf, .svg).
- `show=True` calls `plt.show()`; in notebooks the returned figure displays inline.
- NaN values, such as chunks before the first retrain, appear as gaps, and the chunk axis uses integer ticks.
- matplotlib is imported only when a plot is drawn.

### Demos

| Notebook | Shows |
|---|---|
| `demo/quickstart.ipynb` | Part 1 tour: CSV loading, preprocessing, statistics, sorting and searching, ranking, metrics, gradients, pipeline. |
| `demo/stream_demo.ipynb` | Streaming: loads `demo/stream_data.csv` (6,000 rows, 3% missing values, concept drift) with `read_csv_chunks`; trains `SimpleImputer → StandardScaler → DecisionTreeClassifier` chunk by chunk with `partial_fit` via `StreamTrainer`; plots accuracy, error and memory over time, compares `buffer_size=200` vs `1500`, shows predictions vs ground truth on the latest chunk, and includes both benchmarks. |

Regenerate the demo dataset with `python demo/make_stream_data.py`, then open the notebook from the `demo/` folder.

## Performance and complexity

The following table gives theoretical or typical complexity estimates, rather than benchmark measurements. Actual performance depends on array dimensions, data types, NumPy's implementation, and the cost of user-supplied functions.

Let:

- \(n\) = number of input elements or samples.
- \(d\) = number of features or input dimensions.
- \(c\) = number of classes.
- \(m\) = number of output dimensions.
- \(b\) = number of histogram bins.
- \(q\) = number of queried scores for percentile ranking.

### Sorting and searching

| Operation | Time complexity | Auxiliary space |
|---|---:|---:|
| `stable_sort()` | \(O(n\log n)\) typical | \(O(n)\) typical |
| `multi_key_sort()` | \(O(n\log n)\) | \(O(n)\) |
| `binary_search()` | \(O(\log n)\) | \(O(1)\) |
| `top_k()` | \(O(n + k\log k)\) | \(O(n)\) |
| `quick_select()` | Expected \(O(n)\), worst case \(O(n^2)\) | \(O(n)\) |

Quickselect keeps only the partition containing the target rank, so the expected work is \(n + n/2 + n/4 + \dots = O(n)\). This avoids the \(O(n\log n)\) cost of a full sort. A random pivot from a fixed-seed generator makes the quadratic worst case vanishingly unlikely while keeping results reproducible. `top_k` uses `np.argpartition` (introselect) to find the boundary value and then sorts only the \(k\) selected items.

### Numerical differentiation

Assuming each function evaluation has cost \(C_f\) for `gradient()` or \(C_F\) for `jacobian()`:

| Operation | Function evaluations | Time complexity | Output space |
|---|---:|---:|---:|
| `gradient()` with central differences | \(2d\), plus validation/baseline evaluations if applicable | \(O(dC_f)\) | \(O(d)\) |
| `gradient()` with forward differences | \(d+1\) | \(O(dC_f)\) | \(O(d)\) |
| `jacobian()` with central differences | \(2d\), plus any baseline evaluation | \(O(dC_F)\) | \(O(md)\) |
| `jacobian()` with forward differences | \(d+1\) | \(O(dC_F)\) | \(O(md)\) |

Batched variants may reduce Python-call overhead, but the cost of evaluating the function over the batch still depends on its implementation. Their temporary memory use can be larger because perturbations are constructed together.

### Statistics

For a one-dimensional input of \(n\) elements:

| Function | Typical time complexity | Auxiliary space |
|---|---:|---:|
| `mean()` | \(O(n)\) | \(O(n)\) worst case for conversions or temporaries |
| `median()` | Typically \(O(n)\) using selection-based algorithms | \(O(n)\) worst case |
| `std()` | \(O(n)\) | \(O(n)\) worst case for conversions or temporaries |
| `minimum()` | \(O(n)\) | \(O(n)\) worst case for conversions or temporaries |
| `maximum()` | \(O(n)\) | \(O(n)\) worst case for conversions or temporaries |
| `histogram()` | Typically \(O(n+b)\) | \(O(n+b)\) worst case |
| `quantile()` | Typically selection-based; depends on requested quantiles and NumPy implementation | Depends on input conversion and temporary arrays |

Axis-based operations scale with the total number of input elements. NaN-aware operations may have additional overhead.

### Ranking

| Function | Time complexity | Auxiliary space |
|---|---:|---:|
| `rankdata()` on \(n\) values | \(O(n\log n)\) | \(O(n)\) |
| `rankdata()` on multidimensional arrays | \(O(N\log k)\) typical, where \(N\) is total elements and \(k\) is the ranked axis length | \(O(N)\) |
| `percentile()` | \(O(N\log k + \lvert q\rvert N/k)\) | \(O(N)\) for the sorted copy |
| `percentile_rank()` | \(O(n\log n + q\log n)\) | \(O(n+q)\) |

`rankdata()` sorts each ranked slice once and then assigns every tie group's rank with vectorised array arithmetic. It labels groups with a cumulative sum of value changes, with no Python loop over groups. `percentile_rank()` sorts the reference data once and uses binary-search insertion positions for the queries.

### Evaluation metrics

| Function | Time complexity | Auxiliary space |
|---|---:|---:|
| `accuracy()` | \(O(n)\) | \(O(n)\) |
| `confusion_matrix()` | \(O(nc^2+c^2)\) for the provided matrix-multiplication implementation | \(O(nc+c^2)\) |
| `precision()` | \(O(n)\) | \(O(n)\) |
| `recall()` | \(O(n)\) | \(O(n)\) |
| `f1()` | \(O(n)\) | \(O(n)\) |
| `mse()` | \(O(n)\) | \(O(n)\) |

The confusion-matrix implementation constructs Boolean match arrays and multiplies them to count class combinations. This is more expensive in time and temporary memory than a conventional single-pass counting implementation.

### Preprocessing

For \(n\) samples, \(d\) input features, and \(C\) total one-hot output features:

| Operation | Time complexity | Auxiliary space |
|---|---:|---:|
| `StandardScaler.fit()` | \(O(nd)\) | \(O(d)\) stored statistics |
| `StandardScaler.transform()` | \(O(nd)\) | \(O(nd)\) for the output |
| `MinMaxScaler.fit()` | \(O(nd)\) | \(O(d)\) stored statistics |
| `MinMaxScaler.transform()` | \(O(nd)\) | \(O(nd)\) for the output |
| `OneHotEncoder.fit()` | Typically \(O(nd\log n)\) | \(O(C)\) for stored categories |
| `OneHotEncoder.transform()` | \(O(nC)\) | \(O(nC)\) for the encoded output |

The exact complexity of one-hot encoding depends on the number of unique categories in each feature and the implementation's intermediate arrays.

### Input/output

| Function | Time complexity | Auxiliary space |
|---|---:|---:|
| `read_csv()` | \(O(N)\) typical for \(N\) parsed input values | \(O(N)\) for the resulting array |
| `read_csv_chunks()` | \(O(N)\) in total | \(O(\text{chunk\_size} \times C)\) per chunk |

### Utilities

| Function | Time complexity | Auxiliary space |
|---|---:|---:|
| Activations, `softmax`, `logsumexp` | \(O(n)\) | \(O(n)\) |
| Vector distances on \(n\) rows of \(d\) features | \(O(nd)\) | \(O(nd)\) |
| `pairwise_distances()` | \(O(nmd)\) | \(O(nm)\); \(O(nmd)\) for Manhattan |
| `batch_iterator()` | \(O(n)\) over all batches | \(O(1)\) views, \(O(n)\) when shuffled |

CSV parsing also depends on file size, data types, delimiters, and missing-value handling.

### Pipeline

For a pipeline with \(s\) preprocessing steps, its complexity is the sum of the costs of its component operations.

| Operation | Time complexity | Auxiliary space |
|---|---|---|
| `fit_transform()` | Sum of component fit and transform costs | Depends on intermediate transformed arrays |
| `fit()` | Sum of preprocessing costs and final estimator fitting cost | Depends on component state and intermediate arrays |
| `transform()` | Sum of preprocessing transform costs | Depends on intermediate transformed arrays |
| `predict()` | Sum of preprocessing transform costs and estimator prediction cost | Depends on intermediate arrays and predictions |

These estimates describe algorithmic growth, not measured execution speed. Benchmark results should be reported separately with the machine, Python version, NumPy version, input sizes, and benchmark method.

## Performance Benchmarking

Run every benchmark from the project root (about 5 seconds):

```bash
python -m numcompute.benchmarking
```

Each loop-based implementation is first checked to produce the same result as the vectorised one, and only then timed. Timings are the median of several runs after one warm-up. Data uses fixed random seeds, so the results are reproducible.

**Environment:**

| Machine | OS | Python | NumPy |
|---|---|---|---|
| Intel Core i7-10750H @ 2.60 GHz, 15.8 GB RAM | Windows 11 (10.0.26200) | 3.9.6 | 1.26.0 |

Absolute times vary between machines; the relative comparisons are what matter.

### Batch operations: loop vs vectorised

| Operation | Elements | Python loop (ms) | NumPy (ms) | Speedup |
|---|---:|---:|---:|---:|
| Mean | 100,000 | 11.62 | 1.00 | 11.7× |
| MSE | 100,000 | 39.91 | 0.92 | 43.6× |
| StandardScaler | 50,000 | 62.33 | 5.23 | 11.9× |

### Streaming components: loop vs vectorised

| Operation | Elements | Python loop (ms) | NumPy (ms) | Speedup |
|---|---:|---:|---:|---:|
| Tree `partial_fit` (4,000 samples streamed in chunks of 200, `buffer_size=1000`) | 4,000 | 242.6 | 71.2 | 3.4× |
| Tree `predict` | 20,000 | 28.8 | 6.5 | 4.5× |
| `RunningStats` (Welford, chunks of 500 × 5 features) | 100,000 | 44.7 | 9.3 | 4.8× |

- **Tree `partial_fit`:** the loop version is `LoopDecisionTreeClassifier`, the same tree with its split search in Python loops. Both receive the same chunks and build **identical trees**, which is asserted before timing.
- **Why the tree speedup is lower:** the loop version already uses efficient running class counts, and the vectorised version still loops over nodes in Python.
- **Tree `predict`:** the loop version walks one sample at a time, while the vectorised version moves all samples one tree level per step.

### Effect of `buffer_size`

10,000 samples were streamed in 100 chunks of 100 through `SimpleImputer → StandardScaler → DecisionTreeClassifier(max_depth=6)`. The concept drifts at 50%. Accuracy is test-then-train, so each chunk is scored before the model trains on it.

| `buffer_size` | Accuracy | Accuracy after drift | Retrains | Total retrain time (ms) | Time per retrain (ms) | Peak memory (KB) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.821 | 0.817 | 100 | 512.2 | 5.1 | 65.5 |
| 250 | 0.854 | 0.843 | 33 | 308.7 | 9.4 | 70.6 |
| 500 | 0.869 | 0.853 | 20 | 237.5 | 11.9 | 81.5 |
| 1000 | **0.875** | 0.851 | 10 | 190.4 | 19.0 | 96.3 |
| 2000 | 0.853 | 0.817 | 5 | 148.4 | 29.7 | 131.4 |

**Observations:**
- **Small buffers:** they retrain constantly on little data, so each tree is weaker. They also cost the most total retraining time, because the per-retrain overhead is paid 100 times.
- **Large buffers:** they retrain rarely, but each retrain is slower and the buffer holds more memory. They also react late to the drift; at 2,000, accuracy after the drift falls back to the level of the smallest buffer.
- **The sweet spot:** about 500–1,000, roughly 5–10% of this stream. Peak memory has a floor of about 65 KB, which is the imputer's bounded median summary.

`demo/stream_demo.ipynb` repeats the `buffer_size` experiment on the demo CSV and plots it.

## Testing

Run the full suite from the project root:

```bash
python -m pytest
```

### Test results

Measured with Python 3.9.6, NumPy 1.26.0 and pytest 7.4.3: **877 passed, 0 failed** (about 10 seconds).

| Test file | Tests | What it verifies |
|---|---:|---|
| `test_sort_search.py` | 52 | Sorting, multi-key sort, binary search, top-\(k\), quickselect |
| `test_selection_contract.py` | 20 | Selection contract vs. a stable-sort reference (ties, ordering, NaN, infinity, int64 extremes, strided views, quickselect never calls `np.partition`) |
| `test_utils_numerics.py` | 37 | Activations at ±1000/±inf with warnings treated as errors, logsumexp/softmax stability, distances, pairwise matrices, batching |
| `test_percentile.py` | 26 | All interpolation methods vs. `np.percentile`, NaN policies vs. `np.nanpercentile`, axes, output shapes, extreme q, infinity |
| `test_io_chunks.py` | 20 | Fill/skip missing values, tab delimiter, 2D output, chunked streaming equals a full read, laziness, ragged rows |
| `test_preprocessing.py` | 9 | Ordinary scaling and encoding |
| `test_preprocessing_streaming.py` | 30 | `partial_fit` equals `fit` for any chunking, large-offset stream, running min/max and clip, imputer strategies and on-the-fly updates, append-only one-hot categories |
| `test_preprocessing_boundaries.py` | 44 | Constant columns, fitted feature mismatches, use before `fit`, 1D/empty/NaN/inf/non-numeric input, unknown categories |
| `test_rank.py` | 25 | Ranking methods, NaN policies, percentile ranks |
| `test_rank_vectorised.py` | 14 | Vectorised tie groups vs. an \(O(n^2)\) reference, all-tied and empty input |
| `test_benchmarking.py` | 49 | Loop and NumPy results agree (batch ops, `LoopDecisionTreeClassifier` builds identical trees for Gini and entropy, batch and streaming, loop prediction, loop Welford), `measure_time`, result fields, speedup > 1, `buffer_size` trends (retrains, first scored chunk, memory), buffers larger than the stream, streams with NaN, data generator (drift, missing values, reproducibility), invalid input |
| `test_contracts.py` | 34 | Shared validators, pipeline contracts, metric return types, package exports |
| `test_utils.py` | 24 | Numeric validation, shape checks, safe division |
| `test_optim.py` | 15 | Gradients and Jacobians |
| `test_metrics.py` | 12 | Classification and regression metrics |
| `test_streaming_edge_cases.py` | 102 | Empty chunks, use before fitting/updating, reset, constant columns across chunks, single-row chunks, imputer all-NaN columns and invalid `fill_value`, one-hot NaN and type changes, zero denominators, rolling windows of uneven chunks, fixed-label violations, single-class AUC, NaN/non-numeric/infinite scores, non-monotonic `auc` input; RuntimeWarnings fail the tests |
| `test_metrics_streaming.py` | 38 | ROC/AUC vs. brute force with ties, streaming metrics equal batch metrics, growing confusion-matrix labels, AUC memory per distinct score, rolling windows, reset |
| `test_pipeline.py` | 12 | Pipeline fit/transform/predict |
| `test_pipeline_streaming.py` | 18 | `partial_fit` equals manual chaining, imputer → scaler → tree streams, `predict_proba`, empty chunks, predicting before the tree retrains, missing `partial_fit`, feature-count changes, one-hot width growth |
| `test_stream.py` | 32 | Test-then-train ordering, cumulative accuracy equals manual test-then-train, rolling windows, retrain/buffer/memory logging, peak memory, AUC (including single-class chunks), empty chunks, invalid configuration, `estimate_memory` (containers, cycles, object arrays) |
| `test_visualise.py` | 31 | Figures and line data, NaN gaps, integer chunk ticks, drawing on given axes, saving PNG/SVG/PDF, `show`, invalid input, model comparison labels, confusion-matrix counts, string labels, `max_points`, `StreamTrainer` log plots (headless Agg backend) |
| `test_tree.py` | 69 | Root split vs. exhaustive loop search (Gini and entropy), perfect fit, depth and `min_samples_split` limits, single-class, constant and conflicting data, adjacent floats, ±1e308 values, 1,500-leaf tree, `max_features` options, invalid input, buffered retraining (timing, clearing, buffer-only data, oversized chunks, stable classes, new classes) |
| `test_stats.py` | 11 | Descriptive statistics |
| `test_stats_streaming.py` | 88 | Streaming stats match one-shot results for any chunking (incl. empty/1-row chunks), 1e9-offset stability, constant and all-NaN features, ddof edge cases, merge, windows, NumPy histogram/quantile parity, bounded-memory sketch accuracy and memory bound, no RuntimeWarnings |
| `test_stats_optim_contracts.py` | 64 | Stats input contract (empty, non-numeric, axis bounds, all-NaN slices without warnings, all-equal, strided views, histogram NaN policy, quantile q range) and batched gradient/Jacobian validation |
| `test_io.py` | 1 | CSV loading |

## Team Work Split

| Team Member | Tasks Done |
|---|---|
| Sameena Shafiulla Khan | Setup the initial git repo. Implemented `io.py` (CSV/data loading) and `preprocessing.py` (`StandardScaler`, `MinMaxScaler`, `OneHotEncoder`); wrote unit tests for both modules; contributed to README and documentation. |
| Veeksha Nakka | Implemented `stats.py` (statistical functions and NaN handling), `metrics.py` (classification and regression metrics), and `benchmarking.py`; wrote unit tests for these modules and prepared benchmark results. |
| Krishna Kumar Siraimeetan | Implemented `rank.py` (ranking methods, tie handling, and percentiles) and `utils.py`; wrote unit tests for these modules and created `demo/quickstart.ipynb` after module integration. |
| Aakash Rajan | Implemented `sort_search.py` (sorting, searching, Top-K, and quickselect), `optim.py` (finite-difference gradients and Jacobians), and `pipeline.py`; wrote unit tests for these modules. Contributed to final integration and package setup, README, documentation and demo video |
