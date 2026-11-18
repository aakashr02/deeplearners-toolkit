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
- **Pipelines:** Sequential preprocessing and optional final estimator integration.
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

Precision, recall, and F1 use a specified positive class and are intended for binary classification.

### 7. `preprocessing.py` — Data preprocessing

| Class | Methods | Description |
|---|---|---|
| `StandardScaler` | `fit`, `transform`, `fit_transform` | Centers features and scales by standard deviation. |
| `MinMaxScaler(feature_range=(0, 1))` | `fit`, `transform`, `fit_transform` | Scales features to a specified range. |
| `OneHotEncoder(handle_unknown="error")` | `fit`, `transform`, `fit_transform` | Encodes categorical columns as binary features. |

Fit preprocessing components on training data and reuse the fitted components to transform new data.

**Transformer contract** (shared validation lives in `utils.py`):

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

We compared the performance of NumPy-based vectorised operations against equivalent Python-loop implementations. Three operations were selected for benchmarking: Mean, Mean Squared Error (MSE), and StandardScaler.

The experiments used a fixed random seed of 42 for reproducibility. Each operation was warmed up before performing five timed runs, and the median execution time was recorded.

### Benchmark Results

| Operation | Elements | Python Loop (ms) | NumPy (ms) | Speedup |
|---|---:|---:|---:|---:|
| Mean | 100,000 | 7.0359 | 0.4829 | 14.57× |
| MSE | 100,000 | 22.0071 | 0.3809 | 57.78× |
| StandardScaler | 50,000 | 29.0550 | 0.7463 | 38.93× |

### Observations

All three NumPy implementations were faster than their Python-loop equivalents. MSE achieved the highest speedup of 57.78×, followed by StandardScaler at 38.93× and Mean at 14.57×.

These results highlight the advantages of vectorised numerical operations for the tested input sizes. Execution times may vary depending on the hardware and software environment.

### Running the Benchmarks

Run the following command from the project root:

```bash
python -m numcompute.benchmarking
```

The script generates reproducible input data, measures execution times and prints a comparison table.

## Testing

Run the full suite from the project root:

```bash
python -m pytest
```

### Test results

Measured with Python 3.9.6, NumPy 1.26.0 and pytest 7.4.3: **529 passed, 0 failed** (about 3 seconds).

| Test file | Tests | What it verifies |
|---|---:|---|
| `test_sort_search.py` | 52 | Sorting, multi-key sort, binary search, top-\(k\), quickselect |
| `test_selection_contract.py` | 20 | Selection contract vs. a stable-sort reference (ties, ordering, NaN, infinity, int64 extremes, strided views, quickselect never calls `np.partition`) |
| `test_utils_numerics.py` | 37 | Activations at ±1000/±inf with warnings treated as errors, logsumexp/softmax stability, distances, pairwise matrices, batching |
| `test_percentile.py` | 26 | All interpolation methods vs. `np.percentile`, NaN policies vs. `np.nanpercentile`, axes, output shapes, extreme q, infinity |
| `test_io_chunks.py` | 20 | Fill/skip missing values, tab delimiter, 2D output, chunked streaming equals a full read, laziness, ragged rows |
| `test_preprocessing.py` | 9 | Ordinary scaling and encoding |
| `test_preprocessing_boundaries.py` | 44 | Constant columns, fitted feature mismatches, use before `fit`, 1D/empty/NaN/inf/non-numeric input, unknown categories |
| `test_rank.py` | 25 | Ranking methods, NaN policies, percentile ranks |
| `test_rank_vectorised.py` | 14 | Vectorised tie groups vs. an \(O(n^2)\) reference, all-tied and empty input |
| `test_benchmarking.py` | 21 | Loop and NumPy results agree, `measure_time` behaviour, result fields, invalid sizes, speedup > 1 |
| `test_contracts.py` | 34 | Shared validators, pipeline contracts, metric return types, package exports |
| `test_utils.py` | 24 | Numeric validation, shape checks, safe division |
| `test_optim.py` | 15 | Gradients and Jacobians |
| `test_metrics.py` | 12 | Classification and regression metrics |
| `test_pipeline.py` | 12 | Pipeline fit/transform/predict |
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
