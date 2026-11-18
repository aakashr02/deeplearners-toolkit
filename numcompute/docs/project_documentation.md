# DeepLearners Toolkit — Project Documentation

This document provides detailed technical explanations of all modules, algorithms, design decisions, usage patterns, and testing strategies used in the DeepLearners Scientific Computing Toolkit.

It complements the high‑level README and serves as the developer‑facing reference for the entire project.

---

# 1. Overview

The DeepLearners Toolkit is a modular scientific computing and machine learning package built entirely using **Python + NumPy**.  
It is designed to mimic the structure of real scientific computing libraries (NumPy, SciPy, scikit‑learn) while remaining lightweight and fully transparent.

The toolkit supports:

- Data loading  
- Preprocessing  
- Sorting & searching  
- Ranking & percentiles  
- Statistics  
- Metrics  
- Gradients & Jacobians  
- Pipelines  
- Utilities  
- Benchmarking  

All functionality is implemented from scratch without external ML libraries.

---

# 2. Module Documentation

## 2.1 io.py — Data Loading

### Purpose
Provides simple CSV loading utilities using NumPy.

### Key Functions
```python
read_csv(path, delimiter=",", missing_value=np.nan, dtype=float,
         missing="fill", skip_header=0)
read_csv_chunks(path, chunk_size=1000, ...)   # same options, yields chunks
```
### Behaviour
- Loads CSV files into 2D NumPy arrays, even for a single row or column.
- Supports custom delimiters (`","`, `"\t"`, or `None` for whitespace) and dtype conversion.
- Missing values: `missing="fill"` uses `missing_value`; `missing="skip"` drops affected rows.
- `read_csv_chunks` is a lazy generator, so memory use depends on `chunk_size` rather than file size. `np.vstack` of all chunks equals `read_csv`.
- Raises `FileNotFoundError` for a missing file, and `ValueError` for unknown options or inconsistent column counts.

### Design Notes
- Both functions share one parsing helper around `numpy.genfromtxt`.
- Streaming is the one place a Python loop is necessary: lines are read one at a time and parsed in vectorised blocks of `chunk_size`.

## 2.2 preprocessing.py — Scaling & Encoding

### Purpose
Implements core preprocessing utilities similar to scikit‑learn.

### Components
#### StandardScaler
Computes mean and standard deviation
Produces zero‑mean, unit‑variance features

#### MinMaxScaler
Scales features to a given range (default 0–1)
Useful for algorithms sensitive to absolute scale

#### OneHotEncoder
Converts categorical columns into binary indicator vectors
Handles repeated categories
Supports multiple columns

### Streaming
- `partial_fit(X_chunk)` folds one chunk into the state; `fit` is `reset()` followed by `partial_fit`, so batch and streaming use share one code path.
- `StandardScaler` reuses `stats.RunningStats`, and `MinMaxScaler` keeps a running min and max.
- `SimpleImputer` uses `RunningStats` for the mean, `StreamingQuantile` (bounded sketch) for the median, and merged value counts for the most frequent value.
- `OneHotEncoder` appends newly seen categories after the existing ones, so earlier output columns keep their position.

### API Style
All transformers follow the scikit‑learn pattern:

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

### Design Notes
- Pure NumPy implementation; all three share a `_Transformer` base class and the `utils.validate_2d` / `utils.check_is_fitted` helpers.
- `fit` records `n_features_in_`; `transform` rejects a different column count with `ValueError` instead of broadcasting silently.
- `transform` before `fit` raises `NotFittedError` (a `RuntimeError` and `AttributeError` subclass).
- Constant features: the zero standard deviation or range is replaced by 1 (`scale_`), so outputs stay finite. `StandardScaler` maps such a column to 0 and `MinMaxScaler` to `feature_range[0]`.
- Scalers reject NaN, infinity, non-numeric, 1D and empty input. `OneHotEncoder` rejects unseen categories unless `handle_unknown="ignore"`.

## 2.3 sort_search.py — Sorting, Searching, Top‑K

### Purpose
Implements efficient sorting and searching algorithms.

### Features
Stable sort
Multi‑key sort
Top‑K (`top_k`, alias `topk`) using `np.argpartition`
Explicit, educational quickselect (no `np.partition`)
Binary search

### Example

top_values = top_k(arr, k=5)

### Design Notes
- `_quickselect` is an iterative Hoare FIND. Each round picks a random pivot (fixed seed, so results are reproducible), splits the candidates into better, equal and worse groups, and keeps only the group that contains the target rank. Expected O(n), worst case O(n²).
- The candidate index array stays in increasing order, so the answer inside the equal group is at offset `k - n_better`. This reproduces stable tie order (lower index first).
- `top_k` uses `np.argpartition` (as the specification requires) to find the boundary value. It then takes every strictly better element plus the lowest-index ties needed to make k, and sorts only those k items: O(n + k log k).
- Shared contract: 1D ndarray, NaN rejected, ties broken by lowest index, `top_k` results best-first, and `quick_select(v, r) == top_k(v, r + 1)[-1]`.

## 2.4 rank.py — Ranking & Percentiles
### Purpose
Implements ranking utilities similar to SciPy’s rankdata.

### Features
Average, dense, ordinal, min and max ranking (`rankdata`, alias `rank`)
`percentile(data, q, interpolation='linear'|'lower'|'higher'|'midpoint'|'nearest')`
Percentile rank of scores
Tie handling

### Example

ranks = rank(arr, method="average")
p90 = percentile(arr, 90, interpolation="midpoint")

### Percentile algorithm
- Sort along the axis once (NaN sorts to the end), then count the valid values `n` in each slice.
- Compute `pos = q/100 * (n - 1)` for every `q` and slice at once, take `i = floor(pos)` and `j = ceil(pos)`, and gather both with `take_along_axis`. The interpolation rule then combines `s[i]` and `s[j]`.
- Because `n` is per slice, `nan_policy="omit"` stays fully vectorised even when slices have different numbers of NaNs. Results are verified against `np.percentile` and `np.nanpercentile`.

### Design Notes
- One stable argsort per slice, then fully vectorised tie-group assignment. Group starts are marked where sorted values change; `cumsum` gives each position a group id, and each method's rank (average, min, max, dense, ordinal) is computed per group with array arithmetic and scattered back through the sort order. There is no Python loop over groups.
- Descending order reverses whole groups, so tied values keep their original order.
- O(n log n) time, O(n) space per ranked slice.

## 2.5 stats.py — Statistics
### Purpose
Provides core statistical functions.

### Functions
mean
median
std
min/max
histogram
quantiles
axis‑wise statistics
NaN‑safe versions

### Example

mean_val = mean(arr)

### Streaming statistics
- `RunningStats` keeps a per-feature count, mean, `M2`, min and max. Each chunk's statistics are computed with vectorised NumPy and merged with the Chan–Golub–LeVeque parallel form of Welford's algorithm. Memory is O(n_features), however long the stream.
- `StreamingHistogram` uses fixed edges. Each chunk is binned with one `searchsorted` and one `bincount`. A `deque(maxlen=window)` of per-chunk counts gives the sliding window.
- `StreamingQuantile` has three modes:
  - *Exact* stores the values and evaluates them with `rank.percentile`.
  - *Window* keeps only the last W chunks.
  - *Sketch* (`max_centroids=K`) keeps raw values until there are more than 2K per feature, then merges them into K equal-mass weighted centroids. It interpolates between centroid midpoints, anchored at the exact min and max, so memory is O(K) per feature for the whole stream. The update, compression and query are vectorised across features: rows are padded with (inf, 0) entries, sorted once with `argsort`, and centroids are formed with `bincount`.
- Shared contract (`_StreamingBase`):
  - `update_stats(X_chunk)` / `partial_fit`, `reset()` and `result()`;
  - the feature count is fixed by the first non-empty chunk;
  - `NotFittedError` before any data.

### Design Notes
Pure NumPy implementation
Avoids Python loops
Handles NaN values where appropriate
A constant feature reports variance exactly 0 (detected via min == max), because rounding in chunk means otherwise leaves M2 ≈ 1e-34.

## 2.6 metrics.py — Classification & Regression Metrics
### Purpose
Implements common evaluation metrics.

### Classification Metrics
Accuracy
Precision
Recall
F1 score
Confusion matrix

### Regression Metrics
Mean Squared Error (MSE)

### ROC / AUC
`roc_curve` groups samples by distinct score, so tied scores form one step. `roc_auc_score` integrates that curve with the trapezoidal rule, which counts ties as one half.

### Streaming metrics
- A `_StreamingMetric` base class stores one small array of sufficient statistics per chunk: (correct, n) for accuracy, (TP, FP, FN) for precision, recall and F1, and (sum of squared errors, n) for MSE.
- Without a window these arrays are summed into a running total. With `window=W` they are kept in a `deque(maxlen=W)`, so the oldest chunk drops out automatically.
- `StreamingConfusionMatrix` re-indexes stored matrices into the current label set with `np.ix_`.
- `StreamingAUC` merges (score, positives, negatives) tables with `np.unique` and `bincount`.

### Example

acc = accuracy(y_true, y_pred)

### Design Notes
Works with NumPy arrays
Supports binary and multi‑class cases
Ensures shape consistency

## 2.7 optim.py — Gradients & Jacobians
### Purpose
Implements finite‑difference numerical differentiation.

### Features
Gradient computation
Jacobian computation
Optional step‑size control

### Example

grad = gradient(lambda x: x**2, 3.0)

### Design Notes
- Pure NumPy implementation
- Supports arbitrary scalar functions
- Useful for educational purposes

## 2.8 pipeline.py — Transformer/Estimator Pipeline
### Purpose
Implements a simple pipeline system similar to scikit‑learn.

### Features
- fit, transform, fit_transform
- Pipeline chaining
- Modular design

### Example

- pipeline = Pipeline([StandardScaler()])
- output = pipeline.fit_transform(data)

### Design Notes
- Ensures consistent interface across modules
- Allows sequential transformations
- Easy to extend

## 2.9 utils.py — Shared Utilities
### Purpose
- Provides helper functions used across modules.

### Includes
- Activation functions: `sigmoid`, `relu`, `leaky_relu`, `tanh`, `softmax`, `log_softmax`
- `logsumexp`
- Distance functions: `euclidean_distance`, `manhattan_distance`, `cosine_distance`, `pairwise_distances`
- Batching: `batch_iterator`
- Shared validation: `validate_array`, `validate_2d`, `check_is_fitted`, `validate_int`, `validate_bool`, `NotFittedError`, `safe_divide`

### Numerical stability
- `sigmoid` uses `1/(1+e^{-x})` for `x >= 0` and `e^x/(1+e^x)` for `x < 0`, so `exp` never sees a large positive argument.
- `logsumexp` shifts by the maximum: `m + log(sum(exp(x - m)))`. An infinite maximum is not subtracted, which would give inf − inf = NaN, so all `-inf` gives `-inf` and any `+inf` gives `+inf`.
- `softmax` and `log_softmax` are built on `logsumexp`, so logits of 1000 or more give finite probabilities.
- Pairwise Euclidean distances use `||x||² + ||y||² − 2x·y` and clip small negative rounding errors to 0 before the square root.
- The tests treat NumPy `RuntimeWarning`s as errors, which proves that no overflow happens rather than being silently tolerated.

### Design Notes
- Centralised utility functions
- Avoids duplication across modules

## 2.10 benchmarking.py — Performance Benchmarking
### Purpose
- Compares Python loops vs NumPy vectorised implementations.

### Features
- Time measurement
- Speedup calculation
- Summary tables

### Example

benchmark(python_loop, numpy_sum, data)

### Design Notes
- Used in final report
- Demonstrates efficiency gains
- Helps students understand vectorisation benefits

# 3. Testing
## Approach
- Specification-based tests: each documented contract (shapes, exceptions, constant features, use before fit, tie-breaking) has a test that checks it directly.
- Reference-implementation tests: `top_k`/`quick_select` are compared with a stable-sort reference, and `rankdata` with an O(n²) reference, on hundreds of randomised arrays with heavy ties.
- Benchmark tests: loop and NumPy versions agree, `measure_time` performs one warm-up plus `repeats` runs, result dictionaries have the expected fields, and invalid sizes are rejected.

## Tools
- pytest (with `parametrize`, `monkeypatch`, `capsys`)
- sample CSV files
- synthetic NumPy arrays with fixed random seeds

## Actual results
Run on Python 3.9.6, NumPy 1.26.0, pytest 7.4.3: **597 passed, 0 failed**.

| Area | Test files | Tests |
|---|---|---:|
| Sorting, searching, selection | `test_sort_search.py`, `test_selection_contract.py` | 72 |
| Ranking and percentiles | `test_rank.py`, `test_rank_vectorised.py`, `test_percentile.py` | 65 |
| Utilities | `test_utils.py`, `test_utils_numerics.py` | 61 |
| Preprocessing | `test_preprocessing.py`, `test_preprocessing_boundaries.py`, `test_preprocessing_streaming.py` | 83 |
| Cross-module contracts | `test_contracts.py` | 34 |
| I/O | `test_io.py`, `test_io_chunks.py` | 21 |
| Benchmarking | `test_benchmarking.py` | 21 |
| Statistics and batched optimisation contracts | `test_stats_optim_contracts.py` | 64 |
| Optimisation | `test_optim.py` | 15 |
| Metrics | `test_metrics.py`, `test_metrics_streaming.py` | 50 |
| Pipeline | `test_pipeline.py` | 12 |
| Statistics | `test_stats.py`, `test_stats_streaming.py` | 99 |

## Corrections driven by testing
- Constant columns produced NaN in both scalers → divisor of 1 for zero spread.
- A `(n, 1)` input broadcast silently against a fitted multi-feature scaler → explicit `n_features_in_` check.
- `transform` before `fit` failed with an unhelpful `AttributeError` → `NotFittedError`.
- `quick_select` delegated to `np.partition` and broke ties inconsistently → explicit quickselect with a stable tie rule.
- `benchmark_mean` validated `size` only after generating data → validated first.
- `from numcompute import ...` (as used in this README) failed because `__init__.py` was empty → public API re-exported.
- Missing specified APIs (activations, distances, logsumexp, batching, chunked CSV input, `percentile` interpolation, `rank`/`topk`/`grad` names) → implemented and tested.
- `histogram(..., ignore_nan=False)` silently ignored the flag and dropped NaN → now raises `ValueError`.
- `gradient_batched`/`jacobian_batched` skipped all input validation → now share `gradient`/`jacobian` validation.
- `percentile([inf, inf], 50)` returned NaN because inf − inf = NaN in linear interpolation → returns the shared endpoint when both neighbours are equal.

# 4. Benchmarking Strategy
## Compare
- Python loops
- NumPy vectorised operations

## Metrics
- Execution time
- Speedup factor

## Output
- Tables for final report
- Plots (optional)

# 5. End‑to‑End Workflow (quickstart.ipynb)
The demonstration notebook shows:

- Load data
- Preprocess
- Compute statistics
- Sort/search
- Rank
- Compute metrics
- Compute gradients
- Run pipeline
- Benchmark

This provides a complete walkthrough of the toolkit.

# 6. Limitations
- No GPU support
- No external ML libraries
- No automatic hyperparameter tuning
- Basic metrics only
- Educational focus, not production‑grade

# 7. Future Improvements
- ROC/AUC
- More preprocessing utilities
- More metrics
- CI/CD (GitHub Actions)
- Documentation website (MkDocs/Sphinx)

# 8. Team Responsibilities
- Sameena — IO, preprocessing, tests, documentation
- Veeksha — stats, metrics, benchmarking
- Krishna — ranking, utils, notebook
- Aakash — sorting/searching, optim, pipeline, integration

# 9. Assignment Deliverables
- Full toolkit
- Unit tests
- quickstart notebook
- Benchmark results
- README
- Documentation
- 4‑page report
- 5‑minute demo video

# 10. Conclusion
The DeepLearners Toolkit demonstrates how scientific computing libraries can be built from scratch using only Python and NumPy.
It provides a clean, modular, extensible foundation for numerical computing and machine learning workflows.