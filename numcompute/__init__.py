"""
NumCompute: a lightweight numerical computing toolkit built with NumPy.

The public API of every module is re-exported here, so both
``from numcompute import top_k`` and
``from numcompute.sort_search import top_k`` work.
"""

from numcompute.io import read_csv, read_csv_chunks
from numcompute.metrics import (
    accuracy,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
)
from numcompute.optim import (
    grad,
    gradient,
    gradient_batched,
    jacobian,
    jacobian_batched,
)
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from numcompute.rank import percentile, percentile_rank, rank, rankdata
from numcompute.sort_search import (
    binary_search,
    multi_key_sort,
    quick_select,
    stable_sort,
    top_k,
    topk,
)
from numcompute.stats import (
    RunningStats,
    StreamingHistogram,
    StreamingQuantile,
    histogram,
    maximum,
    mean,
    median,
    minimum,
    quantile,
    std,
    update_stats,
    variance,
)
from numcompute.utils import (
    NotFittedError,
    batch_iterator,
    cosine_distance,
    euclidean_distance,
    leaky_relu,
    log_softmax,
    logsumexp,
    manhattan_distance,
    pairwise_distances,
    relu,
    safe_divide,
    sigmoid,
    softmax,
    tanh,
)

__version__ = "0.1.0"

__all__ = [
    # io
    "read_csv",
    "read_csv_chunks",
    # preprocessing
    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    # sort_search
    "stable_sort",
    "multi_key_sort",
    "binary_search",
    "top_k",
    "topk",
    "quick_select",
    # rank
    "rankdata",
    "rank",
    "percentile",
    "percentile_rank",
    # stats
    "mean",
    "median",
    "std",
    "minimum",
    "maximum",
    "histogram",
    "quantile",
    "variance",
    "RunningStats",
    "StreamingHistogram",
    "StreamingQuantile",
    "update_stats",
    # metrics
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "mse",
    # optim
    "gradient",
    "grad",
    "jacobian",
    "gradient_batched",
    "jacobian_batched",
    # pipeline
    "Pipeline",
    # utils
    "NotFittedError",
    "safe_divide",
    "sigmoid",
    "relu",
    "leaky_relu",
    "tanh",
    "softmax",
    "log_softmax",
    "logsumexp",
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "pairwise_distances",
    "batch_iterator",
]
