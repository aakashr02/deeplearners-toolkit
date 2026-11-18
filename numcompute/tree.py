"""
tree.py
=======

A depth-limited decision tree classifier with buffered streaming
retraining.

``DecisionTreeClassifier.fit(X, y)`` trains a tree from scratch.
``partial_fit(X_chunk, y_chunk)`` appends each chunk to an internal
buffer; once the buffer holds at least ``buffer_size`` samples the tree is
retrained on the buffered data and the buffer is cleared. Between
retrains, ``predict`` uses the most recently trained tree.

Implementation
--------------
The tree is stored as flat NumPy arrays (one entry per node), not as
Python node objects. Growing it needs one Python loop over nodes (each
node's samples differ), but the work inside a node is vectorised:

* every candidate feature is sorted at once (``argsort`` along axis 0);
* one-hot labels are cumulatively summed in that order, giving the class
  counts left of every possible split position for every feature;
* the weighted Gini/entropy impurity of every candidate split is computed
  in one array expression, and ``argmin`` picks the best.

Prediction routes all samples together, one tree level per step.
"""

import time

import numpy as np

from numcompute.utils import NotFittedError, validate_2d, validate_int

_CRITERIA = ("gini", "entropy")
_LEAF = -1


def _impurity(counts, criterion):
    """
    Impurity of class-count vectors along the last axis.

    ``counts`` has shape (..., n_classes); rows with zero total give 0.
    """
    totals = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(totals > 0, totals, 1.0)
    if criterion == "gini":
        return 1.0 - np.sum(p * p, axis=-1)
    # Entropy in bits; 0 * log(0) is defined as 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return -np.sum(p * logs, axis=-1)


class DecisionTreeClassifier:
    """
    Decision tree classifier (Gini or entropy) with buffered retraining.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        Split quality measure.
    max_depth : int or None, default=None
        Maximum depth (the root has depth 0). None grows until leaves are
        pure or too small to split. 0 gives a single leaf.
    min_samples_split : int, default=2
        Minimum number of samples a node needs to be split (at least 2).
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        Number of features considered at each split, drawn at random
        without replacement: an int (1..n_features), a float fraction in
        (0, 1], "sqrt", "log2", or None for all features.
    buffer_size : int, default=500
        Number of buffered samples that triggers a retrain in
        ``partial_fit`` (at least 1).
    random_state : int or None, default=None
        Seed for feature sub-sampling, for reproducible trees.

    Attributes
    ----------
    classes_ : np.ndarray
        Class labels known to the model, sorted. ``predict_proba`` columns
        follow this order.
    n_features_in_ : int
        Number of features.
    n_retrains_ : int
        Number of times ``partial_fit`` retrained the tree.
    n_buffered_ : int
        Samples currently waiting in the buffer.
    last_fit_time_ : float
        Wall-clock seconds taken by the most recent tree build.

    Raises
    ------
    ValueError
        If any hyperparameter is invalid.
    TypeError
        If an integer hyperparameter has the wrong type.

    Examples
    --------
    >>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
    >>> for X_chunk, y_chunk in stream:          # doctest: +SKIP
    ...     tree.partial_fit(X_chunk, y_chunk)
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 max_features=None, buffer_size=500, random_state=None):
        if criterion not in _CRITERIA:
            raise ValueError(
                f"criterion must be 'gini' or 'entropy'; received {criterion!r}"
            )
        if max_depth is not None:
            max_depth = validate_int(max_depth, "max_depth", minimum=0)
        self._check_max_features(max_features)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = validate_int(
            min_samples_split, "min_samples_split", minimum=2
        )
        self.max_features = max_features
        self.buffer_size = validate_int(buffer_size, "buffer_size", minimum=1)
        self.random_state = random_state

        self._known_classes = None
        self._clear_buffer()
        self.n_retrains_ = 0

    # --------------------------------------------------------
    # Validation helpers
    # --------------------------------------------------------

    @staticmethod
    def _check_max_features(max_features):
        if max_features is None or max_features in ("sqrt", "log2"):
            return
        if isinstance(max_features, (bool, np.bool_)):
            raise ValueError("max_features must not be a boolean")
        if isinstance(max_features, (int, np.integer)):
            if max_features < 1:
                raise ValueError("max_features must be at least 1")
            return
        if isinstance(max_features, (float, np.floating)):
            if not 0.0 < max_features <= 1.0:
                raise ValueError("a float max_features must be in (0, 1]")
            return
        raise ValueError(
            "max_features must be None, an int, a float in (0, 1], "
            "'sqrt' or 'log2'"
        )

    def _n_candidate_features(self, n_features):
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features))) if n_features > 1 else 1
        if isinstance(mf, (float, np.floating)):
            return max(1, int(mf * n_features))
        if mf > n_features:
            raise ValueError(
                f"max_features={mf} exceeds the number of features "
                f"({n_features})"
            )
        return int(mf)

    def _validate_Xy(self, X, y, allow_empty=False):
        n_features = getattr(self, "n_features_in_", None)
        if n_features is None and self._buffer_X:
            n_features = self._buffer_X[0].shape[1]

        X = validate_2d(X, name="X", n_features=n_features,
                        allow_empty=allow_empty)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError(f"y must be 1D; received {y.ndim} dimensions")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}"
            )
        if y.dtype.kind == "f" and np.any(np.isnan(y)):
            raise ValueError("y must not contain NaN labels")
        return X, y

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def fit(self, X, y):
        """
        Train a tree from scratch on ``X`` and ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Real, finite features (impute NaN beforehand).
        y : array-like of shape (n_samples,)
            Class labels of any comparable dtype.

        Returns
        -------
        DecisionTreeClassifier
            The fitted tree (``self``).

        Raises
        ------
        TypeError
            If X is not real numeric data.
        ValueError
            If X is not 2D, is empty, contains NaN/inf, y is not 1D or has
            NaN, the lengths differ, or max_features exceeds n_features.

        Notes
        -----
        ``fit`` starts from scratch: it also clears the streaming buffer,
        the known classes and ``n_retrains_``, so data buffered by earlier
        ``partial_fit`` calls cannot leak into later retrains.

        Time complexity: O(D * n * F * (log n + C)) for depth D, n samples,
        F candidate features and C classes (each level sorts every node's
        samples once per candidate feature).
        Space complexity: O(n * F * C) for the cumulative counts of the
        largest node, plus O(nodes * C) for the tree.
        """
        self._clear_buffer()
        for attr in ("n_features_in_", "feature_"):
            self.__dict__.pop(attr, None)

        X, y = self._validate_Xy(X, y)
        self._known_classes = np.unique(y)
        self.n_retrains_ = 0
        self._build(X, y, classes=self._known_classes)
        return self

    def _build(self, X, y, classes):
        """Grow the tree; classes may include labels absent from y."""
        start = time.perf_counter()
        n_samples, n_features = X.shape
        n_candidates = self._n_candidate_features(n_features)
        rng = np.random.default_rng(self.random_state)

        self.classes_ = classes
        self.n_features_in_ = n_features
        n_classes = classes.size
        y_encoded = np.searchsorted(classes, y)
        onehot = np.eye(n_classes)[y_encoded]

        feature, threshold, left, right, value, depth_of = [], [], [], [], [], []

        def new_node(indices, depth):
            feature.append(_LEAF)
            threshold.append(np.nan)
            left.append(_LEAF)
            right.append(_LEAF)
            value.append(onehot[indices].sum(axis=0))
            depth_of.append(depth)
            return len(feature) - 1

        stack = [(new_node(np.arange(n_samples), 0), np.arange(n_samples))]

        while stack:
            node, indices = stack.pop()
            depth = depth_of[node]
            counts = value[node]

            if (
                (self.max_depth is not None and depth >= self.max_depth)
                or indices.size < self.min_samples_split
                or np.count_nonzero(counts) <= 1          # pure node
            ):
                continue

            if n_candidates < n_features:
                features = np.sort(
                    rng.choice(n_features, n_candidates, replace=False)
                )
            else:
                features = np.arange(n_features)

            split = self._best_split(X[indices][:, features],
                                     onehot[indices], counts)
            if split is None:
                continue

            column, cut, go_left = split
            feature[node] = int(features[column])
            threshold[node] = cut

            left_idx, right_idx = indices[go_left], indices[~go_left]
            left[node] = new_node(left_idx, depth + 1)
            right[node] = new_node(right_idx, depth + 1)
            stack.append((right[node], right_idx))
            stack.append((left[node], left_idx))

        self.feature_ = np.array(feature, dtype=np.intp)
        self.threshold_ = np.array(threshold, dtype=float)
        self.children_left_ = np.array(left, dtype=np.intp)
        self.children_right_ = np.array(right, dtype=np.intp)
        self.value_ = np.array(value, dtype=float)
        self.node_depth_ = np.array(depth_of, dtype=np.intp)
        self.last_fit_time_ = time.perf_counter() - start

    def _best_split(self, X, onehot, parent_counts):
        """
        Vectorised search over all candidate features and thresholds.

        Returns ``(column, threshold, go_left_mask)`` or None if no split
        reduces impurity.
        """
        n, n_feat = X.shape
        order = np.argsort(X, axis=0, kind="stable")              # (n, F)
        sorted_x = np.take_along_axis(X, order, axis=0)            # (n, F)

        # Class counts left of each split position i (samples 0..i left).
        left_counts = np.cumsum(onehot[order], axis=0)[:-1]       # (n-1, F, C)
        right_counts = parent_counts - left_counts

        n_left = np.arange(1, n)[:, None]                          # (n-1, 1)
        n_right = n - n_left
        weighted = (
            n_left * _impurity(left_counts, self.criterion)
            + n_right * _impurity(right_counts, self.criterion)
        ) / n                                                      # (n-1, F)

        # Only split between distinct values (ties stay together).
        valid = sorted_x[1:] > sorted_x[:-1]
        if not np.any(valid):
            return None

        weighted = np.where(valid, weighted, np.inf)
        best = int(np.argmin(weighted))       # first minimum: deterministic
        position, column = divmod(best, n_feat)

        parent = _impurity(parent_counts, self.criterion)
        if not weighted[position, column] < parent - 1e-12:
            return None

        low = sorted_x[position, column]
        high = sorted_x[position + 1, column]
        # low/2 + high/2 cannot overflow, unlike low + (high - low)/2 when
        # the values span more than the float range (e.g. -1e308, 1e308).
        cut = low / 2.0 + high / 2.0
        # Rounding can push the midpoint of adjacent floats onto `high`.
        if not low <= cut < high:
            cut = low

        return column, float(cut), X[:, column] <= cut

    # --------------------------------------------------------
    # Streaming
    # --------------------------------------------------------

    def _clear_buffer(self):
        self._buffer_X = []
        self._buffer_y = []
        self.n_buffered_ = 0

    def partial_fit(self, X_chunk, y_chunk):
        """
        Buffer a chunk; retrain once the buffer holds ``buffer_size`` samples.

        The tree is retrained on the buffered data only (not on all data
        ever seen), and the buffer is then cleared. Between retrains,
        ``predict`` keeps using the most recently trained tree.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
            Real, finite features. An empty chunk is a no-op.
        y_chunk : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        DecisionTreeClassifier
            ``self``.

        Raises
        ------
        TypeError
            If X_chunk is not real numeric data.
        ValueError
            If shapes are inconsistent, the feature count differs from
            earlier chunks, or the data contains NaN/inf.

        Notes
        -----
        ``classes_`` covers every label seen by any ``partial_fit`` call,
        so ``predict_proba`` keeps a stable column order even when a
        buffer contains only some of the classes.

        Time complexity: O(n_chunk) to buffer; a retrain costs as ``fit``
        on ``n_buffered_`` samples. Space complexity: O(buffer_size * F).
        """
        X, y = self._validate_Xy(X_chunk, y_chunk, allow_empty=True)
        if X.shape[0] == 0:
            return self

        if self._known_classes is None:
            self._known_classes = np.unique(y)
        else:
            self._known_classes = np.union1d(self._known_classes, y)

        self._buffer_X.append(X)
        self._buffer_y.append(y)
        self.n_buffered_ += X.shape[0]

        if self.n_buffered_ >= self.buffer_size:
            X_all = np.concatenate(self._buffer_X, axis=0)
            y_all = np.concatenate(self._buffer_y, axis=0)
            self._build(X_all, y_all, classes=self._known_classes)
            self.n_retrains_ += 1
            self._clear_buffer()

        return self

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def _check_fitted(self):
        if not hasattr(self, "feature_"):
            raise NotFittedError(
                "DecisionTreeClassifier has no trained tree yet: call fit(), "
                f"or partial_fit() until {self.buffer_size} samples have "
                f"been buffered (currently {self.n_buffered_})"
            )

    def apply(self, X):
        """
        Index of the leaf each sample lands in.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Leaf node indices.

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.
        ValueError
            If X has the wrong shape or contains NaN/inf.

        Notes
        -----
        All samples move down together, one level per step.

        Time complexity: O(n * depth). Space complexity: O(n).
        """
        self._check_fitted()
        X = validate_2d(X, name="X", n_features=self.n_features_in_,
                        allow_empty=True)

        nodes = np.zeros(X.shape[0], dtype=np.intp)
        rows = np.arange(X.shape[0])
        active = self.feature_[nodes] != _LEAF

        while np.any(active):
            current = nodes[active]
            goes_left = (
                X[rows[active], self.feature_[current]]
                <= self.threshold_[current]
            )
            nodes[active] = np.where(goes_left, self.children_left_[current],
                                     self.children_right_[current])
            active = self.feature_[nodes] != _LEAF

        return nodes

    def predict_proba(self, X):
        """
        Class probabilities from the class frequencies in each leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Rows sum to 1; columns follow ``classes_``.

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.
        ValueError
            If X has the wrong shape or contains NaN/inf.

        Notes
        -----
        Time complexity: O(n * depth + n * C).
        """
        leaves = self.apply(X)          # checks the fitted state first
        counts = self.value_[leaves]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels (dtype of ``classes_``). Ties between classes
            in a leaf go to the smallest label.

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.
        ValueError
            If X has the wrong shape or contains NaN/inf.

        Notes
        -----
        Time complexity: O(n * depth + n * C).
        """
        proba = self.predict_proba(X)   # checks the fitted state first
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        """
        Mean accuracy on ``X`` and ``y``.

        Raises
        ------
        NotFittedError, ValueError
            As for ``predict``.

        Notes
        -----
        Time complexity: as ``predict``.
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------

    @property
    def node_count_(self):
        """Total number of nodes in the trained tree."""
        self._check_fitted()
        return int(self.feature_.size)

    def get_depth(self):
        """
        Depth of the trained tree (a single leaf has depth 0).

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.

        Notes
        -----
        Time complexity: O(1) (depths are stored per node).
        """
        self._check_fitted()
        return int(self.node_depth_.max())

    def get_n_leaves(self):
        """
        Number of leaves in the trained tree.

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.

        Notes
        -----
        Time complexity: O(nodes).
        """
        self._check_fitted()
        return int(np.sum(self.feature_ == _LEAF))


__all__ = ["DecisionTreeClassifier"]
