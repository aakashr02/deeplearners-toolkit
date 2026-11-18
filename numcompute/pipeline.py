"""
pipeline.py
===========

Sequential composition of NumCompute components.

Component contracts
-------------------
* Transformer: ``fit(X) -> self`` and ``transform(X) -> ndarray``
  (``fit_transform(X)`` is used when available).
* Estimator (final step only): ``fit(X, y) -> self`` and
  ``predict(X) -> ndarray``. A final step is treated as an estimator
  exactly when it has a ``predict`` method.
* Streaming: every step used by ``Pipeline.partial_fit`` must also have
  ``partial_fit`` (transformers: ``partial_fit(X)``; estimator:
  ``partial_fit(X, y)``).

Shape and fitted-state checks (feature count, use before fit) are
enforced by each component; the pipeline adds structural checks and
raises ``NotFittedError`` when used before fitting.
"""

import numpy as np

from numcompute.utils import NotFittedError


def _has_methods(component, *names):
    return all(callable(getattr(component, name, None)) for name in names)


class Pipeline:
    """Chain named preprocessors and an optional final estimator.

    A pipeline applies components sequentially, passing the output of each
    component to the next. Preprocessors must implement ``fit(X)`` and
    ``transform(X)``. A final estimator must implement ``fit(X, y)`` and
    ``predict(X)``.

    Parameters
    ----------
    steps : sequence of (str, object)
        Ordered ``(name, component)`` pairs with unique string names.

    Attributes
    ----------
    steps : list[tuple[str, object]]
        Ordered sequence of component names and components.
    named_steps : dict[str, object]
        Mapping from step name to component.
    _fitted : bool
        Whether the pipeline has been fitted.
    _final_is_estimator : bool
        Whether the final component is treated as an estimator.

    Raises
    ------
    TypeError
        If ``steps`` is not a sequence of ``(name, component)`` pairs
        with string names.
    ValueError
        If step names are duplicated.

    Notes
    -----
    Time and space complexity are the sums of the costs of the
    component operations, plus the intermediate arrays they produce.
    """

    def __init__(self, steps):
        steps = list(steps)

        for step in steps:
            if not (isinstance(step, tuple) and len(step) == 2):
                raise TypeError(
                    "each step must be a (name, component) tuple"
                )
            if not isinstance(step[0], str):
                raise TypeError("step names must be strings")

        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError("step names must be unique")

        self.steps = steps
        self.named_steps = dict(steps)
        self._fitted = False
        self._final_is_estimator = False

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _require_steps(self):
        if not self.steps:
            raise ValueError("steps must contain at least one component")

    def _require_fitted(self, method):
        if not self._fitted:
            raise NotFittedError(
                f"Pipeline must be fitted before {method}"
            )

    @staticmethod
    def _fit_transform_step(name, step, X):
        """Fit one transformer on X and return its output."""
        if callable(getattr(step, "fit_transform", None)):
            return step.fit_transform(X)

        if _has_methods(step, "fit", "transform"):
            step.fit(X)
            return step.transform(X)

        raise AttributeError(
            f"Step '{name}' does not support transformation; it must "
            "implement fit(X) and transform(X)"
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def fit_transform(self, X):
        """Fit every step as a transformer and return the transformed input.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to fit and transform.

        Returns
        -------
        np.ndarray
            Output of the final transformer; its shape depends on the steps.

        Raises
        ------
        ValueError
            If the pipeline contains no components, or a component rejects
            X (e.g. wrong dimensionality or NaN values).
        AttributeError
            If a component does not support transformation.

        Notes
        -----
        Time complexity: sum of the components' fit and transform costs.
        Space complexity: the largest intermediate array plus state.
        """
        self._require_steps()

        X_current = X
        for name, step in self.steps:
            X_current = self._fit_transform_step(name, step, X_current)

        self._fitted = True
        self._final_is_estimator = False
        return X_current

    def fit(self, X, y=None):
        """Fit the preprocessing steps and the optional final estimator.

        Intermediate components are fitted and applied sequentially. If the
        final component has a ``predict`` method, it is treated as an
        estimator and fitted with the transformed input and ``y``.
        Otherwise it is fitted as a transformer and ``y`` is ignored.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data used to fit the pipeline.
        y : array-like of shape (n_samples,) or None
            Target values for the final estimator. Required when the
            final step is an estimator.

        Returns
        -------
        Pipeline
            The fitted pipeline instance.

        Raises
        ------
        ValueError
            If the pipeline contains no components, or the final step is
            an estimator and ``y`` is None.
        AttributeError
            If a component does not implement the required methods.

        Notes
        -----
        Time complexity: sum of the components' fit/transform costs plus
        the final estimator's fit cost. Space complexity: the largest
        intermediate array plus component state.
        """
        self._require_steps()

        X_current = X
        for name, step in self.steps[:-1]:
            X_current = self._fit_transform_step(name, step, X_current)

        final_name, final_step = self.steps[-1]

        if callable(getattr(final_step, "predict", None)):
            if not callable(getattr(final_step, "fit", None)):
                raise AttributeError(
                    f"Final step '{final_name}' must implement fit(X, y)"
                )
            if y is None:
                raise ValueError(
                    "y is required when the final step is an estimator"
                )
            final_step.fit(X_current, y)
            self._final_is_estimator = True
        else:
            self._fit_transform_step(final_name, final_step, X_current)
            self._final_is_estimator = False

        self._fitted = True
        return self

    def transform(self, X):
        """Transform input data using the fitted preprocessing components.

        Applies each transformer's ``transform`` in order without
        refitting. A final estimator is skipped.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data with the same number of features used in fitting.

        Returns
        -------
        np.ndarray
            Transformed input data.

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted (subclass of RuntimeError).
        ValueError
            If a component rejects X, e.g. a feature-count mismatch.

        Notes
        -----
        Time complexity: sum of the transformers' transform costs.
        Space complexity: the largest intermediate array.
        """
        self._require_fitted("transform")

        X_current = X
        steps_to_apply = (
            self.steps[:-1] if self._final_is_estimator else self.steps
        )

        for _, step in steps_to_apply:
            X_current = step.transform(X_current)

        return X_current

    def predict(self, X):
        """Generate predictions using the fitted final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data with the same number of features used in fitting.

        Returns
        -------
        np.ndarray
            Predictions produced by the final estimator.

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted (subclass of RuntimeError).
        AttributeError
            If the final component is not an estimator.
        ValueError
            If a component rejects X, e.g. a feature-count mismatch.

        Notes
        -----
        Time complexity: sum of the transform costs plus the estimator's
        predict cost. Space complexity: the largest intermediate array.
        """
        self._require_fitted("predict")

        if not self._final_is_estimator:
            raise AttributeError(
                "The final pipeline step is a preprocessor, not an estimator"
            )

        X_current = X
        for _, step in self.steps[:-1]:
            X_current = step.transform(X_current)

        return self.steps[-1][1].predict(X_current)

    def predict_proba(self, X):
        """Transform X and return the final estimator's class probabilities.

        Args:
            X (numpy.ndarray): Input data with the same number of features
                used in fitting.

        Returns:
            numpy.ndarray: Probabilities of shape (n_samples, n_classes).

        Raises:
            NotFittedError: If the pipeline has not been fitted.
            AttributeError: If the final step has no ``predict_proba``.

        Notes
        -----
        Time complexity: sum of the transform costs plus the estimator's
        predict_proba cost.
        """
        self._require_fitted("predict_proba")
        final_name, final_step = self.steps[-1]

        if not callable(getattr(final_step, "predict_proba", None)):
            raise AttributeError(
                f"Final step '{final_name}' does not implement predict_proba"
            )

        X_current = X
        for _, step in self.steps[:-1]:
            X_current = step.transform(X_current)

        return final_step.predict_proba(X_current)

    def partial_fit(self, X_chunk, y_chunk=None):
        """Update every step incrementally with one chunk.

        Each transformer is updated with ``partial_fit`` and then used to
        transform the chunk for the next step, so the chunk is always
        transformed with statistics that include it. A final estimator
        receives ``partial_fit(X_transformed, y_chunk)``.

        Example::

            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("model", DecisionTreeClassifier(max_depth=5,
                                                 buffer_size=500)),
            ])
            for X_chunk, y_chunk in stream:
                pipe.partial_fit(X_chunk, y_chunk)

        Args:
            X_chunk (numpy.ndarray): Chunk of shape (n_samples, n_features).
                An empty chunk is a no-op.
            y_chunk (numpy.ndarray or None): Targets of shape (n_samples,);
                required when the final step is an estimator.

        Returns:
            Pipeline: The updated pipeline instance.

        Raises:
            ValueError: If the pipeline has no steps, or the final step is
                an estimator and ``y_chunk`` is None.
            AttributeError: If a step does not implement ``partial_fit``.
            ValueError / TypeError: As raised by the steps, e.g. for a
                feature-count mismatch between chunks.

        Notes
        -----
        Time complexity: sum of the steps' partial_fit and transform
        costs for one chunk. Space complexity: the largest intermediate
        chunk plus each step's own state.
        """
        self._require_steps()

        for name, step in self.steps:
            if not callable(getattr(step, "partial_fit", None)):
                raise AttributeError(
                    f"Step '{name}' does not support streaming; it must "
                    "implement partial_fit"
                )

        final_name, final_step = self.steps[-1]
        final_is_estimator = callable(getattr(final_step, "predict", None))

        if final_is_estimator and y_chunk is None:
            raise ValueError(
                "y_chunk is required when the final step is an estimator"
            )

        if np.shape(X_chunk)[:1] == (0,):
            return self

        X_current = X_chunk
        for _, step in self.steps[:-1]:
            step.partial_fit(X_current)
            X_current = step.transform(X_current)

        if final_is_estimator:
            final_step.partial_fit(X_current, y_chunk)
        else:
            final_step.partial_fit(X_current)

        self._final_is_estimator = final_is_estimator
        self._fitted = True
        return self


__all__ = ["Pipeline"]
