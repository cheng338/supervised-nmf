"""Guarded reuse of Euclidean KNN queries in explicit calibration grids.

Author: Chang Cheng. This private adapter preserves the public estimator's
decision on uncertain batches; it never moves a voting threshold to break a tie.
"""

import time
import warnings
from numbers import Integral

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class _KNNQueryReuse:
    """Share neighbor order only among fitted heads with identical training data.

    ``models`` contains successful, independently fitted heads indexed by k.
    Unknown sklearn state, sparse inputs, callable weights and other metrics use
    the public selected-k estimator. A query error or any uncertain row also
    falls back for the whole original query batch, preserving its kernel shape.
    Private fitted-state use is deliberately confined to this adapter.
    """

    def __init__(self, models, query):
        self.models = models
        self.query = query
        # Sparse matrices deliberately reject len(); their unsupported path
        # must still reach the ordinary public estimator.
        self.query_rows = query.shape[0] if hasattr(query, "shape") else len(query)
        self.groups = []
        self.by_k = {}
        self.query_seconds = 0.0
        for k, model in models.items():
            reason = self._unsupported(model, query)
            if reason is not None:
                self.by_k[k] = (None, reason)
                continue
            group = next((group for group in self.groups
                          if self._same_geometry(group["model"], model)), None)
            if group is None:
                group = {"model": model, "ks": [], "queried": False,
                         "key": len(self.groups), "warnings": [],
                         "query_seconds": 0.0}
                self.groups.append(group)
            group["ks"].append(k)
            if k > group["model"].n_neighbors:
                group["model"] = model
            self.by_k[k] = (group, None)

    @staticmethod
    def _unsupported(model, query):
        if type(model) is not KNeighborsClassifier:
            return "unsupported_estimator"
        if not isinstance(model.weights, str) or model.weights not in {"uniform", "distance"}:
            return "callable_or_unsupported_weights"
        if getattr(model, "_fit_method", None) not in {"brute", "kd_tree", "ball_tree"}:
            return "unknown_neighbor_backend"
        if (getattr(model, "effective_metric_", None) != "euclidean"
                or getattr(model, "effective_metric_params_", None)):
            return "non_euclidean_metric"
        train = getattr(model, "_fit_X", None)
        classes = getattr(model, "classes_", None)
        encoded = getattr(model, "_y", None)
        if (not isinstance(train, np.ndarray) or not isinstance(query, np.ndarray)
                or train.ndim != 2 or query.ndim != 2
                or train.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or query.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not isinstance(classes, np.ndarray) or classes.ndim != 1
                or not isinstance(encoded, np.ndarray) or encoded.ndim != 1
                or len(encoded) != len(train) or query.shape[1] != train.shape[1]
                or not len(query)):
            return "unsupported_fitted_state_or_array"
        if (not isinstance(model.n_neighbors, Integral) or isinstance(model.n_neighbors, bool)
                or not 1 <= model.n_neighbors <= len(train)):
            return "unsupported_neighbor_count"
        return None

    @staticmethod
    def _same_geometry(left, right):
        return (left._fit_X is right._fit_X
                and left._fit_method == right._fit_method
                and left.weights == right.weights
                and np.array_equal(left.classes_, right.classes_)
                and np.array_equal(left._y, right._y))

    def _query(self, group):
        if group["queried"]:
            return
        group["queried"] = True
        model = group["model"]
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                count = min(len(model._fit_X), max(group["ks"]) + 1)
                group["distances"], group["indices"] = model.kneighbors(
                    self.query, n_neighbors=count)
            except Exception as error:
                # An optimization error must not suppress a valid smaller-k
                # candidate. Each public selected-k call is still attempted.
                group["error_type"] = type(error).__name__
                group["error"] = str(error)
            finally:
                group["query_seconds"] = time.perf_counter() - start
                self.query_seconds += group["query_seconds"]
        group["warnings"] = list(caught)

    def predict(self, k, *, probabilities=False, decision_threshold=None, positive_index=None):
        """Return predictions (or probabilities), diagnostics and shared warnings."""
        model = self.models[k]
        group, reason = self.by_k[k]
        metadata = {"accelerated": False, "shared_query_attempted": False,
                    "public_fallback_query_rows": self.query_rows}
        if group is None or len(group["ks"]) == 1:
            metadata["reason"] = reason or "single_head_backend"
            return self._public(model, probabilities), metadata, []
        self._query(group)
        metadata.update(shared_query_attempted=True, shared_query_key=group["key"],
                        backend=model._fit_method)
        if "error" in group:
            metadata["reason"] = "shared_query_failed"
            return self._public(model, probabilities), metadata, group["warnings"]
        distances, indices = group["distances"], group["indices"]
        distance = distances[:, :k]
        if model.weights == "uniform":
            vote_weights = np.ones_like(distance)
        else:
            zero = distance == 0
            with np.errstate(divide="ignore", over="ignore"):
                vote_weights = 1.0 / distance
            zero_rows = zero.any(axis=1)
            vote_weights[zero_rows] = zero[zero_rows]
        encoded = model._y[indices[:, :k]]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            votes = np.column_stack([
                np.sum(vote_weights * (encoded == j), axis=1, dtype=np.float64)
                for j in range(len(model.classes_))
            ])
            totals = votes.sum(axis=1, keepdims=True)
            probability = votes / totals
        epsilon = np.finfo(vote_weights.dtype).eps
        product = (k + len(model.classes_) + 4) * epsilon
        gamma = product / (1.0 - product) if product < .25 else np.inf
        # A conservative forward-error guard covers differing nonnegative vote
        # reductions. It selects the public calculation, never a new decision.
        guard = 32.0 * gamma
        invalid = (~np.isfinite(probability).all(axis=1)
                   | ~np.isfinite(votes).all(axis=1)
                   | ~np.isfinite(totals[:, 0]) | (totals[:, 0] <= 0))
        if len(model.classes_) > 1:
            largest = np.sort(probability, axis=1)[:, -2:]
            margin = largest[:, 1] - largest[:, 0]
            near_vote = ~np.isfinite(margin) | (margin <= guard)
        else:
            margin = np.ones(self.query_rows)
            near_vote = np.zeros(self.query_rows, dtype=bool)
        near_threshold = np.zeros(self.query_rows, dtype=bool)
        if decision_threshold is not None:
            near_threshold = np.abs(probability[:, positive_index] - decision_threshold) <= guard
        if k == len(model._fit_X):
            near_boundary = np.zeros(self.query_rows, dtype=bool)
        else:
            left, right = distances[:, k - 1], distances[:, k]
            radius = np.maximum(np.abs(left), np.abs(right))
            near_boundary = np.abs(right - left) <= 64.0 * epsilon * radius
        uncertain = invalid | near_vote | near_threshold | near_boundary
        metadata.update(near_boundary_rows=int(near_boundary.sum()),
                        near_vote_rows=int(near_vote.sum()),
                        near_threshold_rows=int(near_threshold.sum()),
                        invalid_vote_rows=int(invalid.sum()),
                        vote_guard=float(guard), uncertain_rows=int(uncertain.sum()))
        if uncertain.any():
            metadata["reason"] = "numerical_or_boundary_uncertainty"
            return self._public(model, probabilities), metadata, group["warnings"]
        metadata.update(accelerated=True, public_fallback_query_rows=0)
        result = probability if probabilities else model.classes_[probability.argmax(axis=1)]
        return result, metadata, group["warnings"]

    def _public(self, model, probabilities):
        return (model.predict_proba(self.query) if probabilities
                else model.predict(self.query))

    def query_records(self):
        """Return JSON-compatible shared-cost records, excluding query arrays."""
        records = []
        for group in self.groups:
            record = {key: value for key, value in group.items()
                      if key in {"key", "ks", "queried", "query_seconds", "error_type", "error"}}
            record["warnings"] = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in group["warnings"]
            ]
            records.append(record)
        return records
