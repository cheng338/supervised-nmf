"""Guarded reuse of Euclidean KNN queries in explicit calibration grids.

Author: Chang Cheng. This private adapter preserves the public estimator's
decision on uncertain batches; it never moves a voting threshold to break a tie.
"""

import time
import warnings
from numbers import Integral

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KDTree, BallTree
from sklearn import __version__ as _SKLEARN_VERSION
try:
    from sklearn.metrics._dist_metrics import EuclideanDistance64 as _FULL_MEMBERSHIP_EUCLIDEAN
except ImportError:
    # Unsupported sklearn layouts retain their ordinary native route.
    _FULL_MEMBERSHIP_EUCLIDEAN = None


# Full-membership checks require these original callable objects.
# Replaced selected-query/decision methods must retain the ordinary fallback.
_FULL_MEMBERSHIP_METHODS = tuple(
    (name, getattr(KNeighborsClassifier, name))
    for name in ("kneighbors", "predict", "predict_proba")
)
_FULL_MEMBERSHIP_TREES = {
    "kd_tree": (KDTree, KDTree.query, KDTree.__getstate__),
    "ball_tree": (BallTree, BallTree.query, BallTree.__getstate__),
}


# Public Joblib contexts can execute user callbacks even with stock KNN.
# Inspect only native configuration and class dictionaries: do not resolve a
# backend by calling get_active_backend(), a factory, or a user descriptor.
from joblib import __version__ as _JOBLIB_CALLBACK_VERSION
from joblib import parallel as _callback_parallel
from sklearn.neighbors import _base as _callback_neighbors_base

_CALLBACK_MISSING = object()


def _callback_static(owner, name):
    for base in owner.__mro__:
        if name in base.__dict__:
            return base.__dict__[name]
    return _CALLBACK_MISSING


_CALLBACK_JOBLIB_FUNCTIONS = tuple(
    (name, getattr(_callback_parallel, name, None))
    for name in ("effective_n_jobs", "get_active_backend", "_get_active_backend", "_get_config_param")
)
_CALLBACK_EFFECTIVE = getattr(_callback_parallel, "effective_n_jobs", None)
_CALLBACK_LOCAL = getattr(_callback_parallel, "_backend", None)
_CALLBACK_DEFAULTS = getattr(_callback_parallel, "default_parallel_config", None)
_CALLBACK_DEFAULT_KEYS = ("backend", "n_jobs", "prefer", "require", "verbose")
_CALLBACK_SENTINELS = ({key: _CALLBACK_DEFAULTS.get(key) for key in _CALLBACK_DEFAULT_KEYS}
                       if type(_CALLBACK_DEFAULTS) is dict else {})
_CALLBACK_CLASS_FIELDS = ("__init__", "__new__", "__getattribute__", "effective_n_jobs",
                          "in_main_thread", "uses_threads", "supports_sharedmem",
                          "__dict__", "nesting_level")
_CALLBACK_STOCK = {}
for _name in ("SequentialBackend", "ThreadingBackend", "LokyBackend", "MultiprocessingBackend"):
    _owner = getattr(_callback_parallel, _name, None)
    if isinstance(_owner, type):
        _CALLBACK_STOCK[_owner] = tuple(_callback_static(_owner, name) for name in _CALLBACK_CLASS_FIELDS)
_CALLBACK_TAG_CHAIN = tuple(
    (owner, owner.__dict__.get("__sklearn_tags__", _CALLBACK_MISSING))
    for owner in KNeighborsClassifier.__mro__
)


def _native_callback_context(model):
    """Permit proof only when omitted native calls contain no user callbacks."""
    try:
        if (type(model) is not KNeighborsClassifier or _JOBLIB_CALLBACK_VERSION != "1.6.0"
                or _CALLBACK_EFFECTIVE is None
                or _callback_neighbors_base.effective_n_jobs is not _CALLBACK_EFFECTIVE
                or "__sklearn_tags__" in model.__dict__
                or "__sklearn_is_fitted__" in model.__dict__):
            return False
        for owner, method in _CALLBACK_TAG_CHAIN:
            if (owner.__dict__.get("__sklearn_tags__", _CALLBACK_MISSING) is not method
                    or "__sklearn_is_fitted__" in owner.__dict__):
                return False
        jobs = model.n_jobs
        if type(jobs) is int and jobs == 1:
            # Original effective_n_jobs returns here without backend lookup.
            return True
        if not (jobs is None or (type(jobs) is int and jobs != 0)):
            return False
        if (_callback_parallel._backend is not _CALLBACK_LOCAL
                or _callback_parallel.default_parallel_config is not _CALLBACK_DEFAULTS
                or type(_CALLBACK_DEFAULTS) is not dict):
            return False
        for name, function in _CALLBACK_JOBLIB_FUNCTIONS:
            if function is None or getattr(_callback_parallel, name, None) is not function:
                return False
        config = getattr(_CALLBACK_LOCAL, "config", _CALLBACK_DEFAULTS)
        if type(config) is not dict:
            return False
        current = {}
        for key in _CALLBACK_DEFAULT_KEYS:
            sentinel = _CALLBACK_SENTINELS[key]
            if _CALLBACK_DEFAULTS.get(key, _CALLBACK_MISSING) is not sentinel:
                return False
            value = config.get(key, _CALLBACK_MISSING)
            if value is _CALLBACK_MISSING:
                return False
            current[key] = (0 if key == "verbose" else None) if value is sentinel else value
        # Non-default backend hints can choose another factory or inspect its
        # flags. Retain the original route rather than emulate that resolver.
        if current["prefer"] is not None or current["require"] is not None:
            return False
        if type(current["verbose"]) is not int:
            return False
        if jobs is None:
            jobs = current["n_jobs"]
            if not (jobs is None or (type(jobs) is int and jobs != 0)):
                return False
        backend = current["backend"]
        if backend is None:
            registry = _callback_parallel.BACKENDS
            name = _callback_parallel.DEFAULT_BACKEND
            if (type(registry) is not dict or type(name) is not str
                    or any(type(key) is not str for key in registry)):
                return False
            factory = registry.get(name)
            if not any(factory is owner for owner in _CALLBACK_STOCK):
                return False
            owner = factory
            state = None
        else:
            owner = type(backend)
            if not any(owner is known for known in _CALLBACK_STOCK):
                return False
        originals = _CALLBACK_STOCK[owner]
        if any(_callback_static(owner, name) is not original
               for name, original in zip(_CALLBACK_CLASS_FIELDS, originals)):
            return False
        if backend is not None:
            # Class descriptors are still the captured stock objects before
            # reading the plain instance dictionary. Unknown class hashing,
            # properties and custom factories are never invoked for admission.
            state = object.__getattribute__(backend, "__dict__")
            if type(state) is not dict:
                return False
            if any(name in state for name in ("effective_n_jobs", "in_main_thread", "__getattribute__")):
                return False
            nesting = state.get("nesting_level", None)
            if not (nesting is None or (type(nesting) is int and nesting >= 0)):
                return False
            for name in ("uses_threads", "supports_sharedmem"):
                if name in state and type(state[name]) is not bool:
                    return False
        return True
    except Exception:
        return False


class _KNNQueryReuse:
    """Share neighbor order only among fitted heads with identical training data.

    ``models`` contains successful, independently fitted heads indexed by k.
    Unknown sklearn state, sparse inputs, callable weights and other metrics use
    the public selected-k estimator. A query error or any uncertain row also
    falls back for the whole original query batch, preserving its kernel shape.
    Private fitted-state use is deliberately confined to this adapter.
    """

    def __init__(self, models, query, *, _study_call_contract=None):
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
        if _study_call_contract is not None:
            self._install_study_contract(_study_call_contract)

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

    def _prefix_votes(self, group):
        # The caller admits only ordinary policy and finite native distances.
        # Keep all per-head guards; this only avoids building unused prefixes.
        if "prefix_votes" not in group and not group.get("prefix_unavailable", False):
            try:
                model = group["model"]
                distances = group["distances"]
                epsilon = np.finfo(distances.dtype).eps
                ks = []
                selected_ks = group["ks"]
                if group.get("study_short_query_active", False):
                    selected_ks = [value for value in selected_ks
                                   if value < len(model._fit_X)]
                for selected_k in sorted(selected_ks):
                    if selected_k == len(model._fit_X):
                        ks.append(selected_k)
                    else:
                        left, right = distances[:, selected_k - 1], distances[:, selected_k]
                        radius = np.maximum(np.abs(left), np.abs(right))
                        near_boundary = np.abs(right - left) <= 64.0 * epsilon * radius
                        if not near_boundary.any():
                            ks.append(selected_k)
                if len(ks) < 2:
                    group["prefix_unavailable"] = True
                    group["prefix_skip_reason"] = "fewer_than_two_boundary_clean_heads"
                    return None
                maximum = ks[-1]
                distance = distances[:, :maximum]
                # Sorted native distances put zero neighbors first; this
                # max-prefix mask equals the original mask at every positive k.
                if model.weights == "uniform":
                    weights = np.ones_like(distance)
                else:
                    zero = distance == 0
                    with np.errstate(divide="ignore", over="ignore"):
                        weights = 1.0 / distance
                    zero_rows = zero.any(axis=1)
                    weights[zero_rows] = zero[zero_rows]
                encoded = model._y[group["indices"][:, :maximum]]
                all_votes = np.empty((len(ks), self.query_rows, len(model.classes_)),
                                     dtype=np.float64)
                starts = np.asarray([0] + ks[:-1], dtype=np.intp)
                # Sorted unique positive ks fill every prefix exactly when H=K.
                complete_prefix = len(ks) == maximum
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for j in range(len(model.classes_)):
                        masked = weights * (encoded == j)
                        if complete_prefix:
                            prefix = np.cumsum(masked, axis=1, dtype=np.float64)
                        else:
                            intervals = np.add.reduceat(masked, starts, axis=1,
                                                        dtype=np.float64)
                            prefix = np.cumsum(intervals, axis=1, dtype=np.float64)
                        for position, selected_k in enumerate(ks):
                            all_votes[position, :, j] = prefix[:, position]
                # Publish only complete selected-k rows, after all classes.
                group["prefix_votes"] = {selected_k: all_votes[position]
                                         for position, selected_k in enumerate(ks)}
            except Exception as error:
                # Internal diagnostics only; optional failure uses the original
                # selected-k calculation and never exposes partial votes.
                group["prefix_unavailable"] = True
                group["prefix_error_type"] = type(error).__name__
                group["prefix_error"] = str(error)
        return group.get("prefix_votes")

    def predict(self, k, *, probabilities=False, decision_threshold=None, positive_index=None):
        """Return predictions (or probabilities), diagnostics and shared warnings."""
        model = self.models[k]
        group, reason = self.by_k[k]
        study_plan = self._study_plan_for_call(
            group, k, probabilities, decision_threshold, positive_index)
        metadata = {"accelerated": False, "shared_query_attempted": False,
                    "public_fallback_query_rows": self.query_rows}
        if group is None or (len(group["ks"]) == 1 and study_plan is None):
            metadata["reason"] = reason or "single_head_backend"
            return self._public(model, probabilities), metadata, []
        planned_full_votes = None
        if study_plan is None:
            self._query(group)
        else:
            planned_full_votes = self._study_query(group, study_plan, k, probabilities)
        metadata.update(shared_query_attempted=True, shared_query_key=group["key"],
                        backend=model._fit_method)
        if (study_plan is not None and study_plan["sole_full_head"]
                and planned_full_votes is None):
            # Baseline sole-head behavior is one public prediction call. Do
            # not add a full shared retry or duplicate its visible warnings.
            metadata["reason"] = "single_head_backend"
            return self._public(model, probabilities), metadata, []
        if "error" in group:
            metadata["reason"] = "shared_query_failed"
            return self._public(model, probabilities), metadata, group["warnings"]
        distances, indices = group["distances"], group["indices"]
        # The shared query arrays are privately owned and immutable for this
        # adapter's lifetime. Qualify the complete max-k-plus-one matrix once.
        if "invalid_distance" not in group:
            group["invalid_distance"] = ~np.isfinite(distances).all(axis=1)
        invalid_distance = group["invalid_distance"]
        if invalid_distance.any():
            # Even finite input can overflow distance arithmetic. In particular
            # inf - inf is NaN and would evade the ordinary boundary-tie guard.
            # Do not qualify cached neighbor order in this state.
            metadata.update(reason="nonfinite_neighbor_distances",
                            invalid_distance_rows=int(invalid_distance.sum()))
            return self._public(model, probabilities), metadata, group["warnings"]
        # Reordering qualification is permitted only under the ordinary
        # NumPy policy: strict underflow/error paths retain their original order.
        # The public selected-k call still receives the entire original batch.
        early_boundary = None
        threshold_is_valid = (
            decision_threshold is None
            or (type(decision_threshold) is float
                and 0.0 <= decision_threshold <= 1.0
                and isinstance(positive_index, Integral)
                and not isinstance(positive_index, bool)
                and 0 <= positive_index < len(model.classes_)))
        if (k < len(model._fit_X) and threshold_is_valid
                and np.geterr() == {"divide": "warn", "over": "warn",
                                    "under": "ignore", "invalid": "warn"}):
            cached_prefix = group.get("prefix_votes")
            if cached_prefix is not None and k in cached_prefix:
                # Successful publication already proved this exact k clean on
                # the same immutable distances. The earlier finite-distance
                # guard established this private mask is all false; the tail
                # only reads it, and creates a fresh uncertainty union.
                early_boundary = invalid_distance
            else:
                boundary_epsilon = np.finfo(distances.dtype).eps
                left, right = distances[:, k - 1], distances[:, k]
                radius = np.maximum(np.abs(left), np.abs(right))
                early_boundary = np.abs(right - left) <= 64.0 * boundary_epsilon * radius
            if early_boundary.any():
                boundary_rows = int(early_boundary.sum())
                metadata.update(reason="numerical_or_boundary_uncertainty",
                                near_boundary_rows=boundary_rows,
                                near_vote_rows=None, near_threshold_rows=None,
                                invalid_vote_rows=None, vote_guard=None,
                                uncertain_rows=None, vote_checks_performed=False,
                                uncertainty_lower_bound_rows=boundary_rows)
                return self._public(model, probabilities), metadata, group["warnings"]
        prefix_votes = None
        if (threshold_is_valid
                and np.geterr() == {"divide": "warn", "over": "warn",
                                    "under": "ignore", "invalid": "warn"}):
            prefix_votes = self._prefix_votes(group)
        early_votes = planned_full_votes
        if (early_votes is None and prefix_votes is None and "prefix_votes" not in group
                and group.get("prefix_skip_reason") == "fewer_than_two_boundary_clean_heads"
                and "prefix_error_type" not in group
                and type(k) is int and k == len(model._fit_X)
                and type(model.weights) is str and model.weights == "uniform"):
            early_votes = self._full_membership_row_votes(
                group, model, k, probabilities=probabilities,
                threshold_is_valid=threshold_is_valid)
        if early_votes is not None:
            votes = early_votes
            metadata["full_membership_vote_basis"] = (
                "short_query_training_histogram" if planned_full_votes is not None
                else "native_query_training_histogram")
        elif prefix_votes is None:
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
        else:
            votes = prefix_votes[k]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            totals = votes.sum(axis=1, keepdims=True)
            probability = votes / totals
        epsilon = np.finfo(distances.dtype).eps
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
        if early_boundary is not None:
            near_boundary = early_boundary
        elif k == len(model._fit_X):
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
            if (type(k) is int and k == len(model._fit_X)
                    and type(model.weights) is str and model.weights == "uniform"
                    and not invalid.any()
                    and (early_votes is not None or self._full_membership_certified(
                        group, model, k, votes=votes, probabilities=probabilities,
                        threshold_is_valid=threshold_is_valid))):
                metadata.update(full_membership_certified=True,
                                full_membership_basis="native_training_histogram",
                                reason="uniform_full_membership_certified")
            else:
                metadata["reason"] = "numerical_or_boundary_uncertainty"
                return self._public(model, probabilities), metadata, group["warnings"]
        metadata.update(accelerated=True, public_fallback_query_rows=0)
        result = probability if probabilities else model.classes_[probability.argmax(axis=1)]
        return result, metadata, group["warnings"]

    def _full_membership_row_votes(self, group, model, k, *, probabilities,
                                   threshold_is_valid):
        """Use the proved stock full-query histogram when prefixes are unavailable."""
        # The original shared query and finite-distance validation still run.
        # Under the common stock-method/magnitude gates, full-k native queries
        # contain every training row. Private query arrays remain immutable.
        # Unknown contexts and optional allocation failures retain native fallback.
        if (type(probabilities) is not bool or not threshold_is_valid
                or group["warnings"]
                or np.geterr() != {"divide": "warn", "over": "warn",
                                  "under": "ignore", "invalid": "warn"}
                or np.geterrcall() is not None):
            return None
        try:
            indices, encoded, classes = group["indices"], model._y, model.classes_
            if (type(indices) is not np.ndarray or indices.dtype != np.dtype(np.intp)
                    or indices.shape != (self.query_rows, k)
                    or type(encoded) is not np.ndarray or encoded.ndim != 1
                    or encoded.dtype != np.dtype(np.intp) or len(encoded) != k
                    or type(classes) is not np.ndarray or classes.ndim != 1
                    or not 1 <= len(classes) <= k <= 2**53
                    or (encoded < 0).any() or (encoded >= len(classes)).any()):
                return None
            context = self._full_membership_context(
                group, model, k, probabilities=probabilities,
                threshold_is_valid=threshold_is_valid)
            if context is None:
                return None
            encoded, classes = context
            expected = np.bincount(encoded, minlength=len(classes))
            cell_count = self.query_rows * len(classes)
            if cell_count > np.iinfo(np.intp).max:
                return None
            # Stock native full-k membership proves these counts for each
            # query row; n<=2**53 makes their float64 conversion exact. Keep a
            # fresh owned matrix so no fitted state or result aliases a cache.
            return np.broadcast_to(
                expected, (self.query_rows, len(classes))).astype(np.float64)
        except Exception:
            # No warning-policy change or partial publication accompanies a
            # failed optional histogram broadcast.
            pass
        return None

    def _full_membership_certified(self, group, model, k, *, votes, probabilities,
                                   threshold_is_valid):
        """Certify existing votes without changing their validation order."""
        context = self._full_membership_context(
            group, model, k, probabilities=probabilities,
            threshold_is_valid=threshold_is_valid)
        if context is None:
            return False
        encoded, classes = context
        try:
            if (type(votes) is not np.ndarray or votes.dtype != np.dtype(np.float64)
                    or votes.shape != (self.query_rows, len(classes))):
                return False
            # The bounded native all-n query visits every training row. Its
            # uniform float64 histogram is exact for n<=2**53. Certify the
            # already computed votes, not the identities of shared indices.
            expected = np.bincount(encoded, minlength=len(classes))
            return bool((votes == expected).all())
        except Exception:
            # This optional proof must not prevent the original selected-k call.
            return False

    def _full_membership_context(self, group, model, k, *, probabilities,
                                 threshold_is_valid):
        """Validate the unchanged native all-training-row context."""
        if not _native_callback_context(model):
            return None
        if (type(k) is not int or type(probabilities) is not bool
                or type(model.weights) is not str or model.weights != "uniform"
                or not threshold_is_valid or _SKLEARN_VERSION != "1.9.1"
                or type(model.metric) is not str
                or model.metric not in {"euclidean", "l2", "minkowski"}
                or not (type(model.p) is int or type(model.p) is float)
                or model.p != 2
                or not (model.metric_params is None
                        or (type(model.metric_params) is dict and not model.metric_params))):
            return None
        try:
            train, query = model._fit_X, self.query
            if (type(train) is not np.ndarray or type(query) is not np.ndarray
                    or train.dtype != query.dtype or train.ndim != 2 or query.ndim != 2
                    or train.shape[1] < 1 or query.shape[1] != train.shape[1]
                    or model.outputs_2d_ is not False
                    or train.shape[1] * np.finfo(np.float64).eps >= 0.25
                    or group["model"] is not model or group["warnings"]
                    or type(model.n_neighbors) is not int
                    or type(model.n_samples_fit_) is not int
                    or not k == model.n_neighbors == model.n_samples_fit_ == len(train)
                    or not 1 <= k <= 2**53):
                return None
            if (np.geterr() != {"divide": "warn", "over": "warn",
                              "under": "ignore", "invalid": "warn"}
                    or np.geterrcall() is not None):
                return None
            for name, original in _FULL_MEMBERSHIP_METHODS:
                if (name in model.__dict__
                        or getattr(getattr(model, name), "__func__", None) is not original):
                    return None
            backend = model._fit_method
            if type(backend) is not str:
                return None
            if backend == "brute":
                if model._tree is not None:
                    return None
            else:
                tree_spec = _FULL_MEMBERSHIP_TREES.get(backend)
                tree = model._tree
                if (tree_spec is None or type(tree) is not tree_spec[0]
                        or type(tree).query is not tree_spec[1]
                        or "query" in getattr(tree, "__dict__", {})
                        or _FULL_MEMBERSHIP_EUCLIDEAN is None
                        or type(tree).__getstate__ is not tree_spec[2]):
                    return None
                # A public fit from a prefit neighbor estimator can copy a
                # tree with a different hidden metric. Inspect the original
                # state descriptor without invoking a custom metric callback.
                tree_state = tree_spec[2](tree)
                if (type(tree_state) is not tuple or len(tree_state) != 13
                        or type(tree_state[11]) is not _FULL_MEMBERSHIP_EUCLIDEAN):
                    return None
            encoded, classes = model._y, model.classes_
            if (type(encoded) is not np.ndarray or encoded.ndim != 1
                    or encoded.dtype != np.dtype(np.intp) or len(encoded) != k
                    or type(classes) is not np.ndarray or classes.ndim != 1
                    or not 1 <= len(classes) <= k
                    or (encoded < 0).any() or (encoded >= len(classes)).any()):
                return None
            # Native ClassMode uses direct differences, unlike the shared
            # Euclidean GEMM route. Keep their squared sums well below the
            # input-dtype limit, including the float32 metric return type.
            bound = float(np.sqrt(float(np.finfo(train.dtype).max)
                                  / (16.0 * train.shape[1])))
            if not (np.max(np.abs(train)) <= bound and np.max(np.abs(query)) <= bound):
                return None
            return encoded, classes
        except Exception:
            # This optional proof must not prevent the original selected-k call.
            return None

    def _install_study_contract(self, contract):
        """Admit only the operation-owned study's fixed, callback-free call loop."""
        try:
            if (type(contract) is not tuple or len(contract) != len(self.models)
                    or len(contract) < 1 or len(self.groups) != 1
                    or len(self.groups[0]["ks"]) != len(self.models)
                    or np.geterr() != {"divide": "warn", "over": "warn",
                                      "under": "ignore", "invalid": "warn"}
                    or np.geterrcall() is not None):
                return
            for row in contract:
                if (type(row) is not tuple or len(row) != 4 or type(row[0]) is not int
                        or type(row[1]) is not bool
                        or not (row[2] is None or (type(row[2]) is float
                                and 0.0 <= row[2] <= 1.0))
                        or not (row[3] is None or type(row[3]) is int)):
                    return
            if (tuple(row[0] for row in contract) != tuple(self.models)
                    or any(row[1:] != contract[0][1:] for row in contract)):
                return
            group = self.groups[0]
            full = group["model"]
            # Brute query width changes selection, not distance computation.
            # Full-n tree membership alone does not prove rounded pruning
            # equivalence for a shorter query; retain full queries for both tree routes.
            if type(full._fit_method) is not str or full._fit_method != "brute":
                return
            n = len(full._fit_X)
            if n not in self.models or self.models[n] is not full:
                return
            smaller = [row[0] for row in contract if row[0] < n]
            count = max(smaller) + 1 if smaller else 1
            if count >= n:
                return
            for key, model in self.models.items():
                if (type(model) is not KNeighborsClassifier
                        or type(model.n_neighbors) is not int or model.n_neighbors != key
                        or type(model.weights) is not str or model.weights != "uniform"
                        or type(model.metric) is not str
                        or model.metric not in {"euclidean", "l2", "minkowski"}
                        or not (type(model.p) is int or type(model.p) is float) or model.p != 2
                        or not (model.metric_params is None or
                                (type(model.metric_params) is dict and not model.metric_params))
                        or not (model.n_jobs is None or (type(model.n_jobs) is int and model.n_jobs != 0))
                        or type(model._fit_method) is not str
                        or type(model.classes_) is not np.ndarray
                        or model.classes_.dtype.kind not in "biufUS"
                        or not _native_callback_context(model)):
                    return
                for name, original in _FULL_MEMBERSHIP_METHODS:
                    if (name in model.__dict__
                            or getattr(getattr(model, name), "__func__", None) is not original):
                        return
                if model._fit_method != "brute":
                    tree_spec = _FULL_MEMBERSHIP_TREES.get(model._fit_method)
                    tree = model._tree
                    if (tree_spec is None or type(tree) is not tree_spec[0]
                            or type(tree).query is not tree_spec[1]
                            or "query" in getattr(tree, "__dict__", {})
                            or _FULL_MEMBERSHIP_EUCLIDEAN is None
                            or type(tree).__getstate__ is not tree_spec[2]):
                        return
                    state = tree_spec[2](tree)
                    if (type(state) is not tuple or len(state) != 13
                            or type(state[11]) is not _FULL_MEMBERSHIP_EUCLIDEAN):
                        return
            _, probabilities, threshold, positive_index = contract[0]
            if threshold is not None and not (
                    type(positive_index) is int and 0 <= positive_index < len(full.classes_)):
                return
            context = self._full_membership_context(
                group, full, n, probabilities=probabilities, threshold_is_valid=True)
            if context is None:
                return
            self._study_contract = contract
            self._study_position = 0
            self._study_plan = {"group": group, "full_k": n, "count": count,
                                "sole_full_head": not smaller}
        except Exception:
            # Optional planning never changes ordinary constructor errors.
            return

    def _study_plan_for_call(self, group, k, probabilities, threshold, positive_index):
        plan = getattr(self, "_study_plan", None)
        if plan is None:
            return None
        position = self._study_position
        expected = self._study_contract[position] if position < len(self._study_contract) else None
        actual = (k, probabilities, threshold, positive_index)
        same = (expected is not None
                and all(type(a) is type(b) for a, b in zip(actual, expected))
                and actual == expected)
        ordinary = (np.geterr() == {"divide": "warn", "over": "warn",
                                   "under": "ignore", "invalid": "warn"}
                    and np.geterrcall() is None)
        if not same or not ordinary:
            if plan["group"].get("study_short_query_active", False):
                # This private mode is supplied only by the fixed study loop.
                # Generic varying-policy/request adapters never enable it.
                raise RuntimeError("The operation-owned KNN call contract changed after query planning.")
            self._study_plan = None
            return None
        self._study_position += 1
        return plan if group is plan["group"] else None

    def _study_query(self, group, plan, k, probabilities):
        """Retain a real native query and charge every attempted query once."""
        if not group["queried"]:
            model = group["model"]
            error = None
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    distances, indices = model.kneighbors(self.query, n_neighbors=plan["count"])
                except Exception as failure:
                    error = failure
                finally:
                    elapsed = time.perf_counter() - start
                    self.query_seconds += elapsed
            group["short_query_attempt_seconds"] = elapsed
            group["shared_query_neighbor_counts"] = [plan["count"]]
            if error is not None or caught:
                if error is not None:
                    group["short_query_error_type"] = type(error).__name__
                    group["short_query_error"] = str(error)
                group["short_query_warning_records"] = [
                    {"category": item.category.__name__, "message": str(item.message)}
                    for item in caught]
                if plan["sole_full_head"]:
                    group.update(queried=True, query_seconds=elapsed, warnings=[],
                                 study_short_query_active=False,
                                 short_query_upgrade_reason="single_head_query_public_fallback")
                else:
                    self._query(group)
                    group["query_seconds"] += elapsed
                    group["shared_query_neighbor_counts"].append(plan["full_k"])
                    group["study_short_query_active"] = False
                    group["short_query_upgrade_reason"] = "short_query_warning_or_error"
            else:
                group.update(queried=True, distances=distances, indices=indices,
                             query_seconds=elapsed, warnings=[], study_short_query_active=True)
        if not group.get("study_short_query_active", False) or k != plan["full_k"]:
            return None
        # Preserve the original complete matrix check before producing votes.
        if "invalid_distance" not in group:
            group["invalid_distance"] = ~np.isfinite(group["distances"]).all(axis=1)
        if group["invalid_distance"].any():
            if plan["sole_full_head"]:
                group["study_short_query_active"] = False
                group["short_query_upgrade_reason"] = "single_head_nonfinite_public_fallback"
            return None
        votes = self._study_full_votes(group, plan, probabilities)
        if votes is not None:
            return votes
        if plan["sole_full_head"]:
            group["study_short_query_active"] = False
            group["short_query_upgrade_reason"] = "single_head_histogram_public_fallback"
            return None
        previous = group["query_seconds"]
        for key in ("distances", "indices", "invalid_distance", "prefix_votes",
                    "prefix_unavailable", "prefix_skip_reason", "prefix_error_type", "prefix_error",
                    "error_type", "error"):
            group.pop(key, None)
        group["queried"] = False
        group["study_short_query_active"] = False
        self._query(group)
        group["query_seconds"] += previous
        group["shared_query_neighbor_counts"].append(plan["full_k"])
        group["short_query_upgrade_reason"] = "full_histogram_not_available"
        return None

    def _study_full_votes(self, group, plan, probabilities):
        try:
            context = self._full_membership_context(
                group, group["model"], plan["full_k"],
                probabilities=probabilities, threshold_is_valid=True)
            if context is None:
                return None
            encoded, classes = context
            if self.query_rows * len(classes) > np.iinfo(np.intp).max:
                return None
            expected = np.bincount(encoded, minlength=len(classes))
            return np.broadcast_to(expected, (self.query_rows, len(classes))).astype(np.float64)
        except Exception:
            return None

    def _public(self, model, probabilities):
        return (model.predict_proba(self.query) if probabilities
                else model.predict(self.query))

    def query_records(self):
        """Return JSON-compatible shared-cost records, excluding query arrays."""
        records = []
        for group in self.groups:
            record = {key: value for key, value in group.items()
                      if key in {"key", "ks", "queried", "query_seconds", "error_type", "error",
                                 "study_short_query_active", "shared_query_neighbor_counts",
                                 "short_query_attempt_seconds", "short_query_warning_records",
                                 "short_query_error_type", "short_query_error",
                                 "short_query_upgrade_reason"}}
            record["warnings"] = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in group["warnings"]
            ]
            records.append(record)
        return records
