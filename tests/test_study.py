"""Equivalence, leakage and cache checks for explicit shared-KNN grids."""

from collections import ChainMap, OrderedDict, UserDict, UserList, deque, namedtuple
from email.message import Message
import json
from operator import itemgetter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import NMF, PCA
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import BallTree, KDTree, KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from supervised_nmf import NMFClassifier
from supervised_nmf.evaluation import calibration_search
from supervised_nmf.screening import BinaryScreening
from supervised_nmf.study import calibration_knn_grid
from supervised_nmf._neighbors import _KNNQueryReuse
from tests.test_neighbor_full_membership import native_queries


class _ArrayCoordinates:
    def __init__(self, values):
        self.values = values

    def __array__(self, dtype=None, copy=None):
        values = np.asarray(self.values, dtype=dtype)
        return values.copy() if copy else values


class _TransformConfiguration:
    def __init__(self, inner):
        self.inner = inner

    def get_params(self, deep=True):
        return {"inner": self.inner}


class _NumericArrayConfiguration(np.ndarray):
    def get_params(self, deep=True):
        return {"inner": self.inner}


class _FitTransformOnly(BaseEstimator):
    def __init__(self, inner):
        self.inner = inner

    def fit_transform(self, X, y=None):
        self.inner.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        return self.inner.transform(X)


class _ReverseFitRows:
    def __init__(self, only_k=None):
        self.only_k = only_k

    def setup(self, estimator, context):
        pass

    def on_fit_task_begin(self, estimator, context):
        pass

    def on_fit_task_end(self, estimator, context, *, X, y):
        if context.task_name == "fit":
            if self.only_k is None or estimator.named_steps["knn"].n_neighbors == self.only_k:
                X[:] = X[::-1].copy()

    def teardown(self, estimator, context):
        pass


class _CallbackInstallingPipeline(Pipeline):
    def __init__(self, steps, *, reverse_rows=False):
        super().__init__(steps)
        self.reverse_rows = reverse_rows

    def set_params(self, **params):
        super().set_params(**params)
        if "reverse_rows" in params:
            callbacks = [_ReverseFitRows()] if self.reverse_rows else []
            self.set_callbacks(*callbacks)
        return self

    def fit_transform(self, X, y=None, **params):
        return self.fit(X, y, **params).transform(X)


class _SlicingIdentityPipeline(Pipeline):
    def set_params(self, **params):
        super().set_params(**params)
        if params.get("identity__validate"):
            super().set_params(identity=FunctionTransformer(func=itemgetter(slice(None)), validate=True))
        return self


_Stage = namedtuple("_Stage", "pipeline")


class _ContainerInstallingTransform(FunctionTransformer):
    def set_params(self, **params):
        super().set_params(**params)
        if "validate" in params:
            configuration = Message()
            if self.validate:
                inner = Pipeline([("identity", FunctionTransformer())]).set_callbacks(_ReverseFitRows())
                configuration = OrderedDict(stage=_Stage(inner))
            self.kw_args = {"configuration": configuration}
        return self


class _CollectionInstallingTransform(FunctionTransformer):
    register_callbacks = False

    def set_params(self, **params):
        super().set_params(**params)
        if "kw_args" in params:
            container = self.kw_args["container"]
            inner = Pipeline([("identity", FunctionTransformer())])
            if self.register_callbacks:
                inner.set_callbacks(_ReverseFitRows())
            if container == "ChainMap":
                models = ChainMap({}, {0: inner})
            elif container == "UserDict":
                models = UserDict({0: inner})
            elif container == "array_parameters":
                models = np.zeros(1, dtype=np.float32).view(_NumericArrayConfiguration)
                models.inner = inner
            elif container == "function_parameters":
                def models():
                    pass

                models.get_params = _TransformConfiguration(inner).get_params
            else:
                models = {"deque": deque, "UserList": UserList}[container]([inner])
            self.kw_args = {"models": models}
        return self


class _RetainedTrainingArray(TransformerMixin, BaseEstimator):
    def __init__(self, dtype="float64", order="C", edit="bounds"):
        self.dtype = dtype
        self.order = order
        self.edit = edit

    def fit(self, X, y=None):
        self.training_ = np.array(X, dtype=self.dtype, order=self.order, copy=True)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        if self.edit == "index_labels":
            self.labels_ = y
            return KDTree(self.training_, leaf_size=2, metric="manhattan")
        if self.edit == "named":
            import pandas as pd
            return pd.DataFrame(self.training_, columns=["first", "second"])
        return self.training_

    def transform(self, X):
        if self.edit == "raises":
            warnings.warn("rejecting validation rows", RuntimeWarning)
            raise ValueError("validation projection failed")
        if self.edit == "index_labels":
            self.labels_[:] = self.labels_[::-1]
        query = np.array(X, dtype=self.dtype, order=self.order, copy=True)
        probe = query[:, 0] == 4
        query[query[:, 0] == 6, 0] = 6.9
        query[probe, 0] = 4.125
        if self.edit == "bounds":
            # Move a point out of its fitted node bounds to within .005 of the probe.
            self.training_[4, 0] = 4.13
        if self.edit == "backend_tie":
            self.training_[4, 1] = .125
            # At k=4, the last neighbor can be either x=4 or x=9.
            query[probe, 0] = 6.5
        if self.edit == "named":
            query[probe, 0] = 5.  # Unnamed query with a tied nearest neighbor.
        return query


class _TrainingNeighborIndex(TransformerMixin, BaseEstimator):
    def __init__(self, index_type, query_form="array"):
        self.index_type = index_type
        self.query_form = query_form

    def fit(self, X, y=None):
        coordinates = np.asarray(X, dtype=np.float64)
        if self.index_type in {"neighbors", "brute"}:
            algorithm = "brute" if self.index_type == "brute" else "kd_tree"
            self.index_ = NearestNeighbors(algorithm=algorithm, metric="manhattan").fit(coordinates)
        else:
            tree = {"kd_tree": KDTree, "ball_tree": BallTree}[self.index_type]
            self.index_ = tree(coordinates, metric="manhattan")
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).index_

    def transform(self, X):
        if self.query_form == "nullable":
            import pandas as pd
            return pd.DataFrame(X, columns=["dtype", "second"]).astype("Float64")
        if self.query_form in {"dok", "lil"}:
            from scipy.sparse import csr_matrix
            return csr_matrix(X).asformat(self.query_form)
        if self.query_form == "array_only":
            return _ArrayCoordinates(X)
        if self.query_form == "list":
            return X.tolist()
        if self.query_form == "memoryview":
            return memoryview(X)
        return X


class SharedKNNGridTests(unittest.TestCase):
    def setUp(self):
        random = np.random.RandomState(23)
        self.y = np.arange(80) % 2
        coefficients = random.uniform(0.1, 1.0, (80, 2))
        coefficients[:, 0] += self.y
        basis = np.column_stack([np.eye(2), random.uniform(0.1, 1.0, (2, 10))])
        self.X = (coefficients @ basis + random.uniform(0, 0.005, (80, 12))).astype(np.float32)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def shared(self, estimator, grid, name="shared", **kwargs):
        return calibration_knn_grid(estimator, grid, self.X, self.y,
                                    checkpoint_dir=Path(self.temp.name) / name, seed=19, **kwargs)

    def assert_equivalent(self, estimator, grid, name):
        original_predict = KNeighborsClassifier.predict
        observed = {}

        def capture(classifier, X):
            predictions = original_predict(classifier, X)
            key = (classifier.n_neighbors, classifier._fit_X.shape[1])
            observed.setdefault(key, []).append(predictions.copy())
            return predictions

        with patch.object(KNeighborsClassifier, "predict", capture):
            ordinary = calibration_search(estimator, grid, self.X, self.y, cv=5, seed=19, refit=False).report
        ordinary_predictions = observed
        observed = {}
        original_shared_predict = _KNNQueryReuse.predict

        def capture_shared(reuse, k, **kwargs):
            result = original_shared_predict(reuse, k, **kwargs)
            classifier = reuse.models[k]
            key = (k, classifier._fit_X.shape[1])
            observed.setdefault(key, []).append(result[0].copy())
            return result

        with patch.object(_KNNQueryReuse, "predict", capture_shared):
            shared = self.shared(estimator, grid, name)
        self.assertEqual(set(observed), set(ordinary_predictions))
        for key, folds in ordinary_predictions.items():
            self.assertEqual(len(folds), len(observed[key]))
            for expected, actual in zip(folds, observed[key]):
                np.testing.assert_array_equal(expected, actual)
        self.assertEqual(len(ordinary["candidates"]), len(shared["candidates"]))
        for expected, actual in zip(ordinary["candidates"], shared["candidates"]):
            self.assertEqual(expected["parameters"], actual["parameters"])
            self.assertTrue(expected["valid"], expected)
            self.assertTrue(actual["valid"], actual)
            self.assertEqual(expected["mean_score"], actual["mean_score"])
            self.assertEqual([fold["score"] for fold in expected["folds"]],
                             [fold["score"] for fold in actual["folds"]])
            for expected_fold, actual_fold in zip(expected["folds"], actual["folds"]):
                for field in ("n_iter", "converged", "last_transform_converged",
                              "n_selected_features", "selected_feature_indices"):
                    self.assertEqual(expected_fold["diagnostics"].get(field), actual_fold["diagnostics"].get(field))
        self.assertEqual(shared["selected_parameters"], ordinary["selected_parameters"])
        self.assertFalse(shared["refit_performed"])
        self.assertFalse(shared["defaults_frozen"])
        return shared

    def test_pipeline_index_origin_is_independent_of_feature_count_metadata(self):
        axis = np.array([0., 1., 2., 4., 6., 7., 8., 9.])
        X = np.tile(np.column_stack((axis, np.zeros_like(axis))), (2, 1))
        y = np.tile([1, 0, 0, 1, 1, 1, 0, 0], 2)
        a, b = np.split(np.arange(len(X)), 2)
        folds = [(a, b), (b, a)]
        fit = KNeighborsClassifier.fit
        for name, transform, backend in (
            ("numeric", _RetainedTrainingArray(edit="backend_tie"), "brute"),
            ("index", _TrainingNeighborIndex("neighbors"), "kd_tree"),
        ):
            with self.subTest(name=name):
                observed = []

                def changed_width(head, values, labels):
                    result = fit(head, values, labels)
                    if isinstance(values, (NearestNeighbors, KNeighborsClassifier)):
                        head.n_features_in_ = head._fit_X.shape[1]
                    else:
                        vars(head).pop("n_features_in_", None)
                    if head.n_neighbors == 4:
                        observed.append(head._fit_method)
                    return result

                model = Pipeline([("map", transform), ("knn", KNeighborsClassifier(leaf_size=2, n_jobs=1))])
                with patch.object(KNeighborsClassifier, "fit", changed_width):
                    scores = []
                    for train, valid in folds:
                        fitted = clone(model).set_params(knn__n_neighbors=4).fit(X[train], y[train])
                        scores.append(np.mean(fitted.predict(X[valid]) == y[valid]))
                    self.assertEqual(observed, [backend, backend])
                    observed.clear()
                    report = calibration_knn_grid(model, {"knn__n_neighbors": [1, 4]}, X, y,
                                                  cv=2, folds=folds, threads=1,
                                                  checkpoint_dir=Path(self.temp.name) / name)
                self.assertEqual(observed, [backend, backend])
                candidate = next(row for row in report["candidates"] if row["parameters"]["knn__n_neighbors"] == 4)
                self.assertTrue(candidate["valid"], candidate)
                self.assertEqual([fold["score"] for fold in candidate["folds"]], scores)

    def test_pipeline_copy_during_shared_fit_can_fit_independently(self):
        from copy import copy, deepcopy
        import pickle

        estimator = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())])
        native_fit = Pipeline.fit
        copies_checked = []

        def copy_before_preprocessing(pipeline, X, y, **params):
            # A monitoring callback can snapshot the operation-owned estimator.
            snapshots = [copy(pipeline), deepcopy(pipeline), pickle.loads(pickle.dumps(pipeline))]
            # A shallow copy normally shares estimators. Give it independent steps
            # before fitting it so its result tests method binding alone.
            snapshots[0].steps = clone(pipeline).steps
            expected = native_fit(clone(pipeline), X, y, **params).predict(X)
            result = native_fit(pipeline, X, y, **params)
            for snapshot in snapshots:
                native_fit(snapshot, X, y, **params)
                np.testing.assert_array_equal(snapshot.predict(X), expected)
                self.assertIs(snapshot._fit.__self__, snapshot)
                copies_checked.append(True)
            return result

        with patch.object(Pipeline, "fit", copy_before_preprocessing):
            report = calibration_knn_grid(estimator, {"knn__n_neighbors": [1, 3]}, self.X, self.y,
                                          cv=2, seed=11, threads=1,
                                          checkpoint_dir=Path(self.temp.name) / "snapshots")
        self.assertEqual(len(copies_checked), 6)
        self.assertTrue(all(row["valid"] for row in report["candidates"]))

    @unittest.skipUnless(hasattr(Pipeline, "set_callbacks"), "requires sklearn Pipeline callbacks")
    def test_pipeline_terminal_callback_refit_and_replacement(self):
        from supervised_nmf.study import _KNNTrainingStateCompatibilityError

        class RefitOnIndex:
            def __init__(self, mode):
                self.mode = mode

            def setup(self, estimator, context):
                if self.mode == "one_shot_labels":
                    estimator.set_callbacks()

            def on_fit_task_begin(self, estimator, context):
                pass

            def on_fit_task_end(self, estimator, context, *, X, y):
                head = estimator.named_steps["knn"]
                if context.task_name != "fit" or (self.mode == "k1_only" and head.n_neighbors != 1):
                    return
                if self.mode == "one_shot_labels":
                    # This fit's callback context survives unregistration in setup.
                    y[:] = 1 - y
                    return
                index = NearestNeighbors(algorithm="kd_tree", leaf_size=2).fit(X)
                if self.mode == "replace":
                    estimator.set_params(knn=clone(head).fit(index, y))
                else:
                    head.fit(index, y)

            def teardown(self, estimator, context):
                pass

        axis = np.array([0., 1., 2., 4., 6., 7., 8., 9.])
        X = np.tile(np.column_stack((axis, np.zeros_like(axis))), (2, 1))
        y = np.tile([1, 0, 0, 1, 1, 1, 0, 0], 2)
        left, right = np.split(np.arange(len(y)), 2)
        options = dict(cv=2, folds=[(left, right), (right, left)], threads=1)
        grid = {"knn__n_neighbors": [1, 4]}
        scores_at_four = {}
        for mode in ("replace", "refit", "k1_only", "one_shot_labels"):
            with self.subTest(mode=mode):
                model = Pipeline([("map", _RetainedTrainingArray(edit="backend_tie")),
                                  ("knn", KNeighborsClassifier(n_jobs=1, leaf_size=2))]).set_callbacks(RefitOnIndex(mode))
                native = calibration_search(model, grid, X, y, refit=False, **options).report
                self.assertTrue(all(row["valid"] for row in native["candidates"]))
                four = next(row for row in native["candidates"] if row["parameters"]["knn__n_neighbors"] == 4)
                scores_at_four[mode] = [fold["score"] for fold in four["folds"]]
                directory = Path(self.temp.name) / mode
                with self.assertRaises(_KNNTrainingStateCompatibilityError):
                    calibration_knn_grid(model, grid, X, y, checkpoint_dir=directory, **options)
                self.assertFalse(directory.exists())
        # At k=4, the ordinary numeric fit uses brute force; the unconditional refit adopts the KDTree.
        self.assertNotEqual(scores_at_four["k1_only"], scores_at_four["refit"])

    @unittest.skipUnless(hasattr(Pipeline, "set_callbacks"), "requires sklearn Pipeline callbacks")
    def test_callback_editing_float32_fit_rows_uses_independent_trees(self):
        from supervised_nmf.study import _KNNTrainingStateCompatibilityError

        X = np.tile(np.arange(4, dtype=np.float32), 2).reshape(-1, 1)
        y = np.tile([0, 0, 1, 1], 2)
        left, right = np.split(np.arange(len(y)), 2)
        options = dict(cv=2, folds=[(left, right), (right, left)], threads=1)
        grid = {"knn__n_neighbors": [1, 3]}
        for only_k in (None, 3):
            with self.subTest(only_k=only_k):
                model = Pipeline([("knn", KNeighborsClassifier(algorithm="kd_tree", n_jobs=1))])
                model.set_callbacks(_ReverseFitRows(only_k))
                for k in grid["knn__n_neighbors"]:
                    train = X[left]
                    fitted = clone(model).set_params(knn__n_neighbors=k).fit(train, y[left])
                    edited = only_k is None or only_k == k
                    np.testing.assert_array_equal(train, X[left][::-1] if edited else X[left])
                    np.testing.assert_array_equal(fitted.predict(X[right]), y[right])
                ordinary = calibration_search(model, grid, X, y, refit=False, checkpoint_path=None, **options).report
                self.assertFalse(ordinary["refit_performed"])
                self.assertEqual([[fold["score"] for fold in row["folds"]] for row in ordinary["candidates"]],
                                 [[1., 1.], [1., 1.]])
                directory = Path(self.temp.name) / f"cold-{only_k}"
                with self.assertRaises(_KNNTrainingStateCompatibilityError) as caught:
                    calibration_knn_grid(model, grid, X, y, checkpoint_dir=directory, **options)
                for text in ("calibration_search", "refit=False", "checkpoint_path=None"):
                    self.assertIn(text, str(caught.exception))
                self.assertFalse(directory.exists())

    @unittest.skipUnless(hasattr(Pipeline, "set_callbacks"), "requires sklearn Pipeline callbacks")
    def test_callback_refusal_preserves_complete_and_partial_checkpoints(self):
        from supervised_nmf.study import _KNNTrainingStateCompatibilityError

        model = Pipeline([("knn", KNeighborsClassifier())])
        grid = {"knn__n_neighbors": [1, 3]}
        directory = Path(self.temp.name) / "warm"
        first = self.shared(model, grid, "warm", cv=2)
        self.assertTrue(all(row["valid"] for row in first["candidates"]))
        model.set_callbacks(_ReverseFitRows(3))
        model.set_callbacks()
        cleared = self.shared(model, grid, "cleared", cv=2)
        self.assertEqual([row["mean_score"] for row in cleared["candidates"]],
                         [row["mean_score"] for row in first["candidates"]])

        def contents():
            return {p.relative_to(directory): (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None
                    for p in directory.rglob("*")}

        model.set_callbacks(_ReverseFitRows(3))
        for state in ("complete", "partial"):
            with self.subTest(state=state):
                if state == "partial":
                    (directory / "report.json").unlink()
                    next((directory / "folds").glob("*.json")).unlink()
                before = contents()
                with patch("supervised_nmf.study._checkpoint_lock", side_effect=AssertionError("checkpoint accessed")), \
                        self.assertRaises(_KNNTrainingStateCompatibilityError):
                    self.shared(model, grid, "warm", cv=2)
                self.assertEqual(contents(), before)

    @unittest.skipUnless(hasattr(Pipeline, "set_callbacks"), "requires sklearn Pipeline callbacks")
    def test_callback_guard_checks_effective_nested_grid_steps(self):
        from sklearn.compose import ColumnTransformer
        from supervised_nmf.study import _KNNTrainingStateCompatibilityError

        clean = Pipeline([("scale", StandardScaler())])
        registered = clone(clean).set_callbacks(_ReverseFitRows())
        grid = {"knn__n_neighbors": [1, 3]}
        for name, inner, extra_grid in (("fixed", registered, {}),
                                        ("replacement", clean, {"prep__scaled": [clean, registered]}),
                                        ("displaced", registered, {"prep__scaled": [clean]})):
            with self.subTest(name=name):
                prep = ColumnTransformer([("scaled", inner, [0])])
                model = Pipeline([("prep", prep), ("knn", KNeighborsClassifier())])
                if name == "displaced":
                    report = self.shared(model, {**grid, **extra_grid}, name, cv=2)
                    self.assertTrue(all(row["valid"] for row in report["candidates"]))
                else:
                    with self.assertRaises(_KNNTrainingStateCompatibilityError):
                        self.shared(model, {**grid, **extra_grid}, name, cv=2)
                    self.assertFalse((Path(self.temp.name) / name).exists())

    @unittest.skipUnless(hasattr(Pipeline, "set_callbacks"), "requires sklearn Pipeline callbacks")
    def test_callback_in_partial_cannot_reuse_clean_checkpoint(self):
        from functools import partial
        from supervised_nmf.study import _KNNTrainingStateCompatibilityError

        def transform(X, *, inner):
            values = X.copy()
            inner.fit(values)
            return values

        def weights(distances, *, inner):
            values = np.ones_like(distances)
            inner.fit(values)
            return values

        inner = Pipeline([("identity", FunctionTransformer())])
        transform_step = FunctionTransformer(partial(transform, inner=inner))
        weight_fn = partial(weights, inner=inner)
        for label, model, parameter in (
            ("pipeline", Pipeline([("map", transform_step), ("knn", KNeighborsClassifier())]), "knn__n_neighbors"),
            ("raw", KNeighborsClassifier(weights=weight_fn), "n_neighbors"),
            ("nmf", NMFClassifier(weights=weight_fn, screening="percentile", percentile=100,
                                  random_state=7), "n_neighbors"),
        ):
            with self.subTest(layout=label):
                inner.set_callbacks()
                grid = {parameter: [1, 3]}
                report = self.shared(model, grid, label, cv=2)
                self.assertTrue(all(row["valid"] for row in report["candidates"]))
                inner.set_callbacks(_ReverseFitRows())
                with self.assertRaises(_KNNTrainingStateCompatibilityError):
                    self.shared(model, grid, label, cv=2)

    @unittest.skipUnless(hasattr(Pipeline, "set_callbacks"), "requires sklearn Pipeline callbacks")
    def test_callback_in_parameter_holder_cannot_reuse_clean_scores(self):
        from supervised_nmf.study import _KNNTrainingStateCompatibilityError

        def transform(X, *, configuration):
            values = X.copy()
            configuration.inner.fit(values)
            return values

        inner = Pipeline([("identity", FunctionTransformer())])
        prep = FunctionTransformer(transform, kw_args={"configuration": _TransformConfiguration(inner)})
        model = Pipeline([("prep", prep), ("knn", KNeighborsClassifier())])
        # Reversing rows swaps classes in the first half and preserves them in the second.
        X = np.array([0, 1, 2, 3, 0, 2, 3, 1], dtype=np.float32).reshape(-1, 1)
        y = (X[:, 0] >= 2).astype(int)
        left, right = np.split(np.arange(len(y)), 2)
        options = dict(cv=2, folds=[(left, right), (right, left)], threads=1)
        grid = {"knn__n_neighbors": [1]}
        directory = Path(self.temp.name) / "holder"
        clean = calibration_knn_grid(model, grid, X, y, checkpoint_dir=directory, **options)
        self.assertEqual([fold["score"] for fold in clean["candidates"][0]["folds"]], [1., 1.])
        inner.set_callbacks(_ReverseFitRows())
        native = calibration_search(model, grid, X, y, refit=False, checkpoint_path=None, **options).report
        self.assertEqual([fold["score"] for fold in native["candidates"][0]["folds"]], [0., 0.])
        with patch("supervised_nmf.study._checkpoint_lock", side_effect=AssertionError("checkpoint accessed")), \
                self.assertRaises(_KNNTrainingStateCompatibilityError):
            calibration_knn_grid(model, grid, X, y, checkpoint_dir=directory, **options)

    @unittest.skipUnless(hasattr(Pipeline, "set_callbacks"), "requires sklearn Pipeline callbacks")
    def test_boolean_grid_setter_installs_callback_on_clean_nested_pipeline(self):
        from supervised_nmf.study import _KNNTrainingStateCompatibilityError

        nested = _CallbackInstallingPipeline([("scale", StandardScaler())])
        model = Pipeline([("prep", nested), ("knn", KNeighborsClassifier(n_neighbors=1))])
        X = np.tile(np.arange(4, dtype=np.float32), 2).reshape(-1, 1)
        y = np.tile([0, 0, 1, 1], 2)
        left, right = np.split(np.arange(len(y)), 2)
        options = dict(cv=2, folds=[(left, right), (right, left)], threads=1)
        grid = {"prep__reverse_rows": [False, True]}
        ordinary = calibration_search(model, grid, X, y, refit=False, checkpoint_path=None, **options).report
        scores = {row["parameters"]["prep__reverse_rows"]: [fold["score"] for fold in row["folds"]]
                  for row in ordinary["candidates"]}
        self.assertEqual(scores, {False: [1., 1.], True: [0., 0.]})

        fresh, warm = (Path(self.temp.name) / name for name in ("fresh", "warm"))
        clean = calibration_knn_grid(model, {"prep__reverse_rows": [False]}, X, y, checkpoint_dir=warm, **options)
        self.assertEqual(clean["candidates"][0]["mean_score"], 1.)
        saved = (warm / "report.json").read_bytes()
        for directory in (fresh, warm):
            with (
                self.subTest(directory=directory.name),
                patch.object(_CallbackInstallingPipeline, "fit", side_effect=AssertionError("fit started")),
                patch("supervised_nmf.study._checkpoint_lock", side_effect=AssertionError("checkpoint accessed")),
                self.assertRaises(_KNNTrainingStateCompatibilityError),
            ):
                calibration_knn_grid(model, grid, X, y, checkpoint_dir=directory, **options)
        self.assertFalse(fresh.exists())
        self.assertEqual((warm / "report.json").read_bytes(), saved)

    def test_setter_installed_slicing_transform_can_score_and_resume(self):
        prep = _SlicingIdentityPipeline([("identity", FunctionTransformer())])
        tail = _CallbackInstallingPipeline([("scale", StandardScaler())])
        model = Pipeline([("prep", prep), ("tail", tail), ("knn", KNeighborsClassifier())])
        grid = {"prep__identity__validate": [True], "knn__n_neighbors": [1, 3]}
        configured = clone(prep).set_params(identity__validate=True)
        self.assertIsInstance(configured.named_steps["identity"].func, itemgetter)
        self.assert_native_scores_and_resume(model, grid, "slicing")

        if hasattr(Pipeline, "set_callbacks"):
            from supervised_nmf.study import _KNNTrainingStateCompatibilityError

            # The callback scan must continue past the slicing function to its sibling.
            with self.assertRaises(_KNNTrainingStateCompatibilityError):
                self.shared(model, {**grid, "tail__reverse_rows": [True]}, "slicing", cv=2)

    def assert_native_scores_and_resume(self, model, grid, name):
        ordinary = calibration_search(model, grid, self.X, self.y, cv=2, seed=19,
                                      refit=False, checkpoint_path=None).report
        first = self.shared(model, grid, name, cv=2)
        for actual, expected in zip(first["candidates"], ordinary["candidates"], strict=True):
            self.assertTrue(actual["valid"], actual)
            self.assertTrue(expected["valid"], expected)
            self.assertEqual(actual["parameters"], expected["parameters"])
            self.assertEqual([fold["score"] for fold in actual["folds"]],
                             [fold["score"] for fold in expected["folds"]])
        resumed = self.shared(model, grid, name, cv=2)
        self.assertEqual(resumed["new_representation_fits"], 0)
        self.assertEqual(resumed["reused_representation_folds"], 2)
        self.assertEqual(resumed["candidates"], first["candidates"])

    def test_setter_installed_standard_containers_and_message(self):
        def transform(X, *, configuration):
            stage = configuration.get("stage")
            if stage is None:
                return X
            values = X.copy()
            stage.pipeline.fit(values)
            return values

        prep = _ContainerInstallingTransform(transform)
        model = Pipeline([("prep", prep), ("knn", KNeighborsClassifier())])
        grid = {"prep__validate": [False], "knn__n_neighbors": [1, 3]}
        self.assert_native_scores_and_resume(model, grid, "containers")
        if hasattr(Pipeline, "set_callbacks"):
            from supervised_nmf.study import _KNNTrainingStateCompatibilityError

            active = clone(prep).set_params(validate=True)
            np.testing.assert_array_equal(active.fit_transform(self.X[:4]), self.X[:4][::-1])
            with self.assertRaises(_KNNTrainingStateCompatibilityError):
                self.shared(model, {**grid, "prep__validate": [True]}, "containers", cv=2)

    def test_setter_installed_collections_cannot_reuse_clean_scores_for_callbacks(self):
        from supervised_nmf.study import _KNNTrainingStateCompatibilityError

        def transform(X, *, models):
            values = X.copy()
            get_params = getattr(models, "get_params", None)
            inner = models[0] if get_params is None else get_params(deep=False)["inner"]
            inner.fit(values)
            return values

        model = Pipeline([("prep", _CollectionInstallingTransform(transform)), ("knn", KNeighborsClassifier())])
        X = np.array([0, 1, 2, 3, 0, 2, 3, 1], dtype=np.float32).reshape(-1, 1)
        y = (X[:, 0] >= 2).astype(int)
        left, right = np.split(np.arange(len(y)), 2)
        options = dict(cv=2, folds=[(left, right), (right, left)], threads=1)
        for container in ("ChainMap", "UserDict", "deque", "UserList", "array_parameters", "function_parameters"):
            with self.subTest(container=container):
                grid = {"prep__kw_args": [{"container": container}], "knn__n_neighbors": [1]}
                directory = Path(self.temp.name) / container
                ordinary = calibration_search(model, grid, X, y, refit=False, checkpoint_path=None, **options)
                native_scores = [fold["score"] for fold in ordinary.report["candidates"][0]["folds"]]
                self.assertEqual(native_scores, [1., 1.])
                for new, reused in ((2, 0), (0, 2)):
                    clean = calibration_knn_grid(model, grid, X, y, checkpoint_dir=directory, **options)
                    self.assertEqual([fold["score"] for fold in clean["candidates"][0]["folds"]], native_scores)
                    self.assertEqual((clean["new_representation_fits"], clean["reused_representation_folds"]),
                                     (new, reused))
                if hasattr(Pipeline, "set_callbacks"):
                    # The public declarations stay the same when the setter registers the callback.
                    with patch.object(_CollectionInstallingTransform, "register_callbacks", True):
                        ordinary = calibration_search(model, grid, X, y, refit=False, checkpoint_path=None, **options)
                        self.assertEqual([fold["score"] for fold in ordinary.report["candidates"][0]["folds"]],
                                         [0., 0.])
                        with (
                            patch("supervised_nmf.study._checkpoint_lock", side_effect=AssertionError("checkpoint accessed")),
                            self.assertRaisesRegex(_KNNTrainingStateCompatibilityError, "registered scikit-learn fit callbacks"),
                        ):
                            calibration_knn_grid(model, grid, X, y, checkpoint_dir=directory, **options)

    def test_fit_transform_only_step_before_knn(self):
        inner = Pipeline([("scale", StandardScaler())])
        model = Pipeline([("prep", _FitTransformOnly(inner)), ("knn", KNeighborsClassifier())])
        grid = {"knn__n_neighbors": [1, 3]}
        self.assert_native_scores_and_resume(model, grid, "fit-transform-only")
        if hasattr(Pipeline, "set_callbacks"):
            from supervised_nmf.study import _KNNTrainingStateCompatibilityError

            inner.set_callbacks(_ReverseFitRows())
            with self.assertRaises(_KNNTrainingStateCompatibilityError):
                self.shared(model, grid, "fit-transform-only", cv=2)

    def test_cd_float32_matches_ordinary_grid_and_retains_coordinates(self):
        estimator = NMFClassifier(solver="cd", screening="percentile", percentile=100,
                                  n_components=2, random_state=7, max_iter=3000, tol=1e-3)
        result = self.assert_equivalent(estimator, {"n_neighbors": [1, 3, 5]}, "cd")
        self.assertEqual(result["unique_representation_fits"], 5)
        self.assertEqual(result["new_representation_fits"], 5)
        self.assertTrue(all(fold["training_representation_dtype"] == "float32" for fold in result["representation_folds"]))
        self.assertEqual(result["tuning_seconds"], sum(result["compute_seconds"].values()))

    def test_spa_matches_ordinary_grid(self):
        estimator = NMFClassifier(solver="spa", screening="percentile", percentile=100,
                                  n_components=2, random_state=7, weights="distance", max_iter=1000)
        result = self.assert_equivalent(estimator, {"n_neighbors": [1, 3]}, "spa")
        self.assertEqual(result["unique_representation_fits"], 5)

    def test_mu_float32_matches_ordinary_grid_and_retains_coordinates(self):
        estimator = NMFClassifier(solver="mu", screening="absolute", score_threshold=1.0,
                                  n_components=2, random_state=7, max_iter=3000, tol=1e-3)
        result = self.assert_equivalent(estimator, {"n_neighbors": [1, 3, 5]}, "mu")
        self.assertEqual(result["unique_representation_fits"], 5)
        self.assertEqual(result["new_representation_fits"], 5)
        self.assertTrue(all(fold["training_representation_dtype"] == "float32"
                            for fold in result["representation_folds"]))
        self.assertEqual(result["tuning_seconds"], sum(result["compute_seconds"].values()))

    def test_pca_pipeline_uses_original_fit_transform_coordinates(self):
        estimator = Pipeline([("screen", BinaryScreening(screening="percentile", percentile=100)),
                              ("pca", PCA(n_components=2, svd_solver="full", random_state=7, whiten=True)),
                              ("knn", KNeighborsClassifier(weights="distance"))])
        grid = {"pca__n_components": [2, 3], "knn__n_neighbors": [1, 3]}
        result = self.assert_equivalent(estimator, grid, "pca")
        self.assertEqual(result["unique_representation_fits"], 10)
        original_transform = PCA.transform
        transformed_shapes = []

        def observe(instance, X):
            transformed_shapes.append(X.shape)
            return original_transform(instance, X)

        with patch.object(PCA, "transform", observe):
            self.shared(estimator, {"knn__n_neighbors": [1, 3, 5]}, "pca-count")
        self.assertEqual(len(transformed_shapes), 5)
        self.assertTrue(all(shape[0] == 16 for shape in transformed_shapes))

    def test_screening_only_and_raw_knn_match_ordinary_grid(self):
        estimator = Pipeline([("screen", BinaryScreening(screening="percentile", percentile=50)),
                              ("knn", KNeighborsClassifier())])
        self.assert_equivalent(estimator, {"knn__n_neighbors": [1, 3, 5]}, "screen")
        self.assert_equivalent(KNeighborsClassifier(weights="distance"), {"n_neighbors": [1, 3, 5]}, "raw")

    def test_duplicate_points_and_distance_ties_preserve_each_prediction(self):
        self.X = np.repeat(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32), 20, axis=0)
        self.y = np.tile([0, 1], 40)
        self.assert_equivalent(KNeighborsClassifier(weights="distance"), {"n_neighbors": [1, 3, 40]}, "ties")

    def test_sd_knn_full_grid_matches_ordinary_and_scales_each_fold_once(self):
        self.y = np.where(self.y, "present", "absent")
        estimator = NMFClassifier(solver="spa", screening="none", n_components=2,
                                  random_state=7, knn_normalization="sd", weights="distance")
        self.assert_equivalent(estimator, {"n_neighbors": [1, 3, 5]}, "sd-reference")
        fit_rows, transform_rows = [], []
        original_fit, original_transform = StandardScaler.fit, StandardScaler.transform

        def observe_fit(scaler, X, y=None, **kwargs):
            fit_rows.append(np.array(X, copy=True))
            return original_fit(scaler, X, y, **kwargs)

        def observe_transform(scaler, X, **kwargs):
            transform_rows.append(np.array(X, copy=True))
            return original_transform(scaler, X, **kwargs)

        with patch.object(StandardScaler, "fit", observe_fit), \
                patch.object(StandardScaler, "transform", observe_transform):
            report = self.shared(estimator, {"n_neighbors": [1, 3, 5]}, "sd-count")
        self.assertEqual(len(fit_rows), 5)
        # One fit_transform of training coordinates and one validation transform
        # per representation fold; candidate KNN fits do not rescale again.
        self.assertEqual([len(X) for X in transform_rows], [64, 16] * 5)
        for rows, fold in zip(fit_rows, report["identity"]["folds"]):
            training = fold["training_indices"]
            reference = estimator.fit(self.X[training], self.y[training])
            np.testing.assert_array_equal(rows, reference.training_coordinates_)

    def test_sd_explicit_threshold_class_order_matches_ordinary_grid(self):
        self.y = np.where(self.y, "present", "absent")
        for threshold in (.37, .5, 0., 1.):
            estimator = NMFClassifier(solver="spa", screening="none", n_components=2,
                                      random_state=7, knn_normalization="sd", weights="distance",
                                      decision_threshold=threshold, positive_label="absent")
            grid = {"n_neighbors": [1, 3, 5, 64]}
            ordinary = calibration_search(estimator, grid, self.X, self.y,
                                           cv=5, seed=19, refit=False).report
            shared = self.shared(estimator, grid, f"sd-threshold-{threshold}")
            self.assertEqual(shared["selected_parameters"], ordinary["selected_parameters"])
            for expected, actual in zip(ordinary["candidates"], shared["candidates"]):
                self.assertTrue(actual["valid"])
                self.assertEqual([row["score"] for row in expected["folds"]],
                                 [row["score"] for row in actual["folds"]])

    def test_shared_costs_are_counted_once_and_resume_never_queries(self):
        estimator = KNeighborsClassifier(algorithm="brute")
        grid = {"n_neighbors": [1, 3, 5, 200]}
        report = self.shared(estimator, grid, "cost-accounting")
        costs = report["compute_seconds"]
        self.assertEqual(report["schema_version"], 4)
        self.assertEqual(costs["knn_shared_query_seconds"],
                         sum(row["neighbor_query_seconds"] for row in report["representation_folds"]))
        self.assertEqual(costs["knn_reuse_setup_seconds"],
                         sum(row["neighbor_reuse_setup_seconds"] for row in report["representation_folds"]))
        self.assertEqual(costs["knn_prediction_seconds"],
                         sum(head["prediction_seconds"] for row in report["representation_folds"]
                             for head in row["neighbors"].values()))
        self.assertEqual(report["tuning_seconds"], sum(costs.values()))
        self.assertTrue(all(len([query for query in fold["neighbor_queries"] if query["queried"]]) == 1
                            for fold in report["representation_folds"]))
        self.assertTrue(all(fold["neighbors"]["200"]["status"] == "failed"
                            for fold in report["representation_folds"]))
        source = report["identity"]["runtime"]["package_source"]
        self.assertTrue(any(name.endswith("_neighbors.py") for name in source))
        with patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("resume must not fit")), \
                patch.object(_KNNQueryReuse, "predict", side_effect=AssertionError("resume must not query")):
            resumed = self.shared(estimator, grid, "cost-accounting")
        self.assertEqual(resumed["compute_seconds"], costs)
        self.assertEqual(resumed["new_representation_fits"], 0)

    def test_shared_query_warning_invalidates_affected_heads(self):
        estimator = KNeighborsClassifier(algorithm="brute", weights="distance")
        def warn_shared(count, result):
            self.assertEqual(count, 4)
            warnings.warn("shared neighbor query incomplete", ConvergenceWarning)

        with native_queries(on_shared=warn_shared):
            report = self.shared(estimator, {"n_neighbors": [1, 3]}, "query-warning")
        self.assertTrue(all(not row["valid"] for row in report["candidates"]))
        self.assertTrue(all(head["status"] == "nonconverged"
                            for fold in report["representation_folds"]
                            for head in fold["neighbors"].values()))
        self.assertTrue(all(fold["neighbor_queries"][0]["warnings"]
                            for fold in report["representation_folds"]))

    def test_screening_is_fit_on_each_training_fold_once_without_refit(self):
        estimator = Pipeline([("screen", BinaryScreening(screening="percentile", percentile=50)),
                              ("knn", KNeighborsClassifier())])
        observed = []
        original_fit = BinaryScreening.fit

        def observe(instance, X, y):
            observed.append((X.copy(), y.copy()))
            return original_fit(instance, X, y)

        with patch.object(BinaryScreening, "fit", observe):
            result = self.shared(estimator, {"knn__n_neighbors": [1, 3, 5]})
        self.assertEqual(len(observed), 5)
        for (X, y), fold in zip(observed, result["identity"]["folds"]):
            np.testing.assert_array_equal(X, self.X[fold["training_indices"]])
            np.testing.assert_array_equal(y, self.y[fold["training_indices"]])
            self.assertFalse(set(fold["training_indices"]) & set(fold["validation_indices"]))

    def test_representation_failures_invalidate_every_corresponding_neighbor_candidate(self):
        estimator = NMFClassifier(solver="cd", screening="absolute", score_threshold=1e20,
                                  n_components=2, random_state=7)
        result = self.shared(estimator, {"n_neighbors": [1, 3]})
        for candidate in result["candidates"]:
            self.assertFalse(candidate["valid"])
            self.assertEqual(candidate["invalid_fold_count"], 5)
            self.assertIsNone(candidate["mean_score"])
        self.assertIsNone(result["selected_parameters"])
        self.assertFalse(result["is_fallback"])

    def test_nonconvergence_and_impossible_k_are_never_success_averaged(self):
        estimator = NMFClassifier(solver="cd", screening="none", n_components=2,
                                  random_state=7, max_iter=1, tol=1e-10)
        result = self.shared(estimator, {"n_neighbors": [1, 3]}, "nonconverged")
        self.assertTrue(all(not row["valid"] for row in result["candidates"]))
        self.assertTrue(any(row["warnings"] for row in result["representation_folds"]))
        raw = self.shared(KNeighborsClassifier(), {"n_neighbors": [1, 1000]}, "large-k")
        self.assertTrue(raw["candidates"][0]["valid"])
        self.assertFalse(raw["candidates"][1]["valid"])
        self.assertIsNone(raw["candidates"][1]["mean_score"])

    def test_projection_warning_invalidates_all_neighbors_for_that_representation(self):
        estimator = NMFClassifier(solver="spa", screening="none", n_components=2, random_state=7)
        original_transform = NMFClassifier.transform

        def warn_on_projection(instance, X):
            coordinates = original_transform(instance, X)
            warnings.warn("test projection nonconvergence", ConvergenceWarning)
            return coordinates

        with patch.object(NMFClassifier, "transform", warn_on_projection):
            result = self.shared(estimator, {"n_neighbors": [1, 3]})
        self.assertTrue(all(not row["valid"] for row in result["candidates"]))
        self.assertTrue(all(row["status"] == "nonconverged" for row in result["representation_folds"]))
        self.assertTrue(all(not row["neighbors"] for row in result["representation_folds"]))

    def test_pipeline_nonconvergence_supersedes_oversized_neighbor_error(self):
        model = Pipeline([("nmf", NMF(n_components=2, init="random", random_state=7, max_iter=1)),
                          ("knn", KNeighborsClassifier())])
        oversized = len(self.y) + 1
        grid = {"knn__n_neighbors": [1, oversized]}
        report = self.shared(model, grid, cv=2)
        self.assertIsNone(report["selected_parameters"])
        for record in report["representation_folds"]:
            self.assertEqual(record["status"], "nonconverged")
            self.assertIn("ConvergenceWarning", [w["category"] for w in record["warnings"]])
            self.assertEqual(record["representation_shape"], [len(self.y) // 2, 2])
            self.assertEqual(record["neighbor_queries"], [])
            self.assertGreater(record["neighbors"]["1"]["fit_seconds"], 0.)
            self.assertEqual(record["neighbors"][str(oversized)]["fit_seconds"], 0.)
            for candidate in report["candidates"]:
                k = candidate["parameters"]["knn__n_neighbors"]
                head = record["neighbors"][str(k)]
                fold = candidate["folds"][record["fold"]]
                for receipt in (head, fold):
                    self.assertEqual(receipt["status"], "nonconverged")
                    self.assertTrue({"error", "error_type", "failure_stage", "score"}.isdisjoint(receipt),
                                    receipt)
                    self.assertEqual(receipt["prediction_seconds"], 0.)
                self.assertEqual(fold["fit_seconds"], head["fit_seconds"])
                self.assertEqual(fold["warnings"], record["warnings"])
        resumed = self.shared(model, grid, cv=2)
        self.assertEqual(resumed["reused_representation_folds"], 2)
        self.assertEqual(resumed["new_representation_fits"], 0)
        for field in ("representation_folds", "candidates", "compute_seconds"):
            self.assertEqual(resumed[field], report[field])

    def test_resume_reuses_folds_and_preserves_shared_compute_totals(self):
        estimator = Pipeline([("screen", BinaryScreening(screening="percentile", percentile=50)),
                              ("knn", KNeighborsClassifier())])
        first = self.shared(estimator, {"knn__n_neighbors": [1, 3]})
        with patch.object(BinaryScreening, "fit", side_effect=AssertionError("Completed fold should not refit")):
            resumed = self.shared(estimator, {"knn__n_neighbors": [1, 3]})
        self.assertEqual(resumed["new_representation_fits"], 0)
        self.assertEqual(resumed["reused_representation_folds"], 5)
        self.assertEqual(first["compute_seconds"], resumed["compute_seconds"])
        self.assertEqual(first["candidates"], resumed["candidates"])

    def test_incompatible_grid_data_and_corrupt_receipts_are_rejected(self):
        estimator = KNeighborsClassifier()
        self.shared(estimator, {"n_neighbors": [1, 3]})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.shared(estimator, {"n_neighbors": [1, 5]})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            calibration_knn_grid(estimator, {"n_neighbors": [1, 3]}, self.X + 1, self.y,
                                 seed=19, checkpoint_dir=Path(self.temp.name) / "shared")
        path = next((Path(self.temp.name) / "shared" / "folds").glob("*.json"))
        envelope = json.loads(path.read_text())
        envelope["payload"]["neighbors"]["1"]["score"] = 0.123
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.shared(estimator, {"n_neighbors": [1, 3]})

    def test_interrupted_fold_keeps_completed_folds_and_failed_attempt(self):
        estimator = Pipeline([("screen", BinaryScreening(screening="percentile", percentile=50)),
                              ("knn", KNeighborsClassifier())])
        original_fit = BinaryScreening.fit
        calls = []

        def interrupt_third(instance, X, y):
            calls.append(X.shape[0])
            if len(calls) == 3:
                raise KeyboardInterrupt("test interruption")
            return original_fit(instance, X, y)

        with patch.object(BinaryScreening, "fit", interrupt_third), self.assertRaises(KeyboardInterrupt):
            self.shared(estimator, {"knn__n_neighbors": [1, 3]})
        attempts = [json.loads(path.read_text())["payload"]
                    for path in (Path(self.temp.name) / "shared" / "attempts").glob("*.json")]
        self.assertEqual(sum(row["state"] == "interrupted" for row in attempts), 1)
        result = self.shared(estimator, {"knn__n_neighbors": [1, 3]})
        self.assertEqual(result["new_representation_fits"], 3)
        self.assertEqual(result["reused_representation_folds"], 2)
        self.assertEqual(result["unique_representation_fits"], 5)

    def test_unsupported_classifier_knobs_and_unseeded_representation_are_rejected(self):
        for estimator, grid, pattern in (
            (DummyClassifier(), {}, "Use NMFClassifier"),
            (KNeighborsClassifier(), {"weights": ["uniform", "distance"]}, "Only n_neighbors"),
            (Pipeline([("knn", KNeighborsClassifier())]), {"knn__metric": ["euclidean"]}, "Only n_neighbors"),
            (NMFClassifier(solver="cd", screening="none"), {"n_neighbors": [1, 3]}, "fixed integer seed"),
        ):
            with self.subTest(estimator=type(estimator).__name__), self.assertRaisesRegex((ValueError, TypeError), pattern):
                self.shared(estimator, grid)

    def test_index_pipeline_reports_numeric_dtype_and_native_feature_warnings(self):
        try:
            import pandas
        except ImportError:
            self.skipTest("pandas is required for named nullable columns")

        X = np.arange(24, dtype=float).reshape(12, 2)
        y = np.arange(12) % 2
        a, b = np.split(np.arange(len(y)), 2)
        folds = [(a, b), (b, a)]
        model = Pipeline([("index", _TrainingNeighborIndex("neighbors", "nullable")),
                          ("knn", KNeighborsClassifier())])
        report = calibration_knn_grid(model, {"knn__n_neighbors": [1, 3]}, X, y, cv=2,
                                      folds=folds, checkpoint_dir=Path(self.temp.name) / "named-index")
        for record in report["representation_folds"]:
            self.assertEqual(record["validation_representation_dtype"], "float64")
        for candidate in report["candidates"]:
            for fold, (train, valid) in zip(candidate["folds"], folds):
                fitted = clone(model).set_params(**candidate["parameters"]).fit(X[train], y[train])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    prediction = fitted.predict(X[valid])
                self.assertEqual(fold["status"], "valid")
                self.assertEqual(fold["score"], np.mean(prediction == y[valid]))
                self.assertEqual(fold["warnings"], [
                    {"category": w.category.__name__, "message": str(w.message)} for w in caught])

    def test_pipeline_candidates_match_fits_before_validation_edits_training_array(self):
        axis = np.array([0., 1., 2., 4., 6., 7., 8., 9.])
        X = np.tile(np.column_stack((axis, np.zeros_like(axis))), (2, 1))
        a, b = np.split(np.arange(len(X)), 2)
        folds = [(a, b), (b, a)]
        for dtype, order, edit, algorithm, ks in (
                ("float64", "C", "bounds", "kd_tree", [1]),
                ("float64", "C", "bounds", "kd_tree", [1, 3]),
                ("float32", "C", "bounds", "kd_tree", [1, 3]),
                ("float64", "F", "bounds", "kd_tree", [1, 3]),
                ("float64", "C", "unchanged", "kd_tree", [1, 3]),
                ("float64", "C", "backend_tie", "auto", [1, 4]),
                ("float64", "C", "index_labels", "kd_tree", [1, 3]),
                ("float64", "C", "named", "kd_tree", [1, 3])):
            with self.subTest(dtype=dtype, order=order, edit=edit, algorithm=algorithm, ks=ks):
                if edit == "named":
                    try:
                        import pandas
                    except ImportError:
                        self.skipTest("pandas is required for named training columns")
                labels = ["blue"] * 4 + ["red"] * 4
                if edit == "backend_tie":
                    labels = ["red", "blue", "blue", "red", "red", "red", "blue", "blue"]
                y = np.tile(labels, 2)
                model = Pipeline([("array", _RetainedTrainingArray(dtype, order, edit)),
                                  ("knn", KNeighborsClassifier(algorithm=algorithm, leaf_size=2, n_jobs=1))])
                directory = Path(self.temp.name) / f"{dtype}-{order}-{edit}-{len(ks)}"
                report = calibration_knn_grid(model, {"knn__n_neighbors": ks}, X, y, cv=2,
                                              folds=folds, threads=1, checkpoint_dir=directory)
                expected_means = []
                for candidate in report["candidates"]:
                    scores = []
                    for fold, (train, valid) in zip(candidate["folds"], folds):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            fitted = clone(model).set_params(**candidate["parameters"]).fit(X[train], y[train])
                            scores.append(np.mean(fitted.predict(X[valid]) == y[valid]))
                        self.assertEqual(fold["warnings"], [
                            {"category": w.category.__name__, "message": str(w.message)} for w in caught])
                    self.assertTrue(candidate["valid"], candidate)
                    self.assertEqual([fold["score"] for fold in candidate["folds"]], scores)
                    self.assertEqual(candidate["mean_score"], np.mean(scores))
                    expected_means.append(np.mean(scores))
                winner = ks[np.argmax(expected_means)]
                self.assertEqual(report["selected_parameters"], {"knn__n_neighbors": winner})

    def test_failed_projection_preserves_executed_head_fit_costs_on_resume(self):
        # Supplying p in both locations makes sklearn emit a fit-time SyntaxWarning.
        model = Pipeline([("array", _RetainedTrainingArray(edit="raises")),
                          ("knn", KNeighborsClassifier(metric="minkowski", metric_params={"p": 2}))])
        grid = {"knn__n_neighbors": [1, 3, len(self.y) + 1]}
        a, b = np.split(np.arange(len(self.y)), 2)
        folds = [(a, b), (b, a)]
        report = self.shared(model, grid, cv=2, folds=folds)
        total = 0.
        for candidate in report["candidates"]:
            k = candidate["parameters"]["knn__n_neighbors"]
            for fold, (train, valid) in zip(candidate["folds"], folds):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = clone(model).set_params(**candidate["parameters"]).fit(self.X[train], self.y[train])
                    fit_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
                    with self.assertRaisesRegex(ValueError, "validation projection failed"):
                        fitted.predict(self.X[valid])
                self.assertEqual(fold["warnings"], [
                    {"category": w.category.__name__, "message": str(w.message)} for w in caught])
                diagnostics = report["representation_folds"][fold["fold"]]["neighbors"].get(str(k))
                if diagnostics is not None:
                    self.assertEqual(diagnostics["warnings"], fit_warnings if k <= 3 else [])
                self.assertEqual(fold["failure_stage"], "projection")
                self.assertEqual(fold["error"], "validation projection failed")
                if k <= 3:
                    self.assertGreater(fold["fit_seconds"], 0.)
                else:
                    self.assertEqual(fold["fit_seconds"], 0.)
                total += fold["fit_seconds"]
        self.assertAlmostEqual(report["compute_seconds"]["knn_fit_seconds"], total)
        resumed = self.shared(model, grid, cv=2, folds=folds)
        self.assertEqual(resumed["new_representation_fits"], 0)
        self.assertEqual(resumed["candidates"], report["candidates"])
        self.assertEqual(resumed["compute_seconds"], report["compute_seconds"])

    def test_pipeline_candidates_preserve_the_fitted_neighbor_index(self):
        # From the origin, Euclidean distance favors (2, 2) and Manhattan favors (3, 0).
        train = np.array([[2, 2], [3, 0], [3.1, 0]])
        query = np.array([[0, 0], [.011, .001], [.023, .004]])
        X = np.vstack([train, train + [30, 0], query, query + [30, 0]]).astype(np.float32)
        y = np.array([13, -7, -7, -7, 13, 13, -7, -7, -7, 13, 13, 13])
        a, b = np.split(np.arange(len(y)), 2)
        folds = [(a, b), (b, a)]
        grid = {"knn__n_neighbors": [1, 3]}
        for index_type, form in (("neighbors", "array"), ("neighbors", "array_only"),
                                 ("neighbors", "list"),
                                 ("kd_tree", "memoryview"),
                                 ("ball_tree", "array"), ("array", "array"),
                                 ("brute", "dok"), ("brute", "lil")):
            with self.subTest(index_type=index_type, form=form):
                index = ("passthrough" if index_type == "array"
                         else _TrainingNeighborIndex(index_type, form))
                model = Pipeline([("index", index), ("knn", KNeighborsClassifier(algorithm="kd_tree"))])
                report = calibration_knn_grid(model, grid, X, y, cv=2, folds=folds,
                                              checkpoint_dir=Path(self.temp.name) / f"{index_type}-{form}")
                for candidate in report["candidates"]:
                    self.assertTrue(candidate["valid"], candidate)
                    scores = []
                    for training, validation in folds:
                        fitted = clone(model).set_params(**candidate["parameters"]).fit(X[training], y[training])
                        scores.append(np.mean(fitted.predict(X[validation]) == y[validation]))
                    self.assertEqual([f["score"] for f in candidate["folds"]], scores)
                    if index_type == "neighbors" and form in {"array", "array_only"}:
                        for fold, (_, validation) in zip(candidate["folds"], folds):
                            route = fold["neighbor_query_reuse"]
                            self.assertFalse(route["shared_query_attempted"])
                            self.assertEqual(route["public_fallback_query_rows"], len(validation))
                # The original native tree uses L1. The numeric and brute heads use L2.
                winner = 3 if index_type in {"array", "brute"} else 1
                self.assertEqual(report["selected_parameters"], {"knn__n_neighbors": winner})


if __name__ == "__main__":
    unittest.main()
