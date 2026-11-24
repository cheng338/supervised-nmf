"""Compare shared KNN calibration with independently fitted and queried heads."""

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

# The fresh interpreter uses the same source and wheel overlay as the parent.
if __name__ == "__main__":
    sys.path[:] = json.loads(sys.argv[1])

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits


def head(k, algorithm="brute"):
    return KNeighborsClassifier(n_neighbors=k, algorithm=algorithm, metric="euclidean", n_jobs=1)


def problem():
    # Each validation row is nearest its partner in the other fold. Small-k
    # decisions have clean boundaries. The first fold's full vote is tied.
    points = np.arange(12, dtype=float)
    X = np.r_[points, points + 0.125][:, None]
    y = np.r_[np.arange(12) % 2, np.repeat([0, 1], [3, 9])]
    first, second = np.arange(12), np.arange(12, 24)
    return X, y, [(first, second), (second, first)]


def assert_public_grid(algorithm, ks, directory, events):
    from supervised_nmf.study import calibration_knn_grid

    X, y, folds = problem()
    expected = {k: [] for k in ks}
    events.clear()
    for train, valid in folds:
        for k in ks:
            model = head(k, algorithm).fit(X[train], y[train])
            try:
                score = float(np.mean(model.predict(X[valid]) == y[valid]))
            except RuntimeError as error:
                expected[k].append((None, type(error).__name__, str(error)))
            else:
                expected[k].append((score, None, None))
    native_events = list(events)

    events.clear()
    report = calibration_knn_grid(
        head(1, algorithm), {"n_neighbors": list(ks)}, X, y, folds=folds, cv=2,
        threads=1, seed=19, checkpoint_dir=directory,
    )
    observed = {
        candidate["parameters"]["n_neighbors"]: [
            (fold.get("score"), fold.get("error_type"), fold.get("error"))
            for fold in candidate["folds"]
        ]
        for candidate in report["candidates"]
    }
    assert observed == expected
    assert events == native_events
    return observed


def changed_predict_body(self, X):
    # Also works in sklearn's defining globals when transplanted into its
    # original function. It has no closure or references to test globals.
    return np.full(len(X), self.classes_[1])


@contextmanager
def changed_code(function, code):
    saved = function.__code__
    try:
        function.__code__ = code
        yield
    finally:
        function.__code__ = saved


def change_public_method(name, events):
    original = getattr(KNeighborsClassifier, name)
    if name == "predict":
        def replacement(self, X):
            events.append(("predict", self.n_neighbors))
            return changed_predict_body(self, X)
    elif name == "predict_proba":
        def replacement(self, X):
            events.append(("predict_proba", self.n_neighbors))
            return np.tile([0.125, 0.875], (len(X), 1))
    else:
        assert name == "kneighbors"

        def replacement(self, X=None, n_neighbors=None, return_distance=True):
            width = self.n_neighbors if n_neighbors is None else n_neighbors
            events.append(("kneighbors", self.n_neighbors, width, return_distance))
            if width == len(self._fit_X):
                raise RuntimeError("selected full-width query")
            return original(self, X, n_neighbors=n_neighbors, return_distance=return_distance)

    return patch.object(KNeighborsClassifier, name, replacement)


def assert_selected_probabilities(algorithm, ks, events):
    from supervised_nmf._neighbors import _KNNQueryReuse

    X, y, folds = problem()
    train, valid = folds[0]
    training = X[train]
    query = X[valid]
    models = {k: head(k, algorithm).fit(training, y[train]) for k in ks}
    calls = tuple((k, True, 0.5, 1) for k in ks) if len(train) in ks else None

    def observe(operation):
        events.clear()
        try:
            probabilities = operation()
        except RuntimeError as error:
            return str(error), list(events)
        return probabilities.tolist(), list(events)

    reuse = _KNNQueryReuse(models, query, _head_fit_input=training, _study_call_contract=calls)
    for k, model in models.items():
        expected = observe(lambda: model.predict_proba(query))
        actual = observe(lambda: reuse.predict(k, probabilities=True, decision_threshold=0.5,
                                              positive_index=1)[0])
        assert actual == expected, (algorithm, k)


def assert_stock_full_query():
    from supervised_nmf._neighbors import _KNNQueryReuse, _native_callback_context, _native_query_methods

    X, y, folds = problem()
    train, valid = folds[0]
    model = head(len(train)).fit(X[train], y[train])
    reuse = _KNNQueryReuse({12: model}, X[valid], _study_call_contract=((12, False, None, None),))
    labels, _, _ = reuse.predict(12)
    np.testing.assert_array_equal(labels, model.predict(X[valid]))
    if (_native_query_methods(model, model) and _native_callback_context(model)
            and (sklearn.__version__, joblib.__version__) in {("1.9.1", "1.6.0"), ("1.9.0", "1.5.3")}):
        assert reuse.query_records()[0].get("study_short_query_active")


def check_class_change(method, when, directory):
    events = []
    if when == "before":
        assert "supervised_nmf" not in sys.modules
    else:
        from supervised_nmf import _neighbors  # Capture the initial bindings.

    if method == "predict_code":
        original = KNeighborsClassifier.predict
        code = changed_predict_body.__code__.replace(
            co_name=original.__code__.co_name, co_qualname=original.__code__.co_qualname,
        )
        change = changed_code(original, code)
    else:
        change = change_public_method(method, events)

    with threadpool_limits(1):
        with change:
            # Auto selects KDTree for the small heads and brute for the endpoint.
            for algorithm, ks in (("brute", (12,)), ("brute", (1, 3, 12)),
                                  ("kd_tree", (1, 3, 12)), ("auto", (1, 3, 12)),
                                  ("brute", (1, 3))):
                subdir = directory / f"{algorithm}-{len(ks)}-{ks[-1]}"
                scores = assert_public_grid(algorithm, ks, subdir, events)
                if method in {"predict", "predict_code"} and 12 in ks:
                    assert [fold[0] for fold in scores[12]] == [0.75, 0.5]
                assert_selected_probabilities(algorithm, ks, events)
        assert_public_grid("brute", (1, 3, 12), directory / "restored", events)
        assert_stock_full_query()


@pytest.mark.parametrize("method", ["predict", "predict_proba", "kneighbors", "predict_code"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_public_method_changed_at_import_boundary(method, when, tmp_path):
    if method == "predict_code" and KNeighborsClassifier.predict.__code__.co_freevars:
        pytest.skip("This scikit-learn predict method is a closure.")
    if when == "after":
        check_class_change(method, when, tmp_path)
        return
    run_fresh(method, tmp_path)


def run_fresh(case, directory):
    paths = [str(Path(entry or ".").resolve()) for entry in sys.path]
    control = None
    if case == "loader_stock":
        # The child must start without this package imported. Measure whether
        # this same installation can share an ordinary query in the parent.
        from supervised_nmf._neighbors import _KNNQueryReuse

        X, y, folds = problem()
        train, valid = folds[0]
        training = X[train]
        models = {k: head(k).fit(training, y[train]) for k in (1, 3)}
        with threadpool_limits(1):
            reuse = _KNNQueryReuse(models, X[valid])
            actual, metadata, _ = reuse.predict(1, probabilities=True)
            np.testing.assert_array_equal(actual, models[1].predict_proba(X[valid]))
        control = metadata["shared_query_attempted"]
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), json.dumps(paths),
         case, str(directory), json.dumps(control)], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("case", ["stock", "changed_method", "missing_origin", "different_origin",
                                  "missing_file", "different_code"])
def test_custom_module_loader_preserves_selected_predictions(case, tmp_path):
    run_fresh("loader_" + case, tmp_path)


def check_custom_loader(case, directory, control):
    from sklearn.metrics._pairwise_distances_reduction import _dispatcher
    from sklearn.neighbors import _base, _classification

    class CustomLoader:
        def __getattr__(self, name):
            raise AssertionError(f"KNN admission accessed custom loader method {name!r}")

    assert "supervised_nmf" not in sys.modules
    events = []
    loader = CustomLoader()
    with ExitStack() as stack:
        stack.enter_context(threadpool_limits(1))
        for module in (_base, _classification, _dispatcher):
            stack.enter_context(patch.object(module, "__loader__", loader))
            stack.enter_context(patch.object(module.__spec__, "loader", loader))
        if case == "missing_origin":
            stack.enter_context(patch.object(_classification.__spec__, "origin", None))
        elif case == "different_origin":
            stack.enter_context(patch.object(_classification.__spec__, "origin", _base.__file__))
        elif case in {"missing_file", "different_code"}:
            source = directory / "_classification.py"
            if case == "different_code":
                source.write_text("class KNeighborsClassifier:\n    def predict(self, X): return None\n",
                                  encoding="utf-8")
            stack.enter_context(patch.object(_classification, "__file__", str(source)))
            stack.enter_context(patch.object(_classification.__spec__, "origin", str(source)))
        elif case == "changed_method":
            stack.enter_context(change_public_method("predict", events))

        from supervised_nmf._neighbors import _KNNQueryReuse

        scores = assert_public_grid("brute", (1, 3, 12), directory / "grid", events)
        if case == "changed_method":
            assert events and [fold[0] for fold in scores[12]] == [0.75, 0.5]
        X, y, folds = problem()
        train, valid = folds[0]
        training, query = X[train], X[valid]
        models = {k: head(k).fit(training, y[train]) for k in (1, 3)}
        reuse = _KNNQueryReuse(models, query)
        for k, model in models.items():
            actual, metadata, _ = reuse.predict(k, probabilities=True)
            np.testing.assert_array_equal(actual, model.predict_proba(query))
            assert metadata["shared_query_attempted"] is (case == "stock" and control)
            if case != "stock":
                assert metadata["reason"] == "changed_neighbor_method"
        assert reuse.query_records()[0]["queried"] is (case == "stock" and control)
        if case == "stock":
            assert_stock_full_query()


@pytest.mark.parametrize("method", ["predict", "predict_proba", "kneighbors"])
@pytest.mark.parametrize("changed_head", [1, 12])
def test_instance_callback_installed_after_adapter_setup(method, changed_head):
    from supervised_nmf._neighbors import _KNNQueryReuse

    X, y, folds = problem()
    train, valid = folds[0]
    training, query = X[train], X[valid]
    models = {k: head(k).fit(training, y[train]) for k in (1, 12)}
    probability = method == "predict_proba"
    calls = tuple((k, probability, None, None) for k in models)
    model = models[changed_head]
    with threadpool_limits(1):
        reuse = _KNNQueryReuse(models, query, _study_call_contract=calls, _head_fit_input=training)
        with patch.object(model, method, side_effect=RuntimeError("selected instance callback")):
            for k, estimator in models.items():
                try:
                    expected = (estimator.predict_proba if probability else estimator.predict)(query)
                except RuntimeError as error:
                    assert k == changed_head and str(error) == "selected instance callback"
                    with pytest.raises(RuntimeError, match="selected instance callback"):
                        reuse.predict(k, probabilities=probability)
                else:
                    # A native classification kernel can answer without calling
                    # kneighbors. In that case its replacement is unobserved.
                    assert k != changed_head or method == "kneighbors"
                    actual, _, _ = reuse.predict(k, probabilities=probability)
                    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("change", ["argkmin_compute", "argkmin_alias", "classmode_selector", "classmode_compute"])
@pytest.mark.skipif(
    (sklearn.__version__, joblib.__version__) not in {("1.9.1", "1.6.0"), ("1.9.0", "1.5.3")},
    reason="The two qualified scikit-learn builds expose these private dispatcher paths.",
)
def test_brute_dispatch_changes_preserve_selected_fold_results(change, tmp_path, monkeypatch):
    from sklearn.neighbors import _base, _classification
    from supervised_nmf import _neighbors  # Capture the initial bindings.

    argkmin = _base.ArgKmin
    classmode = _classification.ArgKminClassMode
    original_compute = argkmin.compute
    events = []

    def observe_compute(cls, X, Y, k, *args, **kwargs):
        events.append(("compute", k))
        if k == len(Y):
            raise RuntimeError("full fold reached the selected dispatcher")
        return original_compute(X, Y, k, *args, **kwargs)

    def fail_selection(cls, X, Y, metric):
        events.append(("select", metric))
        raise RuntimeError("selected ClassMode predicate")

    def observe_classmode(cls, X, Y, k, **kwargs):
        events.append(("classmode", k))
        if k == len(Y):
            raise RuntimeError("selected full-fold ClassMode")
        return np.tile([0.125, 0.875], (len(X), 1))

    with threadpool_limits(1):
        with monkeypatch.context() as changes:
            if change == "argkmin_compute":
                changes.setattr(argkmin, "compute", classmethod(observe_compute))
            elif change == "argkmin_alias":
                class Replacement(argkmin):
                    compute = classmethod(observe_compute)
                changes.setattr(_base, "ArgKmin", Replacement)
            elif change == "classmode_selector":
                changes.setattr(classmode, "is_usable_for", classmethod(fail_selection))
            else:
                # Stock ClassMode excludes Euclidean; select it to make both
                # the changed kernel's answers and its full-width error visible.
                changes.setattr(classmode, "is_usable_for", classmethod(lambda cls, X, Y, metric: True))
                changes.setattr(classmode, "compute", classmethod(observe_classmode))
            for algorithm in ("brute", "auto"):
                scores = assert_public_grid(algorithm, (1, 3, 12), tmp_path / algorithm, events)
                assert events and all(row[1] == "RuntimeError" for row in scores[12])
                assert_selected_probabilities(algorithm, (1, 3, 12), events)
        assert_stock_full_query()


@pytest.mark.parametrize("counts", [(1, 3), (12,), (1, 3, 12)])
def test_group_setup_does_not_compare_custom_weights(counts):
    from supervised_nmf import _neighbors

    class Weight(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            if sys._getframe(1).f_code is _neighbors._KNNQueryReuse.__init__.__code__:
                raise AssertionError("Group setup added a custom weight comparison")
            return super().__eq__(other)

    X, y, folds = problem()
    train, valid = folds[0]
    training, query = X[train], X[valid]
    with threadpool_limits(1):
        models = {k: head(k).set_params(weights=Weight("uniform")).fit(training, y[train])
                  for k in counts}
        calls = tuple((k, True, 0.5, 1) for k in counts) if len(train) in counts else None
        reuse = _neighbors._KNNQueryReuse(models, query, _study_call_contract=calls, _head_fit_input=training)
        for k, model in models.items():
            actual, _, _ = reuse.predict(k, probabilities=True, decision_threshold=0.5, positive_index=1)
            np.testing.assert_array_equal(actual, model.predict_proba(query))


if __name__ == "__main__":
    directory = Path(sys.argv[3])
    if sys.argv[2].startswith("loader_"):
        check_custom_loader(sys.argv[2].removeprefix("loader_"), directory, json.loads(sys.argv[4]))
    else:
        check_class_change(sys.argv[2], "before", directory)
