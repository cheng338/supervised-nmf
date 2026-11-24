"""A saved fold belongs to its fit RNG owner and its place in the draw sequence."""

from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils import check_random_state

from supervised_nmf import evaluation, search, study


class Interrupted(BaseException):
    pass


@pytest.fixture(autouse=True)
def restore_global_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def data():
    rng = np.random.RandomState(7421)
    X = rng.normal(size=(149, 7))
    y = (X[:, 0] * X[:, 1] - .6 * X[:, 2] + rng.normal(size=len(X)) * .4 > 0).astype(int)
    return X, y


def tree():
    return ExtraTreeClassifier(max_depth=3, max_features=2, min_samples_leaf=2, random_state=17)


def science(result, held_out):
    report = result.report
    assert all(candidate["valid"] for candidate in report["candidates"])
    return (
        report["folds"], [[fold["score"] for fold in candidate["folds"]] for candidate in report["candidates"]],
        report["selected_candidate_index"], report["selected_mean_score"],
        None if result.estimator is None else result.estimator.predict(held_out).tolist(),
        None if result.estimator is None else result.estimator.predict_proba(held_out).tolist(),
    )


def mixed_search(data, owner, path=None):
    X, y = data
    np.random.seed(841)
    stream = check_random_state(None) if owner == "ambient" else np.random.RandomState(841)
    return evaluation.calibration_search(
        tree(), {"random_state": [None, stream]}, X[:90], y[:90], cv=3, seed=151,
        checkpoint_path=path, refit=False,
    )


@pytest.mark.parametrize(("writer", "reader"), [("ambient", "private"), ("private", "ambient")])
@pytest.mark.parametrize("partial", [False, True])
def test_mixed_grid_rejects_exchanging_ambient_and_equal_private_rng(tmp_path, data, writer, reader, partial):
    path = tmp_path / "cv.json"
    fresh_writer = science(mixed_search(data, writer), data[0][90:])
    fresh_reader = science(mixed_search(data, reader), data[0][90:])
    assert fresh_writer[0] == fresh_reader[0]
    # The first candidate advances the ambient stream. A clone for the next
    # candidate starts at that later position only if its source is the singleton.
    assert fresh_writer[1][0] == fresh_reader[1][0]
    assert fresh_writer[1][1] != fresh_reader[1][1]

    if partial:
        publish = evaluation.write_json

        def stop(path, report):
            publish(path, report)
            if len(report["candidates"][1]["folds"]) == 1:
                raise Interrupted

        with patch.object(evaluation, "write_json", stop), pytest.raises(Interrupted):
            mixed_search(data, writer, path)
    else:
        assert science(mixed_search(data, writer, path), data[0][90:]) == fresh_writer

    before = path.read_bytes()
    with patch.object(ExtraTreeClassifier, "fit") as fits, patch.object(evaluation, "write_json") as writes:
        with pytest.raises(ValueError, match="incompatible|verifiable"):
            mixed_search(data, reader, path)
        fits.assert_not_called()
        writes.assert_not_called()
    assert path.read_bytes() == before

    if np.__version__.split(".")[0] == "2":
        with patch.object(ExtraTreeClassifier, "fit") as fits:
            if partial:
                with pytest.raises(ValueError, match="random state|ineligible"):
                    mixed_search(data, writer, path)
            else:
                assert science(mixed_search(data, writer, path), data[0][90:]) == fresh_writer
            fits.assert_not_called()


@pytest.mark.parametrize("owner", ["ambient", "private", "split", "integer"])
def test_seeded_partial_fold_recovery_matches_uninterrupted_predictions(tmp_path, data, owner):
    path = tmp_path / "cv.json"
    X, y = data

    def run(path=None, *, ambient=314):
        np.random.seed(ambient)
        split = np.random.RandomState(137)
        fit_rng = {"ambient": check_random_state(None), "private": np.random.RandomState(314),
                   "split": split, "integer": 314}[owner]
        return evaluation.calibration_search(
            tree().set_params(random_state=fit_rng), {"max_depth": [2, 4]}, X[:90], y[:90],
            cv=3, seed=split, checkpoint_path=path,
        )

    publish = evaluation.write_json

    def stop(path, report):
        publish(path, report)
        if len(report["candidates"][0]["folds"]) == 1:
            raise Interrupted

    with patch.object(evaluation, "write_json", stop), pytest.raises(Interrupted):
        run(path)
    ambient = 314 if owner == "ambient" else 919
    oracle = science(run(ambient=ambient), X[90:])
    if owner != "integer" and np.__version__.split(".")[0] != "2":
        with pytest.raises(ValueError, match="incompatible|verifiable"):
            run(path, ambient=ambient)
        return
    sizes = []
    original_fit = ExtraTreeClassifier.fit

    def count(model, rows, labels):
        sizes.append(len(rows))
        return original_fit(model, rows, labels)

    with patch.object(ExtraTreeClassifier, "fit", count):
        assert science(run(path, ambient=ambient), X[90:]) == oracle
    assert sizes == [60] * 5 + [90]


@pytest.mark.parametrize("api", ["grid", "rank"])
def test_shared_study_reader_sees_rng_held_by_callback(tmp_path, data, api):
    X, y = data

    def run(directory, owner):
        np.random.seed(841)
        stream = check_random_state(None) if owner == "ambient" else np.random.RandomState(841)

        def rescale(rows):
            words = stream.get_state()[1][:rows.shape[1]]
            return rows * (1.0 + words.astype(float) / 2**32)

        estimator = Pipeline([("map", FunctionTransformer(rescale)), ("pca", PCA(random_state=23)),
                              ("knn", KNeighborsClassifier())])
        grid = {"pca__n_components": [2], "knn__n_neighbors": [1, 3]}
        if api == "rank":
            report = search.calibration_rank_search(estimator, grid, X[:90], y[:90], checkpoint_dir=directory,
                                                    cv=3, max_expansions=0)["stages"][0]
        else:
            report = study.calibration_knn_grid(estimator, grid, X[:90], y[:90], checkpoint_dir=directory, cv=3)
        assert all(row["valid"] for row in report["candidates"])
        return (report["identity"]["folds"], report["selected_parameters"],
                [[fold["score"] for fold in row["folds"]] for row in report["candidates"]])

    for owner, other in (("ambient", "private"), ("private", "ambient")):
        path = tmp_path / owner
        result = run(path, owner)
        before = {file: file.read_bytes() for file in path.rglob("*.json")}
        with patch.object(Pipeline, "fit") as fits:
            with pytest.raises(ValueError, match="incompatible|verify"):
                run(path, other)
            if owner == "ambient" or np.__version__.split(".")[0] != "2":
                with pytest.raises(ValueError, match="verify"):
                    run(path, owner)
            fits.assert_not_called()
        assert before == {file: file.read_bytes() for file in path.rglob("*.json")}
        if owner == "private" and np.__version__.split(".")[0] == "2":
            oracle = run(tmp_path / "reference", owner)
            assert run(path, owner) == oracle == result


class EntropyTree(ClassifierMixin, BaseEstimator):
    def __init__(self, seed=None):
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.tree_ = tree().set_params(random_state=int(rng.integers(2**30))).fit(X, y)
        self.classes_ = self.tree_.classes_
        return self

    def predict(self, X):
        return self.tree_.predict(X)

    def predict_proba(self, X):
        return self.tree_.predict_proba(X)


@pytest.mark.parametrize("api", ["grid", "rank"])
@pytest.mark.parametrize("parameter_location", ["estimator", "grid"])
def test_partial_study_with_advancing_private_callback_matches_or_refuses(tmp_path, data, api, parameter_location):
    X, y = data

    def run(directory):
        stream = np.random.RandomState(51)

        def project(rows):
            return rows @ stream.normal(size=(rows.shape[1], 4))

        grid = {"pca__n_components": [2], "knn__n_neighbors": [1, 3]}
        transform = FunctionTransformer(project)
        if parameter_location == "grid":
            transform = FunctionTransformer()
            grid["map__func"] = [project]
        estimator = Pipeline([("map", transform), ("pca", PCA(random_state=23)),
                              ("knn", KNeighborsClassifier())])
        args = (estimator, grid, X[:90], y[:90])
        if api == "rank":
            report = search.calibration_rank_search(*args, checkpoint_dir=directory, cv=3, max_expansions=1)
            stages = report["stages"]
        else:
            stages = [study.calibration_knn_grid(*args, checkpoint_dir=directory, cv=3)]
        assert all(c["valid"] for stage in stages for c in stage["candidates"])
        return [(stage["identity"]["folds"], stage["ranked_candidate_indices"],
                 [[f["score"] for f in c["folds"]] for c in stage["candidates"]]) for stage in stages]

    path = tmp_path / "interrupted"
    publish = study._write_receipt

    def stop(file, record):
        publish(file, record)
        if file.parent.name == "folds":
            raise Interrupted

    oracle = run(tmp_path / "uninterrupted")
    with patch.object(study, "_write_receipt", stop), pytest.raises(Interrupted):
        run(path)
    before = {file: file.read_bytes() for file in path.rglob("*.json")}
    original_fit = Pipeline.fit
    with patch.object(Pipeline, "fit", autospec=True, side_effect=original_fit) as fits:
        try:
            result = run(path)
        except ValueError as exc:
            assert "random" in str(exc).lower() or "verif" in str(exc).lower()
            fits.assert_not_called()
            assert before == {file: file.read_bytes() for file in path.rglob("*.json")}
        else:
            assert result == oracle


def test_custom_entropy_source_accepts_fresh_calls_and_recovers_with_declared_integer(tmp_path, data):
    X, y = data

    def run(seed, path):
        return evaluation.calibration_search(EntropyTree(seed), {}, X[:90], y[:90], cv=3, seed=53,
                                             checkpoint_path=path)

    # Fresh entropy has no reproducible origin in NumPy's legacy singleton.
    # This fit exercises fresh-call compatibility. The oracle uses an integer.
    assert all(row["valid"] for row in run(None, tmp_path / "entropy.json").report["candidates"])
    path = tmp_path / "controlled.json"
    run(23, path)
    np.random.seed(19)
    assert science(run(23, path), X[90:]) == science(run(23, None), X[90:])


def rescale_positional(rows, random_state=None):
    return rows * check_random_state(random_state).normal(size=rows.shape[1])


def rescale_keyword_only(rows, *, random_state=None):
    return rows * check_random_state(random_state).normal(size=rows.shape[1])


def rescale_integer(rows, seed=19, /):
    return rows * check_random_state(seed).normal(size=rows.shape[1])


def rescale_required(random_state, /, rows):
    return rows * check_random_state(random_state).normal(size=rows.shape[1])


def rescale_required_keyword(rows, *, random_state):
    return rows * check_random_state(random_state).normal(size=rows.shape[1])


def rescale_variadic(*args):
    random_state, rows = args
    return rows * check_random_state(random_state).normal(size=rows.shape[1])


def capture_random_state(random_state):
    return lambda rows: rescale_positional(rows, random_state)


def capture_rng(rng):
    return lambda rows: rescale_positional(rows, rng)


def capture_seed(seed):
    return lambda rows: rescale_positional(rows, seed)


def capture_offset(offset):
    return lambda rows: rescale_positional(rows, 19) + (0 if offset is None else offset)


@pytest.mark.parametrize("api", ["generic", "shared", "rank"])
@pytest.mark.parametrize(("callback", "fixed"), [
    pytest.param(rescale_positional, False, id="positional-default"),
    pytest.param(rescale_keyword_only, False, id="keyword-default"),
    pytest.param(rescale_integer, True, id="integer-default"),
    pytest.param(partial(rescale_required, None), False, id="positional-binding"),
    pytest.param(partial(rescale_required_keyword, random_state=None), False, id="keyword-binding"),
    pytest.param(partial(rescale_required, 53), True, id="integer-binding"),
    pytest.param(partial(rescale_variadic, None), False, id="variadic-binding"),
    pytest.param(capture_random_state(None), False, id="random-state-closure"),
    pytest.param(capture_rng(None), False, id="rng-closure"),
    pytest.param(capture_seed(None), False, id="seed-closure"),
    pytest.param(capture_seed(19), True, id="integer-closure"),
    pytest.param(capture_offset(None), True, id="unrelated-none-closure"),
])
def test_declared_callback_seed_binds_partial_and_completed_reads(tmp_path, api, callback, fixed):
    data = np.random.RandomState(7421)
    X = data.normal(size=(90, 6))
    y = (X[:, 0] * X[:, 1] - .6 * X[:, 2] + data.normal(size=len(X)) * .4 > 0).astype(int)

    def run(directory, origin):
        np.random.seed(origin)
        estimator = Pipeline([("pca", PCA(random_state=23)), ("map", FunctionTransformer(callback)),
                              ("knn", KNeighborsClassifier(n_jobs=1))])
        args = (estimator, {"pca__n_components": [2], "knn__n_neighbors": [1, 3]}, X, y)
        if api == "generic":
            report = evaluation.calibration_search(*args, seed=53, cv=3, refit=False,
                                                   checkpoint_path=directory / "cv.json").report
            folds = report["folds"]
        else:
            if api == "shared":
                report = study.calibration_knn_grid(*args, seed=53, cv=3, checkpoint_dir=directory)
            else:
                report = search.calibration_rank_search(*args, seed=53, cv=3, checkpoint_dir=directory,
                                                        max_expansions=0)["stages"][0]
            folds = report["identity"]["folds"]
        assert all(row["valid"] for row in report["candidates"])
        return folds, report["selected_parameters"], [[f["score"] for f in c["folds"]] for c in report["candidates"]]

    complete = tmp_path / "complete"
    interrupted = tmp_path / "partial"
    initial = run(complete, 841)
    oracle = run(tmp_path / "independent", 921)
    assert initial[0] == oracle[0]
    assert (initial[2] == oracle[2]) is fixed

    module, name = (evaluation, "write_json") if api == "generic" else (study, "_write_receipt")
    publish = getattr(module, name)

    def stop_after_fold(path, report):
        publish(path, report)
        if ((api == "generic" and len(report["candidates"][0]["folds"]) == 1)
                or (api != "generic" and path.parent.name == "folds")):
            raise Interrupted

    with patch.object(module, name, stop_after_fold), pytest.raises(Interrupted):
        run(interrupted, 841)

    for directory in (complete, interrupted):
        if fixed:
            with patch.object(Pipeline, "fit", autospec=True, side_effect=Pipeline.fit) as fits:
                assert run(directory, 921) == oracle
            remaining_fits = 5 if api == "generic" else 2
            assert fits.call_count == (0 if directory == complete else remaining_fits)
            continue
        before = {path: path.read_bytes() for path in directory.rglob("*.json")}
        with patch.object(Pipeline, "fit", autospec=True, side_effect=Pipeline.fit) as fits:
            with pytest.raises(ValueError, match="incompatible|verify"):
                run(directory, 921)
            if api == "generic" and directory == complete:
                assert run(directory, 841) == initial
            else:
                with pytest.raises(ValueError, match="random|verify"):
                    run(directory, 841)
            fits.assert_not_called()
        assert before == {path: path.read_bytes() for path in directory.rglob("*.json")}
