"""Solver-aware MU defaults, fitted policy ownership and current persistence."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.decomposition import NMF
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf._mu import transform_mu
from supervised_nmf._mu_rowwise import transform_mu_rowwise
from supervised_nmf._neighbors import _KNNQueryReuse
from supervised_nmf.evaluation import calibration_search, _diagnostics
from supervised_nmf.study import calibration_knn_grid


@pytest.fixture(scope="module", autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module", params=[np.float32, np.float64], ids=["float32", "float64"])
def data(request):
    rng = np.random.default_rng(831)
    basis = np.full((2, 12), .1)
    basis[0, ::2] += 2
    basis[1, 1::2] += 2
    latent = rng.uniform(.3, 2, (72, 2))
    x = (latent @ basis).astype(request.param)
    y = (latent[:, 0] > np.median(latent[:, 0])).astype(int)
    q = (np.asarray([[.4, .7], [.6, 1.5], [1.8, .8], [1.9, 1.7],
                     [.3, 1.1], [1.7, .4], [.5, 1.8], [1.9, 1.]]) @ basis).astype(request.param)
    return x, y, q


def estimator(solver="mu", choice="omitted"):
    options = dict(solver=solver, screening="none", n_components=2, n_neighbors=3,
                   weights="uniform", max_iter=20000, tol=1e-4, random_state=17)
    if choice != "omitted":
        options["mu_projection"] = choice
    return NMFClassifier(**options)


@pytest.fixture(scope="module")
def fitted(data):
    cache = {}
    def get(solver, choice):
        key = solver, choice
        if key not in cache:
            cache[key] = estimator(solver, choice).fit(*data[:2])
            assert cache[key].converged_
        return deepcopy(cache[key])
    return get


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def no_new_work():
    stack = ExitStack()
    for owner, name in ((NMFClassifier, "fit"), (NMFClassifier, "_fit_with_screening"),
                        (NMF, "fit_transform"), (KNeighborsClassifier, "fit"), (_KNNQueryReuse, "predict")):
        stack.enter_context(patch.object(owner, name, side_effect=AssertionError("Unexpected new fitting/query work")))
    return stack


@pytest.mark.parametrize("solver", ["spa", "cd", "mu"])
@pytest.mark.parametrize("choice", ["omitted", "auto", "batch", "rowwise"])
def test_solver_aware_policy_resolution_preserves_explicit_choices(data, fitted, solver, choice):
    x, y, q = data
    if solver != "mu" and choice == "rowwise":
        with patch.object(NMF, "fit_transform", side_effect=AssertionError("Invalid policy reached fitting")):
            with pytest.raises(ValueError, match="rowwise.*requires"):
                estimator(solver, choice).fit(x, y)
        return
    model = fitted(solver, choice)
    requested = "auto" if choice == "omitted" else choice
    resolved = ("rowwise" if requested == "auto" else requested) if solver == "mu" else None
    assert model.get_params()["mu_projection"] == requested
    assert model.mu_projection_ == resolved
    assert _diagnostics(model)["mu_projection"] == resolved
    expected_training = "transform" if solver == "spa" else "fit_transform"
    assert model.training_coordinates_mode_ == expected_training
    assert model.training_projection_performed_ == (solver == "spa")
    control = fitted(solver, resolved if solver == "mu" else "batch")
    np.testing.assert_array_equal(model.training_coordinates_, control.training_coordinates_)
    np.testing.assert_array_equal(model.selected_components_, control.selected_components_)
    np.testing.assert_array_equal(model.transform(q), control.transform(q))
    np.testing.assert_array_equal(model.predict(q), control.predict(q))
    np.testing.assert_array_equal(model.predict_proba(q), control.predict_proba(q))
    if solver == "mu":
        operation = transform_mu_rowwise if resolved == "rowwise" else transform_mu
        selected = q[:, model.selected_features_]
        np.testing.assert_array_equal(model.transform(q), operation(deepcopy(model.decomposition_), selected))


@pytest.mark.parametrize("initial,pending,resolved", [
    ("batch", "auto", "rowwise"), ("auto", "batch", "batch"),
    ("rowwise", "auto", "rowwise"), ("auto", "rowwise", "rowwise")])
def test_pending_auto_policy_clone_refit_and_pickle(data, fitted, initial, pending, resolved):
    x, y, q = data
    model = fitted("mu", initial)
    old_policy = model.mu_projection_
    basis, training = model.selected_components_.copy(), model.training_coordinates_.copy()
    old_coordinates, old_labels = model.transform(q), model.predict(q)
    model.set_params(mu_projection=pending)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.mu_projection == pending and restored.mu_projection_ == old_policy
    np.testing.assert_array_equal(restored.transform(q), old_coordinates)
    np.testing.assert_array_equal(restored.predict(q), old_labels)
    assert _diagnostics(restored)["mu_projection"] == old_policy
    fresh = clone(restored)
    assert fresh.mu_projection == pending and not hasattr(fresh, "mu_projection_")
    fresh.fit(x, y)
    restored.fit(x, y)
    for current in (fresh, restored):
        assert current.mu_projection_ == resolved and current.training_coordinates_mode_ == "fit_transform"
        np.testing.assert_array_equal(current.selected_components_, basis)
        np.testing.assert_array_equal(current.training_coordinates_, training)
    np.testing.assert_array_equal(fresh.transform(q), restored.transform(q))


@pytest.mark.parametrize("initial,target", [("spa", "mu"), ("mu", "cd")])
def test_auto_resolves_again_only_when_solver_refits(data, fitted, initial, target):
    model = fitted(initial, "auto")
    old = model.transform(data[2])
    policy = model.mu_projection_
    model.set_params(solver=target)
    assert model.mu_projection == "auto" and model.mu_projection_ == policy
    np.testing.assert_array_equal(model.transform(data[2]), old)
    model.fit(*data[:2])
    assert model.mu_projection_ == ("rowwise" if target == "mu" else None)
    assert model.training_coordinates_mode_ == "fit_transform"


def test_auto_general_calibration_covers_mixed_solver_grid(data):
    result = calibration_search(estimator(), {"solver": ["spa", "cd", "mu"]}, *data[:2],
                                cv=2, seed=19, threads=1, refit=False).report
    assert result["scoring"] == "accuracy" and result["refit_performed"] is False
    assert len(result["candidates"]) == 3
    for candidate in result["candidates"]:
        assert candidate["valid"] and len(candidate["folds"]) == 2
        resolved = "rowwise" if candidate["parameters"]["solver"] == "mu" else None
        assert all(f["diagnostics"]["mu_projection"] == resolved for f in candidate["folds"])
    assert result["selected_parameters"] is not None


def test_auto_shared_calibration_resume_and_requested_policy_identity(data, tmp_path):
    x, y, _ = data
    directory = tmp_path / "auto"
    grid = {"n_neighbors": [1, 3]}
    kwargs = dict(cv=2, seed=19, threads=1, checkpoint_dir=directory)
    first = calibration_knn_grid(estimator(), grid, x, y, **kwargs)
    assert first["new_representation_fits"] == first["unique_representation_fits"] == 2
    assert all(c["valid"] for c in first["candidates"])
    assert all(f["diagnostics"]["mu_projection"] == "rowwise" for f in first["representation_folds"])
    before = snapshot(directory)
    with no_new_work():
        resumed = calibration_knn_grid(estimator(choice="auto"), grid, x, y, **kwargs)
    after = snapshot(directory)
    assert before.keys() == after.keys()
    assert all(h == after[p] for p, h in before.items() if p != "report.json")
    assert resumed["new_representation_fits"] == 0 and resumed["reused_representation_folds"] == 2
    assert resumed["compute_seconds"] == first["compute_seconds"]
    assert resumed["candidates"] == first["candidates"]
    assert resumed["identity"] == first["identity"]
    for explicit in ("batch", "rowwise"):
        with no_new_work(), pytest.raises(ValueError, match="incompatible"):
            calibration_knn_grid(estimator(choice=explicit), grid, x, y, **kwargs)
        assert snapshot(directory) == after
    explicit = calibration_knn_grid(estimator(choice="rowwise"), grid, x, y,
                                   **{**kwargs, "checkpoint_dir": tmp_path / "explicit-rowwise"})
    assert explicit["identity"] != first["identity"]
    assert explicit["selected_parameters"] == first["selected_parameters"]
    for a, b in zip(first["candidates"], explicit["candidates"]):
        assert a["parameters"] == b["parameters"] and a["mean_score"] == b["mean_score"]
        assert [f["score"] for f in a["folds"]] == [f["score"] for f in b["folds"]]


CHILD = r'''
import os, sys
for key in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
source, model_file, query_file, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(source / 'src'))
import pickle
import numpy as np
import supervised_nmf
from supervised_nmf import NMFClassifier
from sklearn.decomposition import NMF
from threadpoolctl import threadpool_limits, threadpool_info
assert Path(supervised_nmf.__file__).resolve() == source / 'src/supervised_nmf/__init__.py'
def forbidden(*args, **kwargs): raise AssertionError('Current persistence attempted a refit')
NMFClassifier.fit = NMFClassifier._fit_with_screening = NMF.fit_transform = forbidden
with threadpool_limits(limits=1):
    model = pickle.loads(model_file.read_bytes())
    assert type(model) is NMFClassifier
    assert model.mu_projection == 'batch' and model.mu_projection_ == 'rowwise'
    q = np.load(query_file, allow_pickle=False)
    coordinates = model.transform(q)
    labels, probabilities = model.predict_with_probabilities(q)
    assert threadpool_info() and all(p['num_threads'] == 1 for p in threadpool_info())
    np.savez(output, coordinates=coordinates, labels=labels, probabilities=probabilities)
'''


def test_current_pickle_in_fresh_process_keeps_fitted_auto_resolution(data, fitted, tmp_path):
    source = Path(__file__).resolve().parents[1]
    model = fitted("mu", "omitted")
    expected = dict(coordinates=model.transform(data[2]), labels=model.predict(data[2]),
                    probabilities=model.predict_proba(data[2]))
    model.set_params(mu_projection="batch")
    model_file, query_file, output = (tmp_path / n for n in ("model.pkl", "query.npy", "out.npz"))
    model_file.write_bytes(pickle.dumps(model))
    np.save(query_file, data[2])
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (model_file, query_file)}
    command = [sys.executable, "-I", "-B", "-X", "utf8", "-c", CHILD,
               str(source), str(model_file), str(query_file), str(output)]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.DEVNULL, start_new_session=os.name == "posix")
    try:
        stdout, stderr = child.communicate(timeout=120)
        assert child.returncode == 0, (stdout.decode(errors="replace"), stderr.decode(errors="replace"))
    finally:
        if child.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
            except ProcessLookupError:
                pass
            child.communicate(timeout=15)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in originals.items())
    with np.load(output, allow_pickle=False) as actual:
        assert set(actual.files) == set(expected)
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key], value)


@pytest.mark.parametrize("choice", ["omitted", "rowwise"])
def test_auto_and_explicit_rowwise_refuse_custom_decomposition_transform(data, fitted, choice):
    model = fitted("mu", choice)
    before = model.decomposition_.components_.copy()
    calls = []
    def custom(selected):
        calls.append(selected.copy())
        return np.full((len(selected), 2), .25, dtype=selected.dtype)
    model.decomposition_.transform = custom
    # Changing a pending constructor choice must not change fitted rowwise policy.
    model.set_params(mu_projection="batch")
    for query in (data[2], np.zeros_like(data[2])):
        with pytest.raises(ValueError, match="rowwise.*standard fitted"):
            model.transform(query)
    assert not calls and model.mu_projection_ == "rowwise"
    np.testing.assert_array_equal(model.decomposition_.components_, before)


def test_missing_fitted_auto_policy_is_not_reconstructed_from_constructor(data, fitted):
    model = fitted("mu", "omitted")
    del model.mu_projection_
    restored = pickle.loads(pickle.dumps(model))
    assert restored.mu_projection == "auto" and not hasattr(restored, "mu_projection_")
    for method in ("transform", "predict", "predict_proba", "predict_with_probabilities"):
        with pytest.raises(AttributeError, match="mu_projection_"):
            getattr(restored, method)(data[2])
