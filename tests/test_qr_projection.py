"""Estimator and process contracts for QR projection. Author: Chang Cheng."""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from supervised_nmf import FeatureSPA, NMFClassifier, _nnls


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(2026102562)
    basis = np.column_stack([np.eye(4), rng.uniform(.05, .6, (4, 60))])
    weights = rng.uniform(.1, 2, (96, 4))
    train = weights @ basis
    query = rng.uniform(.1, 2, (64, 4)) @ basis
    labels = (weights[:, 0] > np.median(weights[:, 0])).astype(int)
    return train, query, labels


def outputs(model, query):
    scores = (model.decision_function(query) if model.classifier == "linear_svm"
              else model.predict_proba(query))
    return model.transform(query), model.predict(query), scores


def assert_outputs(actual, expected, *, exact=False):
    np.testing.assert_array_equal(actual[1], expected[1])
    for index in (0, 2):
        if exact:
            np.testing.assert_array_equal(actual[index], expected[index])
        else:
            np.testing.assert_allclose(actual[index], expected[index], rtol=1e-7, atol=1e-9)


def test_projection_option_does_not_change_spa_basis_fit(classification_data):
    train, query, _ = classification_data
    reference = FeatureSPA(4, nnls_max_iter=20000).fit(train)
    with patch.object(_nnls, "_project_qr", side_effect=AssertionError("Basis entered QR")):
        model = FeatureSPA(4, projection_solver="qr", nnls_max_iter=20000).fit(train)
    for name in ("anchor_indices_", "components_", "fit_coordinates_"):
        np.testing.assert_array_equal(getattr(model, name), getattr(reference, name))
    assert model.reconstruction_err_ == reference.reconstruction_err_
    with patch.object(_nnls.linalg, "qr", wraps=_nnls.linalg.qr) as factor:
        actual = model.transform(query)
    assert factor.call_count == 1
    np.testing.assert_allclose(actual, reference.transform(query), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("head", ["knn", "logistic", "linear_svm"])
def test_classifier_projection_persistence_and_refit(head, classification_data, tmp_path):
    train, query, labels = classification_data
    originals = [array.copy() for array in (train, query, labels)]
    settings = dict(n_components=4, solver="spa", screening="none", classifier=head,
                    n_neighbors=5, projection_workers=1, projection_native_threads=1,
                    projection_batch_size=256, training_coordinates="auto",
                    max_iter=20000, random_state=19)
    reference = NMFClassifier(projection_solver="auto", **settings).fit(train, labels)
    with patch.object(_nnls.linalg, "qr", wraps=_nnls.linalg.qr) as factor:
        model = NMFClassifier(projection_solver="qr", **settings).fit(train, labels)
        expected = outputs(model, query)
    # Training reprojection and the query methods really reach QR.
    assert factor.call_count >= 4
    assert model.decomposition_.projection_solver == "qr"
    np.testing.assert_array_equal(model.components_, reference.components_)
    assert_outputs(expected, outputs(reference, query))
    assert_outputs(outputs(model, query), expected, exact=True)
    # A new solver choice adds no fitted factor/cache/pool attributes.
    assert set(vars(model)) == set(vars(reference))
    assert set(vars(model.decomposition_)) == set(vars(reference.decomposition_))

    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params() == model.get_params()
    assert_outputs(outputs(restored, query), expected, exact=True)
    saved = tmp_path / "model.joblib"
    joblib.dump(model, saved)
    assert_outputs(outputs(joblib.load(saved), query), expected, exact=True)
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        copied.predict(query)

    # The existing execution controls remain changeable without fitting.
    fitted_head = model.classifier_
    model.set_params(projection_solver="auto")
    assert_outputs(outputs(model, query), expected)
    assert model.classifier_ is fitted_head
    model.set_params(projection_solver="qr")
    assert_outputs(outputs(model, query), expected, exact=True)

    invalid = train.copy()
    invalid[0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        model.fit(invalid, labels)
    assert not hasattr(model, "components_")
    assert not hasattr(model, "training_coordinates_")
    with pytest.raises(NotFittedError):
        model.predict(query)
    model.fit(train, labels)
    assert_outputs(outputs(model, query), expected, exact=True)
    # A different fitted basis cannot reuse a previous QR factorization.
    model.fit(train * 2, labels)
    fresh = clone(model).fit(train * 2, labels)
    assert_outputs(outputs(model, query), outputs(fresh, query), exact=True)
    for original, array in zip(originals, (train, query, labels)):
        np.testing.assert_array_equal(array, original)


def test_qr_runs_in_process_workers_at_dispatch_boundaries(tmp_path):
    # An importable test helper wraps the real production worker entry point.
    # Child instrumentation is installed inside each worker, not inherited from
    # a parent-only monkeypatch. The separate interpreter owns all test pools.
    helper = textwrap.dedent('''
        import json
        import os
        from pathlib import Path
        from unittest.mock import patch
        from threadpoolctl import threadpool_info
        from supervised_nmf import _nnls
        from supervised_nmf._projection import _project_block as production_block

        def observed_block(design, rows, solver, maxiter, batch_size, native_threads):
            records = []
            original_qr = _nnls.linalg.qr
            def observed_qr(*args, **kwargs):
                records.append([pool["num_threads"] for pool in threadpool_info()])
                return original_qr(*args, **kwargs)
            succeeded = False
            try:
                with patch.object(_nnls.linalg, "qr", side_effect=observed_qr):
                    answer = production_block(design, rows, solver, maxiter,
                                              batch_size, native_threads)
                succeeded = True
                return answer
            finally:
                entry = dict(pid=os.getpid(), rows=len(rows), solver=solver,
                             qr_calls=len(records), thread_counts=records,
                             succeeded=succeeded)
                path = Path(os.environ["QR_TEST_RECEIPTS"]) / (str(os.getpid()) + ".jsonl")
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry) + "\\n")
    ''')
    (tmp_path / "qr_worker_observation.py").write_text(helper, encoding="utf-8")
    script = textwrap.dedent('''
        import json
        import os
        from pathlib import Path
        import sys
        from unittest.mock import patch
        # Match the existing process-test accommodation for renamed Windows
        # interpreter executables. This is never performed by package code.
        if os.name == "nt" and not Path(sys._base_executable).is_file():
            executable = Path(sys.base_prefix) / "python.exe"
            assert executable.is_file()
            sys._base_executable = str(executable)
        import numpy as np
        from joblib.externals.loky import reusable_executor
        from threadpoolctl import threadpool_limits
        from supervised_nmf import _projection
        from qr_worker_observation import observed_block

        design = np.tile(np.eye(4), (4, 1)) * [1., 1.25, 1.5, 1.75]
        known = (np.arange(513 * 4).reshape(513, 4) % 17) / 8
        rows = known @ design.T
        positions = {}
        def new_receipts():
            entries = []
            for path in Path(os.environ["QR_TEST_RECEIPTS"]).glob("*.jsonl"):
                lines = path.read_text(encoding="utf-8").splitlines()
                entries.extend(json.loads(line) for line in lines[positions.get(path, 0):])
                positions[path] = len(lines)
            return entries
        summaries = []
        try:
            with threadpool_limits(1), patch.object(_projection, "_project_block", observed_block):
                for count in (511, 512, 513):
                    for size in (255, 256, 257):
                        actual = _projection.project_rows(design, rows[:count], solver="qr",
                            workers=2, batch_size=size, native_threads=1, maxiter=1000)
                        np.testing.assert_allclose(actual, known[:count], rtol=1e-12, atol=1e-12)
                        records = new_receipts()
                        assert records and all(r["succeeded"] for r in records)
                        assert all(r["solver"] == "qr" for r in records)
                        process = count >= max(512, 2 * size)
                        assert all((r["pid"] != os.getpid()) == process for r in records)
                        assert sum(r["qr_calls"] for r in records) > 0
                        assert all(r["qr_calls"] == int(r["rows"] >= 32) for r in records)
                        assert all(t == 1 for r in records for pools in r["thread_counts"] for t in pools)
                        assert len({r["pid"] for r in records}) <= 2
                        if process:
                            assert len(reusable_executor._executor._processes) == 2
                        summaries.append(dict(rows=count, batch=size, process=process,
                                              qr_calls=sum(r["qr_calls"] for r in records)))
                # Persistent workers must receive a changed design on each call.
                changed = _projection.project_rows(design * 2, rows, solver="qr",
                    workers=2, batch_size=256, native_threads=1, maxiter=1000)
                np.testing.assert_allclose(changed, known / 2, rtol=1e-12, atol=1e-12)
                demanding = np.tile(design.sum(axis=1), (512, 1))
                try:
                    _projection.project_rows(design, demanding, solver="qr", workers=2,
                        batch_size=256, native_threads=1, maxiter=1)
                except RuntimeError as error:
                    assert "iterations" in str(error).lower()
                else:
                    raise AssertionError("Worker iteration failure was suppressed")
                recovered = _projection.project_rows(design, demanding, solver="qr", workers=2,
                    batch_size=256, native_threads=1, maxiter=1000)
                np.testing.assert_allclose(recovered, np.ones((512, 4)), rtol=1e-12, atol=1e-12)
        finally:
            executor = reusable_executor._executor
            if executor is not None:
                executor.shutdown(wait=True, kill_workers=True)
        print(json.dumps(dict(cases=summaries, changed_design=True,
                              worker_failure_propagated=True, recovery=True)))
    ''')
    root = Path(__file__).resolve().parents[1]
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(root / "src"), env.get("PYTHONPATH", "")))
    env["QR_TEST_RECEIPTS"] = str(receipts)
    for name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["LOKY_MAX_CPU_COUNT"] = "2"
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert len(payload["cases"]) == 9
    assert sum(case["process"] for case in payload["cases"]) == 4
    assert payload["changed_design"] and payload["worker_failure_propagated"] and payload["recovery"]
