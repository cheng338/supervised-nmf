"""Contracts for completed Linux work reuse and clean fallback."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from contextvars import Context, ContextVar, copy_context
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import warnings
from xml.etree import ElementTree

CONTEXT_SENTINEL = ContextVar("snmf_reuse_test_context", default="empty")

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(SOURCE / "scripts"))
sys.path.insert(0, str(SOURCE / "examples"))
import numpy as np
import pytest
import sklearn
import threadpoolctl
from sklearn.neighbors import KNeighborsClassifier
from supervised_nmf import evaluation, study
from supervised_nmf import _completed_reuse as reuse
from supervised_nmf.classifier import NMFClassifier

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux completed-work reuse")

assert Path(reuse.__file__).resolve() == SOURCE / "src/supervised_nmf/_completed_reuse.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, SOURCE / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DEFAULT = load("completed_reuse_default", "scripts/celeba_default_study.py")
PARAMETER = load("completed_reuse_parameter", "examples/celeba_parameter_study.py")


def inactive_threads():
    assert not [t for t in threading.enumerate() if t.name.startswith("snmf-reuse")]


def science(value):
    if isinstance(value, dict):
        return {k: science(v) for k, v in value.items() if k not in {"current_call_wall_seconds", "new_representation_fits", "reused_representation_folds"}}
    if isinstance(value, list):
        return [science(v) for v in value]
    return value


def committed(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [directory / "identity.json", *sorted((directory / "folds").glob("*.json"))]}


@pytest.fixture
def completed(tmp_path):
    rng = np.random.default_rng(747)
    X, y = rng.uniform(.1, 1, (30, 8)), np.arange(30) % 2
    directories = [tmp_path / str(i) for i in range(2)]
    grid = {"n_neighbors": [1, 3]}
    reports = [study.calibration_knn_grid(KNeighborsClassifier(n_jobs=1), grid, X, y,
               cv=3, seed=17, threads=1, checkpoint_dir=d) for d in directories]
    def describe(i):
        return KNeighborsClassifier(n_jobs=1), grid, directories[i], 1
    def invoke(i, environment):
        return study._calibration_knn_grid(KNeighborsClassifier(n_jobs=1), grid, X, y,
               cv=3, seed=17, threads=1, checkpoint_dir=directories[i], _reuse_only=True, _reuse_environment=environment)
    return directories, reports, describe, invoke, X, y


def dummy(tmp_path, count=2):
    paths = [tmp_path / str(i) for i in range(count)]
    for path in paths:
        path.mkdir()
        (path / "identity.json").write_text("hint")
        key = hashlib.sha256(str(path.name).encode()).hexdigest()
        (path / "folds").mkdir()
        (path / "folds" / (key + ".json")).write_text("callback fixture only")
        payload = {"state": "cv_completed", "representation_folds": [{"state": "completed", "key": key}]}
        (path / "report.json").write_text(json.dumps({"sha256": "0" * 64, "payload": payload}))
    def describe(i):
        return KNeighborsClassifier(n_jobs=1), {"n_neighbors": [1]}, paths[i], 1
    return list(range(count)), describe


@pytest.mark.parametrize("workers,count", [(1, 2), (8, 1)])
def test_actual_completed_science_guards_and_no_worker_setter(tmp_path, monkeypatch, completed, workers, count):
    directories, original, describe, invoke, _, _ = completed
    directories, original = directories[:count], original[:count]
    before = [committed(d) for d in directories]
    owner = threading.get_ident()
    original_limits, original_eye = threadpoolctl.threadpool_limits, np.eye
    def limit(*args, **kwargs):
        assert threading.get_ident() == owner, "native setter in a reuse thread"
        return original_limits(*args, **kwargs)
    def eye(*args, **kwargs):
        assert threading.get_ident() == owner, "matrix probe in a reuse thread"
        return original_eye(*args, **kwargs)
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", limit)
    monkeypatch.setattr(study, "threadpool_limits", limit)
    monkeypatch.setattr(evaluation, "threadpool_limits", limit)
    monkeypatch.setattr(np, "eye", eye)
    monkeypatch.setattr(study, "_run_representation_fold", lambda *a, **k: pytest.fail("new fit in reuse"))
    state, config = np.geterr(), sklearn.get_config()
    visited, participants = [], []
    def ordered(i, environment):
        visited.append(i)
        participants.append(threading.get_ident())
        return invoke(i, environment)
    guard_calls = []
    original_guard = reuse._ReadOnlyEnvironment.__call__
    def observed_guard(self, threads):
        guard_calls.append(threading.get_ident())
        return original_guard(self, threads)
    monkeypatch.setattr(reuse._ReadOnlyEnvironment, "__call__", observed_guard)
    result = reuse._try_completed(list(range(count)), workers, describe=describe, invoke=ordered)
    assert visited == list(range(count))
    assert len(set(participants)) == 1 and participants[0] != owner
    assert result is not None
    assert [science(report) for report in result] == [science(report) for report in original]
    assert all(report["new_representation_fits"] == 0 and report["reused_representation_folds"] == 3 for report in result)
    assert [committed(d) for d in directories] == before
    assert participants[0] in guard_calls
    assert np.geterr() == state and sklearn.get_config() == config
    inactive_threads()


def test_completed_reuse_two_thread_startup(tmp_path):
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "2"
    env.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    env.pop("PYTEST_ADDOPTS", None)
    report = tmp_path / "two-thread.xml"
    name = "test_admission_matches_fresh_participant"
    node = f"tests/_completed_reuse_width_control.py::{name}"
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(report), node],
        cwd=SOURCE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
    )
    assert child.returncode == 0, child.stdout
    # A skipped child also exits successfully. Require this specific case to have run.
    cases = ElementTree.parse(report).findall(".//testcase")
    assert len(cases) == 1 and cases[0].get("name") == name and cases[0].find("skipped") is None, child.stdout


def test_provider_preserves_exact_search_and_runtime_identities(completed):
    _, _, _, _, X, y = completed
    evaluation._initialized_provenance(1)
    with threadpoolctl.threadpool_limits(limits=1):
        environment = reuse._ReadOnlyEnvironment(evaluation.provenance(1), provenance=evaluation.provenance, numpy=np, sklearn=sklearn)
        model, candidates = KNeighborsClassifier(n_jobs=1), [{"n_neighbors": 1}]
        assert study._runtime_identity(model, 1, candidates) == study._runtime_identity(model, 1, candidates, _environment=environment)
        args = (model, candidates, X, y, 3, 17, 1)
        assert evaluation._search_identity(*args) == evaluation._search_identity(*args, _environment=environment)


@pytest.mark.parametrize("fault", ["native_count", "library", "numpy_state", "sklearn_state", "io_error"])
def test_fresh_readiness_failure_never_invokes(tmp_path, monkeypatch, fault):
    items, describe = dummy(tmp_path)
    evaluation._initialized_provenance(1)
    before = sorted(threadpoolctl.threadpool_info(), key=lambda row: row["filepath"])
    errors, config = np.geterr(), sklearn.get_config()
    owner = threading.get_ident()
    original = reuse._ReadOnlyEnvironment.__call__
    def bad(self, threads):
        if threading.get_ident() != owner:
            if fault == "io_error":
                raise OSError("readiness I/O failure")
            raise reuse._ReuseUnavailable(fault)
        return original(self, threads)
    monkeypatch.setattr(reuse._ReadOnlyEnvironment, "__call__", bad)
    invoke = lambda *args: pytest.fail("must not invoke after readiness failure")
    if fault == "io_error":
        with pytest.raises(OSError, match="readiness I/O"):
            reuse._try_completed(items, 1, describe=describe, invoke=invoke)
    else:
        assert reuse._try_completed(items, 1, describe=describe, invoke=invoke) is None
    inactive_threads()
    assert sorted(threadpoolctl.threadpool_info(), key=lambda row: row["filepath"]) == before
    assert np.geterr() == errors and sklearn.get_config() == config


@pytest.mark.parametrize("fault", ["count", "inventory", "error_state", "sklearn_config"])
def test_readonly_provider_observes_real_fields_without_normalizing(fault):
    evaluation._initialized_provenance(1)
    with threadpoolctl.threadpool_limits(limits=1):
        original = evaluation.provenance(1)
        changed = deepcopy(original)
        if fault == "count":
            changed["threadpools"][0]["num_threads"] = 2
        elif fault == "inventory":
            changed["threadpools"].append({"filepath": "/new-library.so", "num_threads": 1})
        provider = reuse._ReadOnlyEnvironment(original, provenance=lambda n: changed, numpy=np, sklearn=sklearn)
        if fault == "error_state":
            provider.error_state = {**provider.error_state, "over": "raise"}
        elif fault == "sklearn_config":
            provider.sklearn_config = {**provider.sklearn_config, "assume_finite": not provider.sklearn_config["assume_finite"]}
        with pytest.raises(reuse._ReuseUnavailable):
            provider(1)
        assert original == evaluation.provenance(1)


@pytest.mark.parametrize("outcome", ["none", "late_library", "value_error", "interrupt"])
@pytest.mark.parametrize("stop", [0, 2, 3])
def test_ordered_prefix_stops_and_joins_before_return(tmp_path, monkeypatch, outcome, stop):
    items, describe = dummy(tmp_path, 4)
    evaluation._initialized_provenance(1)
    before = sorted(threadpoolctl.threadpool_info(), key=lambda row: row["filepath"])
    errors, config = np.geterr(), sklearn.get_config()
    owner = threading.get_ident()
    visited, participants, active = [], [], set()
    def invoke(i, environment):
        active.add(i)
        visited.append(i)
        participants.append(threading.get_ident())
        try:
            environment(1)
            if i == stop:
                if outcome == "none":
                    return None
                if outcome == "late_library":
                    raise reuse._ReuseUnavailable("late_inventory")
                if outcome == "value_error":
                    raise ValueError("authoritative invalid record")
                raise KeyboardInterrupt("controlled worker interruption")
            return {"index": i}
        finally:
            active.remove(i)
    if outcome in {"none", "late_library"}:
        assert reuse._try_completed(items, 1, describe=describe, invoke=invoke) is None
    else:
        with pytest.raises(ValueError if outcome == "value_error" else KeyboardInterrupt):
            reuse._try_completed(items, 1, describe=describe, invoke=invoke)
    assert visited == list(range(stop + 1))
    assert len(set(participants)) == 1 and participants[0] != owner
    assert not active
    inactive_threads()
    assert sorted(threadpoolctl.threadpool_info(), key=lambda row: row["filepath"]) == before
    assert np.geterr() == errors and sklearn.get_config() == config


@pytest.mark.parametrize("workers,count", [(2, 2), (8, 34), (0, 1), (True, 1), (9, 1), (1, 0)])
def test_ineligible_gate_precedes_all_prequalification(monkeypatch, workers, count):
    def forbidden(*args, **kwargs):
        pytest.fail("ineligible route reached prequalification or native setup")
    monkeypatch.setattr(evaluation, "_initialized_provenance", forbidden)
    monkeypatch.setattr(reuse, "_ReadOnlyEnvironment", forbidden)
    monkeypatch.setattr(reuse, "_closed_model", forbidden)
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", forbidden)
    assert reuse._try_completed(list(range(count)), workers, describe=forbidden, invoke=forbidden) is None
    inactive_threads()


@pytest.mark.parametrize("policy", ["default", "always", "error"])
def test_warning_grid_uses_unmodified_authoritative_warning_path(tmp_path, monkeypatch, policy):
    items, describe = dummy(tmp_path)
    def warning_description(i):
        _, _, directory, threads = describe(i)
        return NMFClassifier(screening="absolute", random_state=0), {"percentile": [10.]}, directory, threads
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter(policy)
        assert reuse._try_completed(items, 1, describe=warning_description,
                                    invoke=lambda *a: pytest.fail("warning grid entered threads")) is None
        assert not captured
        estimator, grid, _, _ = warning_description(0)
        if policy == "error":
            with pytest.raises(UserWarning, match="Inactive screening"):
                study.screening_grid_diagnostics(estimator, list(study.ParameterGrid(grid)))
        else:
            study.screening_grid_diagnostics(estimator, list(study.ParameterGrid(grid)))
            assert len(captured) == 1 and "Inactive screening" in str(captured[0].message)
    inactive_threads()


@pytest.mark.parametrize("issue", ["missing_fold", "nonfinite_score", "tiny_score", "corrupt", "incompatible", "ownership"])
def test_real_checkpoint_failures_preserve_authority(monkeypatch, completed, issue):
    directories, _, describe, reuse_checkpoint, _, _ = completed
    target = sorted((directories[1] / "folds").glob("*.json"))[0]
    if issue == "missing_fold":
        target.unlink()
    elif issue in {"nonfinite_score", "tiny_score"}:
        envelope = json.loads(target.read_text())
        record = envelope["payload"]
        next(iter(record["neighbors"].values()))["score"] = float("inf") if issue == "nonfinite_score" else 1e-200
        envelope["sha256"] = study._digest(record)
        target.write_text(json.dumps(envelope, allow_nan=True))
    elif issue == "corrupt":
        target.write_text('{"broken":true}')
    elif issue == "incompatible":
        record = study._read_receipt(target)
        record["identity"] = "wrong"
        study._write_receipt(target, record)
    before = [committed(d) for d in directories]
    monkeypatch.setattr(study, "_run_representation_fold", lambda *a, **k: pytest.fail("new fit"))
    visited = []
    def invoke(i, environment):
        visited.append(i)
        return reuse_checkpoint(i, environment)
    if issue == "ownership":
        with study._checkpoint_lock(directories[1]), pytest.raises(RuntimeError, match="Another process"):
            reuse._try_completed([0, 1], 1, describe=describe, invoke=invoke)
    elif issue in {"corrupt", "incompatible"}:
        with pytest.raises(ValueError):
            reuse._try_completed([0, 1], 1, describe=describe, invoke=invoke)
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            assert reuse._try_completed([0, 1], 1, describe=describe, invoke=invoke) is None
            assert not caught
    assert visited == ([] if issue == "missing_fold" else [0, 1])
    assert [committed(d) for d in directories] == before
    inactive_threads()


@pytest.mark.parametrize("module", [DEFAULT, PARAMETER], ids=["default", "parameter"])
def test_scheduler_fallback_uses_whole_original_list_after_return(monkeypatch, module):
    items, calls = [object(), object()], []
    def probe(work, workers, **kwargs):
        assert work == items
        inactive_threads()
        calls.append("probe_finished")
        return None
    class Future:
        def __init__(self, value): self.value = value
        def result(self): return self.value
    class Pool:
        def __init__(self, **kwargs):
            assert calls == ["probe_finished"]
            inactive_threads()
            assert kwargs["mp_context"].get_start_method() == "spawn"
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def submit(self, worker, item):
            calls.append(item)
            return Future(worker(item))
    monkeypatch.setattr(module, "_try_completed", probe)
    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(module, "as_completed", lambda futures: futures)
    monkeypatch.setattr(module, "_worker", lambda item: item)
    assert list(module._completed_or_spawned(items, 2)) == items
    assert calls == ["probe_finished", *items]


def test_default_worker_entry_and_tail_use_explicit_provider(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "train-X.npy", np.ones((30, 2)))
    np.save(cache / "train-a-y.npy", np.arange(30) % 2)
    provider_calls = []
    provider = lambda n: provider_calls.append(n) or evaluation.provenance(n)
    expected = DEFAULT._source_identity(_environment=provider)
    work = (str(tmp_path / "output"), {"job": {"id": "job", "cache": str(cache), "attribute": "a", "seed": 17}},
            {"id": "chunk", "grid": {}}, expected)
    def no_initialize(*a, **k):
        pytest.fail("source guard repeated native initialization")
    monkeypatch.setattr(DEFAULT, "_initialized_provenance", no_initialize)
    monkeypatch.setattr(DEFAULT, "_estimator", lambda job: KNeighborsClassifier())
    def completed_call(*a, **kwargs):
        assert kwargs["_reuse_environment"] is provider
        assert kwargs["_reuse_only"] is True
        return {"candidates": [], "new_representation_fits": 0, "reused_representation_folds": 3}
    monkeypatch.setattr(study, "_calibration_knn_grid", completed_call)
    report = DEFAULT._worker(work, reuse_only=True, _reuse_environment=provider)
    assert report["new_representation_fits"] == 0 and len(provider_calls) == 3


def test_nonlinux_and_custom_classes_keep_original_route(tmp_path, monkeypatch):
    items, describe = dummy(tmp_path)
    with monkeypatch.context() as context:
        context.setattr(reuse.sys, "platform", "win32")
        assert reuse._try_completed(items, 1, describe=lambda i: pytest.fail("nonlinux described"), invoke=None) is None
    class Custom(KNeighborsClassifier):
        pass
    def custom(i):
        _, grid, directory, threads = describe(i)
        return Custom(), grid, directory, threads
    assert reuse._try_completed(items, 1, describe=custom, invoke=lambda *a: pytest.fail("custom thread work")) is None


def test_invalid_grid_hint_preserves_authoritative_invalid_data_precedence(tmp_path, monkeypatch):
    items, describe = dummy(tmp_path)
    invalid_grid = {"n_neighbors": 5}
    def invalid(i):
        estimator, _, directory, threads = describe(i)
        return estimator, invalid_grid, directory, threads
    assert reuse._try_completed(items, 1, describe=invalid,
                               invoke=lambda *a: pytest.fail("invalid hint entered threads")) is None
    X = np.ones((30, 3))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        study.calibration_knn_grid(KNeighborsClassifier(n_jobs=1), invalid_grid, X,
                                  np.arange(30) % 2, checkpoint_dir=tmp_path / "ordinary", threads=1)
    inactive_threads()


def test_tiny_saved_scores_defer_without_changing_strict_underflow(monkeypatch, completed):
    directories, _, describe, invoke, X, y = completed
    for index, path in enumerate(sorted((directories[1] / "folds").glob("*.json"))):
        record = study._read_receipt(path)
        record["neighbors"]["1"]["score"] = 1e-200 if index == 1 else 0.0
        study._write_receipt(path, record)
    before = [committed(d) for d in directories]
    original = np.geterr()
    with np.errstate(all="raise"):
        strict = np.geterr()
        assert reuse._try_completed([0, 1], 1, describe=describe, invoke=invoke) is None
        assert np.geterr() == strict
        # The original worker remains authoritative for a checksum-valid record
        # outside the quiet eligibility. Its direct strict behavior is retained.
        with pytest.raises(FloatingPointError, match="underflow"):
            study.calibration_knn_grid(KNeighborsClassifier(n_jobs=1), {"n_neighbors": [1, 3]}, X, y,
                                       cv=3, seed=17, threads=1, checkpoint_dir=directories[1])
        assert np.geterr() == strict
    assert np.geterr() == original and [committed(d) for d in directories] == before
    inactive_threads()


@pytest.mark.parametrize("position", [0, 1], ids=["first", "last"])
@pytest.mark.parametrize("unrelated", [False, True])
def test_declared_missing_fold_bypasses_before_native_setup(monkeypatch, completed, position, unrelated):
    directories, _, describe, _, _, _ = completed
    missing = sorted((directories[position] / "folds").glob("*.json"))[0]
    missing.unlink()
    if unrelated:
        extra = directories[position] / "folds" / ("f" * 64 + ".json")
        assert not extra.exists()
        extra.write_text("Unrelated files cannot replace the missing declared key.")
    before = [committed(path) for path in directories]
    def forbidden(*args, **kwargs):
        pytest.fail("negative hint reached native warmup, provider or reuse work")
    monkeypatch.setattr(evaluation, "_initialized_provenance", forbidden)
    monkeypatch.setattr(reuse, "_ReadOnlyEnvironment", forbidden)
    monkeypatch.setattr(reuse, "_ReadOnlyEnvironment", forbidden)
    monkeypatch.setattr(reuse, "ThreadPoolExecutor", forbidden)
    assert reuse._try_completed([0, 1], 1, describe=describe, invoke=forbidden) is None

    assert [committed(path) for path in directories] == before
    inactive_threads()


@pytest.mark.parametrize("issue", ["json", "envelope", "payload", "empty", "duplicate", "traversal",
                                  "uppercase", "record_state", "report_state", "checksum_shape", "io_error"])
def test_malformed_presence_hint_only_defers(tmp_path, monkeypatch, issue):
    items, describe = dummy(tmp_path)
    path = describe(0)[2] / "report.json"
    value = json.loads(path.read_text())
    if issue == "json":
        path.write_text("invalid JSON")
    elif issue == "io_error":
        original = Path.read_text
        def unreadable(self, *args, **kwargs):
            if self == path:
                raise PermissionError("controlled report read failure")
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", unreadable)
    else:
        if issue == "envelope":
            value = []
        elif issue == "payload":
            value["payload"] = []
        elif issue == "empty":
            value["payload"]["representation_folds"] = []
        elif issue == "duplicate":
            value["payload"]["representation_folds"] *= 2
        elif issue == "traversal":
            value["payload"]["representation_folds"][0]["key"] = "../" + "a" * 61
        elif issue == "uppercase":
            value["payload"]["representation_folds"][0]["key"] = "A" * 64
        elif issue == "record_state":
            value["payload"]["representation_folds"][0]["state"] = "running"
        elif issue == "report_state":
            value["payload"]["state"] = "running"
        else:
            value["sha256"] = 0
        path.write_text(json.dumps(value))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail("malformed hint reached native setup or reuse work")
    monkeypatch.setattr(evaluation, "_initialized_provenance", forbidden)
    assert reuse._try_completed(items, 1, describe=describe, invoke=forbidden) is None
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before
    inactive_threads()


@pytest.mark.parametrize("bad_grid", [False, True])
def test_bad_hint_preserves_invalid_data_precedence(monkeypatch, completed, bad_grid):
    directories, _, describe, _, X, y = completed
    (directories[0] / "report.json").write_text("malformed derived report")
    grid = {"n_neighbors": 5} if bad_grid else {"n_neighbors": [1, 3]}
    def description(i):
        estimator, _, directory, threads = describe(i)
        return estimator, grid, directory, threads
    assert reuse._try_completed([0, 1], 1, describe=description,
                                invoke=lambda *args: pytest.fail("hint entered reuse")) is None
    invalid = X.copy()
    invalid[0, 0] = np.nan
    before = [committed(path) for path in directories]
    with pytest.raises(ValueError, match="NaN"):
        study.calibration_knn_grid(KNeighborsClassifier(n_jobs=1), grid, invalid, y,
                                   cv=3, seed=17, threads=1, checkpoint_dir=directories[0])
    assert [committed(path) for path in directories] == before
    inactive_threads()


@pytest.mark.parametrize("issue", ["missing_real_fold", "incompatible_identity"])
def test_fabricated_existing_keys_cannot_accept_checkpoint(monkeypatch, completed, issue):
    directories, _, describe, invoke, _, _ = completed
    if issue == "missing_real_fold":
        sorted((directories[1] / "folds").glob("*.json"))[0].unlink()
    else:
        identity_path = directories[1] / "identity.json"
        identity = study._read_receipt(identity_path)
        identity["scoring"] = "fabricated-incompatible-setting"
        study._write_receipt(identity_path, identity)
    for directory in directories:
        fake = "a" * 64
        path = directory / "folds" / (fake + ".json")
        assert not path.exists()
        path.write_text("Present but unrelated to every real expected fold.")
        payload = {"state": "cv_completed", "representation_folds": [{"key": fake, "state": "completed"}]}
        (directory / "report.json").write_text(json.dumps({"payload": payload, "sha256": "0" * 64}))
        assert reuse._declared_folds_present(directory)
    before = [committed(path) for path in directories]
    monkeypatch.setattr(study, "_run_representation_fold", lambda *args, **kwargs: pytest.fail("hint accepted new work"))
    visited = []
    def actual(i, environment):
        visited.append(i)
        return invoke(i, environment)
    if issue == "missing_real_fold":
        assert reuse._try_completed([0, 1], 1, describe=describe, invoke=actual) is None
    else:
        with pytest.raises(ValueError, match="incompatible"):
            reuse._try_completed([0, 1], 1, describe=describe, invoke=actual)
    assert visited == [0, 1]
    assert [committed(path) for path in directories] == before
    inactive_threads()


def test_fold_removed_after_hint_is_checked_by_locked_core(monkeypatch, completed):
    directories, _, describe, invoke, _, _ = completed
    original = reuse._declared_folds_present
    preserved = []
    def race(directory):
        answer = original(directory)
        assert answer
        if directory == directories[1]:
            sorted((directory / "folds").glob("*.json"))[0].unlink()
            preserved.extend(committed(path) for path in directories)
        return answer
    monkeypatch.setattr(reuse, "_declared_folds_present", race)
    monkeypatch.setattr(study, "_run_representation_fold", lambda *args, **kwargs: pytest.fail("new fit after hint race"))
    visited = []
    def actual(i, environment):
        visited.append(i)
        return invoke(i, environment)
    assert reuse._try_completed([0, 1], 1, describe=describe, invoke=actual) is None
    assert visited == [0, 1] and [committed(path) for path in directories] == preserved
    inactive_threads()
    for directory in directories:
        with study._checkpoint_lock(directory):
            pass


def test_incorrect_derived_checksum_does_not_replace_core_validation(monkeypatch, completed):
    directories, original, describe, invoke, _, _ = completed
    for directory in directories:
        report = directory / "report.json"
        value = json.loads(report.read_text())
        value["sha256"] = "0" * 64
        report.write_text(json.dumps(value))
        assert reuse._declared_folds_present(directory)
    before = [committed(path) for path in directories]
    monkeypatch.setattr(study, "_run_representation_fold", lambda *args, **kwargs: pytest.fail("new fit"))
    result = reuse._try_completed([0, 1], 1, describe=describe, invoke=invoke)
    assert [science(report) for report in result] == [science(report) for report in original]
    assert [committed(path) for path in directories] == before
    for directory in directories:
        assert study._read_receipt(directory / "report.json")["state"] == "cv_completed"
    inactive_threads()


def inherited_pool(observations):
    """Force copied caller context even on runtimes whose default is empty."""
    class Pool(RealThreadPoolExecutor):
        def submit(self, function, *args, **kwargs):
            inherited = copy_context()
            def observe():
                observations.append(("before", CONTEXT_SENTINEL.get(), np.geterr()))
                try:
                    return function(*args, **kwargs)
                finally:
                    observations.append(("after", CONTEXT_SENTINEL.get(), np.geterr()))
            return super().submit(inherited.run, observe)
    return Pool


def test_explicit_empty_context_overrides_forced_inheritance_and_negative_control(tmp_path, monkeypatch):
    items, describe = dummy(tmp_path, 1)
    token = CONTEXT_SENTINEL.set("caller")
    owner = threading.get_ident()
    errors, configuration = np.geterr(), sklearn.get_config()
    try:
        # The copied-context control deliberately removes isolation. It must
        # expose the sentinel, proving the forced-inheritance fixture is active.
        for factory, expected in ((copy_context, "caller"), (Context, "empty")):
            outer, observed = [], []
            with monkeypatch.context() as context:
                context.setattr(reuse, "ThreadPoolExecutor", inherited_pool(outer))
                context.setattr(reuse, "Context", factory)
                def invoke(i, environment):
                    environment(1)
                    observed.append((CONTEXT_SENTINEL.get(), threading.get_ident()))
                    CONTEXT_SENTINEL.set("worker")
                    return {"index": i}
                assert reuse._try_completed(items, 1, describe=describe, invoke=invoke) == [{"index": 0}]
            assert len(observed) == 1
            assert observed[0][0] == expected and observed[0][1] != owner
            assert [(phase, value) for phase, value, _ in outer] == [("before", "caller"), ("after", "caller")]
            assert CONTEXT_SENTINEL.get() == "caller"
            assert np.geterr() == errors and sklearn.get_config() == configuration
            inactive_threads()
    finally:
        CONTEXT_SENTINEL.reset(token)


def test_empty_context_retains_real_strict_state_readiness_fallback(tmp_path, monkeypatch):
    items, describe = dummy(tmp_path, 1)
    evaluation._initialized_provenance(1)
    pools = sorted(threadpoolctl.threadpool_info(), key=lambda row: row["filepath"])
    initial_errors, configuration = np.geterr(), sklearn.get_config()
    owner, outer, observed = threading.get_ident(), [], []
    original = reuse._ReadOnlyEnvironment.__call__
    def observe(self, threads):
        if threading.get_ident() != owner:
            observed.append((CONTEXT_SENTINEL.get(), np.geterr()))
        return original(self, threads)
    monkeypatch.setattr(reuse, "ThreadPoolExecutor", inherited_pool(outer))
    monkeypatch.setattr(reuse._ReadOnlyEnvironment, "__call__", observe)
    token = CONTEXT_SENTINEL.set("strict-caller")
    try:
        with np.errstate(all="raise"):
            strict = np.geterr()
            assert reuse._try_completed(items, 1, describe=describe,
                                        invoke=lambda *args: pytest.fail("changed caller state entered reuse")) is None
            assert np.geterr() == strict
        assert observed and all(value == "empty" and state != strict for value, state in observed)
        assert [(phase, value) for phase, value, _ in outer] == [("before", "strict-caller"), ("after", "strict-caller")]
        assert CONTEXT_SENTINEL.get() == "strict-caller"
        assert np.geterr() == initial_errors and sklearn.get_config() == configuration
        assert sorted(threadpoolctl.threadpool_info(), key=lambda row: row["filepath"]) == pools
        inactive_threads()
    finally:
        CONTEXT_SENTINEL.reset(token)


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
@pytest.mark.parametrize("module", [DEFAULT, PARAMETER], ids=["default", "parameter"])
def test_scheduler_platform_gate_preserves_original_dispatch(monkeypatch, platform, module):
    work, calls = [object(), object()], []
    original_probe = module._try_completed
    def probe(items, workers, **kwargs):
        assert items == work
        calls.append("probe")
        if platform != "linux":
            def forbidden(*args, **kwargs):
                pytest.fail("non-Linux helper reached prequalification or work")
            assert original_probe(items, workers, describe=forbidden, invoke=forbidden) is None
        return None
    class Future:
        def __init__(self, result): self.value = result
        def result(self): return self.value
    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def submit(self, function, item):
            calls.append(item)
            return Future(function(item))
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module, "_try_completed", probe)
    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(module, "as_completed", lambda futures: futures)
    monkeypatch.setattr(module, "_worker", lambda item: item)
    assert list(module._completed_or_spawned(work, 1)) == work
    assert calls == ["probe", *work]
