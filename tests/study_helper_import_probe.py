"""Fresh-process assertions for study-helper ownership; no work on import."""

from __future__ import annotations

from contextlib import chdir
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import threading
from unittest.mock import patch


FILES = (
    "examples/celeba_parameter_study.py", "examples/recovery_store.py",
    "examples/numeric_codec.py", "scripts/celeba_default_study.py",
    "scripts/fashion_mnist_study.py", "scripts/profile_celeba_stages.py",
    "scripts/experimental_nmf.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


def rejected(call, error_type, *, restart=True):
    try:
        call()
    except error_type as error:
        if restart:
            assert "restart" in str(error).lower(), str(error)
        return error
    raise AssertionError("The operation accepted an incompatible loaded helper")


def changed(path):
    path.write_bytes(path.read_bytes() + b"\n# Owned test copy changed after import.\n")


def foreign(source, work, relative):
    target = work / "other" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / relative, target)
    assert sha(target) == sha(source / relative)
    return load(target.stem, target)


def alter(helper, path, state):
    if state == "missing_origin":
        helper.__file__ = None
    elif state == "missing_pin":
        del helper._SOURCE_SHA256_AT_IMPORT
    elif state == "stale":
        changed(path)
    elif state == "empty":
        path.write_bytes(b"")
    elif state == "removed":
        path.unlink()
    elif state == "large":
        path.write_bytes(path.read_bytes() + b"\n# " + b"x" * (1024 * 1024 + 1) + b"\n")
    else:
        assert state == "same"


def model_data():
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    X = np.array([[0., 1.], [1., 2.], [3., 1.], [4., 2.]])
    y = np.array([0, 0, 1, 1])
    return KNeighborsClassifier(n_neighbors=1, n_jobs=1).fit(X, y), X, y


def recovery_canonical(source):
    owner = importlib.import_module("recovery_store")
    first = importlib.import_module("examples.celeba_parameter_study")
    second = load("another_study_consumer", source / "examples/celeba_parameter_study.py")
    assert first.recovery_store is second.recovery_store is owner
    lock = owner._FINAL_AGGREGATION_LOCKS_BY_PID[os.getpid()]
    blocked, entered = threading.Event(), threading.Event()
    results = []

    class ObservedLock:
        def acquire(self, *args, **kwargs):
            acquired = lock.acquire(*args, **kwargs)
            if not acquired:
                blocked.set()
            return acquired

        def release(self):
            lock.release()

    def contender():
        with second.recovery_store._final_aggregation_guard() as waited:
            results.append(waited)
            entered.set()

    worker = threading.Thread(target=contender, daemon=True)
    # Signal an actual failed nonblocking acquire, avoiding a scheduling race
    # between a thread's startup signal and its first attempt to take the lock.
    with patch.dict(owner._FINAL_AGGREGATION_LOCKS_BY_PID, {os.getpid(): ObservedLock()}):
        with first.recovery_store._final_aggregation_guard() as waited:
            assert waited is False
            with second.recovery_store._final_aggregation_guard() as nested:
                assert nested is False
            worker.start()
            assert blocked.wait(5)
            assert not entered.is_set()
        worker.join(5)
    assert not worker.is_alive() and entered.is_set() and results == [True]
    assert owner._FINAL_AGGREGATION_LOCKS_BY_PID[os.getpid()] is lock
    try:
        with owner._final_aggregation_guard():
            raise ValueError("owned failed aggregation")
    except ValueError:
        pass
    with owner._final_aggregation_guard() as waited:
        assert waited is False


def recovery_conflict(source, work, state):
    path = source / "examples/recovery_store.py"
    owner = (foreign(source, work, "examples/recovery_store.py") if state == "foreign"
             else load("recovery_store", path))
    if state != "foreign":
        alter(owner, path, state)
    lock = owner._FINAL_AGGREGATION_LOCKS_BY_PID[os.getpid()]
    with owner._final_aggregation_guard():
        rejected(lambda: importlib.import_module("celeba_parameter_study"), ImportError)
        assert "celeba_parameter_study" not in sys.modules
        assert sys.modules["recovery_store"] is owner
        assert owner._FINAL_AGGREGATION_LOCKS_BY_PID[os.getpid()] is lock
    # A caller may remove its conflicting import and retry from the real checkout.
    del sys.modules["recovery_store"]
    current = importlib.import_module("celeba_parameter_study")
    assert current.recovery_store is not owner
    assert Path(current.recovery_store.__file__).resolve() == path


def recovery_terminal(work):
    watched = {"numpy", "scipy", "sklearn", "joblib", "threadpoolctl", "numeric_codec"}
    assert watched.isdisjoint(sys.modules)
    attempts = []

    class ObserveImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] in watched:
                attempts.append(fullname)
            return None

    # Record even optional lookups whose ImportError the helper might catch.
    with patch.object(sys, "meta_path", [ObserveImports(), *sys.meta_path]):
        recovery = importlib.import_module("recovery_store")
        store = recovery.RecoveryStore(work / "store", {"case": "terminal"})
        assert store.commit({"evaluation": {"status": "failed"}})["kind"] == "terminal"
        assert store.load_outcome()["manifest"]["kind"] == "terminal"
    assert not attempts, attempts
    assert watched.isdisjoint(sys.modules)


def recovery_admission(source, work, name):
    study = importlib.import_module("celeba_parameter_study")
    helper = importlib.import_module(name)
    path = source / "examples" / (name + ".py")
    changed(path)
    # The observer still reports fresh facts; admission rejects stale loaded code.
    facts = study.recovery_store.active_runtime_identity(1)
    assert facts["recovery_sources"][path.name] == sha(path)
    _, X, y = model_data()
    from sklearn.neighbors import KNeighborsClassifier
    from supervised_nmf.evaluation import data_fingerprint
    output = work / "fit"
    output.mkdir()
    result = {"attribute": "test", "method": "raw_knn", "evaluation_identity": "evaluation",
              "run_identity": "run", "selected_parameters": {"n_neighbors": 1},
              "final_fit_seconds": 0.}
    manifest = {"splits": {
        "train": {"attributes": {"test": {"data_sha256": data_fingerprint(X, y)}}},
        "validation": {"image_ids": [f"v{i}" for i in range(len(y))]},
    }}
    configuration = {"script_sha256": study._source_hash(),
                     "source_sha256": study._package_source_hashes(), "final_fit_recovery": "all",
                     "data_manifest_sha256": study._digest(manifest)}
    fit_started = []
    original = study.recovery_store._hook

    def observe(event, details=None):
        if event == "fit-start":
            fit_started.append(True)
        return original(event, details)

    with patch.object(study.recovery_store, "_hook", observe):
        rejected(lambda: study._recoverable_final_fit(
            result, output, output / "result.json", KNeighborsClassifier(n_neighbors=1, n_jobs=1),
            {"n_neighbors": 1}, X, y, X, y, manifest, 1, configuration, {}),
            (ImportError, study.recovery_store.RecoveryError))
    assert fit_started == []
    assert not (output / ".fit-recovery/current.json").exists()
    assert not (output / "result.json").exists()
    assert sys.modules[name] is helper


def codec_case(source, work, operation, state):
    recovery = importlib.import_module("recovery_store")
    assert "numeric_codec" not in sys.modules
    path = source / "examples/numeric_codec.py"
    helper = importlib.import_module("numeric_codec")
    model, X, y = model_data()
    store = recovery.RecoveryStore(work / "store", {"case": "model"})
    if operation == "restore":
        store.commit({"purpose": "restore control"}, model=model)
        outcome = store.load_outcome()
        before = {p.relative_to(store.directory): sha(p) for p in store.directory.rglob("*") if p.is_file()}
    if state == "foreign":
        helper = foreign(source, work, "examples/numeric_codec.py")
    else:
        alter(helper, path, state)
    method = "dump_model" if operation == "commit" else "load_model"
    calls = []
    original = getattr(helper, method)

    def observe(*args, **kwargs):
        calls.append(method)
        return original(*args, **kwargs)

    action = (lambda: store.commit({"purpose": "model"}, model=model)) if operation == "commit" else (
        lambda: store.restore_model(outcome))
    with patch.object(helper, method, observe):
        if state == "same":
            value = action()
            if operation == "restore":
                restored, _, _ = value
                assert (restored.predict(X) == y).all()
            assert calls == [method]
        else:
            expected = recovery.RecoveryError if operation == "commit" else (
                FileNotFoundError if state == "removed" else ImportError)
            error = rejected(action, expected, restart=state != "removed")
            if operation == "commit":
                assert isinstance(error.__cause__, FileNotFoundError if state == "removed" else ImportError)
                assert not store.pointer.exists()
                assert not list(store.directory.glob("generations/*/payload"))
            assert calls == []
    if operation == "restore":
        assert {p.relative_to(store.directory): sha(p) for p in store.directory.rglob("*") if p.is_file()} == before
    assert sys.modules["numeric_codec"] is helper
    if state == "missing_pin":
        assert not hasattr(helper, "_SOURCE_SHA256_AT_IMPORT")


def profile_case(source, work, consumer, state):
    path = source / "scripts/profile_celeba_stages.py"
    helper = (foreign(source, work, "scripts/profile_celeba_stages.py") if state == "foreign"
              else importlib.import_module("profile_celeba_stages"))
    name = "celeba_default_study" if consumer == "default" else "fashion_mnist_study"
    if state in ("foreign", "missing_origin"):
        if state == "missing_origin":
            alter(helper, path, state)
        rejected(lambda: importlib.import_module(name), ImportError)
        assert name not in sys.modules
    else:
        current = importlib.import_module(name)
        assert current._runtime_identity is helper._runtime_identity
        assert current.profile_selected_estimator is helper.profile_selected_estimator
        identity = current._source_identity if consumer == "default" else current._sources
        expected = identity()
        assert "experimental_nmf" not in sys.modules
        alter(helper, path, state)
        if state == "same":
            assert identity() == expected
        else:
            rejected(identity, ValueError)
            rejected(lambda: current._check_source(expected), ValueError)
        assert "experimental_nmf" not in sys.modules
    assert sys.modules["profile_celeba_stages"] is helper


def experimental_routes(source, work, state):
    fashion = importlib.import_module("fashion_mnist_study")
    assert "experimental_nmf" not in sys.modules
    helper = (foreign(source, work, "scripts/experimental_nmf.py") if state == "foreign"
              else importlib.import_module("experimental_nmf"))
    import numpy as np
    from sklearn.decomposition import NMF
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from supervised_nmf import BinaryScreening
    from supervised_nmf.evaluation import evaluate_classifier
    X = np.random.RandomState(91).uniform(.3, 1.5, (16, 6))
    y = np.arange(16) % 2
    model = Pipeline([
        ("screen", BinaryScreening(score_threshold=0.)),
        ("nmf", NMF(n_components=2, solver="mu", init="random", random_state=8, max_iter=8, tol=0)),
        ("knn", KNeighborsClassifier(n_neighbors=1, n_jobs=1)),
    ]).fit(X, y)
    reference = evaluate_classifier(model, X[:8], y[:8], threads=1)
    operations = [
        lambda: fashion._estimator({"method": "screened_anls_bpp_absolute", "seed": 8, "max_iter": 8, "tol": 1e-4}),
        lambda: fashion._profile(model, X, y, X[:8], y[:8], reference=reference, sample_ids=None),
        lambda: fashion._model_diagnostics(model),
        lambda: fashion._model_nonconverged(model, []),
        lambda: fashion._training_stationarity(model, X),
        lambda: fashion.evaluate(work / "evaluation", {"evaluation_order": [], "arrays": {}, "plans": []}),
    ]
    # Dataset/CV behavior is covered by the current-workflow tests; this empty
    # aggregate exercises evaluate's helper boundary without another calibration.
    with patch.object(fashion, "aggregate_cv", return_value={"jobs": {}}):
        for operation in operations:
            if state == "same":
                operation()
            else:
                rejected(operation, ImportError)
    assert sys.modules["experimental_nmf"] is helper


def experimental_admission(source, work, state):
    fashion = importlib.import_module("fashion_mnist_study")
    assert "experimental_nmf" not in sys.modules
    expected = fashion._sources()
    assert "experimental_nmf" not in sys.modules
    path = source / "scripts/experimental_nmf.py"
    helper = (foreign(source, work, "scripts/experimental_nmf.py") if state == "foreign"
              else importlib.import_module("experimental_nmf"))
    if state != "foreign":
        alter(helper, path, state)
    estimator = lambda: fashion._estimator({"method": "screened_anls_bpp_absolute", "seed": 8,
                                            "max_iter": 8, "tol": 1e-4})
    if state == "same":
        assert fashion._sources()["runner_source_sha256"] == expected["runner_source_sha256"]
        assert type(estimator().named_steps["nmf"]) is helper.ANLSBPP
    else:
        rejected(fashion._sources, ValueError)
        rejected(lambda: fashion._check_source(expected), ValueError)
        rejected(estimator, ImportError)
    assert sys.modules["experimental_nmf"] is helper
    if state == "missing_pin":
        assert not hasattr(helper, "_SOURCE_SHA256_AT_IMPORT")


def origin_aliases(source, work, route):
    class OriginString(str):
        pass

    if route == "codec":
        owner = importlib.import_module("recovery_store")
        path = source / "examples/numeric_codec.py"
        name = "numeric_codec"
        operation = owner._numeric_codec_module
    else:
        owner = importlib.import_module("fashion_mnist_study")
        path = source / "scripts/experimental_nmf.py"
        name = "experimental_nmf"
        operation = owner._experimental_module
    assert name not in sys.modules
    helper = importlib.import_module(name)
    with chdir(work):
        helper.__file__ = os.path.relpath(path)
        assert operation() is helper
    aliases = [str(path), str(path.parent) + os.sep + "." + os.sep + path.name,
               path, OriginString(str(path))]
    if os.name == "nt":
        aliases.append(str(path).swapcase())
    for alias in aliases:
        helper.__file__ = alias
        assert operation() is helper
    helper.__file__ = str(path)
    assert sys.modules[name] is helper


def main(root, work, case):
    before = {name: sha(root / name) for name in FILES}
    source = work / "source"
    for relative in FILES:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    sys.path[:0] = [str(root / "src"), str(source), str(source / "examples"), str(source / "scripts")]
    try:
        if case == "recovery-canonical":
            recovery_canonical(source)
        elif case == "recovery-terminal":
            recovery_terminal(work)
        elif case.startswith("recovery-conflict-"):
            recovery_conflict(source, work, case.removeprefix("recovery-conflict-"))
        elif case.startswith("recovery-admission-"):
            recovery_admission(source, work, case.removeprefix("recovery-admission-"))
        elif case.startswith("codec-"):
            _, operation, state = case.split("-")
            codec_case(source, work, operation, state)
        elif case.startswith("profile-"):
            _, consumer, state = case.split("-")
            profile_case(source, work, consumer, state)
        elif case.startswith("experimental-routes-"):
            experimental_routes(source, work, case.removeprefix("experimental-routes-"))
        elif case.startswith("experimental-admission-"):
            experimental_admission(source, work, case.removeprefix("experimental-admission-"))
        elif case.startswith("origins-"):
            origin_aliases(source, work, case.removeprefix("origins-"))
        else:
            raise AssertionError(case)
    finally:
        assert {name: sha(root / name) for name in FILES} == before
    print("completed: " + case)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3])
