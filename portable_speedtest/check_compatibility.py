"""Validate the bundled sNMF source in the active environment without installing it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import warnings


ROOT = Path(__file__).resolve().parent
TARGET_VERSIONS = {
    "numpy": "2.2.6", "scipy": "1.15.3", "scikit-learn": "1.7.0",
    "threadpoolctl": "3.6.0", "joblib": "1.5.1",
}


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def array_hash(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def run_checks(report, expect_target_versions=False):
    from benchmark import THREAD_ENVIRONMENT, verify_bundle
    for key, value in THREAD_ENVIRONMENT.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True

    def check(name, operation):
        try:
            details = operation()
        except Exception as error:
            report["checks"].append({"name": name, "passed": False,
                                     "error": f"{type(error).__name__}: {error}"})
            raise
        report["checks"].append({"name": name, "passed": True, "details": details})
        return details

    def integrity():
        protocol, manifest_hash, protocol_hash = verify_bundle()
        report.update(manifest_sha256=manifest_hash, protocol_sha256=protocol_hash)
        return protocol

    protocol = check("frozen_bundle_integrity", integrity)
    assert_true(sys.version_info >= (3, 10), "The compatibility runner requires Python 3.10 or newer.")
    versions = {}
    for name in TARGET_VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report["environment"]["versions"] = versions
    report["target_versions"] = TARGET_VERSIONS
    report["matches_supplied_target_versions"] = versions == TARGET_VERSIONS

    def dependencies():
        missing = [name for name, version in versions.items() if version is None]
        assert_true(not missing, f"Missing dependencies in the active environment: {missing}")
        if expect_target_versions:
            differences = {name: {"expected": TARGET_VERSIONS[name], "actual": version}
                           for name, version in versions.items() if version != TARGET_VERSIONS[name]}
            assert_true(not differences, f"Active versions differ from the supplied target pins: {differences}")
        return {"exact_target_versions_required": expect_target_versions, "versions": versions}

    check("dependency_versions", dependencies)
    import numpy as np
    import scipy
    from scipy.stats import ttest_ind
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_info, threadpool_limits

    def source_import():
        package_root = (ROOT / "package").resolve()
        sys.path.insert(0, str(package_root))
        import supervised_nmf
        from supervised_nmf.datasets import make_original_simulation
        modules = {name: str(Path(module.__file__).resolve())
                   for name, module in sys.modules.items()
                   if (name == "supervised_nmf" or name.startswith("supervised_nmf."))
                   and getattr(module, "__file__", None)}
        assert_true(all(Path(path).is_relative_to(package_root) for path in modules.values()),
                    "An installed or external supervised_nmf module was loaded instead of the bundled source.")
        report["source"] = {"version": supervised_nmf.__version__, "author": supervised_nmf.__author__, "modules": modules}
        return {"bundled_source_only": True, "package_version": supervised_nmf.__version__}

    check("bundled_source_import", source_import)
    from supervised_nmf import BinaryScreening, NMFClassifier
    from supervised_nmf.datasets import make_original_simulation

    data = make_original_simulation(n_samples=256, n_features=64, signal_ratio=0.5, random_state=543)
    rng = np.random.RandomState(544)
    groups = [rng.permutation(np.flatnonzero(data.target == label)) for label in (0, 1)]
    train = rng.permutation(np.concatenate([group[:96] for group in groups]))
    heldout = rng.permutation(np.concatenate([group[96:] for group in groups]))
    X_train, y_train = data.data[train], data.target[train]
    X_test = data.data[heldout]
    report["data"] = {"generator": "make_original_simulation", "random_state": 543, "split_random_state": 544,
                      "training_shape": list(X_train.shape), "heldout_shape": list(X_test.shape),
                      "training_sha256": array_hash(X_train), "heldout_sha256": array_hash(X_test)}

    def screening_reference():
        # scipy's signed statistic is first-group minus second-group; sNMF is class1 minus class0.
        values = X_train.astype(np.float64)
        selector = BinaryScreening(screening="absolute", score_threshold=2.0).fit(values, y_train)
        reference = ttest_ind(values[y_train == 1], values[y_train == 0], axis=0, equal_var=True).statistic
        np.testing.assert_allclose(selector.t_scores_, reference, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(selector.get_support(), np.abs(reference) >= 2.0)
        # Reusing an exactly computed cutoff proves inclusive >= behavior independently of SciPy rounding.
        boundary_index = int(np.argmax(np.abs(selector.t_scores_)))
        boundary = float(abs(selector.t_scores_[boundary_index]))
        inclusive = BinaryScreening(screening="absolute", score_threshold=boundary).fit(values, y_train)
        assert_true(bool(inclusive.get_support()[boundary_index]), "The exact absolute cutoff boundary was excluded.")
        with_constant = np.column_stack([values, np.ones(values.shape[0])])
        zero_cutoff = BinaryScreening(screening="absolute", score_threshold=0).fit(with_constant, y_train)
        assert_true(not zero_cutoff.get_support()[-1], "A globally constant feature survived screening.")
        return {"reference": "scipy.stats.ttest_ind(equal_var=True), class1 minus class0",
                "maximum_absolute_difference": float(np.max(np.abs(selector.t_scores_ - reference))),
                "selected_features": int(selector.get_support().sum()), "inclusive_boundary": boundary}

    with threadpool_limits(limits=1):
        report["environment"]["thread_pools"] = threadpool_info()
        report["environment"]["imported_versions"] = {"numpy": np.__version__, "scipy": scipy.__version__,
                                                         "scikit-learn": sklearn.__version__}
        check("absolute_screening_matches_pooled_t", screening_reference)

        def exercise(dtype):
            train_values, test_values = X_train.astype(dtype), X_test.astype(dtype)
            models, outputs, caught_messages = [], [], []
            for repetition in range(2):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = NMFClassifier(**protocol["model"]).fit(train_values, y_train)
                    before = {"selector_scores": model.selector_.t_scores_.copy(),
                              "selected": model.selected_features_.copy(),
                              "components": model.decomposition_.components_.copy(),
                              "samples": model.n_samples_seen_}
                    coordinates = model.transform(test_values)
                    labels, probabilities = model.predict_with_probabilities(test_values)
                convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
                caught_messages.extend(str(item.message) for item in caught)
                assert_true(not convergence and model.converged_ and model.last_transform_converged_,
                            f"CD fit or held-out projection did not converge: {convergence}")
                assert_true(model.solver == "cd" and model.decomposition_.solver == "cd" and model.screening == "absolute",
                            "The model did not use CD with absolute screening.")
                assert_true(np.isfinite(model.training_coordinates_).all() and np.isfinite(coordinates).all()
                            and np.isfinite(model.components_).all() and np.isfinite(probabilities).all()
                            and np.isfinite(model.reconstruction_err_), "Nonfinite fit or prediction output.")
                assert_true(coordinates.shape == (len(test_values), protocol["model"]["n_components"]),
                            "Held-out transformed coordinates have the wrong shape.")
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)
                assert_true(set(np.unique(labels)) <= set(np.unique(y_train)), "Prediction labels are outside training classes.")
                np.testing.assert_array_equal(before["selector_scores"], model.selector_.t_scores_)
                np.testing.assert_array_equal(before["selected"], model.selected_features_)
                np.testing.assert_array_equal(before["components"], model.decomposition_.components_)
                assert_true(before["samples"] == model.n_samples_seen_, "Held-out prediction refitted training observations.")
                models.append(model)
                outputs.append((coordinates, labels, probabilities))
            np.testing.assert_array_equal(models[0].selected_features_, models[1].selected_features_)
            np.testing.assert_allclose(models[0].components_, models[1].components_, rtol=1e-12, atol=1e-12)
            for first, second in zip(outputs[0], outputs[1]):
                np.testing.assert_array_equal(first, second)
            assert_true(models[0].n_iter_ == models[1].n_iter_, "Repeated deterministic fits used different iteration counts.")
            return {"dtype": str(np.dtype(dtype)), "converged": True, "repeated_fit_reproducible": True,
                    "heldout_did_not_refit": True, "selected_features": int(models[0].n_features_selected_),
                    "iterations": int(models[0].n_iter_), "reconstruction_error": float(models[0].reconstruction_err_),
                    "prediction_sha256": array_hash(outputs[0][1]), "warnings": caught_messages}

        check("cd_absolute_float32_fit_predict_reproduce", lambda: exercise(np.float32))
        check("cd_absolute_float64_fit_predict_reproduce", lambda: exercise(np.float64))
    report["limitations"] = [
        "This validates the recorded interpreter, OS, binaries and small deterministic CD/absolute cases only.",
        "Matching library version numbers on Windows/Python 3.12 does not validate Ubuntu/Python 3.10 or its BLAS builds.",
        "Run this checker on the actual Ubuntu target before accepting that environment; it changes no installed packages.",
        "Package metadata retains Python >=3.11; Python 3.10 direct-source execution is provisional until tested on the target.",
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON report path; existing files are never overwritten")
    parser.add_argument("--expect-target-versions", action="store_true", help="Require the exact supplied target numerical-library versions")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output.resolve() if args.output else ROOT / "results" / f"compatibility-{timestamp}.json"
    if output.exists():
        parser.error(f"Refusing to overwrite existing report: {output}")
    report = {"schema_version": 1, "status": "failed", "checks": [], "created_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                              "machine": platform.machine(), "native_threads_requested": 1}}
    try:
        run_checks(report, args.expect_target_versions)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Compatibility: {report['status']}; {len(report['checks'])} checks; report: {output}")
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
