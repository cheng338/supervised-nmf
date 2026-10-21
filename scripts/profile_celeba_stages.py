"""Measure additional selected-setting fits after a CelebA study has completed.

These are separate profiling fits, not the original CV or final-fit timings.
The script calls the original fit/fit_transform/transform methods once, times
them with reversible wrappers, and requires exact saved validation predictions.
It never tunes parameters or opens the official test partition. Run alone:

    python scripts/profile_celeba_stages.py \
        --run runs/COMPLETED_RUN --data-cache data/CACHE --repeats 1

Authorship: Chang Cheng; instrumentation supports the classification package
derived from Chang Cheng's dissertation and original scripts.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import NMF, PCA
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf.classifier import NMFClassifier
from supervised_nmf.decomposition import FeatureSPA
from supervised_nmf.evaluation import (
    _diagnostics, _digest, _nonconverged, _warning_records,
    comparison_models, data_fingerprint, evaluate_classifier,
    new_run_directory, provenance, write_json,
)
from supervised_nmf.screening import BinaryScreening

ATTRIBUTES = ("Smiling", "Wearing_Necktie")
STAGES = (
    "screening_fit_seconds", "decomposition_fit_seconds",
    "training_projection_seconds", "classifier_fit_seconds",
    "screening_prediction_seconds", "decomposition_projection_seconds",
    "classifier_prediction_seconds",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_receipt(path):
    receipt = _read_json(path)
    if receipt.get("sha256") != _digest(receipt.get("payload")):
        raise ValueError(f"Checksum mismatch: {path}")
    return receipt["payload"]


def _runtime_identity(environment):
    keys = ("user_api", "internal_api", "prefix", "filepath", "version",
            "threading_layer", "architecture")
    libraries = [{key: lib.get(key) for key in keys}
                 for lib in environment["threadpools"]]
    return {"python": environment["python"], "platform": environment["platform"],
            "versions": environment["versions"],
            "numerical_runtime": sorted(libraries, key=lambda x: x.get("filepath") or "")}


class _StageProfiler:
    """Serial-only instrumentation; restore every class method even on failure.

    No coordinates are recomputed: in particular PCA Pipeline training retains
    its original fit_transform result. Timings surround the unchanged methods.
    Components are nested within total fit/predict and must not be added to them.
    """

    def __init__(self):
        self.phase = "fit"
        self.seconds = dict.fromkeys(STAGES, 0.0)
        self.calls = dict.fromkeys(STAGES, 0)
        self._stack = ExitStack()

    def _wrap(self, cls, method, stages):
        original = getattr(cls, method)

        @wraps(original)
        def timed(instance, *args, **kwargs):
            stage = stages.get(self.phase)
            if stage is None:
                return original(instance, *args, **kwargs)
            start = time.perf_counter()
            try:
                return original(instance, *args, **kwargs)
            finally:
                self.seconds[stage] += time.perf_counter() - start
                self.calls[stage] += 1

        self._stack.enter_context(patch.object(cls, method, timed))

    def __enter__(self):
        try:
            self._wrap(BinaryScreening, "fit_transform", {"fit": "screening_fit_seconds"})
            self._wrap(BinaryScreening, "transform", {"prediction": "screening_prediction_seconds"})
            for cls in (NMF, FeatureSPA, PCA):
                self._wrap(cls, "fit_transform", {"fit": "decomposition_fit_seconds"})
                self._wrap(cls, "transform", {"fit": "training_projection_seconds",
                                              "prediction": "decomposition_projection_seconds"})
            for cls in (KNeighborsClassifier, DummyClassifier):
                self._wrap(cls, "fit", {"fit": "classifier_fit_seconds"})
                self._wrap(cls, "predict_proba", {"prediction": "classifier_prediction_seconds"})
            self._wrap(DummyClassifier, "predict", {"prediction": "classifier_prediction_seconds"})
            return self
        except BaseException:
            self._stack.close()
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


def _supported(estimator):
    if type(estimator) in (NMFClassifier, KNeighborsClassifier, DummyClassifier):
        return
    if type(estimator) is Pipeline and estimator.memory is None:
        steps = [type(step) for _, step in estimator.steps]
        if steps in ([BinaryScreening, KNeighborsClassifier],
                     [BinaryScreening, PCA, KNeighborsClassifier]):
            return
    raise TypeError("Profiling supports only the unchanged comparison_models estimators.")


def _compare_predictions(reference, reproduced):
    fields = ("sample_ids", "true_labels", "predicted_labels", "positive_class_probability")
    comparisons = {field: field in reference and field in reproduced
                           and np.array_equal(np.asarray(reference[field]),
                                              np.asarray(reproduced[field]))
                   for field in fields}
    probability_error = None
    if (reference.get("positive_class_probability") is not None
            and reproduced.get("positive_class_probability") is not None):
        old = np.asarray(reference["positive_class_probability"])
        new = np.asarray(reproduced["positive_class_probability"])
        if old.shape == new.shape:
            probability_error = float(np.max(np.abs(old - new), initial=0.0))
    return {"exact_match": all(comparisons.values()), "fields": comparisons,
            "max_absolute_probability_difference": probability_error}


def _expected_stages(estimator):
    expected = {"classifier_fit_seconds", "classifier_prediction_seconds"}
    if isinstance(estimator, NMFClassifier):
        expected.update(("decomposition_fit_seconds", "decomposition_projection_seconds"))
        if estimator.screening != "none":
            expected.add("screening_fit_seconds")
        if estimator.training_coordinates == "transform":
            expected.add("training_projection_seconds")
    elif isinstance(estimator, Pipeline):
        expected.update(("screening_fit_seconds", "screening_prediction_seconds"))
        if "pca" in estimator.named_steps:
            expected.update(("decomposition_fit_seconds", "decomposition_projection_seconds"))
    return expected


def profile_selected_estimator(estimator, X, y, Xv, yv, *, reference, sample_ids):
    """Profile one additional fit; retain failures and verify saved predictions.

    This helper is also exercised by the tiny fixture check. Native libraries
    are held to one thread; no multiprocessing or model selection occurs here.
    """
    _supported(estimator)
    model = clone(estimator)
    record = {"status": "failed", "exact_prediction_match": False,
              "timing": {"total_fit_seconds": None, "prediction_seconds": None},
              "fit_warnings": [], "evaluation": None}
    caught = []
    with threadpool_limits(limits=1), _StageProfiler() as meter:
        try:
            start = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(X, y)
            finally:
                record["timing"]["total_fit_seconds"] = time.perf_counter() - start
            record["diagnostics"] = _diagnostics(model)
            if _nonconverged(model, caught):
                record["status"] = "nonconverged"
                record["reason"] = "Additional profiling fit did not converge; no validation prediction made."
            else:
                meter.phase = "prediction"
                evaluation = evaluate_classifier(model, Xv, yv, threads=1, sample_ids=sample_ids)
                record["evaluation"] = evaluation
                record["timing"]["prediction_seconds"] = evaluation.get("prediction_seconds")
                if evaluation["status"] != "valid":
                    record["status"] = ("nonconverged" if evaluation["status"] == "provisional_nonconvergence"
                                        else "failed")
                    record["reason"] = "Additional profiling prediction was not numerically valid."
                else:
                    check = _compare_predictions(reference["predictions"], evaluation["predictions"])
                    record["reproducibility"] = check
                    record["exact_prediction_match"] = check["exact_match"]
                    record["status"] = "valid" if check["exact_match"] else "prediction_mismatch"
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
        finally:
            record["fit_warnings"] = _warning_records(caught)
            record["timing"].update(meter.seconds)
            # A stage absent from the method is zero. A required stage never
            # reached after an earlier failure is unavailable, rather than zero.
            for stage in _expected_stages(model):
                if meter.calls[stage] == 0:
                    record["timing"][stage] = None
            record["instrumented_method_calls"] = meter.calls
    return record


def _load_study(run, cache):
    if not (run / "comparison.json").exists():
        raise ValueError("The main study is incomplete: comparison.json must exist before profiling.")
    configuration = _read_json(run / "configuration.json")
    main = _read_json(run / "comparison.json")
    manifest = _read_json(cache / "manifest.json")
    original_environment = _read_json(run / "provenance.json")
    current_environment = provenance(1)
    if configuration["threads"] != 1:
        raise ValueError("This profiler requires a main study performed with one native thread.")
    if main.get("official_test_loaded") is not False or manifest.get("official_test_loaded") is not False:
        raise ValueError("Only development studies that did not load official test data are supported.")
    if _digest(manifest) != configuration["data_manifest_sha256"] or manifest != _read_json(run / "data.json"):
        raise ValueError("Data cache manifest differs from the completed study.")
    if current_environment["source_sha256"] != configuration["source_sha256"]:
        raise ValueError("Package source changed since the main study; profiling would not reproduce it.")
    if _runtime_identity(current_environment) != _runtime_identity(original_environment):
        raise ValueError("Python, package, platform, or numerical runtime differs from the main study.")
    main_script = ROOT / "examples" / "celeba_parameter_study.py"
    if hashlib.sha256(main_script.read_bytes()).hexdigest() != configuration["script_sha256"]:
        raise ValueError("The original study script changed since the main study.")
    if set(main["attributes"]) != set(ATTRIBUTES):
        raise ValueError("The completed study must contain both declared CelebA attributes.")
    models = comparison_models(random_state=configuration["seed"], max_iter=configuration["grid"]["max_iter"],
                               tol=configuration["grid"]["tol"])
    matrices, labels = {}, {}
    for split, partition in (("train", 0), ("validation", 1)):
        if manifest["splits"][split]["metadata"]["partition_code"] != partition:
            raise ValueError(f"Unexpected official partition for {split}.")
        X = np.load(cache / f"{split}-X.npy", mmap_mode="r", allow_pickle=False)
        ids = manifest["splits"][split]["image_ids"]
        if len(ids) != len(X) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid image IDs for {split}.")
        matrices[split] = X
        for attribute in ATTRIBUTES:
            y = np.load(cache / f"{split}-{attribute}-y.npy", allow_pickle=False)
            expected = manifest["splits"][split]["attributes"][attribute]["data_sha256"]
            if len(y) != len(X) or data_fingerprint(X, y) != expected:
                raise ValueError(f"Cache fingerprint mismatch for {split}/{attribute}.")
            labels[split, attribute] = y
    if set(manifest["splits"]["train"]["image_ids"]) & set(manifest["splits"]["validation"]["image_ids"]):
        raise ValueError("Training and validation image IDs overlap.")
    jobs, skipped = [], []
    for attribute in ATTRIBUTES:
        if set(main["attributes"][attribute]["models"]) != set(models):
            raise ValueError(f"The completed study has missing or unknown methods for {attribute}.")
        for method, (estimator, _) in models.items():
            selected = main["attributes"][attribute]["models"][method]
            receipt = _read_receipt(run / attribute / method / "development-evaluation.json")
            if selected != receipt or receipt.get("run_identity") != _digest(configuration):
                raise ValueError(f"Completed result differs from its original receipt: {attribute}/{method}.")
            params = receipt.get("selected_parameters")
            if receipt.get("is_fallback") is not False or receipt["evaluation"]["status"] != "valid" or params is None:
                skipped.append({"attribute": attribute, "method": method,
                                "reason": "Main study result is invalid, held, or a fallback.",
                                "original_status": receipt["evaluation"]["status"]})
                continue
            calibration = _read_json(run / attribute / method / "calibration-summary.json")
            if params != calibration["selected_parameters"]:
                raise ValueError(f"Selected parameters differ from calibration: {attribute}/{method}.")
            # Attribute-specific winners must own separate unfitted estimators;
            # setting the second attribute must not change an earlier job.
            estimator = clone(estimator).set_params(**params)
            reference = receipt["evaluation"]
            if reference["data_sha256"] != manifest["splits"]["validation"]["attributes"][attribute]["data_sha256"]:
                raise ValueError(f"Validation fingerprint mismatch: {attribute}/{method}.")
            predictions = reference["predictions"]
            if (predictions["sample_ids"] != manifest["splits"]["validation"]["image_ids"]
                    or not np.array_equal(np.asarray(predictions["true_labels"]), labels["validation", attribute])):
                raise ValueError(f"Validation row ordering differs: {attribute}/{method}.")
            jobs.append((attribute, method, estimator, receipt))
    return configuration, main, manifest, current_environment, matrices, labels, jobs, skipped


def profile_study(run, cache, *, output=None, repeats=1):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    run, cache = Path(run).resolve(), Path(cache).resolve()
    configuration, main, manifest, environment, matrices, labels, jobs, skipped = _load_study(run, cache)
    destination = new_run_directory(output or run / "profiles", "celeba-stage-profile")
    report = {"author": "Chang Cheng", "role": "additional_selected_setting_profiling_fits",
              "status": "running", "source_run": str(run), "source_run_identity": _digest(configuration),
              "source_configuration": configuration, "data_cache": str(cache),
              "data_manifest_sha256": _digest(manifest), "provenance": environment,
              "profiler_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "original_comparison_sha256": hashlib.sha256((run / "comparison.json").read_bytes()).hexdigest(),
              "repeats_requested": repeats, "workers": 1, "native_threads": 1,
              "official_test_loaded": False, "tuning_complete": False, "defaults_frozen": False,
              "original_calibration_call_wall_seconds": main.get("calibration_call_wall_seconds"),
              "records": [], "skipped": skipped,
              "timing_notes": [
                  "Every record is an additional profiling fit, excluded from original CV and final-fit costs.",
                  "Original selected parameters are reused; validation predictions verify reproduction only.",
                  "Components are nested within total fit/prediction; do not add components to those totals.",
                  "Zero decomposition means the method has no MF/PCA; screening-only and raw KNN have none.",
                  "Screening fit wraps original fit_transform; decomposition fit wraps original fit_transform.",
                  "Prediction time is evaluate_classifier's joint labels/probabilities operation, excluding metrics.",
                  "NMF prediction column slicing is in total prediction but not screening_prediction_seconds.",
                  "Wrapping methods adds small measurement overhead; these serial fits differ from parallel CV load.",
                  "Separate repeats are timing/numerical checks with the same seed, not statistical replications.",
              ]}
    write_json(destination / "profiling.json", report)
    try:
        for attribute, method, estimator, original in jobs:
            for repeat in range(repeats):
                if provenance(1)["source_sha256"] != configuration["source_sha256"]:
                    raise ValueError("Package source changed during profiling; remaining fits are held.")
                record = profile_selected_estimator(
                    estimator, matrices["train"], labels["train", attribute], matrices["validation"],
                    labels["validation", attribute], reference=original["evaluation"],
                    sample_ids=manifest["splits"]["validation"]["image_ids"])
                record.update(attribute=attribute, method=method, repeat=repeat,
                              selected_parameters=original["selected_parameters"],
                              original_result_sha256=_digest(original),
                              original_study_timings={
                                  "tuning_compute_seconds": original.get("tuning_compute_seconds"),
                                  "final_fit_seconds": original.get("final_fit_seconds"),
                                  "prediction_seconds": original["evaluation"].get("prediction_seconds")})
                report["records"].append(record)
                write_json(destination / "profiling.json", report)
                print(f"{attribute}/{method} repeat {repeat + 1}: {record['status']}", flush=True)
        report["status"] = "complete"
        report["all_profiled_predictions_match"] = (bool(report["records"])
                                                    and all(r["status"] == "valid" for r in report["records"]))
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(destination / "profiling.json", report)
    print(f"Additional profiling fits saved: {destination.resolve()}", flush=True)
    return destination, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed CelebA parameter-study run.")
    parser.add_argument("--data-cache", type=Path, required=True, help="The exact fingerprinted cache used by the study.")
    parser.add_argument("--output", type=Path, help="Parent for a new unique directory; defaults to RUN/profiles.")
    parser.add_argument("--repeats", type=int, default=1, help="Additional same-seed profiling fits per valid method.")
    args = parser.parse_args()
    _, report = profile_study(args.run, args.data_cache, output=args.output, repeats=args.repeats)
    if not report["all_profiled_predictions_match"]:
        raise SystemExit("Profiling finished with numerical failures or prediction mismatches; inspect profiling.json.")


if __name__ == "__main__":
    main()
