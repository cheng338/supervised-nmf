"""Report a completed explicit CelebA study without fitting or selecting models.

Author: Chang Cheng. Run in the py12_df_aocl environment. Matplotlib and
scikit-learn are used only to draw and verify saved development predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


ATTRIBUTES = ("Smiling", "Wearing_Necktie")
METHODS = {
    "screened_cd_percentile": ("CD NMF, percentile", "#0072B2", "-"),
    "screened_cd_absolute": ("CD NMF, absolute |t|", "#0072B2", "--"),
    "screened_spa_percentile": ("SPA NMF, percentile", "#D55E00", "-"),
    "screened_spa_absolute": ("SPA NMF, absolute |t|", "#D55E00", "--"),
    "screened_pca_percentile": ("PCA, percentile", "#009E73", "-"),
    "screened_pca_absolute": ("PCA, absolute |t|", "#009E73", "--"),
    "screening_only_percentile": ("Screening only, percentile", "#CC79A7", "-"),
    "screening_only_absolute": ("Screening only, absolute |t|", "#CC79A7", "--"),
    "unscreened_cd_nmf": ("Unscreened CD NMF", "#0072B2", ":"),
    "raw_knn": ("Raw KNN", "#222222", "-"),
    "majority_class": ("Majority class", "#888888", "-."),
}
METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "average_precision",
           "precision", "recall", "f1")


def _digest(value):
    # Match the experiment's canonical, sorted JSON receipt representation.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Inputs:
    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def read(self, path, receipt=False):
        raw = path.read_bytes()
        self.hashes[path.relative_to(self.root).as_posix()] = hashlib.sha256(raw).hexdigest()
        result = json.loads(raw)
        if receipt:
            if result.get("sha256") != _digest(result.get("payload")):
                raise ValueError(f"Receipt checksum failed: {path}")
            result = result["payload"]
        return result


def _checked_predictions(result, metadata, expected_truth=None):
    evaluation = result["evaluation"]
    saved = evaluation["predictions"]
    truth = np.asarray(saved["true_labels"])
    predicted = np.asarray(saved["predicted_labels"])
    probability = np.asarray(saved["positive_class_probability"], dtype=float)
    counts = metadata["counts"]
    n = int(sum(counts))
    if any(array.shape != (n,) for array in (truth, predicted, probability)):
        raise ValueError("Saved prediction lengths differ from the validation manifest.")
    if (not np.isin(truth, [0, 1]).all() or not np.isin(predicted, [0, 1]).all()
            or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))):
        raise ValueError("Saved binary labels or positive-class probabilities are invalid.")
    if np.bincount(truth.astype(int), minlength=2).tolist() != counts:
        raise ValueError("Saved validation labels differ from the manifest class counts.")
    if saved["sample_ids"] != metadata["image_ids"]:
        raise ValueError("Saved validation prediction IDs differ from the manifest.")
    if expected_truth is not None and not np.array_equal(truth, expected_truth):
        raise ValueError("Methods were not evaluated on identical ordered validation labels.")
    if evaluation["data_sha256"] != metadata["data_sha256"]:
        raise ValueError("Saved evaluation fingerprint differs from the validation manifest.")
    metrics = evaluation["metrics"]
    if metrics["positive_class"] != 1 or metrics["confusion_matrix_labels"] != [0, 1]:
        raise ValueError("Expected CelebA binary labels 0 and 1 with positive class 1.")
    if not all(counts):
        raise ValueError("Both validation classes are required for PR and ROC plots.")
    checked = {
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "roc_auc": roc_auc_score(truth, probability),
        "average_precision": average_precision_score(truth, probability),
        "precision": precision_score(truth, predicted, zero_division=0),
        "recall": recall_score(truth, predicted, zero_division=0),
        "f1": f1_score(truth, predicted, zero_division=0),
    }
    if any(not np.isclose(metrics[key], value, rtol=1e-12, atol=1e-12)
           for key, value in checked.items()):
        raise ValueError("Saved metrics disagree with the saved predictions.")
    if confusion_matrix(truth, predicted, labels=[0, 1]).tolist() != metrics["confusion_matrix"]:
        raise ValueError("Saved confusion counts disagree with the saved predictions.")
    return truth, probability


def _fold_diagnostics(inputs, directory):
    statuses, errors, warnings = Counter(), Counter(), Counter()
    unique, iterations = set(), []
    costs = {"representation_fit_seconds": 0.0, "validation_projection_seconds": 0.0,
             "knn_fit_seconds": 0.0, "knn_prediction_seconds": 0.0}
    recorded_factorization_seconds = 0.0
    factorization_timed_folds = 0
    for path in sorted(directory.glob("chunk-*/folds/*.json")):
        row = inputs.read(path, receipt=True)
        key = row["key"]
        if key in unique:
            continue
        if row["state"] != "completed":
            raise ValueError(f"Incomplete fold receipt found: {path}")
        unique.add(key)
        statuses[row["status"]] += 1
        costs["representation_fit_seconds"] += row["representation_fit_seconds"]
        costs["validation_projection_seconds"] += row["projection_seconds"]
        costs["knn_fit_seconds"] += sum(item["fit_seconds"] for item in row["neighbors"].values())
        costs["knn_prediction_seconds"] += sum(item["prediction_seconds"] for item in row["neighbors"].values())
        factorization = row.get("diagnostics", {}).get("fit_times", {}).get("factorization")
        if factorization is not None:
            recorded_factorization_seconds += factorization
            factorization_timed_folds += 1
        if row.get("error_type"):
            errors[row["error_type"]] += 1
        # Count a warning category once per representation fold, not per k.
        for category in {warning["category"] for warning in row.get("warnings", [])}:
            warnings[category] += 1
        iteration = row.get("diagnostics", {}).get("n_iter")
        if isinstance(iteration, (int, float)):
            iterations.append(iteration)
    if not factorization_timed_folds:
        recorded_factorization_seconds = None
    return {"unique_completed_representation_folds": len(unique),
            "statuses": dict(sorted(statuses.items())),
            "error_types": dict(sorted(errors.items())),
            "warning_categories_by_fold": dict(sorted(warnings.items())),
            "maximum_recorded_iterations": max(iterations) if iterations else None,
            "compute_seconds": costs,
            "nonadditive_breakdown_seconds": {"recorded_factorization_seconds": recorded_factorization_seconds},
            "factorization_timed_folds": factorization_timed_folds,
            "factorization_timing_note": "The factorization subtotal is already included in representation_fit_seconds and must not be added to compute_seconds. Only folds exposing stage diagnostics are included; earlier failures may lack a factorization breakdown but their total fit time remains recorded.",
            "counting_unit": "unique committed shared representation/fold attempt, independent of neighbor count; a completed receipt need not contain a successful factorization"}


def _execution_transition(inputs, configuration):
    path = inputs.root / "execution-transition.json"
    if not path.exists():
        return None
    transition = inputs.read(path)
    if (transition["workers"] != configuration["workers"]
            or transition["numerical_threads_per_worker"] != configuration["threads"]
            or transition["source_parameters_data_and_folds_unchanged"] is not True):
        raise ValueError("Execution transition differs from the continuation configuration.")
    prior_configuration = Path(transition["prior_run"]) / "configuration.json"
    prior_verified, prior_hash = False, None
    if prior_configuration.exists():
        raw = prior_configuration.read_bytes()
        prior = json.loads(raw)
        if (prior["workers"] != transition["prior_workers"]
                or {key: value for key, value in prior.items() if key != "workers"}
                != {key: value for key, value in configuration.items() if key != "workers"}):
            raise ValueError("The prior and continuation configurations differ beyond worker count.")
        prior_verified, prior_hash = True, hashlib.sha256(raw).hexdigest()
    transferred_folds = 0
    for relative, expected in transition["checkpoint_copy_sha256"].items():
        if "/folds/" not in relative or not relative.endswith(".json"):
            continue  # Chunk report timing metadata may legitimately be refreshed.
        copied = (inputs.root / relative).resolve()
        if not copied.is_relative_to(inputs.root):
            raise ValueError("A transferred checkpoint path leaves the run directory.")
        record = inputs.read(copied, receipt=True)
        if (inputs.hashes[relative] != expected or record["state"] != "completed"):
            raise ValueError("A transferred committed fold differs from the preserved source receipt.")
        transferred_folds += 1
    interrupted = transition.get("uncommitted_attempts_at_interruption", [])
    retried = 0
    for attempt in interrupted:
        attempt_path = (inputs.root / attempt["path"]).resolve()
        if not attempt_path.is_relative_to(inputs.root):
            raise ValueError("An interrupted checkpoint path leaves the run directory.")
        receipt = inputs.read(attempt_path, receipt=True)
        retried_path = attempt_path.parent.parent / "folds" / f"{receipt['key']}.json"
        if retried_path.exists():
            completed = inputs.read(retried_path, receipt=True)
            retried += int(completed["key"] == receipt["key"] and completed["state"] == "completed")
    snapshot = transition.get("prior_interruption_snapshot", {})
    processes = snapshot.get("processes", [])
    snapshot_time = datetime.fromisoformat(snapshot["time"].replace("Z", "+00:00")) if snapshot.get("time") else None
    starts = [datetime.fromisoformat(row["StartTime"].replace("Z", "+00:00")) for row in processes if row.get("StartTime")]
    elapsed = (snapshot_time - min(starts)).total_seconds() if snapshot_time and starts else None
    return {
        "prior_workers": transition["prior_workers"], "continuation_workers": transition["workers"],
        "native_threads_per_worker": transition["numerical_threads_per_worker"],
        "source_parameters_data_and_folds_unchanged": True,
        "prior_configuration_independently_verified": prior_verified, "prior_configuration_sha256": prior_hash,
        "transferred_committed_representation_fold_attempts": transferred_folds,
        "interrupted_uncommitted_attempts": len(interrupted), "retried_fold_attempts_with_committed_receipts": retried,
        "uncommitted_interrupted_factorization_seconds": None,
        "earliest_recorded_prior_process_start_utc": min(starts).astimezone(timezone.utc).isoformat() if starts else None,
        "prior_interruption_snapshot_utc": snapshot_time.astimezone(timezone.utc).isoformat() if snapshot_time else None,
        "prior_process_elapsed_to_snapshot_seconds": elapsed,
        "prior_recorded_process_count": len(processes),
        "prior_recorded_aggregate_process_cpu_seconds": sum(row.get("CPU", 0.0) for row in processes),
        "prior_process_resource_snapshot": processes,
        "transfer_finished_utc": transition.get("transfer_finished_utc"),
        "reason": transition.get("reason"), "end_to_end_wall_span_seconds": None,
        "timing_note": "Mixed worker-count execution. Original committed costs are included once in the final fold totals. Continuation calibration-call wall time excludes the earlier phase. Prior observed-process CPU and elapsed span remain separate; do not add CPU to wall or committed compute. Lost stage times from interrupted uncommitted attempts are unknown. Per-process memory samples are not a whole-run process-tree peak. No complete end-to-end wall span is inferred from mutable file timestamps.",
    }


def summarize(run_dir, scan_folds=False):
    inputs = Inputs(run_dir)
    comparison = inputs.read(run_dir / "comparison.json")
    configuration = inputs.read(run_dir / "configuration.json")
    data = inputs.read(run_dir / "data.json")
    if _digest(data) != configuration["data_manifest_sha256"]:
        raise ValueError("Data manifest differs from the saved run configuration.")
    if set(comparison["attributes"]) != set(ATTRIBUTES):
        raise ValueError("The completed comparison must contain both declared attributes.")
    if comparison.get("official_test_loaded") or data.get("official_test_loaded"):
        raise ValueError("This development report does not accept runs that loaded the official test split.")
    if (comparison["replications"] != 1 or comparison["tuning_complete"] or comparison["defaults_frozen"]):
        raise ValueError("This report describes one provisional development run, before tuning/default decisions.")
    summary = {
        "schema_version": 1, "author": "Chang Cheng", "run_name": run_dir.name,
        "run_identity": _digest(configuration),
        "role": "one_paired_development_comparison", "seed": configuration["seed"],
        "workers": configuration["workers"], "native_threads_per_worker": configuration["threads"],
        "training_cv_folds": configuration["cv"], "selection_metric": configuration["selection_metric"],
        "n_train": len(data["splits"]["train"]["image_ids"]),
        "n_validation": len(data["splits"]["validation"]["image_ids"]),
        "replications": comparison["replications"], "official_test_loaded": False,
        "identity_overlap_checked": data.get("identity_overlap_checked", False),
        "tuning_complete": comparison["tuning_complete"], "defaults_frozen": comparison["defaults_frozen"],
        "grid": configuration["grid"], "fixture_note": data.get("fixture_note"),
        "calibration_call_wall_seconds": comparison["calibration_call_wall_seconds"],
        "qualification_rule": "evaluation.status == valid and is_fallback == false",
        "curves_use_saved_predictions_only": True,
        "fold_diagnostics_scanned": scan_folds, "attributes": {},
        "execution_transition": _execution_transition(inputs, configuration),
    }
    plots = {}
    for attribute in ATTRIBUTES:
        block = comparison["attributes"][attribute]
        if set(block["models"]) != set(METHODS):
            raise ValueError(f"Expected all 11 methods for {attribute}; a partial run cannot be summarized.")
        train_counts = data["splits"]["train"]["attributes"][attribute]["class_counts"]
        validation = data["splits"]["validation"]
        validation_counts = validation["attributes"][attribute]["class_counts"]
        if (block["training_class_counts"] != train_counts
                or block["validation_class_counts"] != validation_counts):
            raise ValueError("Comparison class counts differ from the data manifest.")
        metadata = {"counts": validation_counts, "image_ids": validation["image_ids"],
                    "data_sha256": validation["attributes"][attribute]["data_sha256"]}
        rows, curves, truth = [], {}, None
        for method, (label, _, _) in METHODS.items():
            result = block["models"][method]
            directory = run_dir / attribute / method
            receipt = inputs.read(directory / "development-evaluation.json", receipt=True)
            if receipt != result:
                raise ValueError(f"Comparison and saved evaluation differ: {attribute}/{method}")
            if (result["run_identity"] != _digest(configuration) or result["method"] != method
                    or result["attribute"] != attribute):
                raise ValueError("An evaluation is bound to a different run, attribute or method.")
            calibration = inputs.read(directory / "calibration-summary.json")
            if calibration["selected_parameters"] != result["selected_parameters"]:
                raise ValueError("Calibration summary and final evaluation select different parameters.")
            candidates = calibration["candidates"]
            valid = [row for row in candidates if row["valid"]]
            winner = min(valid, key=lambda row: (-row["mean_balanced_accuracy"], row["index"])) if valid else None
            if result["evaluation_identity"] != _digest({"run": configuration, "attribute": attribute,
                                                         "method": method, "selected_candidate": winner}):
                raise ValueError("Evaluation is bound to different selected CV evidence.")
            evaluation = result["evaluation"]
            qualified = evaluation["status"] == "valid" and not result["is_fallback"]
            metrics = evaluation.get("metrics", {})
            matrix = metrics.get("confusion_matrix")
            confusion = dict(zip(("tn", "fp", "fn", "tp"), np.asarray(matrix).ravel().tolist())) if matrix is not None else None
            row = {"method": method, "label": label, "status": evaluation["status"],
                   "original_result_sha256": _digest(result),
                   "qualified": qualified, "is_fallback": result["is_fallback"],
                   "reason": evaluation.get("reason", evaluation.get("error")),
                   "validation_prevalence": validation_counts[1] / sum(validation_counts),
                   "metrics": {key: metrics.get(key) for key in METRICS}, "confusion_counts": confusion,
                   "selected_parameters": result["selected_parameters"],
                   "cv_balanced_accuracy": result["cv_balanced_accuracy"],
                   "declared_cv_candidates": len(candidates), "valid_cv_candidates": len(valid),
                   "invalid_cv_candidates": len(candidates) - len(valid),
                   "tuning_compute_seconds": result["tuning_compute_seconds"],
                   "final_fit_seconds": result["final_fit_seconds"],
                   "final_fit_breakdown_seconds": {
                       key: result.get("diagnostics", {}).get("fit_times", {}).get(key)
                       for key in ("screening", "factorization", "training_projection", "classifier")},
                   "validation_prediction_seconds": evaluation.get("prediction_seconds"),
                   "peak_memory_bytes": None, "peak_memory_measured": False,
                   "final_fit_warnings": result.get("final_fit_warnings", []),
                   "validation_warnings": evaluation.get("warnings", []),
                   "refinement": calibration["refinement"],
                   "representation_fold_diagnostics": _fold_diagnostics(inputs, directory) if scan_folds else None}
            if qualified:
                truth, probability = _checked_predictions(result, metadata, truth)
                curves[method] = (truth, probability)
            rows.append(row)
        summary["attributes"][attribute] = {
            "training_class_counts": train_counts, "validation_class_counts": validation_counts,
            "training_prevalence": train_counts[1] / sum(train_counts),
            "validation_prevalence": validation_counts[1] / sum(validation_counts), "methods": rows}
        plots[attribute] = curves
    summary["input_sha256"] = dict(sorted(inputs.hashes.items()))
    summary["reporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["reporting_environment"] = {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("numpy", "matplotlib", "scikit-learn")}}
    return summary, plots


def attach_profile(summary, profile_path):
    """Attach a separate matched-prediction timing run without replacing main timings."""
    raw = profile_path.read_bytes()
    profile = json.loads(raw)
    if profile["status"] != "complete":
        raise ValueError("Only a completed post-selection profiling run may be attached.")
    if (profile["source_run_identity"] != summary["run_identity"]
            or _digest(profile["source_configuration"]) != summary["run_identity"]):
        raise ValueError("Profiling refers to another study configuration.")
    if profile["original_comparison_sha256"] != summary["input_sha256"]["comparison.json"]:
        raise ValueError("Profiling refers to different saved comparison evidence.")
    targets = {(attribute, row["method"]): row for attribute, block in summary["attributes"].items()
               for row in block["methods"]}
    for row in targets.values():
        row["post_selection_profiling"] = {"records": [], "qualified_repeats": 0, "median_seconds": {}}
    seen = set()
    for record in profile["records"]:
        key = record["attribute"], record["method"]
        if key not in targets:
            raise ValueError("Profiling contains an unknown attribute or method.")
        row = targets[key]
        repetition = (*key, record["repeat"])
        if repetition in seen:
            raise ValueError("Profiling contains duplicate method repetitions.")
        seen.add(repetition)
        if record["selected_parameters"] != row["selected_parameters"]:
            raise ValueError("Profiling parameters differ from the main selected parameters.")
        if record["original_result_sha256"] != row["original_result_sha256"]:
            raise ValueError("Profiling refers to a different saved method evaluation.")
        qualified = record["status"] == "valid" and record.get("exact_prediction_match") is True
        if qualified and not row["qualified"]:
            raise ValueError("A profiling repeat cannot qualify an unqualified main result.")
        for name, value in record.get("timing", {}).items():
            if value is not None and (not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0):
                raise ValueError(f"Invalid profiling time: {name}")
        row["post_selection_profiling"]["records"].append({"repeat": record["repeat"], "status": record["status"],
            "qualified": qualified, "exact_prediction_match": record.get("exact_prediction_match"),
            "timing": record.get("timing", {}), "error": record.get("error")})
    for row in targets.values():
        attached = row["post_selection_profiling"]
        qualified = [record for record in attached["records"] if record["qualified"]]
        attached["qualified_repeats"] = len(qualified)
        names = sorted({name for record in qualified for name in record["timing"]})
        for name in names:
            values = [record["timing"][name] for record in qualified if record["timing"].get(name) is not None]
            attached["median_seconds"][name] = float(np.median(values)) if values else None
    summary["post_selection_profile"] = {
        "input_sha256": hashlib.sha256(raw).hexdigest(), "source_run_identity": profile["source_run_identity"],
        "repeats_requested": profile["repeats_requested"], "workers": profile["workers"],
        "native_threads": profile["native_threads"], "skipped": profile.get("skipped", []),
        "interpretation": "Separate post-selection timing measurements; only valid exact-prediction matches contribute to medians. Fit/prediction stages are components of their respective totals, not additional costs. Medians are calculated independently by column. Main-run timings are preserved. Timing repetitions are not independent dataset replications."}


def plot_curves(summary, plots, output):
    figures = {}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "figure.facecolor": "white"}):
        for attribute in ATTRIBUTES:
            curves = plots[attribute]
            fig, (pr_axis, roc_axis) = plt.subplots(1, 2, figsize=(14.2, 5.3))
            handles, labels = [], []
            for method, (truth, probability) in curves.items():
                label, color, style = METHODS[method]
                precision, recall, _ = precision_recall_curve(truth, probability)
                false_positive, true_positive, _ = roc_curve(truth, probability)
                handle, = pr_axis.step(recall, precision, where="post", color=color,
                                       linestyle=style, linewidth=1.7, alpha=0.9)
                roc_axis.plot(false_positive, true_positive, color=color, linestyle=style, linewidth=1.7, alpha=0.9)
                handles.append(handle)
                labels.append(label)
            prevalence = summary["attributes"][attribute]["validation_prevalence"]
            pr_axis.axhline(prevalence, color="#555555", linestyle=":", linewidth=1.1)
            pr_axis.text(0.02, prevalence + 0.025, f"Prevalence = {prevalence:.3f}", fontsize=9,
                         bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1})
            roc_axis.plot([0, 1], [0, 1], color="#555555", linestyle=":", linewidth=1.1)
            roc_axis.text(0.55, 0.49, "Chance", fontsize=9, color="#555555")
            pr_axis.set(xlabel="Recall (positive class)", ylabel="Precision (positive class)", title="Precision-recall")
            roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
            for axis in (pr_axis, roc_axis):
                axis.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.04))
                axis.grid(alpha=0.15)
            title_prefix = "Synthetic fixture" if summary["fixture_note"] else "CelebA development validation"
            fig.suptitle(f"{title_prefix}: {attribute.replace('_', ' ')}", x=0.39, y=0.99, fontsize=14)
            subtitle = (f"One paired run | {summary['n_train']:,} training / {summary['n_validation']:,} validation | "
                        f"{len(curves)}/11 methods qualified")
            fig.text(0.39, 0.925, subtitle, ha="center", fontsize=10)
            if handles:
                fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.785, 0.51), frameon=False, fontsize=9)
            else:
                fig.text(0.80, 0.50, "No qualified methods", fontsize=11)
            fig.text(0.01, 0.012, "Author: Chang Cheng. Saved validation predictions; no final test or statistical superiority claim.", fontsize=8)
            fig.subplots_adjust(left=0.06, right=0.77, bottom=0.14, top=0.83, wspace=0.29)
            path = output / f"{attribute.lower()}-pr-roc.png"
            fig.savefig(path, metadata={"Software": "Chang Cheng supervised NMF study reporter"})
            plt.close(fig)
            figures[attribute] = path.name
    return figures


def _number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _parameters(parameters):
    if parameters is None:
        return "No valid selection"
    if not parameters:
        return "None (untuned baseline)"
    aliases = {"n_components": "rank", "n_neighbors": "k", "percentile": "retained %", "score_threshold": "abs(t) cutoff"}
    return "; ".join(f"{aliases.get(key.rsplit('__', 1)[-1], key)}={value}" for key, value in sorted(parameters.items()))


def markdown_report(summary, figures):
    lines = ["# CelebA development comparison", "", "Author: **Chang Cheng**", ""]
    if summary["fixture_note"]:
        lines += [f"**Synthetic fixture — not CelebA evidence.** {summary['fixture_note']}", ""]
    lines += [f"Run: `{summary['run_name']}`. One paired development run with {summary['n_train']:,} training "
              f"and {summary['n_validation']:,} validation observations, seed {summary['seed']}. "
              f"Selection used {summary['training_cv_folds']}-fold training CV and balanced accuracy. "
              "All methods used the same observations within each attribute.", "",
              "The official test partition was not loaded. Grid refinement, numerical sensitivity checks, "
              "and replications remain required; tuning is incomplete and package defaults remain unfrozen. "
              "No confidence intervals or statistical superiority claims are made. "
              + ("Identity overlap was checked in the source run." if summary["identity_overlap_checked"]
                 else "Person-identity overlap has not been audited."), "",
              "Only results with status `valid` and no fallback enter the curves. All methods remain in the tables; "
              "provisional metrics, where present, are unqualified. Positive-class precision, recall and F1 use label 1. "
              "AP is average precision. All proportions use a 0–1 scale.", ""]
    for attribute, block in summary["attributes"].items():
        lines += [f"## {attribute.replace('_', ' ')}", "",
                  f"Training counts (negative, positive): {block['training_class_counts']}; "
                  f"validation counts: {block['validation_class_counts']}. Validation positive prevalence: "
                  f"{block['validation_prevalence']:.3%}.", "",
                  "| Method | Status | Accuracy | Balanced acc. | ROC AUC | AP | Precision+ | Recall+ | F1+ |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in block["methods"]:
            status = row["status"] + ("; fallback" if row["is_fallback"] else "")
            values = [row["label"], status] + [_number(row["metrics"][key]) for key in METRICS]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        lines += ["", f"![Saved validation PR and ROC curves]({figures[attribute]})", "",
                  "| Method | TN | FP | FN | TP | CV balanced acc. | Selected parameters |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for row in block["methods"]:
            confusion = row["confusion_counts"] or {}
            values = [row["label"]] + [confusion.get(key, "—") for key in ("tn", "fp", "fn", "tp")]
            values += [_number(row["cv_balanced_accuracy"]), _parameters(row["selected_parameters"])]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        if attribute == "Wearing_Necktie":
            majority = next(row for row in block["methods"] if row["method"] == "majority_class")
            metrics = majority["metrics"]
            lines += ["", f"**Majority baseline:** accuracy {_number(metrics['accuracy'])}, positive-class recall "
                      f"{_number(metrics['recall'])}, and balanced accuracy {_number(metrics['balanced_accuracy'])}. "
                      "Read accuracy together with positive-class recall because this attribute is imbalanced."]
        qualified_knn = [row for row in block["methods"]
                         if row["qualified"] and row["method"] != "majority_class"]
        if qualified_knn and all(
                any(key.split("__")[-1] == "n_neighbors" and value == 1
                    for key, value in row["selected_parameters"].items())
                for row in qualified_knn):
            lines += ["", "Every qualified tuned method for this attribute selected `k=1`. "
                      "Its predicted positive-class probability is therefore either 0 or 1; "
                      "the ROC and PR curves have few operating points, and ROC AUC equals "
                      "balanced accuracy at the classifier's decision threshold. These scores "
                      "do not provide a finely graded risk ranking."]
        lines += ["", "### Recorded timings (seconds)", "",
                  "| Method | CV compute | Final fit, total | Screening | Matrix factorization | KNN fit | Prediction, total |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in block["methods"]:
            stages = row["final_fit_breakdown_seconds"]
            values = [row["label"], _number(row["tuning_compute_seconds"]), _number(row["final_fit_seconds"])]
            values += [_number(stages[key]) for key in ("screening", "factorization", "classifier")]
            values.append(_number(row["validation_prediction_seconds"]))
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        lines += ["", "CV compute includes committed failed and nonconverged folds and counts shared fits once. "
                  "Shared representation-fit time includes the initial KNN fit used to preserve the original training coordinates; "
                  "the per-neighbor KNN fits are timed separately. Shared fit time is therefore not an MF-only measurement. "
                  "Final-fit stages are recorded by NMFClassifier after selection. Pipeline/raw stage breakdowns "
                  "were not measured separately and appear as —; total fit is not relabeled as KNN-only time. "
                  "Screening, matrix factorization and KNN-fit columns are components of the final-fit total; "
                  "do not add them to that total. "
                  "Prediction includes projection and probability/label computation. Whole-run peak memory was not measured."]
        if "post_selection_profile" in summary:
            lines += ["", "### Separate post-selection profiling (median seconds)", "",
                      "These measurements come from a separate timing run using the selected parameters. "
                      "Only valid repeats whose predictions match the original study exactly contribute. "
                      "Timing repeats do not count as dataset replications; main-study timings above are preserved. "
                      "Screening, decomposition, training projection and classifier-fit times are already included in total fit. "
                      "Prediction substages in summary.json are already included in total prediction. "
                      "Do not add stage times to these totals. Totals include other overhead, and independently calculated "
                      "column medians need not add up to the median total.", "",
                      "| Method | Qualified repeats | Screening fit | Decomposition fit | Training projection | Classifier fit | Total fit | Total prediction |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for row in block["methods"]:
                profiling = row["post_selection_profiling"]
                medians = profiling["median_seconds"]
                values = [row["label"], f"{profiling['qualified_repeats']} / {len(profiling['records'])}"]
                values += [_number(medians.get(key)) for key in (
                    "screening_fit_seconds", "decomposition_fit_seconds", "training_projection_seconds",
                    "classifier_fit_seconds", "total_fit_seconds", "prediction_seconds")]
                lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        lines += ["", "### Numerical status and pending refinement", "",
                  "| Method | Valid / declared CV candidates | Shared representation-fold attempt statuses | Pending diagnostics |",
                  "|---|---:|---|---|"]
        for row in block["methods"]:
            diagnostic = row["representation_fold_diagnostics"]
            numerical = "; ".join(f"{key}: {value}" for key, value in diagnostic["statuses"].items()) if diagnostic else "Not scanned"
            if diagnostic and not diagnostic["unique_completed_representation_folds"]:
                numerical = "No CV representation fits (untuned baseline)"
            parameters = row["refinement"].get("parameters", {})
            pending = "; ".join(f"{key}: {value['status']} (selected {value.get('selected')})" for key, value in parameters.items())
            if not pending:
                pending = "Untuned baseline" if row["method"] == "majority_class" else row["refinement"].get("status", "Unreported")
            if row["reason"]:
                pending = row["reason"] + "; " + pending
            values = [row["label"], f"{row['valid_cv_candidates']} / {row['declared_cv_candidates']}", numerical, pending]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        lines += [""]
    lines += ["## Timing and reproducibility", "",
              f"{'Continuation calibration' if summary['execution_transition'] else 'Calibration'} call wall time: "
              f"{summary['calibration_call_wall_seconds']:.2f} seconds. "
              "Per-method compute and final-fit times are preserved in `summary.json`. Shared representation "
              "fits and projections count once per fold, independent of the number of neighbor values. "
              "Concurrent worker compute times must not be interpreted as elapsed wall time. "
              "Uncommitted interrupted attempts are outside committed-fold compute totals. "
              "No algorithmic speed advantage is inferred from these timings.", "",
              "`summary.json` records all input SHA-256 hashes, the reporter hash, reporting software versions, "
              "the declared grid, selected parameters, full refinement diagnostics, and numerical warning counts. "
              "Regenerate with `python scripts/report_celeba_study.py RUN_DIRECTORY --scan-folds` in `py12_df_aocl`. "
              "The script reads saved results and does not fit models, load images, or alter the study.", ""]
    transition = summary["execution_transition"]
    if transition:
        lines += ["### Worker-count continuation", "",
                  f"This was a mixed **{transition['prior_workers']} → {transition['continuation_workers']} worker** run, "
                  f"with {transition['native_threads_per_worker']} native numerical thread per worker. "
                  "Parameters, source, data, folds, tolerances, and seed were unchanged. "
                  f"The continuation reused {transition['transferred_committed_representation_fold_attempts']} "
                  "hash-verified committed representation-fold receipts, whose costs are included exactly once. "
                  "It was not a fresh run performed entirely at the final worker count.", "",
                  f"The prior observed-process elapsed span to the interruption snapshot was "
                  f"{_number(transition['prior_process_elapsed_to_snapshot_seconds'])} seconds. "
                  f"Aggregate recorded CPU time across {transition['prior_recorded_process_count']} observed prior "
                  f"processes was {_number(transition['prior_recorded_aggregate_process_cpu_seconds'])} seconds. "
                  "These are distinct from the continuation calibration-call wall time above and from committed "
                  "numerical compute. They are not added together.", "",
                  f"{transition['interrupted_uncommitted_attempts']} representation-fold attempts were interrupted "
                  f"before commitment; {transition['retried_fold_attempts_with_committed_receipts']} have committed "
                  "retry receipts. Their lost original factorization time is unknown. The intentional stop is "
                  "an execution transition, not evidence of numerical failure. The earlier run and transition "
                  "manifest preserve the resource and checkpoint history. Per-process memory observations do "
                  "not measure the whole-run process-tree peak. A complete end-to-end wall span, including the "
                  "handoff, was not reconstructed from file timestamps.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed explicit CelebA study directory")
    parser.add_argument("--output-dir", type=Path, help="Report directory; defaults to RUN_DIRECTORY/report")
    parser.add_argument("--scan-folds", action="store_true", help="Count numerical outcomes once per shared representation fold")
    parser.add_argument("--profile", type=Path, help="Optional completed profiling.json from a separate matched-prediction timing run")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output_dir or run_dir / "report").resolve()
    summary, plots = summarize(run_dir, args.scan_folds)
    if args.profile:
        attach_profile(summary, args.profile.resolve())
    output.mkdir(parents=True, exist_ok=True)
    figures = plot_curves(summary, plots, output)
    summary["figures"] = figures
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown_report(summary, figures), encoding="utf-8")
    print(f"Report written: {output}")
    for attribute, block in summary["attributes"].items():
        print(f"{attribute}: {sum(row['qualified'] for row in block['methods'])}/11 qualified methods; "
              f"positive prevalence={block['validation_prevalence']:.4f}")


if __name__ == "__main__":
    main()
