"""Reporting preserves imbalance, missing observations and timing boundaries."""

from copy import deepcopy

from supervised_nmf.evaluation import format_evaluation_summary


def imbalanced_report():
    return {
        "status": "valid",
        "metrics": {
            "accuracy": 1004 / 1024,
            "balanced_accuracy": .5,
            "per_class": [
                {"label": "majority", "precision": 1004 / 1024, "recall": 1, "support": 1004},
                {"label": "minority", "precision": 0, "recall": 0, "support": 20},
            ],
        },
        "training_majority_reference": {
            "available": True, "label": "majority", "accuracy": 1004 / 1024,
            "balanced_accuracy": .5, "class_labels": ["minority", "majority"],
            "training_class_counts": [80, 4016],
        },
    }


def test_high_accuracy_does_not_hide_minority_failure_or_class_balance():
    report = imbalanced_report()
    before = deepcopy(report)
    text = format_evaluation_summary(report)
    assert "| Accuracy | 0.980469 |" in text
    assert "| Balanced accuracy | 0.5 |" in text
    assert "| majority | 0.980469 | 1 | 1004 | 4016 |" in text
    assert "| minority | 0 | 0 | 20 | 80 |" in text
    assert report == before


def test_missing_training_counts_are_not_inferred_from_evaluation():
    report = imbalanced_report()
    report.pop("training_majority_reference")
    text = format_evaluation_summary(report)
    assert "| majority | 0.980469 | 1 | 1004 | unavailable |" in text
    assert "| minority | 0 | 0 | 20 | unavailable |" in text
    assert "training labels were not supplied" in text


def test_large_class_counts_remain_exact_integers():
    report = imbalanced_report()
    report["metrics"]["per_class"][0]["support"] = 1234567
    report["training_majority_reference"]["training_class_counts"][1] = 9876543
    text = format_evaluation_summary(report)
    assert "| majority | 0.980469 | 1 | 1234567 | 9876543 |" in text


def test_failed_prediction_still_reports_known_counts_without_fabricating_metrics():
    report = {
        "status": "failed", "error": "Prediction failed",
        "support_diagnostics": {
            "sample_count": 12, "class_labels": ["negative", "positive"],
            "class_counts": [10, 2],
        },
    }
    text = format_evaluation_summary(report)
    assert "Evaluation status: failed" in text
    assert "| negative | unavailable | unavailable | 10 | unavailable |" in text
    assert "| positive | unavailable | unavailable | 2 | unavailable |" in text
    assert "| Accuracy |" not in text


def test_legacy_missing_precision_and_undefined_metrics_stay_explicit():
    report = {
        "status": "valid",
        "metrics": {
            "accuracy": 1, "balanced_accuracy": None,
            "undefined_reason": "Evaluation split contains only one class.",
            "per_class": [{"label": "absent|class", "recall": 0, "support": 0}],
        },
    }
    text = format_evaluation_summary(report)
    assert "| Balanced accuracy | unavailable |" in text
    assert "Evaluation split contains only one class." in text
    assert "| absent\\|class | unavailable | 0 | 0 | unavailable |" in text


def test_refit_cost_and_fitted_estimator_components_are_not_added():
    report = {
        "tuning_seconds": 12.5, "final_fit_seconds": 3,
        "evaluation": {
            "status": "valid", "prediction_seconds": .2,
            "diagnostics": {"fit_times": {"screening": .1, "factorization": .125,
                                          "classifier": .001, "total": .25}},
        },
    }
    text = format_evaluation_summary(report)
    assert "| Tuning | 12.5 |" in text
    assert "| Final fitting (recorded attempts) | 3 |" in text
    assert "| Prediction | 0.2 |" in text
    assert "| Fit: factorization | 0.125 |" in text
    assert "| Fit: training projection | unavailable |" in text
    assert "| Fit: total | 0.25 |" in text
    assert "unsuccessful refit attempts" in text
    assert "Do not add these overlapping totals" in text
    assert "batch throughput are separate measurements" in text


def test_absent_timing_is_not_synthesized_from_components():
    report = {"diagnostics": {"fit_times": {"screening": .2, "factorization": .3}}}
    text = format_evaluation_summary(report)
    assert "| Final fitting (recorded attempts) | unavailable |" in text
    assert "| Fit: total | unavailable |" in text
    assert "| Prediction | unavailable |" in text
