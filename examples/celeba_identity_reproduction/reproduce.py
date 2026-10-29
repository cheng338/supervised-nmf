"""Reproduce the October 6, 2026 CelebA paired comparisons from saved labels.

Author: Chang Cheng.
This example uses fixed predictions; it does not fit or evaluate new models.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.stats import binomtest


METHODS = ("snmf", "spca", "screen", "raw")
COLUMNS = ("true_label", *METHODS)
HERE = Path(__file__).resolve().parent


def load_predictions(path, expected_sha256):
    """Read the fixed binary fixture, preserving its row and column order."""
    with Path(path).open(encoding="ascii", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or tuple(rows[0]) != COLUMNS:
        raise ValueError(f"Expected CSV columns: {', '.join(COLUMNS)}.")
    if len(rows) < 2 or any(
        len(row) != len(COLUMNS) or any(value not in {"0", "1"} for value in row)
        for row in rows[1:]
    ):
        raise ValueError("Each observation must contain five binary labels.")
    # Hash the parsed label table with LF endings, independent of CSV quoting.
    canonical = "".join(",".join(row) + "\n" for row in rows).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != expected_sha256:
        raise ValueError("The prediction fixture differs from the historical extraction.")
    table = np.asarray(rows[1:], dtype=np.int64)
    if set(table[:, 0]) != {0, 1}:
        raise ValueError("Both true-label classes are required.")
    return table


def paired_statistics(y, first, second):
    """Keep the historical sampling order, arithmetic and quantile convention."""
    differences = (first == y).astype(float) - (second == y).astype(float)
    rng = np.random.default_rng(62107)
    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    means = [
        differences[rng.choice(ix, (10000, len(ix)), replace=True)].mean(axis=1)
        for ix in classes
    ]
    balanced_accuracy = (means[0] + means[1]) / 2
    accuracy = (len(classes[0]) * means[0] + len(classes[1]) * means[1]) / len(y)
    wins = int(np.sum((first == y) & (second != y)))
    losses = int(np.sum((first != y) & (second == y)))
    return {
        "balanced_accuracy_difference": float(
            np.mean([differences[ix].mean() for ix in classes])
        ),
        "balanced_accuracy_ci95": np.quantile(
            balanced_accuracy, [0.025, 0.975], method="linear"
        ).tolist(),
        "accuracy_difference": float(differences.mean()),
        "accuracy_ci95": np.quantile(accuracy, [0.025, 0.975], method="linear").tolist(),
        "snmf_only_correct": wins,
        "comparator_only_correct": losses,
        "mcnemar_exact_two_sided_accuracy_p": (
            float(binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue)
            if wins + losses else 1.0
        ),
    }


def summarize(table):
    """Calculate confusion matrices, class recalls and all three contrasts."""
    table = np.asarray(table)
    if (
        table.ndim != 2 or table.shape[1] != len(COLUMNS) or len(table) == 0
        or table.dtype.kind not in "biuf" or not np.isin(table, (0, 1)).all()
    ):
        raise ValueError("Expected a nonempty table with five binary columns.")
    y = table[:, 0]
    if set(y) != {0, 1}:
        raise ValueError("Both true-label classes are required.")
    methods = {}
    for column, name in enumerate(METHODS, start=1):
        prediction = table[:, column]
        recalls = [float(np.mean(prediction[y == c] == c)) for c in (0, 1)]
        methods[name] = {
            "confusion_matrix": [
                [int(np.sum((y == c) & (prediction == p))) for p in (0, 1)]
                for c in (0, 1)
            ],
            "class_recalls": recalls,
            "accuracy": float(np.mean(prediction == y)),
            "balanced_accuracy": float(np.mean(recalls)),
        }
    return {
        "class_counts": [int(np.sum(y == c)) for c in (0, 1)],
        "methods": methods,
        "snmf_minus_comparator": {
            name: paired_statistics(y, table[:, 1], table[:, column])
            for column, name in enumerate(METHODS[1:], start=2)
        },
    }


def main():
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    table = load_predictions(HERE / "predictions.csv", expected["fixture_sha256"])
    statistics = summarize(table)
    matches = statistics == expected["statistics"]
    print(json.dumps({
        "exact_historical_match": matches,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "statistics": statistics,
    }, indent=2, allow_nan=False))
    if not matches:
        print(
            "Computed values differ from the stored historical values. Compare the "
            "output with expected.json and its recorded library versions; a difference "
            "may reflect library roundoff or sampling changes. This check concerns "
            "historical replay, not package compatibility or statistical validity.",
            file=sys.stderr,
        )
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
