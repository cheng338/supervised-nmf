"""Small subject-separated calibration example. Author: Chang Cheng."""
from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from supervised_nmf import NMFClassifier
from supervised_nmf.evaluation import evaluate_classifier, format_evaluation_summary
from supervised_nmf.search import calibration_rank_search


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.RandomState(42)
    subjects = np.repeat(np.arange(40), 4)
    y = np.tile([0, 0, 1, 1], 40)
    X = rng.uniform(0.1, 1.0, (160, 16))
    X[:, :4] += y[:, None] * 0.7
    ids = np.arange(len(y))
    training, evaluation = subjects < 30, subjects >= 30
    estimator = NMFClassifier(screening="absolute", random_state=42)
    report = calibration_rank_search(
        estimator, {"n_components": [2, 4], "score_threshold": [0.0, 2.0],
                    "n_neighbors": [3, 5]},
        X[training], y[training], cv=5, seed=42, threads=1,
        groups=subjects[training], data_ids=ids[training],
        max_expansions=1, max_rank=8, checkpoint_dir=args.output,
    )
    print("Rank-search stopping reason:", report["stop_reason"])
    if report["selected_parameters"] is None:
        raise RuntimeError("No numerically qualified training candidate.")
    # This is a small API example, not a confirmatory performance study.
    estimator.set_params(**report["selected_parameters"]).fit(X[training], y[training])
    if not estimator.converged_:
        raise RuntimeError("The selected final fit did not converge.")
    result = evaluate_classifier(
        estimator, X[evaluation], y[evaluation], threads=1,
        sample_ids=ids[evaluation], train_sample_ids=ids[training],
        groups=subjects[evaluation], train_groups=subjects[training], train_y=y[training],
    )
    print(format_evaluation_summary({
        "evaluation": result, "tuning_seconds": report["tuning_seconds"],
        "final_fit_seconds": estimator.fit_times_["total"],
    }))


if __name__ == "__main__":
    main()
