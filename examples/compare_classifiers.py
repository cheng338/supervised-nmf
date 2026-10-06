"""Compare KNN and logistic regression on Chang Cheng's original simulation.

Run ``python examples/compare_classifiers.py --quick --threads 1`` for a small
correctness check. Both classifiers use the same training folds and separate
development-validation rows. Screening, factorization and logistic scaling are
fitted within each training fold. This example does not select package defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supervised_nmf.datasets import make_original_simulation
from supervised_nmf.evaluation import (
    comparison_models, new_run_directory, provenance, run_comparison,
    save_data_manifest, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="One candidate per method; correctness check only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--max-iter", type=int, default=20000, help="NMF/NNLS iteration limit.")
    parser.add_argument("--tol", type=float, default=1e-4, help="CD/MU stopping tolerance.")
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--logistic-tol", type=float, default=1e-4)
    parser.add_argument("--logistic-class-weight", choices=("none", "balanced"), default="none")
    args = parser.parse_args()

    class_weight = None if args.logistic_class_weight == "none" else "balanced"
    data = make_original_simulation(n_samples=160 if args.quick else 400,
                                    n_features=60 if args.quick else 120,
                                    signal_ratio=0.1, random_state=args.seed)
    training, validation = train_test_split(np.arange(len(data.target)), test_size=0.25,
                                            stratify=data.target, random_state=args.seed)
    models = {}
    for classifier in ("knn", "logistic"):
        methods = comparison_models(random_state=args.seed, quick=args.quick,
                                    max_iter=args.max_iter, tol=args.tol, classifier=classifier,
                                    logistic_max_iter=args.logistic_max_iter,
                                    logistic_tol=args.logistic_tol,
                                    logistic_class_weight=class_weight)
        # The majority-class rule is shared by both comparison families.
        methods.pop("majority_class_logistic", None)
        models.update(methods)
    run_dir = new_run_directory(args.output, "classifier-smoke" if args.quick else "classifier-development")
    write_json(run_dir / "configuration.json", {
        "author": "Chang Cheng", "quick": args.quick, "seed": args.seed,
        "threads": args.threads, "cv": 5, "development_validation_fraction": 0.25,
        "max_iter": args.max_iter, "tol": args.tol,
        "logistic_max_iter": args.logistic_max_iter, "logistic_tol": args.logistic_tol,
        "logistic_class_weight": class_weight, "models": list(models),
        "run_kind": "correctness_smoke" if args.quick else "development_pilot",
    })
    write_json(run_dir / "provenance.json", provenance(args.threads))
    save_data_manifest(run_dir / "data.json", {
        "generator": "make_original_simulation", "metadata": data.metadata,
        "signal_support": data.signal_support, "training_indices": training,
        "validation_indices": validation, "role": "development_only_not_final_test",
    })
    print(f"Five-fold training calibration across {len(models)} methods; reports: {run_dir.resolve()}", flush=True)
    summary = run_comparison(models, data.data[training], data.target[training],
                             data.data[validation], data.target[validation],
                             output_dir=run_dir, seed=args.seed, threads=args.threads,
                             validation_ids=validation, signal_support=data.signal_support)
    print("\nMethod | validation balanced accuracy | tune (s) | MF (s) | classifier fit (s) | total fit (s) | predict (s) | status")
    for name, record in summary["models"].items():
        evaluation = record["evaluation"]
        score = evaluation.get("metrics", {}).get("balanced_accuracy")
        stages = evaluation.get("diagnostics", {}).get("fit_times", {})

        def formatted(value):
            return "-" if value is None else f"{value:.4f}"

        status = "majority fallback" if record["is_fallback"] else evaluation["status"]
        print(f"{name} | {formatted(score)} | {formatted(record['tuning_seconds'])} | "
              f"{formatted(stages.get('factorization'))} | {formatted(stages.get('classifier'))} | "
              f"{formatted(record['final_fit_seconds'])} | {formatted(evaluation.get('prediction_seconds'))} | {status}",
              flush=True)
    print("MF and classifier stage times are recorded for NMFClassifier; other methods report total fit time.")
    print("These development results do not establish a classifier default or complete tuning.")


if __name__ == "__main__":
    main()
