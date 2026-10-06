"""Prepare CelebA development comparisons based on Chang Cheng's old script.

Images remain outside the package. This example loads official training and
validation partitions, fits/calibrates on training only, and saves validation
predictions. It never loads the official test partition. Final-test evaluation
must wait for a separate explicitly frozen configuration and completed tuning.

Example (replace local paths):
  python examples/celeba_classification.py --dir PATH/img_align_celeba \
      --attributes PATH/list_attr_celeba.txt --partitions PATH/list_eval_partition.txt --quick
Add --classifier logistic to compare NMF, PCA and feature baselines with logistic
regression. The default final classifier remains KNN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supervised_nmf.datasets import load_celeba
from supervised_nmf.evaluation import (
    comparison_models, new_run_directory, provenance, run_comparison, save_data_manifest, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="Directory of aligned/cropped image files.")
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--attribute", default="Smiling", choices=["Smiling", "Wearing_Necktie"])
    parser.add_argument("--max-train", type=int, default=1000)
    parser.add_argument("--max-validation", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--classifier", choices=("knn", "logistic"), default="knn")
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--logistic-tol", type=float, default=1e-4)
    parser.add_argument("--logistic-class-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--quick", action="store_true", help="Single-candidate correctness smoke; not completed tuning.")
    parser.add_argument("--grid-json", type=Path, help="JSON mapping model names to sklearn parameter grids.")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    logistic_class_weight = None if args.logistic_class_weight == "none" else "balanced"
    configuration = {"data_dir": str(args.dir.resolve()), "attributes": str(args.attributes.resolve()),
                     "partitions": str(args.partitions.resolve()), "attribute": args.attribute,
                     "max_train": args.max_train, "max_validation": args.max_validation,
                     "seed": args.seed, "threads": args.threads, "max_iter": args.max_iter,
                     "tol": args.tol, "quick": args.quick, "image_size": [64, 64], "cv": 5,
                     "classifier": args.classifier, "logistic_max_iter": args.logistic_max_iter,
                     "logistic_tol": args.logistic_tol, "logistic_class_weight": logistic_class_weight,
                     "grid_override": json.loads(args.grid_json.read_text(encoding="utf-8")) if args.grid_json else {}}
    if args.resume:
        run_dir = args.resume
        if json.loads((run_dir / "configuration.json").read_text(encoding="utf-8")) != configuration:
            raise ValueError("Resume settings differ from the existing configuration.")
    else:
        run_dir = new_run_directory(args.output, f"celeba-{args.attribute.lower()}-development")
        write_json(run_dir / "configuration.json", configuration)
        write_json(run_dir / "provenance.json", provenance(args.threads))
    training = load_celeba(args.dir, args.attributes, args.partitions, attribute=args.attribute,
                          split="train", max_samples=args.max_train, random_state=args.seed, image_size=(64, 64))
    validation = load_celeba(args.dir, args.attributes, args.partitions, attribute=args.attribute,
                            split="validation", max_samples=args.max_validation,
                            random_state=args.seed, image_size=(64, 64))
    if set(training.image_ids) & set(validation.image_ids):
        raise ValueError("Official training and validation image IDs overlap.")
    save_data_manifest(run_dir / "data.json", {"training_image_ids": training.image_ids,
                                        "validation_image_ids": validation.image_ids,
                                        "official_test_loaded": False,
                                        "identity_overlap_checked": False,
                                        "split_policy": "official_image_partitions"})
    models = comparison_models(random_state=args.seed, quick=args.quick,
                               max_iter=args.max_iter, tol=args.tol, classifier=args.classifier,
                               logistic_max_iter=args.logistic_max_iter, logistic_tol=args.logistic_tol,
                               logistic_class_weight=logistic_class_weight)
    unknown = set(configuration["grid_override"]) - set(models)
    if unknown:
        raise ValueError(f"Unknown grid model names: {sorted(unknown)}")
    for name, grid in configuration["grid_override"].items():
        models[name] = (models[name][0], grid)
    run_comparison(models, training.data, training.target, validation.data, validation.target,
                   output_dir=run_dir, seed=args.seed, threads=args.threads,
                   validation_ids=validation.image_ids)
    print(f"Development reports: {run_dir.resolve()}\nOfficial test data were not loaded. This run does not reselect package defaults.")


if __name__ == "__main__":
    main()
