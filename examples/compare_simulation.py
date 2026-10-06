"""Development comparisons based on Chang Cheng's original simulation script.

Run ``python examples/compare_simulation.py --quick --output runs`` for a small
correctness smoke. Neither quick nor the modest initial default grid completes
tuning or determines package defaults. Use --grid-json to supply expanded grids,
--seeds for replications, and --scenario/--correlation/--signal-strength for harder
designs. --resume RUN_DIRECTORY preserves compatible completed model results.
All metrics here use a held-out development subset; there is no final-test claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

# Also usable from a source checkout before installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supervised_nmf.datasets import make_simulation
from supervised_nmf.evaluation import (
    comparison_models, new_run_directory, provenance, run_comparison, save_data_manifest, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Correctness smoke only; NOT completed tuning.")
    parser.add_argument("--output", type=Path, default=Path("runs"), help="Parent directory for a unique run.")
    parser.add_argument("--resume", type=Path, help="Resume an exactly compatible existing run directory.")
    parser.add_argument("--grid-json", type=Path, help="JSON mapping model names to sklearn parameter grids.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--n-features", type=int)
    parser.add_argument("--n-signal", type=int)
    parser.add_argument("--signal-strength", type=float, default=2.0)
    parser.add_argument("--correlation", type=float, default=0.0)
    parser.add_argument("--correlation-scope", choices=["within_blocks", "all_features"], default="within_blocks",
                        help="Pre-clipping Gaussian correlation within signal/background or across all features.")
    parser.add_argument("--scenario", choices=["additive", "xor"], default="additive")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Replication seeds must be distinct.")
    config = {"quick": args.quick, "seeds": args.seeds,
              "n_samples": args.n_samples if args.n_samples is not None else (120 if args.quick else 400),
              "n_features": args.n_features if args.n_features is not None else (40 if args.quick else 120),
              "n_signal": args.n_signal if args.n_signal is not None else (8 if args.quick else 12),
              "signal_strength": args.signal_strength, "correlation": args.correlation,
              "correlation_scope": args.correlation_scope,
              "scenario": args.scenario, "threads": args.threads,
              "max_iter": args.max_iter, "tol": args.tol,
              "cv": 5, "development_validation_fraction": 0.25,
              "grid_override": json.loads(args.grid_json.read_text(encoding="utf-8")) if args.grid_json else {}}
    if args.resume:
        run_dir = args.resume
        previous = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Resume settings differ from the existing configuration.")
    else:
        run_dir = new_run_directory(args.output, "simulation-smoke" if args.quick else "simulation-development")
        write_json(run_dir / "configuration.json", config)
        write_json(run_dir / "provenance.json", provenance(args.threads))
    repetitions = []
    for seed in args.seeds:
        data = make_simulation(n_samples=config["n_samples"], n_features=config["n_features"],
                               n_signal=config["n_signal"], signal_strength=args.signal_strength,
                               correlation=args.correlation, random_state=seed, scenario=args.scenario,
                               correlation_scope=args.correlation_scope)
        training, validation = train_test_split(np.arange(len(data.target)), test_size=0.25,
                                                stratify=data.target, random_state=seed)
        seed_dir = run_dir / f"seed-{seed}"
        save_data_manifest(seed_dir / "data.json", {"generator": "make_simulation", "seed": seed,
                                            "signal_support": data.signal_support,
                                            "metadata": data.metadata,
                                            "training_indices": training, "validation_indices": validation,
                                            "role": "development_only_not_final_test"})
        models = comparison_models(random_state=seed, quick=args.quick, max_iter=args.max_iter, tol=args.tol)
        unknown = set(config["grid_override"]) - set(models)
        if unknown:
            raise ValueError(f"Unknown grid model names: {sorted(unknown)}")
        for name, grid in config["grid_override"].items():
            models[name] = (models[name][0], grid)
        print(f"Seed {seed}: five-fold development calibration across {len(models)} methods", flush=True)
        summary = run_comparison(models, data.data[training], data.target[training],
                                 data.data[validation], data.target[validation], output_dir=seed_dir,
                                 seed=seed, threads=args.threads, validation_ids=validation,
                                 signal_support=data.signal_support)
        repetitions.append(summary)
        for name, result in summary["models"].items():
            metric = result["evaluation"].get("metrics", {}).get("balanced_accuracy")
            print(f"  {name}: balanced accuracy={metric}, fallback={result['is_fallback']}, "
                  f"status={result['evaluation']['status']}", flush=True)
    aggregated = {}
    for name in repetitions[0]["models"]:
        records = [r["models"][name] for r in repetitions]
        scores = [r["evaluation"].get("metrics", {}).get("balanced_accuracy") for r in records]
        all_valid = all(r["evaluation"]["status"] == "valid" and not r["is_fallback"] for r in records)
        complete = all_valid and all(s is not None for s in scores)
        aggregated[name] = {"replications": len(records), "all_replications_valid": complete,
                            "fallback_count": sum(r["is_fallback"] for r in records),
                            "balanced_accuracy_values": scores,
                            "mean_balanced_accuracy": float(np.mean(scores)) if complete else None,
                            "standard_error_across_replications": float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
                            if complete and len(scores) > 1 else None}
    write_json(run_dir / "replications.json", {"author": "Chang Cheng", "models": aggregated,
                                               "defaults_frozen": False, "tuning_complete": False,
                                               "run_kind": "correctness_smoke" if args.quick else "development_pilot"})
    print(f"Reports: {run_dir.resolve()}\nScientific confirmation remains incomplete; no package defaults were selected.")


if __name__ == "__main__":
    main()
