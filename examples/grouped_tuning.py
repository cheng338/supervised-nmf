"""Small subject-separated calibration example. Author: Chang Cheng."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from supervised_nmf import NMFClassifier
from supervised_nmf.evaluation import (
    _cv_joblib_settings, _digest, _serial_timing_phase, _timing_study_lock,
    data_fingerprint, evaluate_classifier, format_evaluation_summary,
)
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import _cv_wall_attempt, _read_receipt, _write_receipt


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with _timing_study_lock():
        run(args.output)


def run(output):
    output = Path(output)
    rng = np.random.RandomState(42)
    subjects = np.repeat(np.arange(40), 4)
    y = np.tile([0, 0, 1, 1], 40)
    X = rng.uniform(0.1, 1.0, (160, 16))
    X[:, :4] += y[:, None] * 0.7
    ids = np.arange(len(y))
    training, evaluation = subjects < 30, subjects >= 30
    estimator = NMFClassifier(screening="absolute", random_state=42)
    train_X, train_y = X[training], y[training]
    grid = {"n_components": [2, 4], "score_threshold": [0.0, 2.0], "n_neighbors": [3, 5]}
    options = dict(cv=5, seed=42, threads=1, groups=subjects[training], data_ids=ids[training],
                   max_expansions=1, max_rank=8, checkpoint_dir=output)
    joblib = _cv_joblib_settings()
    config = _digest({"grid": grid, "seed": 42, "cv": 5, "max_expansions": 1, "max_rank": 8,
                      "threads": 1, "joblib": joblib, "data": data_fingerprint(X, y), "subjects": subjects})
    journal = output / "grouped-timing"
    history = [_read_receipt(path) for path in journal.glob("*.json")]
    if any(row["configuration_sha256"] != config for row in history):
        raise ValueError("Grouped timing evidence belongs to different data or CV settings.")
    seal = output / "grouped-final-cv.json"
    frozen = seal.exists() or any("fit_execution" in row for row in history)
    prior_work = (output / "identity.json").exists() or any(output.glob("stage-*"))
    path = journal / f"{uuid.uuid4().hex}.json"
    record = {"configuration_sha256": config, "host": platform.node(), "pid": os.getpid(),
              "state": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "cv_joblib": joblib, "cv": []}
    _write_receipt(path, record)

    def search(*, verify):
        with _cv_wall_attempt(output, config, active=not verify, workers=1, native_threads=1,
                              preexisting_work=prior_work) as cv:
            cv.update(operation="verify_completed_folds" if verify else "calibrate", joblib=joblib,
                      invocation_id=path.stem, host=record["host"], pid=record["pid"])
            record["cv"].append(cv)
            _write_receipt(path, record)
            result = calibration_rank_search(estimator, grid, train_X, train_y, **options,
                                             _require_completed=verify)
            cv["complete"] = result is not None
            if result is not None:
                cv["stage_threadpools"] = [stage["active_threadpools"] for stage in result["stages"]]
            return result

    try:
        report = search(verify=True) if frozen or prior_work else None
        if report is None:
            if frozen:
                raise ValueError("Grouped CV folds are missing after final fitting began. Start a new study.")
            report = search(verify=False)
        print("Rank-search stopping reason:", report["stop_reason"])
        if report["selected_parameters"] is None:
            raise RuntimeError("No numerically qualified training candidate.")
        pin = {"configuration_sha256": config, "rank_sha256": report["fingerprint"],
               "selected_parameters": report["selected_parameters"],
               "folds_sha256": _digest([stage["representation_folds"] for stage in report["stages"]])}
        if frozen:
            if not seal.is_file() or _read_receipt(seal) != pin:
                raise ValueError("Grouped CV evidence changed after final fitting began.")
        else:
            _write_receipt(seal, pin)
        estimator.set_params(**report["selected_parameters"])
        record["fit_execution"] = {}
        _write_receipt(path, record)
        with _serial_timing_phase(record["fit_execution"]):
            estimator.fit(train_X, train_y)
        if not estimator.converged_:
            raise RuntimeError("The selected final fit did not converge.")
        record["prediction_execution"] = {}
        _write_receipt(path, record)
        result = evaluate_classifier(
            estimator, X[evaluation], y[evaluation], threads=1,
            sample_ids=ids[evaluation], train_sample_ids=ids[training],
            groups=subjects[evaluation], train_groups=subjects[training], train_y=train_y,
            _serial_timing=record["prediction_execution"],
        )
        record["evaluation_status"] = result["status"]
        record["state"] = "failed" if result["status"] == "failed" else "completed"
        if "error" in result:
            record["error"] = result["error"]
        print(format_evaluation_summary({
            "evaluation": result, "tuning_seconds": record["cv"][-1]["cv_wall_seconds_cumulative"],
            "final_fit_seconds": record["fit_execution"]["seconds"],
        }))
    except BaseException as error:
        record.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_receipt(path, record)
        print(json.dumps({"timing_receipt": str(path), "state": record["state"]}))


if __name__ == "__main__":
    main()
