"""Small training-only numerical smoke check before Chang Cheng's long study.

This does not compute classification scores or open development/test holdouts.
It is an implementation check, not evidence for selecting a solver or tuning.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import fashion_mnist_study as study
from supervised_nmf.evaluation import _jsonable, _warning_records, write_json


def check(output):
    protocol = json.loads((ROOT / "examples" / "fashion_mnist_solver_protocol_v2.json").read_text(encoding="utf-8"))
    X, y, identity = study._dataset(protocol)
    split = study.make_splits(y, protocol)["tshirt-shirt"]
    rows = split["training_indices"]["n2000-seed43"][:512]
    X, y = np.asarray(X[rows]), (y[rows] == 6).astype(np.uint8)
    records = []
    for rank in (5, 20):
        for method in protocol["methods"]:
            job = {"method": method, "seed": 43, "max_iter": protocol["max_iter"], "tol": protocol["tol"]}
            for key in ("anls_max_iter", "anls_nnls_max_iter", "anls_nnls_tol"):
                if key in protocol:
                    job[key] = protocol[key]
            model = study._estimator(job)
            parameters = ({"screen__score_threshold": 2, "nmf__n_components": rank, "knn__n_neighbors": 5}
                          if isinstance(model, Pipeline) else
                          {"score_threshold": 2, "n_components": rank, "n_neighbors": 5})
            model.set_params(**parameters)
            record = {"method": method, "rank": rank, "status": "failed"}
            start = time.perf_counter()
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    model.fit(X, y)
                    record["fit_seconds"] = time.perf_counter() - start
                    if isinstance(model, Pipeline):
                        factor = model.named_steps["nmf"]
                        W = model.named_steps["knn"]._fit_X
                        selected = model.named_steps["screen"].transform(X)
                        projected = model[:-1].transform(X[:16])
                    else:
                        factor = model.decomposition_
                        W = model.training_coordinates_
                        selected = X[:, model.selected_features_]
                        projected = model.transform(X[:16])
                    if not all(np.isfinite(a).all() and np.min(a) >= 0 for a in (W, factor.components_, projected)):
                        raise ValueError("Nonfinite or negative factor/projection.")
                    residual = float(np.linalg.norm(np.asarray(selected, dtype=np.float64) - np.asarray(W, dtype=np.float64) @ factor.components_))
                    record.update(status="finite_nonnegative", basis_dtype=str(factor.components_.dtype),
                                  training_dtype=str(W.dtype), n_iter=_jsonable(getattr(factor, "n_iter_", None)),
                                  reconstruction_norm=residual,
                                  converged=_jsonable(getattr(factor, "converged_", getattr(model, "converged_", None))))
                except Exception as error:
                    record.update(error_type=type(error).__name__, error=str(error))
            record.update(warnings=_warning_records(caught), total_check_seconds=time.perf_counter() - start,
                          numerically_qualified=record["status"] == "finite_nonnegative" and not caught
                          and not study._model_nonconverged(model, caught))
            records.append(record)
            print(json.dumps(record), flush=True)
            write_json(output, {"author": "Chang Cheng", "created_utc": datetime.now(timezone.utc).isoformat(),
                               "role": "implementation_smoke_on_512_declared_training_rows_only",
                               "classification_scores_computed": False, "official_test_loaded": False,
                               "development_validation_evaluated": False,
                               "dataset_manifest_sha256": identity["manifest_file_sha256"],
                               "sources": study._sources(), "records": records})
    return all(record["numerically_qualified"] for record in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "four-solvers-audit" / "training-smoke.json")
    raise SystemExit(0 if check(parser.parse_args().output) else 1)
