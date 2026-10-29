"""Log training-mask feasibility before an explicit grid. Author: Chang Cheng.

No decomposition, classifier, validation-array access or score-based selection
occurs here. A dimensionally eligible tuple can still fail numerical fitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from threadpoolctl import threadpool_limits
from supervised_nmf.evaluation import _digest, data_fingerprint, provenance
from supervised_nmf.screening import BinaryScreening
from plan_celeba_default_study import read, sha, key, box, write_new


def mask_counts(cache, attribute, seed, setting):
    X = np.load(Path(cache) / "train-X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(Path(cache) / f"train-{attribute}-y.npy", allow_pickle=False)
    manifest = read(Path(cache) / "manifest.json")
    if (manifest["official_test_loaded"] or manifest["seed"] != seed
            or manifest["splits"]["train"]["metadata"]["partition_code"] != 0
            or data_fingerprint(X, y) != manifest["splits"]["train"]["attributes"][attribute]["data_sha256"]):
        raise ValueError("Unexpected training cache identity")
    records = []
    with threadpool_limits(limits=1):
        for index, (train, validation) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
            if setting is None:
                count, digest, failure = X.shape[1], None, None
            else:
                try:
                    screen = BinaryScreening(**setting).fit(X[train], y[train])
                    mask = screen.get_support()
                    count, digest, failure = int(mask.sum()), _digest(np.flatnonzero(mask).tolist()), None
                except ValueError as error:
                    # Screening itself rejects an empty mask. Preserve its reason.
                    count, digest, failure = 0, None, str(error)
            records.append({"fold": index, "n_train": len(train), "n_validation": len(validation),
                            "selected_features": count, "mask_sha256": digest, "screening_failure": failure,
                            "training_indices_sha256": _digest(train.tolist())})
    return records


def screen_setting(parameters):
    fields = {name.rsplit("__", 1)[-1]: value for name, value in parameters.items()}
    if "percentile" in fields:
        return {"screening": "percentile", "percentile": fields["percentile"]}
    if "score_threshold" in fields:
        return {"screening": "absolute", "score_threshold": fields["score_threshold"]}
    return None


def materialize(protocol_path, output_path, audit_path):
    started = time.perf_counter()
    protocol = read(protocol_path)
    checked_masks, jobs = {}, []
    for job in protocol["jobs"]:
        kept, excluded, seen = [], [], set()
        for parameters in ParameterGrid(job["grid"]):
            signature = key(parameters)
            if signature in seen:
                continue
            seen.add(signature)
            setting = screen_setting(parameters)
            mask_key = key([job["cache"], job["attribute"], job["seed"], setting])
            if mask_key not in checked_masks:
                checked_masks[mask_key] = mask_counts(job["cache"], job["attribute"], job["seed"], setting)
            folds = checked_masks[mask_key]
            fields = {name.rsplit("__", 1)[-1]: value for name, value in parameters.items()}
            rank = fields.get("n_components", 1)
            feasible = all(row["selected_features"] > 0 and rank <= min(row["selected_features"], row["n_train"])
                           and fields["n_neighbors"] <= row["n_train"] for row in folds)
            if feasible:
                kept.append(box(parameters))
            else:
                excluded.append({"parameters": parameters, "mask_key": mask_key,
                                 "reason": "Empty fold mask, rank exceeds retained-feature/sample dimension, or k exceeds fold size"})
        if not kept:
            raise ValueError(f"No dimensionally eligible candidates: {job['id']}")
        job["grid"] = kept
        job["original_declared_candidates"] = len(seen)
        job["declared_candidates"] = len(kept)
        job["excluded_infeasible_candidates"] = len(excluded)
        unique_representations = {key({k: v for k, v in p.items() if not k.endswith("n_neighbors")})
                                  for p in ParameterGrid(kept)}
        job["nominal_representation_folds"] = 5 * len(unique_representations)
        jobs.append({"id": job["id"], "kept_candidates": len(kept), "excluded": excluded})
        print(f"{job['id']}: eligible {len(kept)}/{len(seen)}", flush=True)
    audit = {"author": "Chang Cheng", "input_protocol_sha256": sha(protocol_path),
             "source_sha256": provenance(1)["source_sha256"], "preflight_script_sha256": sha(__file__),
             "training_only": True, "decomposition_or_classifier_fits": 0, "official_test_loaded": False,
             "screening_fit_seconds_including_io": time.perf_counter() - started,
             "fold_masks": checked_masks, "jobs": jobs,
             "interpretation": "Necessary dimensional checks only; no convergence guarantee and no performance score was computed"}
    # The audit is immutable once its protocol is emitted; do not regenerate timings on resume.
    write_new(audit_path, audit)
    protocol["preflight"] = {"input_protocol_path": str(Path(protocol_path).resolve()),
                             "input_protocol_sha256": sha(protocol_path), "audit_path": str(Path(audit_path).resolve()),
                             "audit_sha256": sha(audit_path)}
    write_new(output_path, protocol)
    print(f"Total eligible: {sum(j['declared_candidates'] for j in protocol['jobs'])} candidates; "
          f"{sum(j['nominal_representation_folds'] for j in protocol['jobs'])} representation-fold attempts")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise ValueError("Preflight artifacts already exist; reuse them or choose new destinations")
    materialize(args.protocol, args.output, args.audit)


if __name__ == "__main__":
    main()
