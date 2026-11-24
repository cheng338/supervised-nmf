"""Public CLI regressions for admission and recovery of a trusted CelebA study."""

from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from supervised_nmf import study as calibration
from supervised_nmf.evaluation import _checkpoint_file
from tests._celeba_recovery_test_support import hashes, receipt, study


HEADERS = ("configuration.json", "data.json", "provenance.json")


class ReachedDispatch(Exception):
    pass


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_failed_parameter_worker_allows_cache_replacement(tmp_path):
    feature_file = tmp_path / "train-X.npy"
    expected = np.arange(30, dtype=np.float32).reshape(10, 3)
    np.save(feature_file, expected)
    np.save(tmp_path / "train-Smiling-y.npy", np.arange(10) % 2)
    replacement = tmp_path / "replacement.npy"
    np.save(replacement, np.zeros_like(expected))
    settings = read(Path(study.__file__).with_name("celeba_starting_grid.json"))
    job = (tmp_path, tmp_path / "run", "Smiling", "raw_knn", "chunk-000",
           {"n_neighbors": [0]}, settings, 17, 1, study._source_hash())

    with patch.object(calibration, "_check_labeled_data", wraps=calibration._check_labeled_data) as validate:
        with pytest.raises(ValueError, match="n_neighbors value must be a positive integer") as failure:
            study._worker(job)
    assert failure.value.__traceback__ is not None
    replacement.replace(feature_file)
    received = validate.call_args.args[0]
    owner = received
    while isinstance(owner, np.ndarray):
        mapping = getattr(owner, "_mmap", None)
        assert mapping is None, "Calibration input remains backed by the worker's map"
        owner = owner.base
    assert not received.flags.writeable
    np.testing.assert_array_equal(received, expected)


def invoke(inputs, run, cache, *, fresh=False, counts=(8, 4), grid=None, policy="off",
           execute=False, threads=1):
    argv = ["study", "--dir", str(inputs / "images"), "--attributes", str(inputs / "labels.csv"),
            "--partitions", str(inputs / "partitions.txt"), "--data-cache", str(cache),
            "--grid-json", str(grid or inputs / "grid.json"), "--workers", "1", "--threads", str(threads),
            "--max-train", str(counts[0]), "--max-validation", str(counts[1]),
            "--final-fit-recovery", policy, "--output" if fresh else "--resume", str(run)]
    dispatch = (nullcontext() if execute else
                patch.object(study, "_completed_or_spawned", side_effect=ReachedDispatch))
    with patch.object(sys, "argv", argv), dispatch:
        study.main()


def initialize(inputs, root, cache, **options):
    with pytest.raises(ReachedDispatch):
        invoke(inputs, root, cache, fresh=True, **options)
    run, = root.iterdir()
    return run


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    image = pytest.importorskip("PIL.Image")
    root = tmp_path_factory.mktemp("faces")
    (root / "images").mkdir()
    labels = ["image_id,Smiling,Wearing_Necktie"]
    partitions = []
    for i in range(12):
        name = f"{i:02}.png"
        image.new("L", (3, 3), color=i * 20).save(root / "images" / name)
        labels.append(f"{name},{i % 2},{(i // 2) % 2}")
        partitions.append(f"{name} {int(i >= 8)}")
    (root / "labels.csv").write_text("\n".join(labels), encoding="utf-8")
    (root / "partitions.txt").write_text("\n".join(partitions), encoding="utf-8")
    write(root / "grid.json", {"n_components": [2], "percentile": [5], "score_threshold": [0],
                               "n_neighbors": [1], "max_iter": 15, "tol": 0.1, "annotation": None})
    return root


@pytest.fixture(scope="module")
def initialized(inputs, tmp_path_factory):
    root = tmp_path_factory.mktemp("cr")
    return initialize(inputs, root / "runs", root / "cache")


@pytest.fixture
def run(initialized, tmp_path):
    return shutil.copytree(initialized, tmp_path / "run")


def reject_before_cache(inputs, run, cache, capsys, message, **options):
    before = hashes(run)
    with (patch.object(study, "prepare_cache", side_effect=AssertionError("cache preparation started")),
          pytest.raises(SystemExit) as error):
        invoke(inputs, run, cache, **options)
    assert error.value.code == 2 and message in capsys.readouterr().err
    assert not cache.exists() and hashes(run) == before


@pytest.mark.parametrize("missing", [None, *HEADERS])
def test_resume_requires_an_initialized_study(tmp_path, capsys, missing):
    run = tmp_path / "typo"
    if missing:
        run.mkdir()
        for name in HEADERS:
            if name != missing:
                write(run / name, {})
    grid = Path(study.__file__).with_name("celeba_starting_grid.json")
    reject_before_cache(tmp_path, run, tmp_path / "cache", capsys, "initialized study", grid=grid)
    assert run.exists() == bool(missing)


@pytest.mark.parametrize("filename,key,value", [
    ("grid.json", "score_threshold", [0.0]),
    ("configuration.json", "threads", True),
    ("provenance.json", "native_threads_requested", 1.0),
    ("provenance.json", "native_threads_requested", True),
    ("provenance.json", "native_threads_requested", 2),
])
def test_resume_checks_request_and_original_record_types(inputs, run, tmp_path, capsys,
                                                        filename, key, value):
    grid = tmp_path / "grid.json"
    shutil.copyfile(inputs / "grid.json", grid)
    path = grid if filename == grid.name else run / filename
    record = read(path)
    record[key] = value
    write(path, record)
    reject_before_cache(inputs, run, tmp_path / "cache", capsys, "incompatible", grid=grid)


@pytest.mark.parametrize("file,token", [("grid.json", "NaN"), ("configuration.json", "1e999")])
def test_nonfinite_numbers_do_not_alias_json_null(inputs, run, tmp_path, capsys, file, token):
    grid = tmp_path / "grid.json"
    shutil.copyfile(inputs / "grid.json", grid)
    path = grid if file == grid.name else run / file
    record = read(path)
    settings = record if file == grid.name else record["grid"]
    assert settings["annotation"] is None
    placeholder = "replace this marker with a raw JSON number"
    settings["annotation"] = placeholder
    raw = json.dumps(record).replace(json.dumps(placeholder), token, 1)
    path.write_text(raw, encoding="utf-8")
    reject_before_cache(inputs, run, tmp_path / "cache", capsys, "nonfinite", grid=grid)


@pytest.mark.parametrize("counts,flag", [((7, 4), "--max-train=7"), ((8, 3), "--max-validation=3")])
def test_resume_checks_both_selected_sample_counts(inputs, run, tmp_path, capsys, counts, flag):
    reject_before_cache(inputs, run, tmp_path / "cache", capsys, flag, counts=counts)


@pytest.mark.parametrize("split,limits", [("train", (9, 4)), ("validation", (8, 5))])
def test_original_capped_cache_is_reusable_but_cannot_be_rebuilt_at_selected_count(
        inputs, tmp_path, capsys, split, limits):
    warm, runs = tmp_path / "warm", tmp_path / "runs"
    with pytest.raises(ValueError, match="Cached sample count"):
        invoke(inputs, runs, warm, fresh=True, counts=limits)
    assert not runs.exists()
    run = initialize(inputs, runs, warm)
    before = hashes(run), hashes(warm)
    with pytest.raises(ReachedDispatch):
        invoke(inputs, run, warm)
    assert (hashes(run), hashes(warm)) == before
    reject_before_cache(inputs, run, tmp_path / "cold", capsys, "identically rebuilt cache")
    i = ("train", "validation").index(split)
    record = read(run / "data.json")["splits"][split]
    assert record["metadata"]["max_samples"] == limits[i] > len(record["image_ids"])


@pytest.mark.parametrize("filename", HEADERS[:2])
def test_resume_respects_a_manifest_writer_then_allows_retry(inputs, run, tmp_path, filename):
    cache = tmp_path / "cache"
    before = hashes(run)
    with _checkpoint_file(run / filename), pytest.raises(RuntimeError, match="writing this checkpoint"):
        invoke(inputs, run, cache)
    assert (cache / "manifest.json").is_file() and hashes(run) == before
    with pytest.raises(ReachedDispatch):
        invoke(inputs, run, cache)
    assert hashes(run) == before


@pytest.mark.parametrize("filename,key", [
    ("configuration.json", "threads"), ("data.json", "seed"),
    ("provenance.json", "native_threads_requested"),
])
def test_resume_rereads_headers_after_cache_preparation(inputs, run, tmp_path, filename, key):
    path = run / filename
    expected = {}
    prepare = study.prepare_cache

    def change_header(*args):
        manifest = prepare(*args)
        record = read(path)
        record[key] = float(record[key])
        write(path, record)
        expected.update(hashes(run))
        return manifest

    with (patch.object(study, "prepare_cache", change_header),
          pytest.raises(ValueError, match="metadata changed during startup")):
        invoke(inputs, run, tmp_path / "cache")
    assert hashes(run) == expected


def test_resume_still_checks_cached_arrays(inputs, initialized, run, tmp_path):
    cache = shutil.copytree(initialized.parent.parent / "cache", tmp_path / "cache")
    path = cache / "train-Smiling-y.npy"
    labels = np.load(path, allow_pickle=False)
    labels[0] ^= 1
    np.save(path, labels, allow_pickle=False)
    before = hashes(run), hashes(cache)
    with pytest.raises(ValueError, match="fingerprint"):
        invoke(inputs, run, cache)
    assert (hashes(run), hashes(cache)) == before


def test_warm_and_cold_resume_recover_the_fit_and_preserve_the_saved_identity(
        inputs, tmp_path_factory, monkeypatch):
    # Keep the real cache, recovery store, final fitting and evaluation in this CLI path.
    root = tmp_path_factory.mktemp("rr")
    model = study.study_models(read(inputs / "grid.json"), 42)["majority_class"]
    monkeypatch.setattr(study, "study_models", lambda settings, seed: {"majority_class": model})
    monkeypatch.setattr(study, "ATTRIBUTES", ("Smiling",))
    warm, cold = root / "warm", root / "cold"
    run = initialize(inputs, root / "runs", warm, policy="all", threads=2).rename(root / "s")
    for name in HEADERS:
        path = run / name
        write(path, dict(reversed(read(path).items())))
    original = {name: (run / name).read_bytes() for name in HEADERS}
    result_path = run / "Smiling" / "majority_class" / "development-evaluation.json"

    def interrupt(event, details=None):
        if event == "manifest-commit":
            raise KeyboardInterrupt

    with patch.object(study.recovery_store, "_hook", interrupt), pytest.raises(KeyboardInterrupt):
        invoke(inputs, run, warm, policy="all", execute=True, threads=2)
    assert not result_path.exists()
    with patch.object(DummyClassifier, "fit", side_effect=AssertionError("recovery refitted")) as fit:
        invoke(inputs, run, cold, policy="all", execute=True, threads=2)
        fit.assert_not_called()
    result = receipt(result_path)
    assert result["run_identity"] == study._digest(json.loads(original["configuration.json"]))
    assert result["evaluation"]["status"] == "valid" and result["recovery"]["fit_reused"]
    assert read(run / "configuration.json")["threads"] == 2
    for phase in ("fit_execution", "prediction_execution"):
        assert result[phase]["application_workers"] == result[phase]["native_threads"] == 1
        assert result[phase]["started_utc"] <= result[phase]["finished_utc"]
    assert result["fit_execution"]["finished_utc"] <= result["prediction_execution"]["started_utc"]
    assert hashes(warm) == hashes(cold)
    completed = result_path.read_bytes()
    with patch.object(study, "evaluate_classifier", side_effect=AssertionError("evaluated again")) as evaluate:
        invoke(inputs, run, warm, policy="all", execute=True, threads=2)
        evaluate.assert_not_called()
    assert result_path.read_bytes() == completed
    assert {name: (run / name).read_bytes() for name in HEADERS} == original


def test_cli_cv_wall_survives_partial_resume_final_failure_and_completed_replay(inputs, tmp_path, monkeypatch):
    settings = read(inputs / "grid.json")
    model = study.study_models(settings, 42)["raw_knn"]
    monkeypatch.setattr(study, "study_models", lambda *args: {"raw_knn": model})
    cache, runs = tmp_path / "cache", tmp_path / "runs"
    key = "1" * 64

    def workers(jobs, count, *, _pins=None):
        for index, (_, output, attribute, method, chunk, *_) in enumerate(jobs):
            directory = Path(output) / attribute / method / chunk
            if index == 1 and not (cache / "continue").exists():
                raise KeyboardInterrupt
            report = {"state": "cv_completed", "active_threadpools": [{"internal_api": "blas", "num_threads": 1}],
                      "representation_folds": [{"key": key, "state": "completed"}],
                      "current_call_wall_seconds": 1., "new_representation_fits": 1, "reused_representation_folds": 0}
            calibration._write_receipt(directory / "report.json", report)
            calibration._write_receipt(directory / "folds" / f"{key}.json", {})
            yield {"attribute": attribute, "method": method, "chunk": chunk, "valid_candidates": 1, "candidates": 1}

    monkeypatch.setattr(study, "_completed_or_spawned", workers)
    with pytest.raises(KeyboardInterrupt):
        invoke(inputs, runs, cache, fresh=True, execute=True)
    run, = runs.iterdir()
    first_path, = (run / "cv-timing").glob("*.json")
    first_bytes, first = first_path.read_bytes(), receipt(first_path)
    assert first["state"] == "interrupted" and first["cv_wall_seconds_this_invocation"] > 0
    (cache / "continue").touch()
    with (patch.object(study, "aggregate_and_evaluate", side_effect=RuntimeError("after CV")),
          pytest.raises(RuntimeError, match="after CV")):
        invoke(inputs, run, cache, execute=True)
    second_path, = set((run / "cv-timing").glob("*.json")) - {first_path}
    second = receipt(second_path)
    assert second["state"] == "completed" and second["cv_wall_seconds_this_invocation"] > 0
    assert second["cv_wall_seconds_cumulative"] == first["cv_wall_seconds_this_invocation"] + second["cv_wall_seconds_this_invocation"]
    with patch.object(study, "aggregate_and_evaluate", return_value={}):
        invoke(inputs, run, cache, execute=True)
    comparison = read(run / "comparison.json")
    assert comparison["cv_wall_seconds_this_invocation"] == 0.
    assert comparison["cv_wall_seconds_cumulative"] == second["cv_wall_seconds_cumulative"]
    assert first_path.read_bytes() == first_bytes


@pytest.mark.parametrize("artifact", ["Smiling/raw_knn/.fit-recovery/attempts/one/start.json",
                                      "Smiling/raw_knn/development-evaluation.json",
                                      "profiles/profile-1/profiling.json"])
def test_orphaned_final_evidence_blocks_parameter_cv_before_dispatch(inputs, run, tmp_path, artifact):
    path = run / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, {})
    with patch.object(study, "_completed_or_spawned") as dispatch, pytest.raises(ValueError, match="fresh CV is closed"):
        invoke(inputs, run, tmp_path / "cache", execute=True)
    dispatch.assert_not_called()


def test_missing_prior_cv_wall_stays_unknown_after_resume(tmp_path):
    source, killed = tmp_path / "original", tmp_path / "killed"
    with calibration._cv_wall_attempt(source, "config", active=True, workers=2, native_threads=1):
        # Capture exactly the state visible to a new process after a hard stop
        # inside a running CV call, before either terminal record is written.
        shutil.copytree(source, killed)
    for active in (True, False):
        with calibration._cv_wall_attempt(killed, "config", active=active, workers=2, native_threads=1) as current:
            pass
        assert current["cv_wall_seconds_cumulative"] is None and current["cv_wall_seconds_recorded"] > 0.

    # Restoring only a previously observed running file in a completed journal
    # is different from restoring an entire genuine in-flight checkpoint.
    with calibration._cv_wall_attempt(source, "config", active=True, workers=2, native_threads=1) as latest:
        path = source / "cv-timing" / (latest["attempt_id"] + ".json")
        running = path.read_bytes()
    path.write_bytes(running)
    with pytest.raises(ValueError, match="terminal record"):
        calibration._validated_cv_wall_receipts(source, "config")


@pytest.mark.parametrize("stop_before", ["register", "promote", "running"])
def test_cv_wall_recovers_interrupted_admission(tmp_path, stop_before):
    def attempt():
        return calibration._cv_wall_attempt(tmp_path, "config", active=True, workers=2, native_threads=1)

    # All three writes precede entry into the CV body.
    registered = stop_before != "register"
    write_receipt = calibration._write_receipt
    replace = Path.replace

    def stop_before_promotion(path, target):
        if stop_before == "promote" and path.name == ".starting":
            raise KeyboardInterrupt
        return replace(path, target)

    def stop_before_publication(path, record):
        if (stop_before == "running" and path.suffix == ".json") or (stop_before == "register" and path.name == "attempts.index"):
            raise KeyboardInterrupt
        write_receipt(path, record)

    with patch.object(Path, "replace", stop_before_promotion), \
            patch.object(calibration, "_write_receipt", side_effect=stop_before_publication), \
            pytest.raises(KeyboardInterrupt), attempt():
        pytest.fail("CV ran before admission finished")
    before = calibration._validated_cv_wall_receipts(tmp_path, "config")
    assert len(before) == int(registered)
    if registered:
        assert (before[0][0].name == ".starting") == (stop_before == "promote")
    with attempt() as result:
        pass
    after = calibration._validated_cv_wall_receipts(tmp_path, "config")
    assert len(after) == int(registered) + 1
    assert all(path.name == row["attempt_id"] + ".json" for path, row in after)
    if registered:
        assert after[0][1]["state"] == "admitted" and after[0][1]["cv_wall_seconds_this_invocation"] == 0.
    assert result["cv_wall_seconds_cumulative"] == result["cv_wall_seconds_this_invocation"]
    # Missing even the most recent attempt is detectable before the next dispatch.
    after[-1][0].unlink()
    with pytest.raises(ValueError, match="Missing registered"), attempt():
        pytest.fail("CV ran with an incomplete journal")


@pytest.mark.parametrize("stop_at_index", [False, True])
def test_failed_cv_receipt_publication_only_exposes_durable_terminal_time(tmp_path, stop_at_index):
    write = calibration._write_receipt

    def interrupted(path, payload):
        at_index = path.name == "attempts.index" and payload.get("terminal_sha256") is not None
        at_receipt = path.suffix == ".json" and "finished_utc" in payload
        if (stop_at_index and at_index) or (not stop_at_index and at_receipt):
            raise OSError("interrupted publication")
        write(path, payload)

    def attempt(active):
        return calibration._cv_wall_attempt(tmp_path, "config", active=active, workers=2, native_threads=1)

    with patch.object(calibration, "_write_receipt", side_effect=interrupted), \
            pytest.raises(OSError, match="interrupted publication"), attempt(True) as record:
        pass
    _, saved = calibration._validated_cv_wall_receipts(tmp_path, "config")[0]
    assert ("finished_utc" in record) == ("finished_utc" in saved) == stop_at_index
    with attempt(False) as replay:
        pass
    assert replay["cv_wall_seconds_cumulative"] == saved["cv_wall_seconds_this_invocation"]
