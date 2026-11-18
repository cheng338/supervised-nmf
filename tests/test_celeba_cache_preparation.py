"""Interrupted CelebA cache preparation, immutable reuse and file ownership."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from supervised_nmf.evaluation import data_fingerprint


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def preparer():
    spec = importlib.util.spec_from_file_location(
        "celeba_cache_preparer_test", ROOT / "scripts/prepare_celeba_default_caches.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def files(preparer, directory):
    return {path.relative_to(directory): preparer.sha(path)
            for path in directory.rglob("*") if path.is_file()}


def rename_roundtrip(path):
    moved = path.with_name("handle-check.npy")
    path.rename(moved)
    moved.rename(path)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    # Only the 16 training images need pixels. The preparer must copy the fixed
    # validation cache and must never decode validation or official-test images.
    Image = pytest.importorskip("PIL.Image", reason="optional Pillow dependency unavailable")
    root = tmp_path_factory.mktemp("cache-input")
    images, validation = root / "images", root / "validation"
    images.mkdir()
    validation.mkdir()
    attributes, partitions = root / "labels.csv", root / "partitions.txt"
    training_ids = [f"{index:06d}.jpg" for index in range(1, 17)]
    validation_ids = [f"{index:06d}.jpg" for index in range(17, 4017)]
    all_ids = training_ids + validation_ids + ["900001.jpg"]
    attributes.write_text("image_id,Smiling,Wearing_Necktie\n" + "".join(
        f"{name},{1 if index % 2 else -1},{1 if (index // 2) % 2 else -1}\n"
        for index, name in enumerate(all_ids)), encoding="utf-8")
    partitions.write_text("".join(f"{name} {0 if index < 16 else 1 if index < 4016 else 2}\n"
                                 for index, name in enumerate(all_ids)), encoding="utf-8")
    for index, name in enumerate(training_ids):
        Image.new("RGB", (7, 9), color=(index * 13, index * 7, index * 5)).save(images / name)
    # This fixed shape is a production input contract. Keep one shared fixture
    # rather than weakening shape checks or generating 4,000 redundant JPEGs.
    values = np.full((4000, 4096), 0.25, dtype=np.float32)
    np.save(validation / "validation-X.npy", values, allow_pickle=False)
    record = {"metadata": {"random_state": 42, "partition_code": 1},
              "image_ids": validation_ids, "attributes": {}}
    for attribute, labels in (("Smiling", np.arange(16, 4016) % 2),
                              ("Wearing_Necktie", (np.arange(16, 4016) // 2) % 2)):
        labels = labels.astype(np.int64)
        np.save(validation / f"validation-{attribute}-y.npy", labels, allow_pickle=False)
        record["attributes"][attribute] = {
            "class_counts": np.bincount(labels, minlength=2).tolist(),
            "data_sha256": data_fingerprint(values, labels)}
    (validation / "manifest.json").write_text(json.dumps({
        "seed": 42, "official_test_loaded": False, "splits": {"validation": record}}), encoding="utf-8")
    return SimpleNamespace(images=images, attributes=attributes, partitions=partitions, validation=validation)


@pytest.fixture
def case(preparer, inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    monkeypatch.setattr(preparer, "SIZES", (8,))
    monkeypatch.setattr(preparer, "SEEDS", (43,))
    dataset = tmp_path / "src/supervised_nmf/datasets.py"
    dataset.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "src/supervised_nmf/datasets.py", dataset)
    calls = []
    loader = preparer.load_celeba

    def tracked(*args, **kwargs):
        calls.append(kwargs["max_samples"])
        return loader(*args, **kwargs)

    monkeypatch.setattr(preparer, "load_celeba", tracked)
    output = tmp_path / "out"

    def run(**kwargs):
        return preparer.prepare(inputs.images, inputs.attributes, inputs.partitions,
                                kwargs.pop("validation", inputs.validation), output, **kwargs)

    return SimpleNamespace(module=preparer, root=tmp_path, output=output, run=run,
                           calls=calls, loader=loader, inputs=inputs)


def test_ordinary_nested_sampling_and_immutable_completed_reuse(case, monkeypatch):
    module = case.module
    monkeypatch.setattr(module, "SIZES", (4, 8))
    result = case.run()
    assert case.calls == [8] and result["recovered_training_payloads"] == []
    assert result["new_largest_sample_decode_calls"] == 1
    for size in (4, 8):
        cache = case.output / f"n{size}-seed43"
        reference = case.loader(case.inputs.images, case.inputs.attributes, case.inputs.partitions,
                                attribute="Smiling", split="train", max_samples=size,
                                random_state=43, image_size=(64, 64))
        manifest = module.read(cache / "manifest.json")
        assert manifest["splits"]["train"]["image_ids"] == reference.image_ids.tolist()
        np.testing.assert_array_equal(np.load(cache / "train-X.npy"), reference.data)
        np.testing.assert_array_equal(np.load(cache / "train-Smiling-y.npy"), reference.target)
        for name in module.ARRAY_NAMES:
            if name.startswith("validation-"):
                assert module.sha(cache / name) == module.sha(case.inputs.validation / name)
    before = {size: files(module, case.output / f"n{size}-seed43") for size in (4, 8)}
    with monkeypatch.context() as guard:
        guard.setattr(module, "load_celeba", lambda *a, **k: pytest.fail("Completed caches were decoded again"))
        resumed = case.run()
    assert resumed["new_largest_sample_decode_calls"] == 0
    assert [row["action"] for row in resumed["caches"]] == ["verified_existing", "verified_existing"]
    assert before == {size: files(module, case.output / f"n{size}-seed43") for size in (4, 8)}


class PreparationInterrupted(KeyboardInterrupt):
    pass


@pytest.mark.parametrize("point,decode_again,payload_reused", [
    ("train-save", True, False), ("receipt-before", True, False),
    ("receipt-after", False, True), ("manifest-before", False, True),
    ("manifest-after", False, False), ("publish-after", False, False),
    ("validation-copy", False, True),
])
def test_recovery_uses_only_committed_work(case, monkeypatch, point, decode_again, payload_reused):
    module = case.module
    save, receipt, write, publish, copy = (np.save, module._write_training_receipt,
                                         module.write_new, module._publish_cache, module.shutil.copy2)

    def fail():
        raise PreparationInterrupted(point)

    def saved(path, *args, **kwargs):
        result = save(path, *args, **kwargs)
        if point == "train-save" and Path(path).name == "train-X.npy":
            fail()
        return result

    def receipted(*args, **kwargs):
        if point == "receipt-before":
            fail()
        result = receipt(*args, **kwargs)
        if point == "receipt-after":
            fail()
        return result

    def written(path, value):
        if point == "manifest-before" and Path(path).name == "manifest.json":
            fail()
        result = write(path, value)
        if point == "manifest-after" and Path(path).name == "manifest.json":
            fail()
        return result

    def published(*args, **kwargs):
        result = publish(*args, **kwargs)
        if point == "publish-after":
            fail()
        return result

    def copied(source, destination, *args, **kwargs):
        result = copy(source, destination, *args, **kwargs)
        if point == "validation-copy" and Path(destination).name == "validation-X.npy":
            fail()
        return result

    with monkeypatch.context() as guard:
        guard.setattr(np, "save", saved)
        guard.setattr(module, "_write_training_receipt", receipted)
        guard.setattr(module, "write_new", written)
        guard.setattr(module, "_publish_cache", published)
        guard.setattr(module.shutil, "copy2", copied)
        with pytest.raises(PreparationInterrupted):
            case.run()
    assert case.calls == [8]
    before = files(module, case.output)
    report = case.run()
    assert case.calls == ([8, 8] if decode_again else [8])
    assert bool(report["recovered_training_payloads"]) is payload_reused
    final = case.output / "n8-seed43"
    manifest = module.read(final / "manifest.json")
    assert manifest["preparation"]["direct_training_decode"] is (not payload_reused)
    relocated = [(module.ROOT / row["stage"] / "c", module.ROOT / row["cache"])
                 for row in report["recovered_complete_stages"]]
    for relative, digest in before.items():
        path = case.output / relative
        for old, new in relocated:
            if path.is_relative_to(old):
                path = new / path.relative_to(old)
                break
        assert module.sha(path) == digest


@pytest.fixture
def payload(preparer, tmp_path):
    directory = tmp_path / "n8-seed43"
    identity = {"training_seed": 43, "training_size": 8, "training_population_size": 16,
                "source_files": {"attributes": {"path": "labels.csv"},
                                 "partitions": {"path": "partitions.txt"}}}
    ids = [f"{index:06d}.jpg" for index in range(8)]
    metadata = {"attribute": "Smiling", "split": "train", "partition_code": 0,
                "partition_size": 16, "random_state": 43, "max_samples": 8, "image_size": [64, 64],
                "sampling": "uniform without replacement, no class balancing",
                "preprocessing": "Pillow L grayscale, resize, flatten, float32 / 255",
                "attributes_path": "labels.csv", "partitions_path": "partitions.txt"}
    stage = preparer._new_stage(directory, identity, tmp_path)
    array = stage / "c/train-X.npy"
    np.save(array, np.full((8, 4096), 0.25, dtype=np.float32), allow_pickle=False)
    preparer._write_training_receipt(stage, directory, identity, ids, metadata, preparer.sha(array))
    return SimpleNamespace(module=preparer, stage=stage, directory=directory, identity=identity,
                           ids=ids, array=array, output=tmp_path)


@pytest.mark.parametrize("damage", [
    "receipt-checksum", "identity", "ids", "metadata", "array-hash", "shape", "dtype", "missing-array",
])
def test_payload_integrity_failures_are_not_silently_rebuilt(payload, damage):
    p = payload
    path = p.stage / "training.json"
    receipt = p.module.read(path)
    value = receipt["payload"]
    if damage == "receipt-checksum":
        receipt["sha256"] = "changed"
    elif damage == "identity":
        value["preparation_identity"]["training_seed"] = 44
    elif damage == "ids":
        value["image_ids"].reverse()
    elif damage == "metadata":
        value["metadata"]["preprocessing"] = "different preprocessing"
    elif damage == "array-hash":
        with p.array.open("ab") as stream:
            stream.write(b"changed")
    elif damage in ("shape", "dtype"):
        shape = (7, 4096) if damage == "shape" else (8, 4096)
        dtype = np.float64 if damage == "dtype" else np.float32
        np.save(p.array, np.zeros(shape, dtype=dtype), allow_pickle=False)
        value["array_sha256"] = p.module.sha(p.array)
    else:
        p.array.unlink()
    if damage != "receipt-checksum":
        # Rechecking the envelope must not bypass identity/metadata/array checks.
        receipt["sha256"] = p.module._receipt_digest(value)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    before = files(p.module, p.output)
    with pytest.raises((ValueError, FileNotFoundError)) as caught:
        p.module._find_training_payload(p.directory, p.identity, p.ids, p.output)
    assert caught.value.__traceback__ is not None
    if p.array.exists():
        # On Windows this also establishes mapping closure while the exception
        # traceback still owns the validator's frame.
        rename_roundtrip(p.array)
    assert files(p.module, p.output) == before


@pytest.mark.parametrize("state", ["valid", "uncommitted", "foreign", "consumed"])
def test_payload_ownership_and_completion_boundary(payload, state):
    p = payload
    if state == "uncommitted":
        (p.stage / "training.json").rename(p.stage / ".j-partial")
    elif state == "foreign":
        marker = p.module.read(p.stage / "identity.json")
        marker["preparation_identity"]["training_seed"] = 44
        (p.stage / "identity.json").write_text(json.dumps(marker), encoding="utf-8")
    elif state == "consumed":
        p.array.unlink()
        (p.stage / "c").rmdir()
    before = files(p.module, p.output)
    found = p.module._find_training_payload(p.directory, p.identity, p.ids, p.output)
    assert (found is not None) is (state == "valid")
    if found is not None:
        assert found[0] == p.stage and found[1]["image_ids"] == p.ids
        rename_roundtrip(p.array)
    assert files(p.module, p.output) == before


@pytest.mark.parametrize("broken_manifest", [False, True])
def test_legacy_incomplete_cache_requires_opt_in_and_preserves_bytes(case, broken_manifest):
    cache = case.output / "n8-seed43"
    cache.mkdir(parents=True)
    (cache / "train-X.npy").write_bytes(b"historical partial file")
    if broken_manifest:
        (cache / "manifest.json").write_text('{"seed":', encoding="utf-8")
    before = files(case.module, cache)
    with pytest.raises(ValueError, match="Incomplete cache exists"):
        case.run()
    assert not case.calls and files(case.module, cache) == before
    report = case.run(recover_incomplete=True)
    preserved, = report["preserved_incomplete_caches"]
    assert files(case.module, case.root / preserved["preserved"]) == before
    assert case.calls == [8]


@pytest.mark.parametrize("damage", ["identity", "checksum"])
def test_committed_cache_errors_remain_strict_with_recovery_enabled(case, damage):
    case.run()
    cache = case.output / "n8-seed43"
    if damage == "identity":
        manifest = case.module.read(cache / "manifest.json")
        manifest["preparation_identity"]["training_seed"] = 44
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    else:
        with (cache / "train-X.npy").open("ab") as stream:
            stream.write(b"changed")
    before = files(case.module, case.output)
    for recover in (False, True):
        with pytest.raises(ValueError):
            case.run(recover_incomplete=recover)
        assert files(case.module, case.output) == before
    assert case.calls == [8] and not list(case.output.glob(".r-*"))


def test_committed_cache_validation_closes_mapping_on_wrong_labels(case):
    case.run()
    cache = case.output / "n8-seed43"
    label_file = cache / "train-Smiling-y.npy"
    labels = np.load(label_file, allow_pickle=False)
    labels[0] = 1 - labels[0]
    np.save(label_file, labels, allow_pickle=False)
    manifest = case.module.read(cache / "manifest.json")
    manifest["array_sha256"][label_file.name] = case.module.sha(label_file)
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = files(case.module, cache)
    with pytest.raises(ValueError, match="Wrong labels") as caught:
        case.run()
    assert caught.value.__traceback__ is not None and case.calls == [8]
    rename_roundtrip(cache / "train-X.npy")
    assert files(case.module, cache) == before


def test_reused_largest_mapping_closes_when_smaller_creation_fails(case, monkeypatch):
    case.run()
    monkeypatch.setattr(case.module, "SIZES", (4, 8))
    original = case.module._new_stage

    def fail_smaller(directory, *args):
        if directory.name == "n4-seed43":
            raise OSError("smaller write interrupted")
        return original(directory, *args)

    with monkeypatch.context() as guard:
        guard.setattr(case.module, "_new_stage", fail_smaller)
        with pytest.raises(OSError, match="smaller write interrupted") as caught:
            case.run()
    assert caught.value.__traceback__ is not None
    rename_roundtrip(case.output / "n8-seed43/train-X.npy")
    report = case.run()
    assert case.calls == [8] and report["new_largest_sample_decode_calls"] == 0


def test_original_validation_mapping_closes_while_failure_traceback_is_retained(case):
    validation = case.root / "bad-validation"
    shutil.copytree(case.inputs.validation, validation)
    manifest = case.module.read(validation / "manifest.json")
    manifest["splits"]["validation"]["attributes"]["Smiling"]["data_sha256"] = "changed"
    (validation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = files(case.module, validation)
    with pytest.raises(ValueError, match="Original validation data/labels") as caught:
        case.run(validation=validation)
    assert caught.value.__traceback__ is not None and not case.calls
    rename_roundtrip(validation / "validation-X.npy")
    assert files(case.module, validation) == before


def test_output_lock_precedes_reads_and_releases_after_interruption(preparer, tmp_path, monkeypatch):
    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    marker = output / ".cache-preparation.lock"
    calls = []

    def interrupted(*args, **kwargs):
        calls.append("entered")
        raise PreparationInterrupted("owner interrupted")

    monkeypatch.setattr(preparer, "_prepare", interrupted)
    with preparer._file_lock(marker, "owned"):
        with pytest.raises(RuntimeError, match="Another process or thread"):
            preparer.prepare("absent", "absent", "absent", "absent", output)
    assert not calls
    with pytest.raises(PreparationInterrupted):
        preparer.prepare("absent", "absent", "absent", "absent", output)
    monkeypatch.setattr(preparer, "_prepare", lambda *a, **k: "reacquired")
    assert preparer.prepare("absent", "absent", "absent", "absent", output) == "reacquired"
    assert calls == ["entered"] and marker.read_bytes() == b"0"


@pytest.mark.parametrize("mode,exit_code", [("normal", 0), ("abrupt", 71)])
def test_process_owner_exit_releases_preparation_lock(preparer, tmp_path, monkeypatch, mode, exit_code):
    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    output, ready, release = tmp_path / "out", tmp_path / "ready", tmp_path / "release"
    child_code = """
import importlib.util
import os
from pathlib import Path
import sys
import time
spec = importlib.util.spec_from_file_location('cache_lock_owner', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = Path(sys.argv[2])
ready, release = Path(sys.argv[3]), Path(sys.argv[4])
def held(*args, **kwargs):
    ready.write_text('owned', encoding='utf-8')
    deadline = time.monotonic() + 30
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Preparation owner was not released')
        time.sleep(0.01)
    if sys.argv[5] == 'abrupt':
        os._exit(71)
    return 'completed'
module._prepare = held
module.prepare('absent', 'absent', 'absent', 'absent', module.ROOT / 'out')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", child_code, str(Path(preparer.__file__).resolve()),
         str(tmp_path), str(ready), str(release), mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=5)
                pytest.fail(f"Preparation owner exited before acquiring lock: {stdout}\n{stderr}")
            if time.monotonic() >= deadline:
                pytest.fail("Preparation owner did not acquire the lock before timeout")
            time.sleep(0.01)
        monkeypatch.setattr(preparer, "_prepare", lambda *a, **k: pytest.fail("Contender entered preparation"))
        with pytest.raises(RuntimeError, match="Another process or thread"):
            preparer.prepare("absent", "absent", "absent", "absent", output)
        # Do not read/hash the marker while Windows holds its byte-range lock.
        release.write_text("release", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == exit_code, (stdout, stderr)
        monkeypatch.setattr(preparer, "_prepare", lambda *a, **k: "reacquired")
        assert preparer.prepare("absent", "absent", "absent", "absent", output) == "reacquired"
        assert (output / ".cache-preparation.lock").read_bytes() == b"0"
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


def test_json_publication_keeps_partial_write_private(preparer, tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    fdopen = os.fdopen

    class BrokenWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, text):
            self.stream.write(text[:3])
            self.stream.flush()
            raise OSError("partial JSON write")

    with monkeypatch.context() as guard:
        guard.setattr(os, "fdopen", lambda *a, **k: BrokenWriter(fdopen(*a, **k)))
        with pytest.raises(OSError, match="partial JSON write"):
            preparer.write_new(target, {"complete": True})
    assert not target.exists()
    preparer.write_new(target, {"complete": True})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        preparer.write_new(target, {"replacement": True})
    assert target.read_bytes() == before


def test_publication_confines_paths_and_never_replaces_a_destination(preparer, tmp_path):
    source, destination = tmp_path / "stage", tmp_path / "existing"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_bytes(b"source")
    (destination / "old.txt").write_bytes(b"old")
    before = files(preparer, tmp_path)
    with pytest.raises(ValueError, match="output tree"):
        preparer._publish_cache(source, tmp_path.parent / "outside", tmp_path)
    with pytest.raises(FileExistsError):
        preparer._publish_cache(source, destination, tmp_path)
    assert files(preparer, tmp_path) == before


def test_compact_staging_supports_a_baseline_valid_long_path(preparer, tmp_path):
    longest = max(preparer.ARRAY_NAMES, key=len)
    padding = 250 - len(str(tmp_path / "x" / "n8000-seed43" / longest)) + 1
    if not 1 <= padding <= 240:
        pytest.skip("The test runner's temporary root cannot fit the controlled path")
    output = tmp_path / ("x" * padding)
    output.mkdir()
    target = output / "n8000-seed43"
    target.mkdir()
    baseline = target / longest
    assert len(str(baseline)) == 250
    baseline.write_bytes(b"baseline path is usable")
    stage = preparer._new_stage(target, {"fixture": True}, output)
    preparer.write_new(stage / "c/manifest.json", {"complete": True})
    assert preparer.read(stage / "c/manifest.json") == {"complete": True}
    assert baseline.read_bytes() == b"baseline path is usable"


@pytest.mark.parametrize("option", [None, 1, "false"])
def test_recovery_option_requires_explicit_boolean(preparer, tmp_path, monkeypatch, option):
    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    output = tmp_path / "not-created"
    with pytest.raises(TypeError, match="must be a boolean"):
        preparer.prepare("absent", "absent", "absent", "absent", output, recover_incomplete=option)
    assert not output.exists()
