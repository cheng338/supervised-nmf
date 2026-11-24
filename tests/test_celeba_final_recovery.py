"""Portable behavioral checks for the CelebA example's final-result recovery."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import hashlib
import json
import pickle
from pathlib import Path
import sys
from threading import Event
from unittest.mock import patch
import warnings
import weakref

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from tests._celeba_recovery_test_support import (
    aggregate, codec, committed, copy_case, data_fingerprint, hashes,
    interrupt_after_commit, make_cv, provenance, receipt, recovery, study,
)


@pytest.fixture(scope="module")
def completed_cv(tmp_path_factory):
    return make_cv(tmp_path_factory.mktemp("cv"))


@pytest.fixture
def case(completed_cv, tmp_path_factory, monkeypatch):
    return copy_case(completed_cv, tmp_path_factory, monkeypatch)


@pytest.fixture(scope="module")
def raw_cv(tmp_path_factory):
    return make_cv(tmp_path_factory.mktemp("rcv"), neighbors=3, raw_cache=True)


@pytest.fixture
def raw_case(raw_cv, tmp_path_factory, monkeypatch):
    return copy_case(raw_cv, tmp_path_factory, monkeypatch)


def test_default_off_keeps_completed_result_reuse_without_model_checkpoints(case, monkeypatch):
    parsed = []
    original = argparse.ArgumentParser.parse_args
    class Parsed(Exception):
        pass
    def capture(parser, *args, **kwargs):
        parsed.append(original(parser, *args, **kwargs))
        raise Parsed
    monkeypatch.setattr(sys, "argv", ["study", "--data-cache", str(case.cache)])
    with patch.object(argparse.ArgumentParser, "parse_args", capture), pytest.raises(Parsed):
        study.main()
    assert parsed[0].final_fit_recovery == "off"
    first = aggregate(case)
    assert "recovery" not in first and not case.fit.exists()
    before = hashes(case.output)
    with patch.object(KNeighborsClassifier, "fit") as fit, patch.object(study, "evaluate_classifier") as evaluate:
        assert aggregate(case) == first
        fit.assert_not_called()
        evaluate.assert_not_called()
    assert hashes(case.output) == before


@pytest.mark.parametrize("damage", ["missing", "extra_empty"])
def test_declared_report_inventory_rejected_before_final_fit(case, damage):
    if damage == "missing":
        case.report.unlink()
    else:
        extra = deepcopy(study._read_report(case.report))
        extra.update(candidates=[], tuning_seconds=1000.)
        study.write_json(case.directory / "chunk-999" / "report.json",
                         {"payload": extra, "sha256": study._digest(extra)})
    before = hashes(case.output)
    with patch.object(KNeighborsClassifier, "fit") as fit, pytest.raises(ValueError):
        aggregate(case)
    fit.assert_not_called()
    assert not case.final.exists()
    # The persistent OS-lock marker is the only permitted new file.
    after = hashes(case.output)
    assert all(after[name] == digest for name, digest in before.items())
    assert set(after) - set(before) <= {Path(".final-evaluation.lock")}


def test_competing_finalizer_fails_then_reuses_the_owners_result(case):
    entered, release = Event(), Event()
    fit_calls = []
    original = KNeighborsClassifier.fit
    def fit(model, *args, **kwargs):
        fit_calls.append(True)
        return original(model, *args, **kwargs)
    def hold(event, details=None):
        if event == "fit-start":
            entered.set()
            if not release.wait(20):
                raise TimeoutError("Owner was not released")
    with patch.object(KNeighborsClassifier, "fit", fit), patch.object(recovery, "_hook", hold):
        with ThreadPoolExecutor(max_workers=1) as executor:
            owner = executor.submit(aggregate, case)
            try:
                assert entered.wait(20)
                with pytest.raises(RuntimeError, match="Another process or thread"):
                    aggregate(case)
            finally:
                release.set()
            first = owner.result(timeout=20)
        assert aggregate(case) == first
    assert len(fit_calls) == 1 and first["evaluation"]["status"] == "valid"


def test_committed_fit_survives_interruption_without_refitting(case):
    case.config["final_fit_recovery"] = "all"
    directory, outcome = interrupt_after_commit(case)
    before = hashes(directory)
    original = outcome["metadata"]
    with patch.object(KNeighborsClassifier, "fit") as fit:
        result = aggregate(case)
        fit.assert_not_called()
    assert result["evaluation"]["status"] == "valid"
    assert result["evaluation"]["predictions"]["predicted_labels"] == case.yv.tolist()
    assert result["recovery"]["fit_reused"] and not result["recovery"]["fit_performed"]
    assert result["recovery"]["new_fit_seconds"] == 0.
    assert result["recovery"]["reconstructed_knn_heads"] == 0
    for name in ("final_fit_seconds", "diagnostics", "final_fit_warnings"):
        assert result[name] == original[name]
    assert hashes(directory) == before


@pytest.mark.parametrize("status", ["held", "failed"])
def test_terminal_outcome_replays_qualification_and_warnings_without_work(case, status):
    case.config["final_fit_recovery"] = "all"
    original = KNeighborsClassifier.fit
    def qualified_fit(model, *args, **kwargs):
        if status == "failed":
            raise ValueError("final fit provider failed")
        fitted = original(model, *args, **kwargs)
        warnings.warn("final fit provider did not converge", ConvergenceWarning)
        return fitted
    with patch.object(KNeighborsClassifier, "fit", qualified_fit):
        directory, outcome = interrupt_after_commit(case)
    assert outcome["kind"] == "terminal" and not (directory / "payload").exists()
    with patch.object(KNeighborsClassifier, "fit") as fit, \
            patch.object(study, "evaluate_classifier") as evaluate, \
            patch.object(recovery.RecoveryStore, "restore_model") as restore:
        result = aggregate(case)
        fit.assert_not_called()
        evaluate.assert_not_called()
        restore.assert_not_called()
    assert result["evaluation"] == outcome["metadata"]["evaluation"]
    assert result["evaluation"]["status"] == status
    assert result["final_fit_warnings"] == outcome["metadata"]["final_fit_warnings"]
    assert result["final_fit_seconds"] == outcome["metadata"]["final_fit_seconds"]
    assert result["recovery"]["outcome_reused"] and not result["recovery"]["fit_reused"]


def test_completed_final_result_precedes_missing_model_state(case):
    case.config["final_fit_recovery"] = "all"
    first = aggregate(case)
    directory, _ = committed(case)
    (directory / "payload" / codec.PAYLOAD_NAME).unlink()
    before = hashes(case.output)
    with patch.object(KNeighborsClassifier, "fit") as fit, \
            patch.object(study, "evaluate_classifier") as evaluate, \
            patch.object(recovery.RecoveryStore, "restore_model") as restore:
        assert aggregate(case) == first
        fit.assert_not_called()
        evaluate.assert_not_called()
        restore.assert_not_called()
    assert hashes(case.output) == before


@pytest.mark.parametrize("policy", ["off", "all"])
def test_ineligible_method_has_no_measured_final_fit(case, policy):
    report = receipt(case.report)
    for candidate in report["candidates"]:
        candidate.update(valid=False, mean_balanced_accuracy=None)
    study.write_json(case.report, {"payload": report, "sha256": study._digest(report)})
    case.config["final_fit_recovery"] = policy
    with patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("ineligible final fit")):
        result = aggregate(case)
    assert result["evaluation"]["status"] == "held"
    assert result["final_fit_seconds"] is None
    assert "fit_execution" not in result and "prediction_execution" not in result


def test_corrupt_final_result_does_not_fall_back_to_a_model_checkpoint(case):
    case.config["final_fit_recovery"] = "all"
    aggregate(case)
    saved = json.loads(case.final.read_text(encoding="utf-8"))
    saved["sha256"] = "0" * 64
    study.write_json(case.final, saved)
    before = hashes(case.output)
    with patch.object(recovery.RecoveryStore, "restore_model") as restore, pytest.raises(ValueError):
        aggregate(case)
    restore.assert_not_called()
    assert hashes(case.output) == before


def test_runtime_change_after_outcome_verification_rejects_before_model_load(case):
    case.config["final_fit_recovery"] = "all"
    interrupt_after_commit(case)
    changed = []
    read = recovery.RecoveryStore.load_outcome
    identity = recovery.active_runtime_identity
    def verified(store):
        outcome = read(store)
        changed.append(True)
        return outcome
    def observed(*args, **kwargs):
        result = identity(*args, **kwargs)
        if changed:
            result["versions"]["joblib"] = "changed-runtime-observation"
        return result
    with patch.object(recovery.RecoveryStore, "load_outcome", verified), \
            patch.object(recovery, "active_runtime_identity", observed), \
            patch.object(recovery.RecoveryStore, "restore_model") as restore, \
            patch.object(KNeighborsClassifier, "fit") as fit, \
            pytest.raises(recovery.RecoveryError, match="Source or active runtime changed"):
        aggregate(case)
    restore.assert_not_called()
    fit.assert_not_called()
    assert not case.final.exists()


@pytest.mark.parametrize("damage", ["version", "checksum", "missing", "truncated"])
def test_owned_codec_payload_mismatch_rejected_before_joblib_load(case, tmp_path, damage):
    with threadpool_limits(limits=1):
        model = KNeighborsClassifier(n_neighbors=1).fit(case.X, case.y)
    directory = tmp_path / "payload"
    metadata = codec.dump_model(model, directory)
    path = directory / codec.PAYLOAD_NAME
    if damage == "version":
        metadata["joblib_version"] = "different-runtime-version"
    elif damage == "checksum":
        metadata["payload"]["sha256"] = "0" * 64
    elif damage == "missing":
        path.unlink()
    else:
        content = path.read_bytes()
        path.write_bytes(content[:len(content) // 2])
    with patch.object(codec.joblib, "load") as load, pytest.raises(codec.CodecError):
        codec.load_model(directory, metadata)
    load.assert_not_called()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_joblib_older_load_signature_and_standard_byte_order(case, tmp_path, dtype):
    with threadpool_limits(limits=1):
        model = KNeighborsClassifier(n_neighbors=1).fit(case.X.astype(dtype), case.y)
        expected = model.predict(case.X.astype(dtype))
        probabilities = model.predict_proba(case.X.astype(dtype))
    model.native_probe_ = np.array([1.25, -0., 2.5], dtype=dtype)
    model.foreign_probe_ = model.native_probe_.astype(np.dtype(dtype).newbyteorder("S"))
    model.scalar_probe_ = np.dtype(dtype).type(-0.)
    model.integer_probe_ = np.int16(-7)
    assert not model.foreign_probe_.dtype.isnative
    directory = tmp_path / "model"
    metadata = codec.dump_model(model, directory)
    actual_load, calls = codec.joblib.load, []
    def older_signature(filename, mmap_mode=None):
        calls.append(mmap_mode)
        return actual_load(filename, mmap_mode=mmap_mode)
    with patch.object(codec.joblib, "load", older_signature):
        restored = codec.load_model(directory, metadata)
    assert calls == [None]  # No newer keyword is required by the adapter.
    with threadpool_limits(limits=1):
        assert np.array_equal(restored.predict(case.X.astype(dtype)), expected)
        assert np.array_equal(restored.predict_proba(case.X.astype(dtype)), probabilities)
    for name in ("native_probe_", "foreign_probe_"):
        # Standard joblib normalizes array byte order; original endian bytes are
        # not the contract. Native dtype, values and signed zero are preserved.
        saved, loaded = getattr(model, name), getattr(restored, name)
        assert loaded.dtype == np.dtype(dtype) and loaded.dtype.isnative
        assert np.array_equal(saved, loaded) and np.array_equal(np.signbit(saved), np.signbit(loaded))
    for name in ("scalar_probe_", "integer_probe_"):
        saved, loaded = getattr(model, name), getattr(restored, name)
        assert type(loaded) is type(saved) and loaded.dtype == saved.dtype
        assert loaded.tobytes() == saved.tobytes()


def test_public_buffer_replays_evaluation_without_fitting_and_final_takes_precedence(raw_case):
    raw_case.config["final_fit_recovery"] = "all"
    evaluations, actual_evaluate = [], study.evaluate_classifier
    def evaluate(*args, **kwargs):
        result = actual_evaluate(*args, **kwargs)
        evaluations.append(deepcopy(result))
        return result
    def stop(event, details=None):
        if event == "evaluation-complete":
            raise KeyboardInterrupt("stop before the final result commits")
    with patch.object(study, "evaluate_classifier", evaluate), \
            patch.object(recovery, "_hook", stop), pytest.raises(KeyboardInterrupt):
        aggregate(raw_case)
    directory, outcome = committed(raw_case)
    assert outcome["codec"]["codec"] == codec.RAW_CODEC and len(evaluations) == 1
    before = hashes(directory)
    with patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("unexpected refit")):
        result = aggregate(raw_case)
    actual, expected = deepcopy(result["evaluation"]), evaluations[0]
    actual.pop("prediction_seconds", None)
    expected.pop("prediction_seconds", None)
    assert actual == expected and hashes(directory) == before
    for key in ("final_fit_seconds", "final_fit_warnings", "diagnostics"):
        assert result[key] == outcome["metadata"][key]
    costs = result["recovery"]
    assert costs["fit_reused"] and not costs["cache_reference_restored"] and not costs["fit_performed"]
    assert costs["new_fit_seconds"] == costs["head_reconstruction_seconds"] == 0.
    assert costs["reconstructed_knn_heads"] == 0
    assert 0 < costs["cache_binding_seconds"] <= costs["model_load_seconds"]
    # Allocation and copying are inside loading; neither is a new fit.
    (directory / "payload" / codec.BUFFER_NAME).unlink()
    with patch.object(codec, "load_model", side_effect=AssertionError("unexpected load")), \
            patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("unexpected fit")), \
            patch.object(study, "evaluate_classifier", side_effect=AssertionError("unexpected evaluation")):
        assert aggregate(raw_case) == result


@pytest.mark.parametrize("change", ["missing", "identity"])
def test_committed_buffer_context_rejects_before_loading(raw_case, change):
    raw_case.config["final_fit_recovery"] = "all"
    directory, outcome = interrupt_after_commit(raw_case)
    assert outcome["codec"]["codec"] == codec.RAW_CODEC
    before, restore = hashes(directory), recovery.RecoveryStore.restore_model
    def mismatched(store, outcome, *, codec_context=None):
        context = None if change == "missing" else dict(codec_context, identity=dict(codec_context["identity"]))
        if context is not None:
            context["identity"]["training_data_sha256"] = "f" * 64
        return restore(store, outcome, codec_context=context)
    with patch.object(recovery.RecoveryStore, "restore_model", mismatched), \
            patch.object(codec.pickle, "load") as load, patch.object(KNeighborsClassifier, "fit") as fit, \
            pytest.raises(recovery.RecoveryError, match="Model reconstruction failed"):
        aggregate(raw_case)
    load.assert_not_called()
    fit.assert_not_called()
    assert not raw_case.final.exists() and hashes(directory) == before




@pytest.fixture
def buffer_case():
    return data()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()



def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')



def data():
    rng = np.random.default_rng(515)
    X = rng.uniform(.1, .9, (48, 4096)).astype(np.float32)
    X[:6] = X[0]
    y = np.tile(np.array([0, 1], dtype=np.int64), 24)
    X.flags.writeable = y.flags.writeable = False
    query = X[[0, 7, 8, 9, 10, 11, 12, 13]].copy()
    truth = y[[0, 7, 8, 9, 10, 11, 12, 13]].copy()
    identity = dict(method='raw_knn', attribute='Smiling', run_identity='1' * 64,
        evaluation_identity='2' * 64, manifest_sha256='3' * 64, training_data_sha256='4' * 64)
    return dict(X=X, y=y, identity=identity), query, truth



def fit(context, k=3, **parameters):
    with threadpool_limits(limits=1):
        return KNeighborsClassifier(n_neighbors=k, n_jobs=1, **parameters).fit(context['X'], context['y'])



def state(value):
    if isinstance(value, np.ndarray):
        return dict(array=True, dtype=value.dtype.str, shape=list(value.shape), strides=list(value.strides),
                    sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    if isinstance(value, np.generic):
        return dict(scalar=value.dtype.str, bytes=value.tobytes().hex())
    if type(value) is dict:
        return {key: state(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return {type(value).__name__: [state(item) for item in value]}
    if value is None or type(value) in (str, int, float, bool):
        return dict(type=type(value).__name__, value=value)
    return dict(type=f'{type(value).__module__}.{type(value).__name__}', joblib_hash=codec.joblib.hash(value))



def forbid_fit(monkeypatch):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Public-buffer restore must not fit')
    monkeypatch.setattr(KNeighborsClassifier, 'fit', forbidden)
    return calls



def save_buffer(tmp_path, context, k=3):
    model = fit(context, k)
    target = tmp_path / 'payload'
    metadata = codec.dump_model(model, target, context=context)
    assert metadata['codec'] == codec.RAW_CODEC
    return model, target, metadata



@pytest.mark.parametrize('k', [1, 3, 25])
def test_exact_native_state_probabilities_ties_original_and_extra_attributes(tmp_path, buffer_case, monkeypatch, k):
    context, query, _ = buffer_case
    model = fit(context, k)
    model.owned_extra_ = {'array': np.arange(15, dtype=np.int16).reshape(3, 5),
        'tuple': (np.float32(-0.), np.uint64(2**63 + 9)), 'text': 'ordinary fitted annotation'}
    before, original_hash = state(vars(model)), codec.joblib.hash(model)
    with threadpool_limits(limits=1):
        expected_labels, expected_proba = model.predict(query), model.predict_proba(query)
    metadata = codec.dump_model(model, tmp_path / 'p', context=context)
    assert metadata['codec'] == codec.RAW_CODEC
    assert set(p.name for p in (tmp_path / 'p').iterdir()) == {'model.pickle'}
    assert metadata['payload']['file_bytes'] < context['X'].nbytes // 4
    assert metadata['external_buffers'] == [{'source': 'training-X', 'nbytes': context['X'].nbytes, 'readonly': True}]
    calls, stats = forbid_fit(monkeypatch), {}
    restored = codec.load_model(tmp_path / 'p', metadata, context=context, stats=stats)
    assert not calls and state(vars(restored)) == before
    assert state(vars(model)) == before and codec.joblib.hash(model) == original_hash
    with threadpool_limits(limits=1):
        np.testing.assert_array_equal(restored.predict(query), expected_labels)
        np.testing.assert_array_equal(restored.predict_proba(query), expected_proba)
    assert stats['reconstructed_knn_heads'] == 0 and stats['reconstruction_seconds'] == 0.
    assert stats['cache_reference_restored'] is False
    assert 0 < stats['cache_binding_seconds'] == stats['buffer_preparation_seconds'] <= stats['codec_load_seconds']
    assert stats['buffer_preparation'] == 'copy'
    # Observational test only; the codec never names this fitted field.
    assert not restored._fit_X.flags.writeable
    assert not np.shares_memory(restored._fit_X, context['X'])



@pytest.mark.parametrize('variant', ['exact_view', 'independent_equal_copy', 'shifted_span', 'prefix',
    'fortran', 'noncontiguous', 'writable_model', 'writable_context', 'float64', 'small_features',
    'non_native', 'no_context', 'multiple_exact_buffers'])
def test_exact_span_and_unsupported_fallback_has_no_partial_artifact(tmp_path, buffer_case, monkeypatch, variant):
    context, _, _ = buffer_case
    fit_context = dict(context)
    if variant == 'exact_view': fit_context['X'] = context['X'].view()
    elif variant == 'independent_equal_copy': fit_context['X'] = context['X'].copy(); fit_context['X'].flags.writeable = False
    elif variant == 'shifted_span':
        backing = np.vstack([context['X'], context['X'][:1]])
        backing.flags.writeable = False
        context['X'], fit_context['X'] = backing[:-1], backing[1:]
    elif variant == 'prefix': fit_context['X'] = context['X'][:-1]; fit_context['y'] = context['y'][:-1]
    elif variant == 'fortran':
        context['X'] = np.asfortranarray(context['X']); context['X'].flags.writeable = False
        fit_context['X'] = context['X']
    elif variant == 'noncontiguous':
        backing = np.repeat(context['X'], 2, axis=1); backing.flags.writeable = False
        context['X'] = backing[:, ::2]; fit_context['X'] = context['X']
    elif variant == 'writable_model': fit_context['X'] = context['X'].copy()
    elif variant == 'writable_context': context['X'] = context['X'].copy(); fit_context['X'] = context['X']
    elif variant == 'float64':
        context['X'] = context['X'].astype(np.float64); context['X'].flags.writeable = False
        fit_context['X'] = context['X']
    elif variant == 'small_features':
        context['X'] = context['X'][:, :16].copy(); context['X'].flags.writeable = False
        fit_context['X'] = context['X']
    elif variant == 'non_native':
        context['X'] = context['X'].astype(np.dtype(np.float32).newbyteorder('S'))
        context['X'].flags.writeable = False
        fit_context['X'] = context['X']
    model = fit(fit_context)
    if variant == 'multiple_exact_buffers': model.owned_extra_alias_ = context['X'].view()
    before = state(vars(model))
    metadata = codec.dump_model(model, tmp_path / 'p', context=None if variant == 'no_context' else context)
    expect_buffer = variant == 'exact_view'
    assert (metadata['codec'] == codec.RAW_CODEC) is expect_buffer
    assert set(p.name for p in (tmp_path / 'p').iterdir()) == ({'model.pickle'} if expect_buffer else {codec.PAYLOAD_NAME})
    calls = forbid_fit(monkeypatch)
    restored = codec.load_model(tmp_path / 'p', metadata, context=context)
    assert not calls and state(vars(model)) == before
    if variant == 'non_native':
        # The full-joblib fallback deliberately uses native endian normalization.
        np.testing.assert_array_equal(restored._fit_X, model._fit_X)
        assert restored._fit_X.dtype.isnative
        assert restored.get_params(deep=False) == model.get_params(deep=False)
    else:
        assert state(vars(restored)) == before



def test_same_object_extra_alias_preserves_pickle_memo_identity(tmp_path, buffer_case, monkeypatch):
    context, _, _ = buffer_case
    model = fit(context)
    model.owned_same_object_ = model._fit_X
    before = state(vars(model))
    metadata = codec.dump_model(model, tmp_path / 'p', context=context)
    assert metadata['codec'] == codec.RAW_CODEC
    calls = forbid_fit(monkeypatch)
    restored = codec.load_model(tmp_path / 'p', metadata, context=context)
    assert not calls and state(vars(restored)) == before and state(vars(model)) == before
    assert restored.owned_same_object_ is restored._fit_X



@pytest.mark.parametrize('parameter', ['algorithm', 'weights', 'n_jobs', 'leaf_size', 'p', 'numpy_neighbors'])
def test_unsupported_public_parameters_use_complete_state(tmp_path, buffer_case, monkeypatch, parameter):
    context, _, _ = buffer_case
    parameters = dict(n_neighbors=3, n_jobs=1)
    if parameter == 'algorithm': parameters['algorithm'] = 'brute'
    elif parameter == 'weights': parameters['weights'] = 'distance'
    elif parameter == 'n_jobs': parameters['n_jobs'] = None
    elif parameter == 'leaf_size': parameters['leaf_size'] = 31
    elif parameter == 'p': parameters['p'] = 1
    else: parameters['n_neighbors'] = np.int64(3)
    with threadpool_limits(limits=1): model = KNeighborsClassifier(**parameters).fit(context['X'], context['y'])
    before = state(vars(model))
    metadata = codec.dump_model(model, tmp_path / 'p', context=context)
    assert metadata['codec'] != codec.RAW_CODEC
    assert set(p.name for p in (tmp_path / 'p').iterdir()) == {codec.PAYLOAD_NAME}
    calls = forbid_fit(monkeypatch)
    restored = codec.load_model(tmp_path / 'p', metadata, context=context)
    assert not calls and state(vars(restored)) == before and state(vars(model)) == before



def test_owned_copy_lifetime_and_independence_after_source_release_or_change(tmp_path, buffer_case, monkeypatch):
    context, _, _ = buffer_case
    cache_path = tmp_path / 'owned-cache.npy'
    np.save(cache_path, context['X'], allow_pickle=False)
    (tmp_path / 'owned-cache-original.npy').write_bytes(cache_path.read_bytes())
    expected, identity, labels = context['X'].copy(), context['identity'], context['y']
    del context
    allocations = []
    actual_array = codec.np.array
    def observed_array(value, *args, **kwargs):
        result = actual_array(value, *args, **kwargs)
        if kwargs.get('copy') is True and kwargs.get('order') == 'C' and getattr(value, 'shape', None) == expected.shape:
            allocations.append(weakref.ref(result))
        return result
    monkeypatch.setattr(codec.np, 'array', observed_array)
    def restore():
        cache = np.load(cache_path, mmap_mode='r', allow_pickle=False)
        cache_ref = weakref.ref(cache)
        scope = dict(X=cache, y=labels, identity=identity)
        model = fit(scope)
        model.owned_same_object_ = model._fit_X
        metadata = codec.dump_model(model, tmp_path / 'buffer', context=scope)
        assert metadata['codec'] == codec.RAW_CODEC
        with patch.object(KNeighborsClassifier, 'fit', side_effect=AssertionError('unexpected refit')):
            restored = codec.load_model(tmp_path / 'buffer', metadata, context=scope)
        assert not np.shares_memory(restored._fit_X, cache)
        return restored, cache_ref
    restored, cache_ref = restore()
    gc.collect()
    assert cache_ref() is None and len(allocations) == 1 and allocations[0]() is not None
    assert not restored._fit_X.flags.writeable and restored.owned_same_object_ is restored._fit_X
    np.testing.assert_array_equal(restored._fit_X, expected)
    writable = np.load(cache_path, mmap_mode='r+', allow_pickle=False)
    writable[0, 0] += np.float32(1.)
    writable.flush()
    assert float(writable[0, 0]) != float(expected[0, 0])
    np.testing.assert_array_equal(restored._fit_X, expected)
    del writable, restored
    gc.collect()
    assert allocations[0]() is None



def test_mutated_training_copy_and_same_size_unrelated_attribute_preserved(tmp_path, buffer_case, monkeypatch):
    context, query, _ = buffer_case
    copied = dict(context)
    copied['X'] = context['X'].copy()
    copied['X'][0] += np.float32(.4)
    copied['X'].flags.writeable = False
    model = fit(copied)
    # Exact cache span appears only in an ordinary extra attribute; fitted
    # training coordinates must remain their actual distinct inline values.
    model.owned_cache_annotation_ = context['X']
    model.owned_unrelated_same_size_ = np.full_like(context['X'], np.float32(.73))
    model.owned_unrelated_same_size_.flags.writeable = False
    before = state(vars(model))
    with threadpool_limits(limits=1): expected = model.predict_proba(query)
    metadata = codec.dump_model(model, tmp_path / 'p', context=context)
    # Discovery aborts before serializing a nonmatching matrix-sized buffer.
    assert metadata['codec'] != codec.RAW_CODEC
    assert set(p.name for p in (tmp_path / 'p').iterdir()) == {codec.PAYLOAD_NAME}
    assert metadata['payload']['file_bytes'] > context['X'].nbytes
    calls = forbid_fit(monkeypatch)
    restored = codec.load_model(tmp_path / 'p', metadata, context=context)
    assert not calls and state(vars(restored)) == before and state(vars(model)) == before
    with threadpool_limits(limits=1): np.testing.assert_array_equal(restored.predict_proba(query), expected)



@pytest.mark.parametrize('variant', ['large_unmatched_first', 'second_exact_span'])
def test_discovery_aborts_before_large_inline_copy_or_second_external_span(tmp_path, buffer_case, monkeypatch, variant):
    context, _, _ = buffer_case
    fitting = dict(context)
    if variant == 'large_unmatched_first':
        fitting['X'] = context['X'].copy()
        fitting['X'].flags.writeable = False
    model = fit(fitting)
    if variant == 'second_exact_span': model.owned_second_span_ = context['X'].view()
    before = state(vars(model))
    original_dump, original_complete = codec.pickle.dump, codec._dump_complete
    observed, fallback = [], []
    def tracked_dump(value, stream, *, protocol, buffer_callback):
        def tracked_buffer(buffer):
            raw = buffer.raw()
            size = raw.nbytes
            raw.release()
            try:
                answer = buffer_callback(buffer)
            except BaseException:
                observed.append(dict(bytes=size, stream_position=stream.tell(), decision='abort'))
                raise
            observed.append(dict(bytes=size, stream_position=stream.tell(), decision='inline' if answer else 'external'))
            return answer
        return original_dump(value, stream, protocol=protocol, buffer_callback=tracked_buffer)
    def complete(value, directory):
        target = Path(directory)
        assert not target.exists() or not any(target.iterdir())
        fallback.append(True)
        return original_complete(value, directory)
    monkeypatch.setattr(codec.pickle, 'dump', tracked_dump)
    monkeypatch.setattr(codec, '_dump_complete', complete)
    metadata = codec.dump_model(model, tmp_path / 'p', context=context)
    assert metadata['codec'] != codec.RAW_CODEC and fallback == [True]
    assert observed[-1]['decision'] == 'abort' and observed[-1]['bytes'] == context['X'].nbytes
    assert observed[-1]['stream_position'] < context['X'].nbytes
    assert not any(row['decision'] == 'inline' and row['bytes'] >= context['X'].nbytes for row in observed)
    assert sum(row['decision'] == 'external' for row in observed) == int(variant == 'second_exact_span')
    assert state(vars(model)) == before
    calls = forbid_fit(monkeypatch)
    restored = codec.load_model(tmp_path / 'p', metadata, context=context)
    assert not calls and state(vars(restored)) == before



def test_actual_discovery_error_propagates_without_fallback_or_partial_file(tmp_path, buffer_case, monkeypatch):
    context, _, _ = buffer_case
    model = fit(context)
    before = state(vars(model))
    original = OSError('owned serializer failure')
    fallbacks = []
    def fail(*args, **kwargs): raise original
    def fallback(*args, **kwargs):
        fallbacks.append(True)
        raise AssertionError('Real serialization errors must not select fallback')
    monkeypatch.setattr(codec.pickle, 'dump', fail)
    monkeypatch.setattr(codec, '_dump_complete', fallback)
    with pytest.raises(OSError) as caught: codec.dump_model(model, tmp_path / 'p', context=context)
    assert caught.value is original and not fallbacks and not (tmp_path / 'p').exists()
    assert state(vars(model)) == before



@pytest.mark.parametrize('damage', ['extra_top', 'runtime', 'numpy_version', 'identity', 'missing_context',
    'writable', 'shape', 'dtype', 'layout', 'bool_shape', 'missing_buffers', 'extra_buffers',
    'buffer_source', 'buffer_size', 'buffer_bool_size', 'buffer_writable', 'payload_size',
    'payload_sha', 'payload_filename', 'missing_file', 'changed_bytes'])
def test_committed_mismatch_rejected_before_loading_or_fallback(tmp_path, buffer_case, monkeypatch, damage):
    context, _, _ = buffer_case
    _, target, metadata = save_buffer(tmp_path, context)
    altered = deepcopy(metadata)
    context = dict(context)
    if damage == 'extra_top': altered['extra'] = True
    elif damage == 'runtime': altered['sklearn_version'] = 'owned-other'
    elif damage == 'numpy_version': altered['numpy_version'] = 'owned-other'
    elif damage == 'identity': altered['cache_identity']['training_data_sha256'] = 'f' * 64
    elif damage == 'missing_context': context = None
    elif damage == 'writable': context['X'] = context['X'].copy()
    elif damage == 'shape': context['X'] = context['X'][:, :8]
    elif damage == 'dtype': context['X'] = context['X'].astype(np.float64); context['X'].flags.writeable = False
    elif damage == 'layout': context['X'] = np.asfortranarray(context['X']); context['X'].flags.writeable = False
    elif damage == 'bool_shape': altered['training']['shape'][0] = True
    elif damage == 'missing_buffers': altered['external_buffers'] = []
    elif damage == 'extra_buffers': altered['external_buffers'] *= 2
    elif damage == 'buffer_source': altered['external_buffers'][0]['source'] = 'other'
    elif damage == 'buffer_size': altered['external_buffers'][0]['nbytes'] -= 4
    elif damage == 'buffer_bool_size': altered['external_buffers'][0]['nbytes'] = True
    elif damage == 'buffer_writable': altered['external_buffers'][0]['readonly'] = False
    elif damage == 'payload_size': altered['payload']['file_bytes'] += 1
    elif damage == 'payload_sha': altered['payload']['sha256'] = '0' * 64
    elif damage == 'payload_filename': altered['payload']['file'] = 'other.pickle'
    elif damage == 'missing_file': (target / 'model.pickle').unlink()
    elif damage == 'changed_bytes': (target / 'model.pickle').write_bytes(b'owned changed payload')
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Invalid committed buffer reached decoder/fallback/fit')
    monkeypatch.setattr(codec.pickle, 'load', forbidden)
    monkeypatch.setattr(codec.pickle, 'loads', forbidden)
    monkeypatch.setattr(codec, '_load_complete', forbidden)
    monkeypatch.setattr(KNeighborsClassifier, 'fit', forbidden)
    with pytest.raises(codec.CodecError): codec.load_model(target, altered, context=context)
    assert not calls



@pytest.mark.parametrize('change', ['unused_buffer', 'extra_consumption', 'trailing_bytes'])
def test_owned_payload_buffer_consumption_contract(tmp_path, buffer_case, monkeypatch, change):
    context, _, _ = buffer_case
    model, target, metadata = save_buffer(tmp_path, context)
    path = target / 'model.pickle'
    if change == 'unused_buffer': path.write_bytes(pickle.dumps(model, protocol=5))
    elif change == 'extra_consumption': path.write_bytes(pickle.dumps(model, protocol=5, buffer_callback=lambda value: False))
    else: path.write_bytes(path.read_bytes() + b'owned unexpected trailing bytes')
    metadata['payload'].update(file_bytes=path.stat().st_size, sha256=sha(path))
    calls = forbid_fit(monkeypatch)
    with pytest.raises(Exception): codec.load_model(target, metadata, context=context)
    assert not calls



@pytest.mark.parametrize('failure', ['allocation', 'after_copy'])
def test_owned_copy_failure_preserves_committed_fit_and_retries_without_fit(raw_case, monkeypatch, failure):
    raw_case.config['final_fit_recovery'] = 'all'
    directory, outcome = interrupt_after_commit(raw_case)
    assert outcome['codec']['codec'] == codec.RAW_CODEC
    before, pointer_before = hashes(directory), sha(raw_case.fit / 'current.json')
    original = MemoryError('owned copy allocation failed')
    actual_array, calls = codec.np.array, []
    def fail(value, *args, **kwargs):
        if kwargs.get('copy') is True and kwargs.get('order') == 'C' and getattr(value, 'shape', None) == raw_case.X.shape:
            calls.append(True)
            if failure == 'after_copy': actual_array(value, *args, **kwargs)
            raise original
        return actual_array(value, *args, **kwargs)
    fits = forbid_fit(monkeypatch)
    with patch.object(codec.np, 'array', fail), pytest.raises(recovery.RecoveryError) as caught:
        aggregate(raw_case)
    cause = caught.value
    while cause.__cause__ is not None: cause = cause.__cause__
    assert cause is original and calls == [True] and not fits
    assert not raw_case.final.exists()
    assert hashes(directory) == before and sha(raw_case.fit / 'current.json') == pointer_before
    failed = [receipt(path)['details'] for path in (raw_case.fit / 'attempts').glob('*/interrupted.json')]
    assert any(row['error_type'] == 'RecoveryError' and str(original) in row['error'] for row in failed)
    result = aggregate(raw_case)
    assert not fits and result['evaluation']['status'] == 'valid'
    assert hashes(directory) == before and sha(raw_case.fit / 'current.json') == pointer_before
    for name in ('final_fit_seconds', 'final_fit_warnings', 'diagnostics'):
        assert result[name] == outcome['metadata'][name]
    cost = result['recovery']
    assert cost['fit_reused'] and cost['outcome_reused'] and not cost['fit_performed']
    assert cost['new_fit_seconds'] == cost['head_reconstruction_seconds'] == 0.
    assert cost['reconstructed_knn_heads'] == 0 and not cost['cache_reference_restored']
    assert 0 < cost['cache_binding_seconds'] <= cost['model_load_seconds']
