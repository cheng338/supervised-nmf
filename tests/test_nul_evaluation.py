"""Reject unsupported class labels before reporting or checkpoint reuse."""

import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from supervised_nmf import evaluation


FIT_CALLS = []
PREDICT_CALLS = []


class LabelProbeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, output="ordinary"):
        self.output = output

    def fit(self, X, y):
        FIT_CALLS.append(len(y))
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        PREDICT_CALLS.append(len(X))
        labels = self.classes_[(X[:, 0].astype(int) % 2)]
        if self.output == "terminal_list":
            return ["a\0" if i % 2 == 0 else "b" for i in range(len(X))]
        if self.output == "terminal_numpy_scalar":
            return np.array([np.str_("a\0") if i % 2 == 0 else "b"
                             for i in range(len(X))], dtype=object)
        return labels

    def predict_proba(self, X):
        positive = X[:, 0].astype(int) % 2
        return np.column_stack([1 - positive, positive]).astype(float)


@pytest.fixture
def data():
    FIT_CALLS.clear()
    PREDICT_CALLS.clear()
    X = np.column_stack([np.arange(20), np.ones(20)])
    return X, np.array(["a", "b"] * 10, dtype=object)


def terminal_labels(kind, length):
    values = ["a\0", "b"] * (length // 2)
    if kind == "list":
        return values
    if kind == "tuple":
        return tuple(values)
    if kind == "object":
        return np.array(values, dtype=object)
    if kind == "numpy_scalar":
        return np.array([np.str_(v) for v in values], dtype=object)
    if kind == "string_dtype":
        if not hasattr(np.dtypes, "StringDType"):
            pytest.skip("StringDType is unavailable in this supported NumPy version")
        return np.array(values, dtype=np.dtypes.StringDType())
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["list", "tuple", "object", "numpy_scalar", "string_dtype"])
def test_fingerprint_rejects_before_lossy_json_or_buffer_hashing(data, kind, monkeypatch):
    X, y = data
    unsafe = terminal_labels(kind, len(y))
    def forbidden_json(value):
        raise AssertionError("Unsafe labels reached JSON serialization")
    monkeypatch.setattr(evaluation, "_jsonable", forbidden_json)
    with pytest.raises(ValueError, match="NUL"):
        evaluation.data_fingerprint(X, unsafe)


@pytest.mark.parametrize("kind", ["list", "object", "numpy_scalar", "string_dtype"])
def test_calibration_rejects_before_identity_or_checkpoint_creation(data, kind, tmp_path, monkeypatch):
    X, y = data
    destination = tmp_path / "uncreated" / "cv.json"
    def forbidden_identity(*args, **kwargs):
        raise AssertionError("Unsafe labels reached checkpoint identity")
    monkeypatch.setattr(evaluation, "_search_identity", forbidden_identity)
    with pytest.raises(ValueError, match="NUL"):
        evaluation.calibration_search(LabelProbeClassifier(), {}, X, terminal_labels(kind, len(y)),
                                      cv=2, checkpoint_path=destination)
    assert not destination.parent.exists()
    assert FIT_CALLS == []


@pytest.mark.parametrize("role", ["evaluation", "training", "fitted_classes"])
@pytest.mark.parametrize("kind", ["list", "object", "numpy_scalar", "string_dtype"])
def test_evaluation_rejects_all_label_inputs_before_prediction(data, role, kind):
    X, y = data
    model = LabelProbeClassifier().fit(X, y)
    kwargs = {}
    labels = y
    if role == "evaluation":
        labels = terminal_labels(kind, len(y))
    elif role == "training":
        kwargs["train_y"] = terminal_labels(kind, len(y))
    else:
        model.classes_ = terminal_labels(kind, 2)
    preserved = pickle.dumps(model)
    with pytest.raises(ValueError, match="NUL"):
        evaluation.evaluate_classifier(model, X, labels, **kwargs)
    assert PREDICT_CALLS == []
    assert pickle.dumps(model) == preserved


@pytest.mark.parametrize("output", ["terminal_list", "terminal_numpy_scalar"])
def test_external_prediction_labels_fail_before_metrics(data, output, monkeypatch):
    X, y = data
    model = LabelProbeClassifier(output=output).fit(X, y)
    def forbidden_metric(*args, **kwargs):
        raise AssertionError("Unsafe predicted labels reached metrics")
    monkeypatch.setattr(evaluation, "precision_recall_fscore_support", forbidden_metric)
    report = evaluation.evaluate_classifier(model, X, y)
    assert report["status"] == "failed"
    assert report["error_type"] == "ValueError"
    assert "NUL" in report["error"]
    assert "metrics" not in report
    assert PREDICT_CALLS == [len(y)]


@pytest.mark.parametrize("fitted_positive", ["a\0", np.str_("a\0")])
def test_fitted_positive_label_rejected_before_diagnostic_serialization(data, fitted_positive):
    X, y = data
    model = LabelProbeClassifier().fit(X, y)
    model.positive_label_ = fitted_positive
    preserved = pickle.dumps(model)
    with pytest.raises(ValueError, match="NUL"):
        evaluation.evaluate_classifier(model, X, y)
    assert PREDICT_CALLS == []
    assert pickle.dumps(model) == preserved


def test_queued_constructor_positive_label_does_not_replace_fitted_policy(data):
    X, y = data
    model = LabelProbeClassifier().fit(X, y)
    model.positive_label_ = "a"
    model.positive_label = np.str_("queued\0")
    assert evaluation.evaluate_classifier(model, X, y)["status"] == "valid"


@pytest.mark.parametrize("role", ["training", "evaluation"])
@pytest.mark.parametrize("kind", ["list", "numpy_scalar"])
def test_comparison_preflight_precedes_directory_and_identity(data, role, kind, tmp_path, monkeypatch):
    X, y = data
    training = terminal_labels(kind, len(y)) if role == "training" else y
    validation = terminal_labels(kind, len(y)) if role == "evaluation" else y
    destination = tmp_path / "uncreated"
    def forbidden_identity(*args, **kwargs):
        raise AssertionError("Unsafe labels reached comparison identity")
    monkeypatch.setattr(evaluation, "_search_identity", forbidden_identity)
    with pytest.raises(ValueError, match="NUL"):
        evaluation.run_comparison({"probe": (LabelProbeClassifier(), {})}, X, training, X, validation,
                                  output_dir=destination)
    assert not destination.exists()
    assert FIT_CALLS == []


@pytest.mark.parametrize("role", ["training", "evaluation"])
@pytest.mark.parametrize("kind", ["list", "numpy_scalar"])
def test_completed_comparison_cannot_reuse_colliding_unsafe_labels(data, role, kind, tmp_path):
    X, y = data
    models = {"probe": (LabelProbeClassifier(), {})}
    first = evaluation.run_comparison(models, X, y, X, y, output_dir=tmp_path)
    fits_after_first = list(FIT_CALLS)
    predictions_after_first = list(PREDICT_CALLS)
    saved = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    # Both representations historically collided with stripped ordinary labels.
    training = terminal_labels(kind, len(y)) if role == "training" else y
    validation = terminal_labels(kind, len(y)) if role == "evaluation" else y
    with pytest.raises(ValueError, match="NUL"):
        evaluation.run_comparison(models, X, training, X, validation, output_dir=tmp_path)
    assert FIT_CALLS == fits_after_first
    assert PREDICT_CALLS == predictions_after_first
    assert saved == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert evaluation.run_comparison(models, X, y, X, y, output_dir=tmp_path) == first
    assert FIT_CALLS == fits_after_first


def test_completed_calibration_rejects_unsafe_labels_without_touching_checkpoint(data, tmp_path):
    X, y = data
    checkpoint = tmp_path / "cv.json"
    evaluation.calibration_search(LabelProbeClassifier(), {}, X, y, cv=2, checkpoint_path=checkpoint)
    saved = checkpoint.read_bytes()
    fits_after_first = list(FIT_CALLS)
    with pytest.raises(ValueError, match="NUL"):
        evaluation.calibration_search(LabelProbeClassifier(), {}, X, terminal_labels("numpy_scalar", len(y)),
                                      cv=2, checkpoint_path=checkpoint)
    assert checkpoint.read_bytes() == saved
    assert FIT_CALLS == fits_after_first


def test_embedded_labels_and_terminal_nul_metadata_remain_supported(data):
    X, ordinary = data
    y = np.array(["a\0x", "b\0y"] * (len(ordinary) // 2), dtype=object)
    model = LabelProbeClassifier().fit(X, y)
    sample_ids = [f"eval-{i}\0" for i in range(len(y))]
    train_ids = [f"train-{i}\0" for i in range(len(y))]
    groups = np.array([f"eval-group-{i % 2}\0" for i in range(len(y))], dtype=object)
    train_groups = np.array([f"train-group-{i % 2}\0" for i in range(len(y))], dtype=object)
    report = evaluation.evaluate_classifier(model, X, y, train_y=y, sample_ids=sample_ids,
                                            train_sample_ids=train_ids, groups=groups,
                                            train_groups=train_groups)
    assert report["status"] == "valid"
    assert report["metrics"]["accuracy"] == 1.0
    assert report["metrics"]["roc_auc"] == 1.0
    assert report["metrics"]["confusion_matrix"] == [[10, 0], [0, 10]]
    assert report["predictions"]["sample_ids"] == sample_ids
    assert report["support_diagnostics"]["group_count"] == 2
    assert evaluation._jsonable(np.array(["metadata\0"], dtype=object)) == ["metadata\0"]


def test_one_class_evaluation_and_comparison_preflight_remain_valid(data, tmp_path):
    X, y = data
    model = LabelProbeClassifier().fit(X, y)
    report = evaluation.evaluate_classifier(model, X[::2], y[::2], train_y=y)
    assert report["status"] == "valid"
    assert report["metrics"]["accuracy"] == 1.0
    assert report["metrics"]["roc_auc"] is None
    comparison = evaluation.run_comparison({"probe": (LabelProbeClassifier(), {})},
                                           X, y.tolist(), X[::2], y[::2].tolist(), output_dir=tmp_path)
    assert comparison["models"]["probe"]["evaluation"]["status"] == "valid"


def test_ordinary_fingerprint_bytes_and_preexisting_fixed_unicode_loss(data):
    X, y = data
    X = np.asarray(X, dtype="<f8")
    assert evaluation.data_fingerprint(X, y) == "18bd5434a63c7b51bc3a8de51d58bab038a749996e9cf42c6a622c4ad466eef5"
    # Zero padding cannot distinguish original intent once callers create U2.
    lost = np.array(["a\0", "b"] * (len(y) // 2), dtype="U2")
    actual = np.array(["a", "b"] * (len(y) // 2), dtype="U2")
    assert evaluation.data_fingerprint(X, lost) == evaluation.data_fingerprint(X, actual)


def test_existing_structural_input_errors_still_precede_label_domain_errors(data):
    X, y = data
    model = LabelProbeClassifier().fit(X, y)
    with pytest.raises(ValueError, match="inconsistent numbers"):
        evaluation.evaluate_classifier(model, X[:-1], terminal_labels("list", len(y)))
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluation.evaluate_classifier(model, X, y, train_y=[["a\0", "b"]])
    assert PREDICT_CALLS == []
