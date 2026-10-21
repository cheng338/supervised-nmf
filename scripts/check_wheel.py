"""Verify the locally built wheel without installing into the conda environment.

Run with `python -I scripts/check_wheel.py` to exclude source-checkout imports.
"""

from email.parser import Parser
from pathlib import Path
import re
import sys
import tarfile
import zipfile

root = Path(__file__).resolve().parents[1]
withdrawn_workflow = ("supervised_nmf/tuning.py",
                      "examples/adaptive_policy.json", "examples/adaptive_support.py",
                      "tests/test_search.py", "tests/test_tuning.py", "tests/test_adaptive_examples.py",
                      "docs/adaptive-tuning.md", "provenance/tuning-reference.json")
version = re.search(r'^version = "([^"]+)"$',
                    (root / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE).group(1)
wheel = root / "dist" / f"supervised_nmf-{version}-py3-none-any.whl"
with zipfile.ZipFile(wheel) as archive:
    metadata_file = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
    metadata = Parser().parsestr(archive.read(metadata_file).decode("utf-8"))
    assert metadata["Author"] == "Chang Cheng"
    assert metadata["Version"] == version
    assert metadata["Requires-Python"] == ">=3.11"
    assert metadata["License-Expression"] == "LicenseRef-All-Rights-Reserved AND BSD-3-Clause"
    license_files = ("LICENSE", "THIRD_PARTY_NOTICES")
    assert set(metadata.get_all("License-File", [])) == set(license_files)
    dist_info = metadata_file.rsplit("/", 1)[0]
    for license_file in license_files:
        assert archive.read(f"{dist_info}/licenses/{license_file}") == (root / license_file).read_bytes(), license_file
    assert "study" in metadata.get_all("Provides-Extra", [])
    assert any("matplotlib" in requirement and 'extra == "study"' in requirement
               for requirement in metadata.get_all("Requires-Dist", []))
    assert not any(n.endswith((".jpg", ".png", ".pdf")) for n in archive.namelist())
    assert not any(set(Path(n).parts) & {"tmp", "data", "runs"} or "ensemble-reference" in n
                   for n in archive.namelist())
    assert not any(n.endswith(withdrawn_workflow) for n in archive.namelist())

sys.path.insert(0, str(wheel))
import numpy as np
from scipy.optimize import nnls
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import supervised_nmf
from supervised_nmf import NMFClassifier
from supervised_nmf.datasets import make_original_simulation
from supervised_nmf.study import calibration_knn_grid
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.validation import build_cv_folds
from supervised_nmf.evaluation import format_evaluation_summary

assert ".whl" in supervised_nmf.__file__, supervised_nmf.__file__
assert supervised_nmf.__author__ == "Chang Cheng"
assert supervised_nmf.__version__ == version
assert callable(calibration_knn_grid)
assert callable(calibration_rank_search)
assert callable(build_cv_folds)
assert callable(format_evaluation_summary)
defaults = NMFClassifier().get_params()
assert defaults["solver"] == "spa", defaults
assert defaults["screening"] == "absolute", defaults
assert defaults["max_iter"] == 20000, defaults
assert defaults["tol"] == 1e-4, defaults
assert defaults["classifier"] == "knn", defaults
assert defaults["knn_normalization"] == "none", defaults
assert defaults["training_coordinates"] == "auto", defaults
data = make_original_simulation(n_samples=60, n_features=20, signal_ratio=0.2)
with threadpool_limits(limits=1):
    # Exercise the installed pair with both method parameters omitted.
    model = NMFClassifier(n_components=2, random_state=42).fit(data.data, data.target)
    labels, probabilities = model.predict_with_probabilities(data.data[:5])
    override_parameters = {
        "solver": "spa" if defaults["solver"] == "cd" else "cd",
        "screening": "absolute" if defaults["screening"] == "percentile" else "percentile",
        "percentile": 20, "score_threshold": 2.0, "n_components": 2,
        "max_iter": 2000, "random_state": 42,
    }
    override = NMFClassifier(**override_parameters).fit(data.data, data.target)
    override_labels, override_probabilities = override.predict_with_probabilities(data.data[:5])
    mu = NMFClassifier(solver="mu", random_state=42).fit(data.data, data.target)
    mu_basis = mu.selected_components_.copy()
    mu_labels, mu_probabilities = mu.predict_with_probabilities(data.data[:5])
    logistic = NMFClassifier(classifier="logistic", random_state=42).fit(data.data, data.target)
    logistic_labels, logistic_probabilities = logistic.predict_with_probabilities(data.data[:5])
assert model.solver == defaults["solver"] and model.screening == defaults["screening"]
assert model.max_iter == 20000
for name, value in override_parameters.items():
    assert override.get_params()[name] == value, name
assert model.converged_ and model.last_transform_converged_
assert override.converged_ and override.last_transform_converged_
assert labels.shape == (5,)
assert probabilities.shape == (5, 2)
assert np.isfinite(probabilities).all()
np.testing.assert_allclose(probabilities.sum(axis=1), 1)
assert override_labels.shape == (5,)
assert override_probabilities.shape == (5, 2)
assert np.isfinite(override_probabilities).all()
np.testing.assert_allclose(override_probabilities.sum(axis=1), 1)
assert mu.solver == "mu" and mu.decomposition_.solver == "mu"
assert mu.converged_ and mu.last_transform_converged_
assert mu_labels.shape == (5,) and mu_probabilities.shape == (5, 2)
assert np.isfinite(mu_probabilities).all()
np.testing.assert_allclose(mu_probabilities.sum(axis=1), 1)
np.testing.assert_array_equal(mu.selected_components_, mu_basis)
assert logistic.classifier == "logistic" and logistic.classifier_converged_
assert logistic.converged_ and logistic.last_transform_converged_
assert logistic_labels.shape == (5,) and logistic_probabilities.shape == (5, 2)
np.testing.assert_allclose(logistic_probabilities.sum(axis=1), 1)
np.testing.assert_array_equal(logistic_labels, logistic.predict(data.data[:5]))

# Check the installed coordinate policy and new heads against public solvers.
with threadpool_limits(limits=1):
    query = np.vstack([data.data[:3], data.data[-3:]]) * np.float32(1.07)
    basis = model.selected_components_.copy()
    selected_train = np.asarray(data.data[:, model.selected_features_], dtype=np.float64)
    selected_query = np.asarray(query[:, model.selected_features_], dtype=np.float64)
    expected_train = np.vstack([nnls(basis.T, row, maxiter=model.max_iter)[0]
                                for row in selected_train])
    expected_query = np.vstack([nnls(basis.T, row, maxiter=model.max_iter)[0]
                                for row in selected_query])
    assert model.training_coordinates_mode_ == "transform"
    assert model.training_projection_performed_ and model.training_projection_converged_
    np.testing.assert_allclose(model.training_coordinates_, expected_train, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(model.transform(query), expected_query, rtol=2e-10, atol=2e-10)
    assert isinstance(model.classifier_, KNeighborsClassifier) and model.knn_normalization_ == "none"
    default_reference = KNeighborsClassifier(n_neighbors=model.n_neighbors).fit(expected_train, data.target)
    np.testing.assert_array_equal(model.predict(query), default_reference.predict(expected_query))
    np.testing.assert_allclose(model.predict_proba(query), default_reference.predict_proba(expected_query))
    for iterative in (override, mu):
        assert iterative.training_coordinates == "auto"
        assert iterative.training_coordinates_mode_ == "fit_transform"
        assert not iterative.training_projection_performed_

    legacy = NMFClassifier(training_coordinates="fit_transform", random_state=42).fit(data.data, data.target)
    assert legacy.training_coordinates_mode_ == "fit_transform"
    assert not legacy.training_projection_performed_
    assert legacy.training_projection_converged_ is None
    np.testing.assert_array_equal(legacy.selected_features_, model.selected_features_)
    np.testing.assert_array_equal(legacy.selected_components_, basis)
    np.testing.assert_array_equal(
        legacy.training_coordinates_, selected_train[:, legacy.decomposition_.anchor_indices_])
    np.testing.assert_allclose(legacy.transform(query), expected_query, rtol=2e-10, atol=2e-10)
    legacy_reference = KNeighborsClassifier(n_neighbors=legacy.n_neighbors).fit(
        legacy.training_coordinates_, data.target)
    np.testing.assert_array_equal(legacy.predict(query), legacy_reference.predict(expected_query))

    sd = NMFClassifier(knn_normalization="sd", random_state=42).fit(data.data, data.target)
    assert sd.knn_normalization_ == "sd"
    assert isinstance(sd.classifier_, Pipeline)
    assert not sd.classifier_.named_steps["scale"].with_mean
    np.testing.assert_array_equal(sd.selected_components_, basis)
    np.testing.assert_allclose(sd.training_coordinates_, expected_train, rtol=2e-10, atol=2e-10)
    sd_query = sd.transform(query)
    np.testing.assert_allclose(sd_query, expected_query, rtol=2e-10, atol=2e-10)
    sd_reference = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("knn", KNeighborsClassifier(n_neighbors=sd.n_neighbors, weights=sd.weights)),
    ]).fit(sd.training_coordinates_, data.target)
    scale_before = sd.classifier_.named_steps["scale"].scale_.copy()
    sd_labels, sd_probabilities = sd.predict_with_probabilities(query)
    np.testing.assert_array_equal(sd_labels, sd_reference.predict(sd_query))
    np.testing.assert_allclose(sd_probabilities, sd_reference.predict_proba(sd_query), rtol=1e-12, atol=1e-12)
    assert sd_probabilities.shape == (len(query), 2) and np.isfinite(sd_probabilities).all()
    assert ((sd_probabilities >= 0) & (sd_probabilities <= 1)).all()
    np.testing.assert_allclose(sd_probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(sd.classifier_.named_steps["scale"].scale_, scale_before)
    assert sd.classifier_.named_steps["scale"].n_samples_seen_ == len(data.target)

    svm = NMFClassifier(classifier="linear_svm", random_state=42).fit(data.data, data.target)
    assert isinstance(svm.classifier_, LinearSVC) and not svm.classifier_.dual
    assert svm.knn_normalization_ is None
    assert not hasattr(svm, "predict_proba") and not hasattr(svm, "predict_with_probabilities")
    np.testing.assert_array_equal(svm.selected_components_, basis)
    np.testing.assert_allclose(svm.training_coordinates_, expected_train, rtol=2e-10, atol=2e-10)
    svm_query = svm.transform(query)
    np.testing.assert_allclose(svm_query, expected_query, rtol=2e-10, atol=2e-10)
    svm_reference = LinearSVC(
        C=svm.svm_C, penalty="l2", loss="squared_hinge", dual=False,
        max_iter=svm.svm_max_iter, tol=svm.svm_tol, class_weight=svm.svm_class_weight,
        random_state=42).fit(svm.training_coordinates_, data.target)
    svm_labels, margins = svm.predict_with_decision_function(query)
    assert margins.shape == (len(query),) and np.isfinite(margins).all()
    np.testing.assert_allclose(margins, svm_reference.decision_function(svm_query), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(svm_labels, svm_reference.predict(svm_query))
    np.testing.assert_array_equal(svm_labels, svm.classes_[(margins > 0).astype(int)])
    np.testing.assert_array_equal(svm_labels, svm.predict(query))
    np.testing.assert_allclose(margins, svm.decision_function(query), rtol=1e-12, atol=1e-12)
    try:
        NMFClassifier(classifier="linear_svm", decision_threshold=.5).fit(data.data, data.target)
    except ValueError as error:
        assert "requires probabilities" in str(error), str(error)
    else:
        raise AssertionError("Linear SVM must reject a probability decision threshold.")
    for fitted in (model, legacy, sd, svm):
        assert fitted.converged_ and fitted.last_transform_converged_
        assert fitted.classifier_converged_
        np.testing.assert_array_equal(fitted.selected_components_, basis)

with tarfile.open(root / "dist" / f"supervised_nmf-{version}.tar.gz") as archive:
    members = archive.getnames()
    for license_file in license_files:
        member = f"supervised_nmf-{version}/{license_file}"
        assert archive.getmember(member).isfile(), member
        with archive.extractfile(member) as stream:
            assert stream.read() == (root / license_file).read_bytes(), license_file
    assert not any(set(Path(n).parts) & {"tmp", "data", "runs"} or "ensemble-reference" in n for n in members)
    assert not any(n.endswith(withdrawn_workflow) for n in members)
    for suffix in ("provenance/original/sim.py", "provenance/original/snmf_am.py",
                   "examples/compare_simulation.py", "examples/celeba_classification.py",
                   "examples/celeba_parameter_study.py", "examples/celeba_starting_grid.json",
                   "src/supervised_nmf/evaluation.py", "src/supervised_nmf/study.py",
                   "tests/test_study.py", "docs/celeba-study.md",
                   "scripts/profile_celeba_stages.py", "scripts/report_celeba_study.py",
                   "scripts/review_celeba_cv.py", "scripts/celeba_default_study.py",
                   "scripts/select_celeba_default.py", "scripts/evaluate_celeba_frozen_default.py",
                   "scripts/prepare_celeba_default_caches.py", "scripts/plan_celeba_default_study.py",
                   "scripts/plan_celeba_default_repetitions.py", "scripts/preflight_celeba_default_grid.py",
                   "examples/celeba_default_decision_protocol.json", "docs/default-selection-protocol.md",
                   "docs/celeba-default-results.md", "docs/theory-implementation.md",
                   "scripts/apply_spca_preference.py", "tests/test_spca_preference.py",
                   "tests/test_default_study.py", "tests/test_default_shortlist.py",
                   "tests/test_select_celeba_default.py", "tests/test_frozen_evaluation.py",
                   "tests/test_core.py", "tests/test_logistic.py", "examples/compare_classifiers.py",
                   "requirements/robust.txt", "docs/compatibility.md", "CITATION.cff",
                   "src/supervised_nmf/search.py", "src/supervised_nmf/validation.py",
                   "tests/test_rank_search.py", "tests/test_group_validation.py",
                   "tests/test_study_improvements.py", "tests/test_phase_reporting.py",
                   "tests/test_spa_blas.py", "examples/grouped_tuning.py",
                   "src/supervised_nmf/_neighbors.py", "src/supervised_nmf/_recovery.py",
                   "src/supervised_nmf/_residual.py", "tests/test_classifier_heads.py",
                   "tests/test_margin_evaluation.py", "tests/test_neighbor_reuse.py",
                   "tests/test_numerical_range_repairs.py", "tests/test_reported_residual.py",
                   "tests/test_training_coordinate_defaults.py",
                   "docs/tuning-improvements.md"):
        assert any(n.endswith(suffix) for n in members), suffix
print("Wheel import, author/license metadata, SPA/CD/MU fit/predict, KNN/logistic/SVM heads, SD KNN, default/legacy training coordinates, explicit override, and source archive contents verified.")
