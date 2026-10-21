"""Verify the locally built wheel without installing into the conda environment.

Run with `python -I scripts/check_wheel.py` to exclude source-checkout imports.
"""

from email.parser import Parser
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

root = Path(__file__).resolve().parents[1]
withdrawn_workflow = ("supervised_nmf/tuning.py",
                      "examples/adaptive_policy.json", "examples/adaptive_support.py",
                      "tests/test_search.py", "tests/test_tuning.py", "tests/test_adaptive_examples.py",
                      "docs/adaptive-tuning.md", "provenance/tuning-reference.json")
excluded_directories = {"tmp", "data", "runs", "output", "references", "portable_speedtest"}
removed_speedtest_files = ("test_portable_speedtest.py", "test_portable_projection.py",
                          "test_portable_manifest_reuse.py", "PROMPT_FOR_OTHER_PC.txt")
version = re.search(r'^version = "([^"]+)"$',
                    (root / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE).group(1)
dist_info_name = f"supervised_nmf-{version}.dist-info"
sdist_name = f"supervised_nmf-{version}"
license_files = ("LICENSE", "THIRD_PARTY_NOTICES")


def relative_path(name):
    """Require one canonical, portable archive-relative path."""
    assert name and "\\" not in name and ":" not in name, name
    path = PurePosixPath(name)
    assert not path.is_absolute() and all(part not in {"", ".", ".."}
                                         for part in name.split("/")), name
    assert str(path) == name, name
    return path


def source_allowlist():
    lines = [line.strip() for line in (root / "MANIFEST.in").read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert lines and lines[0] == "global-exclude *", "MANIFEST.in must clear default discovery first."
    approved = set()
    for line in lines[1:]:
        fields = line.split()
        assert len(fields) == 2 and fields[0] == "include", line
        name = fields[1]
        relative_path(name)
        assert not any(character in name for character in "*?[]"), line
        assert name not in approved, f"Duplicate MANIFEST entry: {name}"
        assert not set(PurePosixPath(name).parts) & excluded_directories, name
        approved.add(name)
    required = {"MANIFEST.in", "pyproject.toml", "README.md", "CITATION.cff", *license_files}
    assert required <= approved, f"Missing source inputs: {sorted(required - approved)}"
    return {name: (root / name).read_bytes() for name in sorted(approved)}


def ancestor_directories(files):
    return {str(parent) for name in files for parent in PurePosixPath(name).parents
            if str(parent) != "."}


approved_source = source_allowlist()
package_source = {name.removeprefix("src/"): content for name, content in approved_source.items()
                  if name.startswith("src/supervised_nmf/") and name.endswith(".py")}
assert "supervised_nmf/__init__.py" in package_source
wheel_metadata = {f"{dist_info_name}/{name}" for name in ("METADATA", "WHEEL", "RECORD", "top_level.txt")}
wheel_licenses = {f"{dist_info_name}/licenses/{name}": approved_source[name] for name in license_files}
generated_source = {"PKG-INFO", "setup.cfg", "src/supervised_nmf.egg-info/SOURCES.txt"}


def verify_wheel_contents(filename):
    expected = set(package_source) | wheel_metadata | set(wheel_licenses)
    directories = ancestor_directories(expected)
    contents, seen = {}, set()
    with zipfile.ZipFile(filename) as archive:
        for member in archive.infolist():
            name = member.filename.rstrip("/") if member.is_dir() else member.filename
            relative_path(name)
            assert name not in seen, f"Duplicate wheel member: {name}"
            seen.add(name)
            kind = stat.S_IFMT(member.external_attr >> 16)
            if member.is_dir():
                assert name in directories and kind in (0, stat.S_IFDIR), name
                continue
            assert kind in (0, stat.S_IFREG), f"Nonregular wheel member: {name}"
            assert name in expected, f"Unexpected wheel member: {name}"
            contents[name] = archive.read(member)
    assert set(contents) == expected, f"Missing wheel files: {sorted(expected - set(contents))}"
    for name, content in {**package_source, **wheel_licenses}.items():
        assert contents[name] == content, f"Wheel source bytes differ: {name}"
    return contents


def verify_sdist_contents(filename):
    expected = set(approved_source) | generated_source
    directories = ancestor_directories(expected)
    contents, seen = {}, set()
    with tarfile.open(filename) as archive:
        for member in archive.getmembers():
            name = member.name.rstrip("/") if member.isdir() else member.name
            path = relative_path(name)
            assert name not in seen, f"Duplicate source archive member: {name}"
            seen.add(name)
            assert path.parts[0] == sdist_name, f"Unexpected source archive root: {name}"
            if len(path.parts) == 1:
                assert member.isdir(), name
                continue
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if member.isdir():
                assert relative in directories, f"Unexpected source directory: {relative}"
                continue
            assert member.isfile(), f"Nonregular source archive member: {name}"
            assert relative in expected, f"Unexpected source archive file: {relative}"
            with archive.extractfile(member) as stream:
                contents[relative] = stream.read()
    assert set(contents) == expected, f"Missing source files: {sorted(expected - set(contents))}"
    for name, content in approved_source.items():
        assert contents[name] == content, f"Source archive bytes differ: {name}"
    sources = contents["src/supervised_nmf.egg-info/SOURCES.txt"].decode("utf-8").splitlines()
    for name in sources:
        relative_path(name)
        assert name in expected, f"Unapproved cached source entry: {name}"
    return contents


def build_from_source(source, output, command):
    output.mkdir()
    code = f"from setuptools.build_meta import {command}; import sys; {command}(sys.argv[1])"
    try:
        subprocess.run([sys.executable, "-I", "-c", code, str(output)], cwd=source,
                       check=True, timeout=120, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        if error.stdout:
            print(error.stdout)
        raise


def verify_sdist_rebuilds(source_contents, wheel_contents):
    """Build without Git, then challenge default discovery and stale SOURCES."""
    with tempfile.TemporaryDirectory(prefix="snmf-sdist-check-") as directory:
        temporary = Path(directory).resolve()
        source = temporary / sdist_name
        source.mkdir()
        # Membership and regular-file checks above precede every write.
        for name, content in source_contents.items():
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        assert not (source / ".git").exists()
        wheel_output = temporary / "wheel"
        build_from_source(source, wheel_output, "build_wheel")
        rebuilt = verify_wheel_contents(wheel_output / wheel.name)
        compared = set(package_source) | set(wheel_licenses) | {f"{dist_info_name}/METADATA"}
        for name in compared:
            assert rebuilt[name] == wheel_contents[name], f"Rebuilt wheel differs: {name}"

        canaries = ("scripts/_packaging_canary.py", "tests/test_packaging_canary.py",
                    "docs/_packaging_canary.md", "examples/_packaging_canary.py",
                    "portable_speedtest/_packaging_canary.py", "src/supervised_nmf/_packaging_canary.py")
        for name in canaries:
            assert name not in approved_source
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"# Packaging exclusion canary.\n")
        with (source / "src/supervised_nmf.egg-info/SOURCES.txt").open("a", encoding="utf-8") as stream:
            stream.write("\n".join(canaries) + "\n")
        sdist_output = temporary / "sdist"
        build_from_source(source, sdist_output, "build_sdist")
        rebuilt_source = verify_sdist_contents(sdist_output / f"{sdist_name}.tar.gz")
        assert set(rebuilt_source) == set(source_contents)
        print("Source archive rebuild and unlisted-file exclusion verified.")


wheel = root / "dist" / f"supervised_nmf-{version}-py3-none-any.whl"
wheel_contents = verify_wheel_contents(wheel)
source_contents = verify_sdist_contents(root / "dist" / f"{sdist_name}.tar.gz")
with zipfile.ZipFile(wheel) as archive:
    metadata_file = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
    metadata = Parser().parsestr(archive.read(metadata_file).decode("utf-8"))
    assert metadata["Author"] == "Chang Cheng"
    assert metadata["Version"] == version
    assert metadata["Requires-Python"] == ">=3.11"
    assert metadata["License-Expression"] == "LicenseRef-All-Rights-Reserved AND BSD-3-Clause"
    assert set(metadata.get_all("License-File", [])) == set(license_files)
    dist_info = metadata_file.rsplit("/", 1)[0]
    for license_file in license_files:
        assert archive.read(f"{dist_info}/licenses/{license_file}") == (root / license_file).read_bytes(), license_file
    assert "study" in metadata.get_all("Provides-Extra", [])
    assert any("matplotlib" in requirement and 'extra == "study"' in requirement
               for requirement in metadata.get_all("Requires-Dist", []))
    assert not any(n.endswith((".jpg", ".png", ".pdf")) for n in archive.namelist())
    assert not any(set(Path(n).parts) & excluded_directories or "ensemble-reference" in n
                   for n in archive.namelist())
    assert not any(n.endswith(withdrawn_workflow) for n in archive.namelist())
    assert not any(Path(n).name in removed_speedtest_files for n in archive.namelist())

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
    assert not any(set(Path(n).parts) & excluded_directories or "ensemble-reference" in n for n in members)
    assert not any(n.endswith(withdrawn_workflow) for n in members)
    assert not any(Path(n).name in removed_speedtest_files for n in members)
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
verify_sdist_rebuilds(source_contents, wheel_contents)
print("Wheel import, author/license metadata, SPA/CD/MU fit/predict, KNN/logistic/SVM heads, SD KNN, default/legacy training coordinates, explicit override, and source archive contents verified.")
