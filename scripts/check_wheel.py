"""Verify the locally built wheel without installing into the conda environment.

Run with `python -I scripts/check_wheel.py` to exclude source-checkout imports.
"""

from email.parser import Parser
from pathlib import Path
import sys
import tarfile
import zipfile

root = Path(__file__).resolve().parents[1]
withdrawn_workflow = ("supervised_nmf/search.py", "supervised_nmf/tuning.py",
                      "examples/adaptive_policy.json", "examples/adaptive_support.py",
                      "tests/test_search.py", "tests/test_tuning.py", "tests/test_adaptive_examples.py",
                      "docs/adaptive-tuning.md", "provenance/tuning-reference.json")
wheel = root / "dist" / "supervised_nmf-0.1.0a0-py3-none-any.whl"
with zipfile.ZipFile(wheel) as archive:
    metadata_file = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
    metadata = Parser().parsestr(archive.read(metadata_file).decode("utf-8"))
    assert metadata["Author"] == "Chang Cheng"
    assert metadata["Version"] == "0.1.0a0"
    assert "study" in metadata.get_all("Provides-Extra", [])
    assert any("matplotlib" in requirement and 'extra == "study"' in requirement
               for requirement in metadata.get_all("Requires-Dist", []))
    assert not any(n.endswith((".jpg", ".png", ".pdf")) for n in archive.namelist())
    assert not any(set(Path(n).parts) & {"tmp", "data", "runs"} or "ensemble-reference" in n
                   for n in archive.namelist())
    assert not any(n.endswith(withdrawn_workflow) for n in archive.namelist())

sys.path.insert(0, str(wheel))
import numpy as np
from threadpoolctl import threadpool_limits
import supervised_nmf
from supervised_nmf import NMFClassifier
from supervised_nmf.datasets import make_original_simulation
from supervised_nmf.study import calibration_knn_grid

assert ".whl" in supervised_nmf.__file__, supervised_nmf.__file__
assert supervised_nmf.__author__ == "Chang Cheng"
assert callable(calibration_knn_grid)
defaults = NMFClassifier().get_params()
assert defaults["solver"] == "spa", defaults
assert defaults["screening"] == "absolute", defaults
assert defaults["max_iter"] == 20000, defaults
assert defaults["tol"] == 1e-4, defaults
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

with tarfile.open(root / "dist" / "supervised_nmf-0.1.0a0.tar.gz") as archive:
    members = archive.getnames()
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
                   "tests/test_core.py", "CITATION.cff"):
        assert any(n.endswith(suffix) for n in members), suffix
print("Wheel import, author metadata, SPA/CD/MU fit/predict, installed defaults, explicit override, and source archive contents verified.")
