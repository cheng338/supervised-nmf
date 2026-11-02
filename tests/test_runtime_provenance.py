"""Cold-process calibration must distinguish lazy loading from runtime changes."""
import json
from pathlib import Path
import subprocess
import sys
import textwrap


def test_fresh_process_calibration_finishes_and_resumes(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    program = "import sys; sys.path.insert(0, " + repr(str(source)) + ")\n"
    program += textwrap.dedent("""
        import json
        import numpy as np
        from supervised_nmf import NMFClassifier
        from supervised_nmf.study import calibration_knn_grid
        X = np.random.RandomState(7).uniform(.1, .3, (80, 12))
        y = np.repeat([0, 1], 40)
        X[y == 1, :4] += .5
        model = NMFClassifier(n_components=2, random_state=7,
                              score_threshold=0, max_iter=1000)
        settings = dict(cv=2, seed=7, threads=1, checkpoint_dir=sys.argv[1])
        first = calibration_knn_grid(model, {"n_neighbors": [1, 3]}, X, y, **settings)
        resumed = calibration_knn_grid(model, {"n_neighbors": [1, 3]}, X, y, **settings)
        assert all(row["valid"] for row in first["candidates"])
        assert first["new_representation_fits"] == 2
        assert resumed["new_representation_fits"] == 0
        assert resumed["reused_representation_folds"] == 2
        print(json.dumps({"completed": True, "reused": resumed["reused_representation_folds"]}))
    """)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(tmp_path / "calibration")],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["completed"]


def test_fresh_process_general_calibration_resumes_with_ordinary_and_extended_ids(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    program = "import sys; sys.path.insert(0, " + repr(str(source)) + ")\n"
    program += textwrap.dedent("""
        import json
        from unittest.mock import Mock
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from supervised_nmf.evaluation import calibration_search
        dtype = np.longdouble if sys.argv[2] == "extended" else np.float64
        rows = np.arange(12, dtype=float)
        y = np.tile([0, 1], 6)
        X = np.column_stack([3 * y + rows / 100, np.sin(rows), np.cos(rows)])
        first_half, second_half = np.arange(6), np.arange(6, 12)
        model = Pipeline([("pca", PCA(n_components=1, svd_solver="full", random_state=7)),
                          ("knn", KNeighborsClassifier(n_neighbors=1))])
        settings = dict(cv=2, seed=7, threads=1, refit=False, checkpoint_path=sys.argv[1],
                        folds=[(first_half, second_half), (second_half, first_half)],
                        data_ids=np.arange(12, dtype=dtype) + dtype("0.5"),
                        groups=np.repeat(np.arange(6, dtype=dtype), 2))
        first = calibration_search(model, {}, X, y, **settings).report
        assert first["state"] == "cv_completed"
        assert first["candidates"][0]["valid"]
        forbid_fit = Mock(side_effect=AssertionError("Completed resume reached fit"))
        PCA.fit = PCA.fit_transform = KNeighborsClassifier.fit = forbid_fit
        resumed = calibration_search(model, {}, X, y, **settings).report
        forbid_fit.assert_not_called()
        assert resumed["fingerprint"] == first["fingerprint"]
        assert resumed["candidates"] == first["candidates"]
        print(json.dumps({"completed": True, "kind": sys.argv[2]}))
    """)
    for kind in ("ordinary", "extended"):
        # Each kind starts cold; a preceding run cannot initialize its libraries.
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", program, str(tmp_path / (kind + ".json")), kind],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, kind + ": " + result.stdout + result.stderr
        assert json.loads(result.stdout) == {"completed": True, "kind": kind}
