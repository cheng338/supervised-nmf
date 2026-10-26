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
