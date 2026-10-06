"""Small runnable example using the selected SPA and absolute-screening defaults."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf.datasets import make_original_simulation


def main():
    data = make_original_simulation(n_samples=200, n_features=100, signal_ratio=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
    )
    # Omitted solver/screening use SPA + absolute screening; tune the cutoff on training data.
    model = NMFClassifier(
        score_threshold=2.0, n_components=2, n_neighbors=5, max_iter=20000, random_state=42,
    )
    with threadpool_limits(limits=1):
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
    print(json.dumps({
        "status": "installed_default_example_not_a_new_selection_study",
        "author": "Chang Cheng",
        "solver": model.solver,
        "screening": model.screening,
        "accuracy": accuracy_score(y_test, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "selected_features": model.selected_features_.tolist(),
        "converged": model.converged_,
        "n_iter": model.n_iter_,
    }, indent=2))


if __name__ == "__main__":
    main()

