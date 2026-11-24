"""A selected KNN prediction must reach its dtype-specific compiled dispatch slot."""

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

if __name__ == "__main__":
    sys.path[:] = json.loads(sys.argv[1])

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits


def compare_predictions(kernel_name, events, changed):
    from supervised_nmf._neighbors import _KNNQueryReuse

    dtype = np.float32 if kernel_name.endswith("32") else np.float64
    train = np.arange(24, dtype=dtype)[:, None]
    query = train[:6] + dtype(0.125)
    labels = np.arange(len(train)) % 2
    models = {}
    for k in (1, len(train)):
        model = KNeighborsClassifier(n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=1)
        models[k] = model.fit(train, labels)
        if "ClassMode" in kernel_name:
            # Both spellings represent the fitted Euclidean geometry. The
            # installed classifier selects ClassMode for the public l2 alias.
            model.set_params(metric="l2")

    def observe(call):
        events.clear()
        try:
            return call().tolist(), list(events)
        except RuntimeError as error:
            return str(error), list(events)

    for probabilities in (False, True):
        calls = tuple((k, probabilities, None, None) for k in models)
        reuse = _KNNQueryReuse(models, query, _study_call_contract=calls, _head_fit_input=train)
        for k, model in models.items():
            selected = model.predict_proba if probabilities else model.predict
            expected = observe(lambda: selected(query))
            actual = observe(lambda: reuse.predict(k, probabilities=probabilities)[0])
            assert actual == expected
            if changed:
                assert expected[1] == [k]
                if k == len(train):
                    assert expected[0] == "compiled full-width query"
        if changed:
            assert not reuse.query_records()[0].get("study_short_query_active")


def check_alias(kernel_name, when):
    from sklearn.metrics._pairwise_distances_reduction import _dispatcher

    events = []
    original = getattr(_dispatcher, kernel_name)

    class ObservedKernel:
        @staticmethod
        def compute(**kwargs):
            events.append(kwargs["k"])
            if kwargs["k"] == len(kwargs["Y"]):
                raise RuntimeError("compiled full-width query")
            return original.compute(**kwargs)

    with threadpool_limits(1):
        if when == "before":
            assert "supervised_nmf" not in sys.modules
        else:
            compare_predictions(kernel_name, events, False)
        with patch.object(_dispatcher, kernel_name, ObservedKernel):
            compare_predictions(kernel_name, events, True)
        compare_predictions(kernel_name, events, False)


@pytest.mark.skipif(
    (sklearn.__version__, joblib.__version__) not in {("1.9.1", "1.6.0"), ("1.9.0", "1.5.3")},
    reason="These dtype-specific dispatchers belong to the qualified scikit-learn builds.",
)
@pytest.mark.parametrize("kernel_name", ["ArgKmin32", "ArgKmin64", "ArgKminClassMode32", "ArgKminClassMode64"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_compiled_alias_reaches_selected_prediction(kernel_name, when):
    if when == "after":
        check_alias(kernel_name, when)
        return
    paths = [str(Path(entry or ".").resolve()) for entry in sys.path]
    child = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                            json.dumps(paths), kernel_name], capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr


if __name__ == "__main__":
    check_alias(sys.argv[2], "before")
