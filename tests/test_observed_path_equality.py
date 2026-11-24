"""Sharing an entry observation must preserve user-defined comparison behavior."""
from pathlib import PurePosixPath

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from supervised_nmf.evaluation import calibration_search


def unchanged_rows(X, *, path):
    return X


@pytest.mark.parametrize("full, entry, later, marked", [(1, 2, 3, False), (3, 2, 4, True)])
def test_path_text_compares_only_against_the_independent_entry_read(tmp_path, full, entry, later, marked):
    # Neighbouring ranks compare equal. Replacing the second read with the
    # first could either invent a drift or hide one at fold publication.
    ranks = iter([full, entry])
    comparisons = []

    class RankedText(str):
        def __eq__(self, other):
            if type(other) is RankedText:
                comparisons.append((self.rank, other.rank))
                return abs(self.rank - other.rank) <= 1
            return super().__eq__(other)

    class RankedPath(PurePosixPath):
        def __str__(self):
            text = RankedText(super().__str__())
            text.rank = next(ranks, later)
            return text

    transform = FunctionTransformer(unchanged_rows, kw_args={"path": RankedPath("registered")})
    model = Pipeline([("map", transform), ("knn", KNeighborsClassifier(n_neighbors=1))])
    X = np.arange(24).reshape(12, 2)
    result = calibration_search(model, {}, X, np.arange(len(X)) % 2, cv=2, refit=False,
                                seed=17, checkpoint_path=tmp_path / "cv.json")
    assert all(f["status"] == "valid" for f in result.report["candidates"][0]["folds"])
    assert result.report.get("resume_eligible", True) is (not marked)
    assert comparisons == [(later, entry)] * (1 if marked else 2)
