"""Reject unsupported class labels before folds, search side effects or cache reuse."""

from contextlib import ExitStack
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf.evaluation import data_fingerprint, provenance
from supervised_nmf.screening import BinaryScreening
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid
from supervised_nmf.validation import build_cv_folds


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


class TerminalNULSearchTests(unittest.TestCase):
    def setUp(self):
        self.y = np.arange(24) % 2
        self.X = np.random.default_rng(841).uniform(0.1, 1, (24, 6))
        self.X[:, :2] += self.y[:, None]
        # Metadata values deliberately end in NUL; label policy must not leak
        # into subject or row identifier validation.
        self.groups = np.array([f"group-{i // 2}\0" for i in range(24)], dtype=object)
        self.ids = np.array([f"row-{i}\0" for i in range(24)], dtype=object)
        group_numbers = np.arange(24) // 2
        self.folds = [(np.flatnonzero(group_numbers % 3 != fold),
                       np.flatnonzero(group_numbers % 3 == fold)) for fold in range(3)]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def estimator(self):
        return Pipeline([
            ("screen", BinaryScreening(screening="percentile", percentile=100)),
            ("pca", PCA(n_components=1, svd_solver="full", random_state=0)),
            ("knn", KNeighborsClassifier(n_neighbors=1)),
        ])

    def labels(self, pair=("a", "a\0")):
        values = [pair[index % 2] for index in range(24)]
        yield "list", values
        yield "tuple", tuple(values)
        yield "object", np.array(values, dtype=object)
        yield "numpy_strings", np.array([np.str_(v) for v in values], dtype=object)
        if hasattr(np, "dtypes") and hasattr(np.dtypes, "StringDType"):
            yield "string_dtype", np.array(values, dtype=np.dtypes.StringDType())

    def call(self, kind, y, directory, **options):
        common = dict(cv=3, seed=7, threads=1, checkpoint_dir=directory,
                      groups=self.groups, data_ids=self.ids, folds=self.folds)
        common.update(options)
        grid = {"pca__n_components": [1, 2], "knn__n_neighbors": [1, 3]}
        if kind == "rank":
            return calibration_rank_search(self.estimator(), grid, self.X, y,
                                           max_expansions=0, **common)
        return calibration_knn_grid(self.estimator(), grid, self.X, y,
                                    execution_order=kind, **common)

    def test_standalone_generated_grouped_and_explicit_folds_reject_exact_labels(self):
        modes = ({}, {"groups": self.groups}, {"groups": self.groups, "folds": self.folds})
        for pair in (("a", "a\0"), ("a\0", "a\0\0"), ("\0", "\0\0"),
                     ("a\0b", "a\0b\0")):
            for representation, y in self.labels(pair):
                before = list(y)
                for options in modes:
                    with self.subTest(pair=pair, representation=representation, options=tuple(options)):
                        with self.assertRaisesRegex(ValueError, "Class labels ending in a NUL"):
                            build_cv_folds(self.X, y, cv=3, seed=7, **options)
                        self.assertEqual(list(y), before)

    def test_searches_reject_before_clone_identity_runtime_lock_or_output(self):
        for kind in ("representation", "fold", "rank"):
            module = "supervised_nmf.search" if kind == "rank" else "supervised_nmf.study"
            for representation, y in self.labels(("a", "long\0")):
                directory = self.directory / f"{kind}-{representation}"
                x_before, y_before = self.X.copy(), list(y)
                with self.subTest(kind=kind, representation=representation), ExitStack() as stack:
                    for name in ("clone", "implementation_sources", "_runtime_identity", "_checkpoint_lock"):
                        stack.enter_context(patch(f"{module}.{name}",
                                                  side_effect=AssertionError(f"Unexpected {name}")))
                    with self.assertRaisesRegex(ValueError, "Class labels ending in a NUL"):
                        self.call(kind, y, directory)
                self.assertFalse(directory.exists())
                np.testing.assert_array_equal(self.X, x_before)
                self.assertEqual(list(y), y_before)
                self.assertTrue(self.X.flags.writeable)
                if isinstance(y, np.ndarray):
                    self.assertTrue(y.flags.writeable)

    def test_input_shape_and_execution_control_errors_keep_their_precedence(self):
        bad_y = ["a\0", "long"] * 12
        with self.assertRaisesRegex(ValueError, "matching rows"):
            build_cv_folds(self.X[:-1], bad_y, cv=3)
        for kind in ("representation", "fold", "rank"):
            directory = self.directory / kind
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "threads"):
                    self.call(kind, bad_y, directory, threads=0)
                self.assertFalse(directory.exists())

    def test_ordinary_generated_folds_match_sklearn(self):
        expected = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=7).split(self.X, self.y))
        for labels in (self.y, self.y.tolist(), tuple(self.y.tolist()),
                       ["absent", "present"] * 12, ["a\0x", "b\0y"] * 12):
            actual = build_cv_folds(self.X, labels, cv=3, seed=7)
            for expected_pair, actual_pair in zip(expected, actual):
                for a, b in zip(expected_pair, actual_pair):
                    np.testing.assert_array_equal(a, b)

    def test_metadata_nuls_remain_valid_in_real_grouped_and_explicit_folds(self):
        for options in ({"groups": self.groups}, {"groups": self.groups, "folds": self.folds}):
            folds = build_cv_folds(self.X, self.y, cv=3, seed=7, **options)
            np.testing.assert_array_equal(np.sort(np.concatenate([b for _, b in folds])), np.arange(24))
            for train, valid in folds:
                self.assertFalse(set(self.groups[train]) & set(self.groups[valid]))

    def test_completed_shared_caches_reject_raw_labels_before_reuse_and_remain_unchanged(self):
        # Both inputs coerce to byte-identical U4 arrays. Without inspection of
        # the original list, the changed labels can reuse the ordinary cache.
        ordinary, unsupported = ["a", "long"] * 12, ["a\0", "long"] * 12
        self.assertEqual(np.asarray(ordinary).dtype, np.asarray(unsupported).dtype)
        self.assertEqual(np.asarray(ordinary).tobytes(), np.asarray(unsupported).tobytes())
        for order in ("representation", "fold"):
            directory = self.directory / order
            with self.subTest(order=order):
                initial = self.call(order, ordinary, directory)
                self.assertEqual(initial["execution_order"]["resolved"], order)
                self.assertTrue(all(row["valid"] for row in initial["candidates"]))
                identity = initial["identity"]["calibration"]
                self.assertEqual(identity["groups"], self.groups.tolist())
                self.assertEqual(identity["data_ids"], self.ids.tolist())
                before = snapshot(directory)
                with patch("supervised_nmf.study._read_receipt", side_effect=AssertionError("Unexpected cache read")), \
                     patch("supervised_nmf.study._checkpoint_lock", side_effect=AssertionError("Unexpected lock")):
                    with self.assertRaisesRegex(ValueError, "Class labels ending in a NUL"):
                        self.call(order, unsupported, directory)
                self.assertEqual(snapshot(directory), before)
                with patch("supervised_nmf.study._run_representation_fold",
                           side_effect=AssertionError("Unexpected representation fit")):
                    resumed = self.call(order, ordinary, directory)
                self.assertEqual(resumed["new_representation_fits"], 0)
                self.assertEqual(resumed["candidates"], initial["candidates"])

    def test_completed_rank_cache_rejects_raw_labels_without_stage_or_file_changes(self):
        ordinary, unsupported = ["a", "long"] * 12, ["a\0", "long"] * 12
        directory = self.directory / "rank-complete"
        initial = self.call("rank", ordinary, directory)
        self.assertTrue(all(row["valid"] for row in initial["candidates"]))
        before = snapshot(directory)
        with patch("supervised_nmf.search._read_receipt", side_effect=AssertionError("Unexpected cache read")), \
             patch("supervised_nmf.search.calibration_knn_grid", side_effect=AssertionError("Unexpected stage")), \
             patch("supervised_nmf.search._checkpoint_lock", side_effect=AssertionError("Unexpected lock")):
            with self.assertRaisesRegex(ValueError, "Class labels ending in a NUL"):
                self.call("rank", unsupported, directory)
        self.assertEqual(snapshot(directory), before)
        with patch("supervised_nmf.study._run_representation_fold",
                   side_effect=AssertionError("Unexpected representation fit")):
            resumed = self.call("rank", ordinary, directory)
        self.assertEqual(resumed["selected_parameters"], initial["selected_parameters"])
        self.assertEqual(resumed["selected_mean_score"], initial["selected_mean_score"])

    def test_leading_and_embedded_nuls_work_in_both_real_shared_orders_and_rank(self):
        labels = ["\0absent", "pre\0sent"] * 12
        for kind in ("representation", "fold", "rank"):
            with self.subTest(kind=kind):
                result = self.call(kind, labels, self.directory / f"embedded-{kind}")
                self.assertTrue(all(row["valid"] for row in result["candidates"]))

    @unittest.skipUnless(hasattr(np.dtypes, "StringDType"), "requires NumPy StringDType")
    def test_variable_width_grouped_search_preserves_checkpoint_dtype_at_every_stage(self):
        labels = np.array(["\0absent", "\0présent"] * 12, dtype=object)
        strings = labels.astype(np.dtypes.StringDType())
        digest = data_fingerprint(self.X, strings)
        self.assertNotEqual(digest, data_fingerprint(self.X, labels))
        for kind in ("representation", "rank"):
            with self.subTest(kind=kind):
                directory = self.directory / kind
                reference = self.call(kind, labels, self.directory / f"{kind}-object", folds=None)
                actual = self.call(kind, strings, directory, folds=None)
                for row in actual["candidates"]:
                    for fold in row["folds"]:
                        self.assertEqual(fold["status"], "valid", fold.get("error"))
                self.assertEqual([row["mean_score"] for row in actual["candidates"]],
                                 [row["mean_score"] for row in reference["candidates"]])
                self.assertEqual(actual["selected_parameters"], reference["selected_parameters"])
                for report in [actual, *actual.get("stages", [])]:
                    self.assertEqual(report["identity"]["calibration"]["data_sha256"], digest)
                resumed = self.call(kind, strings.copy(), directory, folds=None)
                self.assertEqual(resumed["fingerprint"], actual["fingerprint"])
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    self.call(kind, labels, directory, folds=None)

    def test_package_identity_includes_new_label_validator(self):
        import supervised_nmf._labels as labels
        filename = Path(labels.__file__)
        self.assertEqual(provenance()["source_sha256"][filename.name],
                         hashlib.sha256(filename.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
