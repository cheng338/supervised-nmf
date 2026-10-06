"""Bounded rank policy, leakage boundaries and checkpoint checks."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf import NMFClassifier
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid
from supervised_nmf.validation import build_cv_folds


class RankSearchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(71)
        self.X = rng.uniform(0.1, 1, (48, 10))
        self.y = np.arange(48) % 2
        self.X[:, 0] += self.y
        self.estimator = NMFClassifier(screening="none", random_state=3)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.calls = []

    def run_stub(self, scores, grid=None, selected_features=10, **kwargs):
        grid = {"n_components": [1, 2], "n_neighbors": [1, 3]} if grid is None else grid

        def shared(estimator, param_grid, X, y, **options):
            self.calls.append((list(ParameterGrid(param_grid)), options))
            rows = []
            parameter = "n_components" if type(estimator) is NMFClassifier else "pca__n_components"
            for index, params in enumerate(ParameterGrid(param_grid)):
                score = scores.get(params[parameter])
                rows.append({"index": index, "parameters": params, "valid": score is not None,
                             "mean_balanced_accuracy": score,
                             "folds": [{"status": "valid" if score is not None else "failed",
                                        "diagnostics": {"n_selected_features": selected_features}}
                                       for _ in options["folds"]]})
            return {"candidates": rows, "tuning_seconds": 1.0,
                    "new_representation_fits": len(rows) * options["cv"],
                    "reused_representation_folds": 0}

        estimator = kwargs.pop("estimator", self.estimator)
        with patch("supervised_nmf.search.calibration_knn_grid", side_effect=shared):
            return calibration_rank_search(estimator, grid, self.X, self.y,
                                           cv=3, checkpoint_dir=self.directory.name, **kwargs)

    def test_interior_winner_does_not_expand(self):
        result = self.run_stub({1: .6, 2: .8, 4: .7}, grid={"n_components": [1, 2, 4]})
        self.assertEqual(result["selected_parameters"], {"n_components": 2})
        self.assertEqual(result["stop_reason"], "winner_below_upper_rank_boundary")
        self.assertTrue(result["search_complete"])
        self.assertEqual(len(self.calls), 1)

    def test_strict_improvement_at_upper_boundary_expands_other_parameters_fixed(self):
        result = self.run_stub({1: .6, 2: .7, 4: .8, 8: .65})
        self.assertEqual(result["selected_parameters"], {"n_components": 4, "n_neighbors": 1})
        self.assertEqual(self.calls[1][0], [{"n_components": 4, "n_neighbors": 1}])
        self.assertEqual(self.calls[2][0], [{"n_components": 8, "n_neighbors": 1}])
        self.assertTrue(result["search_complete"])
        self.assertFalse(result["tuning_complete"])
        self.assertFalse(result["refit_performed"])
        self.assertFalse(result["global_optimum_claimed"])
        for _, options in self.calls:
            for actual, expected in zip(options["folds"], self.calls[0][1]["folds"]):
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])

    def test_equal_score_preserves_incumbent(self):
        result = self.run_stub({1: .6, 2: .7, 4: .7})
        self.assertEqual(result["selected_parameters"], {"n_components": 2, "n_neighbors": 1})
        self.assertTrue(result["search_complete"])
        self.assertEqual(result["expansions_performed"], 1)

    def test_budget_exhaustion_is_explicitly_unresolved(self):
        result = self.run_stub({1: .6, 2: .7, 4: .8}, max_expansions=1)
        self.assertEqual(result["selected_parameters"]["n_components"], 4)
        self.assertEqual(result["stop_reason"], "expansion_budget_exhausted")
        self.assertFalse(result["search_complete"])
        self.assertTrue(result["unresolved_upper_rank_boundary"])

    def test_zero_budget_still_evaluates_initial_grid(self):
        result = self.run_stub({1: .6, 2: .7}, max_expansions=0)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["expansions_performed"], 0)
        self.assertFalse(result["search_complete"])

    def test_new_rank_is_clamped_to_fold_selected_features(self):
        result = self.run_stub({1: .6, 2: .7, 3: .8}, selected_features=3)
        self.assertEqual(self.calls[1][0][0]["n_components"], 3)
        self.assertEqual(result["stop_reason"], "physical_rank_limit")
        self.assertEqual(result["physical_rank_limit"], 3)

    def test_user_limit_clamps_new_rank_without_changing_initial_grid(self):
        result = self.run_stub({1: .6, 2: .7, 3: .8}, max_rank=3)
        self.assertEqual(self.calls[1][0][0]["n_components"], 3)
        self.assertEqual(result["stop_reason"], "user_rank_limit")
        self.assertFalse(result["search_complete"])
        self.assertTrue(result["unresolved_upper_rank_boundary"])

    def test_failed_expansion_retains_winner_and_unresolved_status(self):
        result = self.run_stub({1: .6, 2: .7})
        self.assertEqual(result["selected_parameters"]["n_components"], 2)
        self.assertEqual(result["stop_reason"], "no_qualified_expansion_candidate")
        self.assertFalse(result["search_complete"])
        self.assertFalse(result["candidates"][-1]["valid"])

    def test_no_initial_qualified_candidate_stops_without_fallback(self):
        result = self.run_stub({})
        self.assertIsNone(result["selected_parameters"])
        self.assertEqual(result["stop_reason"], "no_qualified_initial_candidate")
        self.assertEqual(len(self.calls), 1)

    def test_failed_larger_initial_ranks_do_not_resolve_the_boundary(self):
        result = self.run_stub({1: .6, 2: .7}, grid={"n_components": [1, 2, 4]})
        self.assertEqual(result["selected_parameters"]["n_components"], 2)
        self.assertEqual(result["stop_reason"], "unqualified_upper_rank_candidates")
        self.assertFalse(result["search_complete"])
        self.assertTrue(result["unresolved_upper_rank_boundary"])
        self.assertEqual(len(self.calls), 1)

    def test_pca_arpack_bound_is_strict(self):
        estimator = Pipeline([("pca", PCA(svd_solver="arpack", random_state=3)),
                              ("knn", KNeighborsClassifier())])
        result = self.run_stub({4: .7, 8: .8, 9: .9}, estimator=estimator,
                               grid={"pca__n_components": [4, 8]})
        self.assertEqual(self.calls[1][0], [{"pca__n_components": 9}])
        self.assertEqual(result["physical_rank_limit"], 9)
        self.assertEqual(result["stop_reason"], "physical_rank_limit")

    def test_resume_rejects_changed_expansion_policy(self):
        self.run_stub({1: .6, 2: .7}, max_expansions=0)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.run_stub({1: .6, 2: .7}, max_expansions=1)

    def test_resume_rejects_changed_data_ids(self):
        self.run_stub({1: .6, 2: .7}, max_expansions=0, data_ids=np.arange(48))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.run_stub({1: .6, 2: .7}, max_expansions=0, data_ids=np.arange(48) + 1)

    def test_group_metadata_and_folds_are_forwarded_without_subject_overlap(self):
        groups = np.arange(48) // 2
        pairs = build_cv_folds(self.X, self.y, cv=3, groups=groups)
        result = self.run_stub({1: .6, 2: .7, 4: .6}, groups=groups, folds=pairs)
        self.assertTrue(result["search_complete"])
        for _, options in self.calls:
            np.testing.assert_array_equal(options["groups"], groups)
            for train, valid in options["folds"]:
                self.assertFalse(set(groups[train]) & set(groups[valid]))

    def test_invalid_rank_or_budget_is_rejected_before_any_search(self):
        for options in ({"max_expansions": True}, {"max_rank": 1},
                        {"grid": {"n_components": [0]}},
                        {"grid": [{"n_components": [2]}, {"n_neighbors": [1]}]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_stub({1: .6, 2: .7}, **options)
        self.assertEqual(self.calls, [])

    def test_real_search_matches_its_explicit_candidates_and_resumes(self):
        estimator = Pipeline([("pca", PCA(svd_solver="full", random_state=3)),
                              ("knn", KNeighborsClassifier())])
        grid = {"pca__n_components": [1, 2], "knn__n_neighbors": [1, 3]}
        target = Path(self.directory.name) / "adaptive"
        result = calibration_rank_search(estimator, grid, self.X, self.y, cv=3, seed=4,
                                         checkpoint_dir=target, max_expansions=1)
        explicit = [{key: [value] for key, value in row["parameters"].items()}
                    for row in result["candidates"]]
        expected = calibration_knn_grid(estimator, explicit, self.X, self.y, cv=3, seed=4,
                                        checkpoint_dir=Path(self.directory.name) / "explicit")
        self.assertEqual(result["selected_parameters"], expected["selected_parameters"])
        for actual, reference in zip(result["candidates"], expected["candidates"]):
            self.assertEqual(actual["mean_balanced_accuracy"], reference["mean_balanced_accuracy"])
        resumed = calibration_rank_search(estimator, grid, self.X, self.y, cv=3, seed=4,
                                          checkpoint_dir=target, max_expansions=1)
        self.assertEqual(resumed["selected_parameters"], result["selected_parameters"])
        self.assertEqual(resumed["new_representation_fits"], 0)
        self.assertGreater(resumed["reused_representation_folds"], 0)
        self.assertEqual(resumed["tuning_seconds"], result["tuning_seconds"])


if __name__ == "__main__":
    unittest.main()
