"""Bounded rank policy, leakage boundaries and checkpoint checks."""

from fractions import Fraction
from numbers import Integral
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf import NMFClassifier
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid
from supervised_nmf.validation import build_cv_folds


@Integral.register
class _DisplayRank(Fraction):
    """A cloneable integer whose display deliberately hides its value."""

    def __new__(cls, value, denominator=1):
        rank = super().__new__(cls, value, denominator)
        if rank.denominator != 1:
            raise ValueError("Expected an integer")
        rank._value = value
        return rank

    def get_params(self, deep=True):
        return {"value": self._value}

    def __index__(self):
        return self.numerator

    def __repr__(self):
        return "Declared rank"


class _IndexOnlyRank(_DisplayRank):
    def __int__(self):
        raise NotImplementedError("Use the index protocol")


class _IntOnlyRank(_DisplayRank):
    __index__ = None


class _UnusableRank(_IntOnlyRank):
    def __int__(self):
        raise NotImplementedError("No integer conversion")


class _ConflictingRank(_DisplayRank):
    def __int__(self):
        return self.numerator + 1


class _UnrelatedTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X


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
            scoring = options["scoring"]
            stage_scores = scores[scoring] if scoring in scores else scores
            for index, params in enumerate(ParameterGrid(param_grid)):
                score = stage_scores.get(params[parameter])
                rows.append({"index": index, "parameters": params, "valid": score is not None,
                             "mean_score": score, f"mean_{scoring}": score,
                             "folds": [{"status": "valid" if score is not None else "failed",
                                        "diagnostics": {"n_selected_features": selected_features}}
                                       for _ in options["folds"]]})
            return {"candidates": rows, "tuning_seconds": 1.0,
                    "screening_diagnostics": {"candidates": [
                        {"index": row["index"], "screens": [
                            {"mode": "none", "inactive_grid_parameters": ["score_threshold"]}
                        ] if "score_threshold" in row["parameters"] else []}
                        for row in rows]},
                    "new_representation_fits": len(rows) * options["cv"],
                    "reused_representation_folds": 0}

        estimator = kwargs.pop("estimator", self.estimator)
        directory = Path(self.directory.name) / kwargs.pop("subdir", "default")
        with patch("supervised_nmf.search.calibration_knn_grid", side_effect=shared):
            return calibration_rank_search(estimator, grid, self.X, self.y,
                                           cv=3, checkpoint_dir=directory, **kwargs)

    def test_requested_metric_controls_initial_winner_and_rank_expansion(self):
        scores = {"accuracy": {1: .9, 2: .8, 4: .7},
                  "balanced_accuracy": {1: .5, 2: .7, 4: .8, 8: .6}}
        accuracy = self.run_stub(scores, scoring="accuracy", subdir="accuracy")
        self.assertEqual(accuracy["selected_parameters"]["n_components"], 1)
        self.assertEqual(accuracy["expansions_performed"], 0)
        self.assertEqual(accuracy["selected_mean_score"], .9)
        self.assertEqual(accuracy["selected_mean_accuracy"], .9)
        self.assertNotIn("selected_mean_balanced_accuracy", accuracy)
        self.assertTrue(all(options["scoring"] == "accuracy" for _, options in self.calls))
        self.calls.clear()
        balanced = self.run_stub(scores, scoring="balanced_accuracy", subdir="balanced")
        self.assertEqual(balanced["selected_parameters"]["n_components"], 4)
        self.assertEqual(balanced["expansions_performed"], 2)
        self.assertEqual(balanced["selected_mean_score"], .8)
        self.assertEqual(balanced["selected_mean_balanced_accuracy"], .8)
        self.assertNotIn("selected_mean_accuracy", balanced)
        self.assertTrue(all(options["scoring"] == "balanced_accuracy" for _, options in self.calls))

    def test_metric_change_rejects_rank_checkpoint_before_stage_runs(self):
        self.run_stub({1: .6, 2: .7}, scoring="accuracy", max_expansions=0)
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.run_stub({1: .6, 2: .7}, scoring="balanced_accuracy", max_expansions=0)
        self.assertEqual(self.calls, [])

    def test_invalid_metric_rejected_before_stage_runs(self):
        for scoring in ("roc_auc", "Accuracy", None, 1):
            with self.subTest(scoring=scoring), self.assertRaises(ValueError):
                self.run_stub({1: .6, 2: .7}, scoring=scoring)
        self.assertEqual(self.calls, [])

    def test_expanded_winner_refinement_uses_its_stage_local_diagnostics(self):
        report = self.run_stub({1: .6, 2: .7, 4: .8, 8: .65},
                               grid={"n_components": [1, 2], "score_threshold": [1, 2]})
        selected = report["candidates"][report["selected_candidate_index"]]
        self.assertEqual(selected["stage"], 1)
        self.assertEqual(selected["stage_candidate_index"], 0)
        self.assertNotEqual(selected["index"], selected["stage_candidate_index"])
        self.assertEqual(report["refinement"]["parameters"]["score_threshold"]["status"],
                         "inactive_for_selected_screening_mode")

    def test_interior_winner_does_not_expand(self):
        result = self.run_stub({1: .6, 2: .8, 4: .7}, grid={"n_components": [1, 2, 4]})
        self.assertEqual(result["scoring"], "accuracy")
        self.assertEqual(result["selected_mean_accuracy"], .8)
        self.assertNotIn("selected_mean_balanced_accuracy", result)
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

    def test_durations_are_rejected_as_ranks_and_search_limits(self):
        for duration in (np.timedelta64(3, "ns"), np.timedelta64(3, "D")):
            for options in ({"grid": {"n_components": [duration]}},
                            {"max_rank": duration}, {"max_expansions": duration}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.run_stub({}, **options)
        self.assertEqual(self.calls, [])
        self.assertFalse((Path(self.directory.name) / "default").exists())

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
            self.assertEqual(actual["mean_score"], reference["mean_score"])
            self.assertEqual(actual["mean_accuracy"], reference["mean_accuracy"])
        resumed = calibration_rank_search(estimator, grid, self.X, self.y, cv=3, seed=4,
                                          checkpoint_dir=target, max_expansions=1)
        self.assertEqual(resumed["selected_parameters"], result["selected_parameters"])
        self.assertEqual(resumed["new_representation_fits"], 0)
        self.assertGreater(resumed["reused_representation_folds"], 0)
        self.assertEqual(resumed["tuning_seconds"], result["tuning_seconds"])


class RankCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.X = np.tile([[0., 1., .2], [4., 0., .4]], (8, 1))
        self.y = np.tile([0, 1], 8)
        self.pipeline = Pipeline([("rep", PCA(svd_solver="full", random_state=17)),
                                  ("knn", KNeighborsClassifier(n_neighbors=1, algorithm="brute"))])
        self.rank = "rep__n_components"

    def run_grid(self, function, grid, directory):
        return function(self.pipeline, grid, self.X, self.y, cv=2, seed=17,
                        checkpoint_dir=Path(self.directory.name) / directory)

    def assert_fitted_width(self, stage, candidate, width):
        dimensions = {row["key"]: row["representation_shape"][1] for row in stage["representation_folds"]
                      if row["status"] == "valid"}
        self.assertEqual([dimensions[fold["shared_representation_key"]] for fold in candidate["folds"]],
                         [width, width])

    def test_equal_displays_preserve_exact_ranks_ties_and_completed_resume(self):
        ranks = [1, 2, 2**53, 2**53 + 1, 2**1100]
        grid = {self.rank: [_DisplayRank(value) for value in ranks]}
        report = self.run_grid(calibration_rank_search, grid, "tied")
        self.assertEqual(report["selected_candidate_index"], 0)
        self.assertEqual(report["selected_parameters"], {self.rank: "Declared rank"})
        self.assertEqual(report["stop_reason"], "winner_below_upper_rank_boundary")
        self.assertEqual([r["mean_score"] for r in report["candidates"]], [1., 1., None, None, None])
        for rank, candidate in zip(ranks[:2], report["candidates"][:2]):
            self.assert_fitted_width(report["stages"][0], candidate, rank)
        for result in (report, report["stages"][0]):
            self.assertEqual(result["refinement"]["parameters"][self.rank],
                             {"status": "lower_boundary_requires_expansion_or_constraint",
                              "selected": 1, "values": ranks})
        resumed = self.run_grid(calibration_rank_search, grid, "tied")
        self.assertEqual(resumed["refinement"], report["refinement"])
        self.assertEqual(resumed["new_representation_fits"], 0)
        self.assertGreater(resumed["reused_representation_folds"], 0)
        changed = {self.rank: [_DisplayRank(value) for value in [1, 3, *ranks[2:]]]}
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.run_grid(calibration_rank_search, changed, "tied")

    def test_unqualified_upper_rank_is_included_in_the_refinement(self):
        grid = {self.rank: [_DisplayRank(1), _DisplayRank(9)]}
        report = self.run_grid(calibration_rank_search, grid, "upper")
        self.assertFalse(report["search_complete"])
        self.assertEqual(report["stop_reason"], "unqualified_upper_rank_candidates")
        self.assertEqual(report["refinement"]["parameters"][self.rank]["values"], [1, 9])

    def test_expansion_keeps_the_initial_bounds_of_the_other_pca_step(self):
        estimator = Pipeline([("first", PCA(svd_solver="full", random_state=17)),
                              *self.pipeline.steps])
        secondary = "first__n_components"
        grid = {secondary: [_DisplayRank(2), _DisplayRank(3), _DisplayRank(9)], self.rank: [1]}
        directory = Path(self.directory.name) / "two-pca"
        # The second component separates the classes. Each validation point
        # has a training twin, and both halves have identical covariance.
        X = np.tile([[-5., -1., 0.], [-5., 1., 0.], [5., -1., 0.], [5., 1., 0.]], (4, 1))
        first, second = np.split(np.arange(len(X)), 2)
        options = dict(cv=2, seed=17, folds=[(first, second), (second, first)], max_expansions=1,
                       rank_parameter=self.rank, checkpoint_dir=directory)
        report = calibration_rank_search(estimator, grid, X, self.y, **options)
        self.assertEqual(report["expansions_performed"], 1)
        initial, expanded = report["stages"]
        self.assertEqual([row["valid"] for row in initial["candidates"]], [True, True, False])
        self.assertGreater(expanded["selected_mean_score"], initial["selected_mean_score"])
        self.assertEqual(report["candidates"][report["selected_candidate_index"]]["stage"], 1)
        self.assertEqual(report["selected_parameters"][secondary], "Declared rank")
        root = report["refinement"]["parameters"]
        self.assertEqual(root[secondary], {"status": "lower_boundary_requires_expansion_or_constraint",
                                           "selected": 2, "values": [2, 3, 9]})
        self.assertEqual(root[secondary], initial["refinement"]["parameters"][secondary])
        self.assertEqual(expanded["refinement"]["parameters"][secondary]["status"],
                         "single_value_requires_exploration")
        self.assertEqual(root[self.rank]["values"], [1, 2])
        resumed = calibration_rank_search(estimator, grid, X, self.y, **options)
        self.assertEqual(resumed["refinement"], report["refinement"])
        self.assertEqual(resumed["new_representation_fits"], 0)

    def test_final_report_preserves_mixed_owners_of_a_secondary_rank(self):
        pca = PCA(svd_solver="full", random_state=17)
        estimator = Pipeline([("first", pca), *self.pipeline.steps])
        secondary = "first__n_components"
        grid = [{secondary: [2], self.rank: [1]},
                {"first": [_UnrelatedTransformer()], secondary: [3], self.rank: [1]}]
        report = calibration_rank_search(estimator, grid, self.X, self.y, cv=2, seed=17,
                                         max_expansions=1, rank_parameter=self.rank,
                                         checkpoint_dir=Path(self.directory.name) / "mixed-secondary")
        self.assertEqual(report["expansions_performed"], 1)
        self.assertEqual([row["valid"] for row in report["stages"][0]["candidates"]], [True, True])
        self.assertEqual(report["refinement"]["parameters"][secondary], {"status": "categorical", "selected": 2})

    def test_index_only_and_int_only_ranks_keep_pca_fit_behavior(self):
        for name, rank, qualified in (("index", _IndexOnlyRank(2), True), ("int", _IntOnlyRank(2), False)):
            with self.subTest(protocol=name):
                report = self.run_grid(calibration_rank_search, {self.rank: [1, rank]}, name)
                self.assertEqual([r["valid"] for r in report["candidates"]], [True, qualified])
                self.assertEqual(report["refinement"]["parameters"][self.rank]["values"], [1, 2])
                self.assertEqual(report["search_complete"], qualified)
                self.assertEqual(report["stop_reason"], "winner_below_upper_rank_boundary" if qualified
                                 else "unqualified_upper_rank_candidates")
                if qualified:
                    self.assert_fitted_width(report["stages"][0], report["candidates"][1], 2)

    def test_unusable_and_conflicting_counts_leave_shared_grid_results_intact(self):
        for name, rank, qualified in (("unusable", _UnusableRank(2), False),
                                      ("conflicting", _ConflictingRank(2), True)):
            with self.subTest(protocol=name):
                grid = {self.rank: [rank, 1]}
                report = self.run_grid(calibration_knn_grid, grid, name)
                self.assertEqual([r["valid"] for r in report["candidates"]], [qualified, True])
                entry = report["refinement"]["parameters"][self.rank]
                if qualified:
                    self.assert_fitted_width(report, report["candidates"][0], 2)
                    self.assertEqual(entry, {"status": "categorical", "selected": "Declared rank"})
                else:
                    self.assertEqual(entry, {"status": "single_value_requires_exploration", "selected": 1})
                with self.assertRaises(ValueError):
                    self.run_grid(calibration_rank_search, grid, "bounded-" + name)
                self.assertFalse((Path(self.directory.name) / ("bounded-" + name)).exists())

    def test_selected_longdouble_pca_fraction_and_adjacent_bounds_are_exact(self):
        lower = np.longdouble("0.5")
        upper = np.nextafter(lower, np.longdouble(1))
        grid = {self.rank: [upper, lower, 1]}

        def exact(value):
            if isinstance(value, dict):
                return Fraction(int(value["significand"])) * Fraction(2) ** value["exponent"]
            return Fraction(value)

        # Some sklearn versions reject longdouble fractions at parameter
        # validation. PCA's full solver can fit them when that check is skipped.
        for skip in (False, True):
            with self.subTest(skip_parameter_validation=skip), config_context(skip_parameter_validation=skip):
                report = self.run_grid(calibration_knn_grid, grid, f"fractions-{skip}")
                scores = [r["mean_score"] for r in report["candidates"]]
                entry = report["refinement"]["parameters"][self.rank]
                selected = report["selected_parameters"][self.rank]
                self.assertEqual(entry["selected"], selected)
                self.assertEqual([exact(v) for v in entry["values"]],
                                 [Fraction(*lower.as_integer_ratio()), Fraction(*upper.as_integer_ratio()), 1])
                if skip or scores[0] is not None:
                    self.assertEqual(scores, [1., 1., 1.])
                    self.assertEqual(selected["__supervised_nmf_scalar__"], "longdouble-v1")
                    self.assertEqual(exact(selected), Fraction(*upper.as_integer_ratio()))
                    self.assertEqual(entry["status"], "interior_interval_requires_refinement")
                else:
                    self.assertEqual(scores, [None, None, 1.])
                    self.assertEqual(selected, 1)
                    self.assertEqual(entry["status"], "upper_boundary_requires_expansion_or_constraint")

    def test_integral_longdouble_bound_survives_publication_and_resume(self):
        exponent = min(15000, int(np.finfo(np.longdouble).maxexp) - 1)
        bound = np.ldexp(np.longdouble(1), exponent)
        grid = {self.rank: [1, bound]}
        first = self.run_grid(calibration_knn_grid, grid, "large-float")
        self.assertEqual([row["valid"] for row in first["candidates"]], [True, False])
        entry = first["refinement"]["parameters"][self.rank]
        self.assertEqual(entry["status"], "lower_boundary_requires_expansion_or_constraint")
        self.assertEqual(entry["selected"], 1)
        low, high = entry["values"]
        self.assertEqual(low, 1)
        if isinstance(high, dict):
            self.assertEqual(high, {"__supervised_nmf_scalar__": "longdouble-v1",
                                    "significand": "1", "exponent": exponent})
        else:
            self.assertEqual(int(high), 2**exponent)
        same = self.run_grid(calibration_knn_grid, grid, "large-float")
        self.assertEqual(same["refinement"], first["refinement"])
        self.assertEqual(same["new_representation_fits"], 0)
        self.assertGreater(same["reused_representation_folds"], 0)

    def test_failed_nonfinite_dimensions_do_not_supply_numeric_rank_bounds(self):
        grid = {self.rank: [_DisplayRank(1), 0.5, None, float("nan"), float("inf")]}
        report = self.run_grid(calibration_knn_grid, grid, "fractional")
        self.assertEqual([r["valid"] for r in report["candidates"]], [True, True, True, False, False])
        self.assertEqual(report["refinement"]["parameters"][self.rank],
                         {"status": "upper_boundary_requires_expansion_or_constraint", "selected": 1,
                          "values": [0.5, 1], "includes_none": True})

    def test_mixed_step_owners_keep_a_categorical_axis_and_reject_expansion(self):
        pca = self.pipeline.named_steps["rep"]
        declarations = [{"rep": [pca], self.rank: [1]},
                        {"rep": [_UnrelatedTransformer()], self.rank: [2]}]
        for order, first_two in enumerate((declarations, declarations[::-1])):
            grid = [*first_two, {"rep": [pca], self.rank: [2**1100]}]
            with self.subTest(order=order):
                report = self.run_grid(calibration_knn_grid, grid, f"shared-{order}")
                self.assertEqual([r["valid"] for r in report["candidates"]], [True, True, False])
                self.assertEqual(report["refinement"]["parameters"][self.rank],
                                 {"status": "categorical", "selected": first_two[0][self.rank][0]})
                with self.assertRaisesRegex(TypeError, "Rank expansion"):
                    self.run_grid(calibration_rank_search, grid, f"bounded-{order}")
                self.assertFalse((Path(self.directory.name) / f"bounded-{order}").exists())

    def test_resume_checks_sklearn_parameter_validation_before_reusing_folds(self):
        for search, ranks in ((calibration_knn_grid, [np.longdouble("0.5"), 1]),
                              (calibration_rank_search, [1, 2])):
            with self.subTest(search=search.__name__):
                grid = {self.rank: ranks}
                with config_context(skip_parameter_validation=False):
                    initial = self.run_grid(search, grid, search.__name__)
                    same = self.run_grid(search, grid, search.__name__)
                self.assertEqual(initial["selected_parameters"], same["selected_parameters"])
                self.assertEqual([(row["valid"], row["mean_score"]) for row in initial["candidates"]],
                                 [(row["valid"], row["mean_score"]) for row in same["candidates"]])
                self.assertEqual(same["new_representation_fits"], 0)
                self.assertGreater(same["reused_representation_folds"], 0)
                with config_context(skip_parameter_validation=True), self.assertRaisesRegex(ValueError, "incompatible"):
                    self.run_grid(search, grid, search.__name__)


if __name__ == "__main__":
    unittest.main()
