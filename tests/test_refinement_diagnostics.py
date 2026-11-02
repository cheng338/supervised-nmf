"""Regression checks for refinement reporting with mixed candidate grids."""

from copy import deepcopy
import unittest

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from supervised_nmf.evaluation import calibration_search, refinement_diagnostics


LOWER = "lower_boundary_requires_expansion_or_constraint"
UPPER = "upper_boundary_requires_expansion_or_constraint"
INTERIOR = "interior_interval_requires_refinement"
SINGLE = "single_value_requires_exploration"


class DeclaredNoneFailsClassifier(ClassifierMixin, BaseEstimator):
    """Make the None alternative fail while numeric alternatives qualify."""

    def __init__(self, cutoff=1):
        self.cutoff = cutoff

    def fit(self, X, y):
        if self.cutoff is None:
            raise ValueError("The declared None alternative cannot fit this fixture.")
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self.classes_[np.asarray(X[:, 0], dtype=int) % 2]


class RefinementCandidateTests(unittest.TestCase):
    def test_false_does_not_turn_lower_numeric_boundary_into_interior(self):
        report = refinement_diagnostics(
            [{"value": False}, {"value": 1}, {"value": 2}], {"value": 1}
        )
        self.assertEqual(report["parameters"]["value"], {
            "status": LOWER, "selected": 1, "values": [1.0, 2.0],
        })

    def test_true_does_not_turn_upper_numeric_boundary_into_interior(self):
        report = refinement_diagnostics(
            [{"value": 0.0}, {"value": 0.5}, {"value": True}], {"value": 0.5}
        )
        self.assertEqual(report["parameters"]["value"], {
            "status": UPPER, "selected": 0.5, "values": [0.0, 0.5],
        })

    def test_boolean_alternative_does_not_create_a_numeric_range(self):
        for boolean in (False, True, np.bool_(False), np.bool_(True)):
            with self.subTest(boolean=repr(boolean), scalar_type=type(boolean)):
                report = refinement_diagnostics(
                    [{"value": boolean}, {"value": 2}], {"value": 2}
                )
                self.assertEqual(report["parameters"]["value"], {
                    "status": SINGLE, "selected": 2,
                })

    def test_numpy_booleans_do_not_extend_numeric_endpoints(self):
        cases = (
            ([np.bool_(False), 1, 2], 1, LOWER, [1.0, 2.0]),
            ([0.0, 0.5, np.bool_(True)], 0.5, UPPER, [0.0, 0.5]),
        )
        for values, selected, status, expected_values in cases:
            with self.subTest(values=values):
                report = refinement_diagnostics(
                    [{"value": value} for value in values], {"value": selected}
                )
                self.assertEqual(report["parameters"]["value"], {
                    "status": status, "selected": selected, "values": expected_values,
                })

    def test_selected_python_and_numpy_booleans_are_categorical(self):
        candidates = [{"value": value} for value in (None, False, True, 0, 1, 2)]
        for selected in (False, True, np.bool_(False), np.bool_(True)):
            with self.subTest(selected=repr(selected), scalar_type=type(selected)):
                entry = refinement_diagnostics(candidates, {"value": selected})["parameters"]["value"]
                self.assertEqual(entry, {"status": "categorical", "selected": selected})
                self.assertIs(entry["selected"], selected)

    def test_real_zero_and_one_remain_when_boolean_aliases_are_present(self):
        report = refinement_diagnostics(
            [{"value": value} for value in (False, 0, True, 1, 2)], {"value": 1}
        )
        self.assertEqual(report["parameters"]["value"], {
            "status": INTERIOR, "selected": 1, "values": [0.0, 1.0, 2.0],
        })

    def test_ordinary_integer_boundary_and_interior_statuses_are_preserved(self):
        candidates = [{"value": 3}, {"value": 1}, {"value": 2}]
        for selected, status in ((1, LOWER), (2, INTERIOR), (3, UPPER)):
            with self.subTest(selected=selected):
                report = refinement_diagnostics(candidates, {"value": selected})
                self.assertEqual(report, {
                    "status": "pending_refinement_and_replication",
                    "parameters": {"value": {
                        "status": status, "selected": selected, "values": [1.0, 2.0, 3.0],
                    }},
                })

    def test_python_and_numpy_fractional_values_keep_numeric_order(self):
        values = [np.float64(0.75), np.float32(0.25), 0.5]
        cases = ((values[1], LOWER), (values[2], INTERIOR), (values[0], UPPER))
        for selected, status in cases:
            with self.subTest(selected=repr(selected)):
                entry = refinement_diagnostics(
                    [{"value": value} for value in values], {"value": selected}
                )["parameters"]["value"]
                self.assertEqual(entry, {
                    "status": status, "selected": selected, "values": [0.25, 0.5, 0.75],
                })
                self.assertTrue(all(type(value) is float for value in entry["values"]))
                self.assertIs(entry["selected"], selected)

    def test_python_numpy_integer_and_float_duplicates_share_numeric_values(self):
        selected = np.int64(2)
        report = refinement_diagnostics(
            [{"value": value} for value in (np.int32(1), 1.0, selected, 2.0, np.float64(3), 3)],
            {"value": selected},
        )
        self.assertEqual(report["parameters"]["value"], {
            "status": INTERIOR, "selected": selected, "values": [1.0, 2.0, 3.0],
        })
        self.assertIs(report["parameters"]["value"]["selected"], selected)

    def test_duplicate_real_values_remain_singleton(self):
        report = refinement_diagnostics(
            [{"value": value} for value in (np.int64(2), 2, 2.0)], {"value": 2}
        )
        self.assertEqual(report["parameters"]["value"], {
            "status": SINGLE, "selected": 2,
        })

    def test_numeric_range_reports_declared_none_without_changing_boundaries(self):
        report = refinement_diagnostics(
            [{"value": None}, {"value": 1}, {"value": 2}], {"value": 1}
        )
        self.assertEqual(report["parameters"]["value"], {
            "status": LOWER, "selected": 1, "values": [1.0, 2.0], "includes_none": True,
        })
        self.assertIs(report["parameters"]["value"]["includes_none"], True)

    def test_numeric_singleton_also_reports_declared_none(self):
        report = refinement_diagnostics(
            [{"value": None}, {"value": 2}, {"value": None}], {"value": 2}
        )
        self.assertEqual(report["parameters"]["value"], {
            "status": SINGLE, "selected": 2, "includes_none": True,
        })

    def test_selected_none_keeps_existing_categorical_shape(self):
        report = refinement_diagnostics(
            [{"value": None}, {"value": 1}, {"value": 2}], {"value": None}
        )
        self.assertEqual(report["parameters"]["value"], {
            "status": "categorical", "selected": None,
        })

    def test_strings_do_not_extend_numeric_range_or_become_numeric(self):
        candidates = [{"value": value} for value in ("0", "auto", 1, 2)]
        numeric = refinement_diagnostics(candidates, {"value": 1})["parameters"]["value"]
        self.assertEqual(numeric, {"status": LOWER, "selected": 1, "values": [1.0, 2.0]})
        for selected in ("0", "auto"):
            with self.subTest(selected=selected):
                categorical = refinement_diagnostics(candidates, {"value": selected})["parameters"]["value"]
                self.assertEqual(categorical, {"status": "categorical", "selected": selected})

    def test_categorical_choice_does_not_gain_numeric_none_metadata(self):
        candidates = [{"value": value} for value in (None, "auto", False, 1, 2)]
        for selected in ("auto", False, None):
            with self.subTest(selected=selected):
                report = refinement_diagnostics(candidates, {"value": selected})
                self.assertEqual(report["parameters"]["value"], {
                    "status": "categorical", "selected": selected,
                })

    def test_no_selected_candidate_retains_pending_contract(self):
        report = refinement_diagnostics(
            [{"value": None}, {"value": False}, {"value": 1}], None
        )
        self.assertEqual(report, {"status": "pending_valid_candidate", "parameters": {}})

    def test_missing_parameter_is_distinct_from_explicit_none(self):
        candidates = [{}, {"other": None}, {"value": 1}, {"value": 2}]
        absent = refinement_diagnostics(candidates, {"value": 1})["parameters"]["value"]
        self.assertEqual(absent, {"status": LOWER, "selected": 1, "values": [1.0, 2.0]})
        explicit = refinement_diagnostics(
            candidates + [{"value": None}], {"value": 1}
        )["parameters"]["value"]
        self.assertEqual(explicit, {
            "status": LOWER, "selected": 1, "values": [1.0, 2.0], "includes_none": True,
        })

    def test_none_metadata_is_parameter_specific_in_mixed_grid(self):
        candidates = [
            {"threshold": None, "rank": 1},
            {"threshold": False, "rank": 2},
            {"threshold": "auto", "rank": 1},
            {"threshold": np.float32(0.5), "rank": 2},
            {"threshold": 1.0, "rank": 1},
        ]
        report = refinement_diagnostics(candidates, {"threshold": 0.5, "rank": 1})
        self.assertEqual(report["parameters"], {
            "threshold": {
                "status": LOWER, "selected": 0.5, "values": [0.5, 1.0], "includes_none": True,
            },
            "rank": {"status": LOWER, "selected": 1, "values": [1.0, 2.0]},
        })

    def test_reporting_preserves_inputs_and_selected_scalar_identity(self):
        chosen = np.float32(0.5)
        candidates = [
            {"value": None}, {"value": False}, {"value": chosen},
            {"value": 1.0}, {"other": "auto"},
        ]
        selected = {"value": chosen}
        original_candidates = deepcopy(candidates)
        original_selected = deepcopy(selected)
        original_dictionaries = list(candidates)
        original_scalar_types = [type(candidate.get("value")) for candidate in candidates]

        first = refinement_diagnostics(candidates, selected)
        second = refinement_diagnostics(candidates, selected)

        self.assertEqual(first, second)
        self.assertEqual(candidates, original_candidates)
        self.assertEqual(selected, original_selected)
        for actual, original in zip(candidates, original_dictionaries):
            self.assertIs(actual, original)
        self.assertEqual([type(candidate.get("value")) for candidate in candidates], original_scalar_types)
        self.assertIs(selected["value"], chosen)
        self.assertIs(first["parameters"]["value"]["selected"], chosen)


class RefinementCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack([np.arange(30), np.ones(30)])
        self.y = np.arange(30) % 2

    def test_none_metadata_records_declared_alternative_even_when_every_fold_fails(self):
        result = calibration_search(
            DeclaredNoneFailsClassifier(), {"cutoff": [None, 1, 2]},
            self.X, self.y, cv=3, seed=9, threads=1, refit=False,
        )
        report = result.report
        none_candidate = report["candidates"][0]
        self.assertEqual(none_candidate["parameters"], {"cutoff": None})
        self.assertFalse(none_candidate["valid"])
        self.assertEqual(none_candidate["invalid_fold_count"], 3)
        self.assertIsNone(none_candidate["mean_balanced_accuracy"])
        self.assertEqual([fold["status"] for fold in none_candidate["folds"]], ["failed"] * 3)
        self.assertTrue(all(candidate["valid"] for candidate in report["candidates"][1:]))
        self.assertEqual(report["selected_parameters"], {"cutoff": 1})
        self.assertEqual(report["refinement"]["parameters"]["cutoff"], {
            "status": LOWER, "selected": 1, "values": [1.0, 2.0], "includes_none": True,
        })
        self.assertFalse(report["tuning_complete"])
        self.assertFalse(report["refit_performed"])
        self.assertIsNone(result.estimator)

    def test_no_qualified_candidate_remains_pending_without_none_metadata(self):
        result = calibration_search(
            DeclaredNoneFailsClassifier(), {"cutoff": [None]},
            self.X, self.y, cv=3, seed=9, threads=1, refit=False,
        )
        self.assertIsNone(result.report["selected_parameters"])
        self.assertEqual(result.report["refinement"], {
            "status": "pending_valid_candidate", "parameters": {},
        })
        self.assertIsNone(result.estimator)


if __name__ == "__main__":
    unittest.main()
