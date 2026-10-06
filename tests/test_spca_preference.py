"""Pure preference-amendment regressions. Author: Chang Cheng. No model fits."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_snmf_test_spca_preference", ROOT / "scripts/apply_spca_preference.py")
amendment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(amendment)
METHODS = list(amendment.selector.LABELS)
CDP, CDA, SPP, SPA = METHODS
PROTOCOL = {
    "methods_in_final_tie_order": METHODS,
    "attributes": ["Smiling", "Wearing_Necktie"],
    "timing": {"primary_cost_closeness_ratio": 1.05},
    "decision_rule": {"macro_regret_tolerance": .01, "attribute_regret_tolerance": .02, "numerical_comparison_slack": 1e-12},
    "sensitivity": {"attribute_margin_multiplier": 2},
}


def rows_for(scores=None, costs=None, tuning=None):
    scores = scores or {method: (.8, .7) for method in METHODS}
    costs = costs or {CDP: 4, CDA: 3, SPP: 1, SPA: 2}
    tuning = tuning or {method: 10 for method in METHODS}
    return [{"method": method, "attribute": attribute, "n_train": 2000, "seed": 43, "qualified": True,
             "metrics": {"balanced_accuracy": scores[method][i]}, "tuning_compute_seconds": tuning[method],
             "timing_pairs": [{"fit": costs[method] * .8, "prediction": costs[method] * .2} for _ in range(3)]}
            for method in METHODS for i, attribute in enumerate(PROTOCOL["attributes"])]


class PreferenceTests(unittest.TestCase):
    def test_preference_changes_fastest_overall_to_fastest_qualified_absolute(self):
        rows = rows_for()
        self.assertEqual(amendment.selector.choose(rows, PROTOCOL)["winner"], SPP)
        result = amendment.amended_choose(rows, PROTOCOL)
        self.assertEqual(result["winner"], SPA)
        self.assertEqual(result["methods_before_cost"], [CDA, SPA])

    def test_global_gate_cannot_be_recomputed_among_absolute_only(self):
        scores = {CDP: (.8, .7), CDA: (.75, .65), SPP: (.8, .7), SPA: (.75, .65)}
        result = amendment.amended_choose(rows_for(scores), PROTOCOL)
        self.assertEqual(result["winner"], SPP)
        self.assertFalse(result["screening_preference_applied"])
        self.assertAlmostEqual(result["scores"][SPA]["macro_regret"], .05)

    def test_normalization_keeps_percentile_methods(self):
        rows = rows_for()
        base = amendment.selector.choose(rows, PROTOCOL)
        result = amendment.amended_choose(rows, PROTOCOL)
        self.assertEqual(result["scores"], base["scores"])
        self.assertAlmostEqual(result["scores"][SPA]["normalized_serial_cost"], 2)

    def test_tighter_margin_reapplies_gate_before_preference(self):
        scores = {CDP: (.8, .7), CDA: (.799, .699), SPP: (.8, .7), SPA: (.794, .694)}
        rows = rows_for(scores)
        self.assertEqual(amendment.amended_choose(rows, PROTOCOL)["winner"], SPA)
        self.assertEqual(amendment.amended_choose(rows, PROTOCOL, macro_margin=.005)["winner"], CDA)

    def test_any_attribute_gate_rejects_even_near_macro_absolute(self):
        scores = {CDP: (.8, .7), CDA: (.821, .679), SPP: (.8, .7), SPA: (.8, .7)}
        result = amendment.amended_choose(rows_for(scores), PROTOCOL)
        # CDA improves one attribute but violates the other; the global maxima
        # also make the other methods miss Smiling, so retain original fallback.
        self.assertFalse(result["accuracy_acceptable_methods"])
        self.assertFalse(result["screening_preference_applied"])
        self.assertTrue(result["fallback_used"])

    def test_empty_zero_margin_gate_keeps_original_fallback(self):
        scores = {CDP: (.81, .69), CDA: (.79, .71), SPP: (.805, .705), SPA: (.80, .70)}
        rows = rows_for(scores)
        base = amendment.selector.choose(rows, PROTOCOL, macro_margin=0)
        result = amendment.amended_choose(rows, PROTOCOL, macro_margin=0)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["winner"], base["winner"])
        self.assertFalse(result["screening_preference_applied"])

    def test_five_percent_band_uses_original_tuning_tie_break(self):
        rows = rows_for(costs={CDP: 4, CDA: 2.08, SPP: 1, SPA: 2},
                        tuning={CDP: 10, CDA: 5, SPP: 10, SPA: 10})
        result = amendment.amended_choose(rows, PROTOCOL)
        self.assertEqual(result["methods_within_cost_band"], [CDA, SPA])
        self.assertEqual(result["winner"], CDA)

    def test_unqualified_absolute_cannot_be_preferred(self):
        rows = rows_for()
        next(row for row in rows if row["method"] == SPA)["qualified"] = False
        self.assertEqual(amendment.amended_choose(rows, PROTOCOL)["winner"], CDA)

    def test_no_qualified_method_remains_blocked(self):
        rows = rows_for()
        for row in rows:
            row["qualified"] = False
        result = amendment.amended_choose(rows, PROTOCOL)
        self.assertIsNone(result["winner"])
        self.assertEqual(result["status"], "blocked_no_operationally_qualified_method")

    def test_inputs_preserved_and_duplicate_cases_rejected(self):
        rows, protocol = rows_for(), deepcopy(PROTOCOL)
        before = deepcopy((rows, protocol))
        amendment.amended_choose(rows, protocol)
        self.assertEqual((rows, protocol), before)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            amendment.amended_choose(rows + [deepcopy(rows[0])], protocol)


if __name__ == "__main__":
    unittest.main()
