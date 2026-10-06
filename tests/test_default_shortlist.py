"""Pure training-only shortlist regressions. Author: Chang Cheng.

Synthetic CV records only: no models, private data or saved results are loaded.
"""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_supervised_nmf_test_shortlist_planner", ROOT / "scripts/plan_celeba_default_repetitions.py")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)
shortlist = planner.shortlist


def row(index, rank, score, k=1, valid=True):
    return {"index": index, "parameters": {"n_components": rank, "percentile": 2, "n_neighbors": k},
            "mean_balanced_accuracy": score, "valid": valid}


class ShortlistChecks(unittest.TestCase):
    def test_incumbent_and_distinct_ranked_alternatives(self):
        rows = [row(0, 10, .6), row(1, 20, .8), row(2, 20, .9, k=3),
                row(3, 30, .85), row(4, 40, .95, valid=False)]
        result = shortlist({"candidates": rows}, rows[0]["parameters"])
        self.assertEqual([entry["parameters"]["n_components"] for entry in result["pairs"]], [10, 20, 30])
        self.assertEqual(result["pairs"][1]["best_neighbors"], 3)
        self.assertTrue(result["incumbent_qualified"])

    def test_invalid_incumbent_uses_three_qualified_pairs_and_canonical_ties(self):
        rows = [row(0, 10, None, valid=False), row(1, 20, .8), row(2, 30, .9), row(3, 40, .8)]
        result = shortlist({"candidates": rows}, rows[0]["parameters"])
        self.assertEqual([entry["parameters"]["n_components"] for entry in result["pairs"]], [30, 20, 40])
        self.assertFalse(result["incumbent_qualified"])

    def test_insufficient_diversity_does_not_silently_shrink_budget(self):
        rows = [row(0, 10, .9), row(1, 10, .8, k=3), row(2, 20, .7)]
        with self.assertRaisesRegex(ValueError, "Fewer than three"):
            shortlist({"candidates": rows}, rows[0]["parameters"])

    def test_near_ties_are_reported_without_rounding_selection(self):
        rows = [row(0, 10, .6), row(1, 20, .8), row(2, 30, .8 + 5e-13)]
        result = shortlist({"candidates": rows}, rows[0]["parameters"])
        self.assertEqual([entry["parameters"]["n_components"] for entry in result["pairs"]], [10, 30, 20])
        diagnostics = result["near_tie_diagnostics"]
        self.assertTrue(diagnostics["diagnostic_only"])
        self.assertTrue(diagnostics["selection_uses_exact_scores_and_canonical_order"])
        self.assertEqual(len(diagnostics["pairs"]), 1)
        self.assertEqual(diagnostics["pairs"][0]["first_candidate_index"], 2)
        self.assertGreater(diagnostics["pairs"][0]["score_gap"], 0)
        self.assertLessEqual(diagnostics["pairs"][0]["score_gap"], 1e-12)


if __name__ == "__main__":
    unittest.main()
