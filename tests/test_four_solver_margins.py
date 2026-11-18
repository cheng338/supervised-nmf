"""Supported regret-margin domains and sensitivity decisions.

Author: Chang Cheng. Synthetic summaries require no estimator fits.
"""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import unittest


spec = importlib.util.spec_from_file_location(
    "_four_solver_margin_fixtures", Path(__file__).with_name("test_four_solver_report.py")
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
reporter = fixtures.reporter
METHODS, CD, SPA, MU, BPP = fixtures.METHODS, fixtures.CD, fixtures.SPA, fixtures.MU, fixtures.BPP
FIELDS = ("global_regret_limit", "per_dataset_regret_limit", "per_task_regret_limit")
INVALID_MARGINS = (0, -0.0, -.01, math.nan, math.inf, -math.inf, False, True,
                   "0.01", None, [.01], {"value": .01}, 10 ** 400, .01 + 0j)


def rows_for(*, qualified=True, scores=None, costs=None):
    rows = fixtures.rows_for(scores=scores, costs=costs)
    if not qualified:
        for row in rows:
            row["qualified"] = False
    return rows


class FourSolverMarginTests(unittest.TestCase):
    def test_primary_margins_reject_invalid_values_before_eligibility_return(self):
        for qualified in (True, False):
            rows = rows_for(qualified=qualified)
            for field in FIELDS:
                for value in INVALID_MARGINS:
                    with self.subTest(qualified=qualified, field=field, value=value):
                        rule = deepcopy(fixtures.RULE)
                        rule[field] = value
                        with self.assertRaisesRegex(ValueError, field):
                            reporter.choose(rows, rule)

    def test_explicit_sensitivity_rejects_invalid_values(self):
        # None is choose's documented primary-mode sentinel, not an override.
        for qualified in (True, False):
            rows = rows_for(qualified=qualified)
            for value in INVALID_MARGINS:
                if value is None:
                    continue
                with self.subTest(qualified=qualified, value=value):
                    with self.assertRaisesRegex(ValueError, "sensitivity.*margin|margin.*sensitivity"):
                        reporter.choose(rows, fixtures.RULE, margin=value)

    def test_sensitivity_multiplier_rejects_invalid_values(self):
        for qualified in (True, False):
            rows = rows_for(qualified=qualified)
            for value in INVALID_MARGINS:
                with self.subTest(qualified=qualified, value=value):
                    rule = deepcopy(fixtures.RULE)
                    rule["sensitivities"]["per_task_limit_multiplier"] = value
                    with self.assertRaisesRegex(ValueError, "per_task_limit_multiplier"):
                        reporter.choose(rows, rule, margin=.005)

    def test_effective_task_margin_must_remain_positive_and_finite(self):
        for qualified in (True, False):
            rows = rows_for(qualified=qualified)
            for margin, multiplier in ((math.ulp(0.0), .25), (2., sys.float_info.max)):
                with self.subTest(qualified=qualified, margin=margin, multiplier=multiplier):
                    rule = deepcopy(fixtures.RULE)
                    rule["sensitivities"]["per_task_limit_multiplier"] = multiplier
                    with self.assertRaisesRegex(ValueError, "task|per_task"):
                        reporter.choose(rows, rule, margin=margin)

    def test_slack_rejects_invalid_values_before_eligibility_return(self):
        invalid = (-.01, math.nan, math.inf, -math.inf, False, True, "0", None,
                   [], {}, 10 ** 400, 0j)
        for qualified in (True, False):
            rows = rows_for(qualified=qualified)
            for value in invalid:
                with self.subTest(qualified=qualified, value=value):
                    rule = deepcopy(fixtures.RULE)
                    rule["numerical_comparison_slack"] = value
                    with self.assertRaisesRegex(ValueError, "numerical_comparison_slack"):
                        reporter.choose(rows, rule)

    def test_zero_slack_and_default_slack_remain_valid(self):
        ordinary = reporter.choose(rows_for(), fixtures.RULE)
        rule = deepcopy(fixtures.RULE)
        del rule["numerical_comparison_slack"]
        self.assertEqual(reporter.choose(rows_for(), rule), ordinary)
        rule["numerical_comparison_slack"] = 0
        self.assertEqual(reporter.choose(rows_for(), rule), ordinary)
        blocked = reporter.choose(rows_for(qualified=False), rule)
        self.assertEqual(blocked["status"], "blocked_no_eligible_method")
        json.dumps(blocked, allow_nan=False)

    def test_describe_preflights_every_listed_margin_even_without_workloads(self):
        label = re.escape("global_and_dataset_regret_limits[1]")
        for qualified in (True, False):
            rows = rows_for(qualified=qualified)
            for workloads in ([1], []):
                for value in INVALID_MARGINS:
                    with self.subTest(qualified=qualified, workloads=workloads, value=value):
                        rule = deepcopy(fixtures.RULE)
                        rule["sensitivities"]["global_and_dataset_regret_limits"] = [.005, value, .02]
                        rule["sensitivities"]["prediction_cost_multipliers"] = workloads
                        with self.assertRaisesRegex(ValueError, label):
                            reporter.describe(rows, rule)

    def test_describe_preflights_derived_margins_even_without_workloads(self):
        for qualified in (True, False):
            rows = rows_for(qualified=qualified)
            for margin, multiplier in ((math.ulp(0.0), .25), (2., sys.float_info.max)):
                with self.subTest(qualified=qualified, margin=margin, multiplier=multiplier):
                    rule = deepcopy(fixtures.RULE)
                    rule["sensitivities"].update(global_and_dataset_regret_limits=[.01, margin],
                                                per_task_limit_multiplier=multiplier,
                                                prediction_cost_multipliers=[])
                    with self.assertRaisesRegex(ValueError, "task|per_task"):
                        reporter.describe(rows, rule)

    def test_describe_requires_a_list_of_sensitivity_margins(self):
        for value in (None, 0, ".01", {}, (.005, .01)):
            with self.subTest(value=value):
                rule = deepcopy(fixtures.RULE)
                rule["sensitivities"]["global_and_dataset_regret_limits"] = value
                with self.assertRaisesRegex(ValueError, "global_and_dataset_regret_limits"):
                    reporter.describe(rows_for(qualified=False), rule)

    def test_describe_checks_multiplier_without_workload_evaluations(self):
        for value in INVALID_MARGINS:
            with self.subTest(value=value):
                rule = deepcopy(fixtures.RULE)
                rule["sensitivities"].update(per_task_limit_multiplier=value,
                                            prediction_cost_multipliers=[])
                with self.assertRaisesRegex(ValueError, "per_task_limit_multiplier"):
                    reporter.describe(rows_for(qualified=False), rule)

    def test_describe_rejects_invalid_primary_controls(self):
        for field in (*FIELDS, "numerical_comparison_slack"):
            with self.subTest(field=field):
                rule = deepcopy(fixtures.RULE)
                rule[field] = False
                with self.assertRaisesRegex(ValueError, field):
                    reporter.describe(rows_for(qualified=False), rule)

    def test_primary_choose_only_consumes_primary_controls(self):
        rule = deepcopy(fixtures.RULE)
        rule["sensitivities"] = None
        self.assertEqual(reporter.choose(rows_for(), rule), reporter.choose(rows_for(), fixtures.RULE))

    def test_explicit_sensitivity_does_not_consume_primary_limits(self):
        rule = deepcopy(fixtures.RULE)
        for field in FIELDS:
            del rule[field]
        result = reporter.choose(rows_for(), rule, margin=.005)
        self.assertEqual((result["global_margin"], result["dataset_margin"], result["task_margin"]),
                         (.005, .005, .01))
        self.assertEqual(result["winner"], SPA)

    def test_equal_explicit_sensitivity_has_sensitivity_meaning(self):
        rule = deepcopy(fixtures.RULE)
        rule.update(global_regret_limit=1 / 32, per_dataset_regret_limit=1 / 16,
                    per_task_regret_limit=1 / 8)
        scores = {CD: {"Fashion-MNIST": .8125, "CelebA": .75},
                  SPA: {"Fashion-MNIST": .75, "CelebA": .8125},
                  MU: {"Fashion-MNIST": .5, "CelebA": .5},
                  BPP: {"Fashion-MNIST": .5, "CelebA": .5}}
        rows = rows_for(scores=scores)
        primary = reporter.choose(rows, rule)
        self.assertEqual(primary["accuracy_acceptable_methods"], [CD, SPA])
        self.assertFalse(primary["fallback_used"])
        self.assertEqual((primary["global_margin"], primary["dataset_margin"], primary["task_margin"]),
                         (1 / 32, 1 / 16, 1 / 8))
        for margin in (math.nextafter(1 / 32, 0), 1 / 32, math.nextafter(1 / 32, math.inf)):
            with self.subTest(margin=margin):
                sensitivity = reporter.choose(rows, rule, margin=margin)
                self.assertEqual((sensitivity["global_margin"], sensitivity["dataset_margin"],
                                  sensitivity["task_margin"]), (margin, margin, margin * 2))
                self.assertEqual(sensitivity["accuracy_acceptable_methods"], [])
                self.assertTrue(sensitivity["fallback_used"])
                self.assertEqual(sensitivity["winner"], SPA)
        rule["sensitivities"].update(global_and_dataset_regret_limits=[1 / 32],
                                    prediction_cost_multipliers=[1])
        report = reporter.describe(rows, rule)
        self.assertFalse(report["primary"]["fallback_used"])
        self.assertTrue(all(value["fallback_used"] for value in report["sensitivity"]))

    def test_tiny_positive_margins_are_not_arbitrarily_floored_or_rejected(self):
        tiny = math.ulp(0.0)
        rule = deepcopy(fixtures.RULE)
        rule.update({field: tiny for field in FIELDS})
        decision = reporter.choose(rows_for(), rule)
        self.assertEqual((decision["global_margin"], decision["dataset_margin"], decision["task_margin"]),
                         (tiny, tiny, tiny))
        self.assertEqual(decision["winner"], SPA)
        self.assertTrue(all(score["maximum_normalized_regret"] == 0 for score in decision["scores"].values()))
        json.dumps(decision, allow_nan=False)
        sensitivity = reporter.choose(rows_for(), fixtures.RULE, margin=tiny)
        self.assertEqual(sensitivity["task_margin"], tiny * 2)
        json.dumps(sensitivity, allow_nan=False)
        blocked = reporter.choose(rows_for(qualified=False), rule)
        self.assertEqual(blocked["status"], "blocked_no_eligible_method")

    def test_representable_large_normalized_regrets_remain_supported(self):
        scores = {method: {"Fashion-MNIST": value, "CelebA": value}
                  for method, value in zip(METHODS, (.875, .75, .625, .5))}
        rule = deepcopy(fixtures.RULE)
        rule.update({field: 1e-300 for field in FIELDS})
        decision = reporter.choose(rows_for(scores=scores), rule)
        self.assertEqual(decision["winner"], CD)
        self.assertGreater(decision["scores"][SPA]["maximum_normalized_regret"], 1e298)
        json.dumps(decision, allow_nan=False)

    def test_valid_integer_controls_preserve_their_json_representation(self):
        rule = deepcopy(fixtures.RULE)
        rule.update(global_regret_limit=1, per_dataset_regret_limit=2, per_task_regret_limit=3,
                    numerical_comparison_slack=0)
        decision = reporter.choose(rows_for(), rule)
        self.assertEqual((decision["global_margin"], decision["dataset_margin"], decision["task_margin"]),
                         (1, 2, 3))
        encoded = json.loads(json.dumps(decision, allow_nan=False))
        for field in ("global_margin", "dataset_margin", "task_margin"):
            self.assertIs(type(encoded[field]), int)
        sensitivity = reporter.choose(rows_for(), rule, margin=1)
        self.assertEqual((sensitivity["global_margin"], sensitivity["dataset_margin"], sensitivity["task_margin"]),
                         (1, 1, 2))
        for field in ("global_margin", "dataset_margin", "task_margin"):
            self.assertIs(type(sensitivity[field]), int)

    def test_normalized_regret_overflow_is_rejected_before_ranking(self):
        scores = {CD: {"Fashion-MNIST": .8125, "CelebA": .75},
                  SPA: {"Fashion-MNIST": .75, "CelebA": .8125},
                  MU: {"Fashion-MNIST": .5, "CelebA": .5},
                  BPP: {"Fashion-MNIST": .5, "CelebA": .5}}
        rows = rows_for(scores=scores, costs={CD: 2, SPA: 1.96, MU: 1, BPP: 4})
        for field in FIELDS:
            with self.subTest(field=field):
                rule = deepcopy(fixtures.RULE)
                rule[field] = math.ulp(0.0)
                with self.assertRaisesRegex(ValueError, "(?i)normalized.*regret|regret.*normalized"):
                    reporter.choose(rows, rule)
        with self.assertRaisesRegex(ValueError, "(?i)normalized.*regret|regret.*normalized"):
            reporter.choose(rows, fixtures.RULE, margin=math.ulp(0.0))

    def test_ordinary_describe_produces_strict_json_and_all_sensitivities(self):
        report = reporter.describe(rows_for(), fixtures.RULE)
        self.assertEqual(report["primary"]["winner"], SPA)
        self.assertEqual(len(report["sensitivity"]), 24)
        self.assertTrue(all(value["winner"] == SPA for value in report["sensitivity"]))
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
