"""Numerical phase reporting follows the operation most recently attempted."""
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits
from supervised_nmf import NMFClassifier
from supervised_nmf.evaluation import evaluate_classifier, format_evaluation_summary


class PhaseReportingTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(1)
        self.addCleanup(limits.restore_original_limits)
        rng = np.random.RandomState(87)
        self.y = np.arange(40) % 2
        self.X = rng.uniform(.1, 1, (40, 8))
        self.X[:, :3] += self.y[:, None]

    def test_skipped_training_projection_is_not_reported_as_passed(self):
        model = NMFClassifier(random_state=3, score_threshold=0).fit(self.X, self.y)
        self.assertFalse(model.training_projection_performed_)
        self.assertIsNone(model.training_projection_converged_)
        text = format_evaluation_summary(evaluate_classifier(model, self.X, self.y))
        self.assertIn("| Training projection | not performed |", text)
        model.set_params(training_coordinates="transform").fit(self.X, self.y)
        self.assertTrue(model.training_projection_performed_)
        self.assertTrue(model.training_projection_converged_)

    def test_failed_projection_clears_previous_success_and_recovers(self):
        model = NMFClassifier(random_state=3, score_threshold=0).fit(self.X, self.y)
        model.transform(self.X)
        self.assertTrue(model.last_transform_converged_)
        with patch.object(model.decomposition_, "transform", side_effect=RuntimeError("NNLS iteration limit")):
            result = evaluate_classifier(model, self.X, self.y)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["diagnostics"]["last_transform_converged"])
        self.assertIn("NNLS iteration limit", result["diagnostics"]["last_transform_error"])
        self.assertIn("| Latest projection | failed |", format_evaluation_summary(result))
        model.transform(self.X)
        self.assertTrue(model.last_transform_converged_)
        self.assertIsNone(model.last_transform_error_)

    def test_invalid_projection_shape_does_not_retain_success(self):
        model = NMFClassifier(random_state=3, score_threshold=0).fit(self.X, self.y)
        model.transform(self.X)
        with self.assertRaisesRegex(ValueError, "expected"):
            model.transform(self.X[:, :-1])
        self.assertFalse(model.last_transform_converged_)
        self.assertIn("ValueError", model.last_transform_error_)


if __name__ == "__main__":
    unittest.main()
