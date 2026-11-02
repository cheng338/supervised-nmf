"""Example CLI wiring and resume contracts, with no model fitting or image I/O."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from supervised_nmf import NMFClassifier, datasets, evaluation


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = ("compare_simulation.py", "celeba_classification.py")
SVM_DEFAULTS = {"svm_max_iter": 20000, "svm_tol": 1e-5, "svm_class_weight": None}


class ExampleClassifierCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.invocations = 0

    def _data(self, count=24, prefix="simulation"):
        return SimpleNamespace(
            data=np.arange(count * 6, dtype=float).reshape(count, 6) / 100 + .1,
            target=np.arange(count) % 2, signal_support=np.array([0, 1]), metadata={"fixture": True},
            image_ids=np.array([f"{prefix}-{index}.jpg" for index in range(count)]))

    def _run(self, script, arguments=(), *, resume=None):
        """Execute real parsing/main/factory/file reporting; stop at comparison fitting."""
        self.invocations += 1
        output = self.root / f"output-{self.invocations}"
        argv = [str(EXAMPLES / script), "--quick", "--output", str(output)]
        if script == "celeba_classification.py":
            argv += ["--dir", str(self.root / "images"), "--attributes", str(self.root / "attrs.txt"),
                     "--partitions", str(self.root / "partitions.txt")]
        if resume is not None:
            argv += ["--resume", str(resume)]
        argv += list(arguments)
        created = []
        original_directory = evaluation.new_run_directory

        def directory(*args, **kwargs):
            path = original_directory(*args, **kwargs)
            created.append(path)
            return path

        def load(*args, **kwargs):
            self.assertIn(kwargs["split"], ("train", "validation"))
            available = 24 if kwargs["split"] == "train" else 12
            return self._data(min(kwargs["max_samples"], available), kwargs["split"])

        def comparison(models, *args, **kwargs):
            # Only the surrounding example reporting needs these test records.
            return {"models": {name: {"is_fallback": False,
                    "evaluation": {"status": "valid", "metrics": {"balanced_accuracy": .5}}}
                    for name in models}}

        stdout, stderr = io.StringIO(), io.StringIO()
        original_path = list(sys.path)
        result = SimpleNamespace(error=None, output=output)
        with ExitStack() as stack:
            result.factory = stack.enter_context(patch.object(
                evaluation, "comparison_models", wraps=evaluation.comparison_models))
            result.comparison = stack.enter_context(patch.object(evaluation, "run_comparison", side_effect=comparison))
            result.simulation = stack.enter_context(patch.object(datasets, "make_simulation", side_effect=lambda **kw: self._data()))
            result.loader = stack.enter_context(patch.object(datasets, "load_celeba", side_effect=load))
            result.directory = stack.enter_context(patch.object(evaluation, "new_run_directory", side_effect=directory))
            result.writer = stack.enter_context(patch.object(evaluation, "write_json", wraps=evaluation.write_json))
            result.provenance = stack.enter_context(patch.object(evaluation, "provenance", return_value={"test_fixture": True}))
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(stderr))
            try:
                runpy.run_path(str(EXAMPLES / script), run_name="__main__")
            except (Exception, SystemExit) as error:
                result.error = error
            finally:
                sys.path[:] = original_path
        result.run_dir = Path(resume) if resume is not None else (created[0] if created else None)
        result.stdout, result.stderr = stdout.getvalue(), stderr.getvalue()
        return result

    def _legacy_config(self, script, classifier="knn"):
        common = {"quick": True, "threads": 1, "max_iter": 20000, "tol": 1e-4,
                  "classifier": classifier, "logistic_max_iter": 1000,
                  "logistic_tol": 1e-4, "logistic_class_weight": None, "cv": 5, "grid_override": {}}
        if script == "compare_simulation.py":
            common.update(seeds=[42], n_samples=120, n_features=40, n_signal=8,
                          signal_strength=2., correlation=0., correlation_scope="within_blocks",
                          scenario="additive", development_validation_fraction=.25)
        else:
            common.update(data_dir=str((self.root / "images").resolve()),
                          attributes=str((self.root / "attrs.txt").resolve()),
                          partitions=str((self.root / "partitions.txt").resolve()),
                          attribute="Smiling", max_train=1000, max_validation=500,
                          seed=42, image_size=[64, 64])
        return common

    def _configuration(self, result):
        self.assertIsNone(result.error, repr(result.error))
        return json.loads((result.run_dir / "configuration.json").read_text(encoding="utf-8"))

    def _assert_no_work(self, result):
        for mock in (result.factory, result.comparison, result.simulation, result.loader,
                     result.directory, result.writer, result.provenance):
            mock.assert_not_called()

    def test_legacy_heads_keep_configuration_and_accept_legacy_resume(self):
        for script in SCRIPTS:
            for head in (None, "knn", "logistic"):
                with self.subTest(script=script, classifier=head):
                    options = [] if head is None else ["--classifier", head]
                    first = self._run(script, options)
                    legacy = self._legacy_config(script, head or "knn")
                    self.assertEqual(self._configuration(first), legacy)
                    # Supply an independently constructed old-format configuration.
                    config_path = first.run_dir / "configuration.json"
                    config_path.write_text(json.dumps(legacy), encoding="utf-8")
                    saved = config_path.read_bytes()
                    resumed = self._run(script, options + ["--svm-max-iter", "7", "--svm-tol", ".4",
                                        "--svm-class-weight", "balanced"], resume=first.run_dir)
                    self.assertEqual(self._configuration(resumed), legacy)
                    self.assertEqual(config_path.read_bytes(), saved)
                    self.assertFalse(resumed.directory.called)
                    for call in resumed.factory.call_args_list:
                        self.assertEqual(call.kwargs["classifier"], head or "knn")
                        self.assertFalse(set(SVM_DEFAULTS) & set(call.kwargs))

    def test_svm_defaults_and_custom_controls_are_forwarded_and_persisted(self):
        variants = [([], SVM_DEFAULTS),
                    (["--svm-max-iter", "3333", "--svm-tol", "3e-6", "--svm-class-weight", "balanced"],
                     {"svm_max_iter": 3333, "svm_tol": 3e-6, "svm_class_weight": "balanced"})]
        for script in SCRIPTS:
            for flags, expected in variants:
                with self.subTest(script=script, svm_options=expected):
                    options = ["--classifier", "linear_svm", "--max-iter", "101", "--tol", "2e-4", *flags]
                    first = self._run(script, options)
                    configuration = self._legacy_config(script, "linear_svm")
                    configuration.update(expected, max_iter=101, tol=2e-4)
                    self.assertEqual(self._configuration(first), configuration)
                    for call in first.factory.call_args_list:
                        self.assertEqual({key: call.kwargs[key] for key in expected}, expected)
                    models = first.comparison.call_args.args[0]
                    self.assertEqual(len(models), 13)
                    nmf_count = pipeline_count = 0
                    for model, _ in models.values():
                        if isinstance(model, NMFClassifier):
                            nmf_count += 1
                            self.assertEqual(model.classifier, "linear_svm")
                            self.assertEqual({key: getattr(model, key) for key in expected}, expected)
                            self.assertEqual((model.max_iter, model.tol), (101, 2e-4))
                        elif isinstance(model, Pipeline):
                            pipeline_count += 1
                            head = model.named_steps["linear_svm"]
                            self.assertEqual((head.max_iter, head.tol, head.class_weight),
                                             (expected["svm_max_iter"], expected["svm_tol"], expected["svm_class_weight"]))
                    self.assertEqual((nmf_count, pipeline_count), (7, 5))
                    self.assertEqual(self._configuration(self._run(script, options, resume=first.run_dir)), configuration)

    def test_changed_svm_controls_reject_resume_before_loading_or_writing(self):
        for script in SCRIPTS:
            first = self._run(script, ["--classifier", "linear_svm"])
            self._configuration(first)
            saved = {p.relative_to(first.run_dir): p.read_bytes() for p in first.run_dir.rglob("*") if p.is_file()}
            for change in (["--svm-max-iter", "101"], ["--svm-tol", ".01"],
                           ["--svm-class-weight", "balanced"]):
                with self.subTest(script=script, changed=change[0]):
                    result = self._run(script, ["--classifier", "linear_svm", *change], resume=first.run_dir)
                    self.assertIsInstance(result.error, ValueError)
                    self.assertIn("Resume settings differ", str(result.error))
                    self._assert_no_work(result)
                    self.assertEqual({p.relative_to(first.run_dir): p.read_bytes()
                                      for p in first.run_dir.rglob("*") if p.is_file()}, saved)

    def test_invalid_classifier_or_class_weight_rejected_before_work(self):
        for script in SCRIPTS:
            for options in (["--classifier", "unknown"], ["--svm-class-weight", "unknown"],
                            ["--classifier", "linear_svm", "--svm-class-weight", "unknown"]):
                with self.subTest(script=script, options=options):
                    result = self._run(script, options)
                    self.assertIsInstance(result.error, SystemExit)
                    self.assertEqual(result.error.code, 2)
                    self.assertIn("invalid choice", result.stderr)
                    self._assert_no_work(result)
                    self.assertFalse(result.output.exists())

    def test_grid_json_overrides_cli_defaults_with_correct_parameter_names(self):
        overrides = {
            "screened_spa_percentile_linear_svm": {"n_components": [2], "percentile": [25.], "svm_C": [.25],
                "svm_max_iter": [9876], "svm_tol": [8e-6], "svm_class_weight": [None]},
            "raw_linear_svm": {"linear_svm__C": [.125], "linear_svm__max_iter": [8765],
                "linear_svm__tol": [9e-6], "linear_svm__class_weight": [None]}}
        grid_path = self.root / "grid.json"
        for script in SCRIPTS:
            with self.subTest(script=script):
                grid_path.write_text(json.dumps(overrides), encoding="utf-8")
                options = ["--classifier", "linear_svm", "--svm-max-iter", "4567",
                    "--svm-tol", "5e-5", "--svm-class-weight", "balanced", "--grid-json", str(grid_path)]
                result = self._run(script, options)
                configuration = self._configuration(result)
                self.assertEqual(configuration["grid_override"], overrides)
                self.assertEqual(configuration["svm_max_iter"], 4567)
                models = result.comparison.call_args.args[0]
                for name, grid in overrides.items():
                    estimator, actual_grid = models[name]
                    self.assertEqual(actual_grid, grid)
                    effective = clone(estimator).set_params(**next(iter(ParameterGrid(actual_grid))))
                    if isinstance(effective, NMFClassifier):
                        self.assertEqual((effective.svm_C, effective.svm_max_iter, effective.svm_tol, effective.svm_class_weight),
                                         (.25, 9876, 8e-6, None))
                    else:
                        head = effective.named_steps["linear_svm"]
                        self.assertEqual((head.C, head.max_iter, head.tol, head.class_weight), (.125, 8765, 9e-6, None))
                unchanged, grid = models["unscreened_cd_nmf_linear_svm"]
                self.assertEqual((unchanged.svm_max_iter, unchanged.svm_tol, unchanged.svm_class_weight), (4567, 5e-5, "balanced"))
                self.assertEqual(grid["svm_C"], [1.])
                resumed = self._run(script, options, resume=result.run_dir)
                self.assertEqual(self._configuration(resumed), configuration)
                saved_config = (result.run_dir / "configuration.json").read_bytes()
                changed_grid = json.loads(grid_path.read_text(encoding="utf-8"))
                changed_grid["screened_spa_percentile_linear_svm"]["svm_C"] = [.5]
                grid_path.write_text(json.dumps(changed_grid), encoding="utf-8")
                rejected = self._run(script, options, resume=result.run_dir)
                self.assertIsInstance(rejected.error, ValueError)
                self.assertIn("Resume settings differ", str(rejected.error))
                self._assert_no_work(rejected)
                self.assertEqual((result.run_dir / "configuration.json").read_bytes(), saved_config)

    def test_unknown_grid_model_is_rejected_before_comparison(self):
        grid_path = self.root / "unknown-grid.json"
        grid_path.write_text(json.dumps({"unknown_model": {}}), encoding="utf-8")
        for script in SCRIPTS:
            with self.subTest(script=script):
                result = self._run(script, ["--classifier", "linear_svm", "--grid-json", str(grid_path)])
                self.assertIsInstance(result.error, ValueError)
                self.assertIn("Unknown grid model names", str(result.error))
                self.assertIn("unknown_model", str(result.error))
                result.comparison.assert_not_called()

    def test_simulation_replication_seeds_and_duplicate_rejection(self):
        result = self._run("compare_simulation.py", ["--classifier", "linear_svm", "--seeds", "7", "11"])
        self.assertEqual(self._configuration(result)["seeds"], [7, 11])
        self.assertEqual([c.kwargs["random_state"] for c in result.simulation.call_args_list], [7, 11])
        self.assertEqual([c.kwargs["random_state"] for c in result.factory.call_args_list], [7, 11])
        self.assertEqual([c.kwargs["seed"] for c in result.comparison.call_args_list], [7, 11])
        result.loader.assert_not_called()
        for seed in (7, 11):
            manifest = json.loads((result.run_dir / f"seed-{seed}" / "data.json").read_text(encoding="utf-8"))
            training, validation = set(manifest["training_indices"]), set(manifest["validation_indices"])
            self.assertFalse(training & validation)
            self.assertEqual(training | validation, set(range(24)))
        invalid = self._run("compare_simulation.py", ["--classifier", "linear_svm", "--seeds", "7", "7"])
        self.assertIsInstance(invalid.error, ValueError)
        self.assertIn("seeds must be distinct", str(invalid.error))
        self._assert_no_work(invalid)

    def test_celeba_loads_only_training_and_validation_partitions(self):
        result = self._run("celeba_classification.py", ["--classifier", "linear_svm", "--seed", "37",
                           "--max-train", "9", "--max-validation", "5"])
        self.assertEqual(self._configuration(result)["seed"], 37)
        self.assertEqual([c.kwargs["split"] for c in result.loader.call_args_list], ["train", "validation"])
        self.assertEqual([c.kwargs["max_samples"] for c in result.loader.call_args_list], [9, 5])
        self.assertTrue(all(c.kwargs["random_state"] == 37 for c in result.loader.call_args_list))
        result.simulation.assert_not_called()
        result.comparison.assert_called_once()
        self.assertEqual(result.factory.call_args.kwargs["random_state"], 37)
        manifest = json.loads((result.run_dir / "data.json").read_text(encoding="utf-8"))
        self.assertIs(manifest["official_test_loaded"], False)
        self.assertEqual(len(manifest["training_image_ids"]), 9)
        self.assertEqual(len(manifest["validation_image_ids"]), 5)
        self.assertFalse(set(manifest["training_image_ids"]) & set(manifest["validation_image_ids"]))


if __name__ == "__main__":
    unittest.main()
