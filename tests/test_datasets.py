"""Dataset contract tests using small generated arrays and local image fixtures."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from supervised_nmf.datasets import load_celeba, make_original_simulation, make_simulation


def assert_rng_equal(test, before, after):
    test.assertEqual(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    test.assertEqual(before[2:], after[2:])


class SimulationTests(unittest.TestCase):
    def test_original_preserves_archival_draw_order(self):
        # Frozen small fixture for the historical RandomState(42) generator.
        # Distinct signal/background columns and label ordering detect changed
        # draw order, distribution scales, concatenation, or clipping.
        dataset = make_original_simulation(4, 4, 0.5, 42)
        expected = np.array(
            [
                [1.3973713, 0.88938856, 0.0, 7.8830404],
                [1.5181508, 2.218424, 0.0, 0.0],
                [2.8126774, 2.8126905, 5.1776605, 0.0],
                [4.2633705, 3.6139479, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(dataset.data, expected, rtol=1e-6)
        np.testing.assert_array_equal(dataset.target, [0, 0, 1, 1])
        np.testing.assert_array_equal(dataset.signal_support, [True, True, False, False])
        self.assertEqual(dataset.data.dtype, np.float32)

    def test_generators_leave_global_rng_untouched_and_repeat(self):
        before = np.random.get_state()
        original = make_original_simulation(9, 12, 0.3, 7)
        controlled = make_simulation(10, 12, 4, 2, 0.3, 8)
        xor = make_simulation(10, 12, 4, 2, 0.3, 8, scenario="xor")
        assert_rng_equal(self, before, np.random.get_state())
        np.testing.assert_array_equal(original.data, make_original_simulation(9, 12, 0.3, 7).data)
        np.testing.assert_array_equal(controlled.data, make_simulation(10, 12, 4, 2, 0.3, 8).data)
        np.testing.assert_array_equal(xor.data, make_simulation(10, 12, 4, 2, 0.3, 8, scenario="xor").data)
        self.assertEqual(original.metadata["n_signal"], 3)
        for dataset in (original, controlled, xor):
            self.assertTrue(np.isfinite(dataset.data).all())
            self.assertGreaterEqual(dataset.data.min(), 0)

    def test_xor_is_jointly_informative_without_single_bit_prediction(self):
        dataset = make_simulation(4000, 6, 4, 8, 0.0, 29, scenario="xor")
        bit_a = dataset.data[:, :2].mean(axis=1) > 5
        bit_b = dataset.data[:, 2:4].mean(axis=1) > 5
        self.assertGreater(np.mean((bit_a ^ bit_b) == dataset.target), 0.99)
        # In this finite independent-bit fixture, neither single block predicts
        # the class. This checks the advertised stress case, not an NMF claim.
        for bit in (bit_a, bit_b):
            self.assertLess(abs(np.mean(bit == dataset.target) - 0.5), 0.04)
        self.assertEqual(dataset.metadata["xor_block_boundary"], 2)

    def test_perfect_noise_correlation_and_null_support_are_defined(self):
        dataset = make_simulation(20, 5, 0, 0, 1.0, 12)
        self.assertFalse(dataset.signal_support.any())
        for column in range(1, 5):
            np.testing.assert_array_equal(dataset.data[:, 0], dataset.data[:, column])

    def test_within_block_correlation_and_cross_block_stress_are_distinct(self):
        # Zero class shift isolates the noise correlation from label-induced
        # signal covariance. Clipping changes the requested .8 correlation,
        # but should preserve the block/no-block distinction with ample margin.
        within = make_simulation(6000, 4, 2, 0, 0.8, 16)
        across = make_simulation(6000, 4, 2, 0, 0.8, 16, correlation_scope="all_features")
        within_corr = np.corrcoef(within.data, rowvar=False)
        across_corr = np.corrcoef(across.data, rowvar=False)
        self.assertGreater(within_corr[0, 1], 0.7)
        self.assertGreater(within_corr[2, 3], 0.65)
        self.assertLess(abs(within_corr[0, 2]), 0.05)
        self.assertGreater(across_corr[0, 2], 0.5)
        self.assertEqual(within.metadata["correlation_scope"], "within_blocks")
        self.assertEqual(across.metadata["correlation_scope"], "all_features")
        shifted = make_simulation(10, 4, 2, 1.6, 0, 16)
        self.assertEqual(shifted.metadata["signal_strength"], 1.6)
        self.assertEqual(shifted.metadata["nominal_signal_to_noise"], 2.0)

    def test_invalid_parameters(self):
        for kwargs in (
            {"n_samples": 1}, {"n_features": 0}, {"n_samples": True},
            {"signal_ratio": -0.1}, {"signal_ratio": np.nan},
            {"random_state": -1}, {"signal_ratio": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_original_simulation(**kwargs)
        for kwargs in (
            {"n_signal": 7}, {"n_signal": -1}, {"correlation": 1.1},
            {"correlation": np.nan}, {"signal_strength": -1},
            {"scenario": "unknown"}, {"n_signal": 1, "scenario": "xor"},
            {"correlation_scope": "unknown"},
        ):
            settings = dict(n_samples=10, n_features=6, n_signal=2, signal_strength=1, correlation=0, random_state=1)
            settings.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_simulation(**settings)


@unittest.skipUnless(importlib.util.find_spec("PIL"), "optional Pillow dependency unavailable")
class CelebATests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.ids = [f"{index:06d}.png" for index in range(1, 8)]
        self.labels = [-1, 1, -1, 1, -1, 1, -1]
        self.codes = [0, 0, 0, 1, 1, 2, 2]
        for index, identifier in enumerate(self.ids, start=1):
            Image.new("RGB", (2, 3), (index * 30,) * 3).save(self.images / identifier)
        self.attrs = self.root / "list_attr_celeba.txt"
        self.attrs.write_text(
            "7\nSmiling Wearing_Necktie\n" + "\n".join(
                f"{identifier} {label} 1" for identifier, label in reversed(list(zip(self.ids, self.labels)))
            ) + "\n", encoding="utf-8",
        )
        self.parts = self.root / "list_eval_partition.txt"
        self.parts.write_text(
            "\n".join(f"{identifier} {code}" for identifier, code in reversed(list(zip(self.ids, self.codes)))) + "\n",
            encoding="utf-8",
        )

    def load(self, **kwargs):
        return load_celeba(self.images, self.attrs, self.parts, **kwargs)

    def test_official_text_join_partition_isolation_and_preprocessing(self):
        train = self.load(image_size=(3, 2))
        validation = self.load(split="validation", image_size=(3, 2))
        test = self.load(split="test", image_size=(3, 2))
        np.testing.assert_array_equal(train.image_ids, self.ids[:3])
        np.testing.assert_array_equal(train.target, [0, 1, 0])
        np.testing.assert_array_equal(validation.image_ids, self.ids[3:5])
        np.testing.assert_array_equal(test.image_ids, self.ids[5:])
        self.assertEqual(train.data.shape, (3, 6))
        self.assertEqual(train.data.dtype, np.float32)
        np.testing.assert_allclose(train.data[:, 0], np.array([30, 60, 90]) / 255)
        self.assertFalse(set(train.image_ids) & set(test.image_ids))

    def test_csv_attributes_and_seeded_unbalanced_sampling(self):
        csv_path = self.root / "attrs.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "Smiling"])
            writer.writerows(zip(self.ids, [int(label == 1) for label in self.labels]))
        before = np.random.get_state()
        first = load_celeba(self.images, csv_path, self.parts, max_samples=2, random_state=3)
        second = load_celeba(self.images, csv_path, self.parts, max_samples=2, random_state=3)
        assert_rng_equal(self, before, np.random.get_state())
        np.testing.assert_array_equal(first.image_ids, second.image_ids)
        self.assertTrue(set(first.image_ids).issubset(self.ids[:3]))
        self.assertEqual(len(first.image_ids), 2)
        all_rows = load_celeba(self.images, csv_path, self.parts, max_samples=100)
        np.testing.assert_array_equal(all_rows.target, [0, 1, 0])

    def test_other_partitions_images_need_not_exist(self):
        for identifier in self.ids[3:]:
            (self.images / identifier).unlink()
        self.assertEqual(len(self.load().target), 3)

    def test_missing_label_rejected(self):
        self.attrs.write_text("6\nSmiling\n" + "\n".join(f"{identifier} 1" for identifier in self.ids[1:]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing attribute label"):
            self.load()

    def test_missing_file_rejected_even_when_subsampling(self):
        (self.images / self.ids[0]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing partition image"):
            self.load(max_samples=1)

    def test_duplicate_attributes_and_partitions_rejected(self):
        original = self.attrs.read_text(encoding="utf-8")
        self.attrs.write_text(original.replace("7\n", "8\n", 1) + f"{self.ids[0]} -1 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate image identifier"):
            self.load()
        self.attrs.write_text(original, encoding="utf-8")
        with self.parts.open("a", encoding="utf-8") as stream:
            stream.write(f"{self.ids[0]} 0\n")
        with self.assertRaisesRegex(ValueError, "Duplicate image identifier"):
            self.load()

    def test_invalid_labels_and_partition_codes_rejected(self):
        original = self.attrs.read_text(encoding="utf-8")
        self.attrs.write_text(original.replace(f"{self.ids[0]} -1 1", f"{self.ids[0]} 2 1"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid binary label"):
            self.load()
        self.attrs.write_text(original, encoding="utf-8")
        self.parts.write_text(f"{self.ids[0]} 3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid official partition"):
            self.load()

    def test_missing_attribute_and_bad_options_rejected(self):
        with self.assertRaisesRegex(ValueError, "Attribute.*missing"):
            self.load(attribute="Missing")
        for kwargs in ({"split": "all"}, {"max_samples": 0}, {"image_size": (0, 3)}, {"image_size": (3,)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.load(**kwargs)


if __name__ == "__main__":
    unittest.main()
