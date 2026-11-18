"""Layout, ownership and failure contracts for binary screening."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn import config_context, get_config
from threadpoolctl import threadpool_limits

from supervised_nmf import screening
from supervised_nmf.screening import BinaryScreening


def _data(rows, columns, dtype=np.float64):
    labels = np.arange(rows) % 2
    rng = np.random.default_rng(912)
    values = rng.uniform(0.5, 2.0, size=(rows, columns))
    values += labels[:, None] * 0.35
    return np.array(values, dtype=dtype, order="F"), labels


class ScreeningLayoutTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)

    def assert_same_fit(self, actual, expected):
        actual_state = {k: v for k, v in vars(actual).items() if k.endswith("_")}
        expected_state = {k: v for k, v in vars(expected).items() if k.endswith("_")}
        self.assertEqual(set(actual_state), set(expected_state))
        for name, value in actual_state.items():
            reference = expected_state[name]
            if isinstance(value, np.ndarray):
                self.assertEqual(value.shape, reference.shape, name)
                self.assertEqual(value.dtype, reference.dtype, name)
                self.assertEqual(value.tobytes(), reference.tobytes(), name)
            else:
                self.assertEqual(value, reference, name)

    def reference_fit(self, values, labels, **settings):
        # C input uses the original full-array statistics path. Class membership
        # and represented input values are identical to the F-layout candidate.
        return BinaryScreening(**settings).fit(np.array(values, order="C"), labels.copy())

    def test_large_f_layout_and_block_tails_match_full_statistics(self):
        for dtype in (np.float32, np.float64):
            for columns in (129, 255, 256, 257):
                with self.subTest(dtype=dtype, columns=columns):
                    values, labels = _data(1024, columns, dtype)
                    before, label_before = values.copy(), labels.copy()
                    expected = self.reference_fit(values, labels, score_threshold=0)
                    actual = BinaryScreening(score_threshold=0).fit(values, labels)
                    self.assert_same_fit(actual, expected)
                    np.testing.assert_array_equal(values, before)
                    np.testing.assert_array_equal(labels, label_before)
                    result = actual.transform(values[:7])
                    reference_result = expected.transform(values[:7])
                    np.testing.assert_array_equal(result, reference_result)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.flags.c_contiguous, reference_result.flags.c_contiguous)
                    self.assertEqual(result.flags.f_contiguous, reference_result.flags.f_contiguous)
                    self.assertEqual(result.flags.writeable, reference_result.flags.writeable)
                    self.assertFalse(np.shares_memory(result, values))

    def test_required_cast_layout_at_the_byte_budget(self):
        zeros_like = screening.np.zeros_like
        block_statistics = screening._screening_block_statistics
        for dtype in (np.float32, np.float64):
            for columns in (255, 256, 257):
                with self.subTest(dtype=dtype, columns=columns):
                    values, labels = _data(512, columns, dtype)
                    expected = self.reference_fit(values, labels, score_threshold=0)
                    normalization_inputs = []

                    def observe(array, *args, **kwargs):
                        if array.ndim == 2 and array.dtype == np.float64 and array.shape[0] == 512:
                            normalization_inputs.append(
                                (array.dtype, array.flags.c_contiguous, array.flags.f_contiguous)
                            )
                        return zeros_like(array, *args, **kwargs)

                    with patch.object(screening.np, "zeros_like", side_effect=observe), \
                            patch.object(screening, "_screening_block_statistics",
                                         wraps=block_statistics) as blocked:
                        actual = BinaryScreening(score_threshold=0).fit(values, labels)
                    self.assert_same_fit(actual, expected)
                    self.assertEqual(blocked.call_count, int(columns == 257))
                    self.assertTrue(normalization_inputs)
                    first_dtype, is_c, is_f = normalization_inputs[0]
                    self.assertEqual(first_dtype, np.dtype(np.float64))
                    self.assertEqual(is_c, dtype is np.float32 and columns <= 256)
                    self.assertNotEqual(is_c, is_f)

    def test_narrow_inputs_retain_full_statistics(self):
        zeros_like = screening.np.zeros_like
        block_statistics = screening._screening_block_statistics
        cases = ((512, 128, False), (2048, 128, False), (65536, 2, False),
                 (512, 129, True), (1024, 129, False))
        for rows, columns, expected_c in cases:
            with self.subTest(rows=rows, columns=columns):
                values, labels = _data(rows, columns, np.float32)
                expected = self.reference_fit(values, labels, score_threshold=0)
                normalization_inputs = []

                def observe(array, *args, **kwargs):
                    if array.ndim == 2 and array.dtype == np.float64 and array.shape[0] == rows:
                        normalization_inputs.append((array.flags.c_contiguous, array.flags.f_contiguous))
                    return zeros_like(array, *args, **kwargs)

                with patch.object(screening.np, "zeros_like", side_effect=observe), \
                        patch.object(screening, "_screening_block_statistics",
                                     wraps=block_statistics) as blocked:
                    actual = BinaryScreening(score_threshold=0).fit(values, labels)
                self.assert_same_fit(actual, expected)
                self.assertEqual(blocked.call_count, int(rows == 1024 and columns == 129))
                self.assertTrue(normalization_inputs)
                self.assertEqual(normalization_inputs[0], (expected_c, not expected_c))

    def test_layout_controls_and_read_only_inputs(self):
        values, labels = _data(1024, 257)
        strided = np.empty((2048, 257), order="F")
        strided[::2] = values
        read_only = values.copy(order="F")
        read_only.setflags(write=False)
        unaligned = np.ndarray(values.shape, dtype=np.float64, order="F",
                               buffer=bytearray(values.nbytes + 1), offset=1)
        unaligned[:] = values
        controls = {
            "C": np.array(values, order="C"),
            "strided": strided[::2],
            "read_only": read_only,
            "non_native": values.astype(values.dtype.newbyteorder("S"), order="F"),
            "unaligned": unaligned,
        }
        expected = self.reference_fit(values, labels, score_threshold=0)
        for name, supplied in controls.items():
            with self.subTest(layout=name):
                before = supplied.tobytes()
                actual = BinaryScreening(score_threshold=0).fit(supplied, labels)
                self.assert_same_fit(actual, expected)
                self.assertEqual(supplied.tobytes(), before)
                np.testing.assert_array_equal(actual.transform(supplied[:7]),
                                              expected.transform(values[:7]))

    def test_canonical_layout_requires_native_aligned_plain_arrays(self):
        class ArraySubclass(np.ndarray):
            pass

        values, _ = _data(8, 7)
        unaligned = np.ndarray(values.shape, dtype=np.float64, order="F",
                               buffer=bytearray(values.nbytes + 1), offset=1)
        self.assertTrue(screening._screening_canonical_f(values))
        rejected = (
            values.copy(order="C"), values[::2], values.view(ArraySubclass),
            values.astype(values.dtype.newbyteorder("S")), unaligned,
            values[:1], values[:, :1], values.ravel(),
        )
        for supplied in rejected:
            self.assertFalse(screening._screening_canonical_f(supplied))

    def test_error_policy_and_sklearn_configuration_are_preserved(self):
        block_statistics = screening._screening_block_statistics
        for rows in (64, 1024):
            for under, assume_finite in (("ignore", False), ("warn", False),
                                         ("raise", False), ("ignore", True)):
                with self.subTest(rows=rows, under=under, assume_finite=assume_finite):
                    values, labels = _data(rows, 257, np.float32)
                    with np.errstate(under=under), config_context(assume_finite=assume_finite):
                        before_error = np.geterr().copy()
                        before_config = get_config().copy()
                        expected = self.reference_fit(values, labels, score_threshold=0)
                        with patch.object(screening, "_screening_block_statistics",
                                          wraps=block_statistics) as blocked:
                            actual = BinaryScreening(score_threshold=0).fit(values, labels)
                        self.assert_same_fit(actual, expected)
                        self.assertEqual(blocked.call_count,
                                         int(rows == 1024 and under == "ignore" and not assume_finite))
                        self.assertEqual(np.geterr(), before_error)
                        self.assertEqual(get_config(), before_config)

    def test_constants_cutoffs_and_percentile_ties_match_full_statistics(self):
        values, labels = _data(1024, 257)
        values[:, 0] = 1
        values[:, 1] = labels
        values[:, 2] = 1 - labels
        values[:, 4] = values[:, 3]
        initial = self.reference_fit(values, labels, score_threshold=0)
        cutoff = abs(initial.t_scores_[3])
        settings = [dict(score_threshold=value) for value in
                    (np.nextafter(cutoff, 0), cutoff, np.nextafter(cutoff, np.inf))]
        settings += [dict(screening="percentile", percentile=value) for value in (1, 50, 100)]
        for options in settings:
            with self.subTest(settings=options):
                actual = BinaryScreening(**options).fit(values, labels)
                expected = self.reference_fit(values, labels, **options)
                self.assert_same_fit(actual, expected)
                self.assertFalse(actual.support_[0])
                self.assertEqual(actual.n_classwise_constant_features_, 2)
                self.assertEqual(actual.n_infinite_t_scores_, 2)

    def test_cached_statistics_and_public_outputs_own_their_arrays(self):
        values, labels = _data(1024, 257, np.float32)
        reference = BinaryScreening(score_threshold=0).fit(values, labels)
        cached = BinaryScreening(screening="percentile", percentile=25)._fit_from_statistics(reference)
        direct = self.reference_fit(values, labels, screening="percentile", percentile=25)
        self.assert_same_fit(cached, direct)
        for name, value in vars(cached).items():
            if name.endswith("_") and isinstance(value, np.ndarray):
                self.assertFalse(np.shares_memory(value, getattr(reference, name)), name)
        original_support = cached.support_.copy()
        support = cached.get_support()
        support[:] = ~support
        indices = cached.get_support(indices=True)
        indices[:] = -1
        transformed = cached.transform(values[:7])
        transformed[:] = 0
        np.testing.assert_array_equal(cached.support_, original_support)
        self.assert_same_fit(cached, direct)
        self.assertTrue(np.all(values > 0))

    def test_failure_in_a_later_block_clears_an_earlier_fit(self):
        values, labels = _data(1024, 257)
        model = BinaryScreening(score_threshold=0).fit(values, labels)
        zeros_like = screening.np.zeros_like
        calls = []

        def fail_second(array, *args, **kwargs):
            calls.append(array.shape)
            if len(calls) == 2:
                raise MemoryError("second normalization buffer")
            return zeros_like(array, *args, **kwargs)

        before = values.copy()
        with patch.object(screening.np, "zeros_like", side_effect=fail_second):
            with self.assertRaisesRegex(MemoryError, "second normalization buffer"):
                model.fit(values, labels)
        self.assertEqual(len(calls), 2)
        self.assertFalse(any(name.endswith("_") for name in vars(model)))
        np.testing.assert_array_equal(values, before)
        self.assert_same_fit(model.fit(values, labels),
                             self.reference_fit(values, labels, score_threshold=0))

    def test_numerical_errors_keep_the_original_policy_and_clear_state(self):
        values = np.array(np.tile([0., 1e-200, 1., 1.], (257, 256)).T, order="F")
        labels = np.tile([0, 0, 1, 1], 256)
        for under, message in (("ignore", "underflow"), ("raise", "exceeded float64 range")):
            with self.subTest(under=under), np.errstate(under=under):
                before_error = np.geterr().copy()
                models = []
                for layout in ("C", "F"):
                    supplied = np.array(values, order=layout)
                    model = BinaryScreening(score_threshold=0).fit(supplied + 1, labels)
                    with self.assertRaisesRegex(ValueError, message):
                        model.fit(supplied, labels)
                    self.assertFalse(any(name.endswith("_") for name in vars(model)))
                    models.append(model)
                self.assertEqual(models[0].get_params(), models[1].get_params())
                self.assertEqual(np.geterr(), before_error)

    def test_blocks_use_the_full_value_and_label_snapshot(self):
        values, labels = _data(1024, 257)
        expected = self.reference_fit(values, labels, score_threshold=0)
        original_values, original_labels = values.copy(), labels.copy()
        zeros_like = screening.np.zeros_like
        observations = []

        def mutate_after_snapshot(array, *args, **kwargs):
            if not observations:
                # Both classes and the full subtraction snapshot already exist.
                # The positive input stays nonconstant, while its statistics change.
                values[:] = values[::-1] ** 2
                labels[:] = 1 - labels
            observations.append(array.shape)
            return zeros_like(array, *args, **kwargs)

        with patch.object(screening.np, "zeros_like", side_effect=mutate_after_snapshot):
            actual = BinaryScreening(score_threshold=0).fit(values, labels)
        self.assertGreater(len(observations), 1)
        self.assert_same_fit(actual, expected)
        self.assertFalse(np.array_equal(values, original_values))
        self.assertFalse(np.array_equal(labels, original_labels))


if __name__ == "__main__":
    unittest.main()
