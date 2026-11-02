"""Exact mixed-numeric metadata identity, with native precision where needed."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from supervised_nmf.evaluation import evaluate_classifier
from supervised_nmf.validation import build_cv_folds, validate_metadata


_WIDE = pytest.mark.skipif(np.finfo(np.longdouble).nmant <= 52,
                         reason="This equal-value pair needs wider native precision")
_MIXED_CASES = [pytest.param("extended_equal", marks=_WIDE), "extended_distinct",
                "ordinary_distinct", "numpy_integer_distinct"]


def array_state(values):
    return values.shape, values.strides, values.dtype.str, values.flags.writeable, values.tobytes()


@pytest.fixture
def problem():
    rows = np.arange(12, dtype=float)
    y = np.tile([0, 1], 6)
    X = np.column_stack([3 * y + rows / 100, np.sin(rows), np.cos(rows)])
    first, second = np.arange(6), np.arange(6, 12)
    X.setflags(write=False)
    y.setflags(write=False)
    return X, y, [(first, second), (second, first)]


def mixed_pair(case):
    if case == "extended_equal":
        integer = 2**53 + 1
        return np.longdouble(str(integer)), integer, True
    if case == "extended_distinct":
        return np.longdouble("18446744073709551616"), 2**64 + 1, False
    if case == "ordinary_distinct":
        return np.float64(2**53), 2**53 + 1, False
    return np.float64(2**53), np.int64(2**53 + 1), False


@pytest.mark.parametrize("case", _MIXED_CASES)
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_numeric_uniqueness_uses_exact_values_in_both_orders(case, reverse):
    left, right, equal = mixed_pair(case)
    values = np.asarray([right, left] if reverse else [left, right], dtype=object)
    before = array_state(values)
    preserved = validate_metadata(values, 2, name="ids")
    assert [type(value) for value in preserved] == [type(value) for value in values]
    for actual, original in zip(preserved, values):
        if isinstance(original, np.floating):
            assert actual.as_integer_ratio() == original.as_integer_ratio()
        else:
            assert int(actual) == int(original)
    if equal:
        with pytest.raises(ValueError, match="unique"):
            validate_metadata(values, 2, name="ids", unique=True)
    else:
        assert len(validate_metadata(values, 2, name="ids", unique=True)) == 2
    assert array_state(values) == before


@pytest.mark.parametrize("case", _MIXED_CASES)
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_numeric_group_overlap_uses_exact_values_in_both_orders(problem, case, reverse):
    X, y, folds = problem
    left, right, equal = mixed_pair(case)
    if reverse:
        left, right = right, left
    groups = np.repeat(np.arange(-6, 0, dtype=object), 2)
    groups[:2] = left
    groups[6:8] = right
    before = array_state(groups)
    if equal:
        with pytest.raises(ValueError, match="groups must not overlap"):
            build_cv_folds(X, y, cv=2, groups=groups, folds=folds)
    else:
        result = build_cv_folds(X, y, cv=2, groups=groups, folds=folds)
        for actual, expected in zip(result, folds):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
    assert array_state(groups) == before
    evaluation_groups = np.repeat(np.arange(-6, 0, dtype=object), 2)
    training_groups = np.repeat(np.arange(-16, -10, dtype=object), 2)
    evaluation_groups[:2] = left
    training_groups[:2] = right
    model = DummyClassifier(strategy="prior").fit(X, y)
    options = dict(groups=evaluation_groups, train_groups=training_groups, train_y=y, threads=1)
    if equal:
        with pytest.raises(ValueError, match="groups must not overlap"):
            evaluate_classifier(model, X, y, **options)
    else:
        report = evaluate_classifier(model, X, y, **options)
        assert report["status"] == "valid"
        assert report["support_diagnostics"]["group_count"] == 6
        assert report["support_diagnostics"]["training_group_count"] == 6


def mixed_groups(case):
    if case == "extended_equal":
        values = []
        for offset in range(6):
            integer = 2**53 + 1 + 2 * offset
            values.extend([np.longdouble(str(integer)), integer])
        return np.asarray(values, dtype=object)
    first, second, _ = mixed_pair(case)
    base = 2**64 if case == "extended_distinct" else 2**53
    return np.asarray([value for identity in [first, second, *range(base + 2, base + 6)]
                       for value in (identity, identity)], dtype=object)


@pytest.mark.parametrize("case", _MIXED_CASES[:3])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_numeric_generated_groups_and_report_counts_match_exact_codes(problem, case, reverse):
    X, y, _ = problem
    groups = mixed_groups(case)
    # Subject membership and increasing mathematical order are independently
    # specified by construction, without using NumPy equality or unique().
    expected_codes = np.repeat(np.arange(6), 2)
    if reverse:
        groups = groups[::-1].copy()
        expected_codes = expected_codes[::-1].copy()
    groups.setflags(write=False)
    before = array_state(groups)
    expected = build_cv_folds(X, y, cv=2, seed=7, groups=expected_codes)
    actual = build_cv_folds(X, y, cv=2, seed=7, groups=groups)
    for actual_fold, expected_fold in zip(actual, expected):
        np.testing.assert_array_equal(actual_fold[0], expected_fold[0])
        np.testing.assert_array_equal(actual_fold[1], expected_fold[1])
    model = DummyClassifier(strategy="prior").fit(X, y)
    report = evaluate_classifier(model, X, y, groups=groups, train_y=y, threads=1)
    support = report["support_diagnostics"]
    assert support["group_count"] == 6
    assert support["per_class_group_count"] == [6, 6]
    assert array_state(groups) == before
