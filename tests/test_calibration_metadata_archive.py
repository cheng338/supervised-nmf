"""Cross-check binary class displays and refinement against completed archives."""

from copy import deepcopy
from itertools import permutations
import json
import math
from hashlib import sha256

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from supervised_nmf import BinaryScreening, NMFClassifier, evaluation
from supervised_nmf._calibration_parameters import _check_display, _complex_coordinate, _extended_display
from supervised_nmf._calibration_report import _matches_refinement_axis
from supervised_nmf._identity import configuration_identity


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    path = tmp_path_factory.mktemp("metadata-archive") / "cv.json"
    y = np.tile([0, 1], 8)
    evaluation.calibration_search(
        DecisionTreeClassifier(random_state=7), {"max_depth": [2, 1, 3, None, False, np.int64(2)]},
        y[:, None], y, cv=2, seed=3, refit=False, checkpoint_path=path,
    )
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 0
    assert report["refinement"]["parameters"]["max_depth"] == {
        "status": "interior_interval_requires_refinement", "selected": 2,
        "values": [1.0, 2.0, 3.0], "includes_none": True,
    }
    return report


def save(path, report):
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


def use_legacy_screening_evidence(report):
    report["identity"]["fold_evidence_version"] = 4
    del report["identity"]["screening_diagnostics_sha256"]
    report.pop("refinement_axis_inputs", None)
    report["fingerprint"] = evaluation._digest(report["identity"])


@pytest.mark.parametrize("labels", [[0, 0], [1, True], [-0.0, 0.0], [1, 0]])
def test_scored_numeric_labels_are_distinct_and_ordered(tmp_path, archive, labels):
    report = deepcopy(archive)
    report["class_labels"] = labels
    path = tmp_path / "duplicates.json"
    save(path, report)
    with pytest.raises(ValueError, match="Class labels disagree"):
        evaluation.load_calibration_report(path)
    with pytest.raises(ValueError, match="Class labels disagree"):
        evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


def test_unresolved_legacy_label_display_fails_with_or_without_the_original_digest(tmp_path, archive):
    report = deepcopy(archive)
    report["class_labels"] = [0, 0]
    report["identity"]["fold_evidence_version"] = 3
    del report["identity"]["class_label_evidence"]
    report["fingerprint"] = evaluation._digest(report["identity"])
    path = tmp_path / "legacy.json"
    save(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="(?i)class label"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


def test_missing_bound_label_evidence_is_fatal_with_a_matching_digest(tmp_path, archive):
    report = deepcopy(archive)
    del report["identity"]["class_label_evidence"]
    report["fingerprint"] = evaluation._digest(report["identity"])
    path = tmp_path / "missing-labels.json"
    save(path, report)
    with pytest.raises(ValueError):
        evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize(("kind", "labels"), [("i", ["red", "blue"]), ("u", [-1, 1]), ("i", [False, True]),
                                               ("b", [0, 1]), ("b", [True, False]),
                                               ("i", [-(2**63)-1, 1]), ("i", [0, 2**63]), ("u", [0, 2**64])])
def test_label_displays_agree_with_the_bound_integer_or_boolean_dtype(tmp_path, archive, kind, labels):
    report = deepcopy(archive)
    report["class_labels"] = labels
    report["identity"]["class_label_evidence"] = {"dtype_kind": kind, "labels": labels}
    report["fingerprint"] = evaluation._digest(report["identity"])
    path = tmp_path / "labels.json"
    save(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="Class labels disagree with their recorded numeric dtype"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize(("kind", "labels"), [
    ("f", ["red", "blue"]), ("f", [False, 1.0]), ("f", [1, 2]), ("U", [0, 1]),
    ("U", ["a", "a"]), ("U", ["z", "a"]), ("U", ["z", "a\U0001f600"]),
    ("U", ["a\u0000", "b"]),
    ("f", [{"__supervised_nmf_scalar__": "longdouble-v1", "significand": "3", "exponent": -4},
           {"__supervised_nmf_scalar__": "longdouble-v1", "significand": "1", "exponent": -4}]),
])
def test_native_float_and_text_displays_must_be_possible(tmp_path, archive, kind, labels):
    report = deepcopy(archive)
    report["class_labels"] = labels
    report["identity"]["class_label_evidence"] = {"dtype_kind": kind, "labels": labels}
    report["fingerprint"] = evaluation._digest(report["identity"])
    path = tmp_path / "labels.json"
    save(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="(?i)class labels"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
def test_native_float_labels_from_the_writer(tmp_path, dtype):
    y = np.tile(np.array([-2, 3], dtype=dtype), 8)
    path = tmp_path / "floating-labels.json"
    evaluation.calibration_search(DummyClassifier(), {}, np.ones((len(y), 1)), y, cv=2,
                                  seed=7, refit=False, checkpoint_path=path)
    assert evaluation.load_calibration_report(path) == json.loads(path.read_bytes())


def check_writer_class_labels(path, classes, labels):
    y = np.tile(classes[::-1], 4)
    left, right = np.arange(4), np.arange(4, 8)
    evaluation.calibration_search(DummyClassifier(), {}, np.ones((len(y), 1)), y, cv=2,
                                  folds=[(left, right), (right, left)], refit=False, checkpoint_path=path)
    report = json.loads(path.read_bytes())
    assert report["class_labels"] == labels
    assert report["identity"]["class_label_evidence"] == {"dtype_kind": classes.dtype.kind, "labels": labels}
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report


@pytest.mark.parametrize("classes", [
    (b"", b"\xff"), (b"\t\0\\\n\r", b"A"), (b"a'", b"a'\""), (b"a\0b", b"a\0c"),
    (bytes(range(256)), b"\xff"),
])
def test_fixed_bytes_class_labels_from_the_writer(tmp_path, classes):
    check_writer_class_labels(tmp_path / "bytes.json", np.array(classes, dtype="S"), [repr(v) for v in classes])


@pytest.mark.parametrize(("classes", "labels"), [
    (np.array([-(2**63), 2**63-1], dtype="i8"), [-(2**63), 2**63-1]),
    (np.array([0, 2**64-1], dtype="u8"), [0, 2**64-1]),
    (np.array([b"\0\xff", b"A\0"], dtype="V2"), [r"b'\x00\xff'", r"b'A\x00'"]),
    (np.array([(1,), (2,)], dtype=[("n", "i8")]), [[1], [2]]),
])
def test_integer_limits_and_void_class_labels_from_the_writer(tmp_path, classes, labels):
    check_writer_class_labels(tmp_path / "labels.json", classes, labels)


def test_structured_class_labels_can_have_identical_field_displays(tmp_path):
    class DisplayZero(np.str_):
        def item(self):
            return 0

    classes = np.array([(DisplayZero("a"),), (DisplayZero("b"),)], dtype=[("text", "O")])
    check_writer_class_labels(tmp_path / "structured.json", classes, [[0], [0]])


@pytest.mark.parametrize(("kind", "labels"), [
    ("S", [0, 1]), ("S", ["a", "b"]), ("S", ["b'a'", "b'a'"]), ("S", ["b'A'", r"b'\t'"]),
    ("S", ['b"a"', "b'b'"]), ("S", [r"b'\x61'", "b'b'"]), ("S", [r"b'a\x00'", "b'b'"]),
    ("S", [r"b'\q'", "b'b'"]), ("S", ["b'a' + b'b'", "b'c'"]),
    ("V", [False, [True, 9]]), ("V", [[1], [2, 3]]), ("V", [[], []]), ("V", ["b'a'", "b'bc'"]),
    ("V", ["b'aa'", r"b'\x00b'"]), ("V", [r"b'\x61'", "b'b'"]),
])
def test_bytes_and_void_class_labels_require_possible_displays(tmp_path, archive, kind, labels):
    report = deepcopy(archive)
    report["class_labels"] = labels
    report["identity"]["class_label_evidence"] = {"dtype_kind": kind, "labels": labels}
    report["fingerprint"] = evaluation._digest(report["identity"])
    path = tmp_path / "byte-labels.json"
    save(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="(?i)class labels"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("update", [
    {"status": "lower_boundary_requires_expansion_or_constraint"},
    {"status": "upper_boundary_requires_expansion_or_constraint"},
    {"status": "single_value_requires_exploration", "values": None},
    {"values": [0.0, 2.0, 3.0]}, {"values": [1.0, 2.0]}, {"values": [3.0, 2.0, 1.0]},
    {"values": [True, 2.0, 3.0]}, {"values": [1.0, 2.0, 2.0, 3.0]},
    {"includes_none": None}, {"includes_none": 1},
])
def test_numeric_refinement_agrees_with_the_whole_grid(tmp_path, archive, update):
    report = deepcopy(archive)
    entry = report["refinement"]["parameters"]["max_depth"]
    for name, value in update.items():
        if value is None:
            del entry[name]
        else:
            entry[name] = value
    path = tmp_path / "range.json"
    save(path, report)
    with pytest.raises(ValueError, match="Refinement"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize(("extra", "occurrences"), [(np.float16(np.nan), 2), (np.float64(np.nan), 2),
                                                   (float("nan"), 1)])
def test_nonfinite_coordinates_account_for_the_stored_nulls(extra, occurrences):
    axis = [(configuration_identity(v), evaluation._jsonable(v)) for v in (1., 2., extra, extra)]
    entry = {"selected": 1., "status": "lower_boundary_requires_expansion_or_constraint",
             "values": [1., 2.] + [None] * occurrences}
    assert _matches_refinement_axis(entry, axis, 0, False, True)
    for alternative in ([1., 2.], [1., 2., 42.]):
        assert not _matches_refinement_axis(dict(entry, values=alternative), axis, 0, False, True)


@pytest.mark.parametrize("source", ["python", "numpy", "wide", "opaque"])
def test_short_nan_grid_respects_the_declared_finite_coordinates(source):
    alternatives = {
        "python": (configuration_identity(float("nan")), None),
        "numpy": (configuration_identity(np.float32(np.nan)), None),
        "wide": ({"kind": "numpy_scalar", "dtype": {"str": "<f16"}, "content": "a" * 64}, None),
        "opaque": ({"kind": "estimator"}, "opaque"),
    }
    axis = [(configuration_identity(1.), 1.), alternatives[source], (configuration_identity(2.), 2.)]
    for values, status, allowed in (
        ([1., 2., None], "lower_boundary_requires_expansion_or_constraint", True),
        ([1., None, 2.], "lower_boundary_requires_expansion_or_constraint", True),
        ([None, 1., 2.], "interior_interval_requires_refinement", True),
        ([2., None, 1.], "upper_boundary_requires_expansion_or_constraint", False),
        ([None, 2., 1.], "upper_boundary_requires_expansion_or_constraint", False),
    ):
        entry = {"selected": 1., "status": status, "values": values}
        assert _matches_refinement_axis(entry, axis, 0, False, True, "3.11-3.12") is allowed


@pytest.mark.parametrize(("declared", "values", "allowed"), [
    # Inserting both finite values first leaves no slot between 1 and 2.
    ((1., 2., math.nan), [1., None, 2.], False),
    ((math.nan, 1., 2.), [2., None, 1.], False),
    # 10 occupies slot 2 and 4 occupies slot 4. A NaN in slot 3 survives sorting.
    ((10., 4., math.nan), [10., None, 4.], True),
    ((10., 4., math.nan), [4., None, 10.], False),
    ((10., 4., math.nan), [None, 4., 10.], True),
    # 0 occupies slot 0. The next probe for 32 goes to slot 2, leaving slot 1 free.
    ((0., 32., math.nan), [0., None, 32.], True),
    ((0., 32., math.nan), [None, 0., 32.], False),
    # The dyadic fractions start in adjacent slots 0 and 1 on both widths.
    ((0.5, 1.5, math.nan), [0.5, None, 1.5], False),
    ((0.5, math.nan, 1.5), [0.5, None, 1.5], True),
    # 2**-30 starts in slot 0 on 64-bit and slot 2 on 32-bit CPython.
    ((2**-30, 3., math.nan), [2**-30, None, 3.], True),
    ((2**-30, 3., math.nan), [3., None, 2**-30], False),
    # Both hashes are -2. The second point lands in slot 2 on 32-bit CPython
    # and slot 7 on 64-bit CPython after the unsigned perturb shifts.
    ((-1., -2., math.nan), [-2., None, -1.], True),
    ((-1., -2., math.nan), [-1., None, -2.], False),
    ((-2.5, -1.5, math.nan), [-2.5, None, -1.5], False),
    ((-2.5, -1.5, math.nan), [None, -2.5, -1.5], True),
])
def test_three_point_nan_order_depends_on_the_original_set_insertions(declared, values, allowed):
    axis = [(configuration_identity(v), evaluation._jsonable(v)) for v in declared]
    selected = min(v for v in declared if math.isfinite(v))
    status = ("lower_boundary_requires_expansion_or_constraint" if values[0] == selected
              else "upper_boundary_requires_expansion_or_constraint" if values[-1] == selected
              else "interior_interval_requires_refinement")
    entry = {"selected": selected, "status": status, "values": values}
    assert _matches_refinement_axis(entry, axis, 0, False, True, "3.11-3.12") is allowed
    assert _matches_refinement_axis(entry, axis, 0, False, True, None)


@pytest.mark.parametrize(("declared", "allows_middle"), [
    ((1., math.nan, 2., 1.), True),
    ((1., 2., math.nan, 1.), False),
    ((1., 2., 1., math.nan), False),
    ((False, 1., [], None, math.nan, "ignored", 2., True), True),
    ((1., math.nan, 2., math.nan), True),
    ((1., 2., math.nan, math.nan), False),
    ((1., np.float16(np.nan), 2., np.float32(1.)), True),
])
def test_three_effective_points_keep_their_first_insertion_order(declared, allows_middle):
    axis = [(configuration_identity(v), evaluation._jsonable(v)) for v in declared]
    middle = {"selected": 1., "status": "lower_boundary_requires_expansion_or_constraint",
              "values": [1., None, 2.]}
    reversed_axis = dict(middle, status="upper_boundary_requires_expansion_or_constraint", values=[2., None, 1.])
    assert _matches_refinement_axis(middle, axis, 0, False, True, "3.11-3.12") is allows_middle
    assert not _matches_refinement_axis(reversed_axis, axis, 0, False, True, "3.11-3.12")


def test_opaque_rows_can_change_only_the_insertions_that_follow_them():
    opaque = ({"kind": "estimator"}, "opaque")
    axis = [(configuration_identity(v), evaluation._jsonable(v)) for v in (1., 2., math.nan)]
    entry = {"selected": 1., "status": "lower_boundary_requires_expansion_or_constraint",
             "values": [1., None, 2.]}
    assert _matches_refinement_axis(entry, [opaque, *axis], 0, False, True, "3.11-3.12")
    assert not _matches_refinement_axis(entry, [*axis, opaque], 0, False, True, "3.11-3.12")


@pytest.mark.parametrize("repetitions", [1, 2, 3])
def test_short_axis_accounts_for_tied_wide_bytes_and_a_known_negative_infinity(repetitions):
    # Exhaust the concrete hidden values and original set iteration orders
    # independently, including the collapse with the known negative infinity.
    wide = ({"kind": "numpy_scalar", "dtype": {"str": "<f16"}, "content": "a" * 64}, None)
    raw = [1., 2., -math.inf]
    axis = [(configuration_identity(v), evaluation._jsonable(v)) for v in raw] + [wide] * repetitions
    expected = set()
    for hidden in (-math.inf, math.inf, math.nan):
        if math.isnan(hidden):
            extra = [math.nan] * repetitions
        else:
            extra = [hidden] if hidden not in raw else []
        for order in permutations(raw + extra):
            expected.add(tuple(v if math.isfinite(v) else None for v in sorted(order)))
    for count in range(1, repetitions + 2):
        for values in set(permutations([1., 2.] + [None] * count)):
            i = values.index(1.)
            status = ("lower_boundary_requires_expansion_or_constraint" if i == 0
                      else "upper_boundary_requires_expansion_or_constraint" if i == len(values) - 1
                      else "interior_interval_requires_refinement")
            entry = {"selected": 1., "status": status, "values": list(values)}
            assert _matches_refinement_axis(entry, axis, 0, False, True) == (values in expected)


def test_negative_infinity_sets_the_lower_end_of_a_numeric_axis():
    axis = [(configuration_identity(v), evaluation._jsonable(v)) for v in (1., np.float64(-np.inf))]
    entry = {"selected": 1., "status": "upper_boundary_requires_expansion_or_constraint", "values": [None, 1.]}
    assert _matches_refinement_axis(entry, axis, 0, False, True)
    assert not _matches_refinement_axis(dict(entry, values=[1., None],
        status="lower_boundary_requires_expansion_or_constraint"), axis, 0, False, True)


@pytest.mark.parametrize(("first", "decisive"), [(0., True), (1., False)])
def test_historical_nan_refinement_requires_a_determined_order(tmp_path, first, decisive):
    path = tmp_path / "nan-tolerance.json"
    y = np.tile([0, 1], 12)
    evaluation.calibration_search(LogisticRegression(random_state=3), {"tol": [first, np.nan]},
                                  y[:, None], y, cv=2, seed=3, threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 0
    # Version 5 kept the screening observation but not the sorted NaN axis.
    report["identity"]["fold_evidence_version"] = 5
    report["fingerprint"] = evaluation._digest(report["identity"])
    del report["refinement_axis_inputs"]
    save(path, report)
    if decisive and report["identity"]["python"].startswith(("3.11.", "3.12.")):
        assert evaluation.load_calibration_report(path) == report
    else:
        with pytest.raises(ValueError, match="ordering.*'tol'.*cannot be verified"):
            evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("scalar", [np.complex64(2+3j), np.complex128(2+5j)])
def test_complex_grid_alternative_retains_its_real_coordinate(scalar):
    axis = [(configuration_identity(v), evaluation._jsonable(v)) for v in (1., scalar)]
    entry = {"selected": 1., "status": "lower_boundary_requires_expansion_or_constraint", "values": [1., 2.]}
    assert _matches_refinement_axis(entry, axis, 0, False, True)
    assert not _matches_refinement_axis(dict(entry, values=[1., 42.]), axis, 0, False, True)
    assert not _matches_refinement_axis({"selected": 1., "status": "single_value_requires_exploration"},
                                        axis, 0, False, True)


@pytest.mark.parametrize(("scalar", "coordinate"), [
    (np.complex64(complex(2, math.nan)), 2.0),
    (np.complex128(complex(math.nan, 3)), None),
    (np.clongdouble(complex(2, math.nan)), 2.0),
    (np.clongdouble(complex(-math.inf, 3)), None),
])
def test_nonfinite_complex_alternatives_in_a_completed_search(tmp_path, scalar, coordinate):
    path = tmp_path / "complex.json"
    y = np.tile([0, 1], 12)
    x = np.column_stack([y * 2.0 + .1, 1 - y])
    with pytest.warns(np.exceptions.ComplexWarning):
        evaluation.calibration_search(LogisticRegression(random_state=7), {"C": [1.0, scalar]},
                                      x, y, cv=2, threads=1, seed=5, refit=False, checkpoint_path=path)
    genuine = evaluation.load_calibration_report(path)
    assert genuine["selected_candidate_index"] == 0
    values = genuine["refinement"]["parameters"]["C"]["values"]
    assert len(values) == 2 and 1.0 in values and coordinate in values
    for entry in ({"status": "lower_boundary_requires_expansion_or_constraint", "values": [1.0, 42.0]},
                  {"status": "single_value_requires_exploration"}):
        report = deepcopy(genuine)
        report["refinement"]["parameters"]["C"] = dict(entry, selected=1.0)
        save(path, report)
        with pytest.raises(ValueError, match="Refinement"):
            evaluation.load_calibration_report(path)


def test_complex_nan_payload_does_not_hide_a_nonrepresentable_component():
    raw = np.array([0x40000000, 0x7fc01234], dtype="<u4").view("<c8")[0]
    node = configuration_identity(raw)
    assert _complex_coordinate(node, "(2+nanj)") == 2
    for malformed in ("(2.0+nanj)", "(0.1+nanj)", "(nan+0.1j)"):
        with pytest.raises(ValueError):
            _check_display(node, malformed)
    seen = {}
    _check_display(node, "(2+nanj)", float_finiteness=seen)
    with pytest.raises(ValueError, match="same complex parameter"):
        _check_display(node, "(42+nanj)", float_finiteness=seen)


def test_wide_complex_validates_both_tags_and_retains_a_null_real_part():
    two = {"__supervised_nmf_scalar__": "longdouble-v1", "significand": "1", "exponent": 1}
    tag = {"__supervised_nmf_scalar__": "clongdouble-v1", "real": None, "imag": two}
    node = {"kind": "numpy_scalar", "dtype": {"str": "<c32", "format": ["G"]}, "content": "a" * 64}
    axis = [(configuration_identity(1.), 1.), (node, tag)]
    entry = {"selected": 1., "status": "lower_boundary_requires_expansion_or_constraint"}
    assert _matches_refinement_axis(dict(entry, values=[1., None]), axis, 0, False, True)
    assert not _matches_refinement_axis(dict(entry, values=[1., 42.]), axis, 0, False, True)
    assert not _matches_refinement_axis(dict(entry, status="single_value_requires_exploration"), axis, 0, False, True)
    with pytest.raises(ValueError, match="significand"):
        _check_display(node, dict(tag, imag=dict(two, significand="2")))


@pytest.mark.parametrize("real_bits", [0xfff0000000000000, 0x7ff0000000000000, 0x7ff800000000abcd])
def test_binary64_complex_tag_can_resolve_null_real_with_a_finite_imaginary_part(real_bits):
    raw = np.array([real_bits, 0x4008000000000000], dtype="<u8").tobytes()
    node = {"kind": "numpy_scalar", "dtype": {"str": "<c16", "format": ["G"]}, "content": sha256(raw).hexdigest()}
    three = {"__supervised_nmf_scalar__": "longdouble-v1", "significand": "3", "exponent": 0}
    tag = {"__supervised_nmf_scalar__": "clongdouble-v1", "real": None, "imag": three}
    coordinate = _complex_coordinate(node, tag)
    assert coordinate.hex() == np.frombuffer(raw, dtype="<f8")[0].item().hex()
    with pytest.raises(ValueError, match="declared bytes"):
        _complex_coordinate(node, dict(tag, real=three))


def test_finite_tagged_alternative_supplies_an_exact_refinement_coordinate(tmp_path):
    path = tmp_path / "mixed.json"
    y = np.tile([0, 1], 12)
    x = np.column_stack([y * 2.0 + .1, 1 - y])
    evaluation.calibration_search(LogisticRegression(random_state=7), {"C": [1.0, np.longdouble(2)]},
                                  x, y, cv=2, threads=1, seed=5, refit=False, checkpoint_path=path)
    genuine = evaluation.load_calibration_report(path)
    assert genuine["selected_candidate_index"] == 0
    assert genuine["candidates"][1]["parameters"]["C"]["significand"] == "1"
    assert genuine["refinement"]["parameters"]["C"]["values"] == [1.0, 2.0]
    for changed in ([1.0, 42.0], None):
        report = deepcopy(genuine)
        entry = report["refinement"]["parameters"]["C"]
        if changed is None:
            del entry["values"]
            entry["status"] = "single_value_requires_exploration"
        else:
            entry["values"] = changed
        save(path, report)
        with pytest.raises(ValueError, match="Refinement"):
            evaluation.load_calibration_report(path)


@pytest.mark.parametrize(("significand", "exponent", "expected"), [
    ("-0", 0, "-0x0.0p+0"),
    (str(2**53 + 1), -53, "0x1.0000000000000p+0"),
    (str(2**53 + 3), -53, "0x1.0000000000002p+0"),
    ("-1", -1075, "-0x0.0p+0"),
    (str(2**54 + 1), -1129, "0x0.0000000000001p-1022"),
    (str(2**55 - 3), 969, "0x1.fffffffffffffp+1023"),
    (str(2**54 - 1), 970, "inf"),
    ("-1", 16383, "-inf"),
])
def test_extended_coordinate_is_rounded_once_to_binary64(significand, exponent, expected):
    tag = {"__supervised_nmf_scalar__": "longdouble-v1", "significand": significand, "exponent": exponent}
    assert _extended_display(tag, binary64=False).hex() == expected


@pytest.mark.parametrize(("significand", "exponent"), [("1", 10**100), ("1", -(10**100)), ("1" * 634, 0), ("2", 0)])
def test_legacy_extended_coordinate_is_bounded_and_canonical(significand, exponent):
    axis = [({"kind": "numpy_scalar", "dtype": {"str": "<f16"}},
             {"__supervised_nmf_scalar__": "longdouble-v1", "significand": significand, "exponent": exponent})]
    with pytest.raises(ValueError, match="[Ee]xtended-float"):
        _matches_refinement_axis({"selected": 1., "status": "single_value_requires_exploration"}, axis, 0, False, True)


def test_matching_wide_nonfinite_bytes_contribute_one_infinity_or_separate_nans():
    node = {"kind": "numpy_scalar", "dtype": {"str": "<f16"}, "content": "a" * 64}
    axis = [(configuration_identity(1.), 1.), (node, None), (node, None), (node, None)]
    entry = {"selected": 1., "status": "lower_boundary_requires_expansion_or_constraint"}
    for values in ([1., None], [1., None, None, None]):
        assert _matches_refinement_axis(dict(entry, values=values), axis, 0, False, True)
    for values in ([1., None, None], [1., 2.]):
        assert not _matches_refinement_axis(dict(entry, values=values), axis, 0, False, True)
    assert not _matches_refinement_axis(dict(entry, status="single_value_requires_exploration"), axis, 0, False, True)


@pytest.mark.parametrize("other, displayed", [("\U0001f600", ["\U0001f600", "\U0001f600"]),
                                               ("\ue000", ["\U0001f600", "\ue000"])])
def test_unicode_label_order_can_change_during_json_roundtrip(tmp_path, other, displayed):
    path = tmp_path / "unicode.json"
    y = np.tile([other, "\ud83d\ude00"], 4)
    evaluation.calibration_search(DummyClassifier(), {}, np.ones((len(y), 1)), y,
                                  cv=2, seed=2, refit=False, checkpoint_path=path)
    report = json.loads(path.read_bytes())
    assert report["class_labels"] == displayed
    assert evaluation.load_calibration_report(path) == report


def test_unscored_object_labels_can_share_an_integer_display(tmp_path):
    # Object arrays preserve NumPy scalar subclasses. Label comparisons see
    # the two distinct stored integers, while the writer calls item() for JSON.
    class DisplayAsOne(np.int64):
        def item(self):
            return 1

    path = tmp_path / "object-labels.json"
    y = np.array([DisplayAsOne(1), DisplayAsOne(2)] * 4, dtype=object)
    left, right = np.arange(4), np.arange(4, 8)
    evaluation.calibration_search(DummyClassifier(), {}, np.ones((len(y), 1)), y,
                                  cv=2, folds=[(left, right), (right, left)], refit=False, checkpoint_path=path)
    report = json.loads(path.read_bytes())
    assert report["class_labels"] == [1, 1] and report["selected_candidate_index"] is None
    assert evaluation.load_calibration_report(path) == report


def test_scored_object_labels_can_share_a_bound_numeric_display(tmp_path):
    class DisplayCode(np.str_):
        def item(self):
            return 0

    path = tmp_path / "object-strings.json"
    y = np.array([DisplayCode("a"), DisplayCode("b")] * 8, dtype=object)
    evaluation.calibration_search(DummyClassifier(), {}, np.ones((len(y), 1)), y,
                                  cv=2, seed=7, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 0
    assert report["class_labels"] == [0, 0]
    assert report["identity"]["class_label_evidence"] == {"dtype_kind": "O", "labels": [0, 0]}


@pytest.mark.parametrize("checkpoint", [False, True])
def test_label_evidence_uses_serialized_values_without_copying_user_objects(tmp_path, checkpoint):
    class Label(str):
        def __deepcopy__(self, memo):
            raise AssertionError("Label instances cannot be deep-copied")

    y = np.array([Label("a"), Label("b")] * 8, dtype=object)
    path = tmp_path / "strings.json" if checkpoint else None
    result = evaluation.calibration_search(DummyClassifier(), {}, np.ones((len(y), 1)), y, cv=2, seed=7,
                                           refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path) if checkpoint else result.report
    assert report["selected_candidate_index"] == 0
    bound = report["identity"]["class_label_evidence"]["labels"]
    assert bound == ["a", "b"] and all(type(label) is str for label in bound)


def test_unrelated_parameter_cannot_suppress_boundary_advice(tmp_path, archive):
    report = deepcopy(archive)
    use_legacy_screening_evidence(report)
    index = report["selected_candidate_index"]
    report["screening_diagnostics"]["candidates"][index]["screens"] = [{
        "path": "(root)", "mode": "absolute", "active_parameter": "score_threshold",
        "active_value": 2., "inactive_grid_parameters": ["max_depth"],
    }]
    report["screening_diagnostics"]["inactive_parameters"] = [
        {"parameter": "max_depth", "mode": "absolute", "candidate_indices": [index]}]
    report["refinement"]["parameters"]["max_depth"].update(
        status="inactive_for_selected_screening_mode", screening_mode="absolute")
    path = tmp_path / "false-screen.json"
    save(path, report)
    with pytest.raises(ValueError, match="Inactive refinement"):
        evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


def test_refinement_uses_numpy_text_keys_including_legacy_bytes(tmp_path):
    path = tmp_path / "numpy-key.json"
    y = np.tile([0, 1], 8)
    evaluation.calibration_search(DecisionTreeClassifier(random_state=7), {np.str_("max_depth"): [2, 1, 3]},
                                  y[:, None], y, cv=2, seed=3, threads=1, refit=False, checkpoint_path=path)
    genuine = evaluation.load_calibration_report(path)
    for legacy in (False, True):
        report = deepcopy(genuine)
        config = {k["value"]: v for k, v in report["identity"]["structural_configuration"]["items"]}
        if legacy:
            for declaration in config["candidates"]["items"]:
                key = declaration["items"][0][0]
                del key["key_text"], key["display"]
            report["fingerprint"] = evaluation._digest(report["identity"])
        save(path, report)
        assert evaluation.load_calibration_report(path) == report
        report["refinement"]["parameters"]["max_depth"].update(
            status="lower_boundary_requires_expansion_or_constraint", values=[2., 3., 4.])
        save(path, report)
        with pytest.raises(ValueError, match="Refinement"):
            evaluation.load_calibration_report(path)


class _PercentileMode(str, BaseEstimator):
    """A cloneable string whose structural declaration describes an estimator."""

    def __new__(cls, value="percentile"):
        return str.__new__(cls, value)

    def __init__(self):
        pass


class _ScreenKey(str, BaseEstimator):
    def __new__(cls, value="cut__score_threshold"):
        return str.__new__(cls, value)

    def __init__(self, value="cut__score_threshold"):
        self.value = value

    def __str__(self):
        return "display-key"


class _SharesModeDisplay(_ScreenKey):
    def __str__(self):
        return "cut__screening"


class _CasefoldMode(_PercentileMode):
    def __new__(cls, value="PERCENTILE"):
        return str.__new__(cls, value)

    def __eq__(self, other):
        return isinstance(other, str) and self.casefold() == other.casefold()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.casefold())


def test_recorded_screening_mode_can_compare_differently_from_its_display(tmp_path):
    y = np.tile([0, 1], 8)
    model = Pipeline([("cut", BinaryScreening(screening=_CasefoldMode(), percentile=100)),
                      ("head", DummyClassifier(random_state=3))])
    path = tmp_path / "mode.json"
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(model, {"cut__score_threshold": [1, 2]}, y[:, None], y,
                                      cv=2, seed=3, threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["refinement"]["parameters"]["cut__score_threshold"]["screening_mode"] == "PERCENTILE"
    use_legacy_screening_evidence(report)
    save(path, report)
    with pytest.raises(ValueError, match="cannot be verified"):
        evaluation.load_calibration_report(path)


def test_group_keeps_the_first_equal_mode_while_winner_keeps_its_own_spelling(tmp_path):
    y = np.tile([0, 1], 8)
    model = Pipeline([("cut", BinaryScreening(percentile=100)), ("head", DummyClassifier(random_state=3))])
    path = tmp_path / "mixed-modes.json"
    name = "cut__score_threshold"
    grid = [{"cut__screening": [mode], "head__strategy": [strategy], name: [1]}
            for mode, strategy in [(_CasefoldMode(), "invalid"), ("percentile", "prior")]]
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(model, grid, y[:, None], y, cv=2, seed=3,
                                      threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 1
    assert report["screening_diagnostics"]["inactive_parameters"] == [
        dict(parameter=name, mode="PERCENTILE", candidate_indices=[0, 1])]
    assert report["refinement"]["parameters"][name]["screening_mode"] == "percentile"
    use_legacy_screening_evidence(report)
    save(path, report)
    assert evaluation.load_calibration_report(path) == report


def test_bound_screening_observation_uses_raw_key_before_its_public_display(tmp_path):
    y = np.tile([0, 1], 8)
    model = Pipeline([("cut", BinaryScreening(screening="percentile", percentile=100)), ("head", DummyClassifier())])
    path = tmp_path / "raw-screen-key.json"
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(model, {_ScreenKey(): [1, 2]}, y[:, None], y, cv=2,
                                      seed=3, threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["screening_diagnostics"]["inactive_parameters"][0]["parameter"] == "cut__score_threshold"
    assert report["refinement"]["parameters"]["display-key"]["status"] == "single_value_requires_exploration"
    changed = deepcopy(report)
    changed["refinement_axis_inputs"]["display-key"]["coordinates"] = ["nan"]
    save(path, changed)
    with pytest.raises(ValueError, match="Refinement status or values"):
        evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())
    use_legacy_screening_evidence(report)
    save(path, report)
    with pytest.raises(ValueError, match="axis size cannot be verified"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("with_known", [False, True])
def test_legacy_colliding_screen_keys_use_distinct_raw_declarations_across_screens(tmp_path, with_known):
    y = np.tile([0, 1], 12)
    path = tmp_path / "colliding-cutoffs.json"
    model = Pipeline([("features", BinaryScreening(screening="percentile", percentile=100)),
                      ("cut", NMFClassifier(screening="none", n_components=1, random_state=3))])
    grid = {_ScreenKey("cut__percentile"): [25], _ScreenKey("cut__score_threshold"): [2]}
    if with_known:
        grid["cut__n_neighbors"] = [1]
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(model, grid, np.column_stack([1. + y, 2. - y]), y, cv=2, seed=3,
                                      threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_parameters"] == ({"display-key": 2, "cut__n_neighbors": 1} if with_known else {"display-key": 2})
    use_legacy_screening_evidence(report)
    save(path, report)
    assert evaluation.load_calibration_report(path) == report

    diagnostics = report["screening_diagnostics"]
    screens = {s["path"]: s for s in diagnostics["candidates"][0]["screens"]}
    assert screens["cut"]["inactive_grid_parameters"] == ["cut__percentile", "cut__score_threshold"]
    extra = "features__score_threshold"
    screens["features"]["inactive_grid_parameters"] = [extra]
    diagnostics["inactive_parameters"].append(dict(parameter=extra, mode="percentile", candidate_indices=[0]))
    save(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="more raw parameter keys"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("variant", ["direct", "colliding", "active", "known-mode-collision", "replacement"])
def test_legacy_raw_override_can_control_screening_before_display(tmp_path, variant):
    y = np.tile([0, 1], 12)
    path = tmp_path / "opaque-mode.json"
    cutoff = "cut__score_threshold"
    base = "percentile" if variant == "active" else "absolute"
    mode = "absolute" if variant == "active" else "percentile"
    model = Pipeline([("cut", BinaryScreening(screening=base, percentile=100)), ("head", DummyClassifier(random_state=3))])
    grid = {_ScreenKey("cut__screening"): [mode], (_ScreenKey(cutoff) if variant == "colliding" else cutoff): [1]}
    if variant == "replacement":
        grid = {_ScreenKey("cut"): [BinaryScreening(screening=mode, percentile=100)], cutoff: [1]}
    elif variant == "known-mode-collision":
        grid = {"cut__screening": [mode], _SharesModeDisplay("head__strategy"): ["most_frequent"], cutoff: [1]}
    evaluation.calibration_search(model, grid, np.column_stack([1. + y, 2. - y]), y, cv=2, seed=3,
                                  threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 0
    if variant == "known-mode-collision":
        assert report["selected_parameters"]["cut__screening"] == "most_frequent"
    use_legacy_screening_evidence(report)
    save(path, report)
    if variant in ("known-mode-collision", "colliding"):
        # The raw mode is known, or the sole public parameter is categorical.
        assert evaluation.load_calibration_report(path) == report
    else:
        with pytest.raises(ValueError, match="cannot be verified"):
            evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("variant", ["shared-step", "renamed-descendants", "shared-collection", "two-direct",
                                     "one-direct", "one-hidden-cutoff"])
def test_legacy_opaque_overrides_account_for_joint_screen_modes(tmp_path, variant):
    def features(modes):
        return Pipeline([(name, BinaryScreening(screening=mode, percentile=100))
                         for name, mode in zip(("a", "b"), modes)])

    y = np.tile([0, 1], 12)
    path = tmp_path / "joint.json"
    first, second = ("features__" + name + "__score_threshold" for name in ("a", "b"))
    start_modes = ("absolute", "percentile" if variant == "one-hidden-cutoff" else "absolute")
    if variant == "renamed-descendants":
        start_modes = ("percentile", "percentile")
    model = Pipeline([("features", features(start_modes)), ("head", DummyClassifier(random_state=3))])
    grid = {first: [0], second: [0]}
    if variant in ("shared-step", "renamed-descendants"):
        grid[_ScreenKey("features")] = [features(("percentile", "percentile"))]
    elif variant == "shared-collection":
        grid[_ScreenKey("features__steps")] = [features(("percentile", "percentile")).steps]
    else:
        grid[_ScreenKey("features__a__screening")] = ["percentile"]
        if variant == "two-direct":
            grid[_ScreenKey("features__b__screening")] = ["percentile"]
        elif variant == "one-hidden-cutoff":
            del grid[second]
    evaluation.calibration_search(model, grid, np.column_stack([1. + y, 2. - y]), y, cv=2, seed=3,
                                  threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 0
    use_legacy_screening_evidence(report)
    save(path, report)
    with pytest.raises(ValueError, match="cannot be verified"):
        evaluation.load_calibration_report(path)
    if variant in ("one-direct", "one-hidden-cutoff"):
        diagnostics = report["screening_diagnostics"]
        row = next(s for s in diagnostics["candidates"][0]["screens"] if s["path"] == "features__b")
        row.update(mode="percentile", active_parameter="features__b__percentile", active_value=100,
                   inactive_grid_parameters=[second])
        diagnostics["inactive_parameters"].append(dict(mode="percentile", parameter=second, candidate_indices=[0]))
        if variant == "one-direct":
            report["refinement"]["parameters"][second].update(
                status="inactive_for_selected_screening_mode", screening_mode="percentile")
        save(path, report)
        with pytest.raises(ValueError, match="Inactive refinement"):
            evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("mode", [np.str_("percentile"), _PercentileMode()])
def test_nested_screening_advice_uses_replacement_and_candidate_mode(tmp_path, mode):
    path = tmp_path / "nested.json"
    y = np.tile([0, 1], 8)
    base = Pipeline([("features", Pipeline([("cut", BinaryScreening())])), ("classify", DummyClassifier())])
    grid = {"features__cut": [BinaryScreening(screening="absolute")],
            "features__cut__screening": [mode],
            "features__cut__percentile": [100], "features__cut__score_threshold": [1, 2]}
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(base, grid, y[:, None], y, cv=2, seed=3, threads=1,
                                      refit=False, checkpoint_path=path)
    genuine = evaluation.load_calibration_report(path)
    index = genuine["selected_candidate_index"]
    name = "features__cut__score_threshold"
    assert genuine["refinement"]["parameters"][name]["screening_mode"] == "percentile"
    changes = ["path", "mode", "removed", "empty-with-summary"]
    if type(mode) is np.str_:
        changes.append("erased")
    for change in changes:
        report = deepcopy(genuine)
        use_legacy_screening_evidence(report)
        screen, = report["screening_diagnostics"]["candidates"][index]["screens"]
        entry = report["refinement"]["parameters"][name]
        if change == "path":
            screen["path"] = "cut"
        elif change == "mode":
            entry["screening_mode"] = screen["mode"] = "none"
            screen["active_parameter"] = None
        else:
            screen["inactive_grid_parameters"] = []
            del entry["screening_mode"]
            entry["status"] = "lower_boundary_requires_expansion_or_constraint"
            if change in ("erased", "empty-with-summary"):
                report["screening_diagnostics"]["candidates"][index]["screens"] = []
            if change == "erased":
                for group in report["screening_diagnostics"]["inactive_parameters"]:
                    if index in group["candidate_indices"]:
                        group["candidate_indices"].remove(index)
        save(path, report)
        with pytest.raises(ValueError, match="[Ii]nactive refinement"):
            evaluation.load_calibration_report(path)


@pytest.mark.parametrize("composition", ["union", "columns"])
def test_sklearn_selector_cannot_claim_package_screening(tmp_path, composition):
    step = SelectPercentile()
    features = (FeatureUnion([("select", step)]) if composition == "union"
                else ColumnTransformer([("select", step, [0, 1])]))
    model = Pipeline([("features", features), ("model", DummyClassifier())])
    name = "features__select__percentile"
    y = np.tile([0, 1], 8)
    x = np.random.RandomState(3).normal(size=(len(y), 2))
    path = tmp_path / "selector.json"
    evaluation.calibration_search(model, {name: [50, 100]}, x, y, cv=2, seed=3,
                                  threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    winner = report["selected_candidate_index"]
    assert winner == 0
    use_legacy_screening_evidence(report)
    report["screening_diagnostics"]["candidates"][winner]["screens"] = [dict(
        path="features__select", mode="absolute", active_parameter="features__select__score_threshold",
        active_value=0, inactive_grid_parameters=[name])]
    report["screening_diagnostics"]["inactive_parameters"] = [dict(
        parameter=name, mode="absolute", candidate_indices=[winner])]
    report["refinement"]["parameters"][name].update(status="inactive_for_selected_screening_mode", screening_mode="absolute")
    save(path, report)
    with pytest.raises(ValueError, match="Inactive refinement"):
        evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


def test_column_transformer_replaces_collection_and_updates_an_old_child(tmp_path):
    features = ColumnTransformer([("cut", BinaryScreening(screening="absolute"), [0, 1])])
    model = Pipeline([("features", features), ("classify", DummyClassifier())])
    grid = {"features__transformers": [[("cut", BinaryScreening(screening="percentile", percentile=100), [0, 1])]],
            "features__cut__screening": ["absolute"], "features__cut__score_threshold": [0, 1]}
    path = tmp_path / "replacement.json"
    y = np.tile([0, 1], 8)
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(model, grid, np.column_stack([y, 1-y]), y, cv=2,
                                      seed=3, threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["refinement"]["parameters"]["features__cut__score_threshold"]["screening_mode"] == "percentile"


def test_bound_screening_covers_the_entire_observation_and_resume(tmp_path):
    y = np.tile([0, 1], 8)
    path = tmp_path / "scale-screen.json"
    model = Pipeline([("scale", StandardScaler(with_mean=False)),
                      ("cut", BinaryScreening(screening="percentile", percentile=100)),
                      ("model", DummyClassifier(random_state=3))])
    name = "cut__score_threshold"

    def search():
        with pytest.warns(UserWarning, match="Inactive screening grid"):
            return evaluation.calibration_search(model, {name: [1, 2]}, y[:, None] + 1, y, cv=2, seed=3,
                                                 threads=1, refit=False, checkpoint_path=path)

    search()
    genuine = evaluation.load_calibration_report(path)
    evidence = genuine["identity"]["screening_diagnostics_sha256"]
    assert genuine["identity"]["fold_evidence_version"] == 6
    assert evidence == evaluation._digest(genuine["screening_diagnostics"])
    assert genuine["selected_candidate_index"] == 0
    assert search().report["screening_diagnostics"] == genuine["screening_diagnostics"]
    for change in ("erased", "reversed", "loser"):
        report = deepcopy(genuine)
        diagnostics = report["screening_diagnostics"]
        if change == "erased":
            for row in diagnostics["candidates"]:
                row["screens"] = []
            diagnostics["inactive_parameters"] = []
            report["refinement"]["parameters"][name].update(status="lower_boundary_requires_expansion_or_constraint")
            del report["refinement"]["parameters"][name]["screening_mode"]
        elif change == "reversed":
            diagnostics["candidates"].reverse()
        else:
            diagnostics["candidates"][1]["screens"][0]["active_value"] = 50
        save(path, report)
        original_bytes = path.read_bytes()
        with pytest.raises(ValueError, match="Screening diagnostics"):
            evaluation.load_calibration_report(path, expected_sha256=sha256(original_bytes).hexdigest())
        with pytest.raises(ValueError, match="Screening diagnostics"):
            search()
        assert path.read_bytes() == original_bytes
    for change in ("advice", "group"):
        report = deepcopy(genuine)
        if change == "advice":
            report["refinement"]["parameters"][name].update(status="lower_boundary_requires_expansion_or_constraint")
            del report["refinement"]["parameters"][name]["screening_mode"]
        else:
            report["screening_diagnostics"]["inactive_parameters"][0]["candidate_indices"] = [0]
            report["identity"]["screening_diagnostics_sha256"] = evaluation._digest(report["screening_diagnostics"])
            report["fingerprint"] = evaluation._digest(report["identity"])
        save(path, report)
        with pytest.raises(ValueError, match="Refinement|Grouped screening"):
            evaluation.load_calibration_report(path)


@pytest.mark.parametrize("eligible", [False, True])
def test_screening_evidence_is_required_without_refinement_parameters(tmp_path, eligible):
    y = np.tile([0, 1], 8)
    path = tmp_path / "no-refinement.json"
    evaluation.calibration_search(DecisionTreeClassifier(random_state=3), {} if eligible else {"max_depth": [0]},
                                  y[:, None], y, cv=2, seed=3, threads=1, refit=False, checkpoint_path=path)
    genuine = evaluation.load_calibration_report(path)
    assert genuine["selected_candidate_index"] == (0 if eligible else None)
    assert genuine["refinement"]["parameters"] == {}
    for change in ("missing", "incorrect", "row"):
        report = deepcopy(genuine)
        identity = report["identity"]
        if change == "row":
            report["screening_diagnostics"]["candidates"].clear()
        elif change == "incorrect":
            identity["screening_diagnostics_sha256"] = "0" * 64
        else:
            del identity["screening_diagnostics_sha256"]
        report["fingerprint"] = evaluation._digest(identity)
        save(path, report)
        with pytest.raises(ValueError, match="Screening diagnostics"):
            evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("version", [5, 6])
def test_bound_winning_screen_cannot_contradict_plain_grid(tmp_path, version):
    y = np.tile([0, 1], 8)
    model = Pipeline([("cut", BinaryScreening(screening="percentile", percentile=100)),
                      ("head", DummyClassifier(random_state=3))])
    name = "cut__score_threshold"
    path = tmp_path / "observed.json"
    # The unknown parameter makes set_params fail before the losing screen can be inspected.
    grid = [{name: [1, 2]}, {name: [1], "head__unknown": [1]}]
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(model, grid, y[:, None], y, cv=2, threads=1, seed=3,
                                      refit=False, checkpoint_path=path)
    original = evaluation.load_calibration_report(path)
    assert original["selected_candidate_index"] == 0
    assert original["screening_diagnostics"]["candidates"][2]["screens"] == []
    if version == 5:
        del original["refinement_axis_inputs"]
        original["identity"]["fold_evidence_version"] = version
        original["fingerprint"] = evaluation._digest(original["identity"])
        save(path, original)
        assert evaluation.load_calibration_report(path) == original
    for change in ("active", "omitted", "erased"):
        report = deepcopy(original)
        diagnostics = report["screening_diagnostics"]
        row = diagnostics["candidates"][0]
        if change == "active":
            row["screens"][0]["active_parameter"] = name
        else:
            if change == "omitted":
                row["screens"][0]["inactive_grid_parameters"] = []
            else:
                row["screens"] = []
            diagnostics["inactive_parameters"][0]["candidate_indices"].remove(0)
            entry = report["refinement"]["parameters"][name]
            entry["status"] = "lower_boundary_requires_expansion_or_constraint"
            del entry["screening_mode"]
        report["identity"]["screening_diagnostics_sha256"] = evaluation._digest(diagnostics)
        report["fingerprint"] = evaluation._digest(report["identity"])
        save(path, report)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            with pytest.raises(ValueError, match="Inactive refinement"):
                evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("version", [5, 6])
@pytest.mark.parametrize("scaler, recorded_sklearn", [
    pytest.param(StandardScaler(with_mean=False), None, id="standard"),
    pytest.param(MinMaxScaler(), None, id="minmax"),
    pytest.param(MinMaxScaler(), "1.9.1", id="minmax-1.9.1"),
    pytest.param(RobustScaler(with_centering=False), "1.9.1", id="robust-1.9.1"),
])
def test_bound_screening_with_stock_scaler_must_match_winning_path(tmp_path, version, scaler, recorded_sklearn):
    y = np.tile([0, 1], 8)
    path = tmp_path / "scaled-screen.json"
    name = "cut__score_threshold"
    model = Pipeline([("scale", scaler),
                      ("cut", BinaryScreening(screening="percentile", percentile=100)),
                      ("head", DummyClassifier(random_state=3))])
    grid = [{name: [1, 2]}, {name: [1], "head__unknown": [1]}]
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(model, grid, y[:, None] + 1, y, cv=2, seed=3, threads=1,
                                      refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    if version == 5:
        del report["refinement_axis_inputs"]
        report["identity"]["fold_evidence_version"] = version
    if recorded_sklearn is not None:
        # Both audited sklearn releases use BaseEstimator parameter lookup for this scaler.
        report["identity"]["software"]["scikit-learn"] = recorded_sklearn
    if version == 5 or recorded_sklearn is not None:
        report["fingerprint"] = evaluation._digest(report["identity"])
        save(path, report)
        assert evaluation.load_calibration_report(path) == report
    assert report["selected_candidate_index"] == 0
    diagnostics = report["screening_diagnostics"]
    assert diagnostics["candidates"][2]["screens"] == []
    diagnostics["candidates"][0]["screens"] = []
    diagnostics["inactive_parameters"][0]["candidate_indices"].remove(0)
    entry = report["refinement"]["parameters"][name]
    del entry["screening_mode"]
    entry["status"] = "lower_boundary_requires_expansion_or_constraint"
    report["identity"]["screening_diagnostics_sha256"] = evaluation._digest(diagnostics)
    report["fingerprint"] = evaluation._digest(report["identity"])
    save(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="Inactive refinement"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


class _SwitchingScreen(BinaryScreening):
    def set_params(self, **params):
        super().set_params(**params)
        self.screening = "percentile"
        return self


def test_bound_screening_verifies_one_stock_path_beside_an_opaque_setter(tmp_path):
    y = np.tile([0, 1], 8)
    path = tmp_path / "two-screens.json"
    name = "cut__score_threshold"
    model = Pipeline([("other", _SwitchingScreen(screening="absolute", percentile=100)),
                      ("cut", BinaryScreening(screening="percentile", percentile=100)),
                      ("head", DummyClassifier(random_state=3))])
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(model, {name: [1, 2], "other__score_threshold": [1]},
                                      y[:, None] + 1, y, cv=2, seed=3, threads=1,
                                      refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 0
    entries = report["refinement"]["parameters"]
    assert entries["other__score_threshold"]["screening_mode"] == "percentile"
    assert entries[name]["screening_mode"] == "percentile"
    diagnostics = report["screening_diagnostics"]
    row = diagnostics["candidates"][0]
    row["screens"] = [s for s in row["screens"] if s["path"] != "cut"]
    assert [s["path"] for s in row["screens"]] == ["other"]
    next(g for g in diagnostics["inactive_parameters"] if g["parameter"] == name)["candidate_indices"].remove(0)
    del entries[name]["screening_mode"]
    entries[name]["status"] = "lower_boundary_requires_expansion_or_constraint"
    report["identity"]["screening_diagnostics_sha256"] = evaluation._digest(diagnostics)
    report["fingerprint"] = evaluation._digest(report["identity"])
    save(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="Inactive refinement"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


class _OnceInspectedScaler(StandardScaler):
    def get_params(self, deep=True):
        if deep:
            if getattr(self, "inspected_", False):
                raise ValueError("Only the first deep lookup succeeds on this clone.")
            self.inspected_ = True
        return super().get_params(deep=deep)


def test_bound_screening_retains_empty_failed_inspection_of_eligible_pipeline(tmp_path):
    y = np.tile([0, 1], 8)
    path = tmp_path / "opaque-sibling.json"
    model = Pipeline([("scale", _OnceInspectedScaler(with_mean=False)),
                      ("cut", BinaryScreening(screening="percentile", percentile=100)),
                      ("head", DummyClassifier(random_state=3))])
    name = "cut__score_threshold"
    evaluation.calibration_search(model, {name: [1, 2]}, y[:, None] + 1, y, cv=2, seed=3, threads=1,
                                  refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 0
    assert report["screening_diagnostics"]["candidates"][0]["screens"] == []
    assert report["refinement"]["parameters"][name]["status"] == "lower_boundary_requires_expansion_or_constraint"


def test_legacy_screening_uses_recorded_sklearn_version_for_stock_scaler(tmp_path):
    y = np.tile([0, 1], 8)
    path = tmp_path / "legacy-stock.json"
    name = "cut__score_threshold"
    model = Pipeline([("scale", RobustScaler(with_centering=False)),
                      ("cut", BinaryScreening(screening="percentile", percentile=100)),
                      ("head", DummyClassifier(random_state=3))])
    with pytest.warns(UserWarning, match="Inactive screening grid"):
        evaluation.calibration_search(model, {name: [1, 2]}, y[:, None] + 1, y, cv=2, seed=3, threads=1,
                                      refit=False, checkpoint_path=path)
    original = evaluation.load_calibration_report(path)
    use_legacy_screening_evidence(original)
    for version in ("1.9.0", "1.9.1", "unknown", None):
        report = deepcopy(original)
        software = report["identity"]["software"]
        if version is None:
            del software["scikit-learn"]
        else:
            software["scikit-learn"] = version
        report["fingerprint"] = evaluation._digest(report["identity"])
        save(path, report)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            if version in ("1.9.0", "1.9.1"):
                assert evaluation.load_calibration_report(path, expected_sha256=digest) == report
            else:
                with pytest.raises(ValueError, match="cannot be verified from the legacy screening evidence"):
                    evaluation.load_calibration_report(path, expected_sha256=digest)

        if version in ("1.9.0", "1.9.1"):
            diagnostics = report["screening_diagnostics"]
            diagnostics["candidates"][0]["screens"] = []
            diagnostics["inactive_parameters"][0]["candidate_indices"].remove(0)
            entry = report["refinement"]["parameters"][name]
            del entry["screening_mode"]
            entry["status"] = "lower_boundary_requires_expansion_or_constraint"
            save(path, report)
            for digest in (None, sha256(path.read_bytes()).hexdigest()):
                with pytest.raises(ValueError, match="disagrees with the declared screen"):
                    evaluation.load_calibration_report(path, expected_sha256=digest)


def test_legacy_screening_version_cannot_resolve_opaque_deep_lookup(tmp_path):
    y = np.tile([0, 1], 8)
    path = tmp_path / "legacy-opaque.json"
    model = Pipeline([("scale", _OnceInspectedScaler(with_mean=False)),
                      ("cut", BinaryScreening(screening="percentile", percentile=100)),
                      ("head", DummyClassifier(random_state=3))])
    evaluation.calibration_search(model, {"cut__score_threshold": [1, 2]}, y[:, None] + 1, y, cv=2, seed=3,
                                  threads=1, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["screening_diagnostics"]["candidates"][0]["screens"] == []
    use_legacy_screening_evidence(report)
    report["identity"]["software"]["scikit-learn"] = "1.9.0"
    report["fingerprint"] = evaluation._digest(report["identity"])
    save(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="cannot be verified from the legacy screening evidence"):
            evaluation.load_calibration_report(path, expected_sha256=digest)
