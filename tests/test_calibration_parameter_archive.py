"""Candidate displays and the request identity of a CV-only calibration."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from supervised_nmf import evaluation
from supervised_nmf._calibration_report import _check_display
from supervised_nmf._identity import configuration_identity


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    path = tmp_path_factory.mktemp("parameter-archive") / "original.json"
    labels = np.tile([0, 1], 3)
    evaluation.calibration_search(
        LogisticRegression(random_state=7),
        {"C": [np.float64(0.125)], "fit_intercept": [np.bool_(True)],
         "class_weight": [{0: 1, np.int64(1): 3}]},
        labels[:, None] * 4 - 2, labels, cv=3, seed=9, refit=False, checkpoint_path=path,
    )
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 0
    return report


def save(path, report):
    report["checkpoint_sha256"] = evaluation._digest({
        name: value for name, value in report.items() if name != "checkpoint_sha256"
    })
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


@pytest.mark.parametrize(("key", "replacement"), [
    ("class_weight", {"0": 100, "1": 3}),
    ("class_weight", {"0": 1, "1": 300}),
    ("fit_intercept", False),
    ("C", None),
])
def test_resealed_winning_parameters_must_match_identity(tmp_path, archive, key, replacement):
    report = deepcopy(archive)
    report["candidates"][0]["parameters"][key] = replacement
    report["selected_parameters"][key] = replacement
    refinement = report["refinement"]["parameters"][key]
    refinement["selected"] = replacement
    if replacement is None:
        refinement["status"] = "categorical"
    path = tmp_path / "changed.json"
    save(path, report)
    with pytest.raises(ValueError):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("axis_observed", [False, True])
def test_legacy_null_cannot_have_diagnostics_for_a_numeric_selection(tmp_path, archive, axis_observed):
    report = deepcopy(archive)
    if not axis_observed:
        report["identity"]["fold_evidence_version"] = 5
        del report["refinement_axis_inputs"]
    # Isolate the refinement witness even when sklearn validation was disabled.
    report["identity"]["sklearn_skip_parameter_validation"] = True
    config = dict((key["value"], value) for key, value in report["identity"]["structural_configuration"]["items"])
    declaration = dict((key["value"], value) for key, value in config["candidates"]["items"][0]["items"])
    del declaration["C"]["finite"]
    report["fingerprint"] = evaluation._digest(report["identity"])
    path = tmp_path / "legacy.json"
    save(path, report)
    assert evaluation.load_calibration_report(path) == report
    report["candidates"][0]["parameters"]["C"] = None
    report["selected_parameters"]["C"] = None
    report["refinement"]["parameters"]["C"]["selected"] = None
    save(path, report)
    with pytest.raises(ValueError, match="Refinement axis" if axis_observed else "Refinement status"):
        evaluation.load_calibration_report(path)
    report["refinement"]["parameters"]["C"]["status"] = "categorical"
    save(path, report)
    if axis_observed:
        # The v6 writer retained the numeric coordinate even when the scalar
        # declaration did not record finiteness. A categorical claim conflicts.
        with pytest.raises(ValueError, match="Refinement axis"):
            evaluation.load_calibration_report(path)
    else:
        assert evaluation.load_calibration_report(path) == report


def test_legacy_scalar_finiteness_uses_winning_and_nonwinning_rows(tmp_path):
    path = tmp_path / "legacy.json"
    y = np.tile([0, 1], 8)
    evaluation.calibration_search(
        LogisticRegression(random_state=7), {"C": [np.float64(1.0)], "max_iter": [1, 100]},
        y[:, None] * 4 - 2, y, cv=2, seed=9, refit=False, checkpoint_path=path,
    )
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] == 1
    identity = report["identity"]
    del identity["sklearn_skip_parameter_validation"]
    config = {k["value"]: v for k, v in identity["structural_configuration"]["items"]}
    for declaration in config["candidates"]["items"]:
        params = {k["value"]: v for k, v in declaration["items"]}
        del params["C"]["finite"]
    report["fingerprint"] = evaluation._digest(identity)
    save(path, report)
    assert evaluation.load_calibration_report(path) == report

    for index in (0, 1):
        changed = deepcopy(report)
        changed["candidates"][index]["parameters"]["C"] = None
        if index == changed["selected_candidate_index"]:
            changed["selected_parameters"]["C"] = None
            changed["refinement"]["parameters"]["C"] = {"selected": None, "status": "categorical"}
        save(path, changed)
        with pytest.raises(ValueError, match="finiteness of the same NumPy parameter"):
            evaluation.load_calibration_report(path)


@pytest.mark.parametrize("scalar_type", [np.float16, np.float32, np.float64])
def test_nested_legacy_scalar_binds_only_the_same_storage(scalar_type):
    # A different scalar can genuinely be nonfinite, including a NaN whose
    # payload was lost in the display. Check leaves reached through object arrays.
    finite = configuration_identity(scalar_type(-0.0))
    nonfinite = configuration_identity(scalar_type(np.nan))
    del finite["finite"], nonfinite["finite"]
    node = {"kind": "ndarray", "dtype": {"str": "|O"}, "shape": [3], "content": [finite, nonfinite, finite]}
    _check_display(node, [-0.0, None, -0.0], float_finiteness={})
    with pytest.raises(ValueError, match="finiteness of the same NumPy parameter"):
        _check_display(node, [-0.0, None, None], float_finiteness={})


@pytest.mark.parametrize(("value", "qualified"), [(0.125, True), (np.inf, True), (np.nan, False)])
def test_fitted_legacy_logistic_null_requires_positive_infinity(tmp_path, value, qualified):
    path = tmp_path / "legacy-logistic.json"
    x = np.arange(24, dtype=float)[:, None]
    y = np.tile([0, 1], 12)
    evaluation.calibration_search(LogisticRegression(random_state=7), {"C": [np.float64(value)]},
                                  x, y, cv=2, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["candidates"][0]["valid"] is qualified
    config = {k["value"]: v for k, v in report["identity"]["structural_configuration"]["items"]}
    node = config["candidates"]["items"][0]["items"][0][1]
    del node["finite"]
    report["fingerprint"] = evaluation._digest(report["identity"])
    save(path, report)
    assert evaluation.load_calibration_report(path) == report
    if np.isfinite(value) and report["identity"]["software"]["scikit-learn"] in ("1.9.0", "1.9.1"):
        report["candidates"][0]["parameters"]["C"] = report["selected_parameters"]["C"] = None
        report["refinement"]["parameters"]["C"] = {"selected": None, "status": "categorical"}
        save(path, report)
        with pytest.raises(ValueError, match="Null C"):
            evaluation.load_calibration_report(path)


@pytest.mark.parametrize("mapping", [{0: 1, "0": 2}, {"0": 2, np.int64(0): 1}])
def test_display_name_collisions_keep_the_last_declared_value(mapping):
    display = evaluation._jsonable(mapping)
    declared = configuration_identity(mapping)
    _check_display(declared, display)
    with pytest.raises(ValueError):
        _check_display(declared, {"0": -1})


def test_numpy_bool_raw_noncanonical_storage_is_true():
    declaration = configuration_identity(np.bool_(True))
    declaration["content"] = hashlib.sha256(b"\xff").hexdigest()
    _check_display(declaration, True)
    with pytest.raises(ValueError):
        _check_display(declaration, False)


def test_noncanonical_nan_payloads_remain_readable():
    value = np.array([0x7FF8000000004321], dtype=np.uint64).view(np.float64)[0]
    declared = configuration_identity(value)
    assert declared["finite"] is False
    assert evaluation._jsonable(value) is None
    _check_display(declared, None)
    del declared["finite"]
    _check_display(declared, None)


@pytest.mark.parametrize("dtype", [float, object])
def test_resealed_array_priors_must_match_the_selected_grid_value(tmp_path, dtype):
    path = tmp_path / "priors.json"
    labels = np.tile([0, 1], 8)
    X = (labels * 10 + np.arange(len(labels)) / 100)[:, None]
    evaluation.calibration_search(GaussianNB(), {"priors": [np.array([0.25, 0.75], dtype=dtype)]}, X, labels,
                                  cv=2, seed=9, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_parameters"]["priors"] == [0.25, 0.75]
    changed = [0.75, 0.25]
    report["candidates"][0]["parameters"]["priors"] = changed
    report["selected_parameters"]["priors"] = changed
    report["refinement"]["parameters"]["priors"]["selected"] = changed
    save(path, report)
    with pytest.raises(ValueError):
        evaluation.load_calibration_report(path)


def test_resealed_unicode_array_cannot_change_predicted_class(tmp_path):
    path = tmp_path / "constant.json"
    labels = np.tile(["red", "tan"], 4)
    evaluation.calibration_search(
        DummyClassifier(strategy="constant", random_state=7), {"constant": [np.array(["red"], dtype="U3")]},
        np.arange(len(labels))[:, None], labels, cv=2, refit=False, checkpoint_path=path,
    )
    report = evaluation.load_calibration_report(path)
    assert report["selected_parameters"]["constant"] == ["red"]
    report["candidates"][0]["parameters"]["constant"] = ["tan"]
    report["selected_parameters"]["constant"] = ["tan"]
    report["refinement"]["parameters"]["constant"]["selected"] = ["tan"]
    save(path, report)
    with pytest.raises(ValueError, match="text array parameter"):
        evaluation.load_calibration_report(path)


@pytest.mark.skipif(not hasattr(np.dtypes, "StringDType"), reason="requires NumPy variable-width strings")
def test_variable_width_string_array_checks_labels_and_text_sentinel():
    dtype = np.dtypes.StringDType(na_object="missing")
    values = np.array(["red", None, "missing"], dtype=np.dtypes.StringDType(na_object=None)).astype(dtype)
    declared = configuration_identity(values)
    public = evaluation._jsonable(values)
    assert public == ["red", "missing", "missing"]
    _check_display(declared, public)
    with pytest.raises(ValueError):
        _check_display(declared, ["tan", "missing", "missing"])


@pytest.mark.parametrize("dtype", ["|T8", "|T16"])
def test_released_variable_width_text_element_identities(dtype):
    declared = {"kind": "ndarray", "shape": [1], "dtype": {"str": dtype},
                "content": [{"kind": "str", "value": "red"}]}
    _check_display(declared, ["red"])
    with pytest.raises(ValueError):
        _check_display(declared, ["tan"])


def test_boolean_classes_bind_legacy_float_weights_including_negative_zero():
    declared = configuration_identity({np.float64(-0.0): 1, np.float64(1): 3})
    for key, _ in declared["items"]:
        del key["key_text"]
    _check_display(declared, {"-0.0": 1, "1.0": 3}, [False, True])
    for changed in ({"0.0": 1, "1.0": 3}, {"-0.0": 1, "2.0": 3}):
        with pytest.raises(ValueError):
            _check_display(declared, changed, [False, True])


@pytest.mark.parametrize("new_name", ["2.0", "bogus"])
def test_legacy_float_class_weight_names_use_recorded_class_labels(tmp_path, new_name):
    path = tmp_path / "float-weights.json"
    labels = np.tile([0, 1], 4)
    evaluation.calibration_search(
        LogisticRegression(random_state=7), {"class_weight": [{np.float64(0): 1, np.float64(1): 3}]},
        labels[:, None], labels, cv=2, refit=False, checkpoint_path=path,
    )
    report = evaluation.load_calibration_report(path)
    config = {k["value"]: v for k, v in report["identity"]["structural_configuration"]["items"]}
    for key, _ in config["candidates"]["items"][0]["items"][0][1]["items"]:
        key.pop("key_text")
    report["fingerprint"] = evaluation._digest(report["identity"])
    save(path, report)
    assert evaluation.load_calibration_report(path)["selected_parameters"]["class_weight"] == {"0.0": 1, "1.0": 3}
    weights = {"0.0": 1, new_name: 3}
    report["candidates"][0]["parameters"]["class_weight"] = weights
    report["selected_parameters"]["class_weight"] = weights
    report["refinement"]["parameters"]["class_weight"]["selected"] = weights
    save(path, report)
    with pytest.raises(ValueError, match="dictionary key"):
        evaluation.load_calibration_report(path)


def test_legacy_float_class_keys_preserve_truncated_spelling_and_nan_payload():
    precise = np.float64(1.1234567891234568)
    nan = np.array([0x7ff8000000004321], dtype=np.uint64).view(np.float64)[0]
    mapping = {precise: 1, nan: 3}
    declared = configuration_identity(mapping)
    for key, _ in declared["items"]:
        key.pop("key_text")
    with np.printoptions(legacy="1.13"):
        display = evaluation._jsonable(mapping)
    _check_display(declared, display, [float(precise), 7])


@pytest.mark.parametrize(("label", "renamed"), [
    (1.0, "1.001"), (1.0, "1.00"), (1.0, " 1.0"), (1.0, "01.0"), (10.0, "1_0"),
    (-0.0, "0.0"), (1e-20, "1e-020"), (1e20, "1e20"),
])
def test_known_legacy_float_key_requires_a_possible_numpy_spelling(label, renamed):
    mapping = {np.float64(label): 7}
    declared = configuration_identity(mapping)
    declared["items"][0][0].pop("key_text")
    _check_display(declared, evaluation._jsonable(mapping), [label])
    with pytest.raises(ValueError, match="dictionary key"):
        _check_display(declared, {renamed: 7}, [label])


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_shortest_subnormal_key_can_round_trip_from_a_distant_decimal(dtype):
    key = np.finfo(dtype).smallest_subnormal
    mapping = {key: 1}
    declared = configuration_identity(mapping)
    declared["items"][0][0].pop("key_text")
    _check_display(declared, evaluation._jsonable(mapping), [float(key), 1])


def test_legacy_unicode_keys_and_values_respect_declared_width():
    mapping = {np.str_("case"): 1, np.str_("control"): 3}
    declared = configuration_identity(mapping)
    for key, _ in declared["items"]:
        key.pop("key_text")
        key.pop("display")
    _check_display(declared, {"case": 1, "control": 3})
    for name in ("other_class", "dog"):
        with pytest.raises(ValueError, match="dictionary key"):
            _check_display(declared, {name: 1, "control": 3})
    scalar = configuration_identity(np.str_("red"))
    scalar.pop("display")
    with pytest.raises(ValueError, match="width"):
        _check_display(scalar, "purple")


@pytest.mark.parametrize("text", ["", "a\0", "a\0b", "\U0001f600", "\ud83d\ude00"])
def test_unicode_archive_preserves_padding_and_escaped_surrogates(text):
    value = np.str_(text)
    scalar = configuration_identity(value)
    scalar.pop("display")
    mapping = {value: 1}
    declared = configuration_identity(mapping)
    declared["items"][0][0].pop("key_text")
    declared["items"][0][0].pop("display")
    for array_type in ("<U6", ">U6"):
        array = np.array([text, "short"], dtype=array_type)
        public = json.loads(json.dumps(evaluation._jsonable(array)))
        _check_display(configuration_identity(array), public)
    _check_display(scalar, json.loads(json.dumps(evaluation._jsonable(value))))
    _check_display(declared, json.loads(json.dumps(evaluation._jsonable(mapping))))


@pytest.mark.parametrize("legacy", [True, False])
def test_mixed_class_weight_keys(legacy):
    declared = configuration_identity({0: 1, np.float64(1): 3})
    if legacy:
        declared["items"][1][0].pop("key_text")
    _check_display(declared, {"0": 1, "1.0": 3})
    for changed in ({"0": 99, "1.0": 3}, {"0": 1, "1.0": 99}, {}):
        with pytest.raises(ValueError):
            _check_display(declared, changed)


@pytest.mark.parametrize(("values", "changed"), [
    (np.array([0.25, 0.75]), [None, 0.75]),
    (np.array([np.inf, 0.75], dtype=">f8"), [None, 0.25]),
    (np.array([0xff, 0x00], dtype=np.uint8).view(bool), [True, True]),
])
def test_lossy_array_display_preserves_the_observable_entries(values, changed):
    declared = configuration_identity(values)
    _check_display(declared, evaluation._jsonable(values))
    with pytest.raises(ValueError):
        _check_display(declared, changed)


@pytest.mark.parametrize("legacy", [True, False])
def test_numpy_text_class_weight_keys(legacy):
    mapping = {np.str_("green"): 1, np.str_("blue"): 2}
    declared = configuration_identity(mapping)
    if legacy:
        for key, _ in declared["items"]:
            key.pop("key_text")
            key.pop("display")
    _check_display(declared, {"green": 1, "blue": 2})
    with pytest.raises(ValueError):
        _check_display(declared, {"green": 1, "blue": 222})


@pytest.mark.parametrize("value", [np.array(7, dtype=">u2"), np.empty((2, 0, 3), dtype="i4"),
                                    np.arange(24, dtype=">i4").reshape(4, 6).T,
                                    np.array([np.nan, np.inf, -0.0], dtype=">f8"),
                                    np.array([True, False], dtype=np.bool_)])
def test_array_storage_and_partial_displays(value):
    _check_display(configuration_identity(value), evaluation._jsonable(value))


def test_binary64_extended_mantissa_must_be_exact_and_canonical():
    # Simulate the Windows double-width long double on either host.
    node = configuration_identity(np.float64(2.0))
    node["dtype"]["format"][0] = "g"
    genuine = {"__supervised_nmf_scalar__": "longdouble-v1", "significand": "1", "exponent": 1}
    _check_display(node, genuine)
    for edited in (2.0, dict(genuine, significand="9007199254740993", exponent=-52),
                   dict(genuine, significand="2", exponent=0)):
        with pytest.raises(ValueError):
            _check_display(node, edited)


def test_wide_longdouble_display_tracks_the_portable_value():
    value = np.longdouble(0.125)
    node = configuration_identity(value)
    display = evaluation._jsonable(value)
    _check_display(node, display)
    for edited in ("opaque", dict(display, exponent=display["exponent"] + 1)):
        with pytest.raises(ValueError):
            _check_display(node, edited)
    node.pop("display")
    _check_display(node, display)
    with pytest.raises(ValueError):
        _check_display(node, "opaque")


@pytest.mark.parametrize("other", [None, {"__supervised_nmf_scalar__": "longdouble-v1",
                                         "significand": "1", "exponent": -2}])
def test_repeated_legacy_longdouble_keeps_one_value(other):
    value = np.longdouble(0.125)
    node = configuration_identity(value)
    # Model the wider release ABI on both hosts. Only comparisons between
    # matching bytes are portable when a wide scalar's padding is unknown.
    node["dtype"]["str"] = "<f16"
    node.pop("display", None)
    node.pop("finite", None)
    shown = {"__supervised_nmf_scalar__": "longdouble-v1", "significand": "1", "exponent": -3}
    _check_display(node, other, float_finiteness={})
    seen = {}
    _check_display(node, shown, float_finiteness=seen)
    with pytest.raises(ValueError, match="same extended floating-point parameter"):
        _check_display(node, other, float_finiteness=seen)


def test_concrete_path_is_checked_as_a_historical_string():
    path = Path("calibration.json")
    node = configuration_identity(path)
    _check_display(node, str(path))
    with pytest.raises(ValueError):
        _check_display(node, "other.json")
