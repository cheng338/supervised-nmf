"""JSON scalar dispatch preserves reporting, subclass and failure contracts."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from supervised_nmf import evaluation


class IntegerId(int):
    pass


class FloatingId(float):
    pass


class StringId(str):
    pass


class Label:
    def __repr__(self):
        return "stable-label"


class InvalidRepresentation:
    def __repr__(self):
        raise ValueError("invalid representation")


def callback(value):
    return value


@pytest.mark.parametrize("value", [IntegerId(2**100), StringId("é-Δ"), FloatingId(-0.0),
                                  FloatingId("nan"), FloatingId("inf")])
def test_scalar_subclasses_keep_their_existing_encoding(value):
    result = evaluation._jsonable(value)
    if isinstance(value, float) and not math.isfinite(value):
        assert result is None
    else:
        assert result is value
        if isinstance(value, float):
            assert math.copysign(1., result) == -1.


@pytest.mark.parametrize("length", [15, 16])
def test_integer_first_list_observes_mutations_during_conversion(length):
    identifier = IntegerId(2**100)

    class ClearList(np.int64):
        def item(self):
            values.clear()
            return 19

    class ChangeList(np.int64):
        def item(self):
            values[-1] = identifier
            values.extend([ClearList(0), 99])
            return [True, np.nan]

    values = [7, ChangeList(0), *range(2, length)]
    result = evaluation._jsonable(values)
    assert result == [7, [True, None], *range(2, length - 1), identifier, 19]
    assert result[-2] is identifier
    assert values == []


def test_list_subclass_is_converted_through_its_iterator():
    calls = []

    class Numbers(list):
        def __len__(self):
            raise AssertionError("Unexpected list subclass length probe")

        def __getitem__(self, index):
            raise AssertionError("Unexpected list subclass index probe")

        def __iter__(self):
            calls.append("iter")
            return iter((7, np.nan))

    assert evaluation._jsonable(Numbers(range(16))) == [7, None]
    assert calls == ["iter"]


def test_nested_report_bytes_and_digest_follow_json_contract(tmp_path):
    path_value = Path("records") / "é.json"
    array = np.asarray([0.0, -0.0, np.nan, np.inf])
    array.flags.writeable = False
    before = array.tobytes()
    payload = {1: (None, True, IntegerId(2**100)), "values": array,
               "label": StringId("é-Δ"), "path": path_value,
               "callback": callback, "object": Label()}
    expected = {"1": [None, True, 2**100], "values": [0.0, -0.0, None, None],
                "label": "é-Δ", "path": str(path_value),
                "callback": f"{callback.__module__}.{callback.__qualname__}", "object": "stable-label"}
    expected_digest = hashlib.sha256(json.dumps(expected, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert evaluation._digest(payload) == expected_digest
    output = tmp_path / "report.json"
    evaluation.write_json(output, {"payload": payload, "sha256": expected_digest})
    expected_text = json.dumps({"payload": expected, "sha256": expected_digest},
                               indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert output.read_text(encoding="utf-8") == expected_text
    assert array.tobytes() == before and not array.flags.writeable


@pytest.mark.parametrize("kind", ["list", "dict", "repr"])
def test_unserializable_values_keep_their_existing_failure(kind):
    if kind == "list":
        value = []
        value.append(value)
        error = RecursionError
    elif kind == "dict":
        value = {}
        value["self"] = value
        error = RecursionError
    else:
        value, error = InvalidRepresentation(), ValueError
    with pytest.raises(error):
        evaluation._jsonable(value)
