"""Fit RNG ownership across sklearn's actual clone and checkpoint boundaries."""

import functools
import json
import struct
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_random_state

from supervised_nmf import evaluation
from supervised_nmf._identity import configuration_identity, implementation_sources


numpy2 = pytest.mark.skipif(np.__version__.split(".")[0] != "2",
                            reason="NumPy 2.x preserves stock child lineage in clones")


class DrawClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, draw):
        self.draw = draw

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.label_ = self.draw()
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[self.label_])

    def predict_proba(self, X):
        return (self.predict(X)[:, None] == self.classes_).astype(float)


def pool_callback(layout, shared, array_first):
    rng = np.random.default_rng(841)
    pool = rng.bit_generator.seed_seq.pool
    if layout == "words":
        view = pool.view()
    elif layout == "wide":
        view = pool.view(np.uint64)[::-1]
    elif layout == "matrix":
        view = pool.view(np.uint8).reshape(4, -1)[:, ::-2]
    elif layout == "scalar":
        view = pool[1:2].reshape(())
    elif layout == "indirect":
        view = np.lib.stride_tricks.as_strided(pool.view(np.uint8)[1:],
                                               shape=(2, 2), strides=(1, 2))
    else:
        raise AssertionError(layout)
    array = view if shared else view.copy()
    pair = (array, rng) if array_first else (rng, array)

    def draw():
        first, second = pair
        buffer, stream = (first, second) if isinstance(first, np.ndarray) else (second, first)
        sequence = stream.bit_generator.seed_seq
        before = sequence.generate_state(sequence.pool_size)
        saved = buffer.flat[0]
        try:
            buffer.flat[0] = int(saved) ^ 1
            return int(not np.array_equal(before, sequence.generate_state(sequence.pool_size)))
        finally:
            buffer.flat[0] = saved

    return draw


def calibrate(draw, path=None):
    return calibrate_model(DrawClassifier(draw), path)


def calibrate_model(model, path=None):
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 14 + [1] * 6)
    return evaluation.calibration_search(model, {}, X, y, cv=2,
                                         seed=17, checkpoint_path=path, threads=1)


def science(result):
    X = np.arange(8, dtype=float).reshape(4, 2)
    row = result.report["candidates"][0]
    assert row["valid"]
    return (result.report["identity"]["folds"], [fold["score"] for fold in row["folds"]],
            result.report["selected_parameters"], result.estimator.predict(X).tolist(),
            result.estimator.predict_proba(X).tolist())


@numpy2
@pytest.mark.parametrize("array_first", [False, True])
@pytest.mark.parametrize("layout", ["words", "wide", "matrix", "scalar", "indirect"])
def test_cloned_function_preserves_seed_pool_views(layout, array_first):
    shared = pool_callback(layout, True, array_first)
    copied = pool_callback(layout, False, array_first)
    assert clone(DrawClassifier(shared)).draw is shared
    flags = []
    original = configuration_identity(shared, _split_limits=flags)
    assert shared() == 1 and copied() == 0
    assert configuration_identity(shared) == original
    assert original != configuration_identity(copied)
    observed, other = [], []
    implementation_sources(shared, _observation=observed)
    implementation_sources(copied, _observation=other)
    assert observed != other
    assert "captured fit RNG" in flags
    assert ("unverified fit RNG" in flags) == (layout == "indirect")


@numpy2
def test_pool_views_with_equal_bytes_bind_layout_and_write_access():
    def declaration(offset, reverse, writable):
        generator = np.random.default_rng(841)
        pool = generator.bit_generator.seed_seq.pool
        pool.fill(0)
        view = pool[offset:offset + 2]
        if reverse:
            view = view[::-1]
        view.setflags(write=writable)
        return capture((generator, view))

    cases = [(0, False, True), (1, False, True), (0, True, True), (0, False, False)]
    records = [json.dumps(configuration_identity(declaration(*case))) for case in cases]
    assert len(set(records)) == len(cases)
    assert records[0] == json.dumps(configuration_identity(declaration(*cases[0])))


@numpy2
@pytest.mark.parametrize("layout,array_first", [("words", True), ("wide", False),
                                               ("matrix", True), ("scalar", False),
                                               ("indirect", True)])
@pytest.mark.parametrize("writer_shared", [False, True])
def test_pool_callback_restart_compares_with_uninterrupted_fit(tmp_path, layout, array_first,
                                                               writer_shared):
    def make(shared):
        return pool_callback(layout, shared, array_first)

    archive = tmp_path / "cv.json"
    writer = calibrate(make(writer_shared), archive)
    writer_reference = science(calibrate(make(writer_shared)))
    other_reference = science(calibrate(make(not writer_shared)))
    assert science(writer) == writer_reference
    assert writer_reference[1] != other_reference[1]
    assert writer_reference[-2:] != other_reference[-2:]
    before = archive.read_bytes()
    with patch.object(DrawClassifier, "fit") as fit, patch.object(evaluation, "write_json") as write:
        with pytest.raises(ValueError, match="incompatible|verifiable"):
            calibrate(make(not writer_shared), archive)
        fit.assert_not_called()
        write.assert_not_called()
    assert archive.read_bytes() == before
    if layout == "indirect" and writer_shared:
        with pytest.raises(ValueError, match="verifiable"):
            calibrate(make(writer_shared), archive)
        assert archive.read_bytes() == before
    else:
        assert science(calibrate(make(writer_shared), archive)) == writer_reference


@numpy2
@pytest.mark.parametrize("shared", [False, True])
def test_partial_restart_uses_only_compatible_callback_folds(tmp_path, shared):
    def make(linked):
        return pool_callback("matrix", linked, False)

    class Interrupted(BaseException):
        pass

    archive = tmp_path / "partial.json"
    publish = evaluation.write_json

    def stop_after_first(path, report):
        publish(path, report)
        if len(report["candidates"][0]["folds"]) == 1:
            raise Interrupted

    with patch.object(evaluation, "write_json", stop_after_first), pytest.raises(Interrupted):
        calibrate(make(shared), archive)
    before = archive.read_bytes()
    committed = json.loads(before)["candidates"][0]["folds"]
    with pytest.raises(ValueError, match="incompatible"):
        calibrate(make(not shared), archive)
    assert archive.read_bytes() == before
    resumed = calibrate(make(shared), archive)
    assert science(resumed) == science(calibrate(make(shared)))
    assert resumed.report["candidates"][0]["folds"][:1] == committed


def capture(value):
    def fixed():
        return value
    return fixed


def same_list(payload):
    left, right = payload
    item = left.pop()
    try:
        return int(not right)
    finally:
        left.append(item)


class ContainerClassifier(DrawClassifier):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        pair = self.draw
        if isinstance(pair, slice):
            pair = pair.start, pair.stop
        elif isinstance(pair, np.ndarray) and callable(pair[0]):
            pair = pair[0](), pair[1]()
        self.label_ = same_list(pair)
        return self


def container_model(shared, captured=False):
    rng = np.random.RandomState(19)
    payload = np.empty(2, dtype=object)
    payload[0] = [rng]
    payload[1] = payload[0] if shared else [rng]
    if not captured:
        return ContainerClassifier(payload)

    def draw():
        return same_list(payload)

    return DrawClassifier(draw)


def read_partial(left, *, right):
    return same_list((left, right))


def return_list(value):
    return value


class SeparatePartialsClassifier(DrawClassifier):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.label_ = same_list((self.left(), self.right()))
        return self


@numpy2
@pytest.mark.parametrize("shared", [False, True])
def test_separately_cloned_partial_parameters_match_their_own_reference(tmp_path, shared):
    def make():
        stream = np.random.RandomState(19)
        left = [stream]
        right = left if shared else [stream]
        return SeparatePartialsClassifier(functools.partial(return_list, left),
                                          functools.partial(return_list, right))

    copied = clone(make())
    assert copied.left() is not copied.right()
    assert copied.left()[0] is not copied.right()[0]
    archive = tmp_path / "separate.json"
    write_prefix(archive, make(), True)
    assert science(calibrate_model(make(), archive)) == science(calibrate_model(make()))


def compound_model(shared, kind):
    stream = np.random.RandomState(19)
    left = [stream]
    right = left if shared else [stream]
    if kind == "partial":
        return DrawClassifier(functools.partial(read_partial, left, right=right))
    if kind == "slice":
        return ContainerClassifier(slice(left, right))
    if kind == "object-scalar":
        value = np.empty((), dtype=[("left", object), ("right", object)])
        value["left"][()] = left
        value["right"][()] = right
        return ContainerClassifier(value[()])
    if kind == "partial-pair":
        value = np.empty(2, dtype=object)
        value[0] = functools.partial(return_list, left)
        value[1] = functools.partial(return_list, right)
        return ContainerClassifier(value)
    raise AssertionError(kind)


def cell_model(shared):
    rng = np.random.RandomState(19)

    def access(value):
        def set_value(new):
            nonlocal value
            old = value
            value = new
            return old

        def read():
            return value

        return set_value, read

    set_value, read = access(rng)
    if not shared:
        _, read = access(rng)

    def draw():
        old = set_value(None)
        try:
            return int(read() is None)
        finally:
            set_value(old)

    return DrawClassifier(draw)


def parent_model(view):
    rng = np.random.RandomState(19)
    parent = np.array([0, 1], dtype=np.uint32)
    declared = parent[:1] if view else parent[:1].copy()

    def draw():
        word = int(rng.get_state()[1][0])
        return (word & int(declared.base[1])) if declared.base is not None else 0

    return DrawClassifier(draw)


def write_prefix(path, model, partial):
    if not partial:
        return calibrate_model(model, path)
    publish = evaluation.write_json

    class Interrupted(BaseException):
        pass

    def interrupt(destination, report):
        publish(destination, report)
        if report["candidates"][0]["folds"]:
            raise Interrupted

    with patch.object(evaluation, "write_json", interrupt), pytest.raises(Interrupted):
        calibrate_model(model, path)


@numpy2
@pytest.mark.parametrize("make", [
    container_model, functools.partial(container_model, captured=True), cell_model, parent_model,
    functools.partial(compound_model, kind="partial"), functools.partial(compound_model, kind="slice"),
    functools.partial(compound_model, kind="object-scalar"), functools.partial(compound_model, kind="partial-pair"),
], ids=["object-array", "captured-list", "closure-cell", "array-parent",
        "partial-fields", "slice-fields", "scalar-fields", "two-partials"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_retained_ownership_recovery_matches_uninterrupted_model(tmp_path, make, shared, partial):
    archive = tmp_path / "retained.json"
    writer = write_prefix(archive, make(shared), partial)
    reference = science(calibrate_model(make(shared)))
    other_reference = science(calibrate_model(make(not shared)))
    assert other_reference[1] != reference[1] and other_reference[-2:] != reference[-2:]
    if writer is not None:
        assert science(writer) == reference
    before = archive.read_bytes()
    committed = json.loads(before)["candidates"][0]["folds"]
    with patch.object(evaluation, "write_json") as write:
        with pytest.raises(ValueError, match="incompatible|verifiable"):
            calibrate_model(make(not shared), archive)
        write.assert_not_called()
    assert archive.read_bytes() == before
    if make is parent_model and shared:
        # Even a matching view could see unrecorded values outside its bounds.
        with pytest.raises(ValueError, match="verifiable"):
            calibrate_model(make(shared), archive)
        assert archive.read_bytes() == before
    else:
        recovered = calibrate_model(make(shared), archive)
        assert science(recovered) == reference
        assert recovered.report["candidates"][0]["folds"][:len(committed)] == committed


def test_positional_seed_defaults_use_the_declared_argument_names():
    def seed_default(unrelated, seed=None, /, optional=7):
        pass

    def rng_default(required, rng=None, *, option=8):
        pass

    def state_default(random_state=None):
        pass

    def fixed(seed=11, unrelated=None):
        pass

    for fn, implicit in ((seed_default, True), (rng_default, True), (state_default, True),
                         (functools.partial(rng_default, 4), True), (fixed, False)):
        flags = []
        configuration_identity(fn, _split_limits=flags)
        assert ("implicit RNG" in flags) == implicit
        flags = []
        implementation_sources(fn, _split_limits=flags)
        assert ("implicit RNG" in flags) == implicit


def test_partial_bound_positions_and_keywords_are_classified_by_name():
    def project(random_state, rows):
        pass

    def named(rows, *, seed):
        pass

    def variadic(*args):
        pass

    cases = [
        (functools.partial(project, None), True), (functools.partial(project, 13), False),
        (functools.partial(project, 13, None), False), (functools.partial(named, seed=None), True),
        (functools.partial(named, seed=13), False), (functools.partial(variadic, None), True),
        (functools.partial(abs, None), True),
    ]
    for wrapped, expected in cases:
        for collect in (configuration_identity, implementation_sources):
            flags = []
            collect(wrapped, _split_limits=flags)
            assert ("implicit RNG" in flags) == expected


@numpy2
def test_exotic_rng_ownership_qualifications():
    rng = np.random.default_rng(31)
    dtype = np.dtype(np.uint32, metadata={"stream": rng})
    view = rng.bit_generator.seed_seq.pool.view([("word", "u4")])[0]
    adjusted = functools.partial(return_list, 1)
    adjusted.extra = rng

    def tagged():
        return 1

    tagged.extra = rng
    nested = np.dtype([("field", dtype)])
    subarray = np.dtype((dtype, (2,)))
    assert configuration_identity(subarray) != configuration_identity(np.dtype("V8"))
    for value in (dtype, nested, subarray, capture((rng, view)), adjusted, tagged):
        for collect in (configuration_identity, implementation_sources):
            flags = []
            collect(value, _split_limits=flags)
            assert "unverified fit RNG" in flags
            assert "unverified captured state" in flags
    flags = []
    configuration_identity(capture(view), _split_limits=flags)
    assert "unverified captured state" in flags and "unverified fit RNG" not in flags


@numpy2
@pytest.mark.parametrize("nested", [False, True])
def test_string_dtype_identity_preserves_plain_sentinel_and_coercion(nested):
    def records(options):
        dtype = np.dtypes.StringDType(**options)
        if nested:
            field = np.dtype("u4", metadata={"string_dtype": dtype})
            dtype = np.dtype([("words", field, (2,))])
        observed = []
        implementation_sources(dtype, _observation=observed)
        return tuple(json.dumps(value, sort_keys=True, allow_nan=False)
                     for value in (configuration_identity(dtype), observed[0]))

    nan, negative, payload = (struct.unpack("!d", bytes.fromhex(word))[0] for word in
                              ("7ff8000000000001", "fff8000000000001", "7ff8000000000002"))
    sentinels = (None, True, 1, 1., 1 + 0j, "a", b"a", b"b", nan, negative, payload,
                 complex(nan, payload), complex(nan, negative), complex(payload, negative))
    options = [{}, {"coerce": False}, {"na_object": None, "coerce": False}]
    options.extend({"na_object": value} for value in sentinels)
    full, observed = zip(*(records(settings) for settings in options))
    assert len(set(full)) == len(set(observed)) == len(options)
    assert records({}) == records({"coerce": True})
    assert records({"na_object": negative}) == records({"na_object": -nan})


@numpy2
def test_string_dtype_existing_integer_sentinel_ignores_decimal_display_limit():
    value = 1 << 16000
    original = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        dtype = np.dtypes.StringDType(na_object=value)
        reference = configuration_identity(dtype)
        sys.set_int_max_str_digits(640)
        assert configuration_identity(dtype) == reference
        assert reference["na_object"] == configuration_identity(value)
    finally:
        sys.set_int_max_str_digits(original)


@numpy2
@pytest.mark.parametrize("partial", [False, True])
def test_calibration_rejects_changed_string_dtype_nan_sign(tmp_path, partial):
    def make(negative):
        dtype = np.dtypes.StringDType(na_object=-float("nan") if negative else float("nan"))

        def draw():
            return int(np.signbit(dtype.na_object))

        return DrawClassifier(draw)

    archive = tmp_path / "cv.json"
    writer = science(calibrate_model(make(False)))
    reader = science(calibrate_model(make(True)))
    assert writer[0] == reader[0] and writer[1] != reader[1] and writer[-2:] != reader[-2:]
    write_prefix(archive, make(False), partial)
    before = archive.read_bytes()
    with patch.object(DrawClassifier, "fit") as fit, patch.object(evaluation, "write_json") as write:
        with pytest.raises(ValueError, match="incompatible"):
            calibrate_model(make(True), archive)
        fit.assert_not_called()
        write.assert_not_called()
    assert archive.read_bytes() == before
    assert science(calibrate_model(make(False), archive)) == writer


def test_deterministic_callback_with_array_view_can_recover(tmp_path):
    def make():
        view = np.arange(6)[1::2]

        def deterministic():
            return int(view[0])

        return DrawClassifier(deterministic)

    flags = []
    configuration_identity(make(), _split_limits=flags)
    assert "unverified fit RNG" not in flags
    archive = tmp_path / "deterministic.json"
    write_prefix(archive, make(), True)
    assert science(calibrate_model(make(), archive)) == science(calibrate_model(make()))


def dependencies(value):
    pairs = []
    assert (configuration_identity(value, _fit_rng_dependencies=pairs)
            == configuration_identity(value))
    return pairs


def test_callback_and_later_model_can_depend_on_one_private_engine():
    state = np.random.RandomState(17)
    wrapper = np.random.Generator(state._bit_generator)
    previous = dependencies(capture(state))
    current = dependencies(wrapper)
    assert any(a is b is state._bit_generator and captured and not retained
               for a, captured in previous for b, retained in current)
    independent = dependencies(np.random.RandomState(17))
    assert not any(a is b for a, _ in previous for b, _ in independent)


@numpy2
@pytest.mark.parametrize("entropy", [841, (841,)])
def test_separately_declared_pool_and_callback_buffer_are_dependencies(entropy):
    stream = np.random.default_rng(np.random.SeedSequence(entropy))
    view = stream.bit_generator.seed_seq.pool.view(np.uint8)[1::2]
    for previous, current in ((capture(view), stream), (capture(stream), view)):
        assert any(captured and not retained and isinstance(a, np.ndarray)
                   and isinstance(b, np.ndarray) and np.shares_memory(a, b)
                   for a, captured in dependencies(previous) for b, retained in dependencies(current))
    copied = dependencies(capture(view.copy()))
    owned = dependencies(stream)
    assert not any(isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.may_share_memory(a, b)
                   for a, _ in copied for b, _ in owned)


def test_splitter_and_ambient_engine_ownership_are_distinct():
    ambient = check_random_state(None)
    independent = np.random.RandomState(0)
    independent.set_state(ambient.get_state())
    wrapper = np.random.RandomState(ambient._bit_generator)
    modern = np.random.Generator(ambient._bit_generator)
    flags = []
    singleton_id = configuration_identity(ambient, _split_limits=flags)
    assert "ambient fit RNG" in flags and singleton_id["ambient_fit_rng"] == "singleton"
    assert "ambient_fit_rng" not in configuration_identity(independent)
    assert singleton_id != configuration_identity(independent)
    for value in (wrapper, modern):
        flags = []
        assert configuration_identity(value, _split_limits=flags)["ambient_fit_rng"] == "engine"
        assert "ambient fit RNG" in flags

    splitter = np.random.RandomState(17)
    independent = np.random.RandomState(17)
    assert configuration_identity(splitter) == configuration_identity(independent)
    for value, relation in ((splitter, "object"), (np.random.Generator(splitter._bit_generator), "engine"),
                            (independent, None)):
        flags = []
        entry = configuration_identity(capture(value), _split_seed=splitter, _split_limits=flags)
        assert "captured fit RNG" in flags
        assert entry["closure"][0].get("split_rng") == relation
        assert ("shared split RNG" in flags) == (relation is not None)
        flags = []
        implementation_sources(capture(value), _split_seed=splitter, _split_limits=flags)
        assert ("shared split RNG" in flags) == (relation is not None)


def child_draw(entropy, spawn_key):
    sequence = np.random.SeedSequence(entropy, spawn_key=spawn_key)
    rng = np.random.default_rng(sequence)

    def draw():
        return int(rng.bit_generator.seed_seq.generate_state(1)[0]) % 2

    return draw


@numpy2
@pytest.mark.parametrize("wide_field", ["entropy", "spawn_key"])
def test_arbitrary_width_seed_words_support_fresh_fits_and_restart(tmp_path, wide_field):
    huge = 1 << 16000

    def make(word):
        return child_draw(word, (7,)) if wide_field == "entropy" else child_draw(841, (word,))

    archive = tmp_path / "cv.json"
    reference = science(calibrate(make(huge)))
    assert science(calibrate(make(huge), archive)) == reference
    lineage = configuration_identity(make(huge))["closure"][0]["child_stream"]
    encoded = lineage["entropy"] if wide_field == "entropy" else lineage["spawn_key"][0]
    assert int(encoded, 0) == huge
    json.dumps(lineage, allow_nan=False)
    assert science(calibrate(make(huge), archive)) == reference
    before = archive.read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        calibrate(make(huge + 1), archive)
    assert archive.read_bytes() == before
