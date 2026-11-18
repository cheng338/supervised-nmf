"""Required C-coordinate copies preserve bits, ownership and failure semantics.

The large arrays below exercise only copies. Real estimator checks use tiny
data; admitted large-fit interactions belong to the bounded integration study.
"""
import gc
import inspect
from pathlib import Path
import sys
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import experimental_nmf as experimental

assert Path(experimental.__file__).resolve() == ROOT / "scripts/experimental_nmf.py"


@pytest.fixture(autouse=True)
def observation_state():
    profile = sys.getprofile()
    policy, callback = np.geterr(), np.geterrcall()
    assert profile is None
    with threadpool_limits(limits=1):
        try:
            yield
            assert sys.getprofile() is profile
            assert np.geterr() == policy and np.geterrcall() is callback
        finally:
            sys.setprofile(profile)
            np.seterr(**policy)
            np.seterrcall(callback)


class NumPyForwarder:
    def __getattr__(self, name):
        return getattr(np, name)


def bit_payload(shape=(8192, 20)):
    result = np.empty(shape, order="F")
    # Include signed zeros, subnormal/normal extremes, infinities and distinct
    # quiet/signaling NaNs. No floating arithmetic is used to install the bits.
    words = np.array([0, 0x8000000000000000, 1, 0x0010000000000000,
                      0x7fefffffffffffff, 0x7ff0000000000000,
                      0xfff0000000000000, 0x7ff8000000000042,
                      0xfff8000000000031, 0x7ff0000000000001], dtype=np.uint64)
    result.view(np.uint64).ravel(order="F")[:] = np.resize(words, result.size)
    return result


def assert_independent_copy(source, actual):
    assert actual.shape == source.shape and actual.dtype == source.dtype
    assert actual.tobytes(order="C") == source.tobytes(order="C")
    assert actual.flags.owndata and actual.base is None
    assert actual.flags.c_contiguous and actual.flags.writeable
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("shape,eligible", [
    ((8192,20), True), ((12288,21), True), ((8191,20), False),
    ((8193,20), False), ((4096,20), False), ((8192,19), False),
    ((8192,1), False), ((0,20), False),
])
def test_coordinate_copy_size_boundaries(shape, eligible):
    left = np.zeros(shape, order="F")
    assert experimental._return_coordinates_eligible(left) is eligible
    assert_independent_copy(left, experimental._copy_return_coordinates(left))


@pytest.mark.parametrize("kind", ["c", "strided", "unaligned", "float32", "nonnative", "metadata", "subclass", "vector"])
def test_coordinate_copy_metadata_and_layout_bypass(monkeypatch, kind):
    class Array(np.ndarray):
        pass
    shape = (8192,20)
    if kind == "strided": left = np.zeros((16384,20), order="F")[::2]
    elif kind == "unaligned": left = np.ndarray(shape, dtype=np.float64, buffer=bytearray(8192*20*8+1), offset=1, order="F")
    elif kind == "float32": left = np.zeros(shape, dtype=np.float32, order="F")
    elif kind == "nonnative": left = np.zeros(shape, dtype=np.dtype(np.float64).newbyteorder("S"), order="F")
    elif kind == "metadata": left = np.zeros(shape, dtype=np.dtype(np.float64, metadata={"tag":"retained"}), order="F")
    elif kind == "subclass": left = np.zeros(shape, order="F").view(Array)
    elif kind == "vector": left = np.zeros(8192)
    else: left = np.zeros(shape, order="C")
    class NoScratch(NumPyForwarder):
        def empty(self, *args, **kwargs):
            pytest.fail("Ineligible copies must use the original copy method")
    monkeypatch.setattr(experimental, "np", NoScratch())
    assert experimental._return_coordinates_eligible(left) is False
    actual = experimental._copy_return_coordinates(left)
    assert_independent_copy(left, actual)
    assert actual.dtype.metadata == left.dtype.metadata
    assert type(actual) is type(left)


@pytest.mark.parametrize("readonly", [False, True])
def test_integer_copy_preserves_every_bit_and_policy(monkeypatch, readonly):
    left = bit_payload()
    left.setflags(write=not readonly)
    before = left.tobytes(order="C")
    calls = []
    class Observe(NumPyForwarder):
        def bitwise_or(self, source, scalar, **kwargs):
            calls.append((source.dtype, kwargs["out"].dtype, kwargs["order"]))
            return np.bitwise_or(source, scalar, **kwargs)
    monkeypatch.setattr(experimental, "np", Observe())
    with np.errstate(all="raise"):
        first = experimental._copy_return_coordinates(left)
        second = experimental._copy_return_coordinates(left)
    assert calls == [(np.dtype("uint64"),np.dtype("uint64"),"F")]*2
    assert_independent_copy(left, first)
    assert_independent_copy(left, second)
    assert not np.shares_memory(first, second)
    first.view(np.uint64)[0,0] ^= np.uint64(1)
    assert left.tobytes(order="C") == second.tobytes(order="C") == before
    assert left.flags.writeable is (not readonly)


def test_optional_guard_memory_error_declines_to_original_copy(monkeypatch):
    left = bit_payload()
    def unavailable(_): raise MemoryError("optional metadata")
    monkeypatch.setattr(experimental, "_return_coordinates_eligible", unavailable)
    assert_independent_copy(left, experimental._copy_return_coordinates(left))


def test_required_storage_failure_is_not_retried(monkeypatch):
    left = bit_payload(); calls = []
    class Fail(NumPyForwarder):
        def empty(self, *args, **kwargs):
            calls.append("allocate")
            raise MemoryError("required coordinate storage")
    monkeypatch.setattr(experimental, "np", Fail())
    with pytest.raises(MemoryError, match="required coordinate storage") as caught:
        experimental._copy_return_coordinates(left)
    assert calls == ["allocate"]
    assert caught.value.__context__ is None


@pytest.mark.parametrize("fallback_failure", [False, True])
def test_partial_optional_population_dies_before_required_fallback(monkeypatch, fallback_failure):
    left = bit_payload(); original = left.tobytes(order="C")
    scratch_refs = []; copies = []
    class Partial(NumPyForwarder):
        def empty(self, *args, **kwargs):
            value = np.empty(*args, **kwargs)
            scratch_refs.append(weakref.ref(value))
            return value
        def bitwise_or(self, source, scalar, **kwargs):
            out = kwargs["out"]
            out.fill(np.uint64(0))
            np.bitwise_or(source[:1,:1], scalar, out=out[:1,:1])
            raise MemoryError("optional integer population")
    monkeypatch.setattr(experimental, "np", Partial())
    def observe(frame, event, function):
        if (event == "c_call" and getattr(function,"__name__",None) == "copy"
                and getattr(function,"__self__",None) is left):
            assert len(scratch_refs) == 1 and scratch_refs[0]() is None
            assert sys.exception() is None  # Fallback is outside the trial handler.
            copies.append(True)
            if fallback_failure: raise MemoryError("required original copy")
    sys.setprofile(observe)
    try:
        if fallback_failure:
            with pytest.raises(MemoryError, match="required original copy") as caught:
                experimental._copy_return_coordinates(left)
            assert caught.value.__context__ is None
        else:
            assert_independent_copy(left, experimental._copy_return_coordinates(left))
    finally:
        sys.setprofile(None)
    assert copies == [True] and left.tobytes(order="C") == original


def test_nonmemory_population_error_is_not_hidden(monkeypatch):
    left = bit_payload()
    class Fail(NumPyForwarder):
        def bitwise_or(self, *args, **kwargs): raise RuntimeError("unexpected traversal")
    monkeypatch.setattr(experimental, "np", Fail())
    with pytest.raises(RuntimeError, match="unexpected traversal"):
        experimental._copy_return_coordinates(left)


def test_ineligible_subclass_retains_original_copy_dispatch():
    class Custom(np.ndarray):
        def copy(self, *args, **kwargs):
            assert not args and not kwargs
            return "original subclass method"
    left = np.zeros((8,2)).view(Custom)
    assert experimental._copy_return_coordinates(left) == "original subclass method"


def test_actual_fit_publication_report_return_and_transform_are_independent():
    data = np.zeros((16,5)); before = data.tobytes()
    model = experimental.ANLSBPP(2)
    fit_code = inspect.unwrap(experimental.ANLSBPP.fit_transform).__code__
    copy_code = experimental._copy_return_coordinates.__code__
    order = []; private = []; outputs = []
    def observe(frame, event, value):
        if frame.f_code is copy_code and event == "call" and frame.f_back.f_code is fit_code:
            phase = "publication" if not hasattr(model,"training_coordinates_") else "return"
            order.append(phase)
            private.append(weakref.ref(frame.f_locals["left"]))
            if phase == "publication":
                assert not hasattr(model,"components_")
            else:
                assert hasattr(model,"reconstruction_err_") and hasattr(model,"nnls_diagnostics_")
        if frame.f_code is copy_code and event == "return": outputs.append(weakref.ref(value))
        if (frame.f_code is fit_code and event == "c_call" and getattr(value,"__name__",None) == "copy"
                and getattr(value,"__self__",None) is frame.f_locals.get("H")):
            order.append("H.copy")
    sys.setprofile(observe)
    try: coordinates = model.fit_transform(data)
    finally: sys.setprofile(None)
    assert order == ["H.copy","publication","return"]
    assert outputs[0]() is model.training_coordinates_ and outputs[1]() is coordinates
    gc.collect(); assert all(ref() is None for ref in private)
    assert model.converged_ and model.n_iter_ == 0 and model.reconstruction_err_ == 0
    assert_independent_copy(model.training_coordinates_, coordinates)
    assert not np.shares_memory(model.components_, coordinates)
    coordinates[0,0] = 1
    assert not model.training_coordinates_.any()
    saved_H, saved_W = model.components_.tobytes(), model.training_coordinates_.tobytes()
    with experimental.capture_transform_diagnostics(model) as diagnostics:
        query = model.transform(data[:3])
    assert diagnostics["performed"] and diagnostics["converged"]
    assert model.components_.tobytes() == saved_H and model.training_coordinates_.tobytes() == saved_W
    assert data.tobytes() == before and not np.shares_memory(query, model.training_coordinates_)


def test_publication_failure_does_not_partially_publish_factors(monkeypatch):
    model = experimental.ANLSBPP(2)
    calls = []
    def fail(left):
        calls.append(left.shape)
        assert not hasattr(model,"components_") and not hasattr(model,"training_coordinates_")
        raise MemoryError("required publication")
    monkeypatch.setattr(experimental,"_copy_return_coordinates",fail)
    with pytest.raises(MemoryError,match="required publication"):
        model.fit_transform(np.zeros((8,4)))
    assert calls == [(8,2)]
    assert not hasattr(model,"components_") and not hasattr(model,"training_coordinates_")
    assert not hasattr(model,"reconstruction_err_")
