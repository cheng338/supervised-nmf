"""Unexecuted test support; copy beside selected caller tests for qualification.

Observe factories, preserving the exact production-selected matmul identity.
Only the caller-facing role is wrapped. No source/native arithmetic is patched.
The original protected operation and independent einsum check every returned
role product; this is correctness instrumentation, never a timing harness.
"""
import gc
import hashlib
import weakref

import numpy as np


def bits(value):
    return np.asarray(value).tobytes(order="C")


def state(value):
    return (type(value), value.dtype.str, value.shape, value.strides,
            value.flags.writeable, value.flags.aligned, bits(value))


def digest(value):
    return hashlib.sha256(bits(value)).hexdigest()


def assert_product(left, right, actual, original):
    assert type(actual) is np.ndarray and actual.dtype == np.float64
    assert actual.flags.owndata and actual.flags.c_contiguous
    assert actual.flags.writeable and actual.base is None
    assert not np.shares_memory(actual, left)
    assert not np.shares_memory(actual, right)
    np.testing.assert_array_equal(actual.view(np.uint64), original.view(np.uint64))
    oracle = np.einsum("ik,kj->ij", left, right, optimize=False)
    scale = np.einsum("ik,kj->ij", np.abs(left), np.abs(right), optimize=False)
    # Original independent finite dot-product envelope, unchanged. Exact
    # current bytes above is a separate additional preservation check.
    eps = np.finfo(np.float64).eps
    k = left.shape[1]
    bound = 16 * eps * max(k, 1) * scale + 16 * k * np.nextafter(0.0, 1.0)
    assert np.isfinite(actual).all() and np.isfinite(oracle).all()
    assert np.all(np.abs(actual - oracle) <= bound)


class RoleProbe:
    def __init__(self, *, fail_call=None, failure=RuntimeError):
        self.records = []
        self.bypasses = []
        self.fail_call = fail_call
        self.failure = failure

    def wrap_factory(self, factory, *, bound):
        probe = self

        def observed(*args):
            role = factory(*args)
            operation = args[-1]
            if role is None or role is operation:
                probe.bypasses.append(role is operation)
                return role
            record = dict(role=weakref.ref(role), buffers=[], calls=[], cleared=0)
            probe.records.append(record)

            class ObservedRole:
                def __init__(self):
                    self.delegate = role

                def __call__(self, *operands):
                    left, right = ((self.delegate._left, operands[0])
                                   if bound else operands)
                    before = state(left), state(right)
                    original = operation(left, right)
                    result = self.delegate(*operands)
                    assert_product(left, right, result, original)
                    assert (state(left), state(right)) == before
                    for value in getattr(self.delegate, "_buffers", ()):
                        if value is not None:
                            record["buffers"].append(weakref.ref(value))
                    value = getattr(self.delegate, "_left_buffer", None)
                    if value is not None:
                        record["buffers"].append(weakref.ref(value))
                    record["calls"].append(dict(shape=(left.shape, right.shape),
                                                left=digest(left), right=digest(right)))
                    if len(record["calls"]) == probe.fail_call:
                        # Simulate failure at the caller boundary after real
                        # allocation/native work. Kernel injection coverage is
                        # supplied independently by its existing qualification.
                        raise probe.failure("injected caller role failure")
                    return result

                def clear(self):
                    self.delegate.clear()
                    record["cleared"] += 1
                    assert all(value is None for value in getattr(self.delegate, "_buffers", ()))
                    assert getattr(self.delegate, "_left_buffer", None) is None

            return ObservedRole()

        return observed

    def assert_released(self):
        gc.collect()
        assert self.records
        assert all(record["cleared"] == 1 for record in self.records)
        assert all(record["role"]() is None for record in self.records)
        assert all(ref() is None for record in self.records for ref in record["buffers"])
