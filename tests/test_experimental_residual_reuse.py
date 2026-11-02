"""Final experimental NNLS residual consumes its fresh product safely.

Native products use the ordinary selected MKL or protected AOCL route. Small
independent arithmetic uses einsum without optimization, never unprotected
AOCL GEMM. Differential full-history replay belongs to external qualification.
"""
import gc
import math
from pathlib import Path
import sys
import warnings
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import experimental_nmf as experimental
from supervised_nmf import _aocl

assert Path(experimental.__file__).resolve() == ROOT / "scripts/experimental_nmf.py"
assert Path(_aocl.__file__).resolve() == ROOT / "src/supervised_nmf/_aocl.py"


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        selected = experimental.nnls_matmul()
        assert selected is np.matmul or selected is _aocl.padded_matmul
        yield


def bits(value):
    return np.asarray(value).tobytes(order="C")


def input_state(value):
    return (type(value), value.dtype, value.shape, value.strides,
            value.flags.writeable, value.flags.aligned, bits(value))


def reference(left, right):
    return np.einsum("ik,kj->ij", left, right, optimize=False)


def layout(values, kind):
    if kind == "f":
        result = np.array(values, dtype=np.float64, order="F", copy=True)
    elif kind == "strided":
        storage = np.empty((values.shape[0] * 2, values.shape[1] * 2))
        result = storage[::2, ::2]
        result[:] = values
    else:
        result = np.array(values, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def analytic_problem(m, rank, rhs, kind):
    # Disjoint signed column supports give a known unique NNLS solution:
    # clip the exact unconstrained signed coefficients at zero.
    design = np.zeros((m, rank))
    for row in range(m):
        design[row, row % rank] = (1 if row % 2 else -1) * (1 + (row % 3) / 4)
    signed = (np.arange(rank * rhs).reshape(rank, rhs) % 7 - 2) / 4
    targets = reference(design, signed)
    return layout(design, kind), layout(targets, kind), np.maximum(signed, 0)


def record_solve(monkeypatch, A, B, *, init=None, cap=None, prepared=False):
    """Record actual supported products; no output alias or solver stub."""
    selected = experimental.nnls_matmul()
    events = []

    def product(left, right):
        output = selected(left, right)
        assert type(output) is np.ndarray and output.dtype == np.float64
        assert output.flags.owndata and output.flags.writeable and output.flags.c_contiguous
        assert not np.shares_memory(output, left) and not np.shares_memory(output, right)
        events.append(dict(left=left, right=right, output=output,
                           left_before=left.copy(), right_before=right.copy(),
                           output_before=output.copy()))
        return output

    before = (input_state(A), input_state(B), None if init is None else input_state(init))
    with monkeypatch.context() as patch:
        # Preserve the item6 identity guard when the native selector is padded.
        if selected is _aocl.padded_matmul:
            patch.setattr(_aocl, "padded_matmul", product)
            patch.setattr(experimental, "padded_matmul", product)
        patch.setattr(experimental, "nnls_matmul", lambda: product)
        role = experimental._FixedTrainingRHS(B) if prepared else None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            answer = (role.solve(A, init=init, max_iter=cap) if role is not None
                      else experimental.nnls_block_pivot(A, B, init=init, max_iter=cap))
        role_state = None
        if role is not None and role._prepared is not None:
            cached = role._prepared._right
            role_state = dict(object=role, cache=cached, bytes=bits(cached))
    assert before == (input_state(A), input_state(B), None if init is None else input_state(init))
    return dict(answer=answer, events=events, role_state=role_state,
                warnings=[(row.category.__name__, str(row.message)) for row in caught])


def check_residual_contract(record, A, B):
    coefficients, info = record["answer"]
    final_product, final_gradient = record["events"][-2:]
    residual = final_product["output"]
    assert final_product["left"].shape == A.shape
    assert final_product["right"] is coefficients
    assert final_gradient["right"] is residual
    expected = np.subtract(final_product["output_before"], np.asarray(B, dtype=np.float64))
    assert bits(residual) == bits(expected)
    assert residual.flags.owndata and residual.flags.writeable and residual.flags.c_contiguous
    for value in (A, B, coefficients):
        assert not np.shares_memory(residual, value)
    for event in record["events"]:
        if event is final_product:
            continue
        assert bits(event["output"]) == bits(event["output_before"])
        assert not np.shares_memory(residual, event["output"])
    if record["role_state"] is not None:
        cache = record["role_state"]["cache"]
        assert bits(cache) == record["role_state"]["bytes"]
        assert not np.shares_memory(residual, cache)

    # Independent original-design arithmetic, with a scalar residual norm.
    independent_residual = reference(A, coefficients) - B
    independent_gradient = reference(A.T, independent_residual)
    np.testing.assert_allclose(residual, independent_residual, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(final_gradient["output"], independent_gradient,
                               rtol=1e-11, atol=1e-12)
    residual_norm = math.sqrt(math.fsum(float(x) ** 2 for x in independent_residual.ravel()))
    assert info["residual_norm"] == pytest.approx(residual_norm, rel=1e-12, abs=1e-13)
    dual = float(max(0., -independent_gradient.min()))
    complementarity = float(np.max(np.abs(coefficients * independent_gradient)))
    assert info["dual_violation"] == pytest.approx(dual, rel=1e-11, abs=1e-12)
    assert info["complementarity_max"] == pytest.approx(complementarity, rel=1e-11, abs=1e-12)
    assert info["primal_violation"] == 0 and np.isfinite(coefficients).all()
    assert (coefficients >= 0).all()


@pytest.mark.parametrize("m,rank,rhs", [(8, 3, 8), (9, 3, 8), (8, 3, 9),
                                       (9, 3, 9), (1, 1, 1)])
@pytest.mark.parametrize("kind", ["c", "f", "strided"])
@pytest.mark.parametrize("warm", [False, True])
def test_final_residual_consumes_only_fresh_product(monkeypatch, m, rank, rhs, kind, warm):
    A, B, expected = analytic_problem(m, rank, rhs, kind)
    init = layout(expected, kind) if warm else None
    record = record_solve(monkeypatch, A, B, init=init)
    check_residual_contract(record, A, B)
    coefficients, info = record["answer"]
    np.testing.assert_allclose(coefficients, expected, rtol=1e-11, atol=1e-13)
    assert info["success"] and info["projected_gradient_relative"] <= info["kkt_tolerance"]
    assert not record["warnings"]


def test_capped_solve_keeps_failure_info_warning_and_inputs(monkeypatch):
    rng = np.random.RandomState(10)
    A, B = layout(rng.normal(size=(20, 9)), "f"), layout(rng.normal(size=(20, 8)), "strided")
    record = record_solve(monkeypatch, A, B, cap=1)
    check_residual_contract(record, A, B)
    info = record["answer"][1]
    assert not info["success"] and info["max_iter"] == info["n_iter"] == 1
    assert any(category == "ConvergenceWarning" and
               message.startswith("ANLS-BPP inner NNLS failed its iteration/KKT qualification;")
               for category, message in record["warnings"])


def test_prepared_target_is_not_residual_storage(monkeypatch):
    A, B, init = analytic_problem(9, 3, 8, "c")
    record = record_solve(monkeypatch, A, B, init=init, prepared=True)
    check_residual_contract(record, A, B)
    selected = experimental.nnls_matmul()
    assert (record["role_state"] is not None) is (selected is not np.matmul)
    refs = [weakref.ref(record["events"][-2]["output"])]
    if record["role_state"] is not None:
        refs.append(weakref.ref(record["role_state"]["cache"]))
        role = record["role_state"]["object"]
        cache = record["role_state"]["cache"]
        saved = bits(cache)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coefficients, info = role.solve(A * 2, init=init / 2)
        assert info["success"]
        assert role._prepared._right is cache and bits(cache) == saved
        assert not np.shares_memory(coefficients, cache)
        np.testing.assert_allclose(coefficients, init / 2, rtol=1e-11, atol=1e-13)
        del role, cache, coefficients, info
    del record
    gc.collect()
    assert all(ref() is None for ref in refs)


def edge_operands():
    tiny = np.nextafter(np.float64(0), np.float64(1))
    normal = np.finfo(np.float64).tiny
    large = np.finfo(np.float64).max / 4
    return np.array([[0., -0., 0., -0., 1., -1., tiny, -tiny],
                     [normal, -normal, large, -large, 1., 1., tiny * 2, -tiny * 2]]), np.array([
        [0., 0., -0., -0., 1., -1., -tiny, tiny],
        [tiny, -tiny, -large, large, np.nextafter(1., 0.), np.nextafter(1., 2.), tiny, -tiny]])


def test_subtraction_boundary_exact_bytes_and_signed_zero():
    # Ufunc boundary only: no extreme native product or solver-domain widening.
    product, target = edge_operands()
    before = bits(product), bits(target)
    expected = product - target
    actual = product.copy(order="C")
    actual -= target
    assert bits(actual) == bits(expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    assert (bits(product), bits(target)) == before


def test_subtraction_overflow_preserves_caller_values():
    product = np.array([[np.finfo(float).max]])
    target = np.array([[-np.finfo(float).max]])
    before = bits(product), bits(target)
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError) as old:
            product - target
        actual = product.copy()
        with pytest.raises(FloatingPointError) as new:
            actual -= target
    assert str(new.value) == str(old.value)
    assert (bits(product), bits(target)) == before


@pytest.mark.parametrize("bad", ["nonfinite", "wrong_rows", "overflow_products"])
def test_pre_residual_failure_keeps_caller_inputs(bad):
    A, B, _ = analytic_problem(9, 3, 8, "c")
    A, B = A.copy(), B.copy()
    if bad == "nonfinite":
        A[0, 0] = np.nan
        message = "NaN"
    elif bad == "wrong_rows":
        B = B[:-1].copy()
        message = "equal row counts"
    else:
        A[:] = np.finfo(float).max / 4
        message = "NNLS products overflowed"
    A.setflags(write=False)
    B.setflags(write=False)
    before = input_state(A), input_state(B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(over="warn", invalid="warn"):
            with pytest.raises(ValueError, match=message):
                experimental.nnls_block_pivot(A, B)
    assert (input_state(A), input_state(B)) == before
