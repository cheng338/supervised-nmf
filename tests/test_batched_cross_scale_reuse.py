"""Cross reuse preserves original-space qualification and public decisions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from supervised_nmf import _nnls

assert Path(_nnls.__file__).resolve() == ROOT / 'src/supervised_nmf/_nnls.py'


def reference(left, right):
    return np.einsum('ik,kj->ij', left, right, optimize=False)


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


def problem(rhs=37):
    design = np.zeros((9, 3))
    design[0, 0] = 1.; design[2, 1] = 2.; design[4, 2] = 4.
    truth = (np.arange(3 * rhs).reshape(3, rhs) % 7).astype(float) / 4
    return design, reference(design, truth), truth


def test_batch_returns_original_immutable_cross_and_reuse_retains_other_products():
    design, targets, truth = problem()
    calls = []

    def operation(left, right):
        result = reference(left, right)
        calls.append((left, right, result, result.tobytes()))
        return result

    gram = reference(design.T, design)
    result, converged, cross = _nnls._batched_block(design, targets, gram, 9, matmul=operation)
    assert cross is calls[0][2] and cross.tobytes() == calls[0][3]
    assert converged.all()
    np.testing.assert_array_equal(result, truth)
    original = design.tobytes(), targets.tobytes(), result.tobytes(), cross.tobytes()
    calls.clear()
    default = _nnls._qualified_rows(design, targets, result, matmul=operation)
    assert len(calls) == 4
    retained = [(a.copy(), b.copy()) for a, b, _, _ in (calls[0], calls[1], calls[3])]
    calls.clear()
    reused = _nnls._qualified_rows(design, targets, result, matmul=operation, cross_scale=cross)
    assert len(calls) == 3 and reused.all()
    np.testing.assert_array_equal(default, reused)
    for (left, right, _, _), (a, b) in zip(calls, retained):
        np.testing.assert_array_equal(left, a); np.testing.assert_array_equal(right, b)
    assert original == (design.tobytes(), targets.tobytes(), result.tobytes(), cross.tobytes())


@pytest.mark.parametrize('kind', ['c', 'f', 'strided', 'negative_stride'])
@pytest.mark.parametrize('value', [0., -0., np.nextafter(np.finfo(float).tiny, 0.),
                                  np.finfo(float).tiny, np.nextafter(np.finfo(float).tiny, np.inf), 1.])
def test_qualification_boundary_and_layout_masks_match_default(kind, value):
    design = np.eye(3)
    targets = np.full((3, 5), value)
    if kind == 'f':
        targets = np.array(targets, order='F')
    elif kind == 'strided':
        storage = np.empty((6, 10)); storage[::2, ::2] = targets; targets = storage[::2, ::2]
    elif kind == 'negative_stride':
        targets = targets[:, ::-1]
    targets.setflags(write=False)
    cross = reference(design.T, targets)
    before = targets.tobytes(), targets.strides, cross.tobytes()
    candidates = [np.zeros((3, 5)), np.array(targets, copy=True),
                  np.full((3, 5), -np.finfo(float).tiny), np.full((3, 5), np.inf)]
    for candidate in candidates:
        default = _nnls._qualified_rows(design, targets, candidate, matmul=reference)
        reused = _nnls._qualified_rows(design, targets, candidate, matmul=reference, cross_scale=cross)
        np.testing.assert_array_equal(default, reused)
    assert before == (targets.tobytes(), targets.strides, cross.tobytes())
    assert not targets.flags.writeable


@pytest.mark.parametrize('solver', ['batched', 'auto'])
def test_public_plumbing_binds_cross_to_each_exact_block(monkeypatch, solver):
    monkeypatch.setattr(_nnls, 'nnls_matmul', lambda: reference)
    design, targets, truth = problem()
    saved = []
    batch, qualify = _nnls._batched_block, _nnls._qualified_rows

    def observe_batch(a, b, gram, cap, **kwargs):
        value = batch(a, b, gram, cap, **kwargs)
        saved.append((a, b, value[2], value[2].tobytes()))
        return value

    def observe_qualification(a, b, coefficients, **kwargs):
        aa, bb, cross, original = saved[-1]
        assert a is aa and b is bb
        assert kwargs['cross_scale'] is cross and cross.tobytes() == original
        return qualify(a, b, coefficients, **kwargs)

    monkeypatch.setattr(_nnls, '_batched_block', observe_batch)
    monkeypatch.setattr(_nnls, '_qualified_rows', observe_qualification)
    result = _nnls.project_nnls(design, targets.T, solver=solver, batch_size=13)
    np.testing.assert_array_equal(result, truth.T)
    assert [b.shape[1] for _, b, _, _ in saved] == [13, 13, 11]
    assert len({id(cross) for _, _, cross, _ in saved}) == 3


def test_successful_qr_keeps_default_original_cross_recomputation(monkeypatch):
    monkeypatch.setattr(_nnls, 'nnls_matmul', lambda: reference)
    design, targets, truth = problem()
    calls = []
    qualify = _nnls._qualified_rows

    def observe(a, b, coefficients, **kwargs):
        calls.append(kwargs.get('cross_scale'))
        assert kwargs.get('cross_scale') is None
        return qualify(a, b, coefficients, **kwargs)

    monkeypatch.setattr(_nnls, '_qualified_rows', observe)
    result = _nnls.project_nnls(design, targets.T, solver='qr', batch_size=13)
    np.testing.assert_allclose(result, truth.T, atol=1e-14, rtol=1e-14)
    assert len(calls) == 3


def test_scaled_groups_do_not_share_cross_objects(monkeypatch):
    monkeypatch.setattr(_nnls, 'nnls_matmul', lambda: reference)
    design, targets, truth = problem()
    # Each row group requires its own common scaling with this ordinary design.
    rows = np.concatenate([np.ldexp(targets.T, -700), np.ldexp(targets.T, 700)])
    seen = []
    qualify = _nnls._qualified_rows

    def observe(a, b, coefficients, **kwargs):
        cross = kwargs['cross_scale']
        np.testing.assert_array_equal(cross, reference(a.T, b))
        seen.append((a, b, cross))  # Keep objects live to prevent allocator-id reuse.
        return qualify(a, b, coefficients, **kwargs)

    monkeypatch.setattr(_nnls, '_qualified_rows', observe)
    result = _nnls.project_nnls(design, rows, solver='batched', batch_size=13)
    assert np.isfinite(result).all() and np.all(result >= 0)
    assert len(seen) >= 2 and len({id(x[2]) for x in seen}) == len(seen)


def test_normal_native_route_preserves_analytic_solution():
    design, targets, truth = problem()
    result = _nnls.project_nnls(design, targets.T, solver='batched', batch_size=13)
    np.testing.assert_array_equal(result, truth.T)


# Native qualification comparisons use the currently selected ordinary backend.
# AOCL receives the existing protected helper; no raw native oracle is added.
def native_layout(values, kind):
    if kind == "f":
        result = np.array(values, order="F", copy=True)
    elif kind == "strided":
        storage = np.empty((values.shape[0] * 2, values.shape[1] * 2))
        result = storage[::2, ::2]
        result[:] = values
    elif kind == "negative_stride":
        result = values[::-1].copy()[::-1]
    else:
        result = np.array(values, order="C", copy=True)
    result.setflags(write=False)
    return result


def identity_record(value):
    return (value.shape, value.strides, value.dtype.str, value.flags.writeable,
            value.tobytes(order="C"))


@pytest.mark.parametrize("kind", ["c", "f", "strided", "negative_stride"])
@pytest.mark.parametrize("value", [
    -0.0, np.nextafter(np.finfo(float).tiny, 0.0),
    np.finfo(float).tiny, np.nextafter(np.finfo(float).tiny, np.inf), 1.0,
])
def test_native_cross_bytes_and_boundary_masks_match_recomputation(kind, value):
    design = np.eye(3)
    targets = native_layout(np.full((3, 5), value), kind)
    product = _nnls.nnls_matmul()
    cross = product(design.T, targets)
    original_cross = product(design.T, np.abs(targets))
    assert cross.tobytes(order="C") == original_cross.tobytes(order="C")
    before = identity_record(targets), identity_record(cross)
    candidates = [
        np.zeros((3, 5)), np.array(targets, copy=True),
        np.full((3, 5), -np.finfo(float).tiny),
        np.full((3, 5), np.nan), np.full((3, 5), np.inf),
    ]
    for candidate in candidates:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            expected = _nnls._qualified_rows(
                design, targets, candidate, matmul=product)
            actual = _nnls._qualified_rows(
                design, targets, candidate, matmul=product, cross_scale=cross)
        assert actual.tobytes() == expected.tobytes()
    assert before == (identity_record(targets), identity_record(cross))


@pytest.mark.parametrize("kind", ["c", "f", "strided", "negative_stride"])
def test_native_dense_rounding_and_near_kkt_admission_match_recomputation(kind):
    rng = np.random.default_rng(730127)
    design = np.array(rng.uniform(0.0, 1.0, (37, 5)), order="C")
    truth = rng.uniform(0.25, 1.0, (5, 17))
    targets = native_layout(reference(design, truth), kind)
    product = _nnls.nnls_matmul()
    cross = product(design.T, targets)
    recomputed = product(design.T, np.abs(targets))
    # This deliberately tests bytes, not an allclose relaxation: layout-induced
    # product rounding is one of the unresolved premises of unrestricted D27.
    assert cross.tobytes(order="C") == recomputed.tobytes(order="C")
    before = identity_record(design), identity_record(targets), identity_record(cross)
    for perturbation in (0.0, -1e-12, 1e-14, 1e-12, 1e-10):
        coefficients = truth.copy()
        coefficients[0] += perturbation
        expected = _nnls._qualified_rows(design, targets, coefficients, matmul=product)
        actual = _nnls._qualified_rows(
            design, targets, coefficients, matmul=product, cross_scale=cross)
        assert actual.tobytes() == expected.tobytes()
    assert before == (identity_record(design), identity_record(targets), identity_record(cross))


def public_comparison(monkeypatch, design, rows, solver, *, maxiter=17, batch_size=13):
    """Same native solver, compare new reuse with its unchanged default check."""
    import warnings
    qualified = _nnls._qualified_rows
    scipy_rows = _nnls._scipy_rows
    before = identity_record(design), identity_record(rows)

    def run(recompute):
        fallback, admissions = [], []

        def qualify(a, b, coefficients, **kwargs):
            supplied = kwargs.get("cross_scale")
            if recompute:
                kwargs.pop("cross_scale", None)
            mask = qualified(a, b, coefficients, **kwargs)
            admissions.append((identity_record(a), identity_record(b), mask.tobytes(),
                               supplied is not None))
            return mask

        def scipy(a, b, cap):
            fallback.append((identity_record(a), identity_record(b), cap))
            return scipy_rows(a, b, cap)

        with monkeypatch.context() as patch:
            patch.setattr(_nnls, "_qualified_rows", qualify)
            patch.setattr(_nnls, "_scipy_rows", scipy)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = _nnls.project_nnls(
                        design, rows, solver=solver, maxiter=maxiter, batch_size=batch_size)
                    outcome = ("result", result.dtype.str, result.shape, result.tobytes())
                except (ValueError, FloatingPointError, RuntimeError) as error:
                    result = None
                    outcome = ("error", type(error).__name__, str(error))
            warning_record = [(item.category.__name__, str(item.message)) for item in caught]
        return result, outcome, fallback, admissions, warning_record

    expected = run(True)
    actual = run(False)
    assert actual[1:] == expected[1:]
    assert before == (identity_record(design), identity_record(rows))
    return actual


@pytest.mark.parametrize("kind", ["c", "f", "strided", "negative_stride"])
@pytest.mark.parametrize("route", ["batched", "auto", "qr", "qr_fallback"])
def test_native_public_result_admission_and_fallback_are_exact(monkeypatch, kind, route):
    design, targets, truth = problem()
    rows = targets.T.copy()
    rows[rows == 0.0] = -0.0
    rows = native_layout(rows, kind)
    design.setflags(write=False)
    solver = route
    if route == "qr_fallback":
        solver = "qr"

        def failed_qr(*args, **kwargs):
            raise np.linalg.LinAlgError("forced QR fallback")

        monkeypatch.setattr(_nnls.linalg, "qr", failed_qr)
    actual = public_comparison(monkeypatch, design, rows, solver)
    assert actual[1][0] == "result"
    np.testing.assert_allclose(actual[0], truth.T, rtol=1e-14, atol=0.0)
    assert actual[3]
    assert all(record[3] == (route != "qr") for record in actual[3])


@pytest.mark.parametrize("exponent", [-700, 700])
@pytest.mark.parametrize("solver", ["batched", "qr"])
def test_native_common_binary_scaling_keeps_exact_reuse_decisions(
        monkeypatch, exponent, solver):
    design, targets, truth = problem()
    design = np.ldexp(design, exponent)
    rows = np.ldexp(targets.T, exponent)
    design.setflags(write=False)
    rows.setflags(write=False)
    actual = public_comparison(monkeypatch, design, rows, solver)
    assert actual[1][0] == "result"
    np.testing.assert_allclose(actual[0], truth.T, rtol=1e-14, atol=0.0)


def test_native_independent_extreme_rhs_groups_use_their_own_cross(monkeypatch):
    design, targets, truth = problem()
    rows = np.concatenate([np.ldexp(targets.T, -700), np.ldexp(targets.T, 700)])
    actual = public_comparison(monkeypatch, design, rows, "batched")
    assert actual[1][0] == "result"
    expected = np.concatenate([np.ldexp(truth.T, -700), np.ldexp(truth.T, 700)])
    np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=0.0)
    assert len(actual[3]) >= 2 and all(record[3] for record in actual[3])


def test_native_restrictive_original_cap_and_reference_error_are_unchanged(monkeypatch):
    design, targets, _ = problem()
    actual = public_comparison(monkeypatch, design, targets.T, "batched", maxiter=1)
    # Backend/SciPy versions may return or raise at this cap; compare their
    # actual contract rather than assuming a fixed iteration implementation.
    assert actual[1][0] in {"result", "error"}
    assert actual[2] and all(record[2] == 1 for record in actual[2])
    assert not actual[3]


def test_forced_reference_error_propagates_with_original_cap(monkeypatch):
    design, targets, _ = problem()

    def fail_reference(a, b, cap):
        assert cap == 1
        raise RuntimeError("reference cap failure marker")

    monkeypatch.setattr(_nnls, "_scipy_rows", fail_reference)
    actual = public_comparison(monkeypatch, design, targets.T, "batched", maxiter=1)
    assert actual[1] == ("error", "RuntimeError", "reference cap failure marker")
    assert actual[2] and all(record[2] == 1 for record in actual[2])
    assert not actual[3]
