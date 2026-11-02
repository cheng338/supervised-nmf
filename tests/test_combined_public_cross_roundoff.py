"""Coupled D27/T-ADD contracts on the selected public native route.

AOCL products enter the package's protected helper. MKL keeps its ordinary
selector. The observer records native arguments before the call; it does not
substitute arithmetic, a selector, or a passive solve.
"""
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from supervised_nmf import _aocl, _nnls

assert Path(_nnls.__file__).resolve() == ROOT / "src/supervised_nmf/_nnls.py"
assert Path(_aocl.__file__).resolve() == ROOT / "src/supervised_nmf/_aocl.py"


def snapshot(value):
    return (value.shape, value.strides, value.dtype.str, value.flags.writeable,
            value.tobytes(order="C"))


def independent_product(left, right):
    return np.einsum("ik,kj->ij", left, right, optimize=False)


def target_layout(values, kind):
    if kind == "strided":
        storage = np.empty((2 * values.shape[0], 2 * values.shape[1]))
        result = storage[::2, ::2]
        result[:] = values
    else:
        result = np.array(values, order="C", copy=True)
    result.setflags(write=False)
    return result


@pytest.fixture
def native_route(monkeypatch):
    with threadpool_limits(limits=1):
        selected = _nnls.nnls_matmul()
        assert selected in (np.matmul, _aocl.padded_matmul)
        native = []
        if selected is _aocl.padded_matmul:
            class NativeObserver:
                def __getattr__(self, name):
                    return getattr(np, name)

                def matmul(self, left, right):
                    before = snapshot(left), snapshot(right)
                    # This is reached only through the selected padded helper.
                    assert left.shape[0] % 8 == right.shape[1] % 8 == 0
                    result = np.matmul(left, right)
                    assert before == (snapshot(left), snapshot(right))
                    native.append(dict(left=left, right=right, before=before))
                    return result

            monkeypatch.setattr(_aocl, "np", NativeObserver())
        assert _nnls.nnls_matmul() is selected
        yield selected, native
        assert _nnls.nnls_matmul() is selected


def check_native_route(selected, calls, left, right):
    if selected is np.matmul:
        assert calls == []
        return False
    m, k = left.shape
    n = right.shape[1]
    split = (m >= 8 and n >= 8 and m % 8 and n % 8
             and type(left) is np.ndarray and type(right) is np.ndarray
             and left.dtype == right.dtype == np.float64
             and left.flags.aligned and right.flags.aligned
             and right.flags.c_contiguous and right.strides == (n * 8, 8))
    assert len(calls) == (2 if split else 1)
    assert all(call["left"].shape[1] == call["right"].shape[0] == k for call in calls)
    if split:
        core, fringe = calls
        assert core["left"] is fringe["left"]
        assert core["right"].shape == (k, n & ~7)
        assert core["right"].strides == right.strides
        assert np.shares_memory(core["right"], right)
        assert core["before"][1] == snapshot(right[:, :n & ~7])
        assert not np.shares_memory(fringe["right"], right)
    return bool(split)


def observed_qualification(monkeypatch, qualified, route, design, targets,
                           coefficients, cross):
    """Observe only qualification; preserve the already selected callable."""
    selected, native = route
    products, absolute = [], []
    inputs = [design, targets, coefficients] + ([] if cross is None else [cross])
    before = [snapshot(value) for value in inputs]

    class AbsoluteObserver:
        def __getattr__(self, name):
            return getattr(np, name)

        def abs(self, value, *args, **kwargs):
            result = np.abs(value, *args, **kwargs)
            absolute.append(dict(input=value, result=result, initial=result.copy()))
            return result

    def operation(left, right):
        first = len(native)
        operand_bytes = snapshot(left), snapshot(right)
        result = selected(left, right)
        split = check_native_route(selected, native[first:], left, right)
        products.append(dict(left=left, right=right, result=result,
                             operands=operand_bytes, result_before=snapshot(result),
                             split=split))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(_nnls, "np", AbsoluteObserver())
        mask = qualified(design, targets, coefficients, matmul=operation,
                         cross_scale=cross)
    assert [snapshot(value) for value in inputs] == before
    assert len(products) == (4 if cross is None else 3)
    fitted, gradient = products[:2]
    roundoff = products[-1]
    assert fitted["left"] is design and fitted["right"] is coefficients
    assert snapshot(fitted["result"]) == fitted["result_before"]
    assert gradient["right"].tobytes() == (fitted["result"] - targets).tobytes()
    fitted_abs = [row for row in absolute if row["input"] is fitted["result"]]
    target_abs = [row for row in absolute if row["input"] is targets]
    assert len(fitted_abs) == 1 and len(target_abs) == (2 if cross is None else 1)
    owned = fitted_abs[0]["result"]
    assert owned is roundoff["right"] and owned.flags.owndata and owned.base is None
    assert owned.flags.writeable and owned.flags.c_contiguous
    assert owned.tobytes() == (fitted_abs[0]["initial"] + target_abs[-1]["result"]).tobytes()
    for value in inputs + [fitted["result"], target_abs[-1]["result"]]:
        assert not np.shares_memory(owned, value)
    for row in target_abs:
        assert row["result"].flags.owndata
        assert row["result"].tobytes() == row["initial"].tobytes()
    for row in products:
        assert row["operands"] == (snapshot(row["left"]), snapshot(row["right"]))
        assert row["result_before"] == snapshot(row["result"])
    # These fixtures make the *consumed absolute buffer* a CN right operand.
    if selected is _aocl.padded_matmul:
        assert fitted["split"] and gradient["split"] and roundoff["split"]
    return mask, products


@pytest.mark.parametrize("height,rank,rhs", [(37, 9, 17), (61, 20, 21)])
@pytest.mark.parametrize("kind", ["c", "strided"])
def test_native_dense_cross_roundoff_and_near_kkt_interaction(
        monkeypatch, native_route, height, rank, rhs, kind):
    rng = np.random.default_rng(730127 + rank)
    design = np.array(rng.uniform(0.0, .25, (height, rank)) + np.eye(height, rank), order="C")
    truth = rng.uniform(.25, 1., (rank, rhs))
    targets = target_layout(independent_product(design, truth), kind)
    design.setflags(write=False)
    selected, native = native_route
    first = len(native)
    cross = selected(design.T, targets)
    cached_split = check_native_route(selected, native[first:], design.T, targets)
    absolute_targets = np.abs(targets)
    first = len(native)
    recomputed = selected(design.T, absolute_targets)
    recomputed_split = check_native_route(selected, native[first:], design.T, absolute_targets)
    assert cross.tobytes() == recomputed.tobytes()
    if selected is _aocl.padded_matmul:
        assert cached_split == (kind == "c") and recomputed_split
    cross.setflags(write=False)
    masks = []
    for delta in (0., -1e-12, 1e-14, 1e-12, 1e-8):
        coefficients = truth.copy()
        coefficients[0] += delta
        default, _ = observed_qualification(monkeypatch, _nnls._qualified_rows,
            native_route, design, targets, coefficients, None)
        cached, _ = observed_qualification(monkeypatch, _nnls._qualified_rows,
            native_route, design, targets, coefficients, cross)
        assert cached.tobytes() == default.tobytes()
        masks.append(cached)
    assert masks[0].all() and not masks[-1].any()


@pytest.mark.parametrize("rank,rhs", [(9, 17), (20, 21)])
def test_native_large_rank_zero_and_representability_boundaries(
        monkeypatch, native_route, rank, rhs):
    design = np.eye(2 * rank + 1, rank)
    selected, native = native_route
    for value in (-0., np.nextafter(np.finfo(float).tiny, 0.),
                  np.finfo(float).tiny, np.nextafter(np.finfo(float).tiny, np.inf)):
        targets = target_layout(np.full((len(design), rhs), value), "strided")
        cross = selected(design.T, targets)
        assert cross.tobytes() == selected(design.T, np.abs(targets)).tobytes()
        for coefficients in (np.zeros((rank, rhs)), np.full((rank, rhs), value),
                             np.full((rank, rhs), -np.finfo(float).tiny)):
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                default, _ = observed_qualification(monkeypatch, _nnls._qualified_rows,
                    native_route, design, targets, coefficients, None)
                cached, _ = observed_qualification(monkeypatch, _nnls._qualified_rows,
                    native_route, design, targets, coefficients, cross)
            assert cached.tobytes() == default.tobytes()


@pytest.mark.parametrize("height,rank,batch_size", [(37, 9, 17), (61, 20, 21)])
@pytest.mark.parametrize("layout", ["canonical-full", "c-rows-tail", "strided-target-tail"])
def test_actual_public_full_and_tail_blocks_keep_cross_masks_and_fallback_exact(
        monkeypatch, native_route, height, rank, batch_size, layout):
    rng = np.random.default_rng(51273 + rank)
    design = np.array(rng.uniform(0., .25, (height, rank)) + np.eye(height, rank), order="C")
    count = 37 if layout == "canonical-full" else 2 * batch_size + 9
    truth = rng.uniform(.25, 1., (rank, count))
    targets = independent_product(design, truth)
    if layout == "c-rows-tail":
        rows = np.array(targets.T, order="C", copy=True)
    else:
        rows = target_layout(targets, "strided" if layout == "strided-target-tail" else "c").T
    if layout == "canonical-full":
        batch_size = count
    design.setflags(write=False); rows.setflags(write=False)
    inputs = snapshot(design), snapshot(rows)
    batch, qualified, scipy = _nnls._batched_block, _nnls._qualified_rows, _nnls._scipy_rows
    selected, native = native_route

    def run(recompute):
        batches, admissions, fallback = [], [], []

        def solve(a, b, gram, cap, *, matmul):
            assert cap == 3 * rank and matmul is selected
            result = batch(a, b, gram, cap, matmul=matmul)
            assert len(result) == 3
            batches.append((a, b, result[2], snapshot(result[2])))
            return result

        def qualify(a, b, coefficients, *, matmul, cross_scale=None):
            aa, bb, cross, before = batches[-1]
            assert a is aa and b is bb and cross_scale is cross and matmul is selected
            assert snapshot(cross) == before
            expected_cross = selected(a.T, np.abs(b))
            assert cross.tobytes() == expected_cross.tobytes()
            if layout == "canonical-full":
                assert b.flags.c_contiguous and b.strides == (count * 8, 8)
            elif layout == "strided-target-tail":
                assert not b.flags.c_contiguous and not b.flags.f_contiguous
                assert np.abs(b).flags.c_contiguous
            else:
                assert b.flags.f_contiguous and not b.flags.c_contiguous
            mask, products = observed_qualification(monkeypatch, qualified,
                native_route, a, b, coefficients, None if recompute else cross)
            assert snapshot(cross) == before
            admissions.append((snapshot(a), snapshot(b), mask.tobytes(), snapshot(cross)))
            assert products[-1]["right"].shape == b.shape
            return mask

        def reference(a, b, cap):
            fallback.append((snapshot(a), snapshot(b), cap))
            return scipy(a, b, cap)

        with monkeypatch.context() as patch:
            patch.setattr(_nnls, "_batched_block", solve)
            patch.setattr(_nnls, "_qualified_rows", qualify)
            patch.setattr(_nnls, "_scipy_rows", reference)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _nnls.project_nnls(design, rows, solver="auto",
                                             maxiter=None, batch_size=batch_size)
        assert [b.shape[1] for _, b, _, _ in batches] == (
            [count] if layout == "canonical-full" else [batch_size, batch_size, 9])
        assert len({id(cross) for _, _, cross, _ in batches}) == len(batches)
        assert admissions and all(np.frombuffer(row[2], dtype=np.bool_).any() for row in admissions)
        assert all(cap is None for _, _, cap in fallback)
        assert inputs == (snapshot(design), snapshot(rows))
        return (snapshot(result), admissions, fallback,
                [(item.category.__name__, str(item.message)) for item in caught])

    assert run(False) == run(True)


def test_native_larger_rank_scaled_groups_keep_independent_cross_and_roundoff(
        monkeypatch, native_route):
    rank = 9
    design = np.eye(37, rank)
    truth = np.repeat((np.arange(1, rank + 1, dtype=float) / 4)[:, None], 43, axis=1)
    targets = independent_product(design, truth)
    rows = np.concatenate([np.ldexp(targets.T, -700), np.ldexp(targets.T, 700)])
    design.setflags(write=False); rows.setflags(write=False)
    before = snapshot(design), snapshot(rows)
    qualified, batch, scipy = _nnls._qualified_rows, _nnls._batched_block, _nnls._scipy_rows
    selected, _ = native_route

    def run(recompute):
        batches, admissions, fallback = [], [], []

        def solve(a, b, gram, cap, *, matmul):
            assert cap == 3 * rank and matmul is selected
            result = batch(a, b, gram, cap, matmul=matmul)
            batches.append((a, b, result[2], snapshot(result[2])))
            return result

        def qualify(a, b, coefficients, *, matmul, cross_scale=None):
            aa, bb, cross, original = batches[-1]
            assert a is aa and b is bb and cross_scale is cross and matmul is selected
            assert cross.tobytes() == selected(a.T, np.abs(b)).tobytes()
            mask, _ = observed_qualification(monkeypatch, qualified, native_route,
                a, b, coefficients, None if recompute else cross)
            assert snapshot(cross) == original
            admissions.append((snapshot(a), snapshot(b), mask.tobytes(), original))
            return mask

        def reference(a, b, cap):
            fallback.append((snapshot(a), snapshot(b), cap))
            return scipy(a, b, cap)

        with monkeypatch.context() as patch:
            patch.setattr(_nnls, "_batched_block", solve)
            patch.setattr(_nnls, "_qualified_rows", qualify)
            patch.setattr(_nnls, "_scipy_rows", reference)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _nnls.project_nnls(design, rows, solver="auto",
                                             maxiter=None, batch_size=17)
        assert [b.shape[1] for _, b, _, _ in batches] == [17, 17, 9, 17, 17, 9]
        assert len({id(a) for a, _, _, _ in batches}) == 2
        assert len({id(cross) for _, _, cross, _ in batches}) == 6
        assert all(np.frombuffer(item[2], dtype=np.bool_).all() for item in admissions)
        assert np.isfinite(result).all() and np.all(result >= 0)
        assert before == (snapshot(design), snapshot(rows))
        return (snapshot(result), admissions, fallback,
                [(item.category.__name__, str(item.message)) for item in caught])

    assert run(False) == run(True)
