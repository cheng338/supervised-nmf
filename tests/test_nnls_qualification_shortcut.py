"""Original-space qualification must preserve the solver's SciPy retries."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supervised_nmf import _aocl, _nnls

assert Path(_nnls.__file__).resolve() == ROOT / "src/supervised_nmf/_nnls.py"


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def full_qualification(design, targets, coefficients, *, matmul, cross_scale=None):
    """Always evaluate the original roundoff product on the same backend."""
    feasible = np.isfinite(coefficients).all(axis=0) & (coefficients >= 0).all(axis=0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        fitted = matmul(design, coefficients)
        residual = fitted - targets
        gradient = matmul(design.T, residual)
        if cross_scale is None:
            cross_scale = matmul(design.T, np.abs(targets))
        roundoff_input = np.abs(fitted)
        roundoff_input += np.abs(targets)
        roundoff = matmul(design.T, roundoff_input)
        tolerance = 1e-12 * cross_scale + 64 * np.finfo(float).eps * max(design.shape) * roundoff
        projected = np.where((coefficients > 0) | (gradient < 0), gradient, 0.0)
        finite = (np.isfinite(residual).all(axis=0) & np.isfinite(gradient).all(axis=0)
                  & np.isfinite(tolerance).all(axis=0))
        representable = ~np.any(targets != 0, axis=0) | np.any(cross_scale >= np.finfo(float).tiny, axis=0)
        return feasible & finite & (np.abs(projected) <= tolerance).all(axis=0) & representable


@pytest.mark.parametrize("supply_cross", [False, True])
@pytest.mark.parametrize("uncertain", [False, True])
def test_mixed_block_skips_or_retains_one_full_width_product(supply_cross, uncertain):
    design = np.ones((1, 1))
    # Negative, exact, wider-tolerance-only, and outside-both-tolerances cases.
    coefficients = np.array([[-0.25, 1.0, 1.0 + 1.02e-12, 1.0 + 4e-12]])
    if not uncertain:
        coefficients = coefficients[:, :2]
    targets = np.ones_like(coefficients)
    product = _aocl.padded_matmul
    cross = product(design.T, targets) if supply_cross else None
    expected = full_qualification(design, targets, coefficients, matmul=product, cross_scale=cross)
    np.testing.assert_array_equal(expected, [False, True, True, False][:targets.shape[1]])

    # Count calls to the real helper. A wrapper passed as matmul would disable the shortcut.
    shapes = []

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is product.__code__:
            shapes.append(frame.f_locals["right"].shape)

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        actual = _nnls._qualified_rows(design, targets, coefficients, matmul=product, cross_scale=cross)
    finally:
        sys.setprofile(previous)
    np.testing.assert_array_equal(actual, expected)
    assert shapes == [targets.shape] * (2 + int(not supply_cross) + int(uncertain))


@pytest.mark.parametrize("supply_cross", [False, True])
def test_native_representability_and_finite_roundoff(supply_cross):
    tiny = np.finfo(float).tiny
    cases = [
        (1.0, [0.0, tiny / 2, tiny], [0.0, tiny / 2, tiny], [True, False, True]),
        (2.0**-600, [0.0, 2.0**-600], [0.0, 0.0], [True, False]),
        (0.75, [1.125 * 2.0**1023], [1.5 * 2.0**1023], [False]),
    ]
    product = _nnls.nnls_matmul()
    for value, rhs, candidate, expected in cases:
        design, targets, coefficients = np.array([[value]]), np.array([rhs]), np.array([candidate])
        with np.errstate(under="ignore"):
            cross = product(design.T, targets) if supply_cross else None
        actual = _nnls._qualified_rows(design, targets, coefficients, matmul=product, cross_scale=cross)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("height,rank,rhs", [(0, 2, 3), (2, 0, 3), (2, 2, 0), (0, 0, 0)])
def test_native_empty_masks_keep_their_shape(height, rank, rhs):
    design, targets, coefficients = np.zeros((height, rank)), np.zeros((height, rhs)), np.zeros((rank, rhs))
    actual = _nnls._qualified_rows(design, targets, coefficients, matmul=_nnls.nnls_matmul())
    np.testing.assert_array_equal(actual, np.ones(rhs, dtype=bool), strict=True)


def test_subclass_reductions_and_masked_targets_keep_full_check_behavior():
    class ObservedDesign(np.ndarray):
        def min(self, *args, **kwargs):
            raise AssertionError("unexpected design minimum")

    one = np.ones((1, 1))
    product = _nnls.nnls_matmul()
    for design, targets in [(one.view(ObservedDesign), one), (one, np.ma.array(one, mask=True))]:
        actual = _nnls._qualified_rows(design, targets, one, matmul=product, cross_scale=one)
        expected = full_qualification(design, targets, one, matmul=product, cross_scale=one)
        assert type(actual) is type(expected)
        assert actual.tolist() == expected.tolist()  # Masked positions appear as None.


@pytest.mark.parametrize("installation", ["after-import", "before-import", "before-solver-reload"])
def test_replaced_product_still_raises_on_roundoff(monkeypatch, installation):
    one = np.ones((1, 1))
    native = _aocl.padded_matmul if installation == "before-solver-reload" else np.matmul
    calls = []

    def fresh_module(module):
        spec = spec_from_file_location(module.__name__, module.__file__)
        loaded = module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module.__name__, loaded)
        spec.loader.exec_module(loaded)
        return loaded

    def observed(left, right):
        calls.append(right.copy())
        if len(calls) == 3:
            raise RuntimeError("observed roundoff product")
        return native(left, right)

    helper, solver = _aocl, _nnls
    if installation == "before-solver-reload":
        monkeypatch.setattr(helper, "padded_matmul", observed)
        solver = fresh_module(solver)
    else:
        monkeypatch.setattr(np, "matmul", observed)
        if installation == "before-import":
            helper = fresh_module(helper)
            solver = fresh_module(solver)
    with pytest.raises(RuntimeError, match="observed roundoff product"):
        solver._qualified_rows(one, one, one, matmul=helper.padded_matmul, cross_scale=one)
    assert [right[0, 0] for right in calls] == [1.0, 0.0, 2.0]


# Saved ordinary solver inputs. A one-ulp change in a weak target can change
# the SciPy retry, so preserve the nonzero entries of each 16-by-3 design exactly.
@pytest.fixture(params=["aocl", "mkl"])
def boundary_problem(request):
    if request.param == "aocl":
        entries = ("0x1.432410b636c8ep+0", "0x1.7e385e0700774p+0", "0x1.f277b7282fec0p-1",
                   "0x1.cee9283f5fa14p+0", "0x1.ddb7db6f4c530p-1", "0x1.835f371a92ee6p+0")
        start, exponent, sign, offset = 9368, -68, 1, 166
        weak = "0x1.74891c19913c1p-16"
    else:
        entries = ("0x1.ce18001a3377fp+0", "0x1.209448b77139ap+0", "0x1.28eb81f95f047p+0",
                   "0x1.7fd00c74a7c7ap+0", "0x1.aac9b84d6c2d5p-1", "0x1.28064ae08a292p+0")
        start, exponent, sign, offset = 7576, -67, -1, 22
        weak = "0x1.28e332db2f42dp-15"
    a, b, c, d, e, strong = map(float.fromhex, entries)
    design = np.zeros((16, 3))
    design[:3] = [[a, b, 0.0], [c, 0.0, d], [e, 0.0, 0.0]]
    rows = np.zeros((256, 16))
    # Reconstruct the distinct neighboring targets in their original batch order.
    h = 64 * np.finfo(float).eps * len(design)
    centers = np.arange(start, start + len(rows)) * 2.0**exponent * (1 - sign * h) / (1e-12 + 2 * h)
    rows[:, 0] = centers if sign > 0 else np.nextafter(centers, 0.0)
    rows[:, 1] = strong
    rows[:, 2] = 0.2 * rows[:, 0] * e / a
    assert rows[offset, 0] == float.fromhex(weak)
    return design, rows, offset


@pytest.mark.parametrize("repeated", [False, True], ids=["distinct-256", "repeated-32"])
def test_solver_boundary_preserves_retry_rows_and_coefficient_bytes(monkeypatch, boundary_problem, repeated):
    design, rows, offset = boundary_problem
    if repeated:
        rows = np.tile(rows[offset], (32, 1))
    scipy_rows = _nnls._scipy_rows

    def run(qualify):
        masks, retried = [], []

        def check(*args, **kwargs):
            accepted = qualify(*args, **kwargs)
            masks.append(accepted.tobytes())
            return accepted

        def scipy(a, rejected_rows, cap):
            # Preserve membership and order. Every weak target in the 256-row batch is unique.
            retried.extend(row.tobytes() for row in rejected_rows)
            return scipy_rows(a, rejected_rows, cap)

        with monkeypatch.context() as patch:
            patch.setattr(_nnls, "_qualified_rows", check)
            patch.setattr(_nnls, "_scipy_rows", scipy)
            result = _nnls.project_nnls(design, rows, solver="auto")
        assert len(masks) == 1
        return masks, retried, result.tobytes()

    assert run(_nnls._qualified_rows) == run(full_qualification)
