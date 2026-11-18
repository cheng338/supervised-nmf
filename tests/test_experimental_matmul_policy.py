"""Experimental products follow the selected backend policy without mutation.

Routing checks use reference arithmetic and mocked backend metadata. Baseline
trees fail the source guard before numerical test bodies execute.
"""

import ast
from collections import Counter
import dis
import inspect
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import experimental_nmf as experimental
from supervised_nmf import _aocl, _projection


assert Path(experimental.__file__).resolve() == ROOT / "scripts" / "experimental_nmf.py"
assert Path(_aocl.__file__).resolve() == ROOT / "src" / "supervised_nmf" / "_aocl.py"
SOURCE_TREE = ast.parse(Path(experimental.__file__).read_text(encoding="utf-8"))


def reference_product(left, right):
    return np.einsum("ik,kj->ij", left, right, optimize=False)


def source_product_sites():
    sites = {}
    # The three outer role aliases select either the original callable or its
    # scoped output workspace. Retain original22 fallback sites plus the final-report opt-in, the three
    # consumer sites, and the two quiet-only helper sites as distinct routes.
    # The private residual helper is the diagnostic reconstruction role;
    # its protected native branches have separate operand/storage tests.
    product_names = {"matmul", "outer_reconstruction", "outer_gradient_w", "outer_gradient_h", "_diagnostic_residual"}
    for node in ast.walk(SOURCE_TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if (isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name)
                        and child.func.id in product_names):
                    position = (child.lineno, child.col_offset,
                                child.end_lineno, child.end_col_offset)
                    sites[position] = node.name
    return sites


def handoff_consumer_product_sites():
    """Identify only the three products in the capsule-present PG branch."""
    fit = next(node for node in ast.walk(SOURCE_TREE)
               if isinstance(node, ast.FunctionDef) and node.name == "fit_transform")
    branches = [node for node in ast.walk(fit)
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "_handoff is None or _handoff[3] is None"]
    assert len(branches) == 1
    calls = sorted((child for statement in branches[0].orelse
                    for child in ast.walk(statement)
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id in {"outer_gradient_w", "outer_gradient_h"}),
                   key=lambda child: (child.lineno, child.col_offset))
    assert [child.func.id for child in calls] == [
        "outer_gradient_w", "outer_gradient_w", "outer_gradient_h"]
    return tuple((child.lineno, child.col_offset, child.end_lineno, child.end_col_offset)
                 for child in calls)


def quiet_helper_product_sites():
    """The duplicated quiet tail has two sites; it is not the ordinary route."""
    return {site for site, owner in source_product_sites().items()
            if owner == "_nnls_final_state"}


@pytest.fixture(scope="module", autouse=True)
def candidate_source_guard():
    # In particular, never execute a baseline's unprotected products on AOCL.
    assert callable(getattr(experimental, "nnls_matmul", None)), "Unrouted source tree"
    assert not any(isinstance(node, ast.MatMult) for node in ast.walk(SOURCE_TREE)), (
        "Unprotected matrix-product operator remains in the candidate"
    )


@pytest.fixture(autouse=True)
def one_native_thread(candidate_source_guard):
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def reference_dispatch(monkeypatch):
    selections = []

    def select():
        selections.append(True)
        return reference_product

    monkeypatch.setattr(experimental, "nnls_matmul", select)
    return selections


def analytic_problem():
    # Orthogonal, signed columns: the constrained solution clips the exact
    # unconstrained coordinates. This does not rely on another native solver.
    design = np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0],
                       [0, -2, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    signed_coordinates = np.array([[1, -2, 0, .5], [2, 0, -1, 3],
                                   [0, 1, 2, -1]], dtype=float)
    targets = reference_product(design, signed_coordinates)
    return design, targets, np.maximum(signed_coordinates, 0)


def layout_copy(values, layout):
    if layout == "fortran":
        result = np.array(values, order="F", copy=True)
    elif layout == "strided":
        storage = np.zeros((2 * values.shape[0], 2 * values.shape[1]))
        result = storage[::2, ::2]
        result[:] = values
    else:
        result = np.array(values, order="C", copy=True)
    result.setflags(write=False)
    return result


def unit_initializer(data, rank, **kwargs):
    assert rank == 1
    return np.ones((len(data), 1)), np.ones((1, data.shape[1]))


def test_all_source_products_use_the_selected_callable():
    sites = source_product_sites()
    assert len(sites) == 28
    assert len(quiet_helper_product_sites()) == 2
    assert len(set(sites) - set(handoff_consumer_product_sites())
               - quiet_helper_product_sites()) == 23
    assert Counter(sites.values()) == {
        # Ordinary/replay products remain in the original core. The helper
        # duplicates two quiet-only products; three consumer sites are separate.
        "projected_gradient_kkt": 5, "_nnls_block_pivot": 6,
        "_nnls_final_state": 2, "fit_transform": 15,
    }


def final_report_product_sites():
    fit = next(node for node in ast.walk(SOURCE_TREE)
               if isinstance(node, ast.FunctionDef) and node.name == "fit_transform")
    sites = {}
    for node in ast.walk(fit):
        if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute)
                and target.attr == "reconstruction_err_" for target in node.targets)):
            calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)
                     and isinstance(child.func, ast.Name) and child.func.id == "matmul"]
            assert len(calls) == 1
            call = calls[0]
            explicit = bool(call.keywords)
            if explicit:
                assert len(call.keywords) == 1 and call.keywords[0].arg == "_final_report_copy4"
                assert isinstance(call.keywords[0].value, ast.Constant) and call.keywords[0].value.value is True
            sites[explicit] = (call.lineno, call.col_offset, call.end_lineno, call.end_col_offset)
    assert set(sites) == {False, True}
    return sites


@pytest.mark.parametrize("padded", [False, True], ids=["ordinary-report", "padded-report-opt-in"])
def test_every_product_site_is_exercised_including_nested_calls(monkeypatch, padded):
    observed, selections = set(), []

    def recorded(left, right, **kwargs):
        frame = inspect.currentframe().f_back
        if frame.f_globals is experimental.__dict__ and frame.f_code.co_name == "_diagnostic_residual":
            assert frame.f_back.f_globals is experimental.__dict__
            assert frame.f_back.f_code is experimental.projected_gradient_kkt.__code__
            frame = frame.f_back
        try:
            instruction = next(item for item in dis.get_instructions(
                frame.f_code, show_caches=True) if item.offset == frame.f_lasti)
            position = instruction.positions
            assert position.col_offset is not None, "Python code-position metadata is required"
            site = (position.lineno, position.col_offset,
                    position.end_lineno, position.end_col_offset)
            assert kwargs == ({"_final_report_copy4": True}
                              if padded and site == final_report_product_sites()[True] else {})
            observed.add(site)
        finally:
            del frame
        return reference_product(left, right)

    def select():
        selections.append(True)
        return recorded

    def diagonal_passive_solve(design, targets, gram, cross, passive):
        # Routing coverage has no need to invoke a native factorization. The
        # chosen disjoint-column/rank-one problems have diagonal Gram matrices.
        diagonal = np.diag(gram)
        np.testing.assert_array_equal(gram, np.diag(diagonal))
        answer = np.where(passive, cross / diagonal[:, None], 0.0)
        return answer, 1, 0

    monkeypatch.setattr(experimental, "nnls_matmul", select)
    if padded:
        monkeypatch.setattr(experimental, "padded_matmul", recorded)
    monkeypatch.setattr(experimental, "_passive_solve", diagonal_passive_solve)
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)
    design, targets, expected = analytic_problem()
    coefficients, info = experimental.nnls_block_pivot(
        design, targets, init=np.ones_like(expected))
    assert info["success"]
    np.testing.assert_allclose(coefficients, expected, rtol=0, atol=1e-13)

    data = np.tile(np.array([[2., 4., 6., 8.]]), (6, 1))
    model = experimental.ANLSBPP(1, max_iter=4)
    coordinates = model.fit_transform(data)
    assert model.converged_
    assert model.n_iter_ == 1
    experimental.projected_gradient_kkt(data, coordinates, model.components_)
    with experimental.capture_transform_diagnostics(model) as diagnostic:
        model.transform(data / 2)
    assert diagnostic["converged"]
    assert len(selections) == 6  # direct NNLS, fit, two sweeps, diagnostic, transform NNLS
    # Replaced numerical bindings deliberately decline capsule admission.
    fallback_sites = (set(source_product_sites()) - set(handoff_consumer_product_sites())
                      - quiet_helper_product_sites())
    # Each policy executes exactly one report expression; the parameter pair
    # covers both sites and asserts the opt-in keyword is confined to its site.
    fallback_sites -= {final_report_product_sites()[not padded]}
    assert observed == fallback_sites and len(observed) == 22
    # Controlled private-tail routing only: not natural admission or an added
    # public API obligation. Use the same selected callable and known exact
    # feasible state; the independent profile-only science proves admission.
    scale = np.maximum(np.max(np.abs(reference_product(design.T, targets)), axis=0),
                       np.finfo(float).tiny)
    returned, tail_info, residual, gradient = experimental._nnls_final_state(
        recorded, design, targets, expected.copy(), np.empty(0, dtype=int),
        scale, 1e-10, 1, 3, 1, 0, 0)
    np.testing.assert_allclose(returned, expected, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(residual, reference_product(design, returned) - targets)
    np.testing.assert_array_equal(gradient, reference_product(design.T, residual))
    assert tail_info["success"]
    assert observed == fallback_sites | quiet_helper_product_sites()


@pytest.mark.parametrize("replay", [False, True], ids=["quiet", "gw-replay"])
def test_handoff_consumer_product_sites_use_selected_callable(monkeypatch, replay):
    """Exercise actual fit consumer branches with a controlled valid offer.

    This isolates routing and FPE replay. The supplied offer bypasses producer
    admission deliberately; real admission is covered by profile-only science.
    """
    quiet_site, replay_site, h_site = handoff_consumer_product_sites()
    observed, policies, offers, checks = [], [], [], []
    original_policy = np.geterr().copy()
    injected = FloatingPointError("controlled strict W-gradient boundary")
    fired = []

    def recorded(left, right):
        frame = inspect.currentframe().f_back
        try:
            instruction = next(item for item in dis.get_instructions(
                frame.f_code, show_caches=True) if item.offset == frame.f_lasti)
            position = instruction.positions
            site = (position.lineno, position.col_offset,
                    position.end_lineno, position.end_col_offset)
        finally:
            del frame
        if site in (quiet_site, replay_site, h_site):
            observed.append(site)
            policies.append(np.geterr().copy())
        if site == quiet_site and replay:
            assert not fired
            fired.append(injected)
            raise injected
        return reference_product(left, right)

    def diagonal_passive_solve(design, targets, gram, cross, passive):
        diagonal = np.diag(gram)
        np.testing.assert_array_equal(gram, np.diag(diagonal))
        return np.where(passive, cross / diagonal[:, None], 0.0), 1, 0

    def offer(role, W, H, X, operation):
        assert operation is recorded and role is None
        residual = reference_product(W, H) - X
        gradient = reference_product(W.T, residual)
        payload = (W, H, residual, gradient, np.geterr().copy())
        offers.append(payload)
        return payload

    def usable(payload, role, W, H, X, operation):
        assert payload is offers[-1] and role is None and operation is recorded
        assert payload[0] is W and payload[1] is H
        np.testing.assert_array_equal(payload[2], reference_product(W, H) - X)
        np.testing.assert_array_equal(payload[3], reference_product(W.T, payload[2]))
        checks.append(True)
        return True

    monkeypatch.setattr(experimental, "nnls_matmul", lambda: recorded)
    monkeypatch.setattr(experimental, "_passive_solve", diagonal_passive_solve)
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)
    monkeypatch.setattr(experimental, "_take_handoff", offer)
    monkeypatch.setattr(experimental, "_handoff_usable", usable)
    # Retain controlled consumer routing above the fixed rank-one early gate.
    data = np.tile(np.array([[2., 4., 6., 8.]]), (4097, 1))
    assert data.size > 16384
    model = experimental.ANLSBPP(1, max_iter=4)
    coordinates = model.fit_transform(data)
    assert model.converged_ and model.n_iter_ == 1
    assert len(offers) == 1
    np.testing.assert_allclose(reference_product(coordinates, model.components_),
                               data, rtol=0, atol=1e-13)
    assert observed == ([quiet_site, replay_site, h_site] if replay else [quiet_site])
    assert fired == ([injected] if replay else [])
    # The injected take bypasses initial usability; quiet joint consumption
    # still performs the selected parent's second check. Replay drops the offer.
    assert checks == ([] if replay else [True])
    probe_policy = dict.fromkeys(original_policy, "raise")
    if original_policy["under"] == "ignore":
        probe_policy["under"] = "ignore"
    assert policies == [probe_policy] + ([original_policy, original_policy] if replay else [])
    assert np.geterr() == original_policy


AOCL = {"user_api": "blas", "internal_api": "blis", "prefix": "aocl-libblis",
        "version": "AOCL-BLAS 5.3.0", "architecture": "zen3", "num_threads": 1}
MKL = {"user_api": "blas", "internal_api": "mkl", "prefix": "mkl_rt"}


@pytest.mark.parametrize("pools,padded", [
    ([AOCL], True), ([MKL, AOCL], True), ([MKL], False),
    ([dict(AOCL, prefix="libblis", version="2.0")], False), ([], False),
])
def test_backend_selection_without_unprotected_native_products(monkeypatch, pools, padded):
    calls = []

    def safe_padding_spy(left, right):
        calls.append((left.shape, right.shape))
        return reference_product(left, right)

    monkeypatch.setattr(_projection, "_native_controller",
                        lambda: SimpleNamespace(info=lambda: pools))
    monkeypatch.setattr(_aocl, "padded_matmul", safe_padding_spy)
    selected = _aocl.nnls_matmul()
    assert selected is (safe_padding_spy if padded else np.matmul)
    if padded:
        # Also prove the experimental imported selector reaches this policy.
        design, targets, expected = analytic_problem()
        result, info = experimental.nnls_block_pivot(design, targets)
        assert info["success"] and calls
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
    else:
        # Metadata mocks do not change the actual library. Do not execute a
        # fake-MKL ordinary product when this process might really use AOCL.
        assert not calls


@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_padding_preserves_signed_readonly_operands_and_true_dimensions(monkeypatch, layout):
    left = layout_copy((np.arange(28).reshape(7, 4) - 14) / 8, layout)
    right = layout_copy((np.arange(20).reshape(4, 5) - 10) / 4, layout)
    before = left.copy(), right.copy()
    calls = []

    def safe_native_reference(a, b):
        calls.append((a.copy(), b.copy()))
        return reference_product(a, b)

    monkeypatch.setattr(_aocl.np, "matmul", safe_native_reference)
    result = _aocl.padded_matmul(left, right)
    assert len(calls) == 1
    padded_left, padded_right = calls[0]
    assert padded_left.shape == (8, 4)
    assert padded_right.shape == (4, 8)
    np.testing.assert_array_equal(padded_left[:7], left)
    np.testing.assert_array_equal(padded_right[:, :5], right)
    assert not np.any(padded_left[7:])
    assert not np.any(padded_right[:, 5:])
    np.testing.assert_array_equal(result, reference_product(left, right))
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(right, before[1])
    assert result.dtype == np.float64 and result.flags.c_contiguous
    assert not np.shares_memory(result, left)
    assert not np.shares_memory(result, right)


@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
@pytest.mark.parametrize("warm_start", [False, True])
def test_analytic_signed_nnls_and_original_problem_kkt(reference_dispatch, layout, warm_start):
    design, targets, expected = analytic_problem()
    design, targets = layout_copy(design, layout), layout_copy(targets, layout)
    before = design.copy(), targets.copy()
    initial = np.ones_like(expected) if warm_start else None
    result, info = experimental.nnls_block_pivot(design, targets, init=initial)
    assert info["success"]
    assert len(reference_dispatch) == 1
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
    residual = reference_product(design, result) - targets
    gradient = reference_product(design.T, residual)
    assert np.all(result >= 0)
    assert np.all(gradient >= -1e-12)
    np.testing.assert_allclose(result * gradient, 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(info["residual_norm"], np.sqrt(np.sum(residual * residual)),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(design, before[0])
    np.testing.assert_array_equal(targets, before[1])
    if initial is not None:
        np.testing.assert_array_equal(initial, np.ones_like(expected))


def test_fit_transform_clone_serialization_and_nonmutation(monkeypatch, reference_dispatch):
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)
    truth_w = np.arange(1., 7.)[:, None] / 2
    truth_h = np.array([[.5, 1., 2., 3.]])
    data = layout_copy(reference_product(truth_w, truth_h), "fortran")
    original = data.copy()
    prototype = experimental.ANLSBPP(1, max_iter=10, random_state=7)
    model = clone(prototype)
    coordinates = model.fit_transform(data)
    assert model.converged_
    assert not hasattr(prototype, "components_")
    assert model.get_params() == prototype.get_params()
    assert coordinates.dtype == np.float64
    np.testing.assert_array_equal(coordinates, model.training_coordinates_)
    np.testing.assert_allclose(reference_product(coordinates, model.components_), data,
                               rtol=1e-12, atol=1e-12)
    basis = model.components_.copy()
    with experimental.capture_transform_diagnostics(model) as diagnostic:
        np.testing.assert_allclose(model.transform(data / 2), coordinates / 2,
                                   rtol=1e-12, atol=1e-12)
    assert diagnostic["converged"]
    np.testing.assert_array_equal(model.components_, basis)
    np.testing.assert_array_equal(data, original)
    # A local selected callable must not become fitted/serialized backend state.
    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params() == model.get_params()
    np.testing.assert_allclose(restored.transform(data / 2), coordinates / 2,
                               rtol=1e-12, atol=1e-12)


def test_outer_cap_retains_warning_and_diagnostics(monkeypatch, reference_dispatch):
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)
    data = np.array([[1., 0., 2.], [0., 2., 1.], [2., 1., 0.], [1., 2., 3.]])
    before = data.copy()
    model = experimental.ANLSBPP(1, max_iter=1, tol=0)
    with pytest.warns(ConvergenceWarning, match="outer_iteration_limit"):
        coordinates = model.fit_transform(data)
    assert not model.converged_
    assert model.convergence_reason_ == "outer_iteration_limit"
    assert model.n_iter_ == 1
    assert len(model.objective_history_) == 2
    assert model.objective_history_[1] <= model.objective_history_[0] + 1e-12
    np.testing.assert_array_equal(coordinates, model.training_coordinates_)
    np.testing.assert_array_equal(data, before)


def test_transform_cap_retains_warning_and_basis(reference_dispatch):
    model = experimental.ANLSBPP(2, nnls_max_iter=1)
    model.components_ = np.array([[1., 0.], [1., 1.]])
    model.n_features_in_ = 2
    basis = model.components_.copy()
    queries = layout_copy(np.array([[1., 3.], [2., 6.]]), "strided")
    before = queries.copy()
    with pytest.warns(ConvergenceWarning, match="inner NNLS failed"), experimental.capture_transform_diagnostics(model) as diagnostic:
        result = model.transform(queries)
    assert not diagnostic["converged"]
    assert not diagnostic["diagnostics"]["success"]
    assert result.shape == (2, 2) and np.all(result >= 0)
    np.testing.assert_array_equal(model.components_, basis)
    np.testing.assert_array_equal(queries, before)


def test_canonical_diagnostic_invariance_and_nonmutation(reference_dispatch):
    weights = np.array([[1., 2.], [2., 1.], [3., .5]])
    basis = np.array([[1., 0., 2., 1.], [0., 1., 1., 2.]])
    data = reference_product(weights, basis) + .125
    before = data.copy(), weights.copy(), basis.copy()
    scales = np.array([.5, 2.])
    first = experimental.projected_gradient_kkt(data, weights, basis)
    second = experimental.projected_gradient_kkt(data, weights * scales,
                                                basis / scales[:, None])
    assert len(reference_dispatch) == 2
    for key, value in first["canonical"].items():
        np.testing.assert_allclose(value, second["canonical"][key], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(first["objective"], .5 * data.size * .125 ** 2,
                               rtol=1e-12, atol=1e-12)
    for actual, expected in zip((data, weights, basis), before):
        np.testing.assert_array_equal(actual, expected)
