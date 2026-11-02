"""Selected residual reuse and C-N coupling; no forced private backend route.

Original E1/E2/E3 portable contracts remain byte-exact in their separate files.
These bounded rank9 fixtures must demonstrate actual route entry when run;
source eligibility alone is not a numerical qualification result.
"""
from contextlib import contextmanager
import gc
import inspect
from pathlib import Path
import sys
import warnings
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
import experimental_nmf as experimental
from supervised_nmf import _aocl
import test_experimental_residual_reuse as e1
import test_experimental_outer_residual_reuse as e2
import test_experimental_diagnostic_residual_reuse as e3

assert Path(experimental.__file__).resolve() == ROOT / "scripts/experimental_nmf.py"
assert Path(_aocl.__file__).resolve() == ROOT / "src/supervised_nmf/_aocl.py"
assert all(Path(m.__file__).resolve().parent == ROOT / "tests" for m in (e1, e2, e3))
bits, input_state = e1.bits, e1.input_state

@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        assert experimental.nnls_matmul() in (np.matmul, _aocl.padded_matmul)
        yield

def assert_cn_native(event):
    """Prove the reached selected CN path, including the borrowed right core."""
    left, right = event["left"], event["right"]
    m, k = left.shape
    n = right.shape[1]
    assert m >= 8 and n >= 8 and m % 8 and n % 8
    assert right.flags.c_contiguous and right.strides == (n * 8, 8)
    calls = event["native"]
    mp, nc = (m + 7) // 8 * 8, n // 8 * 8
    assert [(c["left"].shape, c["right"].shape) for c in calls] == [
        ((mp, k), (k, nc)), ((mp, k), (k, 8))]
    assert calls[0]["left"] is calls[1]["left"]
    assert np.shares_memory(calls[0]["right"], right)
    assert calls[0]["right"].strides == right.strides
    assert bits(calls[0]["right_before"]) == bits(event["right_before"][:, :nc])
    assert bits(calls[1]["right_before"][:, :n-nc]) == bits(event["right_before"][:, nc:])
    assert bits(calls[1]["right_before"][:, n-nc:]) == bits(np.zeros((k, 8-(n-nc))))
    assert bits(calls[0]["left_before"][:m]) == bits(event["left_before"])
    assert bits(calls[0]["left_before"][m:]) == bits(np.zeros((mp-m, k)))
    assert bits(left) == bits(event["left_before"])
    assert bits(right) == bits(event["right_before"])
    assert not any(np.shares_memory(event["output"], c["output"]) for c in calls)
    assert not np.shares_memory(event["output"], left)
    assert not np.shares_memory(event["output"], right)

def record_solve(monkeypatch, A, B, *, init=None, cap=None, prepared=False, role=None):
    """Record actual supported products; no output alias or solver stub."""
    selected = experimental.nnls_matmul()
    events = []
    native_calls = []
    class NativeProxy:
        def __getattr__(self, name):
            return getattr(np, name)

        def matmul(self, left, right):
            # Snapshots precede the real call: mutation cannot hide in a later copy.
            left_state, right_state = bits(left), bits(right)
            left_before, right_before = left.copy(), right.copy()
            try:
                output = np.matmul(left, right)
            finally:
                assert bits(left) == left_state and bits(right) == right_state
            native_calls.append(dict(left=left, right=right, output=output,
                                     left_before=left_before, right_before=right_before))
            return output

    def product(left, right):
        left_before, right_before = left.copy(), right.copy()
        start = len(native_calls)
        output = selected(left, right)
        assert bits(left) == bits(left_before) and bits(right) == bits(right_before)
        assert type(output) is np.ndarray and output.dtype == np.float64
        assert output.flags.owndata and output.flags.writeable and output.flags.c_contiguous
        assert not np.shares_memory(output, left) and not np.shares_memory(output, right)
        events.append(dict(left=left, right=right, output=output,
                           left_before=left_before, right_before=right_before,
                           output_before=output.copy(), native=native_calls[start:]))
        return output

    before = (input_state(A), input_state(B), None if init is None else input_state(init))
    with monkeypatch.context() as patch:
        patch.setattr(_aocl, "np", NativeProxy())
        # Preserve the item6 identity guard when the native selector is padded.
        if selected is _aocl.padded_matmul:
            patch.setattr(_aocl, "padded_matmul", product)
            patch.setattr(experimental, "padded_matmul", product)
        patch.setattr(experimental, "nnls_matmul", lambda: product)
        if role is None and prepared:
            role = experimental._FixedTrainingRHS(B)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            answer = (role.solve(A, init=init, max_iter=cap) if role is not None
                      else experimental.nnls_block_pivot(A, B, init=init, max_iter=cap))
        role_state = None
        if role is not None and role._prepared is not None:
            cached = role._prepared._right
            role_state = dict(object=role, cache=cached, bytes=bits(cached))
    assert before == (input_state(A), input_state(B), None if init is None else input_state(init))
    return dict(answer=answer, events=events, role_state=role_state, role=role,
                protected=selected is not np.matmul,
                warnings=[(row.category.__name__, str(row.message)) for row in caught])

def check_cn_residual(record):
    product, gradient = record["events"][-2:]
    residual = product["output"]
    assert gradient["right"] is residual
    assert bits(gradient["right_before"]) == bits(residual)
    if record["protected"]:
        assert_cn_native(product)
        assert_cn_native(gradient)
        assert np.shares_memory(gradient["native"][0]["right"], residual)
    else:
        assert product["native"] == gradient["native"] == []
    assert not np.shares_memory(gradient["output"], residual)

@contextmanager
def capture_native(monkeypatch,before=None,after=None):
    selected=experimental.nnls_matmul()
    norm=experimental.linalg.norm
    records=dict(products=[],norms=[],protected=selected is not np.matmul)
    native_calls=[]
    class NativeProxy:
        def __getattr__(self, name):
            return getattr(np, name)

        def matmul(self, left, right):
            # Snapshots precede the real call: mutation cannot hide in a later copy.
            left_state, right_state = bits(left), bits(right)
            left_before, right_before = left.copy(), right.copy()
            try:
                output = np.matmul(left, right)
            finally:
                assert bits(left) == left_state and bits(right) == right_state
            native_calls.append(dict(left=left, right=right, output=output,
                                     left_before=left_before, right_before=right_before))
            return output


    def multiply(left,right):
        frame=inspect.currentframe().f_back
        if before is not None: before(frame,left,right)
        left_before, right_before = left.copy(), right.copy()
        start = len(native_calls)
        output=selected(left,right)
        assert bits(left) == bits(left_before) and bits(right) == bits(right_before)
        assert type(output) is np.ndarray and output.dtype==np.float64
        assert output.flags.owndata and output.flags.writeable and output.flags.c_contiguous
        assert not np.shares_memory(output,left) and not np.shares_memory(output,right)
        records['products'].append(dict(caller=frame.f_code.co_name,caller_parent=(frame.f_back.f_code.co_name if frame.f_back is not None else None),left=left,right=right,
                                        output=output,original=output.copy(), left_before=left_before,
                                        right_before=right_before,native=native_calls[start:]))
        if after is not None: after(frame,left,right,output)
        return output
    def observed_norm(value,*args,**kwargs):
        frame=inspect.currentframe().f_back
        if frame.f_globals is experimental.__dict__ and frame.f_code.co_name=='fit_transform':
            records['norms'].append(dict(value=value,bytes=bits(value)))
        return norm(value,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(_aocl, "np", NativeProxy())
        if selected is _aocl.padded_matmul:
            patch.setattr(_aocl,'padded_matmul',multiply)
            patch.setattr(experimental,'padded_matmul',multiply)
        patch.setattr(experimental,'nnls_matmul',lambda:multiply)
        patch.setattr(experimental.linalg,'norm',observed_norm)
        yield records


@pytest.mark.parametrize("kind", ["c", "f", "strided"])
@pytest.mark.parametrize("warm", [False, True])
def test_rank9_final_residual_is_borrowed_after_subtraction(monkeypatch, kind, warm):
    A, B, expected = e1.analytic_problem(13, 9, 17, kind)
    init = e1.layout(expected, kind) if warm else None
    record = record_solve(monkeypatch, A, B, init=init)
    e1.check_residual_contract(record, A, B)
    check_cn_residual(record)
    coefficients, info = record["answer"]
    np.testing.assert_allclose(coefficients, expected, rtol=1e-11, atol=1e-13)
    assert info["success"] and info["projected_gradient_relative"] <= info["kkt_tolerance"]
    assert not record["warnings"]


@pytest.mark.parametrize("warm", [False, True])
def test_rank9_prepared_repeat_preserves_cache_and_releases_residuals(monkeypatch, warm):
    A, B, expected = e1.analytic_problem(13, 9, 17, "c")
    init = e1.layout(expected, "c") if warm else None
    first = record_solve(monkeypatch, A, B, init=init, prepared=True)
    e1.check_residual_contract(first, A, B)
    check_cn_residual(first)
    assert first["answer"][1]["success"] and not first["warnings"]
    role = first["role"]
    assert role is not None
    cache = None if first["role_state"] is None else first["role_state"]["cache"]
    saved = None if cache is None else bits(cache)
    assert (cache is not None) == first["protected"]
    second_A, second_init = e1.layout(A * 2, "c"), e1.layout(expected / 2, "c")
    second = record_solve(monkeypatch, second_A, B, init=second_init, role=role)
    e1.check_residual_contract(second, second_A, B)
    check_cn_residual(second)
    assert second["answer"][1]["success"] and not second["warnings"]
    np.testing.assert_allclose(second["answer"][0], expected / 2, rtol=1e-11, atol=1e-13)
    if cache is not None:
        assert role._prepared._right is cache and bits(cache) == saved
        assert second["role_state"]["cache"] is cache
    refs = [weakref.ref(first["events"][-2]["output"]),
            weakref.ref(second["events"][-2]["output"])]
    if cache is not None:
        refs.append(weakref.ref(cache))
    del first, second, role, cache
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_rank9_capped_residual_retains_failure_and_borrowed_gradient(monkeypatch):
    # Preserve the already-qualified cap fixture's eight RHS, then duplicate
    # them to reach 17 columns without discarding its original failing columns.
    rng = np.random.RandomState(10)
    A = e1.layout(rng.normal(size=(20, 9)), "f")
    original_B = rng.normal(size=(20, 8))
    B = e1.layout(np.concatenate((original_B, original_B, original_B[:, :1]), axis=1), "strided")
    record = record_solve(monkeypatch, A, B, cap=1)
    e1.check_residual_contract(record, A, B)
    check_cn_residual(record)
    info = record["answer"][1]
    assert not info["success"] and info["max_iter"] == info["n_iter"] == 1
    assert any(category == "ConvergenceWarning" and
               message.startswith("ANLS-BPP inner NNLS failed its iteration/KKT qualification;")
               for category, message in record["warnings"])


def check_outer_lifetimes(records, X, model, coordinates):
    """Return weakrefs only; observer strong references end with the records."""
    consumed = []
    for norm in records["norms"]:
        matches = [event for event in records["products"]
                   if event["caller"] == "fit_transform" and event["output"] is norm["value"]
                   and event["output"].shape == X.shape
                   and bits(np.subtract(event["original"], np.asarray(X, dtype=np.float64))) == norm["bytes"]]
        if matches:
            assert len(matches) == 1
            consumed.append(matches[0])
    assert len(consumed) == model.n_iter_ == 3
    inner_residuals = []
    for index, event in enumerate(records["products"][:-1]):
        next_event = records["products"][index + 1]
        if (event["caller"] == next_event["caller"] == "_nnls_block_pivot"
                and next_event["right"] is event["output"]):
            # E1 executes in the same fit, distinct from the outer E2 objects.
            assert bits(next_event["right_before"]) == bits(event["output"])
            assert bits(event["output"]) != bits(event["original"])
            assert not np.shares_memory(next_event["output"], event["output"])
            inner_residuals.append(event["output"])
    assert len(inner_residuals) == 2 * model.n_iter_
    for iteration, event in enumerate(consumed):
        residual = event["output"]
        expected = np.subtract(event["original"], np.asarray(X, dtype=np.float64))
        assert bits(residual) == bits(expected)
        uses = [other for other in records["products"] if other["caller"] == "fit_transform"
                and (other["left"] is residual or other["right"] is residual)]
        assert len(uses) == 2 and uses[0]["left"] is residual and uses[1]["right"] is residual
        assert bits(uses[0]["left_before"]) == bits(uses[1]["right_before"]) == bits(residual)
        assert all(bits(other["output"]) == bits(other["original"]) for other in uses)
        assert not any(np.shares_memory(residual, value)
                       for value in (X, coordinates, model.components_, *inner_residuals))
        assert model.objective_history_[iteration + 1] == pytest.approx(.5 * e2.scalar_norm(residual) ** 2, rel=1e-13)
        if records["protected"]:
            assert_cn_native(event)
            assert_cn_native(uses[1])
            assert np.shares_memory(uses[1]["native"][0]["right"], residual)
            # H.T is F storage, so the first gradient uses the retained fallback.
            assert not uses[0]["right"].flags.c_contiguous and len(uses[0]["native"]) == 1
        else:
            assert event["native"] == uses[0]["native"] == uses[1]["native"] == []
    return [weakref.ref(event["output"]) for event in consumed] + [weakref.ref(value) for value in inner_residuals]


def test_rank9_outer_residual_survives_both_gradients_and_refit(monkeypatch):
    X = e2.fit_data((65, 33), "strided")
    original = e2.state(X)
    with capture_native(monkeypatch) as records:
        model, coordinates = e2.fit_model(X, rank=9)
    refs = check_outer_lifetimes(records, X, model, coordinates)
    assert e2.state(X) == original
    del records
    gc.collect()
    assert all(ref() is None for ref in refs)
    # The same estimator refits; no residual/native borrowed view may survive
    # the completed first fit, and the second fit must reach the same contracts.
    with capture_native(monkeypatch) as records:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coordinates = model.fit_transform(X)
    assert model.n_iter_ == 3 and model.nnls_diagnostics_["failed_solves"] == 0
    assert model.convergence_reason_ == "outer_iteration_limit"
    assert len(caught) == 1 and caught[0].category.__name__ == "ConvergenceWarning"
    refs = check_outer_lifetimes(records, X, model, coordinates)
    assert e2.state(X) == original
    del records
    gc.collect()
    assert all(ref() is None for ref in refs)


def check_diagnostic_uses(records, X, W, H, result):
    events = [event for event in records["products"] if event["caller"] == "projected_gradient_kkt" or (event["caller"] == "_diagnostic_residual" and event["caller_parent"] == "projected_gradient_kkt")]
    assert len(events) == 5
    residual = events[0]["output"]
    expected = np.subtract(events[0]["original"], np.asarray(X, dtype=np.float64))
    assert bits(residual) == bits(expected)
    for index, event in enumerate(events[1:], 1):
        side = "left" if index in (1, 3) else "right"
        assert event[side] is residual and bits(event[side + "_before"]) == bits(residual)
        assert bits(event["output"]) == bits(event["original"])
        assert not np.shares_memory(event["output"], residual)
        if records["protected"]:
            if side == "right":
                assert_cn_native(event)
                assert np.shares_memory(event["native"][0]["right"], residual)
            else:
                assert not event["right"].flags.c_contiguous and len(event["native"]) == 1
        else:
            assert event["native"] == []
    if records["protected"]:
        assert_cn_native(events[0])
    else:
        assert events[0]["native"] == []
    assert not any(np.shares_memory(residual, value) for value in (X, W, H))
    x, w, h = (np.asarray(value, dtype=np.float64) for value in (X, W, H))
    direct = e3.product_reference(w, h) - x
    raw = e3.independent_pg(w, h, e3.product_reference(direct, h.T), e3.product_reference(w.T, direct))
    scales = np.asarray([e3.scalar_norm(row) for row in h])
    scales = np.where(scales > 0, scales, 1.)
    cw, ch = w * scales, h / scales[:, None]
    canonical = e3.independent_pg(cw, ch, e3.product_reference(direct, ch.T), e3.product_reference(cw.T, direct))
    for name in raw:
        assert result["raw"][name] == pytest.approx(raw[name], rel=1e-12, abs=1e-12)
        assert result["canonical"][name] == pytest.approx(canonical[name], rel=1e-12, abs=1e-12)
    assert result["reconstruction_error"] == pytest.approx(e3.scalar_norm(direct), rel=1e-13)
    assert result["objective"] == pytest.approx(.5 * e3.scalar_norm(direct) ** 2, rel=1e-13)
    assert result["relative_reconstruction_error"] == pytest.approx(e3.scalar_norm(direct) / e3.scalar_norm(x), rel=1e-13)
    assert result["dtype"] == "float64" and result["convention"].startswith("full unregularized Frobenius NMF;")
    return weakref.ref(residual)


@pytest.mark.parametrize("kind", ["c", "strided"])
@pytest.mark.parametrize("zero_scale_row", [False, True])
def test_rank9_diagnostic_residual_is_unchanged_across_four_uses(monkeypatch, kind, zero_scale_row):
    W = e3.readonly((np.arange(13 * 9).reshape(13, 9) % 5 + 1) / 2, kind)
    h = (np.arange(9 * 17).reshape(9, 17) % 4 + 1) / 2
    if zero_scale_row:
        h[-1] = 0.
    H = e3.readonly(h, kind)
    X = e3.readonly((np.arange(13 * 17).reshape(13, 17) % 6 + 1) / 3, kind)
    original = [e3.state(value) for value in (X, W, H)]
    with capture_native(monkeypatch) as records:
        result = experimental.projected_gradient_kkt(X, W, H)
    ref = check_diagnostic_uses(records, X, W, H, result)
    assert [e3.state(value) for value in (X, W, H)] == original
    del records
    gc.collect()
    assert ref() is None
