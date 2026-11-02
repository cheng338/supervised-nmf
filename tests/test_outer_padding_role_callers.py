"""Conditional draft: actual multi-sweep caller checks if outer reuse wins."""
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path(__file__).resolve().parent)]

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
import experimental_nmf as experimental
from supervised_nmf import _aocl
from role_caller_probe import RoleProbe, state

assert Path(experimental.__file__).resolve() == ROOT / "scripts/experimental_nmf.py"
assert Path(_aocl.__file__).resolve() == ROOT / "src/supervised_nmf/_aocl.py"


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def inputs(shape=(65, 33)):
    # Existing outer residual-consumption regression fixture.
    value = np.random.default_rng(4308).uniform(.05, 2., shape)
    value.setflags(write=False)
    return value


def fit(model, value):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit_transform(value)
    return result, [(item.category.__name__, str(item.message)) for item in caught]


def snapshot(model, result, caught):
    return dict(result=result.copy(), H=model.components_.copy(),
                objectives=np.asarray(model.objective_history_),
                gradients=np.asarray(model.projected_gradient_history_),
                diagnostics=dict(model.nnls_diagnostics_), n_iter=model.n_iter_,
                reason=model.convergence_reason_, warnings=caught)


def assert_snapshot(actual, expected):
    for key in ("result", "H", "objectives", "gradients"):
        np.testing.assert_array_equal(actual[key].view(np.uint64), expected[key].view(np.uint64))
    for key in ("diagnostics", "n_iter", "reason", "warnings"):
        assert actual[key] == expected[key]


def model(rank=5):
    return experimental.ANLSBPP(n_components=rank, max_iter=3, tol=0., random_state=43)


def baseline(monkeypatch, value, rank=5):
    # Disable only the new optional factory; retain the current helper and all
    # existing solver/prepared-input/residual behavior byte-for-byte.
    with monkeypatch.context() as patch:
        patch.setattr(experimental, "make_output_workspace", lambda operation: operation)
        instance = model(rank)
        result, caught = fit(instance, value)
    return snapshot(instance, result, caught)


def test_outer_roles_read_changing_operands_and_refit_cleanly(monkeypatch):
    monkeypatch.setattr(experimental, "nnls_matmul", lambda: _aocl.padded_matmul)
    value = inputs()
    before = state(value)
    expected = baseline(monkeypatch, value)
    probe = RoleProbe()
    monkeypatch.setattr(experimental, "make_output_workspace",
                        probe.wrap_factory(experimental.make_output_workspace, bound=False))
    instance = model()
    for _ in range(2):
        result, caught = fit(instance, value)
        assert_snapshot(snapshot(instance, result, caught), expected)
        assert instance.n_iter_ == 3
        probe.assert_released()
    assert len(probe.records) == 6
    for record in probe.records:
        assert len(record["calls"]) == 3
        assert len({call["left"] for call in record["calls"]}) > 1
        assert len({call["right"] for call in record["calls"]}) > 1
    assert state(value) == before


@pytest.mark.parametrize("fail_call", [1, 2])
def test_outer_error_releases_all_roles_and_refit_recovers(monkeypatch, fail_call):
    monkeypatch.setattr(experimental, "nnls_matmul", lambda: _aocl.padded_matmul)
    value = inputs()
    before = state(value)
    expected = baseline(monkeypatch, value)
    probe = RoleProbe(fail_call=fail_call)
    monkeypatch.setattr(experimental, "make_output_workspace",
                        probe.wrap_factory(experimental.make_output_workspace, bound=False))
    instance = model()
    with pytest.raises(RuntimeError, match="injected caller role failure"):
        fit(instance, value)
    probe.assert_released()
    assert not hasattr(instance, "components_")
    probe.fail_call = None
    result, caught = fit(instance, value)
    assert_snapshot(snapshot(instance, result, caught), expected)
    probe.assert_released()
    assert state(value) == before


def test_outer_aligned_setup_bypasses_factory_for_all_three_roles(monkeypatch):
    monkeypatch.setattr(experimental, "nnls_matmul", lambda: _aocl.padded_matmul)
    value = inputs((64, 32))
    expected = baseline(monkeypatch, value, rank=8)
    def forbidden_factory(operation):
        raise AssertionError("both output dimensions aligned: no role needed")
    monkeypatch.setattr(experimental, "make_output_workspace", forbidden_factory)
    instance = model(8)
    result, caught = fit(instance, value)
    assert_snapshot(snapshot(instance, result, caught), expected)


def test_outer_ordinary_backend_returns_exact_original_callable(monkeypatch):
    value = inputs()
    safe = experimental.nnls_matmul()
    def ordinary(left, right):
        return safe(left, right)
    monkeypatch.setattr(experimental, "nnls_matmul", lambda: ordinary)
    expected = baseline(monkeypatch, value)
    probe = RoleProbe()
    monkeypatch.setattr(experimental, "make_output_workspace",
                        probe.wrap_factory(experimental.make_output_workspace, bound=False))
    instance = model()
    result, caught = fit(instance, value)
    assert_snapshot(snapshot(instance, result, caught), expected)
    assert probe.bypasses == [True, True, True] and probe.records == []
