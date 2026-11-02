"""Exact-repeat mechanism and experimental fit-loop regression tests.

Scripted inner solves isolate stopping priority and resource lifetime. The final
small numerical test exercises the actual solver without a backend-specific
iteration count. A repeated factor pair is not a convergence certificate.
"""
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings
import weakref

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import experimental_nmf as experimental


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def _pair(token, layout="C", readonly=True):
    arrays = (np.arange(6, dtype=np.float64).reshape(3, 2) + 1 + 8 * token,
              np.arange(8, dtype=np.float64).reshape(2, 4) + 1 + 16 * token)
    answer = []
    for value in arrays:
        if layout == "F":
            value = np.asfortranarray(value)
        elif layout == "strided":
            backing = np.zeros((value.shape[0], 2 * value.shape[1]))
            backing[:, ::2] = value
            value = backing[:, ::2]
        if readonly:
            value.setflags(write=False)
        answer.append(value)
    return tuple(answer)


def _signature(pair):
    return tuple((value.shape, value.dtype.str, value.tobytes(order="C")) for value in pair)


def _assert_scalar_event(event):
    assert set(event) == {"method", "detector", "detected_iteration", "matched_iteration", "period"}
    assert event["method"] == "fingerprint"
    assert event["detector"] in ("fingerprint", "brent_layout_fallback")
    assert all(type(event[key]) is int for key in ("detected_iteration", "matched_iteration", "period"))
    assert event["period"] == event["detected_iteration"] - event["matched_iteration"] > 0


@pytest.mark.parametrize("period", [1, 2, 5, 9, 19, 25])
@pytest.mark.parametrize("tail", [0, 7])
def test_first_contiguous_repeat_including_long_periods(period, tail):
    detector = experimental._ExactFactorCycle()
    for iteration in range(1, tail + period + 2):
        token = iteration if iteration <= tail else 100 + (iteration - tail - 1) % period
        pair = _pair(token)
        before = _signature(pair)
        event = detector.observe(*pair, iteration)
        assert _signature(pair) == before
        if iteration < tail + period + 1:
            assert event is None
        else:
            _assert_scalar_event(event)
            assert event["detector"] == "fingerprint"
            assert (event["matched_iteration"], event["period"]) == (tail + 1, period)


def test_forced_hash_collisions_preserve_every_distinct_candidate(monkeypatch):
    monkeypatch.setattr(experimental._ExactFactorCycle, "_fingerprint",
                        staticmethod(lambda w, h: b"constant collision"))
    detector = experimental._ExactFactorCycle()
    for iteration in range(1, 31):
        assert detector.observe(*_pair(iteration), iteration) is None
    event = detector.observe(*_pair(7), 31)
    _assert_scalar_event(event)
    assert (event["matched_iteration"], event["period"]) == (7, 24)


@pytest.mark.parametrize("change", ["W", "H", "ulp", "zero_W", "zero_H", "rescale", "shape"])
def test_hash_hit_requires_both_complete_factor_bit_patterns(monkeypatch, change):
    monkeypatch.setattr(experimental._ExactFactorCycle, "_fingerprint",
                        staticmethod(lambda w, h: b"constant collision"))
    original = _pair(0, readonly=False)
    changed = tuple(value.copy() for value in original)
    if change == "W":
        changed[0][-1, -1] += 1
    elif change == "H":
        changed[1][-1, -1] += 1
    elif change == "ulp":
        changed[0][1, 1] = np.nextafter(changed[0][1, 1], np.inf)
    elif change.startswith("zero_"):
        index = 0 if change == "zero_W" else 1
        original[index][0, 0] = 0.0
        changed[index][0, 0] = -0.0
    elif change == "rescale":
        changed[0][:] *= 2
        changed[1][:] *= 0.5
    else:
        changed = (changed[0].reshape(2, 3), changed[1])
    detector = experimental._ExactFactorCycle()
    assert detector.observe(*original, 1) is None
    assert detector.observe(*changed, 2) is None
    event = detector.observe(*(value.copy() for value in original), 3)
    _assert_scalar_event(event)
    assert event["matched_iteration"] == 1


@pytest.mark.parametrize("layouts", [("F",), ("strided",), ("C", "F", "strided")])
def test_layout_fallback_reports_only_a_verified_logical_repeat(layouts):
    detector = experimental._ExactFactorCycle()
    supplied = {}
    for iteration in range(1, 160):
        token = iteration if iteration <= 7 else 100 + (iteration - 8) % 19
        pair = _pair(token, layouts[(iteration - 1) % len(layouts)])
        signature = _signature(pair)
        event = detector.observe(*pair, iteration)
        supplied[iteration] = signature
        if event is not None:
            _assert_scalar_event(event)
            assert supplied[event["matched_iteration"]] == signature
            assert event["period"] % 19 == 0
            if "C" not in layouts:
                assert event["detector"] == "brent_layout_fallback"
            break
    else:
        pytest.fail("A continuing 19-state sequence was never detected")


@pytest.mark.parametrize("factor", [0, 1])
def test_detector_rejects_non_float64_factors(factor):
    pair = list(_pair(1))
    pair[factor] = pair[factor].astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        experimental._ExactFactorCycle().observe(*pair, 1)


def test_detector_history_releases_all_factor_references():
    def retained():
        detector = experimental._ExactFactorCycle()
        refs = []
        for iteration in range(1, 41):
            pair = _pair(iteration)
            refs.extend(weakref.ref(value) for value in pair)
            assert detector.observe(*pair, iteration) is None
        return detector, refs

    detector, refs = retained()
    gc.collect()
    assert all(reference() is not None for reference in refs)
    del detector
    gc.collect()
    assert all(reference() is None for reference in refs)


class _InjectedFailure(RuntimeError):
    pass


def _scripted_fit(monkeypatch, scenario, model=None):
    """Run the real outer loop with controlled inner/PG outcomes and real arrays."""
    retain_traceback = scenario in ("retained_hash", "retained_equality")
    rank = 2 if scenario == "rank_bypass" else 1
    cap = 12 if retain_traceback else 1 if scenario == "one_sweep" else 4
    tol = 0.0 if scenario == "tol_zero" else 1e-4
    x = np.full((3, 4), float(rank))
    x.setflags(write=False)
    x_before = x.tobytes()
    state = dict(inner_calls=0, pg_calls=0, sweep=0, detector_calls=[], detector_refs=[],
                 workspaces=[], factor_refs=[], nonfinite_injected=False)
    detector_type = experimental._ExactFactorCycle
    original_norm = experimental.linalg.norm

    def remember(value):
        state["factor_refs"].append((weakref.ref(value), value.shape, value.strides, value.tobytes()))
        return value

    def initialize(data, components, **kwargs):
        assert data is x and components == rank
        return remember(np.ones((3, rank))), remember(np.ones((rank, 4)))

    def product(left, right):
        return np.einsum("ik,kj->ij", left, right, optimize=False)

    def solve(a, b, **kwargs):
        state["inner_calls"] += 1
        call = state["inner_calls"]
        sweep, half = (call + 1) // 2, "W" if call % 2 else "H"
        state["sweep"] = sweep
        if scenario == "raise_inner" and sweep == 2 and half == "W":
            raise _InjectedFailure("inner")
        success = not (scenario == "fail_" + half and sweep == 2)
        value = 1.0
        if scenario == "cap" or retain_traceback:
            value = 2.0 ** (sweep if half == "W" else -sweep)
        elif scenario == "objective_failure" and sweep == 2 and half == "W":
            value = 2.0
        answer = remember(np.full((rank, b.shape[1]), value))
        info = dict(success=success, n_iter=1, direct_lstsq_groups=0, single_pivots=0,
                    projected_gradient_relative=0.0)
        return answer, info

    def pg(w, h, gw, gh):
        index = state["pg_calls"]
        state["pg_calls"] += 1
        stationary = scenario == "initial_stationary" and index == 0
        relative_success = scenario == "convergence_wins" and index == 2
        return dict(projected_gradient_norm=0.0 if stationary or relative_success else 1.0)

    class Workspace:
        def __init__(self, operation, role):
            self.operation, self.role, self.clears = operation, role, 0

        def __call__(self, left, right):
            if state["sweep"] == 2:
                if scenario == "raise_reconstruction" and self.role == 0:
                    raise _InjectedFailure("reconstruction")
                if scenario == "raise_gradient" and self.role == 1:
                    raise _InjectedFailure("gradient")
            return self.operation(left, right)

        def clear(self):
            self.clears += 1

    def workspace(operation):
        value = Workspace(operation, len(state["workspaces"]))
        state["workspaces"].append(value)
        return value

    class ObservedDetector(detector_type):
        def __init__(self):
            super().__init__()
            state["detector_refs"].append(weakref.ref(self))

        def observe(self, w, h, iteration):
            state["detector_calls"].append(iteration)
            if scenario == "raise_detector" and iteration == 2:
                raise _InjectedFailure("detector")
            return super().observe(w, h, iteration)

    def fingerprint(w, h):
        if state["sweep"] == 12:
            raise _InjectedFailure("retained hash traceback")
        return detector_type._fingerprint(w, h)

    def equal(left, right):
        if right[0] == 12:
            raise _InjectedFailure("retained equality traceback")
        return detector_type._equal(left, right)

    def norm(value):
        if state["sweep"] == 2 and not state["nonfinite_injected"]:
            state["nonfinite_injected"] = True
            return float("inf")
        return original_norm(value)

    with monkeypatch.context() as patch:
        patch.setattr(experimental, "_initialize_nmf", initialize)
        patch.setattr(experimental, "_pg_values", pg)
        patch.setattr(experimental, "nnls_matmul", lambda: product)
        patch.setattr(experimental, "nnls_block_pivot", solve)
        patch.setattr(experimental, "make_output_workspace", workspace)
        patch.setattr(experimental, "_ExactFactorCycle", ObservedDetector)
        if scenario == "retained_hash":
            patch.setattr(ObservedDetector, "_fingerprint", staticmethod(fingerprint))
        elif scenario == "retained_equality":
            patch.setattr(ObservedDetector, "_fingerprint", staticmethod(lambda w, h: b"collision"))
            patch.setattr(ObservedDetector, "_equal", staticmethod(equal))
        if scenario == "nonfinite_objective":
            patch.setattr(experimental, "linalg", SimpleNamespace(norm=norm))
        if model is None:
            model = experimental.ANLSBPP(n_components=rank, max_iter=cap, tol=tol)
        else:
            model.set_params(n_components=rank, max_iter=cap, tol=tol)
        output, raised = None, None
        # Ordinary exit checks dispose of tracebacks. The two dedicated failure
        # cases deliberately retain them to test explicit history clearing.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                output = model.fit_transform(x)
            except _InjectedFailure as error:
                raised = str(error)
                if retain_traceback:
                    state["retained_error"] = error

    assert (raised is not None) == (scenario.startswith("raise_") or retain_traceback)
    assert x.tobytes() == x_before and not x.flags.writeable
    assert all(workspace.clears == 1 for workspace in state["workspaces"])
    for reference, shape, strides, payload in state["factor_refs"]:
        value = reference()
        if value is not None:
            assert value.shape == shape and value.strides == strides and value.tobytes() == payload
    value = None
    gc.collect()
    if retain_traceback:
        assert len(state["detector_refs"]) == 1
        detector = state["detector_refs"][0]()
        assert detector is not None
        assert detector.history == {} and detector.checkpoint is None
        detector = None
        # Initial factors are followed by W/H for sweeps1..12. The equality
        # traceback legitimately retains its first compared pair (sweep1), and
        # current factor/frame locals may remain. Sweeps2..6 must not survive
        # via either the history dictionary or the in-flight bucket iterator.
        assert len(state["factor_refs"]) == 26
        assert all(reference() is None for reference, *_ in state["factor_refs"][4:14])
    else:
        assert all(reference() is None for reference in state["detector_refs"])
        assert all(reference() is None for reference, *_ in state["factor_refs"])
    assert not hasattr(model, "cycle_hit_factors_")
    if raised is not None:
        assert not hasattr(model, "components_")
        assert model.cycle_event_ is None
        return model, state

    expected = {
        "initial_stationary": ("initial_stationarity", 0, True),
        "convergence_wins": ("relative_projected_gradient", 2, True),
        "fail_W": ("inner_nnls_failure", 1, False),
        "fail_H": ("inner_nnls_failure", 1, False),
        "objective_failure": ("objective_increase_or_nonfinite", 1, False),
        "nonfinite_objective": ("objective_increase_or_nonfinite", 1, False),
        "repeat": ("repeated_iterate", 2, False),
    }.get(scenario, ("outer_iteration_limit", cap, False))
    assert (model.convergence_reason_, model.n_iter_, model.converged_) == expected
    assert len(caught) == int(not model.converged_)
    if caught:
        assert issubclass(caught[0].category, ConvergenceWarning)
        assert model.convergence_reason_ in str(caught[0].message)
    assert len(model.objective_history_) == len(model.projected_gradient_history_) == model.n_iter_ + 1
    assert model.relative_projected_gradient_ == (0.0 if model.converged_ else 1.0)
    if scenario == "repeat":
        _assert_scalar_event(model.cycle_event_)
        assert (model.cycle_event_["detected_iteration"], model.cycle_event_["matched_iteration"]) == (2, 1)
    else:
        assert model.cycle_event_ is None
    if scenario in ("initial_stationary", "tol_zero", "rank_bypass"):
        assert not state["detector_refs"] and not state["detector_calls"]
    if scenario in ("convergence_wins", "fail_W", "fail_H", "objective_failure", "nonfinite_objective"):
        assert state["detector_calls"] == [1]
    assert output.flags.owndata and output.flags.c_contiguous and output.flags.writeable and output.base is None
    assert not np.shares_memory(output, x) and not np.shares_memory(output, model.training_coordinates_)
    np.testing.assert_array_equal(output.view(np.uint64), model.training_coordinates_.view(np.uint64))
    assert model.components_.flags.owndata and not np.shares_memory(model.components_, x)
    return model, state


@pytest.mark.parametrize("scenario", [
    "repeat", "one_sweep", "cap", "tol_zero", "rank_bypass", "initial_stationary",
    "convergence_wins", "fail_W", "fail_H", "objective_failure", "nonfinite_objective",
    "raise_inner", "raise_reconstruction", "raise_gradient", "raise_detector",
])
def test_fit_priority_warning_and_history_cleanup(monkeypatch, scenario):
    _scripted_fit(monkeypatch, scenario)


def test_refit_clears_previous_scalar_event_and_does_not_retain_factor_history(monkeypatch):
    model, _ = _scripted_fit(monkeypatch, "repeat")
    old_event = model.cycle_event_.copy()
    model, _ = _scripted_fit(monkeypatch, "tol_zero", model=model)
    assert old_event["period"] == 1
    assert model.cycle_event_ is None and model.n_iter_ == 4


@pytest.mark.parametrize("scenario", ["retained_hash", "retained_equality"])
def test_retained_failure_traceback_does_not_retain_older_factor_history(monkeypatch, scenario):
    _, state = _scripted_fit(monkeypatch, scenario)
    error = state.pop("retained_error")
    assert error.__traceback__ is not None
    assert state["detector_calls"] == list(range(1, 13))
    # Only after verifying older-history release do we discard the traceback;
    # its current/compared factor locals are allowed to live until this point.
    error.__traceback__ = None
    del error
    gc.collect()
    assert all(reference() is None for reference in state["detector_refs"])
    assert all(reference() is None for reference, *_ in state["factor_refs"])


def test_small_actual_rank1_fit_preserves_truthful_status_and_projection():
    rng = np.random.default_rng(4308)
    x = rng.random((64, 32))
    query = np.ascontiguousarray(x[:8].copy())
    x.setflags(write=False)
    query.setflags(write=False)
    before = (x.tobytes(), query.tobytes())
    model = experimental.ANLSBPP(n_components=1, max_iter=25, tol=1e-4, random_state=43)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = model.fit_transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        projected = model.transform(query)
    assert model.last_transform_converged_
    assert w.shape == (64, 1) and model.components_.shape == (1, 32) and projected.shape == (8, 1)
    assert all(np.isfinite(value).all() and (value >= 0).all() for value in (w, model.components_, projected))
    assert (x.tobytes(), query.tobytes()) == before
    assert not x.flags.writeable and not query.flags.writeable
    assert not hasattr(model, "cycle_hit_factors_")
    assert all(not np.shares_memory(value, original) for value in (w, model.components_, projected) for original in (x, query))
    if model.cycle_event_ is not None:
        _assert_scalar_event(model.cycle_event_)
        assert model.convergence_reason_ == "repeated_iterate" and not model.converged_
        assert model.cycle_event_["detected_iteration"] == model.n_iter_
    if model.converged_:
        assert model.initial_projected_gradient_ == 0 or model.relative_projected_gradient_ <= model.tol
    else:
        assert any(issubclass(item.category, ConvergenceWarning) and model.convergence_reason_ in str(item.message)
                   for item in caught)
    assert model.cycle_event_ is None or model.relative_projected_gradient_ > model.tol
