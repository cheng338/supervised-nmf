"""Logarithmic MU recovery preserves independent rows and observable failures."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import math
import pickle
import struct
import threading
from unittest.mock import patch
import warnings

import numpy as np
import pytest
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import supervised_nmf._mu as mu
import supervised_nmf._mu_rowwise as rowwise


@pytest.fixture(scope="module", autouse=True)
def single_numerical_thread():
    with threadpool_limits(limits=1):
        yield


def fitted_mu(basis, *, max_iter=20000, tol=1e-4):
    """A fixed dictionary isolates projection from joint factor estimation."""
    model = NMF(n_components=len(basis), solver="mu", init="random",
                random_state=0, max_iter=max_iter, tol=tol)
    model.components_ = basis.copy(order="K")
    model.n_components_ = model._n_components = len(basis)
    model.n_features_in_ = basis.shape[1]
    model._beta_loss = 2
    model.n_iter_ = 7
    model.reconstruction_err_ = .25
    return model


def dense_fixture(rows=3):
    rng = np.random.default_rng(230613)
    basis = rng.uniform(.1, .9, (3, 17))
    basis[0, 0] = 2. ** -900
    return rng.uniform(.1, .9, (3, 17))[:rows], basis


def independent_fixture(rows=3):
    basis = np.asarray([[1., 0., 2. ** -900], [0., 1., 0.]])
    query = np.asarray([[1., 1., 1e6], [1., 2., 0.], [2., 1., 3.]])
    return query[:rows], basis


def serial_projection(model, query):
    return np.vstack([mu.transform_mu(model, value[None, :]) for value in query])


@contextmanager
def stop_trace():
    original, events = mu._stopped, []
    def observed(*args):
        answer = original(*args)
        events.append((tuple(float(v) for v in args), bool(answer)))
        return answer
    with patch.object(mu, "_stopped", observed):
        yield events


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_log_product_layout_and_panel_tail_match_scalar_reduction(layout):
    rng = np.random.default_rng(912731)
    left, right = rng.uniform(-5, 5, (4, 1024)), rng.uniform(-5, 5, (65, 1024))
    if layout == "F":
        left, right = np.asfortranarray(left), np.asfortranarray(right)
    elif layout == "strided":
        storage = np.zeros((4, 2048))
        storage[:, ::2] = left
        left = storage[:, ::2]
    before = left.tobytes(), right.tobytes()
    with np.errstate(all="raise"):
        expected = mu._log_cross_reference(left, right)
        actual = mu._log_cross(left, right)
    np.testing.assert_array_equal(actual, expected)
    assert (left.tobytes(), right.tobytes()) == before


@pytest.mark.parametrize("layout", ["C", "F"])
def test_residual_panel_tail_retains_adjacent_log_values(layout):
    rng = np.random.default_rng(912732)
    # The C-order product crosses the feature-panel boundary at256 columns.
    coordinates = rng.uniform(-3, 3, (32, 32))
    basis = rng.uniform(-3, 3, (32, 257))
    if layout == "F":
        coordinates = np.asfortranarray(coordinates)
    observed = np.column_stack([mu._log_sum(coordinates + basis[:, j], axis=1)
                                for j in range(basis.shape[1])])
    observed[:, ::3] = np.nextafter(observed[:, ::3], np.inf)
    observed[:, 1::3] = np.nextafter(observed[:, 1::3], -np.inf)
    shifts = np.linspace(-500, 500, len(coordinates))
    before = [a.tobytes() for a in (observed, coordinates, basis, shifts)]
    with np.errstate(all="raise"):
        expected = mu._log_error_reference(observed, coordinates, basis, shifts)
        actual = mu._log_error(observed, coordinates, basis, shifts)
    assert actual == expected
    assert before == [a.tobytes() for a in (observed, coordinates, basis, shifts)]


def arithmetic_outcome(function, arrays, policy):
    callbacks, result, error = [], None, None
    before_errors, before_callback = np.geterr(), np.geterrcall()
    class Logger:
        def write(self, message):
            callbacks.append(message)
    try:
        np.seterrcall(Logger() if policy == "log" else lambda message, flag: callbacks.append((message, flag)))
        with warnings.catch_warnings(record=True) as caught, np.errstate(all=policy):
            warnings.simplefilter("always")
            try:
                result = function(*arrays)
            except FloatingPointError as exc:
                error = type(exc), str(exc)
        events = [(w.category, str(w.message)) for w in caught]
    finally:
        np.seterrcall(before_callback)
    assert np.geterr() == before_errors and np.geterrcall() is before_callback
    return result, error, events, callbacks


@pytest.mark.parametrize("policy", ["warn", "raise", "ignore", "call", "log"])
@pytest.mark.parametrize("operation", ["cross", "error"])
def test_panel_arithmetic_replays_original_caller_policy(operation, policy):
    maximum = np.finfo(np.float64).max
    if operation == "cross":
        arrays = [np.full((1, 3), maximum), np.full((2, 3), maximum)]
        reference, optimized = mu._log_cross_reference, mu._log_cross
    else:
        arrays = [np.full((1, 3), maximum), np.zeros((1, 2)), np.zeros((2, 3)), np.zeros(1)]
        reference, optimized = mu._log_error_reference, mu._log_error
    before = [a.tobytes() for a in arrays]
    expected = arithmetic_outcome(reference, arrays, policy)
    actual = arithmetic_outcome(optimized, arrays, policy)
    assert actual[1:] == expected[1:]
    np.testing.assert_array_equal(actual[0], expected[0])
    assert before == [a.tobytes() for a in arrays]


@pytest.mark.parametrize("rows,layout", [(2, "C"), (3, "C"), (3, "F"), (3, "readonly")])
def test_vector_recovery_accepts_rows_with_independent_stops(rows, layout):
    query, basis = independent_fixture(rows)
    if layout == "F":
        query, basis = np.asfortranarray(query), np.asfortranarray(basis)
    if layout == "readonly":
        query.flags.writeable = basis.flags.writeable = False
    model = fitted_mu(basis)
    state, inputs = pickle.dumps(model), query.tobytes()
    expected, histories = [], []
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        for value in query:
            with stop_trace() as history:
                expected.append(mu.transform_mu(model, value[None, :]))
            histories.append(history)
        # These rows actually stop at different checks; a shared stop is wrong.
        assert len({len(history) for history in histories}) > 1
        with stop_trace() as actual_history:
            with patch.object(rowwise, "transform_mu", side_effect=AssertionError("Unexpected serial replay")):
                actual = rowwise.transform_mu_rowwise(model, query)
    np.testing.assert_array_equal(actual, np.vstack(expected))
    assert Counter(actual_history) == Counter(event for history in histories for event in history)
    assert pickle.dumps(model) == state and query.tobytes() == inputs


def test_recovery_coordinates_survive_singletons_reversal_and_subset():
    query, basis = independent_fixture()
    model = fitted_mu(basis)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        full = rowwise.transform_mu_rowwise(model, query)
        np.testing.assert_array_equal(rowwise.transform_mu_rowwise(model, query[::-1])[::-1], full)
        np.testing.assert_array_equal(serial_projection(model, query), full)
        np.testing.assert_array_equal(rowwise.transform_mu_rowwise(model, query[[0, 2]]), full[[0, 2]])


@pytest.mark.parametrize("rows,target", [(2, 10), (3, 20)])
def test_actual_stopping_boundary_matches_serial_rows(rows, target):
    query, basis = dense_fixture()
    query = np.repeat(query[:1], rows, axis=0)
    with warnings.catch_warnings(), stop_trace() as oracle:
        warnings.simplefilter("ignore", ConvergenceWarning)
        mu._project_log_mu(query[:1], basis, 21, np.finfo(float).tiny)
    assert len(oracle) == 2
    previous, current, initial, _ = oracle[target // 10 - 1][0]
    assert current < previous
    expression = previous - initial + math.log(-math.expm1(current - previous))
    number = lambda bits: struct.unpack(">d", struct.pack(">Q", bits))[0]
    low, high = 1, 0x7fefffffffffffff
    assert math.log(number(low)) <= expression < math.log(number(high))
    while high - low > 1:
        middle = (low + high) // 2
        if math.log(number(middle)) <= expression:
            low = middle
        else:
            high = middle
    assert np.nextafter(number(low), np.inf) == number(high)
    for tolerance, should_stop in ((number(low), False), (number(high), True)):
        if target == 20:
            assert not mu._stopped(*oracle[0][0][:3], tolerance)
        with warnings.catch_warnings(record=True) as caught, stop_trace() as expected_history:
            warnings.simplefilter("always")
            expected = np.vstack([mu._project_log_mu(value[None, :], basis, 21, tolerance) for value in query])
        for i in range(rows):
            history = expected_history[i * (len(expected_history) // rows):(i + 1) * (len(expected_history) // rows)]
            assert len(history) >= target // 10 and history[target // 10 - 1][1] is should_stop
        with stop_trace() as actual_history, warnings.catch_warnings(record=True) as speculative_warnings:
            warnings.simplefilter("always")
            actual = mu._try_log_row_recovery(query, basis, 21, tolerance)
        assert not speculative_warnings
        assert Counter(actual_history) == Counter(expected_history)
        if caught:
            assert len(caught) == rows and all(w.category is ConvergenceWarning for w in caught)
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)


def recovery_outcome(query, basis, *, vector, cap=11, mutation=None, strict=False):
    model = fitted_mu(basis, max_iter=cap)
    output = np.full((len(query), len(basis)), -777.)
    events, buffers, error = [], [], None
    def emitted(message, category, *args, **kwargs):
        events.append((category, str(message)))
        if len(events) == 1 and mutation == "inplace":
            model.components_[0, 1] *= 1.125
        if len(events) == 1 and mutation == "replace":
            model.components_ = np.asfortranarray(model.components_ * 1.125)
    def serial(decomposition, selected):
        buffers.append(output.copy())
        return mu.transform_mu(decomposition, selected)
    with warnings.catch_warnings(), patch.object(warnings, "showwarning", emitted), np.errstate(all="raise"):
        warnings.simplefilter("error" if strict else "always", ConvergenceWarning)
        try:
            if vector:
                with patch.object(rowwise, "transform_mu", serial):
                    rowwise._recover_rows(model, query, output, np.any(query != 0, axis=1))
            else:
                for index in np.flatnonzero(np.any(query != 0, axis=1)):
                    output[index] = serial(model, query[index:index + 1])[0]
        except (FloatingPointError, ConvergenceWarning) as exc:
            error = type(exc), str(exc)
    return output, model.components_, events, error, buffers


def assert_same_replay(expected, actual):
    for index in (0, 1):
        np.testing.assert_array_equal(actual[index], expected[index])
    assert actual[2:4] == expected[2:4]
    assert len(actual[4]) == len(expected[4]) > 0
    for a, b in zip(actual[4], expected[4]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[4][0], np.full_like(actual[0], -777.))


@pytest.mark.parametrize("mutation,strict", [(None, False), (None, True), ("inplace", False), ("replace", False)])
def test_cap_replay_preserves_warning_order_basis_changes_and_partial_output(mutation, strict):
    query, basis = dense_fixture()
    inputs = query.tobytes(), basis.tobytes()
    expected = recovery_outcome(query, basis, vector=False, mutation=mutation, strict=strict)
    actual = recovery_outcome(query, basis, vector=True, mutation=mutation, strict=strict)
    assert_same_replay(expected, actual)
    if strict:
        assert actual[3][0] is ConvergenceWarning
    else:
        assert actual[3] is None and actual[2]
    assert (query.tobytes(), basis.tobytes()) == inputs


@pytest.mark.parametrize("strict,cap,error", [(False, 21, FloatingPointError), (True, 1, ConvergenceWarning)])
def test_later_range_failure_retains_first_observable_error(strict, cap, error):
    tiny = np.finfo(float).smallest_subnormal
    basis = np.asarray([[tiny, 0., 0.], [0., 1., 0.]])
    query = np.asarray([[tiny, 1., 0.], [tiny, 2., 0.], [1., 1., 0.]])
    expected = recovery_outcome(query, basis, vector=False, cap=cap, strict=strict)
    actual = recovery_outcome(query, basis, vector=True, cap=cap, strict=strict)
    assert_same_replay(expected, actual)
    assert actual[3][0] is error


@pytest.mark.parametrize("kind", ["singleton", "float32", "rank_one", "ordinary", "scalable", "huge", "cap1", "cap9", "cap10"])
def test_unqualified_routes_do_not_enter_vector_solver(kind):
    query, basis = dense_fixture()
    cap, tolerance = 21, 1e-4
    if kind == "singleton":
        query = query[:1]
    elif kind == "float32":
        query = query.astype(np.float32)
        basis = np.maximum(basis, np.finfo(np.float32).smallest_subnormal).astype(np.float32)
    elif kind == "rank_one":
        basis = basis[:1]
    elif kind == "ordinary":
        basis[0, 0] = .25
    elif kind == "scalable":
        basis[0, 0] = 2. ** -520
    elif kind == "huge":
        basis = np.full((2, 17), 2. ** 600)
    else:
        cap = int(kind[3:])
    with patch.object(mu, "_project_log_mu_rows", side_effect=AssertionError("Unexpected vector route")):
        assert mu._try_log_row_recovery(query, basis, cap, tolerance) is None


@pytest.mark.parametrize("policy", ["warn", "raise", "ignore", "call", "log"])
def test_discarded_arithmetic_failure_does_not_leak_caller_notifications(policy):
    query, basis = dense_fixture(2)
    expected = serial_projection(fitted_mu(basis), query)
    before_errors, before_callback = np.geterr(), np.geterrcall()
    events = []
    class Logger:
        def write(self, message):
            events.append(message)
    try:
        np.seterrcall(Logger() if policy == "log" else lambda *args: events.append(args))
        with np.errstate(all=policy), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch.object(mu, "_log_row_errors", side_effect=lambda *a: np.sqrt(np.float64(-1.))):
                actual = rowwise.transform_mu_rowwise(fitted_mu(basis), query)
        assert not events and not caught
    finally:
        np.seterrcall(before_callback)
    assert np.geterr() == before_errors and np.geterrcall() is before_callback
    np.testing.assert_array_equal(actual, expected)


def test_concurrent_recovery_has_no_shared_iteration_or_caller_state():
    query, basis = dense_fixture(2)
    dictionaries = [basis, basis * 1.25]
    expected = [serial_projection(fitted_mu(h, max_iter=11, tol=0.), query) for h in dictionaries]
    before = query.tobytes(), [h.tobytes() for h in dictionaries]
    barrier = threading.Barrier(2)
    def solve(h):
        errors, callback = np.geterr(), np.geterrcall()
        barrier.wait(timeout=10)
        value = mu._try_log_row_recovery(query, h, 11, 0.)
        assert value is not None
        assert np.geterr() == errors and np.geterrcall() is callback
        return value, threading.get_ident()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(solve, h) for h in dictionaries]
        actual = [future.result(timeout=30) for future in futures]
    assert len({identity for _, identity in actual}) == 2
    for (value, _), reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)
    assert (query.tobytes(), [h.tobytes() for h in dictionaries]) == before
