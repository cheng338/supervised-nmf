"""Owned MU workspaces preserve the scalar result and optional fallback.

Generic behavior and decline tests run on every supported numerical stack.
Only observations that require the qualified acceleration skip when its
implementation policy declines. No investigation fixtures are required.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
import scipy
from sklearn.decomposition import NMF
from threadpoolctl import threadpool_limits

import supervised_nmf._mu as mu


@pytest.fixture(scope="module", autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def same_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    # NaN payloads are not a numerical contract; preserve all other bits,
    # including signed zero, instead of accepting a loose tolerance.
    keep = ~np.isnan(expected).ravel()
    a = np.ascontiguousarray(actual).view(np.uint8).reshape(actual.size, actual.dtype.itemsize)
    b = np.ascontiguousarray(expected).view(np.uint8).reshape(expected.size, expected.dtype.itemsize)
    assert a[keep].tobytes() == b[keep].tobytes()


def pair():
    rng = np.random.default_rng(8723)
    return rng.uniform(-5, 5, (4, 4096)), rng.uniform(-5, 5, (17, 4096))


@contextmanager
def observe_reduction():
    """Observe real code without replacing a callable used by admission."""
    owned = mu._owned_log_sum.__code__
    fallback = mu._log_sum.__code__
    scalar = mu._log_cross_reference.__code__
    rows = []
    previous = sys.getprofile()

    def profile(frame, event, result):
        if event == "call" and frame.f_code is owned:
            rows.append(("owned",))
        elif event == "call" and frame.f_code is fallback:
            rows.append(("fallback",))
        elif event == "call" and frame.f_code is scalar:
            rows.append(("scalar", frame.f_locals["left"].tobytes(),
                         frame.f_locals["right"].tobytes()))

    try:
        sys.setprofile(profile)
        yield rows
    finally:
        sys.setprofile(previous)


def require_fast_route():
    if not mu._owned_log_sum.implementation_contract["admitted_implementation"]:
        pytest.skip("Numerical implementation is outside optional owned-reduction policy")
    assert mu._owned_log_sum_compatible(), "Qualified bindings changed before test"


@pytest.mark.parametrize("layout", ["C", "right-F", "left-F", "strided", "readonly", "shared"])
def test_cross_product_preserves_inputs_and_scalar_bits(layout):
    left, right = pair()
    if layout == "right-F":
        right = np.asfortranarray(right)
    elif layout == "left-F":
        left = np.asfortranarray(left)
    elif layout == "strided":
        owner = np.zeros((17, 8192))
        owner[:, ::2] = right
        right = owner[:, ::2]
    elif layout == "readonly":
        left.flags.writeable = right.flags.writeable = False
    elif layout == "shared":
        right = left[:3]
    before = left.tobytes(), right.tobytes(), left.flags.writeable, right.flags.writeable
    expected = mu._log_cross_reference(left, right)
    with np.errstate(all="raise"):
        policy = np.geterr()
        actual = mu._log_cross(left, right)
        assert np.geterr() == policy
    same_bytes(actual, expected)
    assert (left.tobytes(), right.tobytes(), left.flags.writeable, right.flags.writeable) == before
    assert not np.shares_memory(actual, left) and not np.shares_memory(actual, right)


@pytest.mark.parametrize("case", ["ties", "partial-ties", "signed-zero", "subnormal",
                                  "underflow-tail", "mixed-negative-inf", "positive-bound", "negative-bound"])
def test_qualified_owned_reduction_matches_scipy_bits(case):
    require_fast_route()
    products = np.empty((2, 3, 17), dtype=np.float64)
    products.fill(3.)
    if case == "partial-ties":
        products[..., 1::2] = -20.
    elif case == "signed-zero":
        products.fill(0.)
        products[..., 1::2] = -0.
    elif case == "subnormal":
        products.fill(np.finfo(float).smallest_subnormal)
    elif case == "underflow-tail":
        products.fill(-1000.)
        products[..., 0], products[..., 1] = 0., -745.
    elif case == "mixed-negative-inf":
        products.fill(-np.inf)
        products[..., 0] = 0.
    elif case == "positive-bound":
        products.fill(float(2 ** 60))
    elif case == "negative-bound":
        products.fill(-float(2 ** 60))
    original = products.copy()
    expected = mu._log_sum(original, axis=2)
    with np.errstate(all="raise"):
        policy = np.geterr()
        with observe_reduction() as observed:
            actual = mu._owned_log_sum(products)
        assert np.geterr() == policy
    same_bytes(actual, expected)
    assert observed == [("owned",)]
    # Every fixture here distinguishes in-place reduction from an untouched workspace.
    assert products.tobytes() != original.tobytes()


@pytest.mark.parametrize("kind", ["view", "readonly", "F", "float32", "subclass",
                                  "nan", "positive-inf", "all-negative-inf", "above-bound"])
def test_unowned_or_unsupported_products_decline_without_mutation(kind):
    class ArraySubclass(np.ndarray):
        pass

    products = np.arange(102, dtype=np.float64).reshape(2, 3, 17).copy()
    if kind == "view":
        owner = np.empty((2, 3, 34))
        owner[..., :17] = products
        products = owner[..., :17]
    elif kind == "readonly":
        products.flags.writeable = False
    elif kind == "F":
        products = np.asfortranarray(products)
    elif kind == "float32":
        products = products.astype(np.float32)
    elif kind == "subclass":
        products = products.view(ArraySubclass)
    elif kind == "nan":
        products[0, 0, 0] = np.nan
    elif kind == "positive-inf":
        products[0, 0, 0] = np.inf
    elif kind == "all-negative-inf":
        products.fill(-np.inf)
    elif kind == "above-bound":
        products.fill(np.nextafter(float(2 ** 60), np.inf))
    before = products.tobytes(), products.strides, products.flags.writeable
    with np.errstate(all="ignore"):
        expected = mu._log_sum(products, axis=2)
        with observe_reduction() as observed:
            actual = mu._owned_log_sum(products)
    same_bytes(actual, expected)
    assert observed == [("owned",), ("fallback",)]
    assert (products.tobytes(), products.strides, products.flags.writeable) == before


def error_outcome(function, left, right, policy):
    events = []
    previous = np.geterrcall()

    def callback(message, flag):
        events.append((message, flag))

    try:
        np.seterrcall(callback)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(all=policy):
                expected_state = np.geterr()
                try:
                    value, error = function(left, right), None
                except FloatingPointError as exc:
                    value, error = None, (type(exc), str(exc))
                assert np.geterr() == expected_state and np.geterrcall() is callback
        return value, error, [(w.category, str(w.message)) for w in caught], events
    finally:
        np.seterrcall(previous)


@pytest.mark.parametrize("policy", ["raise", "warn", "call", "ignore"])
@pytest.mark.parametrize("kind", ["overflow-late", "invalid-late"])
def test_exceptional_panels_replay_original_scalar_error_order(kind, policy):
    left = np.full((4, 4096), np.finfo(float).max if kind == "overflow-late" else -np.inf)
    right = np.zeros((9, 4096))
    right[4] = np.finfo(float).max if kind == "overflow-late" else np.inf
    before = left.tobytes(), right.tobytes()
    expected = error_outcome(mu._log_cross_reference, left, right, policy)
    actual = error_outcome(mu._log_cross, left, right, policy)
    assert actual[1:] == expected[1:]
    if expected[0] is not None:
        same_bytes(actual[0], expected[0])
    assert (left.tobytes(), right.tobytes()) == before


@pytest.mark.parametrize("change", ["binding", "code", "defaults"])
def test_changed_log_sum_declines_and_uses_the_live_callable(change, monkeypatch):
    left, right = pair()
    original = mu._log_sum
    marker = RuntimeError("live log_sum must remain authoritative")

    def replacement(values, axis=None):
        raise marker

    if change == "binding":
        monkeypatch.setattr(mu, "_log_sum", replacement)
        expected_error = marker
    elif change == "code":
        # Assign a closure-free body, preserving the function object's identity.
        def changed(values, axis=None):
            raise RuntimeError("changed log_sum code")
        monkeypatch.setattr(original, "__code__", changed.__code__)
        expected_error = None
    else:
        monkeypatch.setattr(original, "__defaults__", tuple([None]))
        expected_error = False
    assert not mu._owned_log_sum_compatible()
    with observe_reduction() as observed:
        if expected_error is False:
            same_bytes(mu._log_cross(left, right), mu._log_cross_reference(left, right))
        else:
            with pytest.raises(RuntimeError) as caught:
                mu._log_cross(left, right)
            if expected_error is not None:
                assert caught.value is expected_error
            else:
                assert str(caught.value) == "changed log_sum code"
    assert not any(row[0] == "owned" for row in observed)


def test_replaced_numpy_ufunc_declines_without_changing_results(monkeypatch):
    left, right = pair()
    expected = mu._log_cross_reference(left, right)
    original = np.exp
    before = left.tobytes(), right.tobytes(), np.geterr(), np.geterrcall()

    def delegated(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "exp", delegated)
    assert not mu._owned_log_sum_compatible()
    with observe_reduction() as observed:
        actual = mu._log_cross(left, right)
    same_bytes(actual, expected)
    assert not any(row[0] == "owned" for row in observed)
    assert (left.tobytes(), right.tobytes(), np.geterr(), np.geterrcall()) == before


@pytest.mark.parametrize("reason", ["version", "source", "unavailable"])
def test_optional_implementation_policy_declines_without_disabling_results(reason, monkeypatch):
    contract = mu._owned_log_sum.implementation_contract
    with monkeypatch.context() as local:
        if reason == "version":
            local.setattr(scipy, "__version__", "unqualified-test-version")
        else:
            # Reach the source policy even on an otherwise unqualified stack.
            # These temporary constructor inputs never install an admitted path.
            numpy_version, scipy_version = contract["qualified_numpy_scipy_pairs"][0]
            local.setattr(np, "__version__", numpy_version)
            local.setattr(scipy, "__version__", scipy_version)
            if reason == "source":
                local.setattr(Path, "read_bytes", lambda self: b"unqualified implementation source\n")
            else:
                def unavailable(self):
                    raise OSError("optional implementation source unavailable")
                local.setattr(Path, "read_bytes", unavailable)
        reducer, compatible = mu._make_owned_log_sum()
    assert not reducer.implementation_contract["admitted_implementation"]
    assert not compatible() and reducer.implementation_contract["reason"]
    products = np.arange(102, dtype=np.float64).reshape(2, 3, 17).copy()
    before = products.tobytes()
    same_bytes(reducer(products), mu._log_sum(products, axis=2))
    assert products.tobytes() == before


@pytest.mark.parametrize(("versions", "qualified"), [
    (("2.5.4", "1.18.1"), True), (("2.5.2", "1.18.0"), True),
    (("2.5.4", "1.18.0"), False), (("2.5.2", "1.18.1"), False),
])
def test_numerical_versions_admit_only_exact_pairs(versions, qualified, monkeypatch):
    def unavailable(path):
        if not qualified:
            pytest.fail(f"Unqualified pair tried to read {path}")
        raise OSError("Optional source is unavailable")

    with monkeypatch.context() as local:
        local.setattr(np, "__version__", versions[0])
        local.setattr(scipy, "__version__", versions[1])
        local.setattr(Path, "read_bytes", unavailable)
        reducer, compatible = mu._make_owned_log_sum()
    contract = reducer.implementation_contract
    assert contract["qualified_numpy_scipy_pairs"] == (("2.5.4", "1.18.1"), ("2.5.2", "1.18.0"))
    assert (contract["numpy"], contract["scipy"]) == versions
    assert (contract["required_numpy"], contract["required_scipy"]) == (
        versions if qualified else (None, None))
    assert not compatible()
    if qualified:
        assert contract["reason"].startswith("optional implementation unavailable")
    else:
        assert contract["reason"] == "unqualified numerical version"


@pytest.mark.parametrize("when", ["before_capture", "after_capture"])
@pytest.mark.parametrize(("namespace", "name"), [
    *(("xp", name) for name in ("exp", "sign", "abs", "squeeze", "max", "astype",
                                "sum", "log", "isfinite", "where", "isdtype", "log1p")),
    *(("implementation", name) for name in ("logsumexp", "_logsumexp",
      "_elements_and_indices_with_max_real", "array_namespace", "xp_promote", "xp_size")),
    ("extra", "at"), ("extra", "atleast_nd"), ("at", "__init__"), ("at", "__getitem__"),
    ("at", "set"), ("np", "errstate"), ("np", "sign"),
])
def test_direct_scipy_reduction_callbacks_are_preserved(namespace, name, when, monkeypatch):
    require_fast_route()
    from scipy.special import _logsumexp as implementation
    from scipy._external.array_api_compat import numpy as xp

    owner = {"implementation": implementation, "xp": xp, "extra": implementation.xpx,
             "at": implementation.xpx.at, "np": np}[namespace]
    marker = RuntimeError(f"observed {namespace}.{name}")
    def observed(*args, **kwargs):
        raise marker

    products = np.array([[[-1., -2., -3., -4.]]])
    before = products.tobytes()
    with monkeypatch.context() as local:
        local.setattr(owner, name, observed)
        if name == "logsumexp":
            # Simulate the public import still referring to the same modified binding.
            local.setattr(mu, "logsumexp", observed)
        if when == "before_capture":
            reducer, compatible = mu._make_owned_log_sum()
        else:
            reducer, compatible = mu._owned_log_sum, mu._owned_log_sum_compatible
        assert not compatible()
        with pytest.raises(RuntimeError) as reference:
            mu._log_sum(products, axis=2)
        with pytest.raises(RuntimeError) as actual:
            reducer(products)
        assert reference.value is actual.value is marker
        assert products.tobytes() == before
    assert mu._owned_log_sum_compatible()


@pytest.mark.parametrize("name", ["inf", "nan"])
@pytest.mark.parametrize("when", ["before_capture", "after_capture"])
def test_indexed_update_scalars_decline_before_mutation(name, when, monkeypatch):
    require_fast_route()
    from scipy._external.array_api_compat import numpy as xp

    products = np.array([[[-1., -2., -3.]]])
    before = products.tobytes()
    with monkeypatch.context() as local:
        local.setattr(xp, name, 0.)
        if when == "before_capture":
            reducer, compatible = mu._make_owned_log_sum()
        else:
            reducer, compatible = mu._owned_log_sum, mu._owned_log_sum_compatible
        assert not compatible()
        expected = mu._log_sum(products, axis=2)
        actual = reducer(products)
        same_bytes(actual, expected)
        assert not np.shares_memory(actual, products) and products.tobytes() == before


@pytest.mark.parametrize("target", ["private_body", "indexed_set", "maximum_default"])
def test_preexisting_edits_to_original_scipy_functions_decline(target, monkeypatch):
    require_fast_route()
    from scipy.special import _logsumexp as implementation

    def observed(*args, **kwargs):
        raise RuntimeError("edited SciPy function")

    products = np.array([[[-1., -2., -3.]]])
    before = products.tobytes()
    with monkeypatch.context() as local:
        if target == "maximum_default":
            local.setattr(implementation._elements_and_indices_with_max_real, "__kwdefaults__", {"axis": 0})
        else:
            function = implementation._logsumexp if target == "private_body" else implementation.xpx.at.set
            local.setattr(function, "__code__", observed.__code__)
        reducer, compatible = mu._make_owned_log_sum()
        assert not compatible()
        if target == "maximum_default":
            same_bytes(reducer(products), mu._log_sum(products, axis=2))
        else:
            for operation in (lambda: mu._log_sum(products, axis=2), lambda: reducer(products)):
                with pytest.raises(RuntimeError, match="edited SciPy function"):
                    operation()
        assert products.tobytes() == before
    assert mu._owned_log_sum_compatible()


def test_owned_workspace_fpe_replays_untouched_operands(monkeypatch):
    require_fast_route()
    left, right = pair()
    before = left.tobytes(), right.tobytes()
    expected = mu._log_cross_reference(left, right)
    workspaces = []

    def fail_after_mutation(products):
        assert products.flags.owndata and products.base is None
        assert not np.shares_memory(products, left) and not np.shares_memory(products, right)
        products.fill(np.nan)
        workspaces.append(products.shape)
        raise FloatingPointError("failure after owned workspace mutation")

    # Inject at the owned-workspace boundary; do not weaken the admission gate
    # or claim an allocator-failure test of the unchanged reduction body.
    monkeypatch.setattr(mu, "_owned_log_sum", fail_after_mutation)
    assert mu._owned_log_sum_compatible()
    with observe_reduction() as observed:
        actual = mu._log_cross(left, right)
    same_bytes(actual, expected)
    assert workspaces == [(4, 4, 4096)]
    assert [row for row in observed if row[0] == "scalar"] == [("scalar", *before)]
    assert (left.tobytes(), right.tobytes()) == before


@pytest.mark.parametrize("name", ["_owned_log_sum", "_owned_log_sum_compatible"])
def test_new_helper_replacement_invalidates_handoff_and_scalar_cache(name, monkeypatch):
    rng = np.random.default_rng(765312)
    basis = rng.uniform(.2, .8, (3, 31))
    basis[0, 0] = 2. ** -520
    query = np.ldexp(rng.uniform(.1, .9, (1, 31)), 600)
    logs, exponent = mu._centered_logs(basis, per_row=False)
    gram = mu._log_cross(logs, logs)
    first = mu._log_sum(gram, axis=0)[None, :]
    prefix = mu._CappedPrefix(0, np.empty((0, 3)))
    offer = lambda: mu._handoff_log_setup(prefix, query, basis, logs, exponent, gram, first, 21, 1e-8)
    assert type(offer()) is mu._SuffixSetup
    original = getattr(mu, name)

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(mu, name, replacement)
    assert offer() is prefix
    model = NMF(n_components=3, solver="mu", init="random", random_state=0,
                max_iter=9, tol=0.)
    model.components_ = basis
    model.n_components_ = model._n_components = 3
    model.n_features_in_ = 31
    model._beta_loss = 2
    expected = mu.transform_mu(model, query)
    cache = {"stale": object()}
    actual = mu.transform_mu(model, query, _basis_cache=cache)
    same_bytes(actual, expected)
    assert cache == {}
