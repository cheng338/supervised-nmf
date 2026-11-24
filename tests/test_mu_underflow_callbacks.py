"""Fixed-basis batch MU recovers silent underflow and preserves caller callbacks."""

from functools import wraps
import warnings

import joblib
import numpy as np
import pytest
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf import _mu_dispatch


@pytest.fixture(scope="module", autouse=True)
def single_native_thread():
    with threadpool_limits(limits=1):
        yield


def fitted_classifier(basis):
    rank, features = basis.shape
    training = np.random.default_rng(338).uniform(.1, 1., (36, features)).astype(basis.dtype)
    model = NMFClassifier(
        solver="mu", mu_projection="batch", n_components=rank, screening="none",
        random_state=17, n_neighbors=1, max_iter=1, tol=0.,
    ).fit(training, np.arange(len(training)) % 2)
    model.decomposition_.components_ = basis
    model.selected_components_ = basis.copy()
    model.components_ = basis.copy()
    return model


def ordinary_problem():
    rng = np.random.default_rng(4201)
    query = np.ldexp(rng.uniform(.25, 1.75, (1, 17)).astype(np.float32), 16)
    query[0, 0] = 0
    basis = np.empty((4, 17 * 64), dtype=np.float32)[:, ::64]
    basis[:] = np.ldexp(rng.uniform(.25, 1.75, (4, 17)).astype(np.float32), 8)
    return fitted_classifier(basis), query


def test_unqualified_version_uses_native_batch_transform(monkeypatch):
    model, query = ordinary_problem()
    native = model.decomposition_
    expected = native.transform(query)

    def unexpected_qualification(estimator):
        pytest.fail("An unqualified version entered the checked transform")

    monkeypatch.setattr(_mu_dispatch, "QUALIFIED_VERSION", False)
    monkeypatch.setattr(_mu_dispatch, "_custom_tags", unexpected_qualification)
    np.testing.assert_array_equal(_mu_dispatch.native_projection(native, query), expected)


@pytest.mark.skipif(
    not _mu_dispatch.QUALIFIED_VERSION, reason="Native estimator handles this scikit-learn version"
)
@pytest.mark.parametrize("validation", ["stock", "fitted", "restore", "ignore_underflow"])
def test_mixed_scale_two_update_classification(validation, monkeypatch):
    basis = np.eye(32)
    basis[0, 0], basis[-1, 0], basis[-1, -1] = 1e90, 1e-90, 1e-90
    query = np.zeros((128, 32))
    query[:-1, 0], query[-1, -1] = 1., 1e-60
    model = fitted_classifier(basis)
    model.decomposition_.set_params(max_iter=2, tol=1e-4)
    # For the last example only the final numerator is positive. After the
    # first update all competing coefficients are zero. The second update
    # gives (1e-60 * 1e-90) / (2 * 1e-90**2) = 5e29.
    anchors = np.zeros((2, 32))
    anchors[1, -1] = 5e29
    model.classifier_.fit(anchors, [0, 1])
    model.training_coordinates_ = anchors
    expected_labels = np.zeros(128, dtype=int)
    expected_labels[-1] = 1
    stock_dot = np.dot
    validations = []

    def fitted(estimator):
        validations.append(estimator.components_ is basis)
        if validation == "restore":
            np.dot = stock_dot
        elif validation == "ignore_underflow":
            # Suppress the native signal inside the solve, as can happen in a
            # worker BLAS thread. The genuine one-thread calculation then has
            # the same rounded-zero denominator as the parallel regression.
            np.seterr(under="ignore")
        return True

    @wraps(stock_dot)
    def before_validation(*args, **kwargs):
        return stock_dot(*args, **kwargs)

    if validation != "stock":
        monkeypatch.setattr(NMF, "__sklearn_is_fitted__", fitted, raising=False)
    for operation in ("transform", "predict", "predict_proba"):
        validations.clear()
        if validation == "restore":
            monkeypatch.setattr(np, "dot", before_validation)
        with np.errstate(), pytest.warns(ConvergenceWarning, match="iterations 2 reached"):
            result = getattr(model, operation)(query)
        if validation != "stock":
            assert validations and all(validations)
            assert np.dot is stock_dot
        if operation == "transform":
            assert result.shape == query.shape and result.dtype == query.dtype
            np.testing.assert_allclose(result[-1, -1], 5e29, rtol=2e-11, atol=0)
            np.testing.assert_array_equal(result[-1, :-1], np.zeros(31))
        elif operation == "predict":
            np.testing.assert_array_equal(result, expected_labels)
        else:
            np.testing.assert_array_equal(result, np.eye(2)[expected_labels])


@pytest.mark.parametrize("action", ["read", "zero", "return", "raise"])
@pytest.mark.parametrize("install", ["before_transform", "during_validation"])
def test_first_gram_callback_matches_fitted_sklearn(action, install, monkeypatch):
    stock_dot = np.dot
    sentinel = RuntimeError("first Gram callback")

    def observe(public_classifier):
        model, query = ordinary_problem()
        basis = model.decomposition_.components_
        events = []

        @wraps(stock_dot)
        def observed(left, right, *args, **kwargs):
            gram = left.shape == basis.shape and np.shares_memory(left, right)
            events.append(("gram" if gram else "dot", left.shape, right.shape,
                           np.shares_memory(left, basis), left.tobytes(), right.tobytes()))
            if gram:
                if action == "zero":
                    left[:] = 0
                elif action == "return":
                    return 7 * np.eye(len(left), dtype=left.dtype)
                elif action == "raise":
                    raise sentinel
            return stock_dot(left, right, *args, **kwargs)

        def fitted(estimator):
            events.append(("validate", estimator.components_ is basis))
            if install == "during_validation":
                np.dot = observed
            return True

        with monkeypatch.context() as patch, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            patch.setattr(np, "dot", observed if install == "before_transform" else stock_dot)
            patch.setattr(NMF, "__sklearn_is_fitted__", fitted, raising=False)
            if public_classifier:
                project, values = model.transform, query
            else:
                project, values = model.decomposition_.transform, query[:, model.selected_features_]
            if action == "raise":
                with pytest.raises(RuntimeError) as raised:
                    project(values)
                assert raised.value is sentinel
                coordinates = None
            else:
                coordinates = project(values)
        return (events, [(w.category, str(w.message)) for w in caught],
                basis.tobytes(), coordinates)

    reference = observe(False)
    actual = observe(True)
    kinds = [event[0] for event in reference[0]]
    assert kinds.count("gram") == 1 and kinds.index("validate") < kinds.index("gram")
    assert next(event for event in reference[0] if event[0] == "gram")[3]
    assert actual[:3] == reference[:3]
    if action != "raise":
        assert actual[3].dtype == reference[3].dtype
        np.testing.assert_array_equal(actual[3], reference[3])


@pytest.mark.parametrize("second_action", ["raise", "install_dot"])
def test_loaded_basis_reduction_runs_once(second_action, monkeypatch, tmp_path):
    model, query = ordinary_problem()
    path = tmp_path / "classifier.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path, mmap_mode="r")
    basis = loaded.decomposition_.components_
    assert type(basis) is np.memmap and not basis.flags.writeable
    before = basis.tobytes()
    # The historical batch path checks the retained basis once and then calls
    # the genuine public estimator. sklearn validates its own ndarray view.
    expected = loaded.decomposition_.transform(query[:, loaded.selected_features_])
    stock_dot = np.dot
    original_any = basis.any
    calls = []
    sentinel = RuntimeError("repeated reduction on a retained joblib basis")

    @wraps(stock_dot)
    def observed_dot(*args, **kwargs):
        return stock_dot(*args, **kwargs)

    def observed_any(*args, **kwargs):
        calls.append("any")
        if len(calls) > 1:
            if second_action == "raise":
                raise sentinel
            np.dot = observed_dot
        return original_any(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "dot", stock_dot)
        patch.setattr(basis, "any", observed_any)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = loaded.transform(query)
        assert np.dot is stock_dot
    assert calls == ["any"]
    assert loaded.decomposition_.components_ is basis and basis.tobytes() == before
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)
