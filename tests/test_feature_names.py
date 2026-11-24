"""The fitted input schema belongs to each estimator consuming raw columns."""

import pickle

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from supervised_nmf import BinaryScreening, FeatureSPA, NMFClassifier
from supervised_nmf import _feature_names
from supervised_nmf.evaluation import calibration_search, evaluate_classifier


@pytest.fixture
def named_problem():
    pd = pytest.importorskip("pandas")
    y = np.arange(24) % 2
    X = np.random.default_rng(317).uniform(0.1, 1.1, (len(y), 5))
    X[:, :2] += y[:, None] * [1.5, 2.0]
    X[:, 2] = 0.5
    return pd.DataFrame(X, columns=["A", "B", "constant", "β", "tail"],
                        index=100 + np.arange(len(y))), y


@pytest.mark.parametrize("template", [
    BinaryScreening(score_threshold=0.0),
    FeatureSPA(n_components=2),
    NMFClassifier(n_components=2, score_threshold=0.0, random_state=3),
])
def test_named_transform_and_refit(template, named_problem):
    X, y = named_problem
    model = clone(template).fit(X, y)
    control = clone(template).fit(X.to_numpy(), y)
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    np.testing.assert_allclose(model.transform(X), control.transform(X.to_numpy()))
    restored = pickle.loads(pickle.dumps(model))
    assert not hasattr(clone(restored), "feature_names_in_")
    for bad in (X.iloc[:, ::-1], X.rename(columns={"A": "unseen"}),
                X.iloc[:, :-1], X.assign(extra=1)):
        with pytest.raises(ValueError, match="feature names"):
            restored.transform(bad)
    with pytest.warns(UserWarning, match="feature names"):
        restored.transform(X.to_numpy())
    with pytest.raises(ValueError):
        model.fit(-X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not hasattr(model, "feature_names_in_")

    with pytest.raises((ValueError, TypeError)):
        model.fit(X.set_axis(["A", "A", "C", "D", "E"], axis=1), y)
    with pytest.raises(TypeError):
        model.fit(X.set_axis(["A", 1, 2, 3, 4], axis=1), y)
    model.fit(X.set_axis(range(X.shape[1]), axis=1), y)
    assert not hasattr(model, "feature_names_in_")
    with pytest.warns(UserWarning, match="feature names"):
        model.transform(X)


def test_classifier_joint_prediction_and_fitted_screening(named_problem):
    X, y = named_problem
    template = NMFClassifier(n_components=2, score_threshold=0.0, n_neighbors=3,
                             random_state=3)
    model = template.fit(X, y)
    labels, probabilities = model.predict_with_probabilities(X)
    np.testing.assert_array_equal(labels, model.predict(X))
    np.testing.assert_array_equal(probabilities, model.predict_proba(X))
    for method in (model.predict, model.predict_proba, model.predict_with_probabilities,
                   model.selector_.transform):
        with pytest.raises(ValueError, match="feature names"):
            method(X.iloc[:, ::-1])
    wanted_names = np.asarray(X.columns)[model.get_support()]
    np.testing.assert_array_equal(model.selector_.get_feature_names_out(), wanted_names)
    with pytest.raises(ValueError, match="input_features"):
        model.selector_.get_feature_names_out(X.columns[::-1])
    assert not np.shares_memory(model.selector_.feature_names_in_, model.feature_names_in_)

    # A caller-owned cached selector must retain its original fitted schema.
    donor = BinaryScreening(score_threshold=0.0).fit(X.to_numpy(), y)
    cached = clone(template)._fit_with_screening(X, y, donor)
    assert not hasattr(donor, "feature_names_in_")
    np.testing.assert_array_equal(cached.selector_.get_feature_names_out(), wanted_names)
    with pytest.raises(ValueError, match="feature names"):
        clone(template)._fit_with_screening(X.iloc[:, ::-1], y, model.selector_)


def test_standalone_component_output_and_internal_projection(named_problem):
    X, y = named_problem
    screen = BinaryScreening(score_threshold=0.0)
    spa = FeatureSPA(n_components=2)
    control = make_pipeline(clone(screen), clone(spa)).fit_transform(X, y)
    pipeline = make_pipeline(screen, spa).set_output(transform="pandas")
    projected = pipeline.fit_transform(X, y)
    assert projected.index.equals(X.index)
    assert list(projected.columns) == ["featurespa0", "featurespa1"]
    np.testing.assert_array_equal(pipeline.get_feature_names_out(), projected.columns)
    np.testing.assert_array_equal(spa.feature_names_in_, screen.get_feature_names_out())
    np.testing.assert_allclose(projected.to_numpy(), control)
    with config_context(transform_output="pandas"):
        model = NMFClassifier(n_components=2, score_threshold=0.0)
        assert isinstance(model.fit_transform(X, y), np.ndarray)
        assert isinstance(model.transform(X), np.ndarray)


@pytest.mark.parametrize("solver,mu_projection", [("cd", "auto"), ("mu", "rowwise"), ("mu", "batch")])
@pytest.mark.parametrize("normalization", ["none", "sd"])
@pytest.mark.filterwarnings("error:.*feature names.*:UserWarning")
def test_screened_numeric_projection_in_pandas_output(solver, mu_projection, named_problem, normalization):
    X, y = named_problem
    template = NMFClassifier(solver=solver, mu_projection=mu_projection, n_components=2,
                             score_threshold=0.0, random_state=5, weights="distance",
                             max_iter=2000, knn_normalization=normalization)
    control = clone(template).fit(X.to_numpy(), y)
    expected_labels, expected_proba = control.predict_with_probabilities(X.to_numpy())
    with config_context(transform_output="pandas"):
        model = clone(template).fit(X, y)
        coords = model.transform(X)
        assert isinstance(coords, np.ndarray)
        labels, proba = model.predict_with_probabilities(X)
        np.testing.assert_array_equal(labels, model.predict(X))
        with pytest.raises(ValueError, match="feature names"):
            model.predict(X.iloc[:, ::-1])
    np.testing.assert_allclose(coords, control.transform(X.to_numpy()), rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_allclose(proba, expected_proba, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("fit_mode,query_mode", [("pandas", "default"), ("default", "pandas")])
@pytest.mark.filterwarnings("error:.*feature names.*:UserWarning")
def test_logistic_head_uses_positional_coordinates_across_output_modes(named_problem, fit_mode, query_mode):
    X, y = named_problem
    template = NMFClassifier(n_components=2, classifier="logistic", random_state=3, score_threshold=0.0)
    reference = clone(template).fit(X.to_numpy(), y)
    with config_context(transform_output=fit_mode):
        model = clone(template).fit(X, y)
    with config_context(transform_output=query_mode):
        np.testing.assert_array_equal(model.predict(X), reference.predict(X.to_numpy()))
        np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(X.to_numpy()))


@pytest.mark.parametrize("library", ["polars", "pyarrow"])
def test_optional_named_frame_backend(library, monkeypatch):
    backend = pytest.importorskip(library)
    def optional_import(name):
        if name == library:
            return backend
        raise ModuleNotFoundError(name=name)
    monkeypatch.setattr(_feature_names, "import_module", optional_import)
    values = np.tile([[0., 1.], [1., 0.]], (8, 1))
    names = np.array(["a", "b"], dtype=object)
    X = _feature_names._named_matrix(values, names)
    swapped = _feature_names._named_matrix(values[:, ::-1], names[::-1])
    y = np.arange(len(values)) % 2
    trained = calibration_search(LogisticRegression(), {}, X, y, cv=2)
    np.testing.assert_array_equal(trained.estimator.feature_names_in_, names)
    np.testing.assert_array_equal(evaluate_classifier(trained.estimator, X, y)
                                  ["predictions"]["predicted_labels"], y)
    with pytest.raises(ValueError, match="[Ff]eature names"):
        evaluate_classifier(trained.estimator, swapped, y)


def test_unavailable_table_backends_fail_before_checkpoint(monkeypatch, named_problem, tmp_path):
    X, y = named_problem
    def unavailable(name):
        raise ModuleNotFoundError(name=name)
    monkeypatch.setattr(_feature_names, "import_module", unavailable)
    checkpoint = tmp_path / "study" / "calibration.json"
    with pytest.raises(TypeError, match="pandas, polars or pyarrow"):
        calibration_search(LogisticRegression(), {}, X, y, cv=2, checkpoint_path=checkpoint)
    assert not checkpoint.parent.exists()
    # The canonical workflow container is independent of direct name validation.
    fitted = NMFClassifier(n_components=2, screening="none").fit(X, y)
    with pytest.raises(ValueError, match="feature names"):
        fitted.predict(X.iloc[:, ::-1])
    values = X.to_numpy()
    assert _feature_names._named_matrix(values, None) is values
