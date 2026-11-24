"""Use sklearn's input-column contract alongside the existing numeric validators."""

from importlib import import_module

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data


def _check_input_features(estimator, X, *, reset):
    if type(X) is np.ndarray and not hasattr(estimator, "feature_names_in_"):
        return
    # Width and values are checked by the existing numeric paths. Avoid
    # converting X twice, including for array-like containers without a shape.
    validate_data(estimator, X, reset=reset, skip_check_array=True, ensure_2d=False)
    if reset and hasattr(estimator, "feature_names_in_"):
        names = estimator.feature_names_in_
        if len(set(names)) != len(names):
            raise ValueError("Named input requires unique feature names.")
        estimator.feature_names_in_ = names.copy()


def _input_feature_names(X):
    """Stage training names without publishing learned state before fit succeeds."""
    if type(X) is np.ndarray:
        return None
    metadata = BaseEstimator()
    _check_input_features(metadata, X, reset=True)
    return getattr(metadata, "feature_names_in_", None)


def _named_frame_backend():
    """Choose a table library for rebuilding the validated numeric snapshot."""
    for name in ("pandas", "polars", "pyarrow"):
        try:
            return import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
    raise TypeError("Named numeric workflows require pandas, polars or pyarrow "
                    "to preserve validated column names. For positional input, supply an array.")


def _named_matrix(values, feature_names):
    if feature_names is None:
        return values
    backend = _named_frame_backend()
    if backend.__name__ == "pandas":
        return backend.DataFrame(values, columns=feature_names.copy(), copy=False)
    if backend.__name__ == "polars":
        return backend.DataFrame(values, schema=feature_names.tolist(), orient="row")
    return backend.table({name: values[:, index] for index, name in enumerate(feature_names)})
