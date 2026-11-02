"""Validate row and subject boundaries for Chang Cheng's classification studies."""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def validate_metadata(values, n_samples, *, name, unique=False):
    """Return one scalar ID per row, rejecting missing or ambiguous metadata.

    IDs may be Python/NumPy integers, finite floating-point scalars, or
    nonblank strings. Boolean, complex, temporal, and other objects are
    rejected. Numeric and string IDs may not be mixed in one array. Values
    retain their precision; overlap/group checks use exact numerical identity.
    This validates supplied values, not their origin before caller coercion.
    """
    if values is None:
        return None
    invalid_type = (f"{name} must contain integer, finite floating-point, or nonempty string IDs; "
                    "Boolean, complex, and temporal IDs are not supported.")
    # Object conversion can turn nanosecond temporal arrays into integers.
    if isinstance(values, np.ndarray) and values.dtype.kind in "mM":
        raise ValueError(invalid_type)
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != n_samples:
        raise ValueError(f"{name} must match rows as a one-dimensional array.")
    string_ids = []
    for value in array:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f"{name} must not contain missing IDs.")
            string_ids.append(True)
        elif isinstance(value, (bool, np.bool_, np.timedelta64)):
            # bool and timedelta64 inherit integer behavior but are not IDs.
            raise ValueError(invalid_type)
        elif isinstance(value, (int, np.integer)):
            # Integers are finite; a float conversion could lose identity.
            string_ids.append(False)
        elif isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must not contain missing or nonfinite IDs.")
            string_ids.append(False)
        else:
            raise ValueError(invalid_type)
    if string_ids and any(string_ids) and not all(string_ids):
        raise ValueError(f"{name} must not mix numeric and string IDs.")
    if unique and len(set(_metadata_keys(array).tolist())) != len(array):
        raise ValueError(f"{name} must contain unique IDs.")
    return array


def _metadata_keys(values):
    """Return exact comparison keys for a validated object array of IDs.

    NumPy can narrow mixed numeric comparisons and long-double hashes. Use
    Python numeric keys internally; keep original scalars for reports/callers.
    Builtin-only arrays need no copy, and integral extended values need no
    Fraction. Construct keys once when reusing them across classes or folds.
    """
    if not len(values) or isinstance(values[0], str):
        # Validated string families never contain numeric scalars.
        return values
    keys = values
    for index, value in enumerate(values):
        if isinstance(value, np.number):
            item = value.item()
            if isinstance(item, np.generic):
                numerator, denominator = value.as_integer_ratio()
                item = numerator if denominator == 1 else Fraction(numerator, denominator)
            if keys is values:
                keys = values.copy()
            keys[index] = item
    return keys


def _indices(values, n_samples, role):
    array = np.asarray(values)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"Each {role} fold must contain a nonempty index array.")
    if array.dtype.kind not in "iu":
        raise ValueError(f"Each {role} fold must use integer indices.")
    if np.any(array < 0) or np.any(array >= n_samples):
        raise ValueError(f"Each {role} fold must use in-range indices.")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"Each {role} fold must not contain duplicate indices.")
    return array.astype(np.intp, copy=True)


def build_cv_folds(X, y, *, cv=5, seed=0, groups=None, folds=None):
    """Build or validate an exhaustive binary-classification CV partition.

    Without ``groups``, use shuffled stratified folds. With one subject ID per
    row, use shuffled stratified group folds and reject subject overlap. Custom
    ``folds`` must contain exactly ``cv`` pairs of training/validation indices,
    or dictionaries with ``training_indices`` and ``validation_indices``.
    Each validation row must occur once, each training fold must be its
    complement, and both classes must occur in every validation fold. At least
    two rows per class are required in each training fold for screening.
    """
    y = np.asarray(y)
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("X and y must have matching rows and y must be one-dimensional.")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Calibration currently requires exactly two classes.")
    if (isinstance(cv, (bool, np.bool_)) or not isinstance(cv, Integral)
            or cv < 2 or counts.min() < cv):
        raise ValueError("Every class must have at least cv samples; cv must be >= 2.")
    cv = int(cv)
    groups = validate_metadata(groups, len(y), name="groups")
    group_keys = None if groups is None else _metadata_keys(groups)
    if group_keys is not None and len(set(group_keys.tolist())) < cv:
        raise ValueError("At least cv distinct groups are required.")
    if folds is None:
        splitter = (StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
                    if groups is None else
                    StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed))
        folds = splitter.split(X, y) if groups is None else splitter.split(X, y, group_keys)
    folds = list(folds)
    if len(folds) != cv:
        raise ValueError("Explicit folds must contain exactly cv splits.")
    normalized = []
    validation_counts = np.zeros(len(y), dtype=np.intp)
    all_indices = np.arange(len(y))
    for split in folds:
        if isinstance(split, dict):
            if "training_indices" not in split or "validation_indices" not in split:
                raise ValueError("Fold dictionaries require training_indices and validation_indices.")
            train, valid = split["training_indices"], split["validation_indices"]
        else:
            if not isinstance(split, (tuple, list)) or len(split) != 2:
                raise ValueError("Each fold must be a training/validation index pair.")
            train, valid = split
        train = _indices(train, len(y), "training")
        valid = _indices(valid, len(y), "validation")
        if np.intersect1d(train, valid).size:
            raise ValueError("Training and validation indices must not overlap.")
        if not np.array_equal(np.sort(np.concatenate([train, valid])), all_indices):
            raise ValueError("Every training fold must be the complement of its validation fold.")
        if group_keys is not None and set(group_keys[train].tolist()) & set(group_keys[valid].tolist()):
            raise ValueError("Training and validation groups must not overlap.")
        if any(np.sum(y[valid] == label) == 0 for label in classes):
            raise ValueError("Every validation fold must contain both classes.")
        if any(np.sum(y[train] == label) < 2 for label in classes):
            raise ValueError("Every training fold requires at least two samples per class for screening.")
        validation_counts[valid] += 1
        normalized.append((train, valid))
    if not np.all(validation_counts == 1):
        raise ValueError("Validation folds must partition the rows exactly once.")
    return normalized
