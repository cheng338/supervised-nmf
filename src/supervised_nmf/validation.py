"""Validate row and subject boundaries for Chang Cheng's classification studies."""

from __future__ import annotations

from fractions import Fraction
import sys
import math
from numbers import Integral

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ._labels import _sklearn_labels, _validate_class_labels


# Capture the stock scalar class independently of the mutable public alias.
_LONGDOUBLE_SCALAR_TYPE = np.dtype("longdouble").type


# Native format discovery is only needed by exact long-double key conversion.
# Publish a complete immutable tuple, never a partially initialized route.
_NATIVE_KEY_FORMAT = None


def _native_key_format():
    global _NATIVE_KEY_FORMAT
    cached = _NATIVE_KEY_FORMAT
    if cached is None:
        info = np.finfo(np.longdouble)
        extended_fits_float = (
            sys.float_info.radix == 2
            and info.nmant + 1 <= sys.float_info.mant_dig
            and info.maxexp <= sys.float_info.max_exp
            and info.minexp - info.nmant
                >= sys.float_info.min_exp - sys.float_info.mant_dig
        )
        float_key_route = (sys.float_info.radix == 2
                           and sys.float_info.max_exp >= sys.float_info.mant_dig)
        numerator_limit = 1 << sys.float_info.mant_dig
        numerator_lower = -numerator_limit
        denominator_exponent_max = sys.float_info.mant_dig - sys.float_info.min_exp
        cached = (extended_fits_float, float_key_route, numerator_lower,
                  numerator_limit, denominator_exponent_max)
        _NATIVE_KEY_FORMAT = cached
    return cached


def _validate_threads(value):
    """Normalize execution counts without narrowing the native C-int argument."""
    maximum = int(np.iinfo(np.intc).max)
    if (isinstance(value, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(value, Integral)):
        raise ValueError(f"threads must be a positive integer no greater than {maximum}.")
    value = int(value)
    if not 1 <= value <= maximum:
        raise ValueError(f"threads must be a positive integer no greater than {maximum}.")
    return value


def _validate_refit(value):
    """Accept logical scalars without coercing numeric or array truth values."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("refit must be boolean.")
    return bool(value)


def _validate_fixed_seed(value):
    """Normalize the fixed split seed required by shared and rank searches."""
    message = "seed must be a fixed integer between 0 and 2**32 - 1."
    if (isinstance(value, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(value, Integral)):
        raise ValueError(message)
    value = int(value)
    if not 0 <= value <= 2 ** 32 - 1:
        raise ValueError(message)
    return value


def validate_metadata(values, n_samples, *, name, unique=False):
    """Return one scalar ID per row, rejecting missing or ambiguous metadata.

    IDs may be Python/NumPy integers, finite floating-point scalars, or
    nonblank strings. Boolean, complex, temporal, and other objects are
    rejected. Numeric and string IDs may not be mixed in one array. Values
    retain their precision; overlap/group checks use exact numerical identity.
    This validates supplied values, not their origin before caller coercion.
    """
    return _validate_metadata_snapshot(values, n_samples, name=name, unique=unique)[0]


def _validate_metadata_snapshot(values, n_samples, *, name, unique=False):
    """Keep returned scalar IDs and native checks tied to one owned snapshot."""
    if values is None:
        return None, None
    invalid_type = (f"{name} must contain integer, finite floating-point, or nonempty string IDs; "
                    "Boolean, complex, and temporal IDs are not supported.")
    # Object conversion can turn nanosecond temporal arrays into integers.
    if isinstance(values, np.ndarray) and values.dtype.kind in "mM":
        raise ValueError(invalid_type)
    # Both returned scalar objects and vector checks derive from this
    # same owned numeric snapshot, even if the caller mutates its array later.
    numeric_values = (values.copy() if type(values) is np.ndarray
                      and values.dtype.kind in "iuf" else None)
    if numeric_values is not None and numeric_values.dtype.kind in "mM":
        raise ValueError(invalid_type)
    array = np.asarray(values if numeric_values is None else numeric_values, dtype=object)
    if array.ndim != 1 or len(array) != n_samples:
        raise ValueError(f"{name} must match rows as a one-dimensional array.")
    # Only exact ndarray inputs take this route; subclasses retain their
    # existing conversion and scalar-validation behavior. Homogeneous finite
    # integer/real arrays need no mixed-type canonical keys for uniqueness.
    if numeric_values is not None and numeric_values.dtype.kind in "iuf":
        if numeric_values.dtype.kind == "f" and not np.isfinite(numeric_values).all():
            raise ValueError(f"{name} must not contain missing or nonfinite IDs.")
        if unique and np.unique(numeric_values).size != len(numeric_values):
            raise ValueError(f"{name} must contain unique IDs.")
        return array, numeric_values
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
    return array, None

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
    if (type(values) is np.ndarray and values.ndim == 1
            and values.dtype.kind == "O"
            and type(values[0]) is _LONGDOUBLE_SCALAR_TYPE):
        return _metadata_keys_native(values)
    return _metadata_keys_original(values)


def _metadata_keys_original(values):
    """Run the original loop after the entrypoint's early return."""
    keys = values
    native_format = None
    extended_fits_float = False
    for index, value in enumerate(values):
        if isinstance(value, np.number):
            item = value.item()
            if isinstance(item, np.generic):
                if native_format is None and type(value) is np.longdouble:
                    native_format = _native_key_format()
                    (extended_fits_float, float_key_route, numerator_lower,
                     numerator_limit, denominator_exponent_max) = native_format
                if extended_fits_float and type(value) is np.longdouble:
                    item = float(value)
                else:
                    numerator, denominator = value.as_integer_ratio()
                    if denominator == 1:
                        # Preserve the original integral fast path.
                        item = numerator
                    elif type(value) is not np.longdouble:
                        # Scalar subclasses may override the ratio protocol.
                        # Preserve its original Fraction/error semantics.
                        item = Fraction(numerator, denominator)
                    else:
                        # Exact native nonintegral ratios are reduced dyadics:
                        # this bound is at most float precision, and denominator
                        # >= 2 makes the cached upper-range proof sufficient.
                        fits_float = (float_key_route
                                      and numerator_lower < numerator < numerator_limit)
                        if fits_float:
                            exponent = denominator.bit_length() - 1
                            fits_float = exponent <= denominator_exponent_max
                        item = (math.ldexp(float(numerator), -exponent) if fits_float
                                else Fraction(numerator, denominator))
            if keys is values:
                keys = values.copy()
            keys[index] = item
    return keys


def _metadata_keys_native(values):
    """Keep per-value protocols after first-stock object-array admission."""
    keys = values
    native_format = None
    extended_fits_float = False
    for index, value in enumerate(values):
        if isinstance(value, np.number):
            item = (value if type(value) is _LONGDOUBLE_SCALAR_TYPE
                    else value.item())
            if isinstance(item, np.generic):
                if native_format is None and type(value) is np.longdouble:
                    native_format = _native_key_format()
                    (extended_fits_float, float_key_route, numerator_lower,
                     numerator_limit, denominator_exponent_max) = native_format
                if extended_fits_float and type(value) is np.longdouble:
                    item = float(value)
                else:
                    numerator, denominator = value.as_integer_ratio()
                    if denominator == 1:
                        # Preserve the original integral fast path.
                        item = numerator
                    elif type(value) is not np.longdouble:
                        # Scalar subclasses may override the ratio protocol.
                        # Preserve its original Fraction/error semantics.
                        item = Fraction(numerator, denominator)
                    else:
                        # Exact native nonintegral ratios are reduced dyadics:
                        # this bound is at most float precision, and denominator
                        # >= 2 makes the cached upper-range proof sufficient.
                        fits_float = (float_key_route
                                      and numerator_lower < numerator < numerator_limit)
                        if fits_float:
                            exponent = denominator.bit_length() - 1
                            fits_float = exponent <= denominator_exponent_max
                        item = (math.ldexp(float(numerator), -exponent) if fits_float
                                else Fraction(numerator, denominator))
            if keys is values:
                keys = values.copy()
            keys[index] = item
    return keys


_NATIVE_ID_TYPES = (
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64, np.longdouble,
)


def _exact_metadata_promotion(left_dtype, right_dtype, left_bounds, right_bounds):
    """Prove native comparison casts exact without rewriting report scalars.

    Callers supply supported native dtype descriptors and exact Python-key
    endpoint bounds from their owned snapshots. Unknown floating formats keep
    the canonical-key route; item size alone is never a precision proof.
    """
    target = np.dtype(np.longdouble)
    info = np.finfo(target)
    # Uniform binary formats only; IBM double-double is deliberately absent.
    formats = ((10, -14, 16), (23, -126, 128), (52, -1022, 1024),
               (63, -16382, 16384), (112, -16382, 16384))
    target_format = (info.nmant, info.minexp, info.maxexp)
    if target_format not in formats:
        return None
    precision = info.nmant + 1
    for dtype, bounds in ((left_dtype, left_bounds), (right_dtype, right_bounds)):
        if dtype.type not in _NATIVE_ID_TYPES:
            return None
        if dtype.kind == "f":
            source = np.finfo(dtype)
            if ((source.nmant, source.minexp, source.maxexp) not in formats
                    or source.nmant > info.nmant
                    or source.maxexp > info.maxexp
                    or source.minexp - source.nmant < info.minexp - info.nmant):
                return None
        elif dtype.kind in "iu":
            bits = dtype.itemsize * 8
            if precision >= bits and info.maxexp >= bits and info.minexp <= 0:
                continue
            # All integers in this closed interval are exactly representable.
            # Bounds are exact Python integers, never mixed NumPy comparisons.
            limit = 1 << precision
            if (bounds is None or info.maxexp <= precision or info.minexp > 0
                    or bounds[0] < -limit or bounds[1] > limit):
                return None
        else:
            return None
    return target


def _native_float_key_set(values):
    """Return bulk exact float keys, or None to keep canonical conversion.

    Callers provide finite owned native comparison arrays only after an exact
    promotion was rejected. Unknown native formats retain the original route.
    """
    if type(values) is not np.ndarray or values.dtype.type is not np.longdouble:
        return None
    info = np.finfo(values.dtype)
    if ((info.nmant, info.minexp, info.maxexp) not in (
            (10, -14, 16), (23, -126, 128), (52, -1022, 1024))
            or not _native_key_format()[0]):
        return None
    # Known uniform native formats fit binary64; the existing format proof
    # separately establishes exact Python-float keys. Integer keys never cast.
    # Equal storage dtypes can still have different native scalar classes.
    # A real float64 copy makes tolist produce the proven Python-float keys.
    return set(values.astype(np.float64, copy=True).tolist())


# Sample-ID comparison identities:
#   (None, keys, numeric): exact keys and an owned primitive snapshot.
#   (ordered, None, dtype): owned sorted native values and a dtype marker.
#   (None, keys, None): canonical keys without native promotion.
# Groups and already-promoted comparisons use only (ordered, keys),
# with either field optionally None. Their two-field form suppresses
# sample promotion; the third sample field is not uniformly an array.
def _validated_unique_metadata(values, n_samples, *, name):
    """Retain exact uniqueness data without borrowing the caller's storage."""
    array, numeric = _validate_metadata_snapshot(values, n_samples, name=name)
    if array is None:
        return None, None
    if numeric is not None and numeric.dtype.type in _NATIVE_ID_TYPES:
        # Object conversion of these exact primitive dtypes produces Python
        # int/float values. Native long doubles retain the sorted route.
        if numeric.dtype.type is not np.longdouble:
            keys = set(array.tolist())
            if len(keys) != len(array):
                raise ValueError(f"{name} must contain unique IDs.")
            return array, (None, keys, numeric)
        ordered = np.unique(numeric)
        if ordered.size != len(array):
            raise ValueError(f"{name} must contain unique IDs.")
        return array, (ordered, None, numeric.dtype)
    keys = set(_metadata_keys(array).tolist())
    if len(keys) != len(array):
        raise ValueError(f"{name} must contain unique IDs.")
    return array, (None, keys, None)


def _metadata_overlap(left_array, left_identity, right_array, right_identity):
    """Compare native values only within one dtype; mixed types keep exact keys."""
    left_native, left_keys = left_identity[0], left_identity[1]
    right_native, right_keys = right_identity[0], right_identity[1]
    if (left_native is not None and right_native is not None
            and left_native.dtype == right_native.dtype):
        if not left_native.size or not right_native.size:
            return False
        # Strict bounds prove disjointness without changing touching endpoints.
        if left_native[-1] < right_native[0] or right_native[-1] < left_native[0]:
            return False
        # Reuse the sorted uniqueness results; index only in-range positions.
        if left_native.size > right_native.size:
            left_native, right_native = right_native, left_native
        positions = np.searchsorted(right_native, left_native)
        within = positions < right_native.size
        return bool(np.any(right_native[positions[within]] == left_native[within]))
    # Three fields identify the sample protocol. A non-None third field
    # on both sides admits native promotion: primitive values come from
    # the owned snapshot; sorted native values carry their own dtype.
    # Groups and already-promoted recursion have two fields and skip it.
    if ((left_native is not None or right_native is not None)
            and len(left_identity) == 3 and len(right_identity) == 3
            and left_identity[2] is not None and right_identity[2] is not None):
        left_dtype = (left_native.dtype if left_native is not None else left_identity[2].dtype)
        right_dtype = (right_native.dtype if right_native is not None else right_identity[2].dtype)
        left_bounds = (None if not left_native.size else
                       _metadata_keys(np.asarray(left_native[[0, -1]], dtype=object))) if left_native is not None else (
                           None if not left_keys else (min(left_keys), max(left_keys)))
        right_bounds = (None if not right_native.size else
                        _metadata_keys(np.asarray(right_native[[0, -1]], dtype=object))) if right_native is not None else (
                            None if not right_keys else (min(right_keys), max(right_keys)))
        if left_bounds is None or right_bounds is None:
            return False
        if left_bounds[1] < right_bounds[0] or right_bounds[1] < left_bounds[0]:
            return False
        comparison_dtype = _exact_metadata_promotion(
            left_dtype, right_dtype, left_bounds, right_bounds)
        if comparison_dtype is not None:
            if left_native is not None:
                left_comparison = left_native.astype(comparison_dtype, copy=False)
            else:
                left_comparison = np.sort(left_identity[2].astype(comparison_dtype, copy=False))
            if right_native is not None:
                right_comparison = right_native.astype(comparison_dtype, copy=False)
            else:
                right_comparison = np.sort(right_identity[2].astype(comparison_dtype, copy=False))
            # Sample uniqueness is already established; exact widening and
            # sorting cannot merge values or alter the original scalar IDs.
            return _metadata_overlap(left_array, (left_comparison, None),
                                     right_array, (right_comparison, None))
        if left_keys is None:
            left_keys = _native_float_key_set(left_native)
        if right_keys is None:
            right_keys = _native_float_key_set(right_native)
    if left_keys is None:
        left_keys = set(_metadata_keys(left_array).tolist())
    if right_keys is None:
        right_keys = set(_metadata_keys(right_array).tolist())
    return bool(left_keys & right_keys)

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
    Class labels must not end in a NUL codepoint; group IDs are metadata and
    retain their separate validation rules.
    """
    raw_y = y
    y = np.asarray(y)
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("X and y must have matching rows and y must be one-dimensional.")
    classes, counts = np.unique(y, return_counts=True)
    _validate_class_labels(raw_y, y, classes=classes)
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
        split_y = _sklearn_labels(y)
        folds = splitter.split(X, split_y) if groups is None else splitter.split(X, split_y, group_keys)
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
