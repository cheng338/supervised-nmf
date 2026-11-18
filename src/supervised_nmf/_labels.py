"""Keep class-label validation ahead of lossy string conversions."""

from itertools import repeat

import numpy as np
from sklearn.utils.validation import check_X_y


_TERMINAL_NUL_MESSAGE = (
    "Class labels ending in a NUL codepoint (\\x00) are not supported; "
    "encode the original labels as distinct integers before array conversion."
)


def _label_values(y):
    # Object strings compare their full contents on NumPy versions whose
    # StringDType comparisons stop at NUL. They also support sklearn metrics
    # that attach metadata to the target dtype.
    if issubclass(type(y), np.ndarray) and y.dtype.kind == "T":
        return np.asarray(y).astype(object)
    return y


def _check_labeled_data(X, y, **options):
    """Return validated X, working labels, and labels in the input dtype."""
    target = _label_values(y)
    X, checked = check_X_y(X, target, **options)
    if target is y:
        return X, _label_values(checked), checked
    return X, checked, np.asarray(y).reshape(checked.shape)


def _validate_class_label(value):
    """Check a scalar without a str() conversion that could discard its NUL."""
    if issubclass(type(value), str) and str.endswith(value, "\x00"):
        raise ValueError(_TERMINAL_NUL_MESSAGE)


def _sequence_items(values):
    """Read list/tuple storage without invoking subclass iteration overrides."""
    if issubclass(type(values), list):
        return list.__iter__(values)
    return tuple.__iter__(values)


def _raw_terminal_nul(values):
    """Inspect flat or one-column list/tuple storage without custom dispatch."""
    try:
        return any(map(str.endswith, _sequence_items(values), repeat("\x00")))
    except TypeError:
        # Mixed values and one-column targets need a bounded fallback. Invalid
        # cyclic/deeper structures are left to ordinary shape validation.
        for value in _sequence_items(values):
            if issubclass(type(value), str):
                if str.endswith(value, "\x00"):
                    return True
            elif issubclass(type(value), (list, tuple)):
                for scalar in _sequence_items(value):
                    if issubclass(type(scalar), str) and str.endswith(scalar, "\x00"):
                        return True
        return False


def _validate_class_labels(raw_y, y=None, *, classes=None):
    """Return labels after checking their domain, without imposing shape/counts.

    Keep raw_y across ordinary validation, then pass its validated array as y.
    Existing unique classes may be supplied to avoid scanning an object target.
    Without y this helper applies np.asarray once and returns that array, for
    label-only boundaries such as fingerprints and comparison preflight.

    Numeric inputs need no element scan. Object/StringDType values retain exact
    strings. Fixed-width Unicode can erase terminal NUL, so inspect the retained
    original list/tuple in that case. Padding cannot reveal information already
    lost in a caller-created fixed-width array or custom conversion.
    """
    if issubclass(type(raw_y), str):
        _validate_class_label(raw_y)
    if y is None:
        y = np.asarray(raw_y)
    known = y if classes is None else classes
    if (known.dtype.kind in {"O", "T"}
            and any(issubclass(type(value), str) and str.endswith(value, "\x00")
                    for value in known.flat)):
        raise ValueError(_TERMINAL_NUL_MESSAGE)
    if (y.dtype.kind == "U"
            and issubclass(type(raw_y), (list, tuple))
            and _raw_terminal_nul(raw_y)):
        raise ValueError(_TERMINAL_NUL_MESSAGE)
    return y
