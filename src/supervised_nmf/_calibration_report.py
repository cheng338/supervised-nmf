"""Check the saved scientific outcomes of a completed CV-only calibration."""

import codecs
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import pairwise, permutations
from pathlib import Path

import numpy as np

from ._calibration import _validate_scoring
from ._calibration_parameters import (
    _check_display, _complex_coordinate, _exact_numpy_text, _extended_display, _matches_infinity, _numpy_bool,
    _refinement_key, _string_items,
)


def _json_digest(value):
    """Hash a JSON value using the calibration checkpoint writer's encoding."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validation_support_bounds(sizes, counts):
    # If a fold contains x rows of class 0, its other size-x rows are class 1.
    # Each validation class is present and its training complement has >=2.
    lower = [max(1, size - (counts[1] - 2)) for size in sizes]
    upper = [min(size - 1, counts[0] - 2) for size in sizes]
    min_total, max_total = sum(lower), sum(upper)
    if any(lo > hi for lo, hi in zip(lower, upper)) or not min_total <= counts[0] <= max_total:
        raise ValueError("Validation folds cannot accommodate the recorded class counts.")
    return [(max(lo, counts[0] - (max_total - hi)), min(hi, counts[0] - (min_total - lo)))
            for lo, hi in zip(lower, upper)]


def _all_score_checks(results):
    """False disproves a conjunction; an unknown check alone cannot prove it."""
    certain = True
    for result in results:
        if result is False:
            return False
        if result is None:
            certain = None
    return certain


def _any_score_check(results):
    """A verified witness takes precedence over other unexamined possibilities."""
    absent = False
    for result in results:
        if result is True:
            return True
        if result is None:
            absent = None
    return absent


def _between_nontrivial_scores(score, largest_class):
    # These extrema are attainable with one correct or one incorrect prediction
    # from the larger class. They also bound every smaller retained support.
    return (1 / largest_class) / 2 <= score <= ((largest_class - 1) / largest_class + 1) / 2


_BALANCED_HIT_SEARCH_LIMIT = 4096


def _may_need_balanced_hits(score, a, b):
    """Whether a large fold could have more inverse hit pairs than we examine."""
    if a + b < 2 * _BALANCED_HIT_SEARCH_LIMIT or score in (0, 0.5, 1):
        return False
    common = math.gcd(a, b)
    denominator = 2 * a * (b // common)
    if denominator.bit_count() == 1 and denominator <= 2**53:
        return False
    # The rational-score window is four ulps wide. Across it, each lattice
    # point can have at most gcd(a,b)+1 distinct bounded hit pairs.
    targets = math.floor(4 * math.ulp(score) * denominator) + 1
    return targets * (common + 1) > _BALANCED_HIT_SEARCH_LIMIT


def _possible_balanced_score(score, a, b):
    """Return True for a scorer witness, False for impossibility, or None at the search limit."""
    if score in (0, 0.5, 1):
        return True
    if not _between_nontrivial_scores(score, max(a, b)):
        return False
    divisor = math.gcd(a, b)
    step_a, step_b = b // divisor, a // divisor
    denominator = 2 * a * step_a
    # The metric rounds both recalls and then their mean. A huge fold can
    # make adjacent rational outcomes indistinguishable in binary64.
    tolerance = 2 * math.ulp(score)
    if tolerance * denominator < 0.25:
        target = round(score * denominator)
        if abs(score - target / denominator) > tolerance:
            return False
        targets = range(target, target + 1)
    else:
        # Both positive recalls are <= twice their mean. Their binary64
        # rounding followed by rounding the mean fits within two ulps of score.
        # Keep integer endpoints exact when several lattice points fit.
        center = Fraction(score) * denominator
        radius = Fraction(tolerance) * denominator
        targets = range(max(0, math.ceil(center - radius)), min(denominator, math.floor(center + radius)) + 1)
    # Find bounded integer solutions of hits_a * step_a + hits_b * step_b
    # == target. Coprime steps give one residue for hits_a.
    inverse = pow(step_a, -1, step_b)
    remaining = _BALANCED_HIT_SEARCH_LIMIT
    for target in targets[:_BALANCED_HIT_SEARCH_LIMIT]:
        residue = (target * inverse) % step_b
        minimum = max(0, -((step_b * b - target) // step_a))
        maximum = min(a, target // step_a)
        first = minimum + (residue - minimum) % step_b
        solutions = range(first, maximum + 1, step_b)
        # For power-of-two supports the rational mean is exactly representable.
        if denominator.bit_count() == 1 and denominator <= 2**53:
            if solutions and target / denominator == score:
                return True
            continue
        for hits_a in solutions[:remaining]:
            hits_b = (target - hits_a * step_a) // step_b
            if (hits_a / a + hits_b / b) / 2 == score:
                return True
        if len(solutions) > remaining:
            # Double rounding can vary as the same total is split between
            # the two classes. Also sample the full unexamined interval.
            rest = solutions[remaining:]
            probes = min(len(rest), _BALANCED_HIT_SEARCH_LIMIT)
            for index in range(probes):
                hits_a = rest[index * (len(rest) - 1) // max(1, probes - 1)]
                hits_b = (target - hits_a * step_a) // step_b
                if (hits_a / a + hits_b / b) / 2 == score:
                    return True
            if probes == len(rest) and target == targets.stop - 1:
                return False
            return None
        remaining -= len(solutions)
    return None if targets.stop - targets.start > _BALANCED_HIT_SEARCH_LIMIT else False


_EXTRA_LABEL_WARNING = "y_pred contains classes not in y_true"
_BALANCED_SUPPORT_METHOD = "confusion-matrix-v5"
_REQUEST_CACHE_METHODS = ("confusion-matrix-v1", "confusion-matrix-v2", "confusion-matrix-v3")
_FOLD_CACHE_METHODS = ("confusion-matrix-v4", _BALANCED_SUPPORT_METHOD)
_BALANCED_SUPPORT_METHODS = (*_REQUEST_CACHE_METHODS, *_FOLD_CACHE_METHODS)


def _scalar_rows_required(target_kind, prediction_kind, custom):
    return target_kind in "OV" or prediction_kind in "OV" or custom != "none"


def _possible_retained_score(score, a, b):
    if a and b:
        return _possible_balanced_score(score, a, b)
    n = a + b
    return bool(n) and score == round(score * n) / n


def _require_score_support(supported, contradiction):
    if supported is None:
        raise ValueError("Balanced accuracy cannot be verified from the recorded class counts within the search limit.")
    if not supported:
        raise ValueError(contradiction)


def _legacy_extra_label_score(score, a, b):
    # Some sklearn versions can drop rows when the union of numeric labels
    # rounds a mixed integer/float label. The old warning carries no counts.
    full = _possible_balanced_score(score, a, b)
    if full is True:
        return True
    largest = max(a, b)
    if not _between_nontrivial_scores(score, largest):
        return False
    if (a + 1) * (b + 1) > 4096:
        # Try a single retained class, then a class with recall zero or one.
        # Re-evaluate the scorer arithmetic to account for double rounding.
        if float(Fraction(score).limit_denominator(largest)) == score:
            return True
        other_recall = int(score > 0.5)
        recall = Fraction(2 * score - other_recall).limit_denominator(largest)
        if (other_recall + float(recall)) / 2 == score:
            return True
        return None
    return _any_score_check(_possible_retained_score(score, i, j)
                            for i in range(a + 1) for j in range(b + 1))


def _joint_support_possible(score_sets, sizes, bounds, total, extra_label_scores=None):
    """Check compatibility with one partition and the recorded first-class total."""
    def matches(index, a):
        b = sizes[index] - a
        ordinary = _all_score_checks(_possible_balanced_score(score, a, b) for score in score_sets[index])
        if ordinary is False or extra_label_scores is None or not extra_label_scores[index]:
            return ordinary
        extra = _all_score_checks(_legacy_extra_label_score(score, a, b) for score in extra_label_scores[index])
        return _all_score_checks((ordinary, extra))

    if all(lo == hi for lo, hi in bounds):
        return _all_score_checks(matches(index, lo) for index, (lo, _) in enumerate(bounds))
    if len(sizes) == 2:
        lo, hi = bounds[0]
        return _any_score_check(_all_score_checks(matches(i, n) for i, n in enumerate((a, total - a)))
                                for a in range(lo, hi + 1))

    proven, possible = [], []
    has_unknown = False
    for index, (lo, hi) in enumerate(bounds):
        if score_sets[index] <= {0, 0.5, 1} and (
                extra_label_scores is None or extra_label_scores[index] <= {0, 0.5, 1}):
            certain = options = range(lo, hi + 1)
        else:
            certain, options = [], []
            for a in range(lo, hi + 1):
                result = matches(index, a)
                if result is not False:
                    options.append(a)
                if result is True:
                    certain.append(a)
            has_unknown |= len(certain) != len(options)
        if not options:
            return False
        proven.append(certain)
        possible.append(options)
    witness = _supports_can_sum_to(proven, total)
    if witness is True or not has_unknown:
        return witness
    # Even the unexplored options may collectively contradict the class total.
    return False if _supports_can_sum_to(possible, total) is False else None


def _supports_can_sum_to(fold_options, total):
    slack = 0
    sparse = []
    for options in fold_options:
        if not options:
            return False
        total -= options[0]
        width = options[-1] - options[0]
        if len(options) == width + 1:
            slack += width
        else:
            sparse.append([a - options[0] for a in options])
    # Folds with contiguous support options can supply any count up to slack.
    lower = max(0, total - slack)
    upper = min(total, sum(options[-1] for options in sparse))
    if lower > upper:
        return False
    if lower == 0:
        return True
    if len(sparse) == 1:
        return any(lower <= a <= upper for a in sparse[0])
    # E.g. thousands of folds can each contribute 0 or 2 extra class
    # members. Their sum is even, regardless of the size of the search.
    stride = math.gcd(*(math.gcd(*options) for options in sparse))
    if stride > 1:
        lower, upper = (lower + stride - 1) // stride, upper // stride
        if lower > upper:
            return False
        sparse = [[delta // stride for delta in options] for options in sparse]
    if all(options[-1] + 1 == len(options) for options in sparse):
        return True
    # Keep legacy sparse subset-sum work bounded. New archives store the
    # observed supports and reach the fixed-count branch above.
    if (upper + 1) * sum(len(options) for options in sparse) > 8_000_000:
        # Widely spaced supports can leave very few distinct sums despite
        # the width of their bitset. Search those sums directly while cheap.
        sums = {0}
        remaining = 65_536
        for options in sparse:
            remaining -= len(sums) * len(options)
            if remaining < 0:
                return None
            sums = {value + delta for value in sums for delta in options if value + delta <= upper}
            if max(sums) >= lower:
                return True  # Unprocessed folds can all contribute zero.
        return False
    reachable = 1
    mask = (1 << (upper + 1)) - 1
    for options in sparse:
        next_reachable = 0
        for delta in options:
            if delta > upper:
                break
            next_reachable |= reachable << delta
        reachable = next_reachable & mask
    return bool(reachable >> lower)


def _recorded_support_bounds(identity, sizes, counts):
    bounds = _validation_support_bounds(sizes, counts)
    if identity.get("fold_evidence_version", 0) < 2:
        return bounds
    observed = identity["validation_class_counts"]
    if type(observed) is not list or len(observed) != len(sizes):
        raise ValueError("Incomplete recorded validation class counts.")
    fixed = []
    for pair, size, (lo, hi) in zip(observed, sizes, bounds):
        if (type(pair) is not list or len(pair) != 2 or any(type(k) is not int for k in pair)
                or sum(pair) != size or not lo <= pair[0] <= hi):
            raise ValueError("Invalid recorded validation class counts.")
        fixed.append((pair[0], pair[0]))
    if sum(a for a, _ in fixed) != counts[0]:
        raise ValueError("Recorded validation class counts disagree with their total.")
    return fixed


_LOGISTIC = "sklearn.linear_model._logistic.LogisticRegression"
# These releases share the checked solver iteration, warning and storage rules.
_AUDITED_SOLVER_VERSIONS = ("1.9.0", "1.9.1")
_NMF_CLASSIFIER = "supervised_nmf.classifier.NMFClassifier"
_NMF_HEADS = {
    "sklearn.neighbors._classification.KNeighborsClassifier": "knn",
    _LOGISTIC: "logistic",
    "sklearn.svm._classes.LinearSVC": "linear_svm",
}


def _plain_setting(node):
    kind = node["kind"]
    if kind == "random_state":
        # The writer reserves this kind for an exact NumPy RandomState.
        # Recognize its ordinary MT19937 state even on NumPy versions
        # which cannot verify its child-stream behavior for resuming fits.
        state = node["state"]
        return (state["kind"] == "tuple" and len(state["items"]) == 5
                and state["items"][0] == {"kind": "str", "value": "MT19937"})
    if kind == "numpy_scalar":
        return node["dtype"]["str"][1:2] in {"b", "i", "u", "f", "U", "S"}
    if kind == "dict":
        return all(_plain_setting(key) and _plain_setting(value)
                   for key, value in node["items"])
    return kind in ("none", "bool", "int", "float", "str")


def _stock_settings(estimator, declaration, display, folds=()):
    if estimator["kind"] != "estimator" or estimator["implementation"]["type"] not in (_NMF_CLASSIFIER, _LOGISTIC):
        return None
    base = _string_items(estimator["parameters"])
    if base is None:
        return None
    effective = base.copy()
    for key, value in declaration["items"]:
        # NumPy string keys are also accepted by sklearn's set_params.
        name = _text_choice(key, base)
        if name is None:
            return None
        # A NumPy integer in the base configuration has only a byte digest.
        # A grid override also has a displayed integer. Use that value for
        # range comparisons once it matches this specific declaration.
        if (name in ("max_iter", "svm_max_iter", "logistic_max_iter", "n_components") and _numpy_integer(value)
                or name == "max_iter" and _ordinary_numpy_float(value)):
            value = _bind_numpy_cap(value, display.get(name))
        effective[name] = value
    if (folds and estimator["implementation"]["type"] == _LOGISTIC
            and _text_choice(effective["solver"], ("lbfgs",)) == "lbfgs"
            and (_ordinary_numpy_float(effective["max_iter"]) or _numpy_integer(effective["max_iter"]))):
        # A callback can run another optimizer. Its warning is a witness for
        # this candidate's cap only if the printed cap matches its byte digest.
        messages = (w["message"] for fold in folds for w in fold["warnings"] if w["category"] == "ConvergenceWarning")
        for message in messages:
            cap = _bind_numpy_cap(effective["max_iter"], _lbfgs_warning_limit(message))
            if cap["kind"] in ("float", "int"):
                effective["max_iter"] = cap
                break
    return effective


def _text_choice(node, choices):
    """Resolve an ordinary Python or NumPy text choice from its saved identity."""
    if node["kind"] == "str":
        return node["value"]
    if (node["kind"] != "numpy_scalar"
            or node["dtype"]["str"][1:] not in {f"U{len(choice)}" for choice in choices}):
        return None
    matches = []
    for choice in choices:
        try:
            _check_display(node, choice)
        except ValueError:
            continue
        matches.append(choice)
    return matches[0] if len(matches) == 1 else None


def _ordinary_numpy_float(node):
    return (node["kind"] == "numpy_scalar" and re.fullmatch(r"[<>]f[248]", node["dtype"]["str"])
            and node["dtype"].get("format", [None])[0] != "g")


def _numpy_integer(node):
    return node["kind"] == "numpy_scalar" and re.fullmatch(r"[<>|][iu][1248]", node["dtype"]["str"])


def _bind_numpy_cap(node, value):
    # The display must reproduce the integer or floating-point scalar's bytes.
    if type(value) is int or type(value) is float and math.isfinite(value):
        try:
            _check_display(node, value)
        except ValueError:
            return node
        if type(value) is int:
            return {"kind": "int", "value": str(value)}
        return {"kind": "float", "bits": struct.pack("!d", value).hex()}
    return node


def _literal_iteration_limit(node):
    if node["kind"] == "int":
        return int(node["value"], 0)
    if node["kind"] == "float":
        bits = bytes.fromhex(node["bits"])
        if len(bits) != 8:
            raise ValueError("Invalid floating-point iteration limit.")
        value, = struct.unpack("!d", bits)
        return value if math.isfinite(value) else None
    if node["kind"] == "bool" and type(node["value"]) is bool:
        return int(node["value"])
    if node["kind"] == "numpy_scalar" and node["dtype"]["str"] == "|b1":
        return int(_numpy_bool(node))
    return None


def _matches_iteration_limit(node, count):
    limit = _literal_iteration_limit(node)
    if limit is not None:
        return count == limit
    if not _numpy_integer(node):
        return None  # This representation cannot establish equality or inequality.
    try:
        _check_display(node, count)
    except ValueError:
        return False
    return True


@lru_cache(maxsize=24)
def _small_integer_caps(dtype):
    # A legacy constructor can store just the dtype and a byte digest. Bound
    # recovery to zero and 256 positive values per integer dtype. Larger
    # caps still support an exact equality check and the dtype's upper bound.
    size = int(dtype[2:])
    signed = dtype[1] == "i"
    maximum = (1 << (8 * size - int(signed))) - 1
    order = "big" if dtype[0] == ">" else "little"
    values = {hashlib.sha256(n.to_bytes(size, order, signed=signed)).hexdigest(): n
              for n in range(min(256, maximum) + 1)}
    return maximum, values


def _at_iteration_cap(node, count, solver="NMF"):
    """Reject a count above a known bound, then test whether it reached the cap."""
    cap = _literal_iteration_limit(node)
    if solver == "L-BFGS" and type(cap) is float:
        # sklearn clips before assigning to a signed int32 array, including
        # for negative caps admitted with parameter validation disabled.
        cap = math.trunc(cap)
        if cap < -(2**31):
            return None  # An out-of-range float-to-int cast has no portable result.
    elif solver == "L-BFGS" and cap is not None and cap < -(2**31):
        # A negative integer cap is always the minimum. Some NumPy versions
        # refuse the int32 assignment, while a successful cast keeps its low 32 bits.
        stored = (cap + 2**31) % 2**32 - 2**31
        if count != stored:
            raise ValueError("L-BFGS iteration count disagrees with the cap's signed int32 representation.")
        return True
    if cap is None:
        if not _numpy_integer(node):
            return None
        maximum, values = _small_integer_caps(node["dtype"]["str"])
        if count > maximum:
            raise ValueError(f"{solver} iteration count exceeds its limit's integer dtype.")
        cap = values.get(node["content"])
        if cap is None:
            exact = _matches_iteration_limit(node, count)
            if not exact and solver == "L-BFGS" and count < 0 and node["dtype"]["str"][1:] == "i8":
                # The constructor digest alone cannot distinguish the signed
                # int64 values which narrow to this negative int32 count.
                return None
            return exact
    if count > cap:
        raise ValueError(f"{solver} iteration count exceeds its declared limit.")
    return count == cap


def _lbfgs_warning_limit(message):
    # sklearn formats min(result.nit, max_iter) and max_iter separately.
    # A Boolean cap can produce either the integer or Boolean count spelling.
    match = re.fullmatch(
        r"lbfgs failed to converge after ([0-9.+A-Za-z-]{1,32}) iteration\(s\) \(status=1\):\n"
        r"STOP: TOTAL NO\. OF ITERATIONS REACHED LIMIT\n\n"
        r"Increase the number of iterations to improve the convergence \(max_iter=([0-9.+A-Za-z-]{1,32})\)\.\n"
        r"You might also want to scale the data as shown in:\n"
        r"    https://scikit-learn\.org/stable/modules/preprocessing\.html\n"
        r"Please also refer to the documentation for alternative solver options:\n"
        r"    https://scikit-learn\.org/stable/modules/linear_model\.html#logistic-regression",
        message,
    )
    if match is None:
        return None
    count, cap = match.groups()
    if cap in ("False", "True"):
        limit = int(cap == "True")
        return limit if count in (cap, str(limit)) else None
    if re.fullmatch(r"0|-?[1-9][0-9]{0,18}", cap):
        return int(cap) if count == cap else None
    try:
        limit = float(cap)
    except ValueError:
        return None
    if not math.isfinite(limit) or repr(limit) != cap:
        return None
    # If result.nit equals a floating max_iter, min can return that integer.
    if count == cap or limit >= 0 and limit.is_integer() and count == str(int(limit)):
        return limit
    return None


def _legacy_lbfgs_limit_warning(message, cap, count, sklearn_version):
    if sklearn_version in _AUDITED_SOLVER_VERSIONS:
        limit = _lbfgs_warning_limit(message)
        return limit is not None and _matches_iteration_limit(cap, limit)
    return (_matches_iteration_limit(cap, count)
            and message.startswith(f"lbfgs failed to converge after {count} iteration(s) (status=1):\n"
                                   "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"))


def _nmf_solver_warning(message, solver, head, iterations, settings):
    # Unqualified legacy warning names alone cannot establish their origin.
    # These stock solver messages carry independent iteration/configuration evidence.
    if solver in ("cd", "mu"):
        match = re.fullmatch(r"Maximum number of iterations ([1-9][0-9]{0,18}) reached"
                             r"( for [1-9][0-9]* rows during rowwise MU projection)?\. "
                             r"Increase it to improve convergence\.", message)
        if match and (not match[2] or solver == "mu"):
            return _matches_iteration_limit(settings["max_iter"], int(match[1]))
    if head == "linear_svm" and message == "Liblinear failed to converge, increase the number of iterations.":
        return _matches_iteration_limit(settings["svm_max_iter"], iterations)
    return False


def _positive_nmf_tolerance(node):
    # A completed stock fit already required a finite, nonnegative Real.
    # Exclude both signs of zero for reversible scalar storage. Long-double
    # padding can vary by platform and does not establish a positive value.
    kind = node["kind"]
    if kind == "numpy_scalar":
        dtype = node["dtype"]
        code = dtype["str"][1:]
        if dtype.get("format", [None])[0] == "g":
            return None
        if code in ("f2", "f4", "f8"):
            kind = "float"
        elif re.fullmatch(r"[iu][1248]", code):
            kind = "int"
    zeros = {"float": (0.0, -0.0), "int": (0,), "bool": (False,)}.get(kind)
    if zeros is None:
        return None
    for zero in zeros:
        try:
            _check_display(node, zero)
        except ValueError:
            continue
        return False
    return True


def _check_nmf_iterations(settings, solver, head, diagnostics, plain, sklearn_version):
    n_iter = diagnostics["n_iter"]
    if solver == "spa":
        # FeatureSPA assigns n_iter_ = n_components_ after selecting the anchors.
        if n_iter < 1 or _matches_iteration_limit(settings["n_components"], n_iter) is False:
            raise ValueError("NMF SPA iteration count disagrees with its component count.")
    if sklearn_version not in _AUDITED_SOLVER_VERSIONS:
        return
    if solver in ("cd", "mu"):
        if n_iter < 1:
            raise ValueError("A completed NMF factorization requires at least one iteration.")
        at_cap = _at_iteration_cap(settings["max_iter"], n_iter)
        positive_tol = _positive_nmf_tolerance(settings["tol"])
        if solver == "mu" and at_cap is False and (positive_tol is False or n_iter % 10):
            # MU checks the stopping criterion every ten iterations at positive
            # tolerance. At zero tolerance it runs to the cap.
            raise ValueError("NMF MU cannot stop at the recorded iteration.")
        if plain and at_cap and positive_tol and diagnostics["factorization_converged"]:
            raise ValueError("NMF factorization at its iteration limit lacks its convergence warning.")
    if head == "logistic":
        # sklearn stores min(result.nit, max_iter). The optimizer's status
        # independently determines whether L-BFGS emits a convergence warning.
        _at_iteration_cap(settings["logistic_max_iter"], diagnostics["classifier_n_iter"][0])
    elif head == "linear_svm":
        # The stock primal squared-hinge solver reports 0 through max_iter.
        at_cap = _at_iteration_cap(settings["svm_max_iter"], diagnostics["classifier_n_iter"])
        if plain and at_cap and diagnostics["classifier_converged"]:
            raise ValueError("NMF linear SVM at its iteration limit lacks its convergence warning.")


def _check_nmf_query(fold, diagnostics, scored, plain):
    performed = diagnostics["last_transform_performed"]
    flag = diagnostics["last_transform_converged"]
    messages = diagnostics["last_transform_warnings"]
    error = diagnostics["last_transform_error"]
    seconds = diagnostics["last_transform_seconds"]
    if (type(performed) is not bool or type(messages) is not list or any(type(m) is not str for m in messages)
            or not _number(seconds) or seconds < 0):
        raise ValueError("Invalid NMF query projection diagnostics.")
    if not performed:
        if scored or flag is not None or messages or error is not None or seconds != 0:
            raise ValueError("NMF fold reports results for an unperformed query projection.")
    elif error is not None:
        if (scored or type(error) is not str or flag is not False
                or plain and error != f"{fold['error_type']}: {fold['error']}"):
            raise ValueError("NMF query projection error disagrees with the failed fold.")
    elif flag is not (not messages):
        raise ValueError("NMF query convergence disagrees with its warnings.")
    return performed


def _check_nmf_convergence(fold, version, settings, plain, sklearn_version, decision, checked_warnings):
    # On an exact NMFClassifier these are stored fit attributes and the
    # projection record from this prediction. Custom estimator properties
    # can return different values when diagnostics reads them after scoring.
    scored = fold["status"] != "failed" or "score" in fold
    if not scored and "diagnostics" not in fold:
        # clone/set_params can fail before the evaluator owns a fitted object.
        if fold["fit_seconds"] != 0 or fold["prediction_seconds"] != 0:
            raise ValueError("Timed NMF fit or prediction is missing its diagnostics.")
        return
    diagnostics = fold["diagnostics"]
    queried = _check_nmf_query(fold, diagnostics, scored, plain)
    if not scored and not queried and "fit_times" not in diagnostics:
        # Both release writers clear fitted attributes if fit raises. In this
        # case _diagnostics reports only the empty call-local projection record.
        if any(not key.startswith("last_transform_") for key in diagnostics) or fold["prediction_seconds"] != 0:
            raise ValueError("Failed NMF fit retains contradictory fitted or prediction evidence.")
        return
    performed = diagnostics["training_projection_performed"]
    if (type(performed) is not bool or diagnostics["training_coordinates_mode"] not in ("transform", "fit_transform")
            or performed != (diagnostics["training_coordinates_mode"] == "transform")):
        raise ValueError("NMF fit has inconsistent training projection diagnostics.")
    fitting_warnings = []
    for phase in ("factorization", "training_projection", "classifier"):
        messages = diagnostics[phase + "_convergence_warnings"]
        if type(messages) is not list or any(type(message) is not str for message in messages):
            raise ValueError("Invalid NMF convergence warnings.")
        skipped = phase == "training_projection" and not performed
        expected = None if skipped else not messages
        if (skipped and messages) or diagnostics[phase + "_converged"] is not expected:
            raise ValueError("NMF phase convergence disagrees with its warnings.")
        fitting_warnings.extend(messages)
    if (diagnostics["convergence_warnings"] != fitting_warnings
            or diagnostics["converged"] is not (not fitting_warnings)):
        raise ValueError("NMF fit convergence disagrees with its phase diagnostics.")
    n_iter = diagnostics["n_iter"]
    decomposition_iter = diagnostics["decomposition"]["n_iter"]
    if type(n_iter) is not int or type(decomposition_iter) is not int or n_iter < 0 or decomposition_iter != n_iter:
        raise ValueError("NMF decomposition and fit iteration counts disagree.")

    head = _NMF_HEADS.get(diagnostics["fitted_classifier_type"])
    iterations = diagnostics["classifier_n_iter"]
    normalization = diagnostics["knn_normalization"]
    if head is None or (normalization not in ("sd", "none") if head == "knn" else normalization is not None):
        raise ValueError("NMF fitted head and KNN normalization disagree.")
    if head == "knn":
        # This branch assigns () and None unconditionally after fitting KNN.
        if diagnostics["classifier_convergence_warnings"] or iterations is not None:
            raise ValueError("NMF KNN head has impossible classifier convergence diagnostics.")
    elif head == "logistic":
        if type(iterations) is not list or len(iterations) != 1 or type(iterations[0]) is not int or iterations[0] < 0:
            raise ValueError("Invalid NMF logistic classifier iteration count.")
    elif head == "linear_svm":
        if type(iterations) is not int or iterations < 0:
            raise ValueError("Invalid NMF linear SVM classifier iteration count.")

    if settings is not None:
        declared_head = _text_choice(settings["classifier"], _NMF_HEADS.values())
        if declared_head is not None and declared_head != head:
            raise ValueError("Fitted NMF classifier disagrees with its declared head.")
        if head == "knn":
            declared_normalization = _text_choice(settings["knn_normalization"], ("sd", "none"))
            if declared_normalization is not None and normalization != declared_normalization:
                raise ValueError("NMF KNN normalization disagrees with its declaration.")
        solver = _text_choice(settings["solver"], ("spa", "cd", "mu"))
        mode = _text_choice(settings["training_coordinates"], ("auto", "transform", "fit_transform"))
        if solver == "spa" or solver in ("cd", "mu") and mode is not None:
            uses_projection = solver == "spa" or mode == "transform"
            if performed != uses_projection:
                raise ValueError("NMF training projection disagrees with its declared solver and coordinate mode.")
        _check_nmf_iterations(settings, solver, head, diagnostics, plain, sklearn_version)

    if plain:
        # _record_warnings forwards every captured warning after each phase.
        # Other fold operations can emit additional warnings. Consume matches
        # in order so that one outer warning cannot witness two occurrences.
        # Legacy warnings lack type ancestry and can only corroborate the text.
        forwarded = (w["message"] for w in checked_warnings if version is None or w["is_convergence_warning"])
        phases = fitting_warnings + diagnostics["last_transform_warnings"]
        for message in phases:
            if not any(text == message for text in forwarded):
                raise ValueError("NMF convergence diagnostics lack a corresponding forwarded phase warning.")
        # Conversely, a recognizable stock solver warning cannot disappear
        # from every phase. Warnings from screening, scoring or custom code
        # are not in general part of the fit or query diagnostics.
        extra = Counter(w["message"] for w in checked_warnings if w["category"] == "ConvergenceWarning"
                        and (version is None or w["is_convergence_warning"]))
        extra.subtract(phases)
        if any(count > 0 and _nmf_solver_warning(text, solver, head, iterations, settings)
               for text, count in extra.items()):
            raise ValueError("NMF convergence diagnostics omit a recorded stock solver warning.")
        if (head == "logistic" and version is None and decision == "valid"
                and any(count > 0 and _legacy_lbfgs_limit_warning(text, settings["logistic_max_iter"],
                                                                  iterations[0], sklearn_version)
                        for text, count in extra.items())):
            raise ValueError("Valid NMF fold contradicts its recorded L-BFGS convergence warning.")

    failed = not diagnostics["converged"] or not diagnostics["last_transform_converged"]
    if (decision == "valid" and failed
            or decision == "nonconverged" and not failed
            and (not fold["warnings"] or version is not None and fold["convergence_check"]["reason"] == "estimator")):
        raise ValueError("Fold status disagrees with the NMF convergence diagnostics.")


def _check_lbfgs_fit(fold, settings):
    if fold["status"] == "failed" and "score" not in fold:
        return  # A failed fit can leave partial or empty attributes.
    diagnostics = fold["diagnostics"]
    iterations = diagnostics.get("n_iter")
    stock_count = (diagnostics.get("fitted_classifier_type") == _LOGISTIC and type(iterations) is list
                   and len(iterations) == 1 and type(iterations[0]) is int)
    if not stock_count:
        raise ValueError("Invalid fitted stock L-BFGS iteration diagnostics.")
    count = iterations[0]
    # LogisticRegression.fit stores n_iter_ in a signed int32 array.
    if not -(2**31) <= count < 2**31:
        raise ValueError("L-BFGS iteration count exceeds the fitted int32 storage range.")
    cap = settings["max_iter"]
    at_cap = _at_iteration_cap(cap, count, "L-BFGS")
    # Without parameter validation sklearn can store a negative cap itself.
    if count < 0 and (at_cap is False or cap["kind"] == "none"):
        raise ValueError("Invalid fitted stock L-BFGS iteration count.")


_OUTER_FLAGS = ("converged", "classifier_converged", "last_transform_converged")
_FIT_CHECKS = ("converged", "classifier_converged", "final_classifier_converged")


def _check_observed_flags(fold, estimator_type, tracked_head):
    """Verify the live predicate results, including the writer's short circuit."""
    check = fold["convergence_check"]
    flags = check.get("flag_checks")
    if type(flags) is not dict or any(
            type(item) is not dict or set(item) != {"passed", "exact_scalar"}
            or any(type(value) is not bool for value in item.values()) for item in flags.values()):
        raise ValueError("Invalid recorded convergence flag checks.")
    expected = {"last_transform_converged"}
    reason = check["reason"]
    if reason != "warning":
        for name in _FIT_CHECKS:
            expected.add(name)
            if name not in flags or not flags[name]["passed"]:
                break
    if set(flags) != expected or (reason != "warning" and
            (reason == "estimator") != any(not item["passed"] for item in flags.values())):
        raise ValueError("Recorded convergence flag checks disagree with the decision or its short circuit.")

    diagnostics = fold["diagnostics"]
    if estimator_type in (_LOGISTIC, _NMF_CLASSIFIER) or estimator_type in _TRACKED_HEADS:
        # These direct estimators have no final Pipeline child. Independent
        # stored native flags can also corroborate the recorded truth tests.
        final = flags.get("final_classifier_converged")
        if final is not None and not (final["passed"] and final["exact_scalar"]):
            raise ValueError("Direct estimator has an inconsistent final classifier convergence check.")
        if all(item["exact_scalar"] for item in flags.values()):
            for name in _OUTER_FLAGS:
                if name not in flags:
                    continue
                value = diagnostics.get(name, True)
                # A nonfinite builtin float serializes as null and is truthy.
                if ((value is not None and type(value) not in (bool, int, float))
                        or flags[name]["passed"] is not (value is None or bool(value))):
                    raise ValueError("Native convergence flag disagrees with its recorded truth test.")
    if tracked_head is not None:
        head, prefix = tracked_head
        name = "final_classifier_converged" if prefix else "converged"
        if (name in flags and diagnostics.get("fitted_classifier_type") == head
                and (not flags[name]["exact_scalar"] or flags[name]["passed"] is not diagnostics[prefix + "converged"])):
            raise ValueError("Recorded predicate disagrees with the tracked classifier convergence evidence.")
    if estimator_type == _LOGISTIC and fold["status"] != "failed" and check["warnings_checked"] != len(fold["warnings"]):
        # Only the fit predicates run after warning classification. An opaque
        # flag's truth method can itself emit warnings on the stock estimator.
        if all(flags[name]["exact_scalar"] for name in _FIT_CHECKS if name in flags):
            raise ValueError("Stock fold has an inconsistent convergence warning boundary.")


def _check_fold_convergence(fold, version, settings, estimator_type=None, plain=False, sklearn_version=None):
    status, recorded = fold["status"], fold["warnings"]
    decision = status
    checked_warnings = recorded
    if version is not None and any(type(w["is_convergence_warning"]) is not bool for w in recorded):
        raise ValueError("Invalid archived convergence warning classification.")
    if version is not None and (status != "failed" or "convergence_check" in fold):
        check = fold["convergence_check"]
        checked, reason = check["warnings_checked"], check["reason"]
        decision = "valid" if reason is None else "nonconverged"
        # A catchable exception can arrive between _nonconverged storing its
        # decision and the evaluator assigning the final status. Scoring
        # already finished and the writer collects the fitted diagnostics.
        if status == "failed" and ("score" not in fold or type(fold.get("diagnostics")) is not dict):
            raise ValueError("Failed fold's completed convergence check lacks scoring or fitted evidence.")
        if (type(checked) is not int or not 0 <= checked <= len(recorded)
                or reason not in (None, "warning", "estimator")
                or (status != "failed" and decision != status)
                or (reason == "warning") != any(w["is_convergence_warning"] for w in recorded[:checked])):
            raise ValueError("Fold status disagrees with its recorded convergence check.")
        plain_boundary = (estimator_type == _NMF_CLASSIFIER or
                          estimator_type == _LOGISTIC and version != 6 and
                          (reason == "warning" or not any(name in fold["diagnostics"] for name in _FIT_CHECKS[:2])))
        if status != "failed" and plain_boundary and checked != len(recorded):
            raise ValueError("Stock fold has an inconsistent convergence warning boundary.")
        checked_warnings = recorded[:checked]
    if estimator_type == _NMF_CLASSIFIER:
        _check_nmf_convergence(fold, version, settings, plain, sklearn_version, decision, checked_warnings)
        return
    if estimator_type == _LOGISTIC and status != "failed":
        # Stock logistic creates none of these fields. A present callback
        # flag can serialize as 0 or false even when its live truth was true.
        flags = fold["convergence_check"].get("flag_checks", {}) if version == 6 else {}
        if (decision == "nonconverged" and not any(name in fold["diagnostics"] for name in _OUTER_FLAGS)
                and all(item.get("exact_scalar") is True for item in flags.values())
                and (not recorded or version is not None and fold["convergence_check"]["reason"] == "estimator")):
            raise ValueError("Stock logistic fold disagrees with its convergence warnings or estimator flags.")
    # sklearn clips the outer solver's iteration count to its own cap. The
    # representation of other settings (for example class weights) is immaterial.
    if (estimator_type == _LOGISTIC and settings is not None and sklearn_version in _AUDITED_SOLVER_VERSIONS
            and _text_choice(settings["solver"], ("lbfgs",)) == "lbfgs"):
        _check_lbfgs_fit(fold, settings)


def _target_cache_error(message, method):
    if method in _REQUEST_CACHE_METHODS:
        message += (" The historical report observed the target cache before fitting. "
                    "Its state at scorer entry cannot be independently verified.")
    raise ValueError(message)


def _check_standard_label_domains(fold, target, prediction):
    pred_kind = fold["balanced_accuracy_prediction_kind"]
    if pred_kind == "c":
        raise ValueError("Balanced accuracy cannot score the recorded complex predictions.")
    if (fold["balanced_accuracy_custom_scalar_inputs"] != "none" or target["opaque_unique_cache"]
            or prediction.get("opaque_unique_cache")):
        return
    true_kind = target["dtype_kind"]
    true_text = true_kind == "U" or fold.get("balanced_accuracy_target_plain_strings") is True
    pred_text = pred_kind == "U" or prediction.get("plain_string_values") is True
    # sklearn sorts the union of the native cache keys or the actual labels.
    # It rejects mixed str/non-str keys; bytes and numeric/temporal keys
    # cannot be ordered together. A union drawn from just one domain cannot
    # retain both inputs in any confusion cell. Native byte vectors already
    # fail type_of_target; the union argument also covers byte columns.
    if pred_kind == "S" and (true_text or true_kind in "biufmM"):
        raise ValueError("Balanced accuracy cannot score native byte predictions with the recorded standard "
                         "string, numeric or temporal target and cache states.")
    if true_text and pred_kind in "biufmM" or pred_text and true_kind in "biufmM":
        raise ValueError("Balanced accuracy cannot score mixed string and numeric or temporal labels "
                         "with the recorded cache states.")


_AUDITED_TARGET_VERSIONS = ("1.9.0", "1.9.1")
_FLOAT_OUTSIDE = "outside-int64-roundtrip-v1"
_FLOAT_BOUNDARY = "positive-int64-boundary"
_FLOAT_BOUNDARY_PASSED = "positive-int64-boundary-passed-v1"
_FLOAT_BOUNDARY_FAILED = "positive-int64-boundary-failed-v1"
_FLOAT_NULL = "nonfinite-class-display"
_FLOAT_SCORE_ERROR = ("The original floating class labels cannot be classified by the recorded "
                      "scikit-learn scorer. A score on changed target values cannot establish "
                      "performance on the original labels.")
_MISSING_FLOAT_BOUNDARY = ("The historical report omitted the int64 round-trip observation at +2**63. "
                           "A fold score for the original floating class labels cannot be verified.")
_MISSING_FLOAT_CLASS = ("The historical report displayed a nonfinite floating class as null. Its original "
                        "value and float format cannot be established well enough to verify the fold score.")


def _float_label_domain(values):
    """Necessary conditions for sklearn's float -> int64 -> original dtype check."""
    boundary = False
    wide_significand = False
    for value in values:
        numerator, denominator = value.as_integer_ratio()
        if denominator != 1 or not -(1 << 63) <= numerator <= 1 << 63:
            return _FLOAT_OUTSIDE
        # On a platform with saturating casts, INT64_MAX can round back up
        # to this endpoint in the original float dtype. The negative endpoint
        # is itself representable as an int64.
        boundary |= numerator == 1 << 63
        magnitude = abs(numerator)
        if magnitude:
            wide_significand |= (magnitude // (magnitude & -magnitude)).bit_length() >= 63
    if boundary and wide_significand:
        # A class needing 63 significant bits proves the common dtype can
        # distinguish INT64_MAX from INT64_MAX + 1.
        return _FLOAT_OUTSIDE
    return _FLOAT_BOUNDARY if boundary else None


def _check_original_float_scores(report, error=_FLOAT_SCORE_ERROR):
    if any("score" in fold or report["scoring"] in fold
           for candidate in report["candidates"] for fold in candidate["folds"]):
        raise ValueError(error)


def _check_scorer_dtypes(fold, identity, target):
    if (identity.get("fold_evidence_version", 0) < 4
            or identity.get("software", {}).get("scikit-learn") not in _AUDITED_TARGET_VERSIONS):
        return
    # Both scorers validate the flattened target. Complex arrays fail
    # check_array, and type_of_target rejects a bytes scalar in its first row.
    if target["dtype_kind"] == "c" or target["dtype_kind"] == "S" and target.get("native_text") is True:
        raise ValueError("Fold score contradicts the target dtype rejected by the recorded scikit-learn scorer.")
    witness = fold.get("balanced_accuracy_confusion_rows")
    if "balanced_accuracy_prediction_kind" in fold:
        # Without rows, an opaque cache would itself require missing evidence.
        _check_standard_label_domains(fold, target, {} if witness is None else witness["prediction"])
        return
    if witness is None:
        return
    prediction = witness["prediction"]
    # Prediction shape is absent. A one-column byte array bypasses the
    # scalar check, but native Unicode and bytes cannot both match keys in
    # sklearn's homogeneous native label union. No confusion cell survives.
    # An opaque key can compare equal to both, so keep that case open.
    conflicting_bytes = (prediction["dtype_kind"] == "S" and prediction.get("native_text") is True
                         and (target["dtype_kind"] == "U" and target.get("native_text") is True
                              or target.get("plain_string_values") is True)
                         and target.get("opaque_unique_cache") is False
                         and prediction.get("opaque_unique_cache") is False)
    if prediction["dtype_kind"] != "c" and not conflicting_bytes:
        return
    if identity["balanced_accuracy_support_method"] == "confusion-matrix-v1":
        # A cache callback or warning hook could have changed the array's
        # dtype after sklearn validated it. Even an attainable score does
        # not establish that the original input was scorable.
        raise ValueError("The historical report recorded the prediction dtype after scoring. "
                         "The file cannot establish which dtype the balanced accuracy scorer received.")
    if conflicting_bytes:
        _target_cache_error("Balanced accuracy cannot score the recorded native Unicode target and byte predictions "
                            "with the recorded cache.", identity["balanced_accuracy_support_method"])
    raise ValueError("Balanced accuracy cannot score the recorded complex predictions.")


def _fold_target_evidence(fold, identity):
    target = identity.get("class_label_evidence", {})
    method = identity.get("balanced_accuracy_support_method")
    observes_cache = method in _FOLD_CACHE_METHODS and "score" in fold
    if ("balanced_accuracy_target_cache" in fold) != observes_cache:
        raise ValueError("Missing or unexpected balanced accuracy target cache observation.")
    if observes_cache:
        cache = fold["balanced_accuracy_target_cache"]
        if cache not in ("none", "native", "opaque"):
            raise ValueError("Invalid balanced accuracy target cache observation.")
        target = dict(target, has_unique_cache=cache != "none", opaque_unique_cache=cache == "opaque")
    observes_inputs = method == _BALANCED_SUPPORT_METHOD and "score" in fold
    if ("balanced_accuracy_prediction_kind" in fold) != observes_inputs:
        raise ValueError("Missing or unexpected balanced accuracy prediction kind.")
    if ("balanced_accuracy_custom_scalar_inputs" in fold) != observes_inputs:
        raise ValueError("Missing or unexpected balanced accuracy custom scalar observation.")
    observes_objects = observes_inputs and target["dtype_kind"] == "O"
    if ("balanced_accuracy_target_plain_strings" in fold) != observes_objects:
        raise ValueError("Missing or unexpected balanced accuracy target object-string observation.")
    if observes_inputs:
        kind = fold["balanced_accuracy_prediction_kind"]
        custom = fold["balanced_accuracy_custom_scalar_inputs"]
        if kind not in tuple("biufcmMOSUV"):
            raise ValueError("Invalid balanced accuracy prediction kind.")
        if custom not in ("none", "target", "prediction", "both"):
            raise ValueError("Invalid balanced accuracy custom scalar observation.")
        custom_target = custom in ("target", "both")
        if (target["dtype_kind"] in "US" and target["native_text"] == custom_target
                or target["dtype_kind"] == "O" and custom_target
                or kind == "O" and custom in ("prediction", "both")):
            raise ValueError("Balanced accuracy custom scalar observation contradicts the label dtypes.")
    if observes_objects:
        plain_strings = fold["balanced_accuracy_target_plain_strings"]
        if type(plain_strings) is not bool:
            raise ValueError("Invalid balanced accuracy target object-string observation.")
        target["plain_string_values"] = target["plain_string_values"] and plain_strings
    return target


def _validate_fold(fold, scoring, other_scoring, validation_size, identity, settings, plain):
    status = fold["status"]
    if status not in ("valid", "nonconverged", "failed"):
        raise ValueError("Invalid calibration fold status.")
    if other_scoring in fold:
        raise ValueError("Fold contains a score for a different objective.")
    target = _fold_target_evidence(fold, identity)
    if "balanced_accuracy_support" in fold and (scoring != "balanced_accuracy" or "score" not in fold
                                                or identity.get("fold_evidence_version", 0) < 3):
        raise ValueError("Unexpected balanced accuracy scoring evidence.")
    if "balanced_accuracy_hits" in fold and (scoring != "balanced_accuracy" or "score" not in fold
                                             or identity.get("fold_evidence_version") != 6):
        raise ValueError("Unexpected balanced accuracy hit counts.")
    if "balanced_accuracy_confusion_rows" in fold and (
        scoring != "balanced_accuracy" or "score" not in fold or identity.get("fold_evidence_version") != 6
        or identity.get("balanced_accuracy_support_method") not in _BALANCED_SUPPORT_METHODS
        or "balanced_accuracy_hits" in fold or "balanced_accuracy_support" in fold
    ):
        raise ValueError("Unexpected balanced accuracy confusion rows.")
    for field in ("fit_seconds", "prediction_seconds"):
        if not _number(fold[field]) or fold[field] < 0:
            raise ValueError("Invalid calibration fold duration.")
    if type(fold["warnings"]) is not list or any(
        type(w) is not dict or type(w["message"]) is not str or type(w["category"]) is not str
        for w in fold["warnings"]
    ):
        raise ValueError("Invalid calibration fold warnings.")
    _check_fold_convergence(fold, identity.get("fold_evidence_version"), settings, identity["estimator"], plain,
                            identity.get("software", {}).get("scikit-learn"))
    if status == "failed":
        if type(fold["error_type"]) is not str or type(fold["error"]) is not str:
            raise ValueError("Invalid calibration fold error.")
        # A failure in the convergence check can occur after scoring succeeds.
        if "score" not in fold and scoring not in fold:
            return
    elif "error" in fold or "error_type" in fold:
        raise ValueError("Successful scoring has a contradictory fold error.")
    score, alias = fold["score"], fold[scoring]
    if not _number(score) or not 0 <= score <= 1 or not _number(alias) or alias != score:
        raise ValueError("Invalid calibration fold score or metric alias.")
    _check_scorer_dtypes(fold, identity, target)
    if scoring == "accuracy" and score != round(score * validation_size) / validation_size:
        raise ValueError("Fold accuracy is inconsistent with the validation sample count.")
    return target


def _check_balanced_hits(fold, supports):
    hits = fold["balanced_accuracy_hits"]
    if (type(hits) is not list or len(hits) != 2 or
            any(type(h) is not int or not 0 <= h <= n for h, n in zip(hits, supports))):
        raise ValueError("Invalid balanced accuracy hit counts.")
    recalls = [h / n for h, n in zip(hits, supports) if n]
    if sum(recalls) / len(recalls) != fold["score"]:
        raise ValueError("Balanced accuracy disagrees with its recorded hit counts.")


def _plain_tree_setting(node):
    """Include ordinary class weights and monotonic constraint vectors."""
    if node["kind"] == "dict":
        return all(_plain_tree_setting(item) for pair in node["items"] for item in pair)
    if node["kind"] in ("list", "tuple"):
        return all(_plain_tree_setting(item) for item in node["items"])
    if node["kind"] == "ndarray":
        if node["dtype"] == {"str": "|O", "format": ["O", "|", 1, np.dtype(object).flags]}:
            contents = node["content"]
            return (type(contents) is list and len(contents) == math.prod(node["shape"])
                    and all(_plain_tree_setting(item) for item in contents))
        return node["dtype"]["str"][1:2] in {"b", "i", "u", "f"}
    return _plain_setting(node)


def _retained_support_has_integer_target(report):
    identity = report["identity"]
    if identity["fold_evidence_version"] >= 4:
        kind = identity["class_label_evidence"]["dtype_kind"]
        if kind not in "biuf":
            raise ValueError("Balanced accuracy scoring evidence requires numeric validation labels.")
        return kind in "biu"
    # Version 3 only emitted this witness for native numeric truth. Its
    # class-label display therefore came from a single numeric dtype, even
    # though the version omitted the dtype field. Nonfinite floats became null.
    labels = report["class_labels"]
    if type(labels[0]) is type(labels[1]) and type(labels[0]) in (bool, int) and labels[0] < labels[1]:
        return True
    if _finite_float_class_values(labels) is not None:
        return False
    raise ValueError("Balanced accuracy scoring evidence disagrees with the numeric class labels.")


def _ordinary_score_tree(identity, estimator, declaration):
    return (identity["estimator"] in ("sklearn.tree._classes.DecisionTreeClassifier",
                                      "sklearn.tree._classes.ExtraTreeClassifier")
            and identity.get("software", {}).get("scikit-learn") in ("1.9.0", "1.9.1")
            and estimator["kind"] == "estimator" and _plain_tree_setting(estimator["parameters"])
            and _plain_tree_setting(declaration))


def _check_confusion_rows(fold, full, warning_end, target, stock_tree, method=_BALANCED_SUPPORT_METHOD):
    witness = fold["balanced_accuracy_confusion_rows"]
    if type(witness) is not dict or witness.keys() != {"rows", "warning_index", "prediction", "matrix_size"}:
        raise ValueError("Invalid balanced accuracy confusion rows.")
    pred = witness["prediction"]
    if type(pred) is not dict or pred.get("dtype_kind") not in tuple("biufcmMOSUV"):
        raise ValueError("Invalid balanced accuracy confusion row prediction dtype.")
    fields = {"dtype_kind", "opaque_unique_cache", "has_unique_cache"}
    if pred["dtype_kind"] in "US":
        fields.add("native_text")
        if type(pred.get("native_text")) is not bool:
            raise ValueError("Invalid balanced accuracy confusion row text prediction dtype.")
    elif pred["dtype_kind"] == "O" and method not in ("confusion-matrix-v1", "confusion-matrix-v2"):
        fields.add("plain_string_values")
        if type(pred.get("plain_string_values")) is not bool:
            raise ValueError("Invalid balanced accuracy confusion row object-string observation.")
    if (pred.keys() != fields
            or type(pred["opaque_unique_cache"]) is not bool or type(pred["has_unique_cache"]) is not bool
            or pred["opaque_unique_cache"] and not pred["has_unique_cache"]):
        raise ValueError("Invalid balanced accuracy confusion row prediction dtype.")
    if not {"opaque_unique_cache", "has_unique_cache"} <= target.keys():
        raise ValueError("Missing balanced accuracy confusion row target dtype.")
    nonnumeric = target["dtype_kind"] not in "biuf" or pred["dtype_kind"] not in "biuf"
    target_cache = target["opaque_unique_cache"] or nonnumeric and target["has_unique_cache"]
    pred_cache = pred["opaque_unique_cache"] or nonnumeric and pred["has_unique_cache"]
    scalar_rows = False
    if method == _BALANCED_SUPPORT_METHOD:
        if "balanced_accuracy_prediction_kind" in fold and fold["balanced_accuracy_prediction_kind"] != pred["dtype_kind"]:
            raise ValueError("Balanced accuracy prediction kind disagrees with its confusion row evidence.")
        custom = fold.get("balanced_accuracy_custom_scalar_inputs", "none")
        if pred["dtype_kind"] in "US" and pred["native_text"] == (custom in ("prediction", "both")):
            raise ValueError("Balanced accuracy confusion row text dtype contradicts its custom scalar observation.")
        scalar_rows = _scalar_rows_required(target["dtype_kind"], pred["dtype_kind"], custom)
    if (not (target_cache or pred_cache or scalar_rows)
            or stock_tree and (not (target_cache or scalar_rows) or pred["dtype_kind"] != target["dtype_kind"]
                               or pred.get("native_text") is not target.get("native_text"))):
        _target_cache_error("Balanced accuracy confusion rows contradict the declared cache or stock tree.", method)
    rows, index = witness["rows"], witness["warning_index"]
    size = sum(full)
    if (type(rows) is not list or not 0 < len(rows) <= size or
            any(type(row) is not list or len(row) != 2 or any(type(v) is not int for v in row)
                or row[0] < 1 or not 0 <= row[1] <= row[0] for row in rows)
            or sum(count for count, _ in rows) > size):
        raise ValueError("Invalid balanced accuracy confusion row counts.")
    string_target = target.get("native_text") is True or target.get("plain_string_values") is True
    if (stock_tree and target.get("plain_string_values") is True
            and pred.get("plain_string_values") is False):
        # The stock classifier indexes its fitted class array on prediction.
        raise ValueError("Object predictions contradict the plain string classes of the stock tree.")
    if (stock_tree and string_target and not target["opaque_unique_cache"]
            and (target.get("native_text") is True or pred.get("plain_string_values") is True)):
        # Uncached truth contributes both fitted classes to sklearn's label
        # union, so every stock prediction has a key even with a stale
        # prediction cache. A target cache can omit a class. If only one
        # survives, every retained prediction must match its truth.
        one_cached_class = (target["has_unique_cache"] and len(rows) == 1
                            and rows[0][1] == rows[0][0] <= max(full))
        if pred["opaque_unique_cache"] or ([n for n, _ in rows] != full and not one_cached_class):
            _target_cache_error("Balanced accuracy confusion rows contradict the native text classes of the stock tree.",
                                method)
    dimension = witness["matrix_size"]
    if type(dimension) is not int or dimension < len(rows) or dimension == len(rows) and index is not None:
        raise ValueError("Invalid balanced accuracy confusion row dimension or warning.")
    legacy_object = pred["dtype_kind"] == "O" and method in ("confusion-matrix-v1", "confusion-matrix-v2")
    if (string_target
            and (pred.get("native_text") is True or pred.get("plain_string_values") is True or legacy_object)
            and not target["opaque_unique_cache"] and not pred["opaque_unique_cache"]):
        # sklearn sorts the union of the label keys before lookup, even if a
        # native cache is stale or unordered. Empty rows disappear from the
        # witness. A single surviving class can therefore be either true class.
        observed = [n for n, _ in rows]
        if not target["has_unique_cache"] and not pred["has_unique_cache"]:
            # The label union includes every actual value. Each prediction-only
            # class needs a distinct misclassified sample to populate its column.
            errors = sum(count - hit for count, hit in rows)
            extra_classes = 0 if stock_tree else errors
            possible = observed == full and dimension <= len(full) + extra_classes
        elif len(observed) == 1:
            possible = observed[0] <= max(full) if pred["has_unique_cache"] else observed[0] in full
        elif len(observed) == 2:
            possible = (all(n <= limit for n, limit in zip(observed, full)) if pred["has_unique_cache"]
                        else observed == full)
        else:
            possible = False
        # Without a cached prediction list every predicted value is a matrix
        # key. Without a cached true list both true classes are matrix keys.
        if not possible or not target["has_unique_cache"] and dimension < 2:
            if legacy_object:
                raise ValueError("Historical confusion rows exceed what ordinary string predictions can establish. "
                                 "The archive omitted the object scalar types needed to verify these rows.")
            kind = "native text" if target.get("native_text") is True else "plain string"
            _target_cache_error(f"Balanced accuracy confusion rows contradict the recorded {kind} class counts.",
                                method)
    elif (method == _BALANCED_SUPPORT_METHOD and fold.get("balanced_accuracy_custom_scalar_inputs") == "none"
          and fold.get("balanced_accuracy_target_plain_strings") is True
          and (pred["dtype_kind"] == "U" and pred["native_text"] or pred.get("plain_string_values") is True)
          and not target["has_unique_cache"] and not pred["has_unique_cache"]):
        # Ordinary strings in this fold still form a complete label union
        # when caller-defined values elsewhere obscure the request's classes.
        # Each unpopulated row represents a distinct incorrect prediction.
        if sum(n for n, _ in rows) != size or dimension - len(rows) > sum(n - h for n, h in rows):
            raise ValueError("Uncached string confusion rows disagree with the validation size or "
                             "the number of incorrect predictions.")
    if dimension == 1 and rows[0][0] != rows[0][1]:
        raise ValueError("A one-class balanced accuracy confusion row must contain only correct predictions.")
    if index is not None and (
        type(index) is not int or not 0 <= index < warning_end or
        fold["warnings"][index] != {"category": "UserWarning", "message": _EXTRA_LABEL_WARNING,
                                    "is_convergence_warning": False}
    ):
        raise ValueError("Invalid balanced accuracy confusion row warning.")
    counts, hits = np.asarray(rows, dtype=np.int64).T
    if float(np.mean(hits / counts)) != fold["score"]:
        raise ValueError("Balanced accuracy disagrees with its confusion row counts.")


def _balanced_fold_scores(fold, report, estimator, declaration, scores, extra_label_scores, evidence):
    identity = report["identity"]
    score, index = fold["score"], fold["fold"]
    warnings = fold["warnings"]
    end = (fold["convergence_check"]["warnings_checked"] if identity.get("fold_evidence_version") is not None
           and "convergence_check" in fold else len(warnings))
    opaque = evidence.get("opaque_unique_cache")
    if "balanced_accuracy_confusion_rows" in fold:
        _check_confusion_rows(fold, identity["validation_class_counts"][index], end, evidence,
                              _ordinary_score_tree(identity, estimator, declaration),
                              identity["balanced_accuracy_support_method"])
        return
    target_needs_rows = opaque is True or evidence.get("has_unique_cache") and evidence["dtype_kind"] not in "biuf"
    if identity.get("balanced_accuracy_support_method") == _BALANCED_SUPPORT_METHOD:
        pred_kind = fold["balanced_accuracy_prediction_kind"]
        if (_scalar_rows_required(evidence["dtype_kind"], pred_kind, fold["balanced_accuracy_custom_scalar_inputs"])
                or evidence["has_unique_cache"] and pred_kind not in "biuf"):
            raise ValueError("Missing balanced accuracy confusion rows for the recorded scorer inputs.")
        if pred_kind != evidence["dtype_kind"] and _ordinary_score_tree(identity, estimator, declaration):
            raise ValueError("Balanced accuracy prediction kind contradicts the stock tree target dtype.")
    if target_needs_rows:
        _target_cache_error("Missing balanced accuracy confusion rows for the recorded target cache.",
                            identity.get("balanced_accuracy_support_method"))
    if "balanced_accuracy_support" in fold:
        # These stock trees predict from their fitted class array. Older
        # writers emitted reduced counts only after a floating promotion.
        # A current writer can also observe rows lost to a dtype label cache.
        integer_target = _retained_support_has_integer_target(report)
        measured = identity.get("balanced_accuracy_support_method") in _BALANCED_SUPPORT_METHODS
        software = identity.get("software", {})
        stock_tree = _ordinary_score_tree(identity, estimator, declaration)
        if stock_tree and (not measured and integer_target
                           or evidence.get("has_unique_cache", evidence.get("dtype_has_metadata")) is False):
            _target_cache_error("Balanced accuracy scoring evidence contradicts the stock tree's label dtype.",
                                identity.get("balanced_accuracy_support_method"))
        witness = fold["balanced_accuracy_support"]
        warning_index, retained = witness["warning_index"], witness["class_counts"]
        full = identity["validation_class_counts"][index]
        if warning_index is None:
            if not measured:
                raise ValueError("Balanced accuracy is missing its historical scorer warning.")
        elif (type(warning_index) is not int or not 0 <= warning_index < end
              or warnings[warning_index]["category"] != "UserWarning"
              or warnings[warning_index]["message"] != _EXTRA_LABEL_WARNING
              or warnings[warning_index]["is_convergence_warning"] is not False):
            raise ValueError("Balanced accuracy disagrees with its scorer warning.")
        if (type(retained) is not list or len(retained) != 2
                or any(type(k) is not int or not 0 <= k <= n for k, n in zip(retained, full))
                or not 0 < sum(retained) < sum(full)):
            raise ValueError("Balanced accuracy disagrees with its scoring evidence.")
        if stock_tree and opaque is False and (all(retained) or score != 1.):
            # With native cache keys, either both actual classes survive
            # lookup and every row remains, or only one class survives. Each
            # retained prediction in the latter case matches its truth.
            _target_cache_error("Balanced accuracy reduction contradicts the stock tree's observed label cache.",
                                identity.get("balanced_accuracy_support_method"))
        if "balanced_accuracy_hits" in fold:
            _check_balanced_hits(fold, retained)
        elif measured:
            raise ValueError("Balanced accuracy is missing the correct counts for the observed reduction.")
        else:
            _require_score_support(_possible_retained_score(score, *retained),
                                   "Balanced accuracy disagrees with its scoring evidence.")
        # These sklearn versions can discard a dtype cache on NumPy 2.2.6
        # before counting. Older writers consulted the raw cache afterwards.
        # A stateful cache can also defeat a stock floating tree's ordinary
        # invariant: if both class values survive lookup, every row survives.
        # Old hits came from the same lookup as the suspect denominators.
        if ("balanced_accuracy_support_method" not in identity
                and (software.get("numpy") == "2.2.6" and software.get("scikit-learn") in ("1.9.0", "1.9.1")
                     or stock_tree and all(retained))
                and _possible_balanced_score(score, *full) is not True):
            raise ValueError("Balanced accuracy depends on a historical reduced support that "
                             "cannot be independently verified.")
    elif "balanced_accuracy_hits" in fold:
        _check_balanced_hits(fold, identity["validation_class_counts"][index])
    elif identity.get("fold_evidence_version", 0) < 3 and any(
        w["category"] == "UserWarning" and w["message"] == _EXTRA_LABEL_WARNING
        and w.get("is_convergence_warning", False) is False for w in warnings[:end]
    ):
        extra_label_scores[index].add(score)
    else:
        scores[index].add(score)


def _check_fitted_logistic_nulls(display, settings):
    # Stock sklearn 1.9 validates these intervals before fitting. Of their
    # nonfinite float values only C=+inf is permitted. Earlier writers
    # rendered every nonfinite float as null without a separate finite flag.
    for name in ("C", "tol", "intercept_scaling", "l1_ratio"):
        if name not in display or display[name] is not None:
            continue
        node = settings[name]
        if name == "l1_ratio" and node["kind"] == "none":
            continue
        if name == "C":
            floating = node["kind"] == "float" or node["kind"] == "numpy_scalar" and node["dtype"]["str"][1:2] == "f"
            if floating and _matches_infinity(node, math.inf) is not False:
                continue
        raise ValueError(f"Null {name} contradicts the saved setting of a successfully fitted logistic regression.")


def _check_screening_evidence(report):
    diagnostics = report.get("screening_diagnostics")
    identity = report.get("identity")
    if (type(identity) is not dict or type(diagnostics) is not dict
            or identity.get("screening_diagnostics_sha256") != _json_digest(diagnostics)):
        raise ValueError("Screening diagnostics disagree with the calibration identity.")


def _recorded_inactive_modes(report, estimator, declarations):
    """Read the writer's observation, including losing or unscored candidates."""
    diagnostics = report["screening_diagnostics"]
    grouped = Counter()
    selected = {}
    if len(diagnostics["candidates"]) != len(report["candidates"]):
        raise ValueError("Screening rows disagree with the declared candidate order.")
    # Grouping used live Python equality across *all* candidates. A string
    # subclass can group with a differently spelled literal in another row.
    sklearn_version = report["identity"].get("software", {}).get("scikit-learn")
    plain_tree = _plain_screen_inspection(estimator, sklearn_version)
    plain_grouping = plain_tree and all(_plain_screen_inspection(item, sklearn_version) for item in declarations)
    for index, row in enumerate(diagnostics["candidates"]):
        if type(row["index"]) is not int or row["index"] != index:
            raise ValueError("Screening rows disagree with the declared candidate order.")
        plain_strings = plain_tree and _plain_screen_inspection(declarations[index], sklearn_version)
        for screen in row["screens"]:
            for name in screen["inactive_grid_parameters"]:
                # Diagnostics saw the raw grid key. Its later public str(key)
                # display can differ for a caller-defined string subclass.
                mode = screen["mode"]
                path, _, cutoff = name.rpartition("__")
                if plain_strings and (cutoff not in ("score_threshold", "percentile") or mode not in _ACTIVE_CUTOFF
                        or cutoff == _ACTIVE_CUTOFF[mode] or screen["path"] != (path or "(root)")):
                    raise ValueError("Inactive refinement disagrees with the observed screen path or cutoff.")
                grouped[(name, _json_digest(mode), index) if plain_grouping else index] += 1
                if index == report["selected_candidate_index"]:
                    selected[name] = mode
    recorded = Counter((group["parameter"], _json_digest(group["mode"]), index) if plain_grouping else index
                       for group in diagnostics["inactive_parameters"]
                       for index in group["candidate_indices"])
    if recorded != grouped:
        raise ValueError("Grouped screening diagnostics disagree with the observed candidate rows.")
    winner = report["selected_candidate_index"]
    if type(winner) is int and 0 <= winner < len(declarations):
        declaration = declarations[winner]
        if plain_tree and _plain_screen_inspection(declaration, sklearn_version):
            return _inactive_screening_modes(report, winner, estimator, declarations, bound_observation=True)
        if diagnostics["candidates"][winner]["screens"]:
            # A nonempty observation establishes that cloning, applying the
            # grid and inspecting the entire configured estimator succeeded.
            # An opaque sibling can report its own computed paths, but a
            # stock composition prefixes those paths with that sibling's name.
            display = report["candidates"][winner]["parameters"]
            cache = {}
            overrides = {_refinement_key(key, display, cache): value for key, value in declaration["items"]}
            if None not in overrides:
                for name in overrides:
                    path, _, cutoff = name.rpartition("__")
                    if cutoff not in ("percentile", "score_threshold"):
                        continue
                    component = _screen_component(estimator, path, overrides, sklearn_version)
                    if (component is None or component is NotImplemented or component["kind"] != "estimator"
                            or component["implementation"]["type"] not in (_BINARY_SCREEN, _NMF_CLASSIFIER)):
                        continue
                    modes = _screen_modes(estimator, path, overrides, sklearn_version)
                    if len(modes) == 1:
                        mode, = modes
                        expected = None if cutoff == _ACTIVE_CUTOFF[mode] else mode
                        if selected.get(name) != expected:
                            raise ValueError("Inactive refinement disagrees with the declared screen or screening mode.")
    return selected


_TRACKED_HEADS = {"supervised_nmf.evaluation._TrackedLogisticRegression", "supervised_nmf.evaluation._TrackedLinearSVC"}


def _tracked_head(estimator, declaration):
    """Identify a package head whose convergence flag is a stored fit attribute."""
    if estimator["kind"] != "estimator":
        return None
    root = estimator["implementation"]["type"]
    if root in _TRACKED_HEADS:
        return root, ""
    if root != "sklearn.pipeline.Pipeline":
        return None
    base, overrides = _string_items(estimator["parameters"]), _string_items(declaration)
    if base is None or overrides is None:
        return None
    steps = overrides.get("steps", base["steps"])
    if steps["kind"] not in ("list", "tuple") or not steps["items"]:
        return None
    # Only resolve plain step names. Opaque names can change which child a
    # replacement reaches. The stock Pipeline replaces named steps in lists.
    for pair in steps["items"]:
        if (pair["kind"] not in ("tuple", "list") or len(pair["items"]) != 2
                or pair["items"][0]["kind"] != "str"):
            return None
    key, head = steps["items"][-1]["items"]
    if key["value"] in overrides:
        if steps["kind"] != "list":
            return None
        head = overrides[key["value"]]
    if head["kind"] == "estimator" and head["implementation"]["type"] in _TRACKED_HEADS:
        return head["implementation"]["type"], "classifier_"
    return None


def _tracked_fit_disqualifies(fold, tracked_head, version):
    if tracked_head is None or fold["status"] == "failed":
        return False  # An exception can intervene before the fitted diagnostics are collected.
    head, prefix = tracked_head
    diagnostics = fold["diagnostics"]
    if diagnostics.get("fitted_classifier_type") != head:
        return False
    messages = diagnostics[prefix + "convergence_warnings"]
    converged = diagnostics[prefix + "converged"]
    if (type(messages) is not list or any(type(m) is not str for m in messages)
            or converged is not (not messages)):
        raise ValueError("Tracked classifier convergence disagrees with its fit warnings.")

    # Fit callbacks can add convergence flags on the outer estimator. For a
    # Pipeline, diagnostics overwrite its classifier_converged with the final
    # step's flag, so a clean head alone cannot rule out an estimator decision.
    flags = ("last_transform_converged", "converged" if prefix else "classifier_converged")
    failed = not converged
    # A callback can remove its flag while the evaluator tests its truth.
    # Version 6 records that result before the later diagnostic snapshot.
    all_clean = version != 6 and not prefix and converged and not any(name in diagnostics for name in flags)
    estimator_reason = (fold["convergence_check"]["reason"] == "estimator" if version is not None
                        else not fold["warnings"])
    if (failed and fold["status"] == "valid"
            or all_clean and fold["status"] == "nonconverged" and estimator_reason):
        raise ValueError("Fold status disagrees with the tracked classifier convergence evidence.")
    return failed


def _legacy_fold_disqualifies(fold, estimator_type, tracked_head):
    """Can this fold exclude a challenger independently of warning ancestry?"""
    if _tracked_fit_disqualifies(fold, tracked_head, None):
        return True
    if fold["status"] != "nonconverged":
        return fold["status"] == "failed"
    if not fold["warnings"]:
        return True
    if estimator_type == _NMF_CLASSIFIER:
        diagnostics = fold["diagnostics"]
        return diagnostics["converged"] is False or diagnostics["last_transform_converged"] is False
    return False


def _check_legacy_winner(candidates, winner, estimator, declarations):
    """Check whether missing warning ancestry could change the reported ranking."""
    estimator_type = estimator["implementation"]["type"] if estimator["kind"] == "estimator" else None
    for candidate, declaration in zip(candidates, declarations):
        folds = candidate["folds"]
        tracked_head = _tracked_head(estimator, declaration)
        disqualifications = [_legacy_fold_disqualifies(fold, estimator_type, tracked_head) for fold in folds]
        # A reported failure disqualifies the candidate regardless of a
        # warning's type. So does nonconvergence established independently
        # by a warning-free fold or the stock NMF fit/projection diagnostics.
        if any(disqualifications) or not any(fold["warnings"] for fold in folds):
            continue
        index = candidate["index"]
        if candidate is winner:
            outcome = f"eligibility of selected candidate {index}"
        else:
            # Even an ineligible candidate can retain every fold score. Use
            # the writer's mean and first-index tie rule for the counterfactual.
            if winner is not None:
                mean = np.mean([fold["score"] for fold in folds])
                if (-mean, index) >= (-winner["mean_score"], winner["index"]):
                    continue
            outcome = (f"no-winner result (candidate {index} could qualify)" if winner is None else
                       f"selection of candidate {winner['index']} (challenger {index} could win)")
        raise ValueError(f"Cannot verify the {outcome}: legacy folds omit the warning ancestry "
                         "and decision boundary used to determine convergence.")


def _validate_outcomes(report):
    identity = report["identity"]
    if "fold_evidence_version" in identity and (type(identity["fold_evidence_version"]) is not int
                                                or not 1 <= identity["fold_evidence_version"] <= 6):
        raise ValueError("Unsupported calibration fold evidence version.")
    recorded_modes = None
    if identity.get("fold_evidence_version", 0) >= 5:
        _check_screening_evidence(report)
    scoring = _validate_scoring(report["scoring"])
    if scoring != identity["scoring"]:
        raise ValueError("Calibration scoring disagrees with its identity.")
    if "balanced_accuracy_support_method" in identity:
        if (scoring != "balanced_accuracy" or identity["balanced_accuracy_support_method"] not in _BALANCED_SUPPORT_METHODS
                or identity.get("fold_evidence_version") != 6):
            raise ValueError("Unsupported balanced accuracy support method.")
        if not {"dtype_has_metadata", "has_unique_cache", "opaque_unique_cache"} <= identity["class_label_evidence"].keys():
            raise ValueError("Balanced accuracy is missing its target label-cache observations.")
    other = "balanced_accuracy" if scoring == "accuracy" else "accuracy"
    config = _string_items(identity["structural_configuration"])
    if config is None or config.keys() != {"estimator", "candidates"}:
        raise ValueError("Invalid structural calibration configuration.")
    estimator = config["estimator"]
    if (not isinstance(estimator["kind"], str) or not isinstance(identity["estimator"], str)
            or estimator["kind"] == "estimator" and estimator["implementation"]["type"] != identity["estimator"]):
        raise ValueError("Estimator disagrees with the calibration identity.")
    grid = config["candidates"]
    candidates = report["candidates"]
    if grid["kind"] != "list" or type(grid["items"]) is not list or len(grid["items"]) != len(candidates):
        raise ValueError("Candidate count disagrees with the declared grid.")
    if identity.get("fold_evidence_version", 0) >= 5:
        recorded_modes = _recorded_inactive_modes(report, estimator, grid["items"])

    sizes = [len(split["validation_indices"]) for split in report["folds"]]
    counts = report["class_counts"]
    bounds = _recorded_support_bounds(identity, sizes, counts)
    balanced = [set() for _ in sizes] if scoring == "balanced_accuracy" else None
    extra_label_scores = [set() for _ in sizes] if balanced is not None else None
    eligible = []
    float_finiteness = {}
    for candidate, declaration in zip(candidates, grid["items"]):
        if declaration["kind"] != "dict":
            raise ValueError("Invalid structural candidate declaration.")
        _check_display(declaration, candidate["parameters"], report["class_labels"], float_finiteness)
        settings = _stock_settings(estimator, declaration, candidate["parameters"], candidate["folds"])
        plain = settings is not None and all(_plain_setting(value) for value in settings.values())
        folds = candidate["folds"]
        if (plain and identity["estimator"] == _LOGISTIC and identity.get("sklearn_skip_parameter_validation") is False
                and identity.get("software", {}).get("scikit-learn") in _AUDITED_SOLVER_VERSIONS
                and any(fold["status"] in ("valid", "nonconverged") or "score" in fold for fold in folds)):
            _check_fitted_logistic_nulls(candidate["parameters"], settings)
        version = identity.get("fold_evidence_version")
        tracked_head = _tracked_head(estimator, declaration) if version is not None else None
        for fold in folds:
            index = fold["fold"]
            target = _validate_fold(fold, scoring, other, sizes[index], identity, settings, plain)
            if tracked_head is not None:
                _tracked_fit_disqualifies(fold, tracked_head, version)
            if version == 6 and "convergence_check" in fold:
                _check_observed_flags(fold, identity["estimator"], tracked_head)
            if balanced is not None and "score" in fold:
                _balanced_fold_scores(fold, report, estimator, declaration, balanced, extra_label_scores, target)
        invalid = sum(fold["status"] != "valid" for fold in folds)
        if candidate["valid"] is not (invalid == 0) or type(candidate["invalid_fold_count"]) is not int or (
            candidate["invalid_fold_count"] != invalid
        ):
            raise ValueError("Candidate eligibility disagrees with its fold outcomes.")
        if "mean_" + other in candidate:
            raise ValueError("Candidate contains a mean for a different objective.")
        mean, alias, std = candidate["mean_score"], candidate["mean_" + scoring], candidate["fold_score_std"]
        if invalid:
            if any(value is not None for value in (mean, alias, std)):
                raise ValueError("An ineligible candidate has aggregate scores.")
        else:
            scores = [fold["score"] for fold in folds]
            if (not all(_number(value) for value in (mean, alias, std)) or mean != alias
                    or mean != np.mean(scores) or std != np.std(scores, ddof=1)):
                raise ValueError("Candidate statistics disagree with its fold scores.")
            eligible.append(candidate)

    if balanced is not None:
        contradiction = "Balanced accuracy is impossible for the validation class counts."
        if identity.get("balanced_accuracy_support_method") in (*_REQUEST_CACHE_METHODS, "confusion-matrix-v4"):
            contradiction = ("Balanced accuracy cannot be verified from the recorded validation class counts. "
                             "The historical writer could score caller-defined label operations without saving "
                             "the confusion rows used by the scorer.")
        _require_score_support(_joint_support_possible(balanced, sizes, bounds, counts[0], extra_label_scores),
                               contradiction)
    winner = min(eligible, key=lambda c: (-c["mean_score"], c["index"]), default=None)
    expected = ((winner["index"], winner["parameters"], winner["mean_score"])
                if winner is not None else (None, None, None))
    saved = (report["selected_candidate_index"], report["selected_parameters"], report["selected_mean_score"])
    # JSON comparison also distinguishes true from 1 and 1 from 1.0 in the
    # selected parameter display, matching the writer's verbatim copy.
    if json.dumps(saved, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Selected calibration result disagrees with the eligible ranking.")
    refinement = report["refinement"]
    status = "pending_valid_candidate" if winner is None else "pending_refinement_and_replication"
    parameters = {name: entry["selected"] for name, entry in refinement["parameters"].items()}
    if (refinement["status"] != status
            or json.dumps(parameters, sort_keys=True) != json.dumps(expected[1] or {}, sort_keys=True)):
        raise ValueError("Refinement disagrees with the recorded selection.")
    if parameters:
        _check_refinement(report, grid["items"], winner["index"], estimator, recorded_modes)
    elif identity.get("fold_evidence_version") == 6:
        _check_observed_refinement(report, {})
    if "fold_evidence_version" not in identity:
        _check_legacy_winner(candidates, winner, estimator, grid["items"])


def _refinement_point(value, exact_integers):
    if exact_integers and type(value) is int and value.bit_length() > 53:
        return value
    return float(value)


_MAX_NAN_SORT_POINTS = 6


@lru_cache(maxsize=128)
def _short_nan_orders(finite_count, negative_infinity, positive_infinity, nans, sort_rule):
    # All inputs are plain numbers. Finite ranks preserve every comparison
    # while keeping the cache independent of archive values. The writer did
    # not save set iteration order, so consider every permutation (<= 720).
    points = list(range(finite_count)) + [math.nan] * nans
    if negative_infinity:
        points.append(-math.inf)
    if positive_infinity:
        points.append(math.inf)
    orders = set()
    for order in permutations(points):
        if sort_rule == "3.11-3.12":
            # CPython count_run starts with an ascending or strictly descending
            # run. Reversing a strict descent and each subsequent binary
            # insertion leave no adjacent descent: the insertion's immediate
            # neighbors are its last lower/upper comparisons. Conversely,
            # any order with no adjacent descent is already a complete run.
            if any(b < a for a, b in pairwise(order)):
                continue
        else:
            order = sorted(order)
        orders.add(tuple(v if math.isfinite(v) else None for v in order))
    return orders


def _nonfinite_order_matches(points, finite, values, infinities, nans, sort_rule):
    if len(values) == 1:
        return values == [None] and infinities + nans == 1
    if not nans:
        lower, upper = values[0] is None, values[-1] is None
        return (lower + upper == infinities and (lower or -math.inf not in points)
                and (upper or math.inf not in points) and finite == sorted(finite))
    if len(values) > _MAX_NAN_SORT_POINTS or sort_rule is None:
        return True  # The multiplicity and finite-coordinate checks still apply.
    ranks = {value: i for i, value in enumerate(sorted(finite))}
    observed = tuple(ranks[v] if v is not None else None for v in values)
    for negative in (False, True):
        positive = infinities - negative
        if positive not in (0, 1) or (-math.inf in points and not negative) or (math.inf in points and not positive):
            continue
        if observed in _short_nan_orders(len(finite), negative, bool(positive), nans, sort_rule):
            return True
    return False


def _nonfinite_axis_matches(counts, points, finite, values, min_nans, max_nans, opaque, sort_rule, observed=None):
    # Matching wide bytes mean matching values within the writer's ABI.
    # A group of n such rows adds n freshly converted NaNs or one infinity
    # that can coincide with another group's infinity (padding can differ).
    # Bit s records s NaNs supplied by the wide groups. Track whether zero,
    # one or at least two groups chose infinity, which limits possible signs.
    none, one, multiple = 1, 0, 0
    for n, repetitions in Counter(counts.values()).items():
        block = 1
        while repetitions:
            take = min(block, repetitions)
            shift = n * take
            multiple = (multiple << shift) | multiple | one | (none if take >= 2 else 0)
            one = (one << shift) | (none if take == 1 else 0)
            none <<= shift
            repetitions -= take
            block *= 2
    known_inf = int(-math.inf in points) + int(math.inf in points)
    nulls = len(values) - len(finite)
    observed_nan_count = None if observed is None else observed.count("nan")
    if observed is not None and (math.inf in points and "+inf" not in observed
                                  or -math.inf in points and "-inf" not in observed):
        return False
    for groups, possible in enumerate((none, one, multiple)):
        if not possible:
            continue
        low = int(bool(groups) and not known_inf)
        for wide_inf in range(low, min(2 - known_inf, groups) + 1):
            for opaque_inf in range(min(2 - known_inf - wide_inf, opaque) + 1):
                infinities = known_inf + wide_inf + opaque_inf
                nans = nulls - infinities
                if observed_nan_count is not None and observed_nan_count != nans:
                    continue
                # Unassigned opaque rows can be omitted/duplicates or NaNs.
                low_count = max(0, nans - max_nans - (opaque - opaque_inf))
                high_count = nans - min_nans
                if high_count < low_count or not ((possible >> low_count) & ((1 << (high_count - low_count + 1)) - 1)):
                    continue
                if _nonfinite_order_matches(points, finite, values, infinities, nans, sort_rule):
                    return True
    return False


def _small_nan_orders(positions, nan_index, width):
    # Hash the writer's actual int/float coordinates at the specified pointer
    # width. A binary denominator is always invertible modulo this odd prime.
    modulus = (1 << (61 if width == 64 else 31)) - 1
    declared = [(nan_index, None, 0)]
    for point, index in positions.items():
        numerator, denominator = point.as_integer_ratio()
        h = abs(numerator) * pow(denominator, -1, modulus) % modulus
        if numerator < 0:
            h = -h
        if h == -1:
            h = -2  # CPython reserves -1 as the hash error indicator.
        declared.append((index, point, h % (1 << width)))
    declared.sort()  # First insertions of the three distinct set entries.

    # CPython 3.11–3.12 set_add_entry keeps up to three entries in the original
    # eight slots. The NaN's address hash is lost, so allow any free slot at
    # its insertion. The finite hashes determine every other probe.
    orders = set()
    for nan_slot in range(8):
        slots = {}
        for _, point, perturb in declared:
            if point is None:
                if nan_slot in slots:
                    break
                slot = nan_slot
            else:
                slot = perturb & 7
                while slot in slots:
                    perturb >>= 5
                    slot = (5 * slot + 1 + perturb) & 7
            slots[slot] = point
        else:
            order = [slots[i] for i in sorted(slots)]
            # With at most three entries, count_run and binarysort keep the NaN in
            # place. A middle NaN also keeps the two finite neighbors apart.
            nan_position = order.index(None)
            if nan_position in (0, len(order) - 1):
                order = sorted(positions)
                order.insert(nan_position, None)
            orders.add(tuple(order))
    return orders


def _matches_refinement_axis(entry, axis, missing_keys, inactive, exact_integers, sort_rule="runtime", *,
                             verify_parameter=None, observed=None, optional_points=()):
    # The writer reads the raw candidates for coordinates but classifies the
    # selected parameter after JSON normalization. For example, a 0-D array
    # can have a numeric selection while contributing no point to the axis.
    # The row indices also recover insertion order on a three-coordinate axis.
    points = {}
    unknown = missing_keys
    python_nan_bits = set()
    python_nan_occurrences = 0
    numpy_nans = 0
    wide_nonfinite = Counter()
    first_nan = first_uncertain = None
    for index, (node, display) in enumerate(axis):
        kind = node["kind"]
        dtype = node["dtype"]["str"][1:2] if kind == "numpy_scalar" else ""
        if kind == "float" and display is None:
            # The sign of infinity survives in a Python float declaration.
            bits = int(node["bits"], 16)
            if bits & ((1 << 52) - 1):
                python_nan_bits.add(bits)
                python_nan_occurrences += 1
            else:
                points.setdefault(-math.inf if bits >> 63 else math.inf, index)
        elif kind in ("int", "float") or dtype in ("i", "u", "f"):
            if type(display) in (int, float):
                points.setdefault(_refinement_point(display, exact_integers), index)
            elif dtype == "f" and type(display) is dict:
                points.setdefault(_extended_display(display, binary64=node["dtype"]["str"][2:] == "8"), index)
            elif display is None and re.fullmatch(r"[<>]f[248]", node["dtype"]["str"]):
                if _matches_infinity(node, math.inf):
                    points.setdefault(math.inf, index)
                elif _matches_infinity(node, -math.inf):
                    points.setdefault(-math.inf, index)
                else:
                    # A genuine nonfinite IEEE value other than infinity is
                    # NaN. float(numpy_scalar) makes a new float per grid row.
                    numpy_nans += 1
            elif dtype == "f" and display is None:
                # The extended ABI and its padding can hide the sign of an
                # infinity or a NaN payload. Every possibility displays null.
                wide_nonfinite[node["dtype"]["str"], node["content"]] += 1
            else:
                unknown += 1
        elif dtype == "c":
            coordinate = _complex_coordinate(node, display)
            if coordinate is NotImplemented:
                unknown += 1
            elif coordinate is None:
                wide_nonfinite[node["dtype"]["str"], node["content"]] += 1
            elif math.isnan(coordinate):
                numpy_nans += 1
            else:
                points.setdefault(coordinate, index)
        elif dtype == "m" or kind not in (
                "numpy_scalar", "none", "bool", "str", "bytes", "complex", "ndarray",
                "list", "tuple", "dict", "path") and _optional_axis_point(node, display) is not None:
            unknown += 1
        if first_nan is None and (python_nan_occurrences or numpy_nans):
            first_nan = index
        if first_uncertain is None and (wide_nonfinite or unknown):
            first_uncertain = index

    # Reuse of one Python NaN object can collapse multiple references in a
    # set. Different payloads and separately converted NumPy NaNs cannot.
    min_nans = numpy_nans + len(python_nan_bits)
    max_nans = numpy_nans + python_nan_occurrences
    unknown_from_axis = unknown - missing_keys
    optional = {_refinement_point(value, exact_integers) for value in optional_points} - points.keys()
    # A grid row can contribute at most one raw value, even if several
    # caller-defined keys shared a displayed spelling.
    if optional_points:
        unknown = unknown_from_axis + min(missing_keys, len(optional))
    infinities = int(-math.inf in points) + int(math.inf in points)
    min_extra = int(bool(wide_nonfinite) and not infinities)
    max_extra = sum(wide_nonfinite.values())
    if observed is not None:
        values = [None if type(v) is str else v for v in observed]
    else:
        values = entry.get("values")
    if values is None:
        if (verify_parameter is not None
                and len(points) + min_nans + min_extra <= 1 < len(points) + max_nans + max_extra + unknown):
            raise ValueError(f"Refinement axis size cannot be verified for {verify_parameter!r} from the recorded grid.")
        return ("values" not in entry and len(points) + min_nans + min_extra <= 1
                and (inactive or entry["status"] == "single_value_requires_exploration"))
    minimum_length = 0 if observed is not None else 2
    if type(values) is not list or not minimum_length <= len(values) <= len(points) + max_nans + max_extra + unknown:
        return False
    if any(v is not None and type(v) not in (int, float) for v in values):
        return False
    finite = [v for v in values if v is not None]
    known = [v for v in points if v != math.inf and v != -math.inf]
    nulls = len(values) - len(finite)
    if (len(finite) != len(set(finite)) or not set(known).issubset(finite)
            or len(finite) > len(known) + unknown
            or nulls < infinities + min_nans + min_extra):
        return False
    if not unknown:
        if json.dumps(sorted(finite)) != json.dumps(sorted(known)):
            return False
    elif optional_points and len(set(finite) - points.keys() - optional) > unknown_from_axis:
        return False
    determinate_nan_order = False
    if nulls:
        opaque = unknown - (len(finite) - len(known))
        if optional_points:
            # Only a genuinely unknown value can supply an infinity or NaN.
            # A missing lookup whose possible values are finite stays finite.
            opaque = min(opaque, unknown_from_axis - len(set(finite) - points.keys() - optional))
        if not _nonfinite_axis_matches(wide_nonfinite, points, finite, values, min_nans, max_nans, opaque, sort_rule, observed):
            return False
        if (sort_rule == "3.11-3.12" and not missing_keys and nulls == 1 and 2 <= len(values) <= 3
                and not infinities and len(points) == len(finite)):
            nan_index = None
            if first_nan is not None:
                # One null forces repeated Python NaNs to alias. Once two
                # distinct entries are present an opaque row can only add
                # the third or repeat/omit a point, preserving the order.
                if first_uncertain is None or first_uncertain > sorted((*points.values(), first_nan))[1]:
                    nan_index = first_nan
            elif len(values) == 3 and values[1] is None and first_uncertain is not None:
                # A middle null forces a NaN. Its insertion order is known
                # if only one row could supply it or both finite points
                # preceded every such row.
                if unknown + max_extra == 1 or first_uncertain > max(points.values()):
                    nan_index = first_uncertain
            if nan_index is not None:
                orders = _small_nan_orders(points, nan_index, 32) | _small_nan_orders(points, nan_index, 64)
                if tuple(values) not in orders:
                    return False
                determinate_nan_order = len(orders) == 1 and min_nans == max_nans == 1
    elif finite != sorted(finite):
        return False
    if observed is not None and len(values) < 2:
        return True
    selected = _refinement_point(entry["selected"], exact_integers)
    if selected not in values:
        return False
    location = values.index(selected)
    status = ("lower_boundary_requires_expansion_or_constraint" if location == 0
              else "upper_boundary_requires_expansion_or_constraint" if location == len(values) - 1
              else "interior_interval_requires_refinement")
    if not inactive and entry["status"] != status:
        return False
    if verify_parameter is not None and (unknown or max_extra or max_nans and not determinate_nan_order):
        # The saved declarations do not determine the complete ordered grid.
        # In particular, the allocation of NaNs affects iteration of the
        # writer's set. A compatible displayed order alone cannot verify it.
        raise ValueError(f"Refinement axis membership or ordering for {verify_parameter!r} cannot be verified from the recorded grid.")
    return True


_BINARY_SCREEN = "supervised_nmf.screening.BinaryScreening"
_ACTIVE_CUTOFF = {"absolute": "score_threshold", "percentile": "percentile", "none": None}
_SCREEN_COMPOSITIONS = {
    "sklearn.pipeline.Pipeline": ("steps", 2),
    "sklearn.pipeline.FeatureUnion": ("transformer_list", 2),
    "sklearn.compose._column_transformer.ColumnTransformer": ("transformers", 3),
}


def _ordinary_screen_lookup(implementation, sklearn_version):
    cls = implementation["type"]
    if cls in (_NMF_CLASSIFIER, _BINARY_SCREEN, _LOGISTIC, "sklearn.neighbors._classification.KNeighborsClassifier",
               "sklearn.dummy.DummyClassifier", "sklearn.preprocessing._data.StandardScaler",
               "sklearn.preprocessing._data.MinMaxScaler"):
        return True
    if sklearn_version not in ("1.9.0", "1.9.1") or not cls.startswith("sklearn."):
        return False
    # In 1.9.0 and 1.9.1 the production BaseEstimator get/set_params overrides are
    # named compositions and FrozenEstimator. Gaussian-process kernels have
    # their own parameter protocol. Test helpers also supply special setters.
    parents = {p["type"] for p in implementation.get("implementations", ())}
    return ("sklearn.base.BaseEstimator" in parents and not parents.intersection({
        "sklearn.utils.metaestimators._BaseComposition", "sklearn.frozen._frozen.FrozenEstimator",
        "sklearn.gaussian_process.kernels.Kernel",
    }) and not any(".tests." in p or p.startswith("sklearn.utils._testing.") for p in parents))


def _plain_screen_inspection(node, sklearn_version=None):
    """Identify stock constructor trees whose deep parameter lookup is predictable."""
    if node is NotImplemented:
        return False
    if node["kind"] in ("tuple", "list"):
        return all(_plain_screen_inspection(item, sklearn_version) for item in node["items"])
    if node["kind"] == "dict":
        return all(_plain_screen_inspection(item, sklearn_version) for pair in node["items"] for item in pair)
    if node["kind"] == "estimator":
        cls = node["implementation"]["type"]
        return ((cls in _SCREEN_COMPOSITIONS or _ordinary_screen_lookup(node["implementation"], sklearn_version))
                and _plain_screen_inspection(node["parameters"], sklearn_version))
    return _plain_setting(node)


def _screen_component(estimator, path, overrides, sklearn_version=None):
    """Follow the stock deep-parameter paths the archive can reconstruct."""
    prefix = ""
    for name in path.split("__") if path else ():
        if estimator is NotImplemented or estimator is None:
            return estimator
        if estimator["kind"] in ("none", "bool", "int", "float", "str", "list", "tuple", "dict", "ndarray", "numpy_scalar"):
            return None
        if estimator["kind"] != "estimator":
            return NotImplemented
        cls = estimator["implementation"]["type"]
        params = _string_items(estimator["parameters"])
        if params is None:
            return NotImplemented
        if cls in _SCREEN_COMPOSITIONS:
            collection, arity = _SCREEN_COMPOSITIONS[cls]
            steps = overrides.get(prefix + collection, params[collection])
            if steps is NotImplemented or steps["kind"] not in ("tuple", "list"):
                return NotImplemented
            for pair in steps["items"]:
                if pair["kind"] not in ("tuple", "list") or len(pair["items"]) != arity:
                    return NotImplemented
                key, component = pair["items"][:2]
                spelling = key.get("key_text") if key["kind"] == "numpy_scalar" else None
                if spelling is None:
                    spelling = _text_choice(key, (name,))
                if spelling == name:
                    if re.search(r"[\U00010000-\U0010ffff]", name):
                        if not _exact_numpy_text(key, name):
                            return NotImplemented  # The original spelling is ambiguous in JSON.
                    # _BaseComposition replaces named steps only in a list.
                    if steps["kind"] == "tuple" and prefix + name in overrides:
                        return NotImplemented
                    if arity == 3 and prefix + collection in overrides and any(
                            key == prefix + name or key.startswith(prefix + name + "__") for key in overrides):
                        # ColumnTransformer snapshots the old named children before replacing
                        # its public collection. A nested update can reach the old child.
                        return NotImplemented
                    params[name] = component
                    break
                if spelling is None:
                    # A custom or padded step name could hide a match.
                    return NotImplemented
        elif not _ordinary_screen_lookup(estimator["implementation"], sklearn_version):
            # A third-party get_params(deep=True) can expose computed paths.
            return NotImplemented
        if name not in params:
            return None
        estimator = overrides.get(prefix + name, params[name])
        prefix += name + "__"
    return estimator


def _screen_modes(estimator, path, overrides, sklearn_version=None):
    """Return the modes permitted by the effective screen and its declaration."""
    modes = set(_ACTIVE_CUTOFF)
    component = _screen_component(estimator, path, overrides, sklearn_version)
    if component is NotImplemented:
        return modes
    if component is None or component["kind"] != "estimator":
        return set()
    implementation = component["implementation"]
    cls = implementation["type"]
    if cls not in (_NMF_CLASSIFIER, _BINARY_SCREEN):
        if cls == "sklearn.pipeline.Pipeline" and not path:
            # A step named "" gets the same diagnostic path as the root.
            params = _string_items(component["parameters"])
            steps = overrides.get("steps", params.get("steps")) if params is not None else None
            if steps is None or steps is NotImplemented or steps["kind"] not in ("tuple", "list"):
                return modes
            if any(pair["kind"] in ("tuple", "list") and pair["items"]
                   and _text_choice(pair["items"][0], ("",)) == "" for pair in steps["items"]):
                return modes
        # isinstance recognizes subclasses, which can override set_params.
        mro = {parent["type"] for parent in implementation.get("implementations", ())}
        if not mro or _NMF_CLASSIFIER in mro:
            return modes
        return modes - {"none"} if _BINARY_SCREEN in mro else set()
    if cls == _BINARY_SCREEN:
        modes.remove("none")
    params = _string_items(component["parameters"])
    if params is None:
        return modes
    mode = overrides.get((path + "__" if path else "") + "screening", params["screening"])
    return modes & _declared_screen_modes(mode)


def _declared_screen_modes(mode):
    modes = set(_ACTIVE_CUTOFF)
    if mode is NotImplemented:
        return modes
    if mode["kind"] == "str":
        return modes.intersection([mode["value"]])
    if mode["kind"] == "numpy_scalar" and mode["dtype"]["str"][1:2] == "U":
        if "key_text" in mode:
            return modes.intersection([mode["key_text"]])
        # A legacy scalar retained its dtype and hash, even without a spelling.
        possible = set()
        for choice in modes:
            try:
                _check_display(mode, choice)
            except ValueError:
                continue
            possible.add(choice)
        return possible
    # A str subclass with get_params can have an estimator identity; the
    # recorded MRO still tells us whether isinstance(mode, str) was possible.
    if mode["kind"] == "estimator":
        parents = mode["implementation"].get("implementations")
        if parents is None or any(parent["type"] == "builtins.str" for parent in parents):
            return modes
    return set()


def _legacy_screen_modes(estimator, path, overrides, unknown, sklearn_version=None):
    """Possible recorded modes at a path, including absence of a screen."""
    modes = _screen_modes(estimator, path, overrides, sklearn_version)
    mode_key = (path + "__" if path else "") + "screening"
    if mode_key not in overrides:
        for value in unknown:
            modes.update(_screen_modes(estimator, path, {**overrides, mode_key: value}, sklearn_version))
    # A hidden replacement may install a different child, replace a collection,
    # or change the spelling under which descendants appear in get_params.
    # The root estimator's class is fixed.
    if path and any(value["kind"] in ("estimator", "list", "tuple") for value in unknown):
        return set(_ACTIVE_CUTOFF) | {None}
    if not path and estimator["kind"] == "estimator" and estimator["implementation"]["type"] in _SCREEN_COMPOSITIONS:
        if any(value["kind"] in ("list", "tuple") for value in unknown):
            return set(_ACTIVE_CUTOFF) | {None}
    if not modes or _screen_component(estimator, path, overrides, sklearn_version) is NotImplemented:
        modes.add(None)
    return modes


def _inactive_screening_modes(report, winner, estimator, declarations, *, bound_observation=False):
    sklearn_version = report["identity"].get("software", {}).get("scikit-learn")
    parameters = report["candidates"][winner]["parameters"]
    declared = declarations[winner]["items"]
    # These are the original values belonging to recoverable raw keys, even
    # when another key overwrote their public display during serialization.
    overrides, unknown, cache = {}, [], {}
    for key, value in declared:
        name = _refinement_key(key, parameters, cache)
        if name is None:
            unknown.append(value)
        else:
            overrides[name] = value
    row, = (row for row in report["screening_diagnostics"]["candidates"] if row["index"] == winner)
    observations = {}
    inactive = {}
    for screen in row["screens"]:
        mode = screen["mode"]
        names = screen["inactive_grid_parameters"]
        active = screen["active_parameter"]
        path = screen["path"]
        if path == "(root)":
            # A Pipeline may also name a nested step "(root)". Its actual
            # parameter names still include the prefix.
            if mode == "none" and active is None and not names:
                continue
            if not ((active or "").startswith("(root)__") or any(name.startswith("(root)__") for name in names)):
                path = ""
        prefix = path + "__" if path else ""
        effective_mode = mode
        if mode not in _ACTIVE_CUTOFF:
            # A str subclass can compare as a standard mode while serializing
            # with another spelling. A plain declared mode cannot do this.
            matches = [m for m in _ACTIVE_CUTOFF if active == (prefix + _ACTIVE_CUTOFF[m] if _ACTIVE_CUTOFF[m] else None)]
            if len(matches) != 1:
                raise ValueError("Inactive refinement disagrees with the declared screening mode.")
            effective_mode, = matches
        if active != (prefix + _ACTIVE_CUTOFF[effective_mode] if _ACTIVE_CUTOFF[effective_mode] else None):
            raise ValueError("Inactive refinement disagrees with the active screening parameter.")
        allowed = {prefix + cutoff for cutoff in ("percentile", "score_threshold") if cutoff != _ACTIVE_CUTOFF[effective_mode]}
        claimed = set(names)
        if (type(names) is not list or len(claimed) != len(names) or not claimed <= allowed
                or allowed.intersection(overrides).difference(claimed) or inactive.keys() & claimed):
            raise ValueError("Inactive refinement disagrees with the screening path or parameter.")
        # Empty rows without a tuned cutoff do not affect refinement. Their
        # display can alias a root or a nested empty-named component.
        if names or active in overrides:
            observations[path] = (effective_mode, mode != effective_mode)
        inactive.update(dict.fromkeys(names, mode))
    spare = len(unknown) - len(inactive.keys() - overrides.keys())
    if spare < 0:
        raise ValueError("Inactive refinement requires more raw parameter keys than the candidate declares.")
    grouped = [(group["parameter"], group["mode"]) for group in report["screening_diagnostics"]["inactive_parameters"]
               for index in group["candidate_indices"] if index == winner]
    if Counter(grouped) != Counter(inactive.items()):
        # The group inherits the first matching raw mode object, even when a
        # later winner uses another spelling that compared equal to it.
        if (Counter(name for name, _ in grouped) != Counter(inactive.keys())
                or _plain_screen_inspection(estimator, sklearn_version)
                and all(_plain_screen_inspection(d, sklearn_version) for d in declarations)):
            raise ValueError("Inactive refinement disagrees with the grouped screening diagnostics.")
        for group in report["screening_diagnostics"]["inactive_parameters"]:
            indices = group["candidate_indices"]
            if winner in indices:
                first = report["screening_diagnostics"]["candidates"][indices[0]]
                if not any(group["parameter"] in s["inactive_grid_parameters"] and s["mode"] == group["mode"]
                           for s in first["screens"]):
                    raise ValueError("Grouped screening mode disagrees with its first observed row.")
    # Check structural contradictions in the diagnostic rows even when the
    # display collapsed the raw cutoff names and they affect no public advice.
    by_path = {}
    for path, (mode, custom_spelling) in observations.items():
        modes = by_path[path] = _legacy_screen_modes(estimator, path, overrides, unknown, sklearn_version)
        if mode not in modes or custom_spelling and len(modes - {None}) == 1:
            raise ValueError("Inactive refinement disagrees with the declared screen or screening mode.")

    plain_inspection = (_plain_screen_inspection(estimator, sklearn_version)
                        and all(_plain_screen_inspection(value, sklearn_version) for _, value in declared))
    for name in parameters:
        path, _, cutoff = name.rpartition("__")
        if cutoff not in ("score_threshold", "percentile"):
            continue
        if path not in by_path:
            by_path[path] = _legacy_screen_modes(estimator, path, overrides, unknown, sklearn_version)
        # Record exactly what mark_inactive_refinement could have applied to
        # this displayed name. Uncertainty on another key does not block it.
        possible = {mode if mode is not None and _ACTIVE_CUTOFF[mode] != cutoff else None
                    for mode in by_path[path]}
        if name not in overrides or not plain_inspection:
            # An opaque key may merely display this name. A custom deep lookup
            # can also fail, leaving the writer with an empty screening row.
            possible.add(None)
        observed = inactive.get(name)
        if len(possible) > 1 and not bound_observation:
            raise ValueError(f"Inactive refinement for {name!r} cannot be verified from the legacy screening evidence.")
        if observed not in possible:
            raise ValueError("Inactive refinement disagrees with the declared screen or screening mode.")
    return inactive


def _check_observed_refinement(report, inactive_modes):
    """Project the final writer's ordered numeric inputs into its public advice."""
    selected = report["selected_parameters"] or {}
    axes = report.get("refinement_axis_inputs")
    if type(axes) is not dict or axes.keys() != {name for name, value in selected.items() if type(value) in (int, float)}:
        raise ValueError("Refinement axis observations do not cover the selected numeric parameters.")
    for name, value in selected.items():
        expected = {"selected": value, "status": "categorical"}
        if name in axes:
            observation = axes[name]
            if (type(observation) is not dict or observation.keys() != {"coordinates", "includes_none"}
                    or type(observation["coordinates"]) is not list or type(observation["includes_none"]) is not bool):
                raise ValueError(f"Invalid refinement axis observation for {name!r}.")
            coordinates = observation["coordinates"]
            tokens = {"nan": math.nan, "+inf": math.inf, "-inf": -math.inf}
            if any(type(v) not in (int, float) and (type(v) is not str or v not in tokens)
                   or type(v) is float and not math.isfinite(v)
                   or type(v) is int and v.bit_length() <= 53 for v in coordinates):
                raise ValueError(f"Invalid refinement coordinate for {name!r}.")
            distinct = [v for v in coordinates if v != "nan"]
            comparable = [tokens[v] if type(v) is str else v for v in coordinates]
            order_checkable = ("nan" not in coordinates or len(coordinates) <= _MAX_NAN_SORT_POINTS
                               and report["identity"]["python"].startswith(("3.11.", "3.12.")))
            if (len(set(distinct)) != len(distinct)
                    or order_checkable and any(a > b for a, b in pairwise(comparable))):
                raise ValueError(f"Refinement axis observation for {name!r} is not a sorted unique grid.")
            if len(coordinates) < 2:
                expected["status"] = "single_value_requires_exploration"
            else:
                if value not in coordinates:
                    raise ValueError(f"Selected refinement coordinate is missing for {name!r}.")
                location = coordinates.index(value)
                expected["status"] = ("lower_boundary_requires_expansion_or_constraint" if location == 0
                                      else "upper_boundary_requires_expansion_or_constraint" if location == len(coordinates) - 1
                                      else "interior_interval_requires_refinement")
                expected["values"] = [None if type(v) is str else v for v in coordinates]
            if observation["includes_none"]:
                expected["includes_none"] = True
        if name in inactive_modes:
            expected.update(status="inactive_for_selected_screening_mode", screening_mode=inactive_modes[name])
        if _json_digest(expected) != _json_digest(report["refinement"]["parameters"][name]):
            raise ValueError(f"Refinement for {name!r} disagrees with the observed grid and screen.")


def _axis_sources(declaration, display):
    """Recover raw string lookups separately from the possibly lossy display."""
    known, hidden, cache = {}, [], {}
    collision = len(declaration["items"]) != len(display)
    for key, value in declaration["items"]:
        name = _refinement_key(key, display, cache)
        if not collision and name in display:
            shown = display[name]
        elif value["kind"] == "int":
            shown = int(value["value"], 0)
        elif value["kind"] == "float":
            number = struct.unpack(">d", bytes.fromhex(value["bits"]))[0]
            shown = number if math.isfinite(number) else None
        else:
            shown = value.get("display", NotImplemented)
        if name is None:
            hidden.append((value, shown))
        else:
            known[name] = value, shown
    return known, hidden


def _optional_axis_point(node, display):
    """Possible contribution of a grid entry whose lookup is unresolved."""
    kind = node["kind"]
    if kind in ("bool", "bytes", "complex", "dict", "list", "ndarray", "none", "path", "str", "tuple"):
        return None
    if kind == "numpy_scalar" and node["dtype"]["str"][1:2] not in ("i", "u", "f", "c", "m"):
        return None
    if kind == "estimator":
        parents = node.get("implementation", {}).get("implementations")
        if parents and not any(p["type"] in ("builtins.int", "builtins.float", "numpy.number") for p in parents):
            return None
    if type(display) is int or type(display) is float and math.isfinite(display):
        return display
    return NotImplemented


def _check_refinement(report, declarations, winner, estimator, recorded_modes):
    identity = report["identity"]
    # Run detection changed in CPython 3.13. For other foreign versions the
    # archive cannot establish this interpreter's NaN ordering behavior.
    python = ".".join(identity.get("python", "").split(".")[:2])
    sort_rule = ("3.11-3.12" if python in ("3.11", "3.12")
                 else "runtime" if python == f"{sys.version_info.major}.{sys.version_info.minor}" else None)
    # 0.5.0 and 0.5.1 shipped identical evaluation.py bytes. 0.6.0 and
    # subsequent fold-evidence writers retained integers beyond 53 bits.
    source = identity.get("source_sha256", {}).get("evaluation.py")
    if source == "c5380dbfcce9acb5b2f9b701673c242ba1e3e07eb2379739d1c9fc0312ba8952":
        policies = (False,)
    elif "fold_evidence_version" in identity or source == "15109e637599b82a1bc75670dd04802a305a3b6f89e494ca15d273bb08a9d05c":
        policies = (True,)
    else:
        policies = (True, False)
    raw = [_axis_sources(item, candidate["parameters"])
           for item, candidate in zip(declarations, report["candidates"])]
    inactive_modes = (recorded_modes if recorded_modes is not None else
                      _inactive_screening_modes(report, winner, estimator, declarations))
    has_observation = identity.get("fold_evidence_version") == 6
    if has_observation:
        _check_observed_refinement(report, inactive_modes)
    for name, entry in report["refinement"]["parameters"].items():
        inactive = name in inactive_modes
        if (inactive != (entry["status"] == "inactive_for_selected_screening_mode")
                or inactive and entry.get("screening_mode") != inactive_modes[name]
                or not inactive and "screening_mode" in entry):
            raise ValueError("Inactive refinement disagrees with the recorded screening diagnostics.")
        if type(entry["selected"]) not in (int, float):
            if (not inactive and entry["status"] != "categorical") or "values" in entry or "includes_none" in entry:
                raise ValueError("Refinement status disagrees with the selected parameter type.")
            continue
        axis, optional = [], []
        missing_keys = 0
        optional_none = False
        uncertain_winner_key = name not in raw[winner][0] and re.search(r"[\U00010000-\U0010ffff]", name)
        for known, hidden in raw:
            if name in known and not uncertain_winner_key:
                value, shown = known[name]
                if shown is not NotImplemented or _optional_axis_point(value, shown) is None:
                    axis.append((value, shown))
                    continue
                possibilities = [known[name]]
            else:
                possibilities = hidden + ([known[name]] if name in known else [])
            optional_none |= any(value["kind"] == "none" for value, _ in possibilities)
            points = [_optional_axis_point(value, shown) for value, shown in possibilities]
            points = [point for point in points if point is not None]
            if points:
                optional.extend(points)
                missing_keys += 1
        includes_none = any(node["kind"] == "none" for node, _ in axis)
        if ("includes_none" in entry and (entry["includes_none"] is not True or not (includes_none or optional_none))
                or includes_none and "includes_none" not in entry):
            raise ValueError("Refinement disagrees with the declared None alternative.")
        if optional_none and not includes_none and not has_observation:
            raise ValueError(f"Refinement None alternative for {name!r} cannot be verified from the recorded grid.")
        observation = report["refinement_axis_inputs"][name]["coordinates"] if has_observation else None
        answers = []
        for policy in policies:
            try:
                answers.append(_matches_refinement_axis(entry, axis, missing_keys, inactive, policy, sort_rule,
                                                        verify_parameter=None if has_observation else name, observed=observation,
                                                        optional_points=() if NotImplemented in optional else optional))
            except OverflowError:
                # This integer could never have been narrowed by an older writer.
                continue
        if not any(answers):
            raise ValueError("Refinement status or values disagree with the numeric grid.")
        if not has_observation and not all(answers):
            raise ValueError(f"Refinement coordinate policy for {name!r} cannot be verified from the recorded grid.")


def _unique_json_fields(pairs):
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"Duplicate JSON field in calibration report: {key!r}")
        fields[key] = value
    return fields


def _finite_json_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("Calibration report contains a nonfinite JSON number.")
    return value


def _decode_calibration_report(raw):
    report = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_fields,
                        parse_constant=_finite_json_float, parse_float=_finite_json_float)
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report["schema_version"] != 2):
        raise ValueError("Expected a schema-2 calibration checkpoint.")
    return report


def _float_class_value(label):
    if type(label) is float:
        return label
    # The writer preserves extended floats as exact dyadics on platforms
    # where NumPy cannot unbox them. Two distinct classes can round to the
    # same binary64 value, so compare the saved significands and exponents.
    try:
        _extended_display(label, binary64=False)
    except ValueError as error:
        raise ValueError("Class labels disagree with their recorded floating dtype.") from error
    return int(label["significand"]) * Fraction(2) ** label["exponent"]


def _finite_float_class_values(labels):
    # The writer uses one finite encoding for the common float dtype. A null
    # can stand for any nonfinite value, including on either side of a finite
    # class. Its presence leaves the other class's exact value intact.
    finite = [label for label in labels if label is not None]
    if not (all(type(label) is float for label in finite) or all(type(label) is dict for label in finite)):
        return None
    try:
        values = [_float_class_value(label) for label in finite]
    except ValueError:
        return None  # An object display can also be a dictionary.
    return values if len(values) < 2 or values[0] < values[1] else None


def _archived_float_domain(labels, values):
    domain = _float_label_domain(values)
    if len(values) == len(labels) or domain == _FLOAT_OUTSIDE:
        return domain
    # Among native NumPy floats, only float16 can overflow on int64 -> float.
    # An extended encoding or a finite peer unrepresentable in float16 rules
    # it out. All other native formats cast every int64 to a finite float, so
    # no value represented by the null could have survived the round trip.
    if any(type(label) is dict for label in labels) or any(
            abs(value) > 65504 or float(np.float16(value)) != value for value in values):
        return _FLOAT_OUTSIDE
    return _FLOAT_NULL


def _check_legacy_float_scores(report, labels):
    identity = report["identity"]
    if identity.get("software", {}).get("scikit-learn") not in _AUDITED_TARGET_VERSIONS:
        return
    values = _finite_float_class_values(labels)
    if values is None:
        return
    domain = _archived_float_domain(labels, values)
    if domain is None:
        return
    # Version 3's retained-support witness was emitted only for native
    # numeric targets. Otherwise an object label could have emitted these
    # same two floats as its display.
    proven_numeric = (identity.get("fold_evidence_version") == 3 and identity["scoring"] == "balanced_accuracy"
                      and any("balanced_accuracy_support" in fold and "score" in fold
                              for candidate in report["candidates"] for fold in candidate["folds"]))
    if proven_numeric:
        error = {_FLOAT_OUTSIDE: _FLOAT_SCORE_ERROR, _FLOAT_BOUNDARY: _MISSING_FLOAT_BOUNDARY,
                 _FLOAT_NULL: _MISSING_FLOAT_CLASS}[domain]
    else:
        error = ("The historical report omitted the target dtype. Its floating class displays could represent "
                 "native floats or object labels, so a score on the original labels cannot be verified.")
    _check_original_float_scores(report, error)


def _bytes_class_value(label):
    # The writer stores repr of the bytes returned by ndarray.tolist().
    # Decode only the flat escapes that bytes repr emits.
    if type(label) is str and label.isascii():
        value = re.sub(rb"""\\(?:x[0-9a-f]{2}|[\\'"tnr])""",
                       lambda match: codecs.escape_decode(match[0])[0], label[2:-1].encode("ascii"))
        if repr(value) == label:
            return value
    raise ValueError("Class labels disagree with their recorded bytes dtype.")


def _check_bytes_or_void_labels(kind, labels):
    if kind == "V" and type(labels[0]) is list:
        # A structured dtype returns one tuple per record. Its fields can
        # contain objects with lossy displays, even for two distinct records.
        if type(labels[1]) is not list or not labels[0] or len(labels[0]) != len(labels[1]):
            raise ValueError("Structured class labels must have the same field count.")
        return
    first, second = map(_bytes_class_value, labels)
    # Fixed strings drop all trailing NUL padding. Unstructured voids
    # preserve every byte of their common itemsize, including trailing NULs.
    if (first >= second or kind == "S" and (first.endswith(b"\0") or second.endswith(b"\0"))
            or kind == "V" and len(first) != len(second)):
        raise ValueError("Byte class labels disagree with their recorded dtype or class order.")


def _check_archived_partition(report, identity):
    cv = identity["cv"]
    splits, declared = report["folds"], identity["folds"]
    counts, labels = report["class_counts"], report["class_labels"]
    if (type(cv) is not int or cv < 2 or not isinstance(splits, list) or len(splits) != cv
            or not isinstance(declared, list) or len(declared) != cv):
        raise ValueError("Calibration report has an incomplete fold plan.")
    if (not isinstance(counts, list) or len(counts) != 2
            or any(type(n) is not int or n < cv for n in counts)
            or not isinstance(labels, list) or len(labels) != 2):
        raise ValueError("Calibration report has invalid binary class counts or labels.")
    unordered_numbers = all(type(v) in (int, float, bool) for v in labels) and labels[0] >= labels[1]
    uncertain_labels = False
    float_domain = None
    if type(identity.get("fold_evidence_version")) is int and identity["fold_evidence_version"] >= 4:
        evidence = identity["class_label_evidence"]
        kind = evidence["dtype_kind"]
        if type(kind) is not str or len(kind) != 1 or kind not in "biufcmMOSUVT":
            raise ValueError("Invalid recorded class label dtype kind.")
        new_inputs = identity.get("balanced_accuracy_support_method") == _BALANCED_SUPPORT_METHOD
        if new_inputs and kind == "T":
            raise ValueError("The recorded scorer target dtype must reflect StringDType normalization.")
        if ("plain_string_values" in evidence) != (new_inputs and kind == "O") or (
                "plain_string_values" in evidence and type(evidence["plain_string_values"]) is not bool):
            raise ValueError("Missing or invalid target object-string observation.")
        if identity.get("balanced_accuracy_support_method") in _BALANCED_SUPPORT_METHODS and (
            kind in "US" and type(evidence.get("native_text")) is not bool or
            kind not in "US" and "native_text" in evidence
        ):
            raise ValueError("Invalid native text target dtype observation.")
        if ("dtype_has_metadata" in evidence and
                (identity["scoring"] != "balanced_accuracy" or type(evidence["dtype_has_metadata"]) is not bool)):
            raise ValueError("Invalid target dtype metadata observation.")
        if ("opaque_unique_cache" in evidence and
                (type(evidence["opaque_unique_cache"]) is not bool or "dtype_has_metadata" not in evidence
                 or evidence["opaque_unique_cache"] and not evidence["dtype_has_metadata"]
                 or identity.get("balanced_accuracy_support_method") not in _BALANCED_SUPPORT_METHODS)):
            raise ValueError("Invalid target unique-label cache observation.")
        if ("has_unique_cache" in evidence and
                (type(evidence["has_unique_cache"]) is not bool or "opaque_unique_cache" not in evidence
                 or evidence["has_unique_cache"] and not evidence["dtype_has_metadata"]
                 or evidence["opaque_unique_cache"] and not evidence["has_unique_cache"])):
            raise ValueError("Invalid target unique-label cache observation.")
        if json.dumps(evidence["labels"], sort_keys=True) != json.dumps(labels, sort_keys=True):
            raise ValueError("Class labels disagree with their recorded request identity.")
        if kind == "b" and (labels[0] is not False or labels[1] is not True):
            raise ValueError("Class labels disagree with their recorded numeric dtype.")
        if kind in "iu":
            # Only the kind is retained. The widest native integer dtype in
            # each family bounds every possible original width.
            low, stop = (-(2**63), 2**63) if kind == "i" else (0, 2**64)
            if any(type(label) is not int or not low <= label < stop for label in labels):
                raise ValueError("Class labels disagree with their recorded numeric dtype.")
        if kind == "U" or evidence.get("plain_string_values") is True:
            if any(type(label) is not str or label.endswith("\x00") for label in labels):
                raise ValueError("Class labels disagree with their recorded Unicode dtype.")
            if labels[0] >= labels[1]:
                # JSON joins adjacent high/low surrogates. Splitting every astral
                # character gives the least possible original first label.
                units = labels[0].encode("utf-16-be", errors="surrogatepass")
                least_first = "".join(chr(unit) for unit, in struct.iter_unpack(">H", units))
                if least_first >= labels[1]:
                    raise ValueError("Unicode class labels cannot represent two ascending native classes.")
        if kind in "SV":
            _check_bytes_or_void_labels(kind, labels)
        if kind == "f":
            floats = _finite_float_class_values(labels)
            if floats is None:
                raise ValueError("Floating class labels must have the same encoding and increase strictly.")
            if identity.get("software", {}).get("scikit-learn") in _AUDITED_TARGET_VERSIONS:
                float_domain = _archived_float_domain(labels, floats)
            elif len(floats) < len(labels):
                float_domain = _FLOAT_NULL
        elif kind in "biu" and unordered_numbers:
            raise ValueError("A native numeric target must have distinct numeric class labels in ascending order.")
    else:
        # Scored object targets can contain string subclasses whose item()
        # emits a duplicate or reversed integer. Older files omit the dtype.
        uncertain_labels = unordered_numbers and any(
            "score" in fold for row in report["candidates"] for fold in row["folds"])
        _check_legacy_float_scores(report, labels)
    if "original_float_target_policy" in identity:
        policy = identity["original_float_target_policy"]
        if None in labels or not (float_domain == _FLOAT_OUTSIDE and policy == _FLOAT_OUTSIDE
                or float_domain == _FLOAT_BOUNDARY
                and policy in (_FLOAT_BOUNDARY_FAILED, _FLOAT_BOUNDARY_PASSED)):
            raise ValueError("Invalid original floating class label qualification in the request identity.")
    else:
        policy = None
    if float_domain == _FLOAT_OUTSIDE or policy == _FLOAT_BOUNDARY_FAILED:
        _check_original_float_scores(report)
    elif float_domain == _FLOAT_BOUNDARY and policy is None:
        _check_original_float_scores(report, _MISSING_FLOAT_BOUNDARY)
    elif float_domain == _FLOAT_NULL:
        _check_original_float_scores(report, _MISSING_FLOAT_CLASS)
    rows = sum(counts)
    validated = set()
    for split, pair in zip(splits, declared):
        train, valid = split["training_indices"], split["validation_indices"]
        if (not isinstance(pair, list) or len(pair) != 2
                or not isinstance(train, list) or not isinstance(valid, list)
                or len(train) < 4 or len(valid) < 2 or len(train) + len(valid) != rows):
            raise ValueError("Calibration report has an invalid training or validation fold.")
        for indices, original in zip((train, valid), pair):
            # bools and integral floats can compare equal to integer row indices.
            if (not isinstance(original, list) or indices != original
                    or any(type(i) is not int for i in original)
                    or any(type(i) is not int or not 0 <= i < rows for i in indices)):
                raise ValueError("Calibration fold indices disagree with the recorded identity.")
        validation_rows = set(valid)
        training_rows = set(train)
        if (len(training_rows) != len(train) or len(validation_rows) != len(valid)
                or not training_rows.isdisjoint(validation_rows)
                or not validated.isdisjoint(validation_rows)):
            raise ValueError("Calibration folds must partition the rows without overlap.")
        validated.update(validation_rows)
    if len(validated) != rows:
        raise ValueError("Calibration validation folds do not cover every row.")

    candidates = report["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Calibration report requires at least one candidate.")
    for index, candidate in enumerate(candidates):
        records = candidate["folds"]
        if (type(candidate["index"]) is not int or candidate["index"] != index
                or not isinstance(records, list) or len(records) != cv):
            raise ValueError("Calibration report has incomplete candidate or fold coverage.")
        fold_ids = [record["fold"] for record in records]
        if (any(type(i) is not int or not 0 <= i < cv for i in fold_ids)
                or len(set(fold_ids)) != cv):
            raise ValueError("A calibration candidate has invalid or duplicate fold indices.")
    return uncertain_labels


def _check_completed_cv_report(report, identity):
    """Check a completed CV result before returning it to a caller."""
    try:
        if (report["state"] != "cv_completed" or identity["refit"] is not False
                or any(report[key] is not False for key in
                       ("refit_performed", "is_fallback", "tuning_complete", "defaults_frozen"))
                or report["final_fit_attempts"] != [] or "pending_final_fit" in report
                or type(report["final_fit_seconds"]) not in (float, int) or report["final_fit_seconds"] != 0):
            raise ValueError("Expected a completed CV-only calibration without refitting.")
        duration = report.get("tuning_seconds")
        if not _number(duration) or duration < 0:
            raise ValueError("Invalid or missing tuning_seconds in completed calibration.")
        pools = report.get("active_threadpools")
        if type(pools) is not list or any(type(pool) is not dict for pool in pools):
            raise ValueError("Invalid or missing active_threadpools in completed calibration.")
        uncertain_labels = _check_archived_partition(report, identity)
        _validate_outcomes(report)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("Malformed calibration report.") from error
    if uncertain_labels:
        raise ValueError("The legacy numeric class display is ambiguous. The original class labels "
                         "cannot be independently verified.")


def load_calibration_report(path, *, expected_sha256=None):
    """Read a completed schema-2 CV-only calibration checkpoint by file path.

    Return the historical report dictionary after checking its hashes, folds,
    scores and selection. An optional expected_sha256 verifies the exact
    bytes against a separately trusted record. A result whose decisive
    evidence cannot be verified raises ValueError, including when the digest
    matches. Malformed, inconsistent or unsupported reports also raise
    ValueError. File access errors propagate as OSError.
    """
    try:
        if expected_sha256 is not None and (not isinstance(expected_sha256, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256)):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal SHA-256 digest.")
        raw = Path(path).read_bytes()
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
            raise ValueError("Calibration archive does not match the expected whole-file SHA-256.")
        report = _decode_calibration_report(raw)
        payload = {k: v for k, v in report.items() if k != "checkpoint_sha256"}
        if report["checkpoint_sha256"] != _json_digest(payload):
            raise ValueError("Calibration checkpoint checksum mismatch.")
        identity = report["identity"]
        if (not isinstance(identity, dict) or type(identity.get("identity_schema")) is not int
                or identity["identity_schema"] != 2 or report["fingerprint"] != _json_digest(identity)):
            raise ValueError("Invalid embedded calibration identity or fingerprint.")
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("Malformed calibration report.") from error
    _check_completed_cv_report(report, identity)
    return report
