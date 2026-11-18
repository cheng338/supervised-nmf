"""Read-only audit comparisons for relocated structural-identity receipts.

This is not a checkpoint-resume policy. Reuse and resume continue to require
the package's exact identities. An exported receipt may be audited on another
machine when implementation names/hashes and every typed parameter agree.
Only implementation-file locations are allowed to differ.
"""

from collections import Counter
from pathlib import PurePosixPath, PureWindowsPath
import re


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_QUALIFIED = re.compile(r"[A-Za-z0-9_.<>]+\Z")


def _source_path(value):
    return (isinstance(value, str) and bool(value)
            and not any(character in value for character in ("\x00", "\n", "\r"))
            and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()))


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _qualified(value):
    return (isinstance(value, str) and "." in value
            and _QUALIFIED.fullmatch(value) is not None)


def _implementation_container(record):
    # Constructor dictionaries are encoded as kind=dict/items and cannot match
    # these implementation records, even if callers use path/sha256/source keys.
    return ((set(record) == {"type", "source"} and _qualified(record["type"]))
            or (record.get("kind") in {"function", "native_callable"}
                and _qualified(record.get("name"))))


def _same(actual, expected, *, implementation_source=False):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            return False
        if implementation_source and set(expected) == {"path", "sha256"}:
            return (_source_path(actual["path"]) and _source_path(expected["path"])
                    and _hash(actual["sha256"]) and _hash(expected["sha256"])
                    and actual["sha256"] == expected["sha256"])
        owner = _implementation_container(expected)
        return all(_same(actual[key], value,
                         implementation_source=owner and key == "source")
                   for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same(a, e) for a, e in zip(actual, expected))
    return actual == expected


def verify_configuration_identity(actual, expected):
    """Require the same typed configuration, allowing implementation relocation."""
    if not _same(actual, expected):
        raise ValueError("Read-only audit structural identity differs in type, code hash, or constructor state.")
    return True


def _source_inventory(value):
    if not isinstance(value, dict):
        raise ValueError("Read-only audit implementation source inventory must be a mapping.")
    inventory = Counter()
    for key, digest in value.items():
        if not isinstance(key, str):
            raise ValueError("Read-only audit implementation source key is invalid.")
        name, separator, path = key.partition("@")
        if not separator or not _qualified(name) or not _source_path(path) or not _hash(digest):
            raise ValueError("Read-only audit implementation source name, path, or hash is invalid.")
        # Preserve multiplicity: relocation cannot silently merge separate
        # implementations sharing a qualified name and source digest.
        inventory[name, digest] += 1
    return inventory


def verify_implementation_sources(actual, expected):
    """Require identical qualified source names/hashes, permitting new locations."""
    if _source_inventory(actual) != _source_inventory(expected):
        raise ValueError("Read-only audit implementation source names or hashes differ.")
    return True


def verify_shared_knn_cv_receipt(report):
    """Check current complete shared-KNN CV receipts for the frozen BA readers.

    This contract belongs to calibration_knn_grid's schema-4 reports, not
    general calibration_search reports or historical receipt formats. Callers
    still verify identities, fold qualification, means, rankings and sources.
    """
    identity = report["identity"]
    calibration = identity["calibration"]
    for record, field, expected in ((report, "schema_version", 4),
                                    (identity, "schema_version", 4),
                                    (calibration, "identity_schema", 2)):
        if type(record.get(field)) is not int or record[field] != expected:
            raise ValueError("Unsupported structural CV identity schema.")
    if report.get("state") != "cv_completed":
        raise ValueError("Current shared-KNN CV receipt is not complete.")
    if report.get("refit_performed") is not False:
        raise ValueError("Current shared-KNN CV receipt must not contain a final refit.")
    if any(record.get("scoring") != "balanced_accuracy"
           for record in (report, identity, calibration)):
        raise ValueError("CV scoring differs from the declared balanced-accuracy objective.")

    def aliases(record, generic, named, *, required=True, nullable=False):
        present = (generic in record, named in record)
        if not required and present == (False, False):
            return
        if not all(present) or record[generic] != record[named]:
            raise ValueError(f"Current shared-KNN CV score aliases differ or are missing: {generic}/{named}.")
        for key in (generic, named):
            value = record[key]
            if nullable and value is None:
                continue
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"Current shared-KNN CV score is not a finite number in [0, 1]: {key}.")

    for candidate in report["candidates"]:
        if type(candidate.get("valid")) is not bool:
            raise ValueError("Current shared-KNN CV candidate qualification must be boolean.")
        aliases(candidate, "mean_score", "mean_balanced_accuracy", nullable=not candidate["valid"])
        if not candidate["valid"] and candidate["mean_score"] is not None:
            raise ValueError("Invalid shared-KNN CV candidate must have None mean scores.")
        for fold in candidate["folds"]:
            # Failed folds may have no scores; nonconverged scored folds can
            # retain both. Neither case permits contradictory partial aliases.
            valid = fold["status"] == "valid"
            aliases(fold, "score", "balanced_accuracy", required=valid, nullable=not valid)
    selected_mean = max((row["mean_balanced_accuracy"] for row in report["candidates"]
                         if row["valid"]), default=None)
    aliases(report, "selected_mean_score", "selected_mean_balanced_accuracy", nullable=selected_mean is None)
    if report["selected_mean_score"] != selected_mean:
        raise ValueError("Current shared-KNN CV selected score differs from the qualified local best.")
    return True
