"""Private guarded reuse of completed Linux study checkpoints."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
import json
import math
import sys
import warnings


class _ReuseUnavailable(Exception):
    """An explicit scheduling qualification failed; ordinary spawn remains authoritative."""


def _pool_key(rows):
    if not rows or any(not isinstance(row.get("filepath"), str) for row in rows):
        raise _ReuseUnavailable("native_inventory_unavailable")
    if len({row["filepath"] for row in rows}) != len(rows):
        raise _ReuseUnavailable("duplicate_native_inventory")
    return sorted(rows, key=lambda row: row["filepath"])


class _ReadOnlyEnvironment:
    def __init__(self, expected, *, provenance, numpy, sklearn):
        self.expected = _pool_key(expected["threadpools"])
        if any(row["num_threads"] != 1 for row in self.expected):
            raise _ReuseUnavailable("coordinator_native_count")
        self.provenance = provenance
        self.numpy = numpy
        self.sklearn = sklearn
        self.error_state = numpy.geterr()
        self.sklearn_config = sklearn.get_config()

    def __call__(self, threads):
        if type(threads) is not int or threads != 1:
            raise _ReuseUnavailable("native_threads_not_one")
        if self.numpy.geterr() != self.error_state or self.sklearn.get_config() != self.sklearn_config:
            raise _ReuseUnavailable("thread_local_configuration")
        environment = self.provenance(threads)
        if _pool_key(environment["threadpools"]) != self.expected:
            raise _ReuseUnavailable("native_inventory_or_count_changed")
        return environment


def _plain(value):
    if value is None or type(value) in (bool, str, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in (list, tuple):
        return all(_plain(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain(item) for key, item in value.items())
    return False


def _closed_model(estimator):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from .classifier import NMFClassifier
    from .screening import BinaryScreening
    if type(estimator) not in (PCA, KNeighborsClassifier, Pipeline, NMFClassifier, BinaryScreening):
        return False
    parameters = estimator.get_params(deep=False)
    if type(estimator) is Pipeline:
        if parameters["memory"] is not None:
            return False
        steps = parameters.pop("steps")
        if type(steps) not in (list, tuple) or not steps:
            return False
        if not all(type(step) is tuple and len(step) == 2 and type(step[0]) is str and _closed_model(step[1]) for step in steps):
            return False
        if type(steps[-1][1]) is not KNeighborsClassifier:
            return False
    return _plain(parameters)


def _quiet_record(record, validation_rows):
    """Scheduling eligibility only; malformed records remain ordinary-worker inputs."""
    if record.get("status") != "valid":
        return True
    neighbors = record.get("neighbors")
    if type(neighbors) is not dict:
        return False
    for row in neighbors.values():
        if type(row) is not dict:
            return False
        score = row.get("score")
        if type(score) not in (int, float) or not 0 <= score <= 1 or not math.isfinite(score):
            return False
        # The CLI uses unweighted accuracy/balanced accuracy. A positive score
        # cannot be smaller than half of one validation observation's weight.
        # This also keeps report std squaring away from strict underflow.
        if score != 0 and score < 0.5 / validation_rows:
            return False
    return True


def _declared_folds_present(directory):
    """Negative scheduling hint only; locked core validation remains authoritative."""
    envelope = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if type(envelope) is not dict or type(envelope.get("payload")) is not dict:
        return False
    checksum = envelope.get("sha256")
    if (type(checksum) is not str or len(checksum) != 64
            or any(char not in "0123456789abcdef" for char in checksum)):
        return False
    report = envelope["payload"]
    records = report.get("representation_folds")
    if report.get("state") != "cv_completed" or type(records) is not list or not records:
        return False
    keys = set()
    for record in records:
        if type(record) is not dict or record.get("state") != "completed":
            return False
        key = record.get("key")
        if (type(key) is not str or len(key) != 64
                or any(char not in "0123456789abcdef" for char in key) or key in keys):
            return False
        keys.add(key)
    # The report is derived and may be stale or fabricated. Existence can only
    # decline this optimization; it never accepts any checkpoint or identity.
    return all((directory / "folds" / f"{key}.json").is_file() for key in keys)


def _try_completed(items, workers, *, describe, invoke):
    """Reuse in one fresh thread only when effective parallelism equals one."""
    if sys.platform != "linux" or not items:
        return None
    if type(workers) is not int or workers < 1 or workers > 8:
        return None
    if min(workers, len(items)) != 1:
        return None
    try:
        descriptors = [describe(item) for item in items]
        hints = all(threads == 1 and (directory / "identity.json").is_file() and (directory / "report.json").is_file()
                    for _, _, directory, threads in descriptors)
    except Exception:
        return None
    if not hints:
        return None
    from sklearn.model_selection import ParameterGrid
    from ._search_diagnostics import screening_grid_diagnostics
    # This context exists only before any executor/thread is created. The unchanged
    # spawned call emits warnings/errors when prequalification sees a warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            for estimator, grid, _, _ in descriptors:
                if not _closed_model(estimator) or not _plain(grid):
                    return None
                screening_grid_diagnostics(estimator, list(ParameterGrid(grid)))
        except Exception:
            # This was only a hint. Ordinary source/data/grid validation decides
            # the authoritative error and its precedence after full spawn.
            return None
    if caught:
        return None
    try:
        if not all(_declared_folds_present(directory) for _, _, directory, _ in descriptors):
            return None
    except Exception:
        # Defer parse, I/O and shape failures to the original worker path,
        # preserving its data/grid/source/checkpoint error precedence.
        return None
    import numpy as np
    import sklearn
    from threadpoolctl import threadpool_limits
    from .evaluation import _initialized_provenance, provenance
    # Warm before the outer context is created, so it sees probe-loaded libraries.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _initialized_provenance(1)
    if caught:
        return None
    try:
        with threadpool_limits(limits=1):
            environment = _ReadOnlyEnvironment(provenance(1), provenance=provenance, numpy=np, sklearn=sklearn)

            def ordered_reuse():
                # This is one fresh Python participant. Its real native counts
                # and thread-local NumPy/sklearn state qualify before any work.
                environment(1)
                results = []
                for item in items:
                    result = invoke(item, environment)
                    if result is None:
                        raise _ReuseUnavailable("incomplete_or_parent_identity")
                    results.append(result)
                environment(1)
                return results

            # One submitted callable owns the ordered sequence. A miss, internal
            # qualification signal or authoritative error stops that sequence;
            # executor exit joins before restoring limits or spawning fallback.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="snmf-reuse") as pool:
                results = pool.submit(Context().run, ordered_reuse).result()
            environment(1)
        return results
    except _ReuseUnavailable:
        return None
