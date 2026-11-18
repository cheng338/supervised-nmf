"""Isolated native-format initialization check used by metadata contract tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import sys
import threading
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("--import-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--policy", choices=("raise", "warn", "ignore"), required=True)
parser.add_argument("--warning-policy", choices=("error", "always"), required=True)
parser.add_argument("--concurrent", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.import_root))
import numpy as np
from supervised_nmf import validation

assert Path(validation.__file__).resolve() == (args.import_root / "supervised_nmf/validation.py").resolve()
assert validation._NATIVE_KEY_FORMAT is None
values = np.asarray([np.longdouble("0.5"), np.longdouble("-0"), np.longdouble("7.25")], dtype=object)
expected = [Fraction(*value.as_integer_ratio()) for value in values]
original_finfo = np.finfo
original_cache = original_finfo._finfo_cache
before_error, before_filters = np.geterr(), list(warnings.filters)
calls = []
barrier = threading.Barrier(2) if args.concurrent else None


def observed_finfo(dtype):
    assert dtype is np.longdouble and validation._NATIVE_KEY_FORMAT is None
    calls.append(threading.get_ident())
    if barrier is not None:
        barrier.wait(timeout=15)
    return original_finfo(dtype)


def keys_once():
    before = np.geterr()
    with np.errstate(all=args.policy):
        active = np.geterr()
        keys = validation._metadata_keys(values)
        assert np.geterr() == active and list(keys) == expected
        assert [hash(value) for value in keys] == [hash(value) for value in expected]
        cache = validation._NATIVE_KEY_FORMAT
        assert cache is not None
        assert list(validation._metadata_keys(values)) == list(keys)
        assert np.geterr() == active
    assert np.geterr() == before
    return cache, keys.tolist()


try:
    # Keep NumPy cache manipulation inside this child, restoring its identity.
    original_finfo._finfo_cache = {}
    np.finfo = observed_finfo
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(args.warning_policy)
        if args.concurrent:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(keys_once) for _ in range(2)]
                results = [future.result(timeout=30) for future in futures]
            assert len(set(calls)) == len(calls) == 2
            assert results[0] == results[1]
            assert results[0][0] == validation._NATIVE_KEY_FORMAT
        else:
            keys_once()
            assert len(calls) == 1
        assert not caught, [(type(row.message).__name__, str(row.message)) for row in caught]
finally:
    np.finfo = original_finfo
    original_finfo._finfo_cache = original_cache
assert np.finfo is original_finfo and original_finfo._finfo_cache is original_cache
assert np.geterr() == before_error and warnings.filters == before_filters
with args.output.open("x", encoding="utf-8") as stream:
    json.dump({"status": "passed", "concurrent": args.concurrent, "finfo_calls": len(calls),
               "numpy_cache_identity_restored": True, "error_and_warning_state_preserved": True}, stream)
