"""The public package and SciPy reduction remain usable if optional MU checks decline."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import warnings

if __name__ == "__main__":
    # Retain the dependency builds and package source selected by the parent.
    sys.path[:] = json.loads(sys.argv[2])

import pytest


@pytest.mark.parametrize("case", ["source_none", "missing_at_import", "none_at_import",
                                  "removed_after_import", "np.iterable", "np.asanyarray",
                                  "np.array", "xp.asarray", "xp.result_type", "xp.atleast_1d",
                                  "nonregular_logsumexp", "nonregular_at"])
def test_public_import_and_fallback_with_incomplete_optional_scipy(case):
    from supervised_nmf import _mu

    if not _mu._owned_log_sum_compatible():
        pytest.skip("This regression needs an otherwise qualified SciPy implementation")
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), case, json.dumps(sys.path)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (result.stdout, result.stderr)


def check_in_fresh_process(case):
    import numpy as np
    from scipy.special import logsumexp
    from sklearn.decomposition import NMF  # Finish sklearn initialization before the mutation.

    implementation = importlib.import_module("scipy.special._logsumexp")
    xp = importlib.import_module("scipy._external.array_api_compat.numpy")
    native_min = xp.min
    assert "supervised_nmf" not in sys.modules
    if "." in case:
        owner, name = case.split(".")
        check_array_binding(np if owner == "np" else xp, name, np, logsumexp)
        return
    if case in ("nonregular_logsumexp", "nonregular_at"):
        module = (implementation if case == "nonregular_logsumexp" else
                  importlib.import_module("scipy._external.array_api_extra._lib._at"))
        check_nonregular_source(module, np, logsumexp)
        return
    if case == "source_none":
        implementation.__file__ = None
    elif case == "missing_at_import":
        del xp.min
    elif case == "none_at_import":
        xp.min = None

    from supervised_nmf import BinaryScreening, FeatureSPA, NMFClassifier, _mu
    assert all(isinstance(cls, type) for cls in (BinaryScreening, FeatureSPA, NMFClassifier, NMF))
    if case == "removed_after_import":
        assert _mu._owned_log_sum_compatible()
        del xp.min
    assert not _mu._owned_log_sum_compatible()

    products = np.array([[[-np.inf, -745., 0., 0.], [-2., -2., -1., -8.]]])
    before = products.tobytes()
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="raise"):
        warnings.simplefilter("always")
        # The reference MU reduction ignores underflow in negligible summands.
        with np.errstate(under="ignore"):
            expected = logsumexp(products, axis=2)
        actual = _mu._owned_log_sum(products)
    assert not caught
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert not np.shares_memory(actual, products) and products.tobytes() == before
    if case == "removed_after_import":
        xp.min = native_min
        assert _mu._owned_log_sum_compatible()
        assert _mu._owned_log_sum(products).tobytes() == expected.tobytes()
        assert products.tobytes() != before


def check_nonregular_source(module, np, logsumexp):
    products = np.array([[[-1., -2., -3.]]])
    before = products.tobytes()
    expected = logsumexp(products, axis=2)
    read_bytes = Path.read_bytes
    with TemporaryDirectory() as folder:
        path = (Path(folder) / 'optional_source.py').resolve()
        if os.name == 'nt':
            path.mkdir()
        else:
            os.mkfifo(path)

        def read_regular(other):
            assert other != path, 'optional qualification attempted to read a nonregular file'
            return read_bytes(other)

        with patch.object(module, '__file__', str(path)), patch.object(Path, 'read_bytes', read_regular):
            from supervised_nmf import _mu
            assert not _mu._owned_log_sum_compatible()
            assert _mu._owned_log_sum(products).tobytes() == expected.tobytes()
            assert products.tobytes() == before


def check_array_binding(owner, name, np, logsumexp):
    # Allow other dependencies to finish importing before arming the observer.
    native = getattr(owner, name)
    marker = RuntimeError(name)
    armed = False
    def observed(*args, **kwargs):
        if armed:
            raise marker
        return native(*args, **kwargs)

    products = np.array([[[-1., -2., -3.]]])
    before = products.tobytes()
    expected = logsumexp(products, axis=2)
    def check_fallback(reduce, compatible):
        assert not compatible()
        for operation in (lambda: logsumexp(products, axis=2), lambda: reduce(products)):
            with pytest.raises(RuntimeError) as caught:
                operation()
            assert caught.value is marker
            assert products.tobytes() == before

    with patch.object(owner, name, observed):
        from supervised_nmf import _mu
        armed = True
        check_fallback(_mu._owned_log_sum, _mu._owned_log_sum_compatible)

    reduce, compatible = _mu._make_owned_log_sum()
    assert compatible()
    with patch.object(owner, name, observed):
        check_fallback(reduce, compatible)
    assert compatible()
    assert reduce(products).tobytes() == expected.tobytes()
    assert products.tobytes() != before


if __name__ == "__main__":
    check_in_fresh_process(sys.argv[1])
