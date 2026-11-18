"""Bound only native-thread environment changes made during test imports."""
from contextlib import contextmanager
import os


THREAD_ENVIRONMENT = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                      'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                      'OMP_THREAD_LIMIT', 'MKL_DYNAMIC')


@contextmanager
def preserve_thread_environment():
    before = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT}
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
