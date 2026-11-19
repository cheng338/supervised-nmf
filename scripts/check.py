"""Run the source test suite with pytest."""

from pathlib import Path
import os
import sys


def main():
    if sys.platform == "linux":
        # The completed-reuse tests need a fresh worker to see single-threaded native pools.
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "1"
    if os.name == "nt":
        import ctypes
        # Limit this policy to this test process and its children. Native
        # failures still terminate and retain their fault-handler traceback.
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetErrorMode.restype = ctypes.c_uint
        kernel.SetErrorMode.argtypes = [ctypes.c_uint]
        kernel.SetErrorMode.restype = ctypes.c_uint
        kernel.SetErrorMode(kernel.GetErrorMode() | 0x0001 | 0x0002)
    os.environ["PYTHONFAULTHANDLER"] = "1"
    import faulthandler
    faulthandler.enable()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    import pytest
    return pytest.main(["-v", str(root / "tests")])


if __name__ == "__main__":
    # Spawned Windows workers import this entry point before receiving a job.
    # Test discovery/execution must stay in the original process.
    raise SystemExit(main())

