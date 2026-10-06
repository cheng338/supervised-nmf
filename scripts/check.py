"""Run local tests without installing the package or third-party test runners."""

from pathlib import Path
import os
import sys
import unittest

def main():
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
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return not result.wasSuccessful()


if __name__ == "__main__":
    # Spawned Windows workers import this entry point before receiving a job.
    # Test discovery/execution must stay in the original process.
    raise SystemExit(main())

