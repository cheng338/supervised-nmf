"""Print the active numerical environment without changing it."""

import importlib.metadata
import json
import os
import platform
import sys
from threadpoolctl import threadpool_info
import numpy  # Load BLAS before querying thread pools.
import scipy
import sklearn

print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "versions": {name: importlib.metadata.version(name) for name in
                 ["numpy", "scipy", "scikit-learn", "threadpoolctl"]},
    "thread_pools": threadpool_info(),
}, indent=2))
