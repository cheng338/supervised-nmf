"""Study helpers retain one checkout owner and reject stale loaded code."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).with_name("study_helper_import_probe.py")
CASES = [
    "recovery-canonical", "recovery-terminal",
    *[f"recovery-conflict-{state}" for state in ("foreign", "missing_origin", "missing_pin")],
    "recovery-admission-recovery_store", "recovery-admission-numeric_codec",
    *[f"codec-{operation}-{state}" for operation in ("commit", "restore")
      for state in ("same", "foreign", "missing_pin", "stale")],
    "codec-commit-empty", "codec-restore-large", "codec-restore-removed",
    *[f"profile-{consumer}-{state}" for consumer in ("default", "fashion")
      for state in ("same", "foreign", "stale")],
    "profile-default-missing_pin", "profile-fashion-missing_origin",
    "experimental-routes-same", "experimental-routes-foreign",
    *[f"experimental-admission-{state}" for state in ("same", "foreign", "missing_pin", "stale")],
    "origins-codec", "origins-experimental",
]


@pytest.mark.parametrize("case", CASES)
def test_study_helper_import_boundary(case, tmp_path_factory):
    # Short owned roots also keep Windows recovery payload names bounded.
    work = tmp_path_factory.mktemp("hi")
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(PROBE), str(ROOT), str(work), case],
        env=environment, capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    assert result.returncode == 0, (case, result.stdout, result.stderr)
    assert result.stdout.strip() == "completed: " + case
