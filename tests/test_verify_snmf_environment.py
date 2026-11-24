import importlib.metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from supervised_nmf import __version__


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_snmf_environment.py"


def test_verifier_creates_nested_receipt_and_preserves_it(tmp_path, monkeypatch, capsys):
    output = tmp_path / "new" / "nested" / "receipt.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])
    installed_version = importlib.metadata.version
    real_run = subprocess.run

    def version(name):
        # A source checkout need not have wheel metadata.
        return __version__ if name == "supervised-nmf" else installed_version(name)

    def run(args, *positional, **kwargs):
        if args == [sys.executable, "-m", "pip", "check"]:
            return subprocess.CompletedProcess(args, 0, stdout="")
        return real_run(args, *positional, **kwargs)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(subprocess, "run", run)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    original = output.read_bytes()
    receipt = json.loads(original)
    checks = receipt["checks"]
    assert len(checks) == 6
    assert {(row["solver"], row["classifier"]) for row in checks} == {
        (solver, head) for solver in ("spa", "cd", "mu") for head in ("knn", "logistic")
    }
    assert all(row["passed"] and row["serialization_exact"] for row in checks), checks
    assert receipt["passed"] is True
    summary = json.loads(capsys.readouterr().out)
    assert summary["passed"] is True and summary["checks"] == len(checks)
    assert summary["output"] == str(output) and summary["failed_checks"] == []
    assert list(output.parent.iterdir()) == [output]

    with pytest.raises(FileExistsError):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert output.read_bytes() == original
