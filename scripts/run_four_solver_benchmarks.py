"""Run Chang Cheng's frozen four-solver comparison without interactive steps.

All training CV on both datasets precedes development validation. The official
test sets are not read. Each phase is resumable through its own checked receipts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FASHION_PROTOCOL = ROOT / "examples/fashion_mnist_solver_protocol_v2.json"
DEFAULT_CELEBA_PROTOCOL = ROOT / "examples/celeba_four_solver_protocol.json"
ERROR_DIALOG_POLICY = {"scope": "this process and inherited children only", "preserve_existing_error_mode": True,
                       "windows_added_error_mode_flags": 3, "python_faulthandler": True,
                       "system_registry_changed": False}


def suppress_crash_dialogs():
    """Keep native crashes in logs, without changing machine-wide settings."""
    os.environ["PYTHONFAULTHANDLER"] = "1"
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetErrorMode.restype = ctypes.c_uint
        kernel32.SetErrorMode.argtypes = [ctypes.c_uint]
        kernel32.SetErrorMode.restype = ctypes.c_uint
        current = kernel32.GetErrorMode()
        kernel32.SetErrorMode(current | 0x0001 | 0x0002)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def output_lock(output):
    """Prevent two complete orchestrations from overlapping on one output."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".orchestrator.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another orchestration already owns this output directory.") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def freeze_run(output, fashion_protocol=DEFAULT_FASHION_PROTOCOL, celeba_protocol=DEFAULT_CELEBA_PROTOCOL):
    """Freeze exact selected protocols and sources without starting any fitting."""
    output = Path(output).resolve()
    suppress_crash_dialogs()
    output.mkdir(parents=True, exist_ok=True)
    fashion_protocol, celeba_protocol = Path(fashion_protocol).resolve(), Path(celeba_protocol).resolve()
    protocol_paths = {"fashion": fashion_protocol.relative_to(ROOT).as_posix(),
                      "celeba": celeba_protocol.relative_to(ROOT).as_posix()}
    files = list((ROOT / "src").rglob("*.py")) + [
        Path(__file__), ROOT / "scripts" / "fashion_mnist_study.py",
        ROOT / "scripts" / "celeba_four_solver_study.py",
        ROOT / "scripts" / "experimental_nmf.py",
        ROOT / "scripts" / "report_four_solver_benchmarks.py",
        ROOT / "scripts" / "profile_celeba_stages.py",
        ROOT / "scripts" / "prepare_fashion_mnist.py",
        ROOT / "scripts" / "check_four_solver_training.py",
        ROOT / "scripts" / "prepare_four_solver_recovery.py",
        fashion_protocol, celeba_protocol,
        ROOT / "docs" / "fashion-mnist-protocol.md",
        ROOT / "docs" / "nmf-solver-comparison.md",
        ROOT / "docs" / "numerical-repair-20261004.md",
    ]
    identity = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in files}
    freeze_path = output / "source-freeze.json"
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (freeze["files"] != identity or freeze["python_executable"] != sys.executable
                or freeze.get("protocol_paths") != protocol_paths):
            raise RuntimeError("Frozen source/environment changed; preserve this run and use a new output directory.")
    else:
        freeze = {"author": "Chang Cheng", "created_utc": now(), "files": identity,
                  "protocol_paths": protocol_paths,
                  "error_dialog_policy": ERROR_DIALOG_POLICY,
                  "python_executable": sys.executable, "official_test_loaded": False,
                  "historical_celeba_outcomes_known": True,
                  "fashion_development_validation_outcomes_seen_before_freeze": False}
        for path in files:
            destination = output / "source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        write_json(freeze_path, freeze)
    return freeze


def run(output, fashion_protocol=DEFAULT_FASHION_PROTOCOL, celeba_protocol=DEFAULT_CELEBA_PROTOCOL):
    if Path(sys.prefix).name != "py12_df_aocl":
        raise RuntimeError("Use the author-requested py12_df_aocl conda environment.")
    output = Path(output).resolve()
    freeze = freeze_run(output, fashion_protocol, celeba_protocol)
    identity = freeze["files"]
    fashion_protocol, celeba_protocol = Path(fashion_protocol).resolve(), Path(celeba_protocol).resolve()
    state_path = output / "status.json"
    state = (json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists()
             else {"author": "Chang Cheng", "created_utc": now(), "status": "running",
                   "official_test_loaded": False, "phase_order": [], "phases": {}})
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    phases = [
        ("fashion-cv", "fashion_mnist_study.py", fashion_protocol, "fashion", "cv"),
        ("celeba-cv", "celeba_four_solver_study.py", celeba_protocol, "celeba", "cv"),
        ("fashion-evaluate", "fashion_mnist_study.py", fashion_protocol, "fashion", "evaluate"),
        ("celeba-evaluate", "celeba_four_solver_study.py", celeba_protocol, "celeba", "evaluate"),
        ("report", "report_four_solver_benchmarks.py", None, None, None),
    ]
    state["phase_order"] = [phase[0] for phase in phases]
    state["error_dialog_policy"] = ERROR_DIALOG_POLICY
    state["status"] = "running"
    write_json(state_path, state)
    for name, script, protocol, subdirectory, mode in phases:
        if state["phases"].get(name, {}).get("status") == "complete":
            continue
        if any(sha256(ROOT / path) != expected for path, expected in identity.items()):
            raise RuntimeError("Source changed during the experiment; remaining phases are held.")
        command = ([sys.executable, str(ROOT / "scripts" / script), "--run", str(output)]
                   if name == "report" else
                   [sys.executable, str(ROOT / "scripts" / script), "--protocol",
                    str(protocol), "--output", str(output / subdirectory),
                    "--mode", mode])
        if mode == "cv":
            command += ["--workers", "8"]
        attempt = len(state["phases"].get(name, {}).get("attempts", [])) + 1
        log_path = output / f"{name}-attempt{attempt}.log"
        previous = state["phases"].get(name, {})
        attempts = previous.get("attempts", [])
        record = {"started_utc": now(), "command": command, "log": str(log_path), "status": "running"}
        attempts.append(record)
        state["phases"][name] = {"status": "running", "attempts": attempts}
        state["current_phase"] = name
        write_json(state_path, state)
        print(json.dumps({"phase": name, "status": "started", "log": str(log_path)}), flush=True)
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, check=False)
        record.update(completed_utc=now(), wall_seconds=time.perf_counter() - start,
                      exit_code=process.returncode,
                      status="complete" if process.returncode == 0 else "failed")
        state["phases"][name]["status"] = record["status"]
        if process.returncode:
            state["status"] = "failed"
            write_json(state_path, state)
            raise RuntimeError(f"{name} failed; inspect {log_path}. All records are preserved.")
        write_json(state_path, state)
        print(json.dumps({"phase": name, "status": "complete", "wall_seconds": record["wall_seconds"]}), flush=True)
    state.update(status="complete", completed_utc=now(), current_phase=None,
                 decision_status="Development evidence and automatic decision report complete; review numerical limitations and public API integration before changing defaults.")
    write_json(state_path, state)
    print(json.dumps({"status": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "four-solvers-20261003")
    parser.add_argument("--fashion-protocol", type=Path, default=DEFAULT_FASHION_PROTOCOL)
    parser.add_argument("--celeba-protocol", type=Path, default=DEFAULT_CELEBA_PROTOCOL)
    args = parser.parse_args()
    with output_lock(args.output):
        run(args.output, args.fashion_protocol, args.celeba_protocol)
