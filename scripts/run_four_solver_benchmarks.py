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
import stat
import subprocess
import sys
import time
import uuid

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
    """Stage a complete JSON file and atomically replace the target."""
    path = Path(path)
    payload = json.dumps(value, indent=2) + "\n"
    temporary = None
    try:
        for _ in range(10):
            candidate = path.parent / (".f" + uuid.uuid4().hex[:12])
            try:
                stream = candidate.open("x", encoding="utf-8")
            except FileExistsError:
                continue
            temporary = candidate
            break
        else:
            raise FileExistsError(f"Could not create a unique temporary file for {path}")
        with stream:
            stream.write(payload)
        for attempt in range(6):
            try:
                temporary.replace(path)
                temporary = None
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * 2 ** attempt)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


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


def _protocol_key(selected):
    """Return the checkout-relative key for this physical protocol file."""
    path = Path(selected).resolve(strict=True)
    relative = path.relative_to(ROOT)
    # Windows path comparisons can fold case even inside a case-sensitive directory.
    checkout = Path(*path.parts[:len(ROOT.parts)])
    if not checkout.samefile(ROOT) or not path.samefile(ROOT / relative):
        raise ValueError(f"Selected protocol {path} does not belong to checkout {ROOT} at {relative}")
    return relative.as_posix()


def _check_archive_root_spelling(archive):
    """Check the stored name when the filesystem resolves the archive entry."""
    if os.path.lexists(archive) and archive.name not in os.listdir(archive.parent):
        raise FileExistsError(f"Archive directory {archive.parent} does not contain the exact name {archive.name!r}")


def _archive_children(archive, keys):
    """List the required child names for each directory using literal path spellings."""
    children = {str(archive.parent): {archive.name}}
    for key in keys:
        parent = archive
        for name in key.split("/"):
            children.setdefault(str(parent), set()).add(name)
            parent = parent / name
    return children


def _archive_files(files, archive):
    """Copy each selected name to a separate file without writing through an alias."""
    sources = {path.relative_to(ROOT).as_posix(): path for path in files}
    live_files = {}
    for source in sources.values():
        info = source.stat()
        live_files[info.st_dev, info.st_ino] = source

    existing_files = {}
    missing = []
    existing = []
    for key, source in sources.items():
        destination = archive / key
        try:
            info = destination.stat()
        except FileNotFoundError:
            missing.append((key, source, destination))
            continue
        file_id = (info.st_dev, info.st_ino)
        if file_id in live_files:
            raise shutil.SameFileError(f"{destination} is a selected source file ({live_files[file_id]})")
        if file_id in existing_files:
            raise FileExistsError(f"Archive destinations {existing_files[file_id]} and {destination} share a file")
        if not stat.S_ISREG(info.st_mode) or destination.is_symlink() or info.st_nlink > 1:
            raise FileExistsError(f"Archive destination {destination} must be an independent regular file")
        existing_files[file_id] = destination
        existing.append((key, source, destination))

    directories = _archive_children(archive, sources)
    for parent in directories:
        try:
            info = os.lstat(parent)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
                os.name == "nt" and info.st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT):
            raise FileExistsError(f"Archive directory {parent} redirects to another directory")

    # The destination filesystem decides which names can coexist. Create all new
    # files exclusively before updating any previous, unpublished partial copies.
    identity = dict.fromkeys(sources)

    def copy_entries(entries, mode):
        for key, source, destination in entries:
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_bytes()
            with destination.open(mode) as stream:
                stream.write(content)
            shutil.copystat(source, destination)
            identity[key] = hashlib.sha256(content).hexdigest()

    copy_entries(missing, "xb")
    for parent, names in directories.items():
        absent = names.difference(os.listdir(parent))
        if absent:
            raise FileExistsError(f"Archive directory {parent} does not contain the exact names {sorted(absent)!r}")
    copy_entries(existing, "wb")
    return identity


def freeze_run(output, fashion_protocol=DEFAULT_FASHION_PROTOCOL, celeba_protocol=DEFAULT_CELEBA_PROTOCOL):
    """Freeze exact selected protocols and sources without starting any fitting."""
    output = Path(output).resolve()
    suppress_crash_dialogs()
    output.mkdir(parents=True, exist_ok=True)
    protocol_paths = {"fashion": _protocol_key(fashion_protocol), "celeba": _protocol_key(celeba_protocol)}
    files = list((ROOT / "src").rglob("*.py")) + [
        Path(__file__).resolve(), ROOT / "scripts" / "fashion_mnist_study.py",
        ROOT / "scripts" / "celeba_four_solver_study.py",
        ROOT / "scripts" / "experimental_nmf.py",
        ROOT / "scripts" / "report_four_solver_benchmarks.py",
        ROOT / "scripts" / "_audit_identity.py",
        ROOT / "scripts" / "profile_celeba_stages.py",
        ROOT / "scripts" / "prepare_fashion_mnist.py",
        ROOT / "scripts" / "check_four_solver_training.py",
        ROOT / protocol_paths["fashion"], ROOT / protocol_paths["celeba"],
        ROOT / "research" / "fashion-mnist-protocol.md",
        ROOT / "research" / "nmf-solver-comparison.md",
        ROOT / "research" / "numerical-repair-20261004.md",
    ]
    archive = output / "source"
    freeze_path = output / "source-freeze.json"
    if freeze_path.exists():
        identity = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (freeze["files"] != identity or freeze["python_executable"] != sys.executable
                or freeze.get("protocol_paths") != protocol_paths):
            raise RuntimeError("Frozen source/environment changed; preserve this run and use a new output directory.")
        _check_archive_root_spelling(archive)
    else:
        _check_archive_root_spelling(archive)
        created = now()
        identity = _archive_files(files, archive)
        freeze = {"author": "Chang Cheng", "created_utc": created, "files": identity,
                  "protocol_paths": protocol_paths,
                  "error_dialog_policy": ERROR_DIALOG_POLICY,
                  "python_executable": sys.executable, "official_test_loaded": False,
                  "historical_celeba_outcomes_known": True,
                  "fashion_development_validation_outcomes_seen_before_freeze": False}
        write_json(freeze_path, freeze)
    return freeze


def run(output, fashion_protocol=DEFAULT_FASHION_PROTOCOL, celeba_protocol=DEFAULT_CELEBA_PROTOCOL):
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
