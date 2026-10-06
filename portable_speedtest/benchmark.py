"""Portable CPU/RAM throughput benchmark for Chang Cheng's supervised NMF.

The parent uses only the standard library (and optional psutil). Every numerical
worker is a Windows-safe spawned process with explicit native thread limits.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import re
import struct
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE",
}


def thread_environment(native_threads):
    return {name: str(native_threads) if value == "1" else value for name, value in THREAD_ENVIRONMENT.items()}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_bundle():
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest.json must contain a nonempty files mapping.")
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path == manifest_path or not path.is_file():
            raise ValueError(f"Invalid manifest path: {name}")
        if not isinstance(expected, str) or digest(path) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}. Rebuild the bundle before running.")
    required = {"benchmark.py", "compare.py", "sweep.py", "check_compatibility.py", "protocol.json",
                "data/workload.npz", "package/supervised_nmf/__init__.py"}
    if not required <= set(files):
        raise ValueError("Manifest does not bind required benchmark, protocol, data, and package files.")
    # An unmanifested Python module could shadow a checked package module.
    for path in ROOT.glob("*.py"):
        if path.name not in files:
            raise ValueError(f"Unmanifested Python source: {path.name}")
    for folder in (ROOT / "package",):
        for path in folder.rglob("*.py"):
            if path.relative_to(ROOT).as_posix() not in files:
                raise ValueError(f"Unmanifested Python source: {path.relative_to(ROOT)}")
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8-sig"))
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported protocol schema.")
    model = protocol.get("model", {})
    if model.get("solver") != "cd" or model.get("screening") != "absolute":
        raise ValueError("Protocol must explicitly request CD with absolute screening.")
    required_model = {"solver", "screening", "n_components", "score_threshold", "n_neighbors", "max_iter",
                      "tol", "random_state", "training_coordinates"}
    if not required_model <= set(model):
        raise ValueError("Protocol must explicitly define all model/numerical parameters.")
    return protocol, digest(manifest_path), digest(ROOT / "protocol.json")


def array_headers(path):
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for field in ("X_train", "y_train", "X_test", "y_test"):
            with archive.open(field + ".npy") as stream:
                if stream.read(6) != b"\x93NUMPY":
                    raise ValueError("Invalid NPY data header.")
                version = tuple(stream.read(2))
                size_format = "<H" if version == (1, 0) else "<I"
                length = struct.unpack(size_format, stream.read(struct.calcsize(size_format)))[0]
                if length > 65536:
                    raise ValueError("NPY header exceeds supported size.")
                header = ast.literal_eval(stream.read(length).decode("latin1"))
                headers[field] = {"shape": list(header["shape"]), "dtype": header["descr"]}
    if len(headers["X_train"]["shape"]) != 2 or len(headers["X_test"]["shape"]) != 2:
        raise ValueError("Workload matrices must be two dimensional.")
    return headers


def memory_info():
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {"total_bytes": int(memory.total), "available_bytes": int(memory.available), "source": "psutil"}
    except ImportError:
        if sys.platform.startswith("linux"):
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, value = line.split(":", 1)
                values[name] = int(value.strip().split()[0]) * 1024
            return {"total_bytes": values["MemTotal"], "available_bytes": values.get("MemAvailable", values["MemFree"]),
                    "source": "/proc/meminfo"}
        if os.name == "nt":
            import ctypes
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]
            status = Status()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return {"total_bytes": int(status.total_phys), "available_bytes": int(status.avail_phys),
                        "source": "GlobalMemoryStatusEx"}
        return {"total_bytes": None, "available_bytes": None, "source": "unavailable"}


def hardware():
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "machine": platform.machine(), "processor": platform.processor(),
              "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"), "logical_cpus": len(affinity) if affinity else os.cpu_count(),
              "host_logical_cpus": os.cpu_count(), "affinity": affinity,
              "physical_cores": None, "memory": memory_info(), "psutil_available": False}
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                result["processor"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
                result["affinity"] = [index for index in range(ctypes.sizeof(ctypes.c_size_t) * 8) if process_mask.value & (1 << index)]
                result["logical_cpus"] = len(result["affinity"]) or result["logical_cpus"]
            length = wintypes.DWORD(0)
            kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
            if length.value:
                buffer = ctypes.create_string_buffer(length.value)
                if kernel.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(length)):
                    offset, cores = 0, 0
                    while offset + 8 <= length.value:
                        relationship, size = struct.unpack_from("<II", buffer.raw, offset)
                        if size < 8 or offset + size > length.value:
                            break
                        cores += relationship == 0
                        offset += size
                    result["physical_cores"] = cores or None
        except (OSError, AttributeError):
            pass
    try:
        import psutil
        result.update(physical_cores=psutil.cpu_count(logical=False), psutil_available=True,
                      affinity=psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else None)
        if result["affinity"]:
            result["logical_cpus"] = len(result["affinity"])
    except ImportError:
        if sys.platform.startswith("linux"):
            blocks = Path("/proc/cpuinfo").read_text().strip().split("\n\n")
            pairs = set()
            for block in blocks:
                fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
                fields = {key.strip(): value.strip() for key, value in fields.items()}
                if "physical id" in fields and "core id" in fields:
                    pairs.add((fields["physical id"], fields["core id"]))
                result["processor"] = fields.get("model name", result["processor"])
            result["physical_cores"] = len(pairs) or None
    return result


def system_cpu_ticks():
    if sys.platform.startswith("linux"):
        values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        total = sum(values)
        idle = values[3] + values[4]
        return total - idle, total
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            unpack = lambda value: (value.dwHighDateTime << 32) | value.dwLowDateTime
            total = unpack(kernel) + unpack(user)
            return total - unpack(idle), total
    return None


def process_rss(pid):
    if sys.platform.startswith("linux"):
        try:
            return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError):
            return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                        ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                        ("quota_peak_paged", ctypes.c_size_t), ("quota_paged", ctypes.c_size_t),
                        ("quota_peak_nonpaged", ctypes.c_size_t), ("quota_nonpaged", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
        open_process = ctypes.windll.kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        close_handle = ctypes.windll.kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        memory = ctypes.windll.psapi.GetProcessMemoryInfo
        memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        handle = open_process(0x0400 | 0x0010, False, pid)
        if handle:
            try:
                counters = Counters()
                counters.cb = ctypes.sizeof(counters)
                if memory(handle, ctypes.byref(counters), counters.cb):
                    return int(counters.working_set)
            finally:
                close_handle(handle)
    return None


def native_pools_valid(pools, expected=1):
    return bool(pools) and all(isinstance(pool.get("num_threads"), int) and pool["num_threads"] == expected for pool in pools)


def _array_hash(array):
    value = hashlib.sha256()
    value.update(str(array.shape).encode("ascii"))
    value.update(array.dtype.str.encode("ascii"))
    value.update(array.tobytes(order="C"))
    return value.hexdigest()


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _worker(worker_id, repeat_id, configuration, model_params, start_event, shared_start,
            cancel_event, events, output_path):
    """Child entry point: set native limits before importing NumPy or sklearn."""
    native_threads = configuration["native_threads_per_worker"]
    os.environ.update(configuration["thread_environment"])
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "package"))
    try:
        import importlib.metadata
        import warnings
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning
        from threadpoolctl import threadpool_info, threadpool_limits
        import supervised_nmf
        from supervised_nmf import NMFClassifier

        if not Path(supervised_nmf.__file__).resolve().is_relative_to(ROOT / "package"):
            raise RuntimeError("Bundled supervised_nmf source did not win import resolution.")
        with threadpool_limits(limits=native_threads):
            pools = threadpool_info()
            if not native_pools_valid(pools, native_threads):
                raise RuntimeError(f"Cannot verify {native_threads} native threads per pool: {pools}")
            X_train, y_train, X_test, y_test = _load_worker_data(np, configuration, worker_id)
            if (X_train.dtype != np.float32 or X_test.dtype != np.float32 or not np.isfinite(X_train).all()
                    or not np.isfinite(X_test).all() or (X_train < 0).any() or (X_test < 0).any()):
                raise ValueError("Workload requires finite nonnegative float32 matrices.")
            data_identity = {"X_train": _array_hash(X_train), "y_train": _array_hash(y_train),
                             "X_test": _array_hash(X_test), "y_test": _array_hash(y_test)}
            reference = None

            def one_fit(task_id, window_start=None):
                nonlocal reference
                model = NMFClassifier(**model_params)
                started = time.perf_counter()
                fit_finished = prediction_finished = None
                failure = None
                labels = probabilities = None
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        model.fit(X_train, y_train)
                        fit_finished = time.perf_counter()
                        labels, probabilities = model.predict_with_probabilities(X_test)
                        prediction_finished = time.perf_counter()
                    except Exception as exc:
                        failure = {"type": type(exc).__name__, "message": str(exc)}
                # This is the completion timestamp; no quality checks, hashing,
                # serialization, disk writes, or queue operations precede it.
                completed = prediction_finished if prediction_finished is not None else time.perf_counter()
                convergence_warnings = [str(item.message) for item in caught
                                        if issubclass(item.category, ConvergenceWarning)]
                record = {"repeat": repeat_id, "worker": worker_id, "task": task_id,
                          "model_seed": model_params["random_state"], "status": "failed",
                          "started_offset_seconds": started - window_start if window_start is not None else None,
                          "completed_offset_seconds": completed - window_start if window_start is not None else None,
                          "started_before_deadline": window_start is not None and started < window_start + configuration["duration_seconds"],
                          "completed_within_window": window_start is not None and completed <= window_start + configuration["duration_seconds"],
                          "timing": {"fit_seconds": fit_finished - started if fit_finished is not None else None,
                                     "prediction_seconds": prediction_finished - fit_finished if prediction_finished is not None else None,
                                     "fit_plus_prediction_seconds": completed - started},
                          "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in caught],
                          "error": failure, "quality": None}
                if failure is None:
                    finite_outputs = bool(np.isfinite(probabilities).all() and np.isfinite(model.reconstruction_err_)
                                          and np.isfinite(model.training_coordinates_).all()
                                          and np.isfinite(model.selected_components_).all())
                    probability_valid = bool(probabilities.shape == (len(y_test), 2)
                                             and np.all(probabilities >= 0) and np.all(probabilities <= 1)
                                             and np.allclose(probabilities.sum(axis=1), 1, atol=1e-7, rtol=0))
                    converged = bool(model.converged_ and model.last_transform_converged_ and not convergence_warnings)
                    quality = {"converged": converged, "finite_outputs": finite_outputs,
                               "valid_probabilities": probability_valid, "n_iter": int(model.n_iter_),
                               "selected_features": int(model.n_features_selected_),
                               "selected_mask_sha256": _array_hash(model.selected_features_),
                               "labels_sha256": _array_hash(labels), "probabilities_sha256": _array_hash(probabilities),
                               "reconstruction_error": _finite(model.reconstruction_err_),
                               "accuracy": float(np.mean(labels == y_test)),
                               "basis_dtype": str(model.selected_components_.dtype),
                               "coordinates_dtype": str(model.training_coordinates_.dtype),
                               "labels_match_warmup": None, "selected_mask_matches_warmup": None,
                               "iterations_match_warmup": None, "max_probability_delta_warmup": None}
                    record["timing"].update({"stage_" + key + "_seconds": float(value) for key, value in model.fit_times_.items()})
                    record["timing"]["projection_seconds"] = float(model.last_transform_seconds_)
                    if reference is None:
                        reference = {"labels": labels.copy(), "probabilities": probabilities.copy(),
                                     "selected": model.selected_features_.copy(), "n_iter": int(model.n_iter_)}
                    else:
                        quality.update(labels_match_warmup=bool(np.array_equal(labels, reference["labels"])),
                                       selected_mask_matches_warmup=bool(np.array_equal(model.selected_features_, reference["selected"])),
                                       iterations_match_warmup=int(model.n_iter_) == reference["n_iter"],
                                       max_probability_delta_warmup=float(np.max(np.abs(probabilities - reference["probabilities"]), initial=0)))
                    stable = task_id == 0 or (quality["labels_match_warmup"] and quality["selected_mask_matches_warmup"]
                                               and quality["iterations_match_warmup"] and quality["max_probability_delta_warmup"] == 0)
                    record["quality"] = quality
                    record["status"] = "qualified" if converged and finite_outputs and probability_valid and stable and not caught else "invalid"
                return record

            warm_record = one_fit(0)
            if warm_record["status"] != "qualified":
                events.put({"event": "warmup_failed", "worker": worker_id, "record": warm_record})
                return
            reference_name = f"reference-repeat-{repeat_id:02d}-worker-{worker_id:03d}.npz"
            np.savez_compressed(Path(output_path) / reference_name, **reference)
            events.put({"event": "ready", "worker": worker_id, "pid": os.getpid(), "threadpools": pools,
                        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
                        "data_sha256": data_identity, "shapes": {"train": list(X_train.shape), "test": list(X_test.shape)},
                        "reference_file": reference_name, "warmup": warm_record})
            while not start_event.wait(0.1):
                if cancel_event.is_set():
                    return
            common_start = shared_start.value
            while time.perf_counter() < common_start:
                if cancel_event.wait(min(0.01, max(0, common_start - time.perf_counter()))):
                    return
            task_id = 0
            while not cancel_event.is_set() and time.perf_counter() < common_start + configuration["duration_seconds"]:
                task_id += 1
                record = one_fit(task_id, common_start)
                events.put({"event": "record", "worker": worker_id, "record": record})
                if record["status"] != "qualified":
                    break
            events.put({"event": "done", "worker": worker_id, "finished_offset_seconds": time.perf_counter() - common_start,
                        "threadpools": threadpool_info()})
    except BaseException as exc:
        events.put({"event": "error", "worker": worker_id, "error_type": type(exc).__name__,
                    "error": str(exc), "traceback": traceback.format_exc()})


def _load_worker_data(np, configuration, worker_id):
    if configuration["mode"] == "matched":
        with np.load(ROOT / "data" / "workload.npz", allow_pickle=False) as arrays:
            return tuple(np.array(arrays[name], copy=True, order="C") for name in ("X_train", "y_train", "X_test", "y_test"))
    n_features = configuration["n_features"]
    n_signal = int(n_features * configuration["signal_ratio"])
    rng = np.random.RandomState(configuration["data_seed"] + worker_id)

    def generate(rows):
        values = np.empty((rows, n_features), dtype=np.float32)
        target = np.zeros(rows, dtype=np.int32)
        target[rows // 2:] = 1
        for first in range(0, rows, 256):
            last = min(rows, first + 256)
            block = np.empty((last - first, n_features), dtype=np.float64)
            block[:, :n_signal] = (1.0 + 2.0 * target[first:last, None]
                                    + rng.normal(0, 0.8, (last - first, n_signal)))
            block[:, n_signal:] = rng.normal(3.0, 9.0, (last - first, n_features - n_signal))
            np.maximum(block, 0, out=block)
            values[first:last] = block
        return values, target

    train, train_y = generate(configuration["n_train"])
    test, test_y = generate(configuration["n_test"])
    return train, train_y, test, test_y


def _sample(processes, samples, started, previous_ticks):
    try:
        rss_values = [process_rss(pid) for pid in [os.getpid()] + [child.pid for child in processes if child.pid is not None]]
        ticks = system_cpu_ticks()
        cpu = None
        if ticks is not None and previous_ticks[0] is not None:
            busy = ticks[0] - previous_ticks[0][0]
            total = ticks[1] - previous_ticks[0][1]
            cpu = 100 * busy / total if total > 0 else None
        previous_ticks[0] = ticks
        memory = memory_info()
        samples.append({"offset_seconds": time.perf_counter() - started, "system_cpu_percent": cpu,
                        "process_tree_rss_bytes": sum(value for value in rss_values if value is not None) if any(value is not None for value in rss_values) else None,
                        "system_available_bytes": memory["available_bytes"], "system_total_bytes": memory["total_bytes"]})
    except (OSError, ValueError, AttributeError):
        return


def summarize_task_counts(records, duration_seconds, n_train, n_features):
    """Count only qualified completions inside the window; preserve drain."""
    completed = [record for record in records if record["completed_within_window"] and record["started_before_deadline"]]
    qualified = [record for record in completed if record["status"] == "qualified"]
    drained = [record for record in records if not record["completed_within_window"]]
    elapsed = max((record["completed_offset_seconds"] for record in records), default=0)
    batch_valid = sum(record["status"] == "qualified" for record in records)
    return {"completed_within_window": len(completed), "qualified_completed": len(qualified),
            "qualified_fits_per_second": len(qualified) / duration_seconds,
            "qualified_fits_per_minute": 60 * len(qualified) / duration_seconds,
            "qualified_training_input_values_per_second": len(qualified) * n_train * n_features / duration_seconds,
            "drain_completed": len(drained), "drain_qualified_completed": sum(record["status"] == "qualified" for record in drained),
            "batch_elapsed_seconds_including_drain": elapsed,
            "batch_qualified_completed_including_drain": batch_valid,
            "batch_qualified_fits_per_second_including_drain": batch_valid / elapsed if elapsed > 0 else None,
            "batch_qualified_input_values_per_second_including_drain": batch_valid * n_train * n_features / elapsed if elapsed > 0 else None,
            "window_had_zero_completions": len(completed) == 0}


def run_repeat(repeat_id, configuration, model, output, startup_timeout, drain_timeout):
    context = mp.get_context("spawn")
    events = context.Queue()
    start_event, cancel_event = context.Event(), context.Event()
    common_start = context.Value("d", 0)
    processes = [context.Process(target=_worker, args=(worker, repeat_id, configuration, model,
                 start_event, common_start, cancel_event, events, str(output)), name=f"snmf-worker-{worker}")
                 for worker in range(configuration["workers"])]
    readiness, records, done, errors, samples = [], [], {}, [], []
    started = time.perf_counter()
    launched_at = None
    previous_ticks = [system_cpu_ticks()]
    last_sample = 0
    try:
        for child in processes:
            child.start()
        while len(done) < len(processes):
            now = time.perf_counter()
            if now - last_sample >= 0.5:
                _sample(processes, samples, started, previous_ticks)
                last_sample = now
            if launched_at is None and now - started > startup_timeout:
                errors.append({"event": "timeout", "phase": "warmup", "seconds": startup_timeout})
                break
            if launched_at is not None and now > launched_at + configuration["duration_seconds"] + drain_timeout:
                errors.append({"event": "timeout", "phase": "drain", "seconds": drain_timeout})
                break
            try:
                message = events.get(timeout=0.1)
            except queue.Empty:
                bad = [child for child in processes if child.exitcode is not None and child.exitcode != 0]
                if bad:
                    errors.extend({"event": "worker_exit", "worker": child.name, "exitcode": child.exitcode} for child in bad)
                    break
                if all(child.exitcode is not None for child in processes):
                    errors.append({"event": "incomplete", "reason": "Workers exited without all completion receipts."})
                    break
                continue
            if message["event"] == "ready":
                readiness.append(message)
                print(f"  repeat {repeat_id}: {len(readiness)}/{len(processes)} workers ready", flush=True)
                if len(readiness) == len(processes):
                    # Small common lead time lets every child leave its barrier.
                    launched_at = time.perf_counter() + 0.2
                    common_start.value = launched_at
                    start_event.set()
                    print(f"  measuring {configuration['duration_seconds']:g} seconds", flush=True)
            elif message["event"] == "record":
                records.append(message["record"])
            elif message["event"] == "done":
                done[message["worker"]] = message
            else:
                errors.append(message)
                break
    finally:
        cancel_event.set()
        start_event.set()
        for child in processes:
            if child.pid is not None:
                child.join(timeout=0.1)
        for child in processes:
            if child.is_alive():
                child.terminate()
        for child in processes:
            if child.pid is not None:
                child.join(timeout=0.5)
        events.close()
        events.cancel_join_thread()
    finished = time.perf_counter()
    duration = configuration["duration_seconds"]
    integrity = not errors and len(done) == len(processes) and all(
        native_pools_valid(item["threadpools"], configuration["native_threads_per_worker"]) for item in done.values())
    if any(record["status"] != "qualified" for record in records):
        integrity = False
    rows = configuration["n_train"]
    features = configuration["n_features"]
    counts = summarize_task_counts(records, duration, rows, features)
    rss_samples = [sample["process_tree_rss_bytes"] for sample in samples if sample["process_tree_rss_bytes"] is not None]
    timed_samples = [sample for sample in samples if launched_at is not None and
                     launched_at - started <= sample["offset_seconds"] <= launched_at - started + duration]
    cpu_samples = [sample["system_cpu_percent"] for sample in timed_samples if sample["system_cpu_percent"] is not None]
    return {"repeat": repeat_id, "status": "qualified" if integrity and counts["qualified_completed"] > 0 else "invalid",
            "numerical_integrity_qualified": integrity, "window_seconds": duration,
            "total_wall_seconds": finished - started,
            "warmup_and_startup_seconds": launched_at - started if launched_at is not None else None,
            **counts,
            "drain_wall_seconds": max(0, finished - launched_at - duration) if launched_at is not None else None,
            "worker_readiness": readiness, "worker_completion": list(done.values()), "records": records,
            "errors": errors, "resource_samples": samples,
            "sampled_peak_process_tree_rss_bytes": max(rss_samples, default=None),
            "cpu_utilization_percent_mean": sum(cpu_samples) / len(cpu_samples) if cpu_samples else None,
            "system_peak_used_ram_bytes": max((sample["system_total_bytes"] - sample["system_available_bytes"]
                for sample in samples if sample["system_total_bytes"] is not None and sample["system_available_bytes"] is not None), default=None)}


def describe_configuration(args, protocol, provenance):
    headers = array_headers(ROOT / "data" / "workload.npz")
    workers = 1 if args.smoke else (provenance["logical_cpus"] or 1) if args.workers == "auto" else int(args.workers)
    native_threads = 1 if args.smoke else getattr(args, "native_threads", 1)
    seconds = 3.0 if args.smoke else args.seconds if args.seconds is not None else float(protocol.get("duration_seconds", 60))
    repeats = 1 if args.smoke else args.repeats if args.repeats is not None else int(protocol.get("repeats", 3))
    if workers < 1 or seconds <= 0 or repeats < 1 or not math.isfinite(seconds):
        raise ValueError("Workers, seconds, and repeats must be positive.")
    if isinstance(native_threads, bool) or not isinstance(native_threads, int) or native_threads < 1:
        raise ValueError("Native threads must be a positive integer.")
    cpu_limit = provenance["logical_cpus"] or 1
    if workers * native_threads > cpu_limit:
        raise ValueError(f"Requested {workers} workers x {native_threads} native threads exceeds the {cpu_limit} available logical CPUs. "
                         "Choose --workers and --native-threads within that CPU budget.")
    n_train, n_features = headers["X_train"]["shape"]
    n_test = headers["X_test"]["shape"][0]
    memory = provenance["memory"]
    available = memory["available_bytes"]
    total = memory["total_bytes"]
    mode = "matched" if args.smoke else args.mode
    reserve = max(4 * GIB, int(0.1 * total)) if total is not None else None
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("--memory-fraction must be greater than zero and at most one.")
    budget = int(max(0, available - reserve) * args.memory_fraction) if available is not None and reserve is not None else None
    capacity = protocol.get("capacity", {})
    bytes_per_feature = 40 if mode == "capacity" else 96
    worker_overhead = 256 * 1024 ** 2
    if mode == "capacity":
        if capacity.get("generator", "original_simulation_v1") != "original_simulation_v1":
            raise ValueError("Unknown capacity generator.")
        n_features = int(capacity.get("input_features", 2048))
        n_test = int(capacity.get("test_rows", 256))
        bytes_per_feature = int(capacity.get("bytes_per_train_feature", 40))
        worker_overhead = int(capacity.get("worker_overhead_bytes", 256 * 1024 ** 2))
        if n_features < 1 or n_test < 4 or n_test % 2 or bytes_per_feature < 32 or worker_overhead < 0:
            raise ValueError("Invalid capacity dimensions or memory estimate.")
        if args.train_rows is not None:
            n_train = args.train_rows
        else:
            if budget is None:
                raise ValueError("Cannot size capacity mode without memory telemetry; provide --train-rows explicitly.")
            per_worker_data = budget // workers - worker_overhead - 16 * n_test * n_features
            n_train = int(per_worker_data // (bytes_per_feature * n_features))
            n_train -= n_train % 2
        if n_train < 32 or n_train % 2:
            raise ValueError("Capacity training rows must be an even integer >=32. There may be insufficient memory for all workers; "
                             "close applications or select --workers explicitly.")
    elif args.train_rows is not None and not args.smoke:
        raise ValueError("--train-rows applies only to capacity mode.")
    per_worker = int(bytes_per_feature * n_train * n_features + 16 * n_test * n_features + worker_overhead)
    configuration = {"mode": mode, "smoke": args.smoke, "workers": workers, "worker_selection": "smoke" if args.smoke else args.workers,
                     "duration_seconds": seconds, "repeats": repeats, "native_threads_per_worker": native_threads,
                     "total_requested_native_threads": workers * native_threads,
                     "n_train": n_train, "n_test": n_test, "n_features": n_features,
                     "dtype": "float32", "estimated_peak_bytes_per_worker": per_worker,
                     "estimated_peak_total_bytes": workers * per_worker,
                     "estimated_peak_bytes": workers * per_worker, "memory_budget_bytes": budget,
                     "memory_reserve_bytes": reserve, "memory_fraction": args.memory_fraction,
                     "bytes_per_train_feature_estimate": bytes_per_feature, "worker_overhead_bytes_estimate": worker_overhead,
                     "training_rows_selection": "explicit" if args.train_rows is not None else "memory_capacity" if mode == "capacity" else "frozen_bundle",
                     "signal_ratio": float(capacity.get("signal_ratio", 0.25)), "data_seed": int(capacity.get("data_seed", 43)),
                     "data_seed_policy": "Capacity worker data seed = data_seed + worker index; fixed across repeats.",
                     "generator": "original_simulation_v1" if mode == "capacity" else "frozen_bundle",
                     "generator_note": "Chunked original-simulation distribution; random draw order differs from the historical monolithic generator.",
                     "model": protocol["model"],
                     "memory_estimate_note": "Conservative estimate, not a promise of RAM use; screening creates float64 intermediates.",
                     "model_seed_policy": "Same frozen model random_state for every warm-up and measured task.",
                     "metric_note": "Window rate counts valid completions before the common deadline. Batch rate including drain is separate. "
                         "Capacity mode changes training rows with RAM; input values/sec is capacity efficiency, not equal-work CPU speedup.",
                     "thread_environment": thread_environment(native_threads), "workload_headers": headers}
    if not 0 < configuration["signal_ratio"] <= 1 or not 0 <= configuration["data_seed"] < 2 ** 32 - workers:
        raise ValueError("Invalid signal ratio or per-worker data seed range.")
    configuration["auto_memory_guard_passed"] = budget is not None and workers * per_worker <= budget
    configuration["memory_guard_passed"] = configuration["auto_memory_guard_passed"]
    return configuration


def write_reports(output, report):
    import statistics
    valid = [item for item in report["repeats"] if item["status"] == "qualified"]
    batch_valid = [item for item in report["repeats"] if item.get("numerical_integrity_qualified")
                   and item["batch_qualified_fits_per_second_including_drain"] is not None]
    report["status"] = "qualified" if len(valid) == report["configuration"]["repeats"] else "invalid"
    rates = [item["qualified_fits_per_second"] for item in valid]
    values = [item["qualified_training_input_values_per_second"] for item in valid]
    report["aggregate"] = {"qualified_repeats": len(valid), "requested_repeats": report["configuration"]["repeats"],
                           "median_qualified_fits_per_second": statistics.median(rates) if rates else None,
                           "min_qualified_fits_per_second": min(rates, default=None),
                           "max_qualified_fits_per_second": max(rates, default=None),
                           "median_qualified_training_input_values_per_second": statistics.median(values) if values else None}
    report["aggregate"].update(
        qualified_fits_per_second_median=statistics.median(rates) if rates else None,
        qualified_input_values_per_second_median=statistics.median(values) if values else None,
        batch_qualified_fits_per_second_median_including_drain=statistics.median(
            item["batch_qualified_fits_per_second_including_drain"] for item in batch_valid) if batch_valid else None,
        batch_qualified_input_values_per_second_median_including_drain=statistics.median(
            item["batch_qualified_input_values_per_second_including_drain"] for item in batch_valid) if batch_valid else None,
        system_peak_used_ram_bytes=max((item["system_peak_used_ram_bytes"] for item in report["repeats"]
                                       if item["system_peak_used_ram_bytes"] is not None), default=None),
        benchmark_peak_rss_bytes=max((item["sampled_peak_process_tree_rss_bytes"] for item in report["repeats"]
                                     if item["sampled_peak_process_tree_rss_bytes"] is not None), default=None),
        cpu_utilization_percent_mean=statistics.mean(item["cpu_utilization_percent_mean"] for item in report["repeats"]
            if item["cpu_utilization_percent_mean"] is not None) if any(item["cpu_utilization_percent_mean"] is not None
                                                                      for item in report["repeats"]) else None)
    json_write(output / "report.json", report)
    fields = ["repeat", "worker", "task", "status", "started_offset_seconds", "completed_offset_seconds",
              "completed_within_window", "fit_seconds", "prediction_seconds", "fit_plus_prediction_seconds",
              "n_iter", "selected_features", "reconstruction_error", "labels_sha256", "probabilities_sha256"]
    with (output / "tasks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for repeat in report["repeats"]:
            for record in repeat["records"]:
                row = {**record, **record["timing"], **(record["quality"] or {})}
                writer.writerow({field: row.get(field) for field in fields})
    config = report["configuration"]
    lines = ["sNMF CD + absolute cutoff full-PC benchmark", f"Label: {report['label']}", f"Status: {report['status']}",
             f"Workers: {config['workers']} ({config.get('native_threads_per_worker', 1)} native numerical threads each)",
             f"Shape per worker: train {config['n_train']} x {config['n_features']}; test {config['n_test']} x {config['n_features']}",
             f"Measured window: {config['duration_seconds']:g} seconds; repeats: {config['repeats']}"]
    for repeat in report["repeats"]:
        lines.append(f"Repeat {repeat['repeat']}: {repeat['qualified_completed']} qualified completions; "
                     f"{repeat['qualified_fits_per_minute']:.3f} fits/min; {repeat['status']}; "
                     f"{repeat['drain_completed']} completions after deadline excluded")
        if repeat["batch_qualified_fits_per_second_including_drain"] is not None:
            lines.append(f"  Separate batch rate including drain: {60 * repeat['batch_qualified_fits_per_second_including_drain']:.3f} fits/min; "
                         f"elapsed {repeat['batch_elapsed_seconds_including_drain']:.3f} seconds")
        if repeat["window_had_zero_completions"]:
            lines.append("  No fit completed before deadline; window rate is zero. Use the separately labeled batch result or a longer window.")
    if rates:
        lines.append(f"Median qualified fits/min: {60 * statistics.median(rates):.3f}")
    lines.extend([config["metric_note"], "Source, model, data, environment, warnings, iterations, and output hashes are in report.json.",
                  "RAM and CPU samples are observations at 0.5-second intervals and can miss short peaks."])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None)
    parser.add_argument("--workers", default="auto", help="auto uses every logical CPU; an integer explicitly selects a worker count")
    parser.add_argument("--native-threads", type=int, default=1,
                        help="Native numerical threads per worker; workers x threads must fit the available CPU count")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--mode", choices=("capacity", "matched"), default="capacity")
    parser.add_argument("--train-rows", type=int, help="Explicit even training rows per capacity worker; otherwise derived from available RAM")
    parser.add_argument("--memory-fraction", type=float, default=0.9, help="Fraction of available RAM after OS reserve to allocate in the peak estimate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One worker, a three-second window, one repeat; validation only")
    parser.add_argument("--describe", action="store_true", help="Verify the bundle and print hardware/workload provenance without fitting")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--drain-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    protocol, manifest_hash, protocol_hash = verify_bundle()
    provenance = hardware()
    configuration = describe_configuration(args, protocol, provenance)
    if args.describe:
        print(json.dumps({"provenance": provenance, "configuration": configuration,
                          "manifest_sha256": manifest_hash, "protocol_sha256": protocol_hash}, indent=2))
        return 0
    if not args.label:
        parser.error("--label is required for a benchmark run")
    if args.startup_timeout <= 0 or args.drain_timeout <= 0:
        parser.error("Timeouts must be positive")
    if not args.smoke and not configuration["memory_guard_passed"]:
        raise ValueError("Requested worker memory estimate exceeds the safe available RAM budget, or RAM is unknown. "
                         "Inspect --describe, close applications, reduce --workers or --train-rows, and retry. "
                         "Explicit worker counts do not bypass the memory guard.")
    label_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", args.label).strip("-")[:70] or "pc"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output.resolve() if args.output else ROOT / "results" / f"{timestamp}-{label_slug}"
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    report = {"schema_version": 1, "author": "Chang Cheng", "label": args.label,
              "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
              "manifest_sha256": manifest_hash, "protocol": protocol, "configuration": configuration,
              "provenance": provenance, "smoke": args.smoke, "repeats": []}
    json_write(output / "report.json", report)
    print(f"Output: {output}", flush=True)
    try:
        for repeat in range(1, configuration["repeats"] + 1):
            result = run_repeat(repeat, configuration, protocol["model"], output, args.startup_timeout, args.drain_timeout)
            report["repeats"].append(result)
            if result["worker_readiness"]:
                report["provenance"]["versions"] = result["worker_readiness"][0]["versions"]
                report["provenance"]["blas"] = result["worker_readiness"][0]["threadpools"]
            write_reports(output, report)
            if result["status"] != "qualified":
                print("Benchmark failed qualification. See report.json; partial timings are not a qualified comparison.", flush=True)
                return 2
    except KeyboardInterrupt:
        report["interrupted"] = True
        write_reports(output, report)
        print("Interrupted; workers stopped and partial report retained.", flush=True)
        return 130
    print((output / "summary.txt").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Benchmark error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
