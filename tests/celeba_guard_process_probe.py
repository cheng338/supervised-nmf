"""Fresh stdlib-only process checks for finalization lock ownership."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time


def load(path):
    spec = importlib.util.spec_from_file_location("celeba_guard_only_recovery", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def no_numerical_modules():
    names = ("numpy", "scipy", "sklearn", "threadpoolctl", "joblib")
    found = sorted(name for name in sys.modules if name.split(".")[0] in names)
    assert not found, found
    return found


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def begin_other_thread_wait(module):
    observed = []
    started = threading.Event()
    entered = threading.Event()

    def attempt():
        started.set()
        try:
            with module._final_aggregation_guard():
                observed.append("entered")
                entered.set()
        except BaseException as error:
            observed.append(type(error).__name__ + ": " + str(error))

    # A failed assertion must not keep this bounded diagnostic process alive.
    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    assert started.wait(timeout=5)
    assert not entered.wait(timeout=0.15), "Other thread entered while owner held the guard"
    assert thread.is_alive() and observed == []
    return thread, entered, observed


def finish_other_thread_wait(waiter):
    thread, entered, observed = waiter
    thread.join(timeout=5)
    assert not thread.is_alive(), "Other thread did not enter after owner released the guard"
    assert entered.is_set() and observed == ["entered"], observed
    return {"waited_while_owned": True, "entered_after_release": True,
            "thread_completed": True}


def main():
    source, output, invoking_pid, mode = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    module = load(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert invoking_pid != os.getpid()
    if mode == "fresh":
        with module._final_aggregation_guard():
            waiter = begin_other_thread_wait(module)
            proof = {"mode": mode, "pid": os.getpid(), "invoking_pid": invoking_pid,
                "source_sha256": source_hash,
                "registered_pids": list(module._FINAL_AGGREGATION_LOCKS_BY_PID),
                "numerical_modules": no_numerical_modules(), "guard_entered": True}
        proof["other_thread_blocking"] = finish_other_thread_wait(waiter)
        write_new(output, proof)
        return
    if mode != "fork" or not hasattr(os, "fork"):
        raise RuntimeError("Fork guard probe requires a platform with os.fork")
    # Parent and child below are both this isolated stdlib-only process tree.
    # Keep the original parent lock referenced so object IDs cannot be reused.
    with module._final_aggregation_guard():
        parent_pid = os.getpid()
        parent_lock = module._FINAL_AGGREGATION_LOCKS_BY_PID[parent_pid]
        child_pid = os.fork()
        if child_pid == 0:
            try:
                with module._final_aggregation_guard():
                    own_pid = os.getpid()
                    own_lock = module._FINAL_AGGREGATION_LOCKS_BY_PID[own_pid]
                    assert list(module._FINAL_AGGREGATION_LOCKS_BY_PID) == [own_pid]
                    assert own_lock is not parent_lock
                    proof = {"mode": mode, "pid": own_pid, "parent_pid": parent_pid,
                        "invoking_pid": invoking_pid, "source_sha256": source_hash,
                        "registered_pids": list(module._FINAL_AGGREGATION_LOCKS_BY_PID),
                        "fresh_child_lock": own_lock is not parent_lock,
                        "numerical_modules": no_numerical_modules(), "guard_entered": True}
                write_new(output, proof)
                os._exit(0)
            except BaseException as error:
                write_new(output.with_suffix(".error.json"), {"type": type(error).__name__, "message": str(error)})
                os._exit(1)
        deadline = time.monotonic() + 20
        while True:
            done, status = os.waitpid(child_pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() > deadline:
                os.kill(child_pid, signal.SIGKILL)
                os.waitpid(child_pid, 0)
                raise TimeoutError("Forked guard-only child did not exit")
            time.sleep(0.01)
        assert os.waitstatus_to_exitcode(status) == 0, status
        assert list(module._FINAL_AGGREGATION_LOCKS_BY_PID) == [parent_pid]
        assert module._FINAL_AGGREGATION_LOCKS_BY_PID[parent_pid] is parent_lock
        waiter = begin_other_thread_wait(module)
        parent_proof = {"pid": parent_pid, "registered_pids": [parent_pid],
            "parent_lock_unchanged": True,
            "numerical_modules": no_numerical_modules()}
    parent_proof["other_thread_blocking"] = finish_other_thread_wait(waiter)
    write_new(output.with_suffix(".parent.json"), parent_proof)


if __name__ == "__main__":
    main()
