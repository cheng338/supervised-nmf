"""Fork isolation is checked without forking an initialized numerical process."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tests._celeba_recovery_test_support import recovery_module
pytestmark = pytest.mark.skipif(not hasattr(os, 'fork'), reason='Requires os.fork')


def test_forked_child_gets_new_guard_while_parent_lock_stays_owned(tmp_path, recovery_module):
    assert hasattr(os, "fork"), "Select this declared platform-specific module only on fork platforms"
    recovery = recovery_module
    source = Path(recovery.__file__).resolve()
    output = tmp_path / "fork-proof.json"
    child = Path(__file__).with_name("celeba_guard_process_probe.py")
    result = subprocess.run([sys.executable, "-I", "-S", "-B", str(child), str(source),
                             str(output), str(os.getpid()), "fork"],
                            capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(output.read_text(encoding="utf-8"))
    parent = json.loads(output.with_suffix(".parent.json").read_text(encoding="utf-8"))
    assert len({os.getpid(), proof["pid"], proof["parent_pid"]}) == 3
    assert proof["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert proof["guard_entered"] and proof["fresh_child_lock"]
    assert proof["registered_pids"] == [proof["pid"]]
    assert parent["pid"] == proof["parent_pid"] and parent["parent_lock_unchanged"]
    assert parent["registered_pids"] == [parent["pid"]]
    assert parent["other_thread_blocking"] == {"waited_while_owned": True,
        "entered_after_release": True, "thread_completed": True}
    assert proof["numerical_modules"] == parent["numerical_modules"] == []
