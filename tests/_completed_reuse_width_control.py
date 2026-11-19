"""Run explicitly in a fresh Linux pytest process with two native threads."""
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest
import sklearn
import threadpoolctl

from tests.test_completed_reuse import committed, completed, inactive_threads, reuse, science, study


def inventory():
    return sorted(threadpoolctl.threadpool_info(), key=lambda row: row["filepath"])


def test_admission_matches_fresh_participant(completed, monkeypatch):
    directories, original, describe, reuse_checkpoint, _, _ = completed
    starting = inventory()
    assert any(row["num_threads"] == 2 for row in starting), starting
    errors, config = np.geterr(), sklearn.get_config()
    saved = [committed(directory) for directory in directories]

    # Some native limits are thread-local. Measure a separate participant under
    # the coordinator's limit before asking the scheduler to make its decision.
    with threadpoolctl.threadpool_limits(limits=1):
        coordinator = inventory()
        with ThreadPoolExecutor(max_workers=1) as pool:
            participant = pool.submit(inventory).result()
    assert coordinator and all(row["num_threads"] == 1 for row in coordinator), coordinator
    assert inventory() == starting

    owner = threading.get_ident()
    observed, visits = [], []
    original_guard = reuse._ReadOnlyEnvironment.__call__

    def observe_guard(self, threads):
        if threading.get_ident() != owner:
            observed.append(inventory())
        return original_guard(self, threads)

    def invoke(i, environment):
        visits.append(i)
        return reuse_checkpoint(i, environment)

    def forbid_fit(*args, **kwargs):
        pytest.fail("completed checkpoint started a representation fit")

    monkeypatch.setattr(reuse._ReadOnlyEnvironment, "__call__", observe_guard)
    monkeypatch.setattr(study, "_run_representation_fold", forbid_fit)
    result = reuse._try_completed([0, 1], 1, describe=describe, invoke=invoke)
    assert observed and observed[0] == participant
    if participant == coordinator:
        assert visits == [0, 1]
        assert [science(report) for report in result] == [science(report) for report in original]
        assert all(report["new_representation_fits"] == 0 and report["reused_representation_folds"] == 3
                   for report in result)
    else:
        assert visits == [] and result is None
    assert [committed(directory) for directory in directories] == saved
    assert inventory() == starting and np.geterr() == errors and sklearn.get_config() == config
    inactive_threads()
    for directory in directories:
        with study._checkpoint_lock(directory):
            pass
