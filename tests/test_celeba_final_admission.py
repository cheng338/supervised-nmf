"""Process-local finalization ownership, completed reuse and queued source checks."""
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event, RLock, Thread, get_ident
from unittest.mock import patch
import warnings
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_info, threadpool_limits
from tests._celeba_recovery_test_support import guard_support, guard_cv, guard_cases

@pytest.fixture
def isolated_guard(guard_support, monkeypatch):
    # A release regression must not strand the suite on its process-wide lock.
    # The tested guard still uses a real RLock through its ordinary PID lookup.
    monkeypatch.setitem(guard_support.recovery._FINAL_AGGREGATION_LOCKS_BY_PID,
                        os.getpid(), RLock())


def daemon_submit(function, *args):
    """Bound a failure probe without executor teardown joining a leaked lock."""
    future = Future()
    def run():
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)
    Thread(target=run, daemon=True).start()
    return future


def invoke(support, case):
    # The caller must not create a competing threadpool context before admission.
    return support.study.aggregate_and_evaluate(case.output, case.cache, case.models,
        case.manifest, 1, case.config)['Smiling']['models']['raw_knn']

def pools():
    return {row['filepath']:row['num_threads'] for row in threadpool_info()}


def warning_state():
    return (warnings.filters, tuple(warnings.filters), warnings._showwarnmsg_impl,
            warnings.showwarning)


def assert_warning_state(before):
    assert warnings.filters is before[0]
    assert tuple(warnings.filters) == before[1]
    assert warnings._showwarnmsg_impl is before[2]
    assert warnings.showwarning is before[3]


@pytest.mark.parametrize("policy", ["off", "all"])
def test_same_thread_nested_disjoint_finalization_preserves_lifo_warning_scopes(
        guard_support, guard_cases, policy):
    support = guard_support
    outer, inner = guard_cases
    for case in guard_cases:
        case.config["final_fit_recovery"] = policy
    original_fit = KNeighborsClassifier.fit
    nested_results = []
    entered = False
    before = warning_state()

    def fit(model, *args, **kwargs):
        nonlocal entered
        if not entered:
            entered = True
            nested_results.append(support.aggregate(inner))
        return original_fit(model, *args, **kwargs)

    with patch.object(KNeighborsClassifier, "fit", fit):
        result = support.aggregate(outer)
    assert result["evaluation"]["status"] == "valid"
    assert len(nested_results) == 1 and nested_results[0]["evaluation"]["status"] == "valid"
    assert outer.final.exists() and inner.final.exists()
    assert_warning_state(before)


def test_separate_fresh_process_can_enter_while_parent_owns_guard(guard_support, tmp_path):
    recovery = guard_support.recovery
    source = Path(recovery.__file__).resolve()
    output = tmp_path / "fresh-process-proof.json"
    child = Path(__file__).with_name("celeba_guard_process_probe.py")
    before = warning_state()
    with recovery._final_aggregation_guard():
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(child), str(source),
                                 str(output), str(os.getpid()), "fresh"],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        proof = json.loads(output.read_text(encoding="utf-8"))
        assert proof["pid"] != os.getpid() and proof["invoking_pid"] == os.getpid()
        assert proof["registered_pids"] == [proof["pid"]]
        assert proof["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert proof["guard_entered"] and proof["numerical_modules"] == []
        assert proof["other_thread_blocking"] == {"waited_while_owned": True,
            "entered_after_release": True, "thread_completed": True}
    assert_warning_state(before)


def test_guard_waits_then_enters_after_owner_release(guard_support, isolated_guard):
    support = guard_support
    attempted, entered = Event(), Event()
    def contender():
        attempted.set()
        with support.recovery._final_aggregation_guard():
            entered.set()
    with support.recovery._final_aggregation_guard():
        future = daemon_submit(contender)
        assert attempted.wait(10)
        assert not entered.wait(.15)
    future.result(timeout=10)
    assert entered.is_set()


@pytest.mark.parametrize('error_type',[ValueError,KeyboardInterrupt])
def test_blocking_guard_exception_releases_for_other_thread(guard_support, error_type, isolated_guard):
    with pytest.raises(error_type):
        with guard_support.recovery._final_aggregation_guard():
            raise error_type('owned failure')
    def enter():
        with guard_support.recovery._final_aggregation_guard():
            return True
    assert daemon_submit(enter).result(timeout=10)


@pytest.mark.parametrize('policy',['off','all'])
def test_disjoint_calls_queue_and_preserve_warning_ownership(guard_support,guard_cases,policy):
    support = guard_support
    owner, contender = guard_cases
    for case in guard_cases:
        case.config['final_fit_recovery'] = policy
    entered, release, contender_prepared = Event(),Event(),Event()
    owner_id = []
    original_fit, original_write = KNeighborsClassifier.fit, support.study.write_json
    before, native_before = warning_state(), pools()
    def fit(model,*args,**kwargs):
        is_owner = get_ident() == owner_id[0]
        warnings.warn('owned-A' if is_owner else 'owned-B',UserWarning)
        if is_owner:
            entered.set()
            if not release.wait(20):
                raise TimeoutError('Owner not released')
        return original_fit(model,*args,**kwargs)
    def write(path,value):
        result = original_write(path,value)
        if path == contender.directory/'calibration-summary.json':
            contender_prepared.set()
        return result
    def first():
        owner_id.append(get_ident())
        return invoke(support,owner)
    with patch.object(KNeighborsClassifier,'fit',fit), patch.object(support.study,'write_json',write), ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(first)
        try:
            assert entered.wait(10)
            b = executor.submit(invoke,support,contender)
            assert contender_prepared.wait(10)
            with pytest.raises(FutureTimeout):
                b.result(timeout=.15)
            assert not contender.final.exists()
        finally:
            release.set()
        ar,br = a.result(timeout=20),b.result(timeout=20)
    assert ar['evaluation']['status'] == br['evaluation']['status'] == 'valid'
    am = [row['message'] for row in ar['final_fit_warnings']]
    bm = [row['message'] for row in br['final_fit_warnings']]
    assert am.count('owned-A') == bm.count('owned-B') == 1
    assert 'owned-B' not in am and 'owned-A' not in bm
    assert_warning_state(before)
    assert pools() == native_before


def test_blocking_owner_preserves_different_requested_native_limits(guard_support):
    support = guard_support
    entered,release,attempted = Event(),Event(),Event()
    before = pools()
    def owner():
        with support.recovery._final_aggregation_guard(), threadpool_limits(limits=2):
            observed = pools()
            entered.set()
            if not release.wait(20):
                raise TimeoutError('Native owner not released')
            assert pools() == observed
    def contender():
        attempted.set()
        with support.recovery._final_aggregation_guard(), threadpool_limits(limits=1):
            assert all(value == 1 for value in pools().values())
    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(owner)
        try:
            assert entered.wait(10)
            b = executor.submit(contender)
            assert attempted.wait(10)
            with pytest.raises(FutureTimeout):
                b.result(timeout=.15)
        finally:
            release.set()
        a.result(timeout=10)
        b.result(timeout=10)
    assert pools() == before


@pytest.mark.parametrize('policy',['off','all'])
def test_callback_wait_cycle_is_bounded_and_reported(guard_support,guard_cases,policy):
    """Shows the wait cycle without leaving either thread blocked indefinitely."""
    support = guard_support
    owner, contender = guard_cases
    for case in guard_cases:
        case.config['final_fit_recovery'] = policy
    owner_id, nested, timed_out = [],[],[]
    ready = Event()
    original_fit, original_write = KNeighborsClassifier.fit,support.study.write_json
    before = warning_state()
    def write(path,value):
        result = original_write(path,value)
        if path == contender.directory/'calibration-summary.json':
            ready.set()
        return result
    def fit(model,*args,**kwargs):
        if get_ident() == owner_id[0]:
            future = executor.submit(invoke,support,contender)
            nested.append(future)
            assert ready.wait(10)
            try:
                future.result(timeout=.15)
            except FutureTimeout:
                timed_out.append(True)
            else:
                raise AssertionError('Blocking policy did not retain operation ownership')
        return original_fit(model,*args,**kwargs)
    def first():
        owner_id.append(get_ident())
        return invoke(support,owner)
    with patch.object(KNeighborsClassifier,'fit',fit),patch.object(support.study,'write_json',write),ThreadPoolExecutor(max_workers=2) as executor:
        ar = executor.submit(first).result(timeout=20)
        assert len(nested) == len(timed_out) == 1
        br = nested[0].result(timeout=20)
    assert ar['evaluation']['status'] == br['evaluation']['status'] == 'valid'
    assert_warning_state(before)


def test_ownership_remains_held_between_methods(guard_support,guard_cases):
    support = guard_support
    owner, contender = guard_cases
    for case in guard_cases:
        case.config['final_fit_recovery'] = 'all'
    owner.models['majority_class'] = (DummyClassifier(strategy='most_frequent'),{})
    reached,release,ready = Event(),Event(),Event()
    owner_id = []
    original_write = support.study.write_json
    import builtins
    def printing(*args,**kwargs):
        if get_ident() == owner_id[0] and args[:2] == ('Smiling','raw_knn'):
            reached.set()
            if not release.wait(20):
                raise TimeoutError('Between-method owner not released')
        return builtins.print(*args,**kwargs)
    def write(path,value):
        result = original_write(path,value)
        if path == contender.directory/'calibration-summary.json':
            ready.set()
        return result
    def first():
        owner_id.append(get_ident())
        return invoke(support,owner)
    before = warning_state()
    with patch.object(support.study,'print',printing,create=True),patch.object(support.study,'write_json',write),ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(first)
        try:
            assert reached.wait(10) and owner.final.is_file()
            b = executor.submit(invoke,support,contender)
            assert ready.wait(10)
            with pytest.raises(FutureTimeout):
                b.result(timeout=.15)
        finally:
            release.set()
        assert a.result(timeout=20)['evaluation']['status'] == 'valid'
        assert b.result(timeout=20)['evaluation']['status'] == 'valid'
    assert (owner.output/'Smiling/majority_class/development-evaluation.json').is_file()
    assert_warning_state(before)


@pytest.mark.parametrize("policy", ["off", "all"])
def test_completed_other_output_succeeds_during_active_fit(guard_support, guard_cases, policy):
    support = guard_support
    owner, completed = guard_cases
    for case in guard_cases:
        case.config["final_fit_recovery"] = policy
    expected = invoke(support, completed)
    assert expected["evaluation"]["status"] == "valid"
    entered, release = Event(), Event()
    owner_thread, recovery_entries = [], []
    original_fit = KNeighborsClassifier.fit
    original_recover = support.study._recoverable_final_fit
    before = warning_state()

    def fit(model, *args, **kwargs):
        if get_ident() == owner_thread[0]:
            entered.set()
            if not release.wait(40):
                raise TimeoutError("Owner was not released")
        return original_fit(model, *args, **kwargs)

    def recover(*args, **kwargs):
        recovery_entries.append(get_ident())
        return original_recover(*args, **kwargs)

    def run_owner():
        owner_thread.append(get_ident())
        return invoke(support, owner)

    with patch.object(KNeighborsClassifier, "fit", fit), \
            patch.object(support.study, "_recoverable_final_fit", recover), \
            threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_owner)
        try:
            assert entered.wait(20)
            assert executor.submit(invoke, support, completed).result(timeout=10) == expected
            assert recovery_entries == owner_thread
        finally:
            release.set()
        assert first.result(timeout=30)["evaluation"]["status"] == "valid"
    assert_warning_state(before)


@pytest.mark.parametrize("policy", ["off", "all"])
def test_completed_separate_outputs_overlap_without_process_guard(guard_support, guard_cases, policy):
    support = guard_support
    owner, contender = guard_cases
    for case in guard_cases:
        case.config["final_fit_recovery"] = policy
    expected = [invoke(support, case) for case in guard_cases]
    entered, release = Event(), Event()
    owner_thread = []
    original_read = support.study._read_report
    before = warning_state()

    def read(path, *args, **kwargs):
        if get_ident() == owner_thread[0] and path == owner.final:
            entered.set()
            if not release.wait(40):
                raise TimeoutError("Completed reader was not released")
        return original_read(path, *args, **kwargs)

    def run_owner():
        owner_thread.append(get_ident())
        return invoke(support, owner)

    with patch.object(support.study, "_read_report", read), \
            patch.object(support.recovery, "_final_aggregation_guard",
                         side_effect=AssertionError("Completed reuse acquired process guard")), \
            patch.object(support.study, "_recoverable_final_fit",
                         side_effect=AssertionError("Completed reuse entered recovery")), \
            ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_owner)
        try:
            assert entered.wait(20)
            assert executor.submit(invoke, support, contender).result(timeout=10) == expected[1]
        finally:
            release.set()
        assert first.result(timeout=30) == expected[0]
    assert_warning_state(before)


def test_completed_same_output_retains_original_file_lock_error(guard_support, guard_cases):
    support, case = guard_support, guard_cases[0]
    case.config["final_fit_recovery"] = "all"
    expected = invoke(support, case)
    entered, release = Event(), Event()
    owner_thread = []
    original_read = support.study._read_report
    before = warning_state()

    def read(path, *args, **kwargs):
        if get_ident() == owner_thread[0] and path == case.final:
            entered.set()
            if not release.wait(40):
                raise TimeoutError("Completed reader was not released")
        return original_read(path, *args, **kwargs)

    def run_owner():
        owner_thread.append(get_ident())
        return invoke(support, case)

    with patch.object(support.study, "_read_report", read), \
            patch.object(support.recovery, "_final_aggregation_guard",
                         side_effect=AssertionError("Completed reuse acquired process guard")), \
            patch.object(support.study, "_recoverable_final_fit",
                         side_effect=AssertionError("Completed reuse entered recovery")), \
            ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_owner)
        try:
            assert entered.wait(20)
            second = executor.submit(invoke, support, case)
            with pytest.raises(RuntimeError, match="Another process or thread is evaluating this study") as caught:
                second.result(timeout=10)
            assert "Another thread is finalizing" not in str(caught.value)
        finally:
            release.set()
        assert first.result(timeout=30) == expected
    assert_warning_state(before)


@pytest.mark.parametrize("exception_type", [ValueError, KeyboardInterrupt])
def test_failure_after_lazy_acquisition_releases_for_another_thread(
        guard_support, guard_cases, exception_type, isolated_guard):
    support, case = guard_support, guard_cases[0]
    case.config["final_fit_recovery"] = "all"
    before = warning_state()
    with patch.object(support.study, "_recoverable_final_fit", side_effect=exception_type("after-admission")):
        with pytest.raises(exception_type, match="after-admission"):
            invoke(support, case)
    assert not case.final.exists()
    # A same-thread retry alone could hide a leaked reentrant lock.
    assert daemon_submit(invoke, support, case).result(timeout=30)["evaluation"]["status"] == "valid"
    assert_warning_state(before)


@contextmanager
def blocking_admission_observer(support):
    """Delegate every real RLock operation; signal its blocking acquisition.

    This occurs only after the real nonblocking attempt failed. Merely
    signalling before calling the guard could race the owner release and would
    not establish that the new conditional check was actually eligible.
    """
    locks = support.recovery._FINAL_AGGREGATION_LOCKS_BY_PID
    pid, owner = os.getpid(), get_ident()
    actual = locks[pid]
    reached = Event()
    class Delegate:
        def acquire(self, *args, **kwargs):
            blocking = kwargs.get('blocking', args[0] if args else True)
            if blocking and get_ident() != owner:
                reached.set()
            return actual.acquire(*args, **kwargs)
        def release(self):
            return actual.release()
    with patch.dict(locks, {pid: Delegate()}):
        yield reached


@pytest.mark.parametrize('policy', ['off', 'all'])
@pytest.mark.parametrize('changed_source', ['script', 'package'])
def test_queued_actual_source_change_precedes_clone_fit(guard_support, guard_cases, policy, changed_source, tmp_path, monkeypatch):
    support = guard_support
    case, retry = guard_cases
    case.config['final_fit_recovery'] = retry.config['final_fit_recovery'] = policy
    # Hash actual changed bytes in test-owned copies, never edit checkout files.
    if changed_source == 'script':
        source = tmp_path / 'study.py'
        shutil.copyfile(support.study.__file__, source)
        monkeypatch.setattr(support.study, '__file__', str(source))
    else:
        from supervised_nmf import evaluation as evaluation_module
        package = tmp_path / 'package'
        package.mkdir()
        for path in Path(evaluation_module.__file__).parent.glob('*.py'):
            shutil.copyfile(path, package / path.name)
        source = package / 'evaluation.py'
        monkeypatch.setattr(evaluation_module, '__file__', str(source))
    assert source.is_file()
    original_bytes = source.read_bytes()
    changed_bytes = original_bytes + b'\n# Controlled test-owned queued-source change.\n'
    original_guard = support.recovery._final_aggregation_guard
    original_clone, original_fit = support.study.clone, KNeighborsClassifier.fit
    calls = []
    before = warning_state()

    def clone(*args, **kwargs):
        calls.append('clone')
        return original_clone(*args, **kwargs)

    def fit(*args, **kwargs):
        calls.append('fit')
        return original_fit(*args, **kwargs)

    value, error = None, None
    with ThreadPoolExecutor(max_workers=1) as executor, blocking_admission_observer(support) as attempted:
        with patch.object(support.study, 'clone', clone), patch.object(KNeighborsClassifier, 'fit', fit):
            try:
                with original_guard():
                    future = executor.submit(invoke, support, case)
                    assert attempted.wait(timeout=10), 'Caller did not reach admission after preflight'
                    assert not future.done()
                    source.write_bytes(changed_bytes)
                try:
                    value = future.result(timeout=20)
                except BaseException as caught:
                    error = caught
            finally:
                source.write_bytes(original_bytes)
            assert source.read_bytes() == original_bytes
            assert isinstance(error, (ValueError, support.recovery.RecoveryError)), repr(error)
            assert 'Source' in str(error)
            assert value is None and calls == []
            assert not case.final.exists()
            calls.clear()
            retried = executor.submit(invoke, support, retry).result(timeout=20)
            assert retried['evaluation']['status'] == 'valid'
            assert calls == ['clone', 'fit']
    assert_warning_state(before)


@pytest.mark.parametrize('policy', ['off', 'all'])
def test_only_contended_off_adds_one_source_recheck(guard_support, guard_cases, policy):
    support, case = guard_support, guard_cases[0]
    case.config['final_fit_recovery'] = policy
    original_guard = support.recovery._final_aggregation_guard
    original_script, original_package = support.study._source_hash, support.study._package_source_hashes
    calls = {'script': 0, 'package': 0}

    def script():
        calls['script'] += 1
        return original_script()

    def package():
        calls['package'] += 1
        return original_package()

    with ThreadPoolExecutor(max_workers=1) as executor, blocking_admission_observer(support) as attempted, \
         patch.object(support.study, '_source_hash', script), \
         patch.object(support.study, '_package_source_hashes', package):
        with original_guard():
            future = executor.submit(invoke, support, case)
            assert attempted.wait(timeout=10)
            assert calls == {'script': 1, 'package': 1}
        result = future.result(timeout=20)
    assert result['evaluation']['status'] == 'valid'
    expected = 4 if policy == 'all' else 2
    assert calls == {'script': expected, 'package': expected}


def test_uncontended_off_and_completed_keep_original_source_scan_count(guard_support, guard_cases):
    support, case = guard_support, guard_cases[0]
    case.config['final_fit_recovery'] = 'off'
    original_script, original_package = support.study._source_hash, support.study._package_source_hashes
    calls = {'script': 0, 'package': 0}

    def script():
        calls['script'] += 1
        return original_script()

    def package():
        calls['package'] += 1
        return original_package()

    def forbidden():
        raise AssertionError('Completed result attempted ownership')

    with patch.object(support.study, '_source_hash', script), \
         patch.object(support.study, '_package_source_hashes', package):
        first = invoke(support, case)
        assert first['evaluation']['status'] == 'valid'
        assert calls == {'script': 1, 'package': 1}
        calls.update(script=0, package=0)
        with patch.object(support.recovery, '_final_aggregation_guard', forbidden):
            assert invoke(support, case) == first
        assert calls == {'script': 1, 'package': 1}


def test_guard_reports_only_actual_contention_and_remains_reentrant(guard_support):
    support = guard_support
    observed = []
    def attempt():
        with support.recovery._final_aggregation_guard() as waited:
            observed.append(waited)
    with ThreadPoolExecutor(max_workers=1) as executor, blocking_admission_observer(support) as attempted:
        with support.recovery._final_aggregation_guard() as first:
            assert first is False
            with support.recovery._final_aggregation_guard() as nested:
                assert nested is False
            future = executor.submit(attempt)
            assert attempted.wait(timeout=10) and not future.done()
        future.result(timeout=10)
    assert observed == [True]
