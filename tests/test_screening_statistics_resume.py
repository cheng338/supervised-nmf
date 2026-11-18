"""Fresh-process partial recovery with real prior consumers and immutable events."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests._screening_recovery_support import screening_policy, stable_bytes, study

pytestmark = pytest.mark.usefixtures('screening_policy')


def child(root, mode, order, case):
    script = Path(__file__).with_name('_screening_recovery_child.py')
    source = Path(study.__file__).resolve().parents[1]
    command = [sys.executable, '-I', '-B', str(script), str(source), str(root), mode, order, case]
    environment = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'AOCL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment[name] = '1'
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, env=environment)
    assert completed.returncode == 0, (completed.returncode, completed.stdout, completed.stderr)
    result = json.loads((root/(mode+'-'+case+'.json')).read_bytes())
    assert result['passed'] and result['pid'] != os.getpid()
    assert result['source_sha256'] == {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(study.__file__).parent.glob('*.py'))}
    return result


@pytest.mark.parametrize('order', ['representation', 'fold'])
def test_partial_consumer_repair_and_second_interruption_across_processes(tmp_path, order):
    prepared = child(tmp_path, 'prepare', order, 'template')
    template = tmp_path/'template'
    preserved = stable_bytes(template)
    assert preserved == prepared['before']
    tokens = {prepared['process_token']}
    for case in ('healthy', 'repair', 'broken'):
        shutil.copytree(template, tmp_path/case)
        assert stable_bytes(tmp_path/case) == preserved

    # Healthy reuse uses the old fold event with a previously committed consumer.
    for mode in ('resume', 'completed'):
        value = child(tmp_path, mode, order, 'healthy')
        assert value['process_token'] not in tokens
        tokens.add(value['process_token'])

    # Lose only learned bytes. Interrupt after the repaired representation fit,
    # leaving its new event orphaned, then restore it in another fresh process.
    (tmp_path/'repair/statistics-events'/('payload-'+prepared['target_event']+'.npz')).unlink()
    for mode in ('repair_interrupt', 'resume', 'completed'):
        value = child(tmp_path, mode, order, 'repair')
        assert value['process_token'] not in tokens
        tokens.add(value['process_token'])

    # Independent damage creates earlier missing work before a retained broken
    # consumer. Preflight must reject before creating even a running attempt.
    (tmp_path/'broken/statistics-events'/('event-'+prepared['target_event']+'.json')).unlink()
    (tmp_path/'broken/folds'/(prepared['first_key']+'.json')).unlink()
    before = stable_bytes(tmp_path/'broken')
    value = child(tmp_path, 'missing_authority', order, 'broken')
    assert value['process_token'] not in tokens
    assert stable_bytes(tmp_path/'broken') == before
    assert stable_bytes(template) == preserved


def test_separate_process_cannot_enter_an_owned_checkpoint(tmp_path):
    directory = tmp_path/'blocked'
    directory.mkdir()
    with study._checkpoint_lock(directory):
        result = child(tmp_path, 'blocked', 'representation', 'blocked')
        assert result['call']['counts']['statistics'] == result['call']['counts']['representations'] == 0
        assert not (directory/'attempts').exists() and not (directory/'statistics-events').exists()
