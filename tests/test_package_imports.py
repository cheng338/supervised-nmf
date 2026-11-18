"""Public import behavior must also hold in a fresh interpreter."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests._package_import_probe import CASES


@pytest.mark.parametrize('case', CASES)
def test_public_import_in_fresh_process(case, tmp_path):
    spec = importlib.util.find_spec('supervised_nmf')
    assert spec is not None and spec.origin is not None
    package_root = Path(spec.origin).resolve().parent
    assert Path(spec.origin).name == '__init__.py'
    helper = Path(__file__).with_name('_package_import_probe.py').resolve()
    output = tmp_path / 'result'
    environment = os.environ.copy()
    for key in ('BLIS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_MAX_ACTIVE_LEVELS',
                'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS'):
        environment[key] = '1'
    environment.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE',
                       PYTHONDONTWRITEBYTECODE='1')
    arguments = [sys.executable, '-I', '-B', str(helper),
                 '--package-root', str(package_root), '--case', case,
                 '--output', str(output)]
    result = subprocess.run(arguments, env=environment, capture_output=True,
                            text=True, timeout=240)
    receipt_path = output / 'RESULT.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.is_file() else {}
    assert result.returncode == 0, (result.stdout, result.stderr, receipt)
    assert receipt['status'] == 'passed' and receipt['case'] == case
    assert Path(receipt['package_root']).resolve() == package_root
    assert receipt['origins']['supervised_nmf'] == '__init__.py'
