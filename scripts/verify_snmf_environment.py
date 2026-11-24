"""Verify an installed package on small synthetic data without opening study data."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits
from supervised_nmf import NMFClassifier
from supervised_nmf.evaluation import evaluate_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous verification receipts')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(23)
    y = np.arange(80) % 2
    coordinates = rng.uniform(.1, 1, (80, 2))
    coordinates[:, 0] += y
    basis = np.column_stack([np.eye(2), rng.uniform(.1, 1, (2, 10))])
    X = (coordinates @ basis + rng.uniform(0, .005, (80, 12))).astype(np.float32)
    query = (X[:12] * 1.03).copy()
    checks = []
    with threadpool_limits(limits=1):
        for solver in ('spa', 'cd', 'mu'):
            for head in ('knn', 'logistic'):
                row = {'solver': solver, 'classifier': head, 'passed': False}
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model = NMFClassifier(n_components=2, solver=solver,
                            screening='absolute', score_threshold=0,
                            classifier=head, n_neighbors=5, random_state=11,
                            max_iter=20000, logistic_class_weight='balanced').fit(X, y)
                        before = model.selected_components_.copy()
                        probability = model.predict_proba(query)
                        prediction = model.predict(query)
                        report = evaluate_classifier(model, query, y[:12], threads=1)
                    assert model.converged_ and report['status'] == 'valid', report
                    assert report['diagnostics']['last_transform_converged']
                    assert report['diagnostics']['last_transform_performed']
                    assert report['diagnostics']['last_transform_error'] is None
                    assert not any(name.startswith('last_transform_') for name in vars(model))
                    assert not any(issubclass(w.category, (ConvergenceWarning, RuntimeWarning)) for w in caught)
                    assert probability.shape == (12, 2) and np.isfinite(probability).all()
                    np.testing.assert_allclose(probability.sum(axis=1), 1)
                    np.testing.assert_array_equal(prediction, probability[:, 1] > .5)
                    np.testing.assert_array_equal(before, model.selected_components_)
                    with tempfile.TemporaryDirectory(prefix='synthetic-model-', dir=args.output.parent) as folder:
                        path = Path(folder) / 'model.joblib'
                        joblib.dump(model, path)
                        np.testing.assert_array_equal(joblib.load(path).predict_proba(query), probability)
                    row.update(passed=True, converged=True, finite_probabilities=True,
                               predictions_consistent=True, basis_unchanged=True,
                               serialization_exact=True)
                except Exception as exc:
                    row.update(error=f'{type(exc).__name__}: {exc}')
                checks.append(row)
        pools = threadpool_info()
    pip = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    result = {'passed': all(r['passed'] for r in checks) and pip.returncode == 0,
              'scope': 'Six installed-estimator fit/predict/serialization checks on synthetic data; not the full source or portable-study test suite',
              'scientific_data_used': False, 'python': platform.python_version(),
              'executable': sys.executable, 'versions': {n: importlib.metadata.version(n)
                for n in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl', 'supervised-nmf')},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checks': checks, 'numerical_threads': 1, 'threadpools': pools,
              'pip_check': pip.stdout.strip(), 'pip_check_exit': pip.returncode}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'checks': len(checks),
                      'failed_checks': [r for r in checks if not r['passed']],
                      'output': str(args.output)}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
