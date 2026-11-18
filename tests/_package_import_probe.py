"""Fresh-process checks for public imports and eager numerical guards.

Importing this helper uses only the standard library. Each case runs in a
separate child bound to the package source selected by its parent test.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import traceback

CASES = ('namespace', 'class_attribute', 'module_attribute', 'from_module',
         'direct_submodule', 'star', 'reload_before_use', 'reload_rebind',
         'reload_missing', 'reload_deleted', 'concurrent', 'concurrent_failure', 'failure_class',
         'failure_module', 'anchors', 'anchor_mutations', 'class_lifecycle')


def package():
    return importlib.import_module('supervised_nmf')


def namespace_facts(module):
    return dict(raw_class='NMFClassifier' in vars(module), raw_module='classifier' in vars(module),
        module_loaded='supervised_nmf.classifier' in sys.modules,
        names=list(module.__all__), advertised=sorted(set(module.__all__) & set(dir(module))))


def capture_anchors():
    declarations = {
        'supervised_nmf._mu': ('_NMF_TRANSFORM', '_owned_log_sum', '_owned_log_sum_compatible',
                               '_SUFFIX_CACHE_FUNCTIONS', '_SUFFIX_SETUP_FUNCTIONS'),
        'supervised_nmf._mu_rowwise': ('_SUFFIX_CACHE_TRANSFORM', '_SUFFIX_SETUP_TRY'),
        'supervised_nmf._nnls': ('_QR_ORIGINAL_QR', '_QR_ORIGINAL_COND'),
        'supervised_nmf._projection': ('_QR_STANDARD_BLOCK', '_QR_CONTROLLER',
                                     '_QR_NATIVE_CONTROLLER', '_QR_MKL_BINDING'),
        'supervised_nmf.screening': ('_SCREENING_MEAN_READY', '_SCREENING_NORMALIZE_READY'),
    }
    result = {}
    for name, attributes in declarations.items():
        module = sys.modules[name]
        result[name] = (module, tuple((key, vars(module)[key]) for key in attributes))
    return result


def same_anchors(before, after):
    assert before.keys() == after.keys()
    for name in before:
        assert before[name][0] is after[name][0]
        for (old_name, old), (new_name, new) in zip(before[name][1], after[name][1]):
            assert old_name == new_name and old is new, (name, old_name)


class ClassifierFinder(importlib.abc.MetaPathFinder):
    def __init__(self, failures=0):
        self.failures = failures
        self.executions = 0
        self.partial = object()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'supervised_nmf.classifier':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        original = spec.loader
        observer = self
        class Loader(importlib.abc.Loader):
            def create_module(self, module_spec):
                return original.create_module(module_spec)
            def exec_module(self, module):
                observer.executions += 1
                if observer.failures:
                    observer.failures -= 1
                    module.NMFClassifier = observer.partial
                    raise ImportError('deliberate classifier initialization failure')
                return original.exec_module(module)
        spec.loader = Loader()
        return spec


def run_case(case, output):
    evidence = {}
    if case == 'concurrent_failure':
        finder = ClassifierFinder(failures=1)
        barrier = threading.Barrier(8)
        sys.meta_path.insert(0, finder)
        try:
            deferred = package()
            assert finder.executions == 0
            def access(index):
                barrier.wait(timeout=30)
                try:
                    module = deferred
                    value = module.NMFClassifier if index % 2 else module.classifier.NMFClassifier
                    return True, value
                except ImportError as error:
                    return False, str(error)
            with ThreadPoolExecutor(max_workers=8) as pool:
                answers = list(pool.map(access, range(8)))
            failures = [value for success, value in answers if not success]
            successes = [value for success, value in answers if success]
            assert failures == ['deliberate classifier initialization failure']
            assert len(successes) == 7 and finder.executions == 2
            module = package()
            assert all(value is module.NMFClassifier for value in successes)
            assert module.NMFClassifier is module.classifier.NMFClassifier is not finder.partial
            evidence.update(readers=8, failures=failures, recovered_readers=7,
                            classifier_executions=finder.executions,
                            failed_at='attribute access')
        finally:
            sys.meta_path.remove(finder)
        return evidence
    if case.startswith('failure_'):
        finder = ClassifierFinder(failures=1)
        sys.meta_path.insert(0, finder)
        try:
            module = package()
            assert finder.executions == 0
            name = 'NMFClassifier' if case == 'failure_class' else 'classifier'
            try:
                getattr(module, name)
            except ImportError as error:
                assert str(error) == 'deliberate classifier initialization failure'
            else:
                raise AssertionError('Deferred import failure was hidden')
            assert sys.modules['supervised_nmf'] is module
            assert 'NMFClassifier' not in vars(module) and 'classifier' not in vars(module)
            assert 'supervised_nmf.classifier' not in sys.modules
            module = package()
            actual = module.NMFClassifier
            assert actual is module.classifier.NMFClassifier and actual is not finder.partial
            assert finder.executions == 2
            evidence.update(failed_at='first attribute access',
                            executions=finder.executions, retry_succeeded=True)
        finally:
            sys.meta_path.remove(finder)
        return evidence
    module = package()
    evidence['before'] = namespace_facts(module)
    assert evidence['before']['raw_class'] is False
    assert evidence['before']['module_loaded'] is False
    if case == 'namespace':
        assert module.__all__ == ['NMFClassifier', 'FeatureSPA', 'BinaryScreening']
        assert set(module.__all__) | {'classifier'} <= set(dir(module))
        for name in ('missing_public_export', '__missing_private_export__'):
            assert not hasattr(module, name)
            try:
                getattr(module, name)
            except AttributeError as error:
                assert str(error) == f"module 'supervised_nmf' has no attribute {name!r}"
            else:
                raise AssertionError('Unknown name resolved')
        assert namespace_facts(module) == evidence['before']
        evidence['deferred_exports'] = 'Raw namespace and classifier import are deferred'
    elif case in ('class_attribute', 'module_attribute', 'from_module', 'direct_submodule', 'star'):
        if case == 'class_attribute':
            first = module.NMFClassifier
        elif case == 'module_attribute':
            first = module.classifier.NMFClassifier
        elif case == 'from_module':
            from supervised_nmf import classifier
            first = classifier.NMFClassifier
        elif case == 'direct_submodule':
            import supervised_nmf.classifier as classifier
            first = classifier.NMFClassifier
        else:
            exports = {}
            exec('from supervised_nmf import *', exports)
            assert set(exports) - {'__builtins__'} == set(module.__all__)
            assert exports['FeatureSPA'] is module.FeatureSPA
            assert exports['BinaryScreening'] is module.BinaryScreening
            first = exports['NMFClassifier']
        from supervised_nmf import NMFClassifier
        from supervised_nmf.classifier import NMFClassifier as direct
        assert first is NMFClassifier is direct is module.NMFClassifier is module.classifier.NMFClassifier
        assert first.__module__ == 'supervised_nmf.classifier' and first.__qualname__ == 'NMFClassifier'
        assert vars(module)['NMFClassifier'] is first
    elif case == 'reload_before_use':
        anchors = capture_anchors()
        spa, screen = module.FeatureSPA, module.BinaryScreening
        assert importlib.reload(module) is module
        assert module.FeatureSPA is spa and module.BinaryScreening is screen
        same_anchors(anchors, capture_anchors())
        assert namespace_facts(module) == evidence['before']
    elif case in ('reload_rebind', 'reload_missing', 'reload_deleted'):
        original = module.NMFClassifier
        classifier = module.classifier
        anchors = capture_anchors()
        if case == 'reload_rebind':
            sentinel = object()
            classifier.NMFClassifier = sentinel
            try:
                importlib.reload(module)
                assert vars(module)['NMFClassifier'] is sentinel
            finally:
                classifier.NMFClassifier = original
                importlib.reload(module)
            assert module.NMFClassifier is original
        elif case == 'reload_deleted':
            del module.NMFClassifier
            importlib.reload(module)
            assert vars(module)['NMFClassifier'] is original
        else:
            del classifier.NMFClassifier
            try:
                try:
                    importlib.reload(module)
                except ImportError:
                    pass
                else:
                    raise AssertionError('Missing class on reload was hidden')
                assert vars(module)['NMFClassifier'] is original
            finally:
                classifier.NMFClassifier = original
                importlib.reload(module)
            assert module.NMFClassifier is original
        same_anchors(anchors, capture_anchors())
    elif case == 'concurrent':
        finder = ClassifierFinder()
        barrier = threading.Barrier(8)
        def access(index):
            barrier.wait(timeout=30)
            if index % 3 == 0:
                return module.NMFClassifier
            if index % 3 == 1:
                return module.classifier.NMFClassifier
            return importlib.import_module('supervised_nmf.classifier').NMFClassifier
        sys.meta_path.insert(0, finder)
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                answers = list(pool.map(access, range(8)))
            assert all(value is module.NMFClassifier for value in answers)
            assert finder.executions == 1
        finally:
            sys.meta_path.remove(finder)
        evidence.update(readers=8, classifier_executions=finder.executions)
    elif case == 'anchors':
        import numpy as np
        import scipy.linalg
        from sklearn.decomposition import NMF
        import threadpoolctl
        anchors = capture_anchors()
        mu, rowwise = sys.modules['supervised_nmf._mu'], sys.modules['supervised_nmf._mu_rowwise']
        nnls = sys.modules['supervised_nmf._nnls']
        assert mu._NMF_TRANSFORM is NMF.transform
        assert rowwise._SUFFIX_CACHE_TRANSFORM is mu.transform_mu
        assert rowwise._SUFFIX_SETUP_TRY is mu._try_log_row_recovery
        assert nnls._QR_ORIGINAL_QR is scipy.linalg.qr and nnls._QR_ORIGINAL_COND is np.linalg.cond
        state = np.geterr()
        before_pools = threadpoolctl.threadpool_info()
        module.NMFClassifier
        same_anchors(anchors, capture_anchors())
        assert np.geterr() == state
        after_pools = threadpoolctl.threadpool_info()
        assert sorted(before_pools, key=repr) == sorted(after_pools, key=repr)
        evidence.update(native_pools=before_pools,
            eager_modules=sorted(anchors), owned_reduction_ready=mu._owned_log_sum_compatible())
    elif case == 'anchor_mutations':
        import numpy as np
        from sklearn.decomposition import NMF
        import scipy.special._logsumexp as reduction
        mu = sys.modules['supervised_nmf._mu']
        anchor = mu._NMF_TRANSFORM
        old_defaults = reduction.logsumexp.__defaults__
        ready = mu._owned_log_sum_compatible()
        replacement = lambda *args, **kwargs: None
        NMF.transform = replacement
        reduction.logsumexp.__defaults__ = tuple(list(old_defaults or ()) + [object()])
        try:
            module.NMFClassifier
            assert mu._NMF_TRANSFORM is anchor and mu._NMF_TRANSFORM is not NMF.transform
            assert mu._standard_mu_basis(NMF(n_components=2, solver='mu'), np.ones((2, 2))) is None
            assert not mu._owned_log_sum_compatible()
        finally:
            NMF.transform = anchor
            reduction.logsumexp.__defaults__ = old_defaults
        assert mu._owned_log_sum_compatible() is ready
        evidence.update(owned_reduction_before=ready, changed_owned_reduction=False,
                        transform_replacement_not_recaptured=True)
    elif case == 'class_lifecycle':
        from sklearn.base import clone
        cls = module.NMFClassifier
        model = cls(n_components=3, screening='none', random_state=17)
        for value in (clone(model), copy.copy(model), copy.deepcopy(model),
                      pickle.loads(pickle.dumps(model, protocol=5))):
            assert type(value) is cls and value.get_params(deep=False) == model.get_params(deep=False)
        blob = output / 'model.pkl'
        blob.write_bytes(pickle.dumps(model, protocol=5))
        target = output / 'child'
        argv = [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
            '--package-root', str(ARGS.package_root.resolve()), '--case', 'pickle_child',
            '--output', str(target), '--pickle', str(blob)]
        result = subprocess.run(argv, env=os.environ.copy(), capture_output=True, text=True, timeout=120)
        (output / 'child.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        assert result.returncode == 0, result.stdout + result.stderr
        evidence['child'] = json.loads((target / 'RESULT.json').read_text())
        assert evidence['child']['status'] == 'passed'
    elif case == 'pickle_child':
        before = namespace_facts(module)
        value = pickle.loads(ARGS.pickle.read_bytes())
        assert type(value) is module.NMFClassifier is module.classifier.NMFClassifier
        assert value.n_components == 3 and value.random_state == 17
        evidence.update(before_unpickle=before, actual_class_module=type(value).__module__)
    else:
        raise AssertionError(case)
    evidence['after'] = namespace_facts(module)
    return evidence


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package_root = args.package_root.resolve(strict=True)
    assert (package_root / '__init__.py').is_file()
    assert sys.dont_write_bytecode and not any(
        name == 'supervised_nmf' or name.startswith('supervised_nmf.') for name in sys.modules)
    sys.path.insert(0, str(package_root.parent))
    result = dict(status='failed', case=args.case, package_root=str(package_root),
                  executable=sys.executable, python=sys.version)
    try:
        result['evidence'] = run_case(args.case, output)
        origins = {}
        for name, module in tuple(sys.modules.items()):
            if name == 'supervised_nmf' or name.startswith('supervised_nmf.'):
                path = Path(module.__file__).resolve()
                origins[name] = path.relative_to(package_root).as_posix()
        assert origins['supervised_nmf'] == '__init__.py'
        result.update(status='passed', origins=origins)
    except BaseException:
        result['error'] = traceback.format_exc()
    with (output / 'RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(status=result['status'], case=args.case)), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--case', choices=[*CASES, 'pickle_child'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pickle', type=Path)
    ARGS = parser.parse_args()
    sys.exit(main(ARGS))
