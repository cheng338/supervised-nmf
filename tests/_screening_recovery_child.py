"""One bounded subprocess phase for persistent-statistics recovery tests."""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from unittest.mock import patch
import zipfile


def main():
    source, root, mode, order, case = sys.argv[1:]
    source, root = Path(source).resolve(), Path(root).resolve()
    sys.path[:0] = [str(source), str(Path(__file__).resolve().parents[1])]
    import numpy as np
    from sklearn import config_context
    from threadpoolctl import threadpool_limits
    from supervised_nmf import NMFClassifier, study
    from supervised_nmf.screening import BinaryScreening
    from tests._screening_recovery_support import assert_accounting, event_paths, metadata, receipts, science, stable_bytes

    assert Path(study.__file__).resolve() == source/'supervised_nmf/study.py'
    assert order in ('representation', 'fold')
    assert mode in ('prepare', 'resume', 'repair_interrupt', 'completed', 'missing_authority', 'blocked')
    root.mkdir(exist_ok=True)
    directory = root/case
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(study.__file__).parent.glob('*.py'))}
    random = np.random.RandomState(43)
    y = np.tile([0, 1], 60)
    X = random.uniform(.1, 1., (120, 24)).astype(np.float32)
    X[:, :8] += (y*.2)[:, None]
    grid = {'n_components': [2, 3], 'score_threshold': [0], 'n_neighbors': [1, 3]}
    original_fit = NMFClassifier._fit_with_screening

    @contextmanager
    def observed():
        assert sys.getprofile() is None
        result = {'statistics': 0, 'representations': 0, 'payload_opens': 0}
        codes = {BinaryScreening.fit.__code__: 'statistics', original_fit.__code__: 'representations'}
        archive = zipfile.ZipFile.__init__.__code__
        def observe(frame, event, arg):
            if event != 'call':
                return
            label = codes.get(frame.f_code)
            if label:
                result[label] += 1
            elif frame.f_code is archive:
                value = frame.f_locals.get('file')
                if not isinstance(value, (str, Path)):
                    value = getattr(value, 'name', None)
                if isinstance(value, (str, Path)) and Path(value).name.startswith('payload-'):
                    result['payload_opens'] += 1
        sys.setprofile(observe)
        try:
            yield result
        finally:
            sys.setprofile(None)

    def call(path, *, early=True, interrupt=None):
        completed = []
        def fit(model, *args, **kwargs):
            result = original_fit(model, *args, **kwargs)
            completed.append(model.n_components)
            if len(completed) == interrupt:
                raise KeyboardInterrupt('successful representation before receipt')
            return result
        context = patch.object(NMFClassifier, '_fit_with_screening', fit) if interrupt else nullcontext()
        error, report = None, None
        with context, observed() as counts:
            try:
                report = study.calibration_knn_grid(NMFClassifier(solver='spa', random_state=43),
                    grid, X, y, cv=3, seed=43, threads=1, scoring='balanced_accuracy',
                    execution_order=order, checkpoint_dir=path,
                    screening_statistics_recovery='early' if early else 'off')
            except BaseException as caught:
                error = {'type': type(caught).__name__, 'message': str(caught)}
        assert NMFClassifier._fit_with_screening is original_fit
        return {'report': report, 'error': error, 'counts': counts, 'completed_fit_ranks': completed}

    with config_context(assume_finite=False), threadpool_limits(limits=1):
        np.seterr(divide='warn', over='warn', under='ignore', invalid='warn')
        np.seterrcall(None)
        result = {'mode': mode, 'case': case, 'order': order, 'source_sha256': pins,
                  'pid': os.getpid(), 'process_token': uuid.uuid4().hex}
        if mode == 'prepare':
            assert case == 'template' and not directory.exists()
            reference = call(root/'reference', early=False)
            assert reference['error'] is None and all(row['valid'] for row in reference['report']['candidates'])
            partial = call(directory, interrupt=6)
            assert partial['error']['type'] == 'KeyboardInterrupt'
            assert partial['error']['message'] == 'successful representation before receipt'
            assert partial['counts']['statistics'] == 3 and partial['counts']['representations'] == 6
            rows = receipts(directory)
            attempts = [study._read_receipt(path) for path in (directory/'attempts').glob('*.json')]
            assert len(rows) == 5 and len(attempts) == 6 and len(event_paths(directory)) == 3
            schedule = ([(rank, fold) for fold in range(3) for rank in (2, 3)] if order == 'fold'
                        else [(rank, fold) for rank in (2, 3) for fold in range(3)])
            assert partial['completed_fit_ranks'] == [rank for rank, _ in schedule]
            assert {(row['parameters']['n_components'], row['fold']) for row in rows} == set(schedule[:-1])
            missing = next(row for row in attempts if row['state'] == 'interrupted')
            target = next(row for row in rows if row['parameters']['n_components'] == 2 and row['fold'] == 2)
            first = next(row for row in rows if row['parameters']['n_components'] == 2 and row['fold'] == 0)
            result.update(oracle_science=science(reference['report']), before=stable_bytes(directory),
                          target_event=target['screening_statistics_event_id'], target_key=target['key'],
                          first_key=first['key'], missing_key=missing['key'], call=partial)
        elif mode == 'blocked':
            actual = call(directory)
            assert actual['error']['type'] == 'RuntimeError' and 'Another process is writing' in actual['error']['message']
            assert actual['counts']['statistics'] == actual['counts']['representations'] == 0
            result['call'] = actual
        else:
            prepared = json.loads((root/'prepare-template.json').read_bytes())
            assert prepared['source_sha256'] == pins
            before = stable_bytes(directory)
            rows_before = receipts(directory)
            events_before = {path.name: path.read_bytes().hex() for path in event_paths(directory)}
            actual = call(directory, interrupt=1 if mode == 'repair_interrupt' else None)
            after = stable_bytes(directory)
            assert all(after.get(name) == pin for name, pin in before.items() if name != 'report.json')
            assert all((directory/'statistics-events'/name).read_bytes().hex() == data for name, data in events_before.items())
            result.update(before=before, after=after, call=actual)
            if mode == 'missing_authority':
                assert actual['error']['type'] == 'StatisticsRecoveryError'
                assert actual['report'] is None and after == before
                assert actual['counts'] == {'statistics': 0, 'representations': 0, 'payload_opens': 0}
            elif mode == 'repair_interrupt':
                assert actual['error']['type'] == 'KeyboardInterrupt'
                assert actual['counts']['statistics'] == actual['counts']['representations'] == 1
                assert receipts(directory) == rows_before and len(rows_before) == 5
                new = [path for path in event_paths(directory) if path.name not in events_before]
                assert len(new) == 1 and len(event_paths(directory)) == 4
                value = metadata(new[0])
                assert value['origin_record_key'] == prepared['missing_key']
                assert not any(row['screening_statistics_event_id'] == value['event_id'] for row in rows_before)
                result['replacement_event'] = value['event_id']
            else:
                assert actual['error'] is None and science(actual['report']) == prepared['oracle_science']
                report = actual['report']
                assert_accounting(report)
                assert actual['counts']['statistics'] == 0
                assert actual['counts']['representations'] == (0 if mode == 'completed' else 1)
                assert len(receipts(directory)) == 6
                if mode == 'completed':
                    assert actual['counts']['payload_opens'] == 0
                    assert {name: pin for name, pin in after.items() if name != 'report.json'} == {
                        name: pin for name, pin in before.items() if name != 'report.json'}
                    previous = json.loads((root/('resume-'+case+'.json')).read_bytes())['call']['report']
                    assert report['representation_folds'] == previous['representation_folds']
                    assert report['compute_seconds'] == previous['compute_seconds']
                    for field in ('events', 'durable_statistics_seconds', 'union_statistics_seconds', 'union_tuning_seconds'):
                        assert report['screening_statistics_recovery'][field] == previous['screening_statistics_recovery'][field]
                else:
                    assert report['new_representation_fits'] == 1 and report['reused_representation_folds'] == 5
                    assert report['screening_statistics_recovery']['current_call']['restored_events'] == 1
                ledger = report['screening_statistics_recovery']
                assert ledger['orphan_event_count'] == 0
                if case == 'repair':
                    repaired = json.loads((root/'repair_interrupt-repair.json').read_bytes())
                    assert repaired['source_sha256'] == pins
                    assert ledger['durable_event_count'] == 4
                    old = next(row for row in ledger['events'] if row['event_id'] == prepared['target_event'])
                    new = next(row for row in ledger['events'] if row['event_id'] == repaired['replacement_event'])
                    assert old['consumers'] == [prepared['target_key']]
                    assert new['consumers'] == [prepared['missing_key']]
                else:
                    assert case == 'healthy' and ledger['durable_event_count'] == 3
        assert pins == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(study.__file__).parent.glob('*.py'))}
        result['passed'] = True
        with (root/(mode+'-'+case+'.json')).open('x', encoding='utf8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
