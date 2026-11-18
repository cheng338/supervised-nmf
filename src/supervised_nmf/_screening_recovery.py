"""Durable screening-statistics events for opt-in study recovery.

A committed JSON event is the authority; its immutable NPZ payload is
separate. Metadata-only accounting and full learned-state restore have
distinct checks. No pickle or global cache is used.
"""
import io
import json
import math
import os
from pathlib import Path
import re
import time
import uuid
import zipfile
import hashlib
import numpy as np
from sklearn import get_config
from .screening import BinaryScreening
from ._labels import _validate_class_labels
from .study import StatisticsRecoveryError

ARRAYS = {"classes_", "class_counts_", "scores_", "t_scores_",
          "statistic_offsets_", "statistic_scales_", "scaled_pooled_variances_",
          "scaled_standard_errors_", "support_", "selected_features_", "constant_features_"}


SCALARS = {"n_features_in_", "n_samples_seen_", "n_features_selected_",
           "n_constant_features_", "n_infinite_scores_", "n_infinite_t_scores_",
           "n_f_overflow_", "n_classwise_constant_features_"}


VECTORS = ARRAYS - {"classes_", "class_counts_", "selected_features_", "constant_features_"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _plain_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _packed_state(screen):
    state = vars(screen)
    fitted = {k for k in state if k.endswith("_") and not k.startswith("__")}
    if fitted != ARRAYS | SCALARS:
        raise ValueError("Unsupported fitted-state fields")
    arrays, descriptors = {}, {}
    for name in sorted(ARRAYS):
        value = np.asarray(state[name])
        descriptor = {"shape": list(value.shape), "dtype": value.dtype.str}
        if value.dtype.hasobject:
            # Fixed-width Unicode would lose trailing NULs. ASCII hex preserves
            # exact Python string values, including lone surrogate code points.
            if name != "classes_" or not all(type(x) is str for x in value):
                raise ValueError("Unsupported object-label storage")
            value = np.asarray([item.encode("utf-8", errors="surrogatepass").hex() for item in value], dtype=str)
            descriptor["object_string_encoding"] = "utf8-surrogatepass-hex-v1"
        arrays[name] = value
        descriptor["stored_dtype"] = value.dtype.str
        descriptors[name] = descriptor
    scalars = {}
    for name in SCALARS:
        if isinstance(state[name], (bool, np.bool_)) or not isinstance(state[name], (int, np.integer)):
            raise ValueError("Unsupported scalar state")
        scalars[name] = int(state[name])
    return arrays, {"arrays": descriptors, "scalars": scalars}


def _restored_state(arrays, metadata, p, n):
    if not isinstance(metadata, dict) or set(arrays) != ARRAYS or set(metadata) != {"arrays", "scalars"}:
        raise ValueError("Invalid fitted-state fields")
    if (not isinstance(metadata["arrays"], dict) or not isinstance(metadata["scalars"], dict)
            or set(metadata["arrays"]) != ARRAYS or set(metadata["scalars"]) != SCALARS):
        raise ValueError("Invalid fitted-state metadata")
    scalars = metadata["scalars"]
    if any(type(x) is not int for x in scalars.values()):
        raise ValueError("Invalid fitted-state scalar")
    if scalars["n_features_in_"] != p or scalars["n_samples_seen_"] != n:
        raise ValueError("Training shape mismatch")
    if any(value < 0 for value in scalars.values()):
        raise ValueError("Invalid negative fitted-state scalar")
    result = BinaryScreening(screening="absolute", score_threshold=0.0)
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise ValueError("Expected an array value in fitted state")
        descriptor = metadata["arrays"][name]
        if not isinstance(descriptor, dict) or set(descriptor) - {"shape", "dtype", "stored_dtype", "object_string_encoding"}:
            raise ValueError("Invalid array descriptor schema")
        if value.dtype.hasobject or list(value.shape) != descriptor["shape"] or value.dtype.str != descriptor["stored_dtype"]:
            raise ValueError("Invalid array descriptor")
        if value.ndim != 1:
            raise ValueError("Expected one-dimensional fitted state")
        if name in VECTORS and value.size != p:
            raise ValueError("Feature dimension mismatch")
        if name in {"classes_", "class_counts_"} and value.size != 2:
            raise ValueError("Class dimension mismatch")
        if name in {"selected_features_", "constant_features_"}:
            if value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value >= p):
                raise ValueError("Invalid feature indices")
        if name == "support_" and value.dtype.kind != "b":
            raise ValueError("Invalid support dtype")
        if name not in {"classes_", "support_", "class_counts_", "selected_features_", "constant_features_"} and value.dtype != np.dtype("float64"):
            raise ValueError("Invalid numeric statistics dtype")
        if "object_string_encoding" in descriptor:
            if (name != "classes_" or value.dtype.kind != "U" or descriptor["dtype"] != "|O"
                    or descriptor["object_string_encoding"] != "utf8-surrogatepass-hex-v1"):
                raise ValueError("Invalid object-string encoding")
            decoded = []
            for encoded in value:
                raw = bytes.fromhex(str(encoded))
                if raw.hex() != encoded:
                    raise ValueError("Invalid canonical object-string hex")
                decoded.append(raw.decode("utf-8", errors="surrogatepass"))
            value = np.asarray(decoded, dtype=object)
        elif value.dtype.str != descriptor["dtype"]:
            raise ValueError("Original dtype mismatch")
        setattr(result, name, value)
    if result.class_counts_.dtype.kind not in "iu" or np.any(result.class_counts_ < 2) or int(result.class_counts_.sum()) != n:
        raise ValueError("Invalid class counts")
    if scalars["n_features_selected_"] != len(result.selected_features_) or scalars["n_constant_features_"] != len(result.constant_features_):
        raise ValueError("Invalid feature counts")
    if not np.array_equal(np.flatnonzero(result.support_), result.selected_features_):
        raise ValueError("Inconsistent selected features")
    if not np.array_equal(np.unique(result.constant_features_), result.constant_features_):
        raise ValueError("Invalid constant feature ordering")
    if np.any(result.support_[result.constant_features_]) or np.any(result.t_scores_[result.constant_features_] != 0):
        raise ValueError("Inconsistent constant feature state")
    for name in {"statistic_offsets_", "statistic_scales_", "scaled_pooled_variances_", "scaled_standard_errors_"}:
        value = getattr(result, name)
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Invalid finite nonnegative statistics")
    if np.isnan(result.t_scores_).any() or np.isnan(result.scores_).any() or np.any(result.scores_ < 0):
        raise ValueError("Invalid t/F scores")
    if (scalars["n_infinite_scores_"] != int(np.isinf(result.scores_).sum())
            or scalars["n_infinite_t_scores_"] != int(np.isinf(result.t_scores_).sum())
            or scalars["n_f_overflow_"] != int((np.isfinite(result.t_scores_) & np.isinf(result.scores_)).sum())
            or scalars["n_classwise_constant_features_"] != scalars["n_infinite_t_scores_"]):
        raise ValueError("Inconsistent fitted-state diagnostics")
    for name, value in scalars.items():
        setattr(result, name, value)
    return result

def _event_hook(stage, details):
    """Private interruption boundary; no-op ordinarily."""


def _policy_reason():
    if np.geterr() != {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}:
        return 'nondefault_numpy_error_policy'
    if np.geterrcall() is not None:
        return 'numpy_error_callback'
    if get_config()['assume_finite'] is not False:
        return 'assume_finite_enabled'
    return None


def _check_state_metadata(state):
    if (type(state) is not dict or set(state) != {'arrays', 'scalars'}
            or type(state['arrays']) is not dict or set(state['arrays']) != ARRAYS
            or type(state['scalars']) is not dict or set(state['scalars']) != SCALARS):
        raise ValueError('Invalid fitted-state metadata fields')
    scalars = state['scalars']
    if any(type(value) is not int or value < 0 for value in scalars.values()):
        raise ValueError('Invalid fitted-state scalar metadata')
    p, n = scalars['n_features_in_'], scalars['n_samples_seen_']
    if (p < 1 or n < 4 or not 1 <= scalars['n_features_selected_'] <= p
            or scalars['n_features_selected_'] + scalars['n_constant_features_'] != p
            or scalars['n_classwise_constant_features_'] != scalars['n_infinite_t_scores_']
            or scalars['n_infinite_t_scores_'] + scalars['n_f_overflow_'] != scalars['n_infinite_scores_']
            or scalars['n_infinite_scores_'] > p):
        raise ValueError('Inconsistent fitted-state counter metadata')
    for name, descriptor in state['arrays'].items():
        if (type(descriptor) is not dict
                or not {'shape', 'dtype', 'stored_dtype'} <= set(descriptor)
                or set(descriptor) - {'shape', 'dtype', 'stored_dtype', 'object_string_encoding'}):
            raise ValueError('Invalid fitted-state array descriptor')
        expected = (2 if name in {'classes_', 'class_counts_'} else
                    scalars['n_features_selected_'] if name == 'selected_features_' else
                    scalars['n_constant_features_'] if name == 'constant_features_' else p)
        if descriptor['shape'] != [expected]:
            raise ValueError('Inconsistent fitted-state descriptor shape')
        if type(descriptor['dtype']) is not str or type(descriptor['stored_dtype']) is not str:
            raise ValueError('Invalid fitted-state dtype descriptor')
        dtype, stored = np.dtype(descriptor['dtype']), np.dtype(descriptor['stored_dtype'])
        if stored.hasobject:
            raise ValueError('Object arrays cannot be stored as numeric event members')
        encoding = descriptor.get('object_string_encoding')
        if encoding is not None:
            if (name != 'classes_' or dtype.kind != 'O' or stored.kind != 'U'
                    or encoding != 'utf8-surrogatepass-hex-v1'):
                raise ValueError('Invalid event label encoding')
        elif dtype != stored:
            raise ValueError('Unexpected stored/original dtype mismatch')
        if name in VECTORS - {'support_'} and dtype != np.dtype('float64'):
            raise ValueError('Invalid numeric statistic dtype metadata')
        if name == 'support_' and dtype.kind != 'b':
            raise ValueError('Invalid support dtype metadata')
        if name in {'class_counts_', 'selected_features_', 'constant_features_'} and dtype.kind not in 'iu':
            raise ValueError('Invalid integer array dtype metadata')


class StatisticsEventStore:
    """One invocation's owned event index, under the caller's directory lock.

    Metadata is independently integrity checked for accounting. Learned-array
    members are read only for an eligible missing fold, and then checked in full.
    Invalid committed metadata is fatal. Valid metadata with unavailable
    learned payload preserves historical cost and permits recomputation.
    Payloads and temporary files without a committed JSON are not events.
    """

    def __init__(self, directory, identity, runtime, estimator, candidates,
                 threads, runtime_guard, *, layout_eligible, fold_shapes):
        self.root = Path(directory) / 'statistics-events'
        self.identity = identity
        self.runtime = runtime
        self.estimator, self.candidates = estimator, candidates
        self.threads, self.runtime_guard = threads, runtime_guard
        self.layout_eligible = layout_eligible
        self.fold_shapes = tuple(fold_shapes)
        self.events = None
        self.current = dict(actual_statistics_fits=0, restored_events=0,
                            published_events=0, metadata_read_seconds=0.0,
                            array_read_seconds=0.0, write_seconds=0.0,
                            guard_seconds=0.0, bypasses=[], publication_failures=[], unusable_payload_events=[])

    def eligibility_reason(self):
        return 'unsupported_layout_or_parameters' if not self.layout_eligible else _policy_reason()

    def _read_metadata(self, path):
        try:
            raw = path.read_bytes()
            envelope = json.loads(raw)
            if type(envelope) is not dict or set(envelope) != {'payload', 'sha256'}:
                raise ValueError('Invalid event envelope')
            value = envelope['payload']
            if type(value) is not dict or envelope['sha256'] != sha(_plain_json(value)):
                raise ValueError('Event metadata checksum mismatch')
            required = {'schema', 'event_id', 'identity', 'fold', 'origin_record_key',
                        'origin_seconds', 'state', 'arrays', 'policy'}
            if set(value) != required or value['schema'] != 1:
                raise ValueError('Invalid event metadata schema')
            event_id = value['event_id']
            if (type(event_id) is not str or re.fullmatch('[0-9a-f]{32}', event_id) is None
                    or path.name != 'event-' + event_id + '.json'):
                raise ValueError('Invalid event identifier')
            if value['identity'] != self.identity or value['policy'] != 'stock-default-silent-v1':
                raise ValueError('Statistics event belongs to a different identity or policy')
            if type(value['fold']) is not int or value['fold'] < 0:
                raise ValueError('Invalid event fold')
            if (type(value['origin_record_key']) is not str
                    or re.fullmatch('[0-9a-f]{64}', value['origin_record_key']) is None):
                raise ValueError('Invalid originating fold key')
            seconds = value['origin_seconds']
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid event duration')
            if type(value['arrays']) is not dict or set(value['arrays']) != ARRAYS:
                raise ValueError('Invalid event array inventory')
            for name, descriptor in value['arrays'].items():
                if (type(descriptor) is not dict or set(descriptor) != {'member', 'bytes', 'sha256'}
                        or descriptor['member'] != name + '.npy'
                        or type(descriptor['bytes']) is not int or descriptor['bytes'] < 0
                        or type(descriptor['sha256']) is not str
                        or re.fullmatch('[0-9a-f]{64}', descriptor['sha256']) is None):
                    raise ValueError('Invalid event array descriptor')
            _check_state_metadata(value['state'])
            if value['fold'] >= len(self.fold_shapes):
                raise ValueError('Statistics event fold is outside this study')
            p, n = self.fold_shapes[value['fold']]
            scalars = value['state']['scalars']
            if scalars['n_features_in_'] != p or scalars['n_samples_seen_'] != n:
                raise ValueError('Statistics event training shape differs from its fold')
            return value
        except Exception as error:
            raise StatisticsRecoveryError(f'Invalid committed statistics event {path.name}: {error}') from error

    @staticmethod
    def committed_paths(directory):
        # Path.glob suppresses enumeration OSError on supported Python versions;
        # an inaccessible existing ledger must not masquerade as an empty one.
        try:
            entries = os.scandir(directory)
        except FileNotFoundError:
            return []
        except OSError as error:
            raise StatisticsRecoveryError('Cannot enumerate committed statistics events') from error
        try:
            with entries:
                return sorted(Path(entry.path) for entry in entries
                              if Path(entry.name).match('event-*.json'))
        except OSError as error:
            raise StatisticsRecoveryError('Cannot enumerate committed statistics events') from error

    def _index(self):
        if self.events is None:
            start = time.perf_counter()
            try:
                events = {}
                for path in self.committed_paths(self.root):
                    if path.is_symlink() or not path.is_file():
                        raise StatisticsRecoveryError('Unexpected linked/non-file statistics event')
                    event = self._read_metadata(path)
                    if event['event_id'] in events:
                        raise StatisticsRecoveryError('Duplicate statistics event identifier')
                    events[event['event_id']] = event
                self.events = events
            finally:
                self.current['metadata_read_seconds'] += time.perf_counter() - start
        return self.events

    def load(self, fold, record, p, n):
        reason = self.eligibility_reason()
        if reason is not None:
            self.current['bypasses'].append({'fold': fold, 'reason': reason})
            return None
        for event in self._index().values():
            if event['fold'] != fold:
                continue
            start = time.perf_counter()
            path = self.root / ('event-' + event['event_id'] + '.json')
            payload = self.root / ('payload-' + event['event_id'] + '.npz')
            try:
                # Metadata is historical accounting authority. Its loss or
                # invalidity never becomes a learned-payload cache miss.
                if self._read_metadata(path) != event:
                    raise StatisticsRecoveryError('Statistics event changed after indexing')
                arrays = {}
                try:
                    if payload.is_symlink() or not payload.is_file():
                        raise ValueError('Unexpected linked/non-file statistics payload')
                    with zipfile.ZipFile(payload, 'r') as archive:
                        entries = archive.infolist()
                        names = [row.filename for row in entries]
                        expected = {name + '.npy' for name in ARRAYS}
                        if len(names) != len(expected) or set(names) != expected:
                            raise ValueError('Unexpected or missing learned-state members')
                        if any(row.compress_type != zipfile.ZIP_STORED or row.flag_bits & 1 for row in entries):
                            raise ValueError('Unsupported learned-state member encoding')
                        for name, descriptor in event['arrays'].items():
                            if archive.getinfo(descriptor['member']).file_size != descriptor['bytes']:
                                raise ValueError('Statistics member size mismatch: ' + name)
                            raw = archive.read(descriptor['member'])
                            if len(raw) != descriptor['bytes'] or sha(raw) != descriptor['sha256']:
                                raise ValueError('Statistics member checksum mismatch: ' + name)
                            value = np.load(io.BytesIO(raw), allow_pickle=False)
                            if type(value) is not np.ndarray:
                                if hasattr(value, 'close'):
                                    value.close()
                                raise ValueError('Statistics member is not an NPY ndarray')
                            arrays[name] = value
                    restored = _restored_state(arrays, event['state'], p, n)
                    _validate_class_labels(restored.classes_)
                except MemoryError as error:
                    raise StatisticsRecoveryError('Cannot allocate committed statistics payload') from error
                except Exception as error:
                    # Recheck authority before treating only its arrays as
                    # unavailable; never erase or rewrite old event evidence.
                    if self._read_metadata(path) != event:
                        raise StatisticsRecoveryError('Statistics metadata changed during failed payload read') from error
                    arrays.clear()
                    self.current['unusable_payload_events'].append(dict(
                        event_id=event['event_id'], fold=fold,
                        error_type=type(error).__name__, error=str(error)))
                    continue
                self.current['restored_events'] += 1
                try:
                    _event_hook('after_statistics_loaded', {'event_id': event['event_id'], 'fold': fold})
                except Exception as error:
                    raise StatisticsRecoveryError('Statistics restore observer failed') from error
                return restored, event
            finally:
                self.current['array_read_seconds'] += time.perf_counter() - start
        return None

    def publish(self, fold, base, seconds, record, *, silent):
        reason = self.eligibility_reason()
        if reason is not None or not silent:
            self.current['bypasses'].append({'fold': fold, 'reason': reason or 'statistics_emitted_warning'})
            return None
        self._index()
        start = time.perf_counter()
        guard_before = self.current['guard_seconds']
        event_id = uuid.uuid4().hex
        details = {'event_id': event_id, 'fold': fold}
        temporary_payload = self.root / ('.payload-' + event_id + '.tmp')
        payload = self.root / ('payload-' + event_id + '.npz')
        temporary = self.root / ('.event-' + event_id + '.tmp')
        final = self.root / ('event-' + event_id + '.json')
        committed = False
        try:
            try:
                arrays, state = _packed_state(base)
                _check_state_metadata(state)
                encoded_arrays = {}
                for name in sorted(arrays):
                    buffer = io.BytesIO()
                    np.save(buffer, arrays[name], allow_pickle=False)
                    encoded_arrays[name] = buffer.getvalue()
            except (ValueError, TypeError) as error:
                self.current['bypasses'].append(dict(fold=fold, reason='codec_unsupported',
                                                     error_type=type(error).__name__, error=str(error)))
                return None
            value = dict(schema=1, event_id=event_id, identity=self.identity, fold=fold,
                         origin_record_key=record['key'], origin_seconds=seconds,
                         state=state, arrays={}, policy='stock-default-silent-v1')
            self.root.mkdir(parents=True, exist_ok=True)
            _event_hook('before_payload_write', details)
            with temporary_payload.open('xb') as stream:
                with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
                    for name, raw in encoded_arrays.items():
                        member = name + '.npy'
                        value['arrays'][name] = {'member': member, 'bytes': len(raw), 'sha256': sha(raw)}
                        archive.writestr(member, raw)
            _event_hook('after_payload_write', details)
            _event_hook('before_metadata_write', details)
            with temporary.open('xb') as stream:
                stream.write(_plain_json({
                    'payload': value, 'sha256': sha(_plain_json(value))}))
            _event_hook('after_metadata_write', details)
            _event_hook('before_payload_commit', details)
            if payload.exists():
                raise StatisticsRecoveryError('Statistics payload identifier collision')
            temporary_payload.replace(payload)
            _event_hook('after_payload_commit', details)
            # The final source/runtime check remains after all serialization
            # and precommit hooks. A payload without JSON is uncommitted.
            _event_hook('before_event_commit', details)
            guard_start = time.perf_counter()
            try:
                try:
                    current = self.runtime_guard(self.estimator, self.threads, self.candidates)
                except Exception as error:
                    raise StatisticsRecoveryError('Cannot verify source/runtime before statistics event commit') from error
                if current != self.runtime:
                    raise StatisticsRecoveryError('Source or numerical runtime changed before statistics event commit')
            finally:
                self.current['guard_seconds'] += time.perf_counter() - guard_start
            if self.eligibility_reason() is not None:
                self.current['bypasses'].append({'fold': fold, 'reason': 'policy_changed_before_commit'})
                return None
            if final.exists():
                raise StatisticsRecoveryError('Statistics event identifier collision')
            temporary.replace(final)
            committed = True
            self.events[event_id] = value
            self.current['published_events'] += 1
            try:
                _event_hook('after_event_commit', details)
            except Exception as error:
                raise StatisticsRecoveryError('Statistics event committed before a later publication observer failed') from error
            return value
        except OSError as error:
            if committed:
                raise StatisticsRecoveryError('Statistics event committed before a later publication observer failed') from error
            # Optional publication failure does not invalidate already-computed
            # statistics; no final artifact is trusted until replacement succeeds.
            self.current['publication_failures'].append(dict(fold=fold, error_type=type(error).__name__, error=str(error)))
            return None
        finally:
            self.current['write_seconds'] += time.perf_counter() - start - (self.current['guard_seconds'] - guard_before)
            # Uncommitted temporary/orphan payload evidence is retained;
            # only the final JSON replacement commits a durable event.

    def validate_consumers(self, receipts):
        # Newly published events arrive in computation order; reloaded ones
        # arrive in filename order. Use one stable order for records and sums.
        events = dict(sorted(self._index().items()))
        consumers = {event_id: [] for event_id in events}
        backed_seconds = 0.0
        backed_fits = 0
        for record in receipts:
            event_id = record.get('screening_statistics_event_id')
            if event_id is None:
                continue
            if (type(event_id) is not str or event_id not in events
                    or type(record.get('fold')) is not int
                    or events[event_id]['fold'] != record['fold']):
                raise StatisticsRecoveryError('Completed fold references a missing or incompatible statistics event')
            event = events[event_id]
            consumers[event_id].append(record['key'])
            seconds = record.get('screening_statistics_seconds')
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                raise StatisticsRecoveryError('Invalid statistics-event consumer duration')
            performed = record.get('screening_statistics_fit_performed')
            reused = record.get('screening_statistics_event_reused')
            if type(reused) is not bool or type(performed) is not bool or reused is performed:
                raise StatisticsRecoveryError('Invalid statistics-event consumer flags')
            if performed:
                if (reused or record['key'] != event['origin_record_key']
                        or seconds != event['origin_seconds']):
                    raise StatisticsRecoveryError('Inconsistent original statistics event cost')
                backed_seconds += seconds
                backed_fits += 1
            elif seconds != 0:
                raise StatisticsRecoveryError('Reused statistics charged as newly performed work')
        return events, consumers, backed_seconds, backed_fits

    def summary(self, receipts, old_costs):
        events, consumers, backed_seconds, backed_fits = self.validate_consumers(receipts)
        durable = sum(event['origin_seconds'] for event in events.values())
        receipt_seconds = old_costs['screening_statistics_seconds']
        union = receipt_seconds - backed_seconds + durable
        return dict(schema=1, requested='early', eligible=self.eligibility_reason() is None,
                    bypass_reason=self.eligibility_reason(),
                    events=[dict(event_id=key, fold=event['fold'], origin_seconds=event['origin_seconds'],
                                 consumers=consumers[key]) for key, event in events.items()],
                    durable_event_count=len(events), durable_statistics_seconds=durable,
                    orphan_event_count=sum(not rows for rows in consumers.values()),
                    receipt_statistics_seconds=receipt_seconds,
                    receipt_event_backed_seconds=backed_seconds,
                    receipt_event_backed_fits=backed_fits,
                    union_statistics_seconds=union,
                    union_tuning_seconds=sum(old_costs.values()) - backed_seconds + durable,
                    current_call=dict(self.current),
                    timing_note='Original fold costs/fit flags remain actual committed receipt work. The union removes receipt-backed event costs once before adding every unique successful durable event, including orphans. Uncommitted attempts are not inferred to be free or included in this ledger. Metadata-only ledger reads do not validate learned arrays until restore. Intact metadata with unusable learned arrays retains its historical cost; a replacement computation creates a separate event and never rewrites that authority.')

    def close(self):
        self.events = None
        self.estimator = self.candidates = self.runtime_guard = self.runtime = None
