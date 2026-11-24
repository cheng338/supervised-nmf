"""A serial KDTree query still configures a backend and asks it for a batch size."""

import json
from pathlib import Path
import sys
import warnings

if __name__ == "__main__":
    sys.path[:] = [str(Path(__file__).resolve().parent), *json.loads(sys.argv[1])]

import pytest


HOOKS = (
    "ThreadingBackend.configure", "ThreadingBackend.effective_n_jobs",
    "SequentialBackend.__init__", "SequentialBackend.configure",
    "SequentialBackend.effective_n_jobs", "SequentialBackend.compute_batch_size",
    "joblib_parallel.memstr_to_bytes", "sklearn_parallel.get_config", "sklearn_parallel.config_context",
)


def _run_hook(hook, directory):
    from joblib import _parallel_backends, parallel as joblib_parallel
    from sklearn.utils import parallel as sklearn_parallel
    from test_neighbor_tree_delayed import _measure, _phases
    from threadpoolctl import threadpool_limits

    if hook == "stock":
        with threadpool_limits(1):
            return _measure(directory, [])

    if hook in ("ThreadingBackend", "LokyBackend", "gen_even_slices"):
        # The public Joblib lookup can still resolve the registered factory
        # when its optional module alias is absent.
        records = {}
        assert "supervised_nmf" not in sys.modules
        owner = joblib_parallel
        if hook == "gen_even_slices":
            from sklearn import utils
            from sklearn.neighbors import _base

            assert _base.gen_even_slices is utils.gen_even_slices
            owner = utils
        original = getattr(owner, hook)
        with threadpool_limits(1):
            for phase in ("before_import", "stock", "after_capture", "restored"):
                if phase in ("before_import", "after_capture"):
                    delattr(owner, hook)
                try:
                    records[phase] = _measure(directory / phase, [])
                finally:
                    setattr(owner, hook, original)
        return records

    class_name, method = hook.split(".")
    events = []

    def observed():
        events.append(hook)
        warnings.warn(hook, UserWarning)
        if len(events) % 2 == 0:
            raise RuntimeError(hook)

    modules = {"joblib_parallel": joblib_parallel, "sklearn_parallel": sklearn_parallel}
    if class_name in modules:
        owner = modules[class_name]
        original = vars(owner)[method]

        def selected(*args, **kwargs):
            observed()
            return original(*args, **kwargs)
    else:
        receiver = getattr(_parallel_backends, class_name)
        owner = next(cls for cls in receiver.__mro__ if method in vars(cls))
        original = vars(owner)[method]

        def selected(self, *args, **kwargs):
            if type(self) is receiver:
                observed()
            return original(self, *args, **kwargs)

    return _phases(directory, events, method, (owner,), selected, caller_alias=False)


@pytest.fixture(scope="module")
def stock_dispatch(tmp_path_factory):
    from tests.test_neighbor_tree_delayed import _collect

    return _collect(Path(__file__).resolve(), "stock", tmp_path_factory.mktemp("stock-dispatch"))


@pytest.mark.parametrize("hook", HOOKS)
def test_selected_serial_backend_calls(hook, stock_dispatch, tmp_path):
    from tests.test_neighbor_tree_delayed import _collect

    phases = _collect(Path(__file__).resolve(), hook, tmp_path)
    for phase, record in phases.items():
        native = record["native"]
        assert record["actual"] == native, phase
        if phase in ("before_import", "after_capture"):
            assert record["queried"] == [False, False]
            assert native["events"] == [hook] * 4
            assert native["probability_events"] == [hook] * 2
            expected_warning = [{"category": "UserWarning", "message": hook}]
            for rows in (native["scores"], native["probabilities"]):
                assert all("value" in row and row["warnings"] == expected_warning for row in rows[::2])
                assert all(row == {"error_type": "RuntimeError", "error": hook, "warnings": expected_warning}
                           for row in rows[1::2])
    assert phases["restored"] == phases["stock"] == stock_dispatch


@pytest.mark.parametrize("alias", ["ThreadingBackend", "LokyBackend", "gen_even_slices"])
def test_missing_backend_alias_preserves_public_prediction(alias, stock_dispatch, tmp_path):
    from tests.test_neighbor_tree_delayed import _collect

    phases = _collect(Path(__file__).resolve(), alias, tmp_path)
    for phase, record in phases.items():
        assert record["actual"] == record["native"] == stock_dispatch["native"]
        if phase in ("before_import", "after_capture"):
            assert record["queried"] == [False, False]
    assert phases["restored"] == phases["stock"] == stock_dispatch


if __name__ == "__main__":
    print(json.dumps(_run_hook(sys.argv[2], Path(sys.argv[3]))))
