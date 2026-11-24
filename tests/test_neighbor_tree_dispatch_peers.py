"""Selected calls through sklearn tree dispatch and its Joblib base class."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

if __name__ == "__main__":
    sys.path[:] = [str(Path(__file__).resolve().parent), *json.loads(sys.argv[1])]

import pytest


def _peer(name, directory):
    import joblib
    from sklearn import utils
    from sklearn.utils import _chunking, _param_validation, parallel
    from test_neighbor_tree_delayed import _phases

    events = []
    if name == "executed_task":
        wrapper = parallel._FuncWrapper
        original_call = wrapper.__call__

        def observed_task(self, *args, **kwargs):
            if getattr(self.function, "__name__", None) == "query":
                X, k, return_distance = args
                events.append([k, return_distance])
                if k == 3:
                    raise RuntimeError("selected task width 3")
                return original_call(self, X[::-1], k, return_distance)
            return original_call(self, *args, **kwargs)

        return _phases(directory, events, "__call__", (wrapper,), observed_task, caller_alias=False)

    if name == "logger_init":
        logger = joblib.parallel.Logger
        original_init = logger.__init__

        def observed_init(self, *args, **kwargs):
            if isinstance(self, parallel.Parallel):
                events.append("construct")
                if len(events) % 2 == 0:
                    raise RuntimeError("second Parallel construction")
            return original_init(self, *args, **kwargs)

        return _phases(directory, events, "__init__", (logger,), observed_init, caller_alias=False)

    if name in ("Parallel", "joblib_call"):
        inherited = joblib.Parallel.__call__ if name == "joblib_call" else parallel.Parallel.__call__

        def observed_call(self, iterable):
            def observe():
                for function, args, kwargs in iterable:
                    if getattr(function, "__name__", None) == "query":
                        _, k, return_distance = args
                        events.append([k, return_distance])
                        if k == 3:
                            raise RuntimeError("selected Parallel width 3")
                    yield function, args, kwargs

            return inherited(self, observe())

        if name == "joblib_call":
            return _phases(directory, events, "__call__", (joblib.Parallel,), observed_call, caller_alias=False)

        class Replacement(parallel.Parallel):
            __call__ = observed_call

        return _phases(directory, events, name, (parallel,), Replacement)

    original = utils.gen_even_slices
    assert _chunking.gen_even_slices is original

    def selected_slice(marker):
        events.append(marker)
        if len(events) % 2 == 0:
            raise RuntimeError("second selected slice request")

    if name == "gen_even_slices":
        def replacement(n, packs, **kwargs):
            selected_slice([n, packs])
            return original(n, packs, **kwargs)
    else:
        assert name == "redecorated_slicer"
        raw = original.__wrapped__
        constraints = raw._skl_parameter_constraints

        class ObservedConstraints(dict):
            def __contains__(self, parameter):
                if parameter == "n":
                    selected_slice("validate n")
                return super().__contains__(parameter)

        # Reuse sklearn's actual decorator and generator with a different
        # validation callback, then restore the metadata the decorator sets.
        with patch.object(raw, "_skl_parameter_constraints", constraints):
            replacement = _param_validation.validate_params(
                ObservedConstraints(constraints), prefer_skip_nested_validation=True,
            )(raw)
    return _phases(directory, events, "gen_even_slices", (_chunking, utils), replacement)


@pytest.mark.parametrize("name", [
    "Parallel", "gen_even_slices", "redecorated_slicer", "joblib_call", "executed_task", "logger_init",
])
def test_selected_tree_dispatch_peer(name, tmp_path):
    from tests.test_neighbor_tree_delayed import _check, _collect

    _check(_collect(Path(__file__).resolve(), name, tmp_path), name)


if __name__ == "__main__":
    print(json.dumps(_peer(sys.argv[2], Path(sys.argv[3]))))
