"""Keep current regressions distinct from archived study qualification.

Collection scopes historical CLI thread-environment assignments. Test bodies
own their state; restoration cannot rewind initialized native libraries.
"""
import pytest

from tests._thread_import_state import preserve_thread_environment


def pytest_addoption(parser):
    parser.addoption(
        '--run-archived-qualification', action='store_true', default=False,
        help='Enable archived study qualification; original source/runtime checks and study opt-ins still apply.',
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-archived-qualification'):
        return
    for item in items:
        marker = item.get_closest_marker('archived_qualification')
        if marker is not None:
            reason = marker.kwargs.get('reason', 'Archived study qualification requires its original source/runtime')
            item.add_marker(pytest.mark.skip(
                reason=reason + '; enable explicitly with --run-archived-qualification',
            ))


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if isinstance(collector, pytest.Module):
        with preserve_thread_environment():
            yield
    else:
        yield
