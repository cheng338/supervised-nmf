"""The explicitly accepted CN warning multiplicity on a rejected overflow."""
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import experimental_nmf as experimental
from supervised_nmf import _aocl

assert Path(experimental.__file__).resolve() == ROOT / 'scripts/experimental_nmf.py'
assert Path(_aocl.__file__).resolve() == ROOT / 'src/supervised_nmf/_aocl.py'


def state(value):
    return (value.tobytes(), value.shape, value.strides, value.dtype.str,
            value.flags.writeable, value.flags.aligned, value.flags.owndata)


@pytest.mark.parametrize('overflow', ['warn', 'ignore', 'raise'])
def test_rejected_overflow_preserves_explicit_policy_contract(overflow, request):
    a = np.full((13, 9), 1e200, dtype=np.float64, order='C')
    b = np.zeros((13, 9), dtype=np.float64, order='C')
    a.setflags(write=False)
    b.setflags(write=False)
    before = state(a), state(b)
    entries = {'core': 0, 'passive': 0, 'selector': 0}
    products = []
    prior_profile = sys.getprofile()
    prior_errstate = np.geterr()
    assert prior_profile is None
    with threadpool_limits(limits=1):
        selected = _aocl.nnls_matmul()
        assert selected is np.matmul or selected is _aocl.padded_matmul
        protected = selected is _aocl.padded_matmul
        functions = (experimental.nnls_block_pivot, experimental._nnls_block_pivot,
                     experimental._passive_solve, experimental.nnls_matmul,
                     experimental.padded_matmul, _aocl.nnls_matmul, _aocl.padded_matmul)

        def observe(frame, event, arg):
            if event == 'call':
                if frame.f_code is experimental._nnls_block_pivot.__code__:
                    entries['core'] += 1
                elif frame.f_code is experimental._passive_solve.__code__:
                    entries['passive'] += 1
                elif frame.f_code is _aocl.padded_matmul.__code__:
                    left, right = frame.f_locals['left'], frame.f_locals['right']
                    products.append((left.shape, right.shape, right.strides,
                                     type(left) is np.ndarray and type(right) is np.ndarray,
                                     left.dtype == right.dtype == np.float64,
                                     bool(left.flags.aligned and right.flags.aligned)))
            elif event == 'return' and frame.f_code is _aocl.nnls_matmul.__code__:
                assert arg is selected
                entries['selector'] += 1

        expected_type = FloatingPointError if overflow == 'raise' else ValueError
        expected_message = ('overflow encountered in matmul' if overflow == 'raise'
                            else 'NNLS products overflowed; rescale the input.')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with np.errstate(over=overflow, divide='warn', invalid='warn', under='ignore'):
                sys.setprofile(observe)
                try:
                    with pytest.raises(expected_type) as error:
                        experimental.nnls_block_pivot(a, b, init=None, max_iter=None, tol=1e-10)
                finally:
                    sys.setprofile(prior_profile)
        assert type(error.value) is expected_type
        assert str(error.value) == expected_message
        count = (2 if protected else 1) if overflow == 'warn' else 0
        assert [(item.category, str(item.message)) for item in caught] == [
            (RuntimeWarning, 'overflow encountered in matmul')] * count
        assert entries == {'core': 1, 'passive': 0, 'selector': 1}
        expected_product = ((9, 13), (13, 9), (72, 8), True, True, True)
        assert products == [expected_product] * ((1 if overflow == 'raise' else 2) if protected else 0)
        assert all(x is y for x, y in zip(functions, (
            experimental.nnls_block_pivot, experimental._nnls_block_pivot,
            experimental._passive_solve, experimental.nnls_matmul,
            experimental.padded_matmul, _aocl.nnls_matmul, _aocl.padded_matmul)))
        assert _aocl.nnls_matmul() is selected
    assert before == (state(a), state(b))
    assert np.geterr() == prior_errstate and sys.getprofile() is prior_profile
    request.node.user_properties.extend([
        ('selected_protected', protected), ('accepted_warning_count', count),
        ('reference_warning_count', 1 if overflow == 'warn' else 0)])
