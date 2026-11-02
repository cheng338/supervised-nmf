"""A focused temporary-consumption contract, portable within a package checkout.

Native products retain the normal MKL/protected AOCL selection. Independent
small arithmetic uses einsum without optimization, never raw AOCL GEMM.
Other qualified temporary-consumption changes are permitted by these tests.
"""
from contextlib import contextmanager
import gc
import inspect
import math
from pathlib import Path
import sys
import warnings
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
import experimental_nmf as experimental
from supervised_nmf import _aocl

assert Path(experimental.__file__).resolve() == ROOT / 'scripts/experimental_nmf.py'
assert Path(_aocl.__file__).resolve() == ROOT / 'src/supervised_nmf/_aocl.py'

@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        assert experimental.nnls_matmul() in (np.matmul, _aocl.padded_matmul)
        yield

def bits(value): return np.asarray(value).tobytes(order='C')
def state(value):
    return (type(value),value.dtype,value.shape,value.strides,value.flags.writeable,value.flags.aligned,bits(value))
def product_reference(left,right):
    return np.einsum('ik,kj->ij',left,right,optimize=False)
def scalar_norm(value):
    return math.sqrt(math.fsum(float(x)**2 for x in np.asarray(value).ravel()))
def readonly(value,kind='c',dtype=np.float64):
    value=np.asarray(value,dtype=dtype)
    if kind=='strided':
        storage=np.empty((value.shape[0]*2,value.shape[1]*2),dtype=dtype)
        result=storage[::2,::2];result[:]=value
    else:
        result=np.array(value,dtype=dtype,order='F' if kind=='f' else 'C',copy=True)
    result.setflags(write=False)
    return result

@contextmanager
def capture_native(monkeypatch,before=None,after=None):
    selected=experimental.nnls_matmul()
    norm=experimental.linalg.norm
    records=dict(products=[],norms=[])
    def multiply(left,right):
        frame=inspect.currentframe().f_back
        if before is not None: before(frame,left,right)
        output=selected(left,right)
        assert type(output) is np.ndarray and output.dtype==np.float64
        assert output.flags.owndata and output.flags.writeable and output.flags.c_contiguous
        assert not np.shares_memory(output,left) and not np.shares_memory(output,right)
        records['products'].append(dict(caller=frame.f_code.co_name,left=left,right=right,
                                        output=output,original=output.copy()))
        if after is not None: after(frame,left,right,output)
        return output
    def observed_norm(value,*args,**kwargs):
        frame=inspect.currentframe().f_back
        if frame.f_globals is experimental.__dict__ and frame.f_code.co_name=='fit_transform':
            records['norms'].append(dict(value=value,bytes=bits(value)))
        return norm(value,*args,**kwargs)
    with monkeypatch.context() as patch:
        if selected is _aocl.padded_matmul:
            patch.setattr(_aocl,'padded_matmul',multiply)
            patch.setattr(experimental,'padded_matmul',multiply)
        patch.setattr(experimental,'nnls_matmul',lambda:multiply)
        patch.setattr(experimental.linalg,'norm',observed_norm)
        yield records

def fit_data(shape=(64,32),kind='c'):
    return readonly(np.random.default_rng(4308).uniform(.05,2.,shape),kind)
def fit_model(X,rank=4):
    model=experimental.ANLSBPP(n_components=rank,max_iter=3,tol=0.,random_state=43)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        coordinates=model.fit_transform(X)
    assert model.nnls_diagnostics_['failed_solves']==0
    assert model.n_iter_==3 and model.convergence_reason_=='outer_iteration_limit'
    assert len(caught)==1 and caught[0].category.__name__=='ConvergenceWarning'
    return model,coordinates

def disjoint_problem(kind='c'):
    design=np.zeros((9,3))
    for row in range(9): design[row,row%3]=(1 if row%2 else -1)*(1+row%3/4)
    signed=(np.arange(24).reshape(3,8)%7-2)/4
    return readonly(design,kind),readonly(product_reference(design,signed),kind),np.maximum(signed,0.)



@pytest.mark.parametrize('shape,rank,kind',[((64,32),4,'c'),((65,33),5,'f'),((65,33),5,'strided')])
def test_outer_objective_consumes_its_own_product(monkeypatch,shape,rank,kind):
    X=fit_data(shape,kind); original=state(X)
    with capture_native(monkeypatch) as records:
        model,coordinates=fit_model(X,rank)
    # Identify the actual object consumed by the objective norm; no source-line
    # counts and no prohibition on other independent temporary optimizations.
    consumed=[]
    for norm in records['norms']:
        matches=[event for event in records['products'] if event['caller']=='fit_transform'
                 and event['output'] is norm['value'] and event['output'].shape==X.shape
                 and bits(np.subtract(event['original'],np.asarray(X,dtype=np.float64)))==norm['bytes']]
        if matches: consumed.append(matches[0])
    assert len(consumed)==model.n_iter_==3
    for index,event in enumerate(consumed):
        residual=event['output']
        assert bits(residual)==bits(np.subtract(event['original'],np.asarray(X,dtype=np.float64)))
        assert any(other['left'] is residual or other['right'] is residual for other in records['products'])
        assert not any(np.shares_memory(residual,value) for value in (X,coordinates,model.components_))
        assert model.objective_history_[index+1]==pytest.approx(.5*scalar_norm(residual)**2,rel=1e-13)
    assert state(X)==original
    refs=[weakref.ref(event['output']) for event in consumed]
    del consumed,event,residual,records,norm,matches
    gc.collect()
    assert all(ref() is None for ref in refs)
    # Refit owns a new lifetime and retains only the ordinary fitted arrays.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit_transform(X)
    assert state(X)==original and model.n_iter_==3

def test_stationary_and_failed_inner_solve_do_not_enter_outer_objective(monkeypatch):
    zero=readonly(np.zeros((16,8)),'f'); before=state(zero)
    stationary=experimental.ANLSBPP(n_components=3,max_iter=3,random_state=43)
    stationary.fit_transform(zero)
    assert stationary.n_iter_==0 and stationary.convergence_reason_=='initial_stationarity'
    assert state(zero)==before
    X=fit_data(); before=state(X)
    original=experimental._nnls_block_pivot
    def failed(*args,**kwargs):
        coefficients,info=original(*args,**kwargs)
        return coefficients,dict(info,success=False)
    with capture_native(monkeypatch) as records:
        with monkeypatch.context() as patch:
            patch.setattr(experimental,'_nnls_block_pivot',failed)
            model=experimental.ANLSBPP(n_components=4,max_iter=3,random_state=43)
            with pytest.warns(experimental.ConvergenceWarning,match='inner_nnls_failure'):
                model.fit_transform(X)
    assert model.n_iter_==0 and model.convergence_reason_=='inner_nnls_failure'
    assert len(model.objective_history_)==1 and state(X)==before
    for norm in records['norms']:
        for event in records['products']:
            if event['caller']=='fit_transform' and event['output'] is norm['value'] and event['output'].shape==X.shape:
                # A combined reverse-norm candidate may own this norm input.
                assert norm['bytes']==bits(np.subtract(np.asarray(X),event['original']))
