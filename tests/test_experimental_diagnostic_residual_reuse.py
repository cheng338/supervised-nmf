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
        records['products'].append(dict(caller=frame.f_code.co_name,caller_parent=(frame.f_back.f_code.co_name if frame.f_back is not None else None),left=left,right=right,
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



def independent_pg(W,H,gradient_W,gradient_H):
    projected_W=np.where((W>0)|(gradient_W<0),gradient_W,0.)
    projected_H=np.where((H>0)|(gradient_H<0),gradient_H,0.)
    projected=math.hypot(scalar_norm(projected_W),scalar_norm(projected_H))
    total=math.hypot(scalar_norm(gradient_W),scalar_norm(gradient_H))
    return dict(projected_gradient_norm=projected,gradient_norm=total,
                relative_projected_gradient=projected/total if total else 0.,
                primal_violation=float(max(0.,-W.min(),-H.min())))

@pytest.mark.parametrize('kind,dtype',[('c',np.float64),('f',np.float32),('strided',np.float64),('c',np.int64)])
def test_diagnostic_residual_has_fresh_storage_and_independent_values(monkeypatch,kind,dtype):
    W=readonly((np.arange(27).reshape(9,3)%5+1)/2,kind,dtype)
    H=readonly((np.arange(21).reshape(3,7)%4+1)/2,kind,dtype)
    X=readonly((np.arange(63).reshape(9,7)%6+1)/3,kind,dtype)
    original=[state(value) for value in (X,W,H)]
    with capture_native(monkeypatch) as records:
        result=experimental.projected_gradient_kkt(X,W,H)
    events=[event for event in records['products'] if event['caller']=='projected_gradient_kkt' or (event['caller']=='_diagnostic_residual' and event['caller_parent']=='projected_gradient_kkt')]
    residual=events[0]['output']
    expected=np.subtract(events[0]['original'],np.asarray(X,dtype=np.float64))
    assert bits(residual)==bits(expected)
    assert sum(event['left'] is residual or event['right'] is residual for event in events[1:])==4
    assert not any(np.shares_memory(residual,value) for value in (X,W,H))
    x,w,h=(np.asarray(value,dtype=np.float64) for value in (X,W,H))
    direct=product_reference(w,h)-x
    raw=independent_pg(w,h,product_reference(direct,h.T),product_reference(w.T,direct))
    scales=np.asarray([scalar_norm(row) for row in h]);scales=np.where(scales>0,scales,1.)
    cw,ch=w*scales,h/scales[:,None]
    canonical=independent_pg(cw,ch,product_reference(direct,ch.T),product_reference(cw.T,direct))
    for key in raw:
        assert result['raw'][key]==pytest.approx(raw[key],rel=1e-12,abs=1e-12)
        assert result['canonical'][key]==pytest.approx(canonical[key],rel=1e-12,abs=1e-12)
    assert result['reconstruction_error']==pytest.approx(scalar_norm(direct),rel=1e-13)
    assert result['objective']==pytest.approx(.5*scalar_norm(direct)**2,rel=1e-13)
    assert [state(value) for value in (X,W,H)]==original
    ref=weakref.ref(residual)
    del events,residual,records
    gc.collect();assert ref() is None

def test_zero_and_rejected_diagnostics_preserve_inputs():
    X=readonly(np.zeros((9,7)));W=readonly(np.zeros((9,3)));H=readonly(np.zeros((3,7)))
    original=[state(value) for value in (X,W,H)]
    result=experimental.projected_gradient_kkt(X,W,H)
    assert result['objective']==result['reconstruction_error']==0.
    assert result['raw']['relative_projected_gradient']==result['canonical']['relative_projected_gradient']==0.
    with pytest.raises(ValueError,match='Incompatible'):
        experimental.projected_gradient_kkt(X,W,H[:,:-1])
    assert [state(value) for value in (X,W,H)]==original
