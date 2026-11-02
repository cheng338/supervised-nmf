"""Separate protected one-axis decomposition; every native dot retains full K."""
import gc
from pathlib import Path
import sys
import weakref
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from supervised_nmf import _aocl
assert Path(_aocl.__file__).resolve()==ROOT/'src/supervised_nmf/_aocl.py'
AXIS='column'
POLICY='5e283198c322520ef8829590b1b4b7de55502b290281e9efc4f23bdbdd9b3efa'

@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):yield

def reference(a,b):return np.einsum('ik,kj->ij',np.asarray(a),np.asarray(b),optimize=False)
def bits(a):return np.ascontiguousarray(a,dtype=np.float64).view(np.uint64)
def state(a):return a.tobytes(),a.shape,a.strides,a.dtype.str,a.flags.writeable,a.flags.aligned
def operands(m,k,n):
    a=np.array((np.arange(m*k).reshape(m,k)%13-6)/8,dtype=float,order='C')
    b=np.array((np.arange(k*n).reshape(k,n)%17-8)/16,dtype=float,order='C')
    if a.size:a.flat[0]=-0.
    if b.size:b.flat[0]=-0.
    return a,b

def layout(a,kind):
    if kind=='f':return np.array(a,order='F')
    if kind=='strided':
        storage=np.empty(tuple(2*d for d in a.shape));view=storage[::2,::2];view[:]=a;return view
    if kind=='c_view':
        storage=np.empty((a.shape[0]+2,a.shape[1]));view=storage[1:-1];view[:]=a;return view
    return a

@pytest.mark.parametrize('m,k,n',[(9,7,9),(15,13,17),(20,17,454),(17,1,15),(20,37,20)])
@pytest.mark.parametrize('other_layout',['c','f','strided'])
@pytest.mark.parametrize('borrowed_view',[False,True])
def test_two_full_k_products_preserve_padding_and_cover_output(monkeypatch,m,k,n,other_layout,borrowed_view):
    a,b=operands(m,k,n)
    if AXIS=='column':a=layout(a,other_layout);b=layout(b,'c_view' if borrowed_view else 'c')
    else:b=layout(b,other_layout);a=layout(a,'c_view' if borrowed_view else 'c')
    a.setflags(write=False);b.setflags(write=False);before=state(a),state(b)
    calls=[];allocated=[]
    class Proxy:
        def __getattr__(self,name):return getattr(np,name)
        def empty(self,shape,**kw):
            value=np.empty(shape,**kw);value.fill(np.nan);allocated.append(('empty',value));return value
        def zeros(self,shape,**kw):
            value=np.zeros(shape,**kw);allocated.append(('zeros',value));return value
        def matmul(self,left,right):
            assert left.shape[0]%8==right.shape[1]%8==0
            assert left.shape[1]==right.shape[0]==k
            assert np.isfinite(left).all() and np.isfinite(right).all()
            value=reference(left,right);calls.append((left,right,value));return value
    monkeypatch.setattr(_aocl,'np',Proxy());actual=_aocl.padded_matmul(a,b)
    mp,npad=(m+7)//8*8,(n+7)//8*8;mc,nc=m//8*8,n//8*8
    assert len(calls)==2 and [tag for tag,_ in allocated]==['zeros','zeros','empty']
    assert allocated[-1][1] is actual
    if AXIS=='column':
        assert [(x.shape,y.shape) for x,y,_ in calls]==[((mp,k),(k,nc)),((mp,k),(k,8))]
        full,core,fringe=calls[0][0],calls[0][1],calls[1][1]
        assert calls[1][0] is full and allocated[0][1] is full and allocated[1][1] is fringe
        old=np.zeros((mp,k));old[:m]=a
        np.testing.assert_array_equal(bits(full),bits(old))
        assert np.shares_memory(core,b) and core.strides==b.strides
        np.testing.assert_array_equal(bits(core),bits(b[:,:nc]))
        np.testing.assert_array_equal(bits(fringe[:,:n-nc]),bits(b[:,nc:]))
        assert not np.any(bits(fringe[:,n-nc:]))
    else:
        assert [(x.shape,y.shape) for x,y,_ in calls]==[((mc,k),(k,npad)),((8,k),(k,npad))]
        core,full,fringe=calls[0][0],calls[0][1],calls[1][0]
        assert calls[1][1] is full and allocated[0][1] is fringe and allocated[1][1] is full
        old=np.zeros((k,npad));old[:,:n]=b
        np.testing.assert_array_equal(bits(full),bits(old))
        assert np.shares_memory(core,a) and core.strides==a.strides
        np.testing.assert_array_equal(bits(core),bits(a[:mc]))
        np.testing.assert_array_equal(bits(fringe[:m-mc]),bits(a[mc:]))
        assert not np.any(bits(fringe[m-mc:]))
    for x in (full,fringe):assert x.dtype==np.float64 and x.flags.c_contiguous and x.flags.aligned
    np.testing.assert_array_equal(actual,reference(a,b))
    assert np.isfinite(actual).all() and actual.flags.owndata and actual.flags.c_contiguous and actual.flags.writeable and actual.base is None
    assert all(not np.shares_memory(actual,x) for x in (a,b,*[p for _,_,p in calls]))
    assert before==(state(a),state(b))

class Subclass(np.ndarray):pass

@pytest.mark.parametrize('kind',['f','strided','negative','float32','swapped','subclass','unaligned'])
def test_borrowed_core_exclusions_keep_original_single_product(monkeypatch,kind):
    a,b=operands(19,13,21);value=b if AXIS=='column' else a
    if kind in ('f','strided'):value=layout(value,kind)
    elif kind=='negative':value=value[:,::-1].copy()[:,::-1]
    elif kind=='float32':value=value.astype(np.float32)
    elif kind=='swapped':value=value.astype(np.dtype(float).newbyteorder('S'))
    elif kind=='subclass':value=value.view(Subclass)
    else:value.setflags(align=False)
    if AXIS=='column':b=value
    else:a=value
    a.setflags(write=False);b.setflags(write=False);before=state(a),state(b);calls=[]
    class Proxy:
        def __getattr__(self,name):return getattr(np,name)
        def matmul(self,x,y):calls.append((x,y));return reference(x,y)
    monkeypatch.setattr(_aocl,'np',Proxy());result=_aocl.padded_matmul(a,b)
    assert len(calls)==1 and (calls[0][0].shape,calls[0][1].shape)==((24,13),(13,24))
    olda,oldb=np.zeros((24,13)),np.zeros((13,24));olda[:19]=a;oldb[:,:21]=b
    np.testing.assert_array_equal(bits(calls[0][0]),bits(olda));np.testing.assert_array_equal(bits(calls[0][1]),bits(oldb))
    np.testing.assert_array_equal(result,reference(np.asarray(a,dtype=float),np.asarray(b,dtype=float)))
    assert before==(state(a),state(b))

@pytest.mark.parametrize('kind',['float32','subclass','unaligned'])
def test_other_operand_type_and_alignment_preserve_fallback(monkeypatch,kind):
    a,b=operands(19,13,21);value=a if AXIS=='column' else b
    if kind=='float32':value=value.astype(np.float32)
    elif kind=='subclass':value=value.view(Subclass)
    else:value.setflags(align=False)
    if AXIS=='column':a=value
    else:b=value
    calls=[]
    class Proxy:
        def __getattr__(self,name):return getattr(np,name)
        def matmul(self,x,y):calls.append((x.shape,y.shape));return reference(x,y)
    monkeypatch.setattr(_aocl,'np',Proxy());result=_aocl.padded_matmul(a,b)
    assert calls==[((24,13),(13,24))]
    np.testing.assert_array_equal(result,reference(np.asarray(a,dtype=float),np.asarray(b,dtype=float)))

def test_contiguous_singleton_with_noncanonical_stride_keeps_fallback(monkeypatch):
    a,b=operands(19,1,21)
    if AXIS=='column':b=np.arange(21,dtype=float)[None,:];value=b
    else:a=np.arange(19,dtype=float)[:,None];value=a
    assert value.flags.c_contiguous and value.strides!=(value.shape[1]*8,8)
    calls=[]
    class Proxy:
        def __getattr__(self,name):return getattr(np,name)
        def matmul(self,x,y):calls.append((x.shape,y.shape));return reference(x,y)
    monkeypatch.setattr(_aocl,'np',Proxy());result=_aocl.padded_matmul(a,b)
    assert calls==[((24,1),(1,24))]
    np.testing.assert_array_equal(result,reference(a,b))

@pytest.mark.parametrize('m,k,n',[(5,7,9),(9,7,5),(8,7,9),(9,7,8),(16,7,16),(0,7,9),(9,7,0),(9,0,9)])
def test_other_geometry_preserves_single_native_call(monkeypatch,m,k,n):
    calls=[]
    class Proxy:
        def __getattr__(self,name):return getattr(np,name)
        def matmul(self,a,b):
            calls.append((a.shape,b.shape));assert a.shape[0]%8==b.shape[1]%8==0
            return reference(a,b)
    monkeypatch.setattr(_aocl,'np',Proxy());actual=_aocl.padded_matmul(np.ones((m,k)),np.ones((k,n)))
    assert len(calls)==1
    np.testing.assert_array_equal(actual,np.full((m,n),k,dtype=float))

def native_compare(a,b,request):
    m,k=a.shape;n=b.shape[1];mp,npad=(m+7)//8*8,(n+7)//8*8
    before=state(a),state(b);aa,bb=np.zeros((mp,k)),np.zeros((k,npad));aa[:m]=a;bb[:,:n]=b
    selected=_aocl.nnls_matmul();protected=selected is _aocl.padded_matmul
    expected=selected(aa,bb)[:m,:n].copy(order='C')
    actual=_aocl.padded_matmul(a,b);equal=actual.tobytes()==expected.tobytes()
    request.node.user_properties.extend([('selected_protected',protected),('native_bytes_equal',equal),
        ('native_maximum_absolute_difference',float(np.max(np.abs(actual-expected)))),('backend_relevance_policy_sha256',POLICY)])
    if protected:assert equal
    else:
        # The directly invoked private branch is unused by normal MKL solves.
        # Keep exact-byte differences visible and require independent sanity;
        # the actual selected MKL solver still has an exact-science gate.
        oracle=reference(a,b);scale=reference(np.abs(a),np.abs(b))
        bound=16*np.finfo(float).eps*max(1,k)*scale+16*max(1,k)*np.nextafter(0.,1.)
        assert np.all(np.abs(actual-oracle)<=bound)
    assert np.isfinite(actual).all() and actual.flags.owndata and actual.flags.c_contiguous and actual.flags.writeable and actual.base is None
    assert not np.shares_memory(actual,a) and not np.shares_memory(actual,b)
    assert before==(state(a),state(b))

@pytest.mark.parametrize('m,k,n',[(20,454,20),(20,8000,454),(9,37,17)])
def test_actual_selected_native_bytes_match_original_padding(m,k,n,request):
    rng=np.random.default_rng(20261103+m+k+n);a=rng.uniform(-1,1,(m,k));b=rng.uniform(-1,1,(k,n))
    a.setflags(write=False);b.setflags(write=False);native_compare(a,b,request)

@pytest.mark.parametrize('kind',['signed-zero','subnormal','tiny','large','cancellation'])
def test_actual_selected_edge_bytes_match_original_padding(kind,request):
    a,b=operands(19,13,21)
    if kind=='signed-zero':a.fill(0.);b.fill(0.);a[::2]=-0.;b[:,::2]=-0.
    elif kind in ('subnormal','tiny','large'):
        exponent={'subnormal':-530,'tiny':-500,'large':500}[kind];a=np.ldexp(a,exponent);b=np.ldexp(b,exponent)
    else:
        a[:,1::2]=-a[:,:-1:2];b[1::2]=b[:-1:2]
    a.setflags(write=False);b.setflags(write=False);native_compare(a,b,request)

@pytest.mark.parametrize('failure_at',[None,1,2])
def test_workspaces_release_after_success_or_native_error(monkeypatch,failure_at):
    refs=[];calls=0
    class Proxy:
        def __getattr__(self,name):return getattr(np,name)
        def zeros(self,*args,**kw):
            value=np.zeros(*args,**kw);refs.append(weakref.ref(value));return value
        def empty(self,*args,**kw):
            value=np.empty(*args,**kw);refs.append(weakref.ref(value));return value
        def matmul(self,a,b):
            nonlocal calls
            calls+=1
            if calls==failure_at:raise RuntimeError('test product failure')
            value=reference(a,b);refs.append(weakref.ref(value));return value
    monkeypatch.setattr(_aocl,'np',Proxy());a,b=operands(19,13,21);before=state(a),state(b)
    def attempt():
        try:return _aocl.padded_matmul(a,b)
        except RuntimeError:
            assert failure_at is not None
            return None
    result=attempt();gc.collect()
    assert calls==(failure_at if failure_at else 2)
    assert sum(ref() is not None for ref in refs)==(1 if failure_at is None else 0)
    assert all(ref() is None or ref() is result for ref in refs)
    assert before==(state(a),state(b))
    if result is not None:
        saved=result.copy();later=_aocl.padded_matmul(a*2,b)
        np.testing.assert_array_equal(result,saved);assert not np.shares_memory(later,result)
