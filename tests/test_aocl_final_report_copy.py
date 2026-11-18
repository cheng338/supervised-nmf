"""The final-report opt-in changes copying, never native products or defaults.

Scoped array subclasses expose writes to private padding only. Native-product
observers use independent einsum arithmetic, so backend metadata is never used
to authorize an unprotected fringe GEMM in these routing/failure tests.
"""
from pathlib import Path
import sys
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import experimental_nmf as experimental
from supervised_nmf import _aocl

assert Path(_aocl.__file__).resolve() == ROOT / "src/supervised_nmf/_aocl.py"
assert Path(experimental.__file__).resolve() == ROOT / "scripts/experimental_nmf.py"


@pytest.fixture(autouse=True)
def one_native_thread():
    policy, callback = np.geterr(), np.geterrcall()
    with threadpool_limits(limits=1):
        yield
    assert np.geterr() == policy and np.geterrcall() is callback


def bits(value): return np.asarray(value).tobytes(order="C")


def key_record(key):
    if isinstance(key,tuple): return tuple(key_record(v) for v in key)
    if isinstance(key,slice): return (key.start,key.stop,key.step)
    return key


class PaddingObserver:
    def __init__(self, fault=None):
        self.fault = fault
        self.allocations = 0
        self.writes = []
        self.native = []
        self.refs = []
        self.output_copies = 0
        self.fired = False

    def __getattr__(self,name): return getattr(np,name)

    def zeros(self, shape, **kwargs):
        self.allocations += 1
        if self.fault == "allocation%d" % self.allocations:
            raise MemoryError("required padding allocation")
        observer = self
        ordinal = self.allocations
        class Buffer(np.ndarray):
            def __setitem__(self,key,value):
                observer.writes.append((ordinal,key_record(key)))
                block = ordinal == 1 and isinstance(key,tuple)
                if block and key[1].start == 4 and not observer.fired:
                    if observer.fault in ("population", "population-and-fallback", "population-runtime"):
                        np.ndarray.__setitem__(self,key,value)
                        observer.fired = True
                        if observer.fault == "population-runtime": raise RuntimeError("unexpected population")
                        raise MemoryError("optional partial population")
                if ordinal == 1 and observer.fault == "population-and-fallback" and observer.fired and not block:
                    assert sys.exception() is None
                    raise MemoryError("required fallback population")
                return np.ndarray.__setitem__(self,key,value)
        value = np.zeros(shape,**kwargs).view(Buffer)
        self.refs.append(weakref.ref(value))
        return value

    def matmul(self,left,right):
        assert left.shape[0] % 8 == right.shape[1] % 8 == 0
        self.native.append(dict(left_shape=left.shape,right_shape=right.shape,
                                left_bits=bits(left),right_bits=bits(right)))
        if self.fault == "native": raise MemoryError("required native product")
        return np.einsum("ik,kj->ij", left, right, optimize=False, order="C")

    def array(self,*args,**kwargs):
        self.output_copies += 1
        if self.fault == "output": raise MemoryError("required output copy")
        return np.array(*args,**kwargs)


def operands(m=4096,k=16,n=3,left_order="F",right_order="C"):
    left = np.full((m,k),.5,order=left_order)
    right = np.full((k,n),.25,order=right_order)
    left.setflags(write=False); right.setflags(write=False)
    return left,right


def run_report(monkeypatch,left,right,opt_in,observer):
    before = bits(left),bits(right)
    with monkeypatch.context() as patch:
        patch.setattr(_aocl,"np",observer)
        result = _aocl.padded_matmul(left,right,_final_report_copy4=opt_in)
    assert (bits(left),bits(right)) == before
    assert result.flags.owndata and result.base is None and result.flags.writeable and result.flags.c_contiguous
    assert not np.shares_memory(result,left) and not np.shares_memory(result,right)
    np.testing.assert_array_equal(result,np.full((left.shape[0],right.shape[1]),left.shape[1]/8))
    assert all(ref() is None for ref in observer.refs)
    return result


@pytest.mark.parametrize("m,k,opt_in,blocked", [
    (4096,16,True,True), (8192,12,True,True), (8192,21,True,True),
    (4096,12,True,False), (8192,11,True,False), (8191,16,True,False),
    (4096,16,False,False), (4096,16,1,False), (4096,16,np.bool_(True),False),
])
def test_report_gate_and_default_copy_share_exact_padded_geometry(monkeypatch,m,k,opt_in,blocked):
    left,right = operands(m,k)
    observer = PaddingObserver()
    run_report(monkeypatch,left,right,opt_in,observer)
    assert observer.allocations == 2 and observer.output_copies == 1 and len(observer.native) == 1
    writes = [key for ordinal,key in observer.writes if ordinal == 1]
    expected = [((None,None,None),(j,j+4,None)) for j in range(0,k,4)] if blocked else [(None,m,None)]
    assert writes == expected
    event = observer.native[0]; mp = (m+7)&~7
    assert event['left_shape'] == (mp,k) and event['right_shape'] == (k,8)
    expected_left = np.zeros((mp,k)); expected_left[:m] = left
    expected_right = np.zeros((k,8)); expected_right[:,:3] = right
    assert event['left_bits'] == bits(expected_left) and event['right_bits'] == bits(expected_right)


@pytest.mark.parametrize("kind", ["left-c", "left-strided", "left-subclass", "left-unaligned", "right-f", "right-subclass", "aligned-columns"])
def test_opt_in_does_not_expand_original_operand_routes(monkeypatch,kind):
    class Array(np.ndarray): pass
    left,right = operands()
    if kind == "left-c": left = np.array(left,order="C")
    elif kind == "left-strided": left = np.full((8192,16),.5,order="F")[::2]
    elif kind == "left-subclass": left = left.view(Array)
    elif kind == "left-unaligned":
        left = np.ndarray((4096,16),buffer=bytearray(4096*16*8+1),offset=1,order="F"); left.fill(.5)
    elif kind == "right-f": right = np.array(right,order="F")
    elif kind == "right-subclass": right = right.view(Array)
    elif kind == "aligned-columns": right = np.full((16,8),.25)
    left.setflags(write=False); right.setflags(write=False)
    first, second = PaddingObserver(), PaddingObserver()
    normal = run_report(monkeypatch,left,right,False,first)
    flagged = run_report(monkeypatch,left,right,True,second)
    assert bits(normal) == bits(flagged)
    assert first.allocations == second.allocations and first.writes == second.writes
    assert first.native == second.native


def test_partial_optional_copy_is_fully_overwritten_before_native(monkeypatch):
    left,right = operands(k=21)
    ordinary,failed = PaddingObserver(),PaddingObserver("population")
    run_report(monkeypatch,left,right,False,ordinary)
    run_report(monkeypatch,left,right,True,failed)
    assert failed.fired and failed.allocations == 2 and len(failed.native) == 1
    assert failed.native == ordinary.native
    assert [key for ordinal,key in failed.writes if ordinal==1] == [
        ((None,None,None),(0,4,None)),((None,None,None),(4,8,None)),(None,4096,None)]


@pytest.mark.parametrize("fault,allocations,native,copies", [
    ("allocation1",1,0,0), ("allocation2",2,0,0),
    ("population-and-fallback",2,0,0), ("native",2,1,0), ("output",2,1,1),
])
def test_required_report_failures_propagate_without_retry(monkeypatch,fault,allocations,native,copies):
    left,right = operands(); before = bits(left),bits(right)
    observer = PaddingObserver(fault)
    with monkeypatch.context() as patch:
        patch.setattr(_aocl,"np",observer)
        with pytest.raises(MemoryError,match="required") as caught:
            _aocl.padded_matmul(left,right,_final_report_copy4=True)
    assert caught.value.__context__ is None
    assert observer.allocations == allocations and len(observer.native) == native and observer.output_copies == copies
    assert (bits(left),bits(right)) == before


def test_unexpected_report_population_error_is_not_swallowed(monkeypatch):
    left,right = operands(); observer = PaddingObserver("population-runtime")
    monkeypatch.setattr(_aocl,"np",observer)
    with pytest.raises(RuntimeError,match="unexpected population"):
        _aocl.padded_matmul(left,right,_final_report_copy4=True)
    assert observer.fired and not observer.native and observer.output_copies == 0


def test_fit_uses_opt_in_only_after_publication_and_before_return(monkeypatch):
    events = []; original_copy = experimental._copy_return_coordinates
    model = experimental.ANLSBPP(2)
    def selected(left,right,**kwargs):
        if kwargs:
            assert kwargs == {"_final_report_copy4":True}
            assert hasattr(model,"components_") and hasattr(model,"training_coordinates_")
            assert not hasattr(model,"nnls_diagnostics_")
            events.append("report")
        return np.einsum("ik,kj->ij",left,right,optimize=False)
    def copy(left):
        events.append("return" if hasattr(model,"reconstruction_err_") else "publication")
        return original_copy(left)
    monkeypatch.setattr(experimental,"nnls_matmul",lambda:selected)
    monkeypatch.setattr(experimental,"padded_matmul",selected)
    monkeypatch.setattr(experimental,"_copy_return_coordinates",copy)
    data = np.zeros((8,4)); coordinates = model.fit_transform(data)
    assert events == ["publication","report","return"]
    events.clear()
    diagnostic = experimental.projected_gradient_kkt(data,coordinates,model.components_)
    with experimental.capture_transform_diagnostics(model) as transform:
        model.transform(data[:2])
    assert not events and diagnostic['reconstruction_error'] == 0 and transform['converged']


def test_ordinary_selected_callable_never_receives_private_report_keyword(monkeypatch):
    calls = []
    def ordinary(left,right):
        calls.append((left.shape,right.shape))
        return np.einsum("ik,kj->ij",left,right,optimize=False)
    monkeypatch.setattr(experimental,"nnls_matmul",lambda:ordinary)
    model = experimental.ANLSBPP(2).fit(np.zeros((8,4)))
    assert model.converged_ and model.reconstruction_err_ == 0 and calls
