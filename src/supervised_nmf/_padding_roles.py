"""Scoped padded operand and output storage for guarded matrix products."""

from ._aocl import np, _ndarray, padded_matmul

class _NativeOutputWorkspace:
    """Call-local owner: two exact native output slots, fresh logical results.

    Caller must scope an instance to one fit role or projection design group.
    No inputs or returned arrays are retained. A failure clears every slot.
    """

    def __init__(self):
        self._buffers = [None, None]
        self._used = 0

    def clear(self):
        self._buffers[0] = self._buffers[1] = None
        self._used = 0

    def _product(self, left, right):
        index = self._used
        if index >= 2:
            raise AssertionError('More than two protected products in one call')
        self._used += 1
        shape = (left.shape[0], right.shape[1])
        buffer = self._buffers[index]
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=np.float64, order='C')
            self._buffers[index] = buffer
        np.matmul(left, right, out=buffer)
        return buffer

    def __call__(self, left, right):
        self._used = 0
        try:
            return self._compute(left, right)
        except BaseException:
            self.clear()
            raise
        finally:
            for index in range(self._used, 2):
                self._buffers[index] = None

    def _compute(self, left, right):
        m, k = left.shape
        n = right.shape[1]
        mp, npad = (m + 7 & ~7, n + 7 & ~7)
        if mp == m:
            if npad == n:
                return np.matmul(left, right)
            if type(left) is _ndarray and left.flags.c_contiguous and (left.dtype == np.float64) and left.flags.aligned and (left.strides == (k * 8, 8)):
                b = np.zeros((k, npad), dtype=np.float64)
                b[..., :n] = right
                product = self._product(left, b)
                return np.array(product[:m, :n], order='C', copy=True)
        elif npad == n and type(right) is _ndarray and right.flags.c_contiguous and (right.dtype == np.float64) and right.flags.aligned and (right.strides == (n * 8, 8)):
            a = np.zeros((mp, k), dtype=np.float64)
            a[:m] = left
            product = self._product(a, right)
            return np.array(product[:m, :n], order='C', copy=True)
        if mp != m and npad != n and (m >= 8) and (n >= 8) and (type(left) is _ndarray) and (type(right) is _ndarray) and (left.dtype == np.float64) and (right.dtype == np.float64) and left.flags.aligned and right.flags.aligned and right.flags.c_contiguous and (right.strides == (n * 8, 8)):
            nc = n & ~7
            a = np.zeros((mp, k), dtype=np.float64)
            b = np.zeros((k, 8), dtype=np.float64)
            a[:m] = left
            b[:, :n - nc] = right[:, nc:]
            result = np.empty((m, n), dtype=np.float64)
            result[:, :nc] = self._product(a, right[:, :nc])[:m]
            result[:, nc:] = self._product(a, b)[:m, :n - nc]
            return result
        a = np.zeros((mp, k), dtype=np.float64)
        b = np.zeros((k, npad), dtype=np.float64)
        a[:m] = left
        b[..., :n] = right
        product = self._product(a, b)
        return np.array(product[:m, :n], order='C', copy=True)

def make_output_workspace(operation):
    """Return original nonpadded backend or one private mutable-output role."""
    if operation is not padded_matmul:
        return operation
    return _NativeOutputWorkspace()
