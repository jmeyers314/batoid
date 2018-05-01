import numpy as np
from .utils import lazy_property


class Lattice:
    """Simple container class for an ndarray + primitive lattice vectors.

    Parameters
    ----------
    arr : (N1, ... Nd), ndarray
        d-dimensional ndarray with dimensions N1, N2, ..., Nd
    primitiveVector : (d, d) ndarray
        Primitive lattice vectors.  E.g., primitiveVector[0] should contain the lattice vector for
        the first dimension.

    Notes
    -----
        The coords attribute will contain the coordinates of each lattice point determined by the
        coordinate index and the primitive lattice vectors.  E.g., in 2-dimensions,

        lattice.coord[i, j] == (i - N1/2) * primitiveVector[0] + (j - N2/2) * primitiveVector[1]
    """
    def __init__(self, arr, primitiveVectors):
        primitiveVectors = np.atleast_2d(primitiveVectors)

        assert arr.ndim == len(primitiveVectors), "Not enough primitiveVectors for array"
        assert arr.ndim == len(primitiveVectors[0]), "primitiveVectors are too small for array"

        self.arr = arr
        self.primitiveVectors = primitiveVectors

    @lazy_property
    def coords(self):
        # 2D output should be [N1, N2, 2]
        # 3D output should be [N1, N2, N3, 3]
        # and so on...
        ns = []
        for d in np.arange(self.arr.ndim):
            ns.append(np.arange(-self.arr.shape[d]/2, self.arr.shape[d]/2))
        ns = np.meshgrid(*ns, indexing='ij')
        return np.matmul(np.moveaxis(ns, 0, -1), self.primitiveVectors)
