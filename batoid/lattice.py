import numpy as np
from .utils import lazy_property


def primitiveToLattice(primitiveVectors, Ns):
    # 2D output should be [N1, N2, 2]
    # 3D output should be [N1, N2, N3, 3]
    # and so on...
    ns = []
    for d in np.arange(len(Ns)):
        ns.append(np.arange(-(Ns[d]//2), -(-Ns[d]//2)))
    ns = np.meshgrid(*ns, indexing='ij')
    return np.matmul(np.moveaxis(ns, 0, -1), primitiveVectors)


class Lattice:
    """Container class for an ndarray + primitive lattice vectors.

    Used as the output type for several of the analysis algorithms, including
    PSFs and wavefronts.

    Parameters
    ----------
    array : ndarray, shape (N1, N2, ..., Nd)
        d-dimensional ndarray with dimensions N1, N2, ..., Nd
    primitiveVector : (d, d) ndarray
        Primitive lattice vectors.  E.g., primitiveVector[0] should contain the
        lattice vector for the first dimension.

    Notes
    -----
        The ``coords`` attribute will contain the coordinates of each lattice
        point determined by the coordinate index and the primitive lattice
        vectors.  E.g., in 2-dimensions

            lattice.coord[i, j] = (i - N1//2) * primitiveVector[0] + (j - N2//2) * primitiveVector[1]

        Note, this convention places

            lattice.coord[0,0] = (-N1//2) * primitiveVector[0] + (-N2/2) * primitiveVector[1]

        but the location of lattice.coord[-1,-1] depends on whether the size of
        each dimension is even or odd.  An odd-sized dimension is "centered", in
        that the 0-coordinate is precisely in the middle of the dimension.  An
        even-sized dimension will be slightly decentered, with more negative
        points than positive points.  The above convention is the same as for
        numpy.fft.fftfreq.
    """
    def __init__(self, array, primitiveVectors):
        primitiveVectors = np.atleast_2d(primitiveVectors)

        if array.ndim != len(primitiveVectors):
            raise ValueError("Not enough primitiveVectors for array")
        if array.ndim != len(primitiveVectors[0]):
            raise ValueError("primitiveVectors are too small for array")

        self.array = array
        self.primitiveVectors = primitiveVectors

    @lazy_property
    def coords(self):
        return primitiveToLattice(self.primitiveVectors, self.array.shape)

    def __eq__(self, rhs):
        if not isinstance(rhs, Lattice): return False
        return (
            np.array_equal(self.array, rhs.array) and
            np.array_equal(self.primitiveVectors, rhs.primitiveVectors)
        )

    def __hash__(self):
        return hash((
            "Lattice",
            tuple(self.array.ravel()),
            tuple(self.primitiveVectors.ravel())
        ))

    def __repr__(self):
        return f"Lattice({self.array!r}, {self.primitiveVectors!r})"
