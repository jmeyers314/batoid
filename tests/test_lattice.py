import numpy as np
import batoid

from test_helpers import timer


@timer
def test_lattice_coords():
    np.random.seed(5)

    # Check 1D
    for _ in np.arange(10):
        N = np.random.randint(1, 2000)
        arr = np.ones((N,))
        primitiveVector = np.random.uniform(-1.0, 1.0)

        lattice = batoid.Lattice(arr, primitiveVector)

        np.testing.assert_allclose(
            np.squeeze(lattice.coords),
            np.arange(-(N//2), -(-N//2))*primitiveVector
        )

    # Check 2D
    for _ in np.arange(10):
        N1 = np.random.randint(1, 200)
        N2 = np.random.randint(1, 200)
        arr = np.ones((N1, N2))
        pv1 = np.random.uniform(-1.0, 1.0, size=2)
        pv2 = np.random.uniform(-1.0, 1.0, size=2)

        lattice = batoid.Lattice(arr, np.vstack([pv1, pv2]))

        for _ in np.arange(100):
            i = np.random.randint(0, N1)
            j = np.random.randint(0, N2)
            np.testing.assert_allclose(
                lattice.coords[i,j],
                (i-N1//2)*pv1 + (j-N2//2)*pv2
            )

    # Check 3D
    for _ in np.arange(10):
        N1 = np.random.randint(1, 20)
        N2 = np.random.randint(1, 20)
        N3 = np.random.randint(1, 20)
        arr = np.ones((N1, N2, N3))
        pv1 = np.random.uniform(-1.0, 1.0, size=3)
        pv2 = np.random.uniform(-1.0, 1.0, size=3)
        pv3 = np.random.uniform(-1.0, 1.0, size=3)

        lattice = batoid.Lattice(arr, np.vstack([pv1, pv2, pv3]))

        coords = lattice.coords
        for __ in np.arange(100):
            i = np.random.randint(0, N1)
            j = np.random.randint(0, N2)
            k = np.random.randint(0, N3)
            np.testing.assert_allclose(
                lattice.coords[i,j,k],
                (i-N1//2)*pv1 + (j-N2//2)*pv2 + (k-N3//2)*pv3
            )


if __name__ == '__main__':
    test_lattice_coords()
