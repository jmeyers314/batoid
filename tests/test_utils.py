import batoid
from test_helpers import timer

import numpy as np

@timer
def test_gnomicDirCos():
    np.random.seed(5)
    u = np.random.uniform(-0.1, 0.1, size=1000)
    v = np.random.uniform(-0.1, 0.1, size=1000)

    # Test round trip
    u1, v1 = batoid.utils.dirCosToGnomic(*batoid.utils.gnomicToDirCos(u, v))
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in the other direction
    alpha = np.random.uniform(-0.1, 0.1, size=1000)
    beta = np.random.uniform(-0.1, 0.1, size=1000)
    gamma = np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.gnomicToDirCos(
        *batoid.utils.dirCosToGnomic(alpha, beta, gamma)
    )
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # For really tiny angles, u/v should be basically the same as alpha/beta
    u = np.random.uniform(-1e-6, 1e-6, size=1000)
    v = np.random.uniform(-1e-6, 1e-6, size=1000)
    alpha, beta, gamma = batoid.utils.gnomicToDirCos(u, v)
    np.testing.assert_allclose(alpha, u, rtol=0, atol=1e-8)
    np.testing.assert_allclose(beta, v, rtol=0, atol=1e-8)

    # Check normalization of direction cosines
    np.testing.assert_allclose(np.sqrt(alpha*alpha+beta*beta+gamma*gamma), 1, rtol=0, atol=1e-15)


@timer
def test_gnomicSpherical():
    np.random.seed(57)
    u = np.random.uniform(-0.1, 0.1, size=1000)
    v = np.random.uniform(-0.1, 0.1, size=1000)

    # Test round trip
    u1, v1 = batoid.utils.sphericalToGnomic(*batoid.utils.gnomicToSpherical(u, v))
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in other direction
    phi = np.random.uniform(0.0, 0.1, size=1000)
    theta = np.random.uniform(-np.pi, np.pi, size=1000)
    phi1, theta1 = batoid.utils.gnomicToSpherical(*batoid.utils.sphericalToGnomic(phi, theta))
    np.testing.assert_allclose(phi, phi1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(theta, theta1, rtol=1e-10, atol=1e-12)

    # Check u**2 + v**2 = tan(phi)**2
    u, v = batoid.utils.sphericalToGnomic(phi, theta)
    np.testing.assert_allclose(np.tan(phi)**2, u**2+v**2, rtol=0, atol=1e-17)

    # Check v/u = tan(theta)
    np.testing.assert_allclose(np.tan(theta), v/u, rtol=1e-15, atol=0)


if __name__ == '__main__':
    test_gnomicDirCos()
    test_gnomicSpherical()
