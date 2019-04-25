import batoid
from test_helpers import timer

import numpy as np


@timer
def test_normalized():
    for _ in range(1000):
        x = np.random.uniform()
        y = np.random.uniform()
        z = np.random.uniform()
        w = np.random.uniform()

        np.testing.assert_allclose(
            np.linalg.norm(batoid.utils.normalized([x])),
            1.0,
            rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.norm(batoid.utils.normalized([x, y])),
            1.0,
            rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.norm(batoid.utils.normalized([x, y, z])),
            1.0,
            rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.norm(batoid.utils.normalized([x, y, z, w])),
            1.0,
            rtol=0, atol=1e-10
        )


@timer
def test_gnomonicDirCos():
    np.random.seed(5)
    u = np.random.uniform(-0.1, 0.1, size=1000)
    v = np.random.uniform(-0.1, 0.1, size=1000)

    # Test round trip
    u1, v1 = batoid.utils.dirCosToGnomonic(*batoid.utils.gnomonicToDirCos(u, v))
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in the other direction
    alpha = np.random.uniform(-0.1, 0.1, size=1000)
    beta = np.random.uniform(-0.1, 0.1, size=1000)
    gamma = np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.gnomonicToDirCos(
        *batoid.utils.dirCosToGnomonic(alpha, beta, gamma)
    )
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # For really tiny angles, u/v should be basically the same as alpha/beta
    u = np.random.uniform(-1e-6, 1e-6, size=1000)
    v = np.random.uniform(-1e-6, 1e-6, size=1000)
    alpha, beta, gamma = batoid.utils.gnomonicToDirCos(u, v)
    np.testing.assert_allclose(alpha, u, rtol=0, atol=1e-8)
    np.testing.assert_allclose(beta, v, rtol=0, atol=1e-8)

    # Check normalization of direction cosines
    np.testing.assert_allclose(np.sqrt(alpha*alpha+beta*beta+gamma*gamma), 1, rtol=0, atol=1e-15)


@timer
def test_postelDirCos():
    np.random.seed(5)
    u = np.random.uniform(-0.1, 0.1, size=1000)
    v = np.random.uniform(-0.1, 0.1, size=1000)

    # Test round trip
    u1, v1 = batoid.utils.dirCosToPostel(*batoid.utils.postelToDirCos(u, v))
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in the other direction
    alpha = np.random.uniform(-0.1, 0.1, size=1000)
    beta = np.random.uniform(-0.1, 0.1, size=1000)
    gamma = np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.postelToDirCos(
        *batoid.utils.dirCosToPostel(alpha, beta, gamma)
    )
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # For really tiny angles, u/v should be basically the same as alpha/beta
    u = np.random.uniform(-1e-6, 1e-6, size=1000)
    v = np.random.uniform(-1e-6, 1e-6, size=1000)
    alpha, beta, gamma = batoid.utils.postelToDirCos(u, v)
    np.testing.assert_allclose(alpha, u, rtol=0, atol=1e-8)
    np.testing.assert_allclose(beta, v, rtol=0, atol=1e-8)

    # Check normalization of direction cosines
    np.testing.assert_allclose(np.sqrt(alpha*alpha+beta*beta+gamma*gamma), 1, rtol=0, atol=1e-15)


@timer
def test_gnomonicSpherical():
    np.random.seed(57)
    u = np.random.uniform(-0.1, 0.1, size=1000)
    v = np.random.uniform(-0.1, 0.1, size=1000)

    # Test round trip
    u1, v1 = batoid.utils.sphericalToGnomonic(*batoid.utils.gnomonicToSpherical(u, v))
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in other direction
    phi = np.random.uniform(0.0, 0.1, size=1000)
    theta = np.random.uniform(-np.pi, np.pi, size=1000)
    phi1, theta1 = batoid.utils.gnomonicToSpherical(*batoid.utils.sphericalToGnomonic(phi, theta))
    np.testing.assert_allclose(phi, phi1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(theta, theta1, rtol=1e-10, atol=1e-12)

    # Check u**2 + v**2 = tan(phi)**2
    u, v = batoid.utils.sphericalToGnomonic(phi, theta)
    np.testing.assert_allclose(np.tan(phi)**2, u**2+v**2, rtol=0, atol=1e-17)

    # Check v/u = tan(theta)
    np.testing.assert_allclose(np.tan(theta), v/u, rtol=1e-15, atol=0)


@timer
def test_sphericalToDirCos():
    np.random.seed(577)
    phi = np.random.uniform(0.0, 0.1, size=1000)
    theta = np.random.uniform(-np.pi, np.pi, size=1000)

    # Test round trip
    phi1, theta1 = batoid.utils.dirCosToSpherical(*batoid.utils.sphericalToDirCos(phi, theta))
    np.testing.assert_allclose(phi, phi1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(theta, theta1, rtol=1e-10, atol=1e-12)

    # Test round trip in other direction
    alpha = np.random.uniform(-0.1, 0.1, size=1000)
    beta = np.random.uniform(-0.1, 0.1, size=1000)
    gamma = np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.sphericalToDirCos(
        *batoid.utils.dirCosToSpherical(alpha, beta, gamma)
    )
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # Check normalization of direction cosines
    np.testing.assert_allclose(np.sqrt(alpha1*alpha1+beta1*beta1+gamma1*gamma1), 1, rtol=0, atol=1e-15)


@timer
def test_composition():
    np.random.seed(5772)

    # Let's try spherical -> dirCos -> gnomonic = spherical -> gnomonic
    phi = np.random.uniform(0.0, 0.1, size=1000)
    theta = np.random.uniform(-np.pi, np.pi, size=1000)

    u1, v1 = batoid.utils.dirCosToGnomonic(*batoid.utils.sphericalToDirCos(phi, theta))
    u2, v2 = batoid.utils.sphericalToGnomonic(phi, theta)
    np.testing.assert_allclose(u1, u2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-10)

    # And cycle: gnomonic -> spherical -> dirCos = gnomonic -> dirCos
    u = np.random.uniform(-0.1, 0.1, size=1000)
    v = np.random.uniform(-0.1, 0.1, size=1000)
    a1, b1, c1 = batoid.utils.sphericalToDirCos(*batoid.utils.gnomonicToSpherical(u, v))
    a2, b2, c2 = batoid.utils.gnomonicToDirCos(u, v)
    np.testing.assert_allclose(a1, a2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b1, b2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(c1, c2, rtol=1e-10, atol=1e-10)

    # And cycle: dirCos -> gnomonic -> spherical = dirCos -> spherical
    a = np.random.uniform(-0.1, 0.1, size=1000)
    b = np.random.uniform(-0.1, 0.1, size=1000)
    c = np.sqrt(1 - a*a - b*b)
    ph1, th1 = batoid.utils.gnomonicToSpherical(*batoid.utils.dirCosToGnomonic(a, b, c))
    ph2, th2 = batoid.utils.dirCosToSpherical(a, b, c)
    np.testing.assert_allclose(ph1, ph2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(th1, th2, rtol=1e-10, atol=1e-10)

    # And reverse direction: gnomonic -> dirCos -> spherical = gnomonic -> spherical
    u = np.random.uniform(-0.1, 0.1, size=1000)
    v = np.random.uniform(-0.1, 0.1, size=1000)
    ph1, th1 = batoid.utils.dirCosToSpherical(*batoid.utils.gnomonicToDirCos(u, v))
    ph2, th2 = batoid.utils.gnomonicToSpherical(u, v)
    np.testing.assert_allclose(ph1, ph2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(th1, th2, rtol=1e-10, atol=1e-10)

    # and cycle: spherical -> gnomonic -> dirCos = spherical -> dirCos
    phi = np.random.uniform(0.0, 0.1, size=1000)
    theta = np.random.uniform(-np.pi, np.pi, size=1000)
    a1, b1, c1 = batoid.utils.gnomonicToDirCos(*batoid.utils.sphericalToGnomonic(phi, theta))
    a2, b2, c2 = batoid.utils.sphericalToDirCos(phi, theta)
    np.testing.assert_allclose(a1, a2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b1, b2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(c1, c2, rtol=1e-10, atol=1e-10)

    # and cycle: dirCos -> spherical -> gnomonic = dirCos -> gnomonic
    a = np.random.uniform(-0.1, 0.1, size=1000)
    b = np.random.uniform(-0.1, 0.1, size=1000)
    c = np.sqrt(1 - a*a - b*b)
    u1, v1 = batoid.utils.sphericalToGnomonic(*batoid.utils.dirCosToSpherical(a, b, c))
    u2, v2 = batoid.utils.dirCosToGnomonic(a, b, c)
    np.testing.assert_allclose(u1, u2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-10)


@timer
def test_jacobian():
    np.random.seed(57721)

    u = np.random.uniform(-0.1, 0.1, size=1000)
    v = np.random.uniform(-0.1, 0.1, size=1000)
    phi, theta = batoid.utils.gnomonicToSpherical(u, v)

    jac1 = batoid.utils.dSphericalDGnomonic(u, v)
    jac2 = batoid.utils.dGnomonicDSpherical(phi, theta)

    # Check that the product of the jacobian and its inverse is the identity matrix
    np.testing.assert_allclose(
        np.matmul(np.transpose(jac1, (2,0,1)), np.transpose(jac2, (2,0,1))),
        np.transpose(np.tile(np.eye(2)[:,:,None], 1000), (2,0,1)),
        rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        np.matmul(np.transpose(jac2, (2,0,1)), np.transpose(jac1, (2,0,1))),
        np.transpose(np.tile(np.eye(2)[:,:,None], 1000), (2,0,1)),
        rtol=0, atol=1e-15
    )

    # Check d(u, v)/d(phi, theta) against finite difference
    du = dv = 1e-8
    phi2, theta2 = batoid.utils.gnomonicToSpherical(u+du, v)
    phi3, theta3 = batoid.utils.gnomonicToSpherical(u, v+dv)
    np.testing.assert_allclose(
        jac1[0,0,:],
        (phi2-phi)/du,
        atol=1e-5
    )
    np.testing.assert_allclose(
        jac1[0,1,:],
        (phi3-phi)/du,
        atol=1e-5
    )
    np.testing.assert_allclose(
        jac1[1,0,:],
        np.sin(phi)*(theta2-theta)/du,
        atol=1e-5
    )
    np.testing.assert_allclose(
        jac1[1,1,:],
        np.sin(phi)*(theta3-theta)/dv,
        atol=1e-5
    )

    # Check d(phi, theta)/d(u, v) too
    dphi = 1e-8
    dtheta = 1e-8

    u2, v2 = batoid.utils.sphericalToGnomonic(phi+dphi, theta)
    u3, v3 = batoid.utils.sphericalToGnomonic(phi, theta+dtheta)

    np.testing.assert_allclose(
        jac2[0,0,:],
        (u2-u)/dphi,
        atol=1e-5
    )
    np.testing.assert_allclose(
        jac2[0,1,:],
        (u3-u)/dtheta/np.sin(phi),
        atol=1e-5
    )
    np.testing.assert_allclose(
        jac2[1,0,:],
        (v2-v)/dphi,
        atol=1e-5
    )
    np.testing.assert_allclose(
        jac2[1,1,:],
        (v3-v)/dtheta/np.sin(phi),
        atol=1e-5
    )


if __name__ == '__main__':
    test_normalized()
    test_gnomonicDirCos()
    test_postelDirCos()
    test_gnomonicSpherical()
    test_sphericalToDirCos()
    test_composition()
    test_jacobian()
