import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff
import pytest
from batoid.utils import normalized


hasGalSim = True
try:
    import galsim
except ImportError:
    hasGalSim = False


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_horner2d():
    np.random.seed(5)
    for _ in range(1000):
        nx = np.random.randint(1, 20)
        ny = np.random.randint(1, 20)
        arr = np.random.normal(size=(nx, ny))
        x = np.random.normal()
        y = np.random.normal()

        np.testing.assert_allclose(
            batoid._batoid.horner2d(x, y, arr),
            galsim.utilities.horner2d(x, y, arr),
            atol=1e-17,
            rtol=1e-17
        )


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_sag():
    np.random.seed(57)
    jmaxmax=200
    for _ in range(100):
        jmax = np.random.randint(1, jmaxmax)
        coef = np.random.normal(size=jmax+1)*1e-3
        R_outer = np.random.uniform(0.5, 5.0)
        R_inner = np.random.uniform(0.0, 0.65*R_outer)

        gz = galsim.zernike.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        bz = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)

        x = np.random.uniform(-R_outer, R_outer, size=5000)
        y = np.random.uniform(-R_outer, R_outer, size=5000)
        w = np.hypot(x, y) < R_outer
        x = x[w]
        y = y[w]

        np.testing.assert_allclose(
            np.clip(gz.evalCartesian(x, y), -1.0, 1.0),
            np.clip(bz.sag(x, y), -1.0, 1.0),
            atol=1e-9,
            rtol=1e-6
        )

        np.testing.assert_allclose(
            gz.evalCartesian(x[::5], y[::5]),
            bz.sag(x[::5], y[::5]),
            atol=1e-9,
            rtol=1e-6
        )


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_properties():
    np.random.seed(577)
    jmaxmax=200
    for _ in range(100):
        jmax = np.random.randint(1, jmaxmax)
        coef = np.random.normal(size=jmax+1)*1e-3
        R_outer = np.random.uniform(0.5, 5.0)
        R_inner = np.random.uniform(0.0, 0.8*R_outer)
        zernike = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)

        assert np.all(zernike.coef == coef)
        assert zernike.R_outer == R_outer
        assert zernike.R_inner == R_inner
        do_pickle(zernike)


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_intersect():
    np.random.seed(5772)
    jmaxmax=50
    for i in range(100):
        jmax = np.random.randint(1, jmaxmax)
        coef = np.random.normal(size=jmax+1)*1e-6
        R_outer = np.random.uniform(0.5, 5.0)
        R_inner = np.random.uniform(0.0, 0.8*R_outer)
        zernike = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        for j in range(100):
            x = np.random.normal(0.0, 1.0)
            y = np.random.normal(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r0 = batoid.Ray(x, y, -10000, 0, 0, 1)
            r = zernike.intersect(r0)
            np.testing.assert_allclose(r.r[0], x, rtol=0, atol=1e-9)
            np.testing.assert_allclose(r.r[1], y, rtol=0, atol=1e-9)
            np.testing.assert_allclose(r.r[2], zernike.sag(x, y), rtol=0, atol=1e-9)


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_intersect_vectorized():
    np.random.seed(57721)
    jmaxmax=50
    r0s = [batoid.Ray([np.random.normal(0.0, 0.1),
                       np.random.normal(0.0, 0.1),
                       np.random.normal(10.0, 0.1)],
                      [np.random.normal(0.0, 0.1),
                       np.random.normal(0.0, 0.1),
                       np.random.normal(-1.0, 0.1)],
                      np.random.normal(0.0, 0.1))
            for i in range(100)]
    r0s = batoid.RayVector(r0s)

    for i in range(100):
        jmax = np.random.randint(1, jmaxmax)
        coef = np.random.normal(size=jmax+1)*1e-3
        R_outer = np.random.uniform(0.5, 5.0)
        R_inner = np.random.uniform(0.0, 0.8*R_outer)
        zernike = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)

        r1s = zernike.intersect(r0s.copy())
        r2s = batoid.RayVector([zernike.intersect(r0.copy()) for r0 in r0s])
        assert r1s == r2s


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_normal():
    np.random.seed(5772156)
    jmaxmax = 100
    for i in range(100):
        jmax = np.random.randint(1, jmaxmax)
        coef = np.random.normal(size=jmax+1)*1e-3
        R_outer = np.random.uniform(0.5, 5.0)
        R_inner = np.random.uniform(0.0, 0.8*R_outer)

        zernike = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        gradx = zernike.Z.gradX
        grady = zernike.Z.gradY

        x = np.random.uniform(-R_outer, R_outer, size=500)
        y = np.random.uniform(-R_outer, R_outer, size=500)
        w = np.hypot(x, y) < R_outer
        x = x[w]
        y = y[w]

        norm1 = zernike.normal(x, y)
        norm2 = np.stack([-gradx(x, y), -grady(x, y), np.ones_like(x)], axis=1)
        norm2 /= np.sqrt(np.sum(norm2**2, axis=1))[:, None]
        np.testing.assert_allclose(norm1, norm2, atol=1e-9)


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_ne():
    objs = [
        batoid.Zernike([0,0,0,0,1]),
        batoid.Zernike([0,0,0,1]),
        batoid.Zernike([0,0,0,0,1], R_outer=1.1),
        batoid.Zernike([0,0,0,0,1], R_inner=0.8),
        batoid.Zernike([0,0,0,0,1], R_outer=1.1, R_inner=0.8),
        batoid.Quadric(10.0, 1.0)
    ]
    all_obj_diff(objs)


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_fail():
    zernike = batoid.Zernike([0,0,0,0,1])
    ray = batoid.Ray([0,0,zernike.sag(0,0)-1], [0,0,-1])
    ray = zernike.intersect(ray)
    assert ray.failed

    ray = batoid.Ray([0,0,zernike.sag(0,0)-1], [0,0,-1])
    zernike.intersect(ray)
    assert ray.failed


if __name__ == '__main__':
    test_horner2d()
    test_sag()
    test_properties()
    test_intersect()
    test_intersect_vectorized()
    test_normal()
    test_ne()
    test_fail()
