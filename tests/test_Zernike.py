import batoid
import galsim
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff, rays_allclose


@timer
def test_horner2d():
    rng = np.random.default_rng(5)
    for _ in range(1000):
        nx = rng.integers(1, 21)
        ny = rng.integers(1, 21)
        arr = rng.normal(size=(ny, nx))
        x = rng.normal()
        y = rng.normal()
        np.testing.assert_allclose(
            batoid._batoid.horner2d(x, y, arr.ctypes.data, nx, ny),
            galsim.utilities.horner2d(x, y, arr),
            atol=1e-13,
            rtol=1e-13
        )


@timer
def test_properties():
    rng = np.random.default_rng(57)
    for _ in range(10):
        jmax = rng.integers(4, 55)
        coef = rng.normal(size=jmax+1)*1e-3
        R_outer = rng.uniform(0.5, 5.0)
        R_inner = rng.uniform(0.0, 0.65*R_outer)
        zernike = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)

        assert R_outer == zernike.R_outer
        assert R_inner == zernike.R_inner
        assert np.array_equal(coef, zernike.coef)

        do_pickle(zernike)


@timer
def test_sag():
    rng = np.random.default_rng(577)
    for _ in range(10):
        jmax = rng.integers(4, 100)
        coef = rng.normal(size=jmax+1)*1e-3
        R_outer = rng.uniform(0.5, 5.0)
        R_inner = rng.uniform(0.0, 0.65*R_outer)

        gz = galsim.zernike.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        bz = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        lim = 0.7*R_outer

        for j in range(100):
            x = rng.uniform(-lim, lim)
            y = rng.uniform(-lim, lim)
            result = bz.sag(x, y)
            np.testing.assert_allclose(
                gz.evalCartesian(x, y),
                result,
                rtol=0,
                atol=1e-12
            )
            assert isinstance(result, float)

        # Check vectorization
        x = rng.uniform(-lim, lim, size=(10, 10))
        y = rng.uniform(-lim, lim, size=(10, 10))
        np.testing.assert_allclose(
            gz.evalCartesian(x, y),
            bz.sag(x, y),
            rtol=0,
            atol=1e-12
        )
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            bz.sag(x[::5,::2], y[::5,::2]),
            gz.evalCartesian(x, y)[::5,::2],
            rtol=0,
            atol=1e-12
        )


@timer
def test_normal():
    rng = np.random.default_rng(5772)
    for _ in range(10):
        jmax = rng.integers(4, 100)
        coef = rng.normal(size=jmax+1)*1e-3
        R_outer = rng.uniform(0.5, 5.0)
        R_inner = rng.uniform(0.0, 0.65*R_outer)

        gz = galsim.zernike.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        bz = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        lim = 0.7*R_outer

        for j in range(10):
            x = rng.uniform(-lim, lim)
            y = rng.uniform(-lim, lim)
            dzdx = gz.gradX(x, y)
            dzdy = gz.gradY(x, y)
            nz = 1./np.sqrt(dzdx*dzdx + dzdy*dzdy + 1)
            nx = -dzdx*nz
            ny = -dzdy*nz
            prediction = np.array([nx, ny, nz]).T
            np.testing.assert_allclose(
                bz.normal(x, y),
                prediction,
                atol=1e-13,
                rtol=1e-13
            )
        # Check vectorization
        x = rng.uniform(-lim, lim, size=(10, 10))
        y = rng.uniform(-lim, lim, size=(10, 10))
        dzdx = gz.gradX(x, y)
        dzdy = gz.gradY(x, y)
        nz = 1./np.sqrt(dzdx*dzdx + dzdy*dzdy + 1)
        nx = -dzdx*nz
        ny = -dzdy*nz
        prediction = np.moveaxis(np.array([nx, ny, nz]), 0, -1)

        np.testing.assert_allclose(
            bz.normal(x, y),
            prediction,
            atol=1e-13,
            rtol=1e-13
        )
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            bz.normal(x[::5,::2], y[::5,::2]),
            prediction[::5, ::2],
            rtol=1e-13, atol=1e-13
        )


@timer
def test_intersect():
    rng = np.random.default_rng(57721)
    size = 10_000

    for i in range(10):
        jmax = rng.integers(4, 100)
        coef = rng.normal(size=jmax+1)*1e-3
        R_outer = rng.uniform(0.5, 5.0)
        R_inner = rng.uniform(0.0, 0.65*R_outer)

        zernike = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        lim = 0.7*R_outer

        zernikeCoordSys = batoid.CoordSys(origin=[0, 0, -1])
        x = rng.uniform(-lim, lim, size=size)
        y = rng.uniform(-lim, lim, size=size)
        z = np.full_like(x, -10.0)
        # If we shoot rays straight up, then it's easy to predict the
        # intersection
        vx = np.zeros_like(x)
        vy = np.zeros_like(x)
        vz = np.ones_like(x)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        np.testing.assert_allclose(rv.z, -10.0)
        rv2 = batoid.intersect(zernike, rv.copy(), zernikeCoordSys)
        assert rv2.coordSys == zernikeCoordSys

        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(
            rv2.z, zernike.sag(x, y)-1,
            rtol=0, atol=1e-12
        )

        # Check default intersect coordTransform
        rv2 = rv.copy().toCoordSys(zernikeCoordSys)
        batoid.intersect(zernike, rv2)
        assert rv2.coordSys == zernikeCoordSys
        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(
            rv2.z, zernike.sag(x, y)-1,
            rtol=0, atol=1e-12
        )


@timer
def test_reflect():
    rng = np.random.default_rng(577215)
    size = 10_000

    for i in range(10):
        jmax = rng.integers(4, 36+1)
        coef = rng.normal(size=jmax+1)*1e-3
        R_outer = rng.uniform(0.5, 5.0)
        R_inner = rng.uniform(0.0, 0.65*R_outer)

        zernike = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        lim = 0.7*R_outer

        x = rng.uniform(-lim, lim, size=size)
        y = rng.uniform(-lim, lim, size=size)
        z = np.full_like(x, -1.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.full_like(x, 1)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.reflect(zernike, rv.copy())
        rvr2 = zernike.reflect(rv.copy())
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = zernike.normal(rvr.x, rvr.y)

        # Test law of reflection
        a0 = np.einsum("ad,ad->a", normal, rv.v)[~rvr.failed]
        a1 = np.einsum("ad,ad->a", normal, -rvr.v)[~rvr.failed]
        np.testing.assert_allclose(
            a0, a1,
            rtol=0, atol=1e-12
        )

        # Test that rv.v, rvr.v and normal are all in the same plane
        np.testing.assert_allclose(
            np.einsum(
                "ad,ad->a",
                np.cross(normal, rv.v),
                rv.v
            )[~rvr.failed],
            0.0,
            rtol=0, atol=1e-12
        )


@timer
def test_refract():
    rng = np.random.default_rng(5772156)
    size = 10_000

    for i in range(10):
        jmax = rng.integers(4, 100)
        coef = rng.normal(size=jmax+1)*1e-3
        R_outer = rng.uniform(0.5, 5.0)
        R_inner = rng.uniform(0.0, 0.65*R_outer)

        zernike = batoid.Zernike(coef, R_outer=R_outer, R_inner=R_inner)
        lim = 0.7*R_outer
        m0 = batoid.ConstMedium(rng.normal(1.2, 0.01))
        m1 = batoid.ConstMedium(rng.normal(1.3, 0.01))
        x = rng.uniform(-lim, lim, size=size)
        y = rng.uniform(-lim, lim, size=size)
        z = np.full_like(x, -10.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.sqrt(1-vx*vx-vy*vy)/m0.n
        rv = batoid.RayVector(x, y, z, vx, vy, vz, t=0)
        rvr = batoid.refract(zernike, rv.copy(), m0, m1)
        rvr2 = zernike.refract(rv.copy(), m0, m1)
        rays_allclose(rvr, rvr2, atol=1e-13)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = zernike.normal(rvr.x, rvr.y)

        # Test Snell's law
        s0 = np.sum(np.cross(normal, rv.v*m0.n)[~rvr.failed], axis=-1)
        s1 = np.sum(np.cross(normal, rvr.v*m1.n)[~rvr.failed], axis=-1)
        np.testing.assert_allclose(
            m0.n*s0, m1.n*s1,
            rtol=0, atol=1e-9
        )

        # Test that rv.v, rvr.v and normal are all in the same plane
        np.testing.assert_allclose(
            np.einsum(
                "ad,ad->a",
                np.cross(normal, rv.v),
                rv.v
            )[~rvr.failed],
            0.0,
            rtol=0, atol=1e-12
        )


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


@timer
def test_fail():
    zernike = batoid.Zernike([0,0,0,0,1])  # basically a paraboloid
    rv = batoid.RayVector(0, 0, -10, 0, 0.99, np.sqrt(1-0.99**2))  # Too shallow
    rv2 = batoid.intersect(zernike, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([True]))
    # This one passes
    rv = batoid.RayVector(0, 0, -1, 0, 0, -1)
    rv2 = batoid.intersect(zernike, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([False]))


if __name__ == '__main__':
    test_horner2d()
    test_properties()
    test_sag()
    test_normal()
    test_intersect()
    test_reflect()
    test_refract()
    test_ne()
    test_fail()
