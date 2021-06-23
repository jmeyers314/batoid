import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff, init_gpu, rays_allclose


@timer
def test_properties():
    rng = np.random.default_rng(5)
    for i in range(100):
        R = rng.normal(0.0, 0.3)  # negative allowed
        conic = rng.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        assert quad.R == R
        assert quad.conic == conic
        do_pickle(quad)


def quadric_sag(R, conic):
    def f(x, y):
        r2 = x*x + y*y
        den = R*(1+np.sqrt(1-(1+conic)*r2/R/R))
        return r2/den
    return f


def quadric_normal(R, conic):
    def f(x, y):
        r = np.hypot(x, y)
        dzdr = r/(R*np.sqrt(1-r*r*(1.+conic)/R/R))
        nz = 1./np.sqrt(1+dzdr*dzdr)
        return np.dstack([-x/r*dzdr*nz, -y/r*dzdr*nz, nz])
    return f


@timer
def test_sag():
    rng = np.random.default_rng(57)
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)
        conic = rng.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        for j in range(100):
            lim = 0.7*abs(R)/np.sqrt(1+conic) if conic > -1 else 1
            x = rng.uniform(-lim, lim)
            y = rng.uniform(-lim, lim)
            result = quad.sag(x, y)
            np.testing.assert_allclose(result, quadric_sag(R, conic)(x, y))
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check vectorization
        x = rng.uniform(-lim, lim, size=(10, 10))
        y = rng.uniform(-lim, lim, size=(10, 10))
        np.testing.assert_allclose(quad.sag(x, y), quadric_sag(R, conic)(x, y))
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            quad.sag(x[::5,::2], y[::5,::2]),
            quadric_sag(R, conic)(x, y)[::5,::2]
        )


@timer
def test_normal():
    rng = np.random.default_rng(577)
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)
        conic = rng.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        for j in range(10):
            lim = 0.7*abs(R)/np.sqrt(1+conic) if conic > -1 else 1
            x = rng.uniform(-lim, lim)
            y = rng.uniform(-lim, lim)
            result = quad.normal(x, y)
            np.testing.assert_allclose(
                result,
                quadric_normal(R, conic)(x, y)[0,0]
            )
        # Check 0,0
        np.testing.assert_equal(quad.normal(0, 0), np.array([0, 0, 1]))
        # Check vectorization
        x = rng.uniform(-lim, lim, size=(10, 10))
        y = rng.uniform(-lim, lim, size=(10, 10))
        np.testing.assert_allclose(
            quad.normal(x, y),
            quadric_normal(R, conic)(x, y)
        )
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            quad.normal(x[::5,::2], y[::5,::2]),
            quadric_normal(R, conic)(x, y)[::5, ::2]
        )


@timer
def test_intersect():
    rng = np.random.default_rng(5772)
    size = 10_000
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)
        conic = rng.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        quadCoordSys = batoid.CoordSys(origin=[0, 0, -1])
        lim = min(0.7*abs(R)/np.sqrt(1+conic) if conic > -1 else 10, 10)
        x = rng.uniform(-lim, lim, size=size)
        y = rng.uniform(-lim, lim, size=size)
        z = np.full_like(x, -100.0)
        # If we shoot rays straight up, then it's easy to predict the intersection
        vx = np.zeros_like(x)
        vy = np.zeros_like(x)
        vz = np.ones_like(x)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        np.testing.assert_allclose(rv.z, -100.0)
        rv2 = batoid.intersect(quad, rv.copy(), quadCoordSys)
        assert rv2.coordSys == quadCoordSys

        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(rv2.z, quad.sag(x, y)-1, rtol=0, atol=1e-9)

        # Check default intersect coordTransform
        rv2 = rv.copy().toCoordSys(quadCoordSys)
        batoid.intersect(quad, rv2)
        assert rv2.coordSys == quadCoordSys
        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(rv2.z, quad.sag(x, y)-1, rtol=0, atol=1e-9)


@timer
def test_reflect():
    rng = np.random.default_rng(57721)
    size = 10_000
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)
        conic = rng.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        lim = min(0.7*abs(R)/np.sqrt(1+conic) if conic > -1 else 10, 10)
        x = rng.uniform(-lim, lim, size=size)
        y = rng.uniform(-lim, lim, size=size)
        z = np.full_like(x, -100.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.full_like(x, 1)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.reflect(quad, rv.copy())
        rvr2 = quad.reflect(rv.copy())
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = quad.normal(rvr.x, rvr.y)

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
    rng = np.random.default_rng(577215)
    size = 10_000
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)
        conic = rng.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        m0 = batoid.ConstMedium(rng.normal(1.2, 0.01))
        m1 = batoid.ConstMedium(rng.normal(1.3, 0.01))
        lim = min(0.7*abs(R)/np.sqrt(1+conic) if conic > -1 else 10, 10)
        x = rng.uniform(-lim, lim, size=size)
        y = rng.uniform(-lim, lim, size=size)
        z = np.full_like(x, -100.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.sqrt(1-vx*vx-vy*vy)/m0.n
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.refract(quad, rv.copy(), m0, m1)
        rvr2 = quad.refract(rv.copy(), m0, m1)
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = quad.normal(rvr.x, rvr.y)

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
        batoid.Quadric(10.0, 1.0),
        batoid.Quadric(11.0, 1.0),
        batoid.Quadric(10.0, 1.1),
        batoid.Sphere(10.0)
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    quad = batoid.Quadric(1.0, 0.0)
    rv = batoid.RayVector(0, 10, 0, 0, 0, -1)  # Too far to side
    rv2 = batoid.intersect(quad, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([True]))
    # This one passes
    rv = batoid.RayVector(0, 0, -1, 0, 0, -1)
    rv2 = batoid.intersect(quad, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([False]))


@timer
def test_sphere():
    rng = np.random.default_rng(5772156)
    size = 1000
    for i in range(100):
        R = 1/rng.normal(0.0, 0.3)
        conic = 0.0
        quad = batoid.Quadric(R, conic)
        sphere = batoid.Sphere(R)
        lim = 0.7*abs(R)
        x = rng.uniform(-lim, lim, size=size)
        y = rng.uniform(-lim, lim, size=size)
        np.testing.assert_allclose(
            quad.sag(x,y), sphere.sag(x, y),
            rtol=0, atol=1e-11
        )
        np.testing.assert_allclose(
            quad.normal(x,y), sphere.normal(x, y),
            rtol=0, atol=1e-11
        )


@timer
def test_paraboloid():
    rng = np.random.default_rng(57721566)
    size = 1000
    for i in range(100):
        R = 1/rng.normal(0.0, 0.3)
        conic = -1.0
        quad = batoid.Quadric(R, conic)
        para = batoid.Paraboloid(R)
        lim = 0.7*abs(R)
        x = rng.uniform(-lim, lim, size=size)
        y = rng.uniform(-lim, lim, size=size)
        np.testing.assert_allclose(
            quad.sag(x,y), para.sag(x, y),
            rtol=0, atol=1e-11
        )
        np.testing.assert_allclose(
            quad.normal(x,y), para.normal(x, y),
            rtol=0, atol=1e-11
        )


if __name__ == '__main__':
    init_gpu()
    test_properties()
    test_sag()
    test_normal()
    test_intersect()
    test_reflect()
    test_refract()
    test_ne()
    test_fail()
    test_sphere()
    test_paraboloid()
