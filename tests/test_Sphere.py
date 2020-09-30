import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff, init_gpu, rays_allclose


@timer
def test_properties():
    rng = np.random.default_rng(5)
    for i in range(100):
        R = rng.normal(0.0, 0.3)  # negative allowed
        sphere = batoid.Sphere(R)
        assert sphere.R == R
        do_pickle(sphere)


@timer
def test_sag():
    rng = np.random.default_rng(57)
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)
        sphere = batoid.Sphere(R)
        for j in range(10):
            x = rng.uniform(-0.7*abs(R), 0.7*abs(R))
            y = rng.uniform(-0.7*abs(R), 0.7*abs(R))
            result = sphere.sag(x, y)
            np.testing.assert_allclose(
                result,
                R*(1-np.sqrt(1.0-(x*x + y*y)/R/R))
            )
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check 0,0
        np.testing.assert_allclose(sphere.sag(0, 0), 0.0, rtol=0, atol=1e-17)
        # Check vectorization
        x = rng.uniform(-0.7*abs(R), 0.7*abs(R), size=(10, 10))
        y = rng.uniform(-0.7*abs(R), 0.7*abs(R), size=(10, 10))
        np.testing.assert_allclose(
            sphere.sag(x, y),
            R*(1-np.sqrt(1.0-(x*x + y*y)/R/R))
        )
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            sphere.sag(x[::5,::2], y[::5,::2]),
            R*(1-np.sqrt(1.0-(x*x + y*y)/R/R))[::5,::2]
        )
        do_pickle(sphere)


@timer
def test_normal():
    rng = np.random.default_rng(577)
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)
        sphere = batoid.Sphere(R)
        for j in range(10):
            x = rng.uniform(-0.7*abs(R), 0.7*abs(R))
            y = rng.uniform(-0.7*abs(R), 0.7*abs(R))
            result = sphere.normal(x, y)
            r = np.hypot(x, y)
            rat = r/R
            dzdr = rat/np.sqrt(1-rat*rat)
            nz = 1/np.sqrt(1+dzdr*dzdr)
            normal = np.array([-x/r*dzdr*nz, -y/r*dzdr*nz, nz])
            np.testing.assert_allclose(result, normal)
        # Check 0,0
        np.testing.assert_equal(sphere.normal(0, 0), np.array([0, 0, 1]))
        # Check vectorization
        x = rng.uniform(-0.7*abs(R), 0.7*abs(R), size=(10, 10))
        y = rng.uniform(-0.7*abs(R), 0.7*abs(R), size=(10, 10))
        r = np.hypot(x, y)
        rat = r/R
        dzdr = rat/np.sqrt(1-rat*rat)
        nz = 1/np.sqrt(1+dzdr*dzdr)
        normal = np.dstack([-x/r*dzdr*nz, -y/r*dzdr*nz, nz])
        np.testing.assert_allclose(
            sphere.normal(x, y),
            normal
        )
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            sphere.normal(x[::5,::2], y[::5,::2]),
            normal[::5, ::2]
        )


@timer
def test_intersect():
    rng = np.random.default_rng(5772)
    size = 10_000
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)
        sphereCoordSys = batoid.CoordSys(origin=[0, 0, -1])
        sphere = batoid.Sphere(R)
        x = rng.uniform(-0.3*abs(R), 0.3*abs(R), size=size)
        y = rng.uniform(-0.3*abs(R), 0.3*abs(R), size=size)
        z = np.full_like(x, -2*abs(R))
        # If we shoot rays straight up, then it's easy to predict the intersection
        vx = np.zeros_like(x)
        vy = np.zeros_like(x)
        vz = np.ones_like(x)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        np.testing.assert_allclose(rv.z, -2*abs(R))
        rv2 = batoid.intersect(sphere, rv.copy(), sphereCoordSys)
        assert rv2.coordSys == sphereCoordSys

        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(rv2.z, sphere.sag(x, y)-1, rtol=0, atol=1e-9)

        # Check default intersect coordTransform
        rv2 = rv.copy().toCoordSys(sphereCoordSys)
        batoid.intersect(sphere, rv2)
        assert rv2.coordSys == sphereCoordSys
        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(rv2.z, sphere.sag(x, y)-1, rtol=0, atol=1e-9)


@timer
def test_reflect():
    rng = np.random.default_rng(57721)
    size = 10_000
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)
        sphere = batoid.Sphere(R)
        x = rng.uniform(-0.3*abs(R), 0.3*abs(R), size=size)
        y = rng.uniform(-0.3*abs(R), 0.3*abs(R), size=size)
        z = np.full_like(x, -2*abs(R))
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.full_like(x, 1)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.reflect(sphere, rv.copy())
        rvr2 = sphere.reflect(rv.copy())
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = sphere.normal(rvr.x, rvr.y)

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
        sphere = batoid.Sphere(R)
        m0 = batoid.ConstMedium(rng.normal(1.2, 0.01))
        m1 = batoid.ConstMedium(rng.normal(1.3, 0.01))
        x = rng.uniform(-0.3*abs(R), 0.3*abs(R), size=size)
        y = rng.uniform(-0.3*abs(R), 0.3*abs(R), size=size)
        z = np.full_like(x, -2*abs(R))
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.sqrt(1-vx*vx-vy*vy)/m0.n
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.refract(sphere, rv.copy(), m0, m1)
        rvr2 = sphere.refract(rv.copy(), m0, m1)
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = sphere.normal(rvr.x, rvr.y)

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
        batoid.Sphere(1.0),
        batoid.Sphere(2.0),
        batoid.Plane()
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    sphere = batoid.Sphere(1.0)
    rv = batoid.RayVector(0, 10, 0, 0, 0, -1)  # Too far to side
    rv2 = batoid.intersect(sphere, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([True]))
    # This one passes
    rv = batoid.RayVector(0, 0, 0, 0, 0, -1)
    rv2 = batoid.intersect(sphere, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([False]))


if __name__ == '__main__':
    test_properties()
    test_sag()
    test_normal()
    test_intersect()
    test_reflect()
    test_refract()
    test_ne()
    test_fail()
