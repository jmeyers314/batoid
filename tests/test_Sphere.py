import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff


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
    for i in range(10):
        R = rng.normal(0.0, 0.3)
        sphereCoordSys = batoid.CoordSys(origin=[0, 0, -1])
        sphere = batoid.Sphere(R)
        x = rng.uniform(-0.7*abs(R), 0.7*abs(R), size=size)
        y = rng.uniform(-0.7*abs(R), 0.7*abs(R), size=size)
        z = np.full_like(x, -100.0)
        # If we shoot rays straight up, then it's easy to predict the intersection
        vx = np.zeros_like(x)
        vy = np.zeros_like(x)
        vz = np.ones_like(x)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        np.testing.assert_allclose(rv.z, -100.0)
        coordTransform = batoid.CoordTransform(rv.coordSys, sphereCoordSys)
        rv2 = batoid.intersect(sphere, rv.copy(), coordTransform)
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
    # This one should fail, since already passed the paraboloid.
    rv = batoid.RayVector(0, 0, -1, 0, 0, -1)
    rv2 = batoid.intersect(sphere, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([True]))
    # This one passes
    rv = batoid.RayVector(0, 0, -1, 0, 0, +1)
    rv2 = batoid.intersect(sphere, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([False]))


if __name__ == '__main__':
    test_properties()
    test_sag()
    test_normal()
    test_intersect()
    test_ne()
    test_fail()
