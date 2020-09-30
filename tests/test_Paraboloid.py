import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff, init_gpu, rays_allclose


@timer
def test_properties():
    rng = np.random.default_rng(5)
    for i in range(100):
        R = rng.normal(0.0, 0.3)  # negative allowed
        para = batoid.Paraboloid(R)
        assert para.R == R
        do_pickle(para)


@timer
def test_sag():
    rng = np.random.default_rng(57)
    for i in range(100):
        R = rng.normal(0.0, 0.3)
        para = batoid.Paraboloid(R)
        for j in range(10):
            x = rng.normal()
            y = rng.normal()
            result = para.sag(x, y)
            np.testing.assert_allclose(result, (x*x + y*y)/2/R)
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check vectorization
        x = rng.normal(size=(10, 10))
        y = rng.normal(size=(10, 10))
        np.testing.assert_allclose(para.sag(x, y), (x*x + y*y)/2/R)
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(para.sag(x[::5,::2], y[::5,::2]), ((x*x + y*y)/2/R)[::5,::2])


@timer
def test_normal():
    rng = np.random.default_rng(577)
    for i in range(100):
        R = rng.normal(0.0, 0.3)
        para = batoid.Paraboloid(R)
        for j in range(10):
            x = rng.normal()
            y = rng.normal()
            result = para.normal(x, y)
            normal = np.array([-x/R, -y/R, 1])
            normal /= np.sqrt(np.sum(np.square(normal)))
            np.testing.assert_allclose(result, normal)
        # Check 0,0
        np.testing.assert_equal(para.normal(0, 0), np.array([0, 0, 1]))
        # Check vectorization
        x = rng.normal(size=(10, 10))
        y = rng.normal(size=(10, 10))
        normal = np.dstack([-x/R, -y/R, np.ones_like(x)])
        normal /= np.sqrt(np.sum(np.square(normal), axis=-1))[..., None]
        np.testing.assert_allclose(
            para.normal(x, y),
            normal
        )
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            para.normal(x[::5,::2], y[::5,::2]),
            normal[::5, ::2]
        )


@timer
def test_intersect():
    rng = np.random.default_rng(5772)
    size = 10_000
    for i in range(10):
        R = 1./rng.normal(0.0, 0.3)
        paraCoordSys = batoid.CoordSys(origin=[0, 0, -1])
        para = batoid.Paraboloid(R)
        x = rng.normal(0.0, 1.0, size=size)
        y = rng.normal(0.0, 1.0, size=size)
        z = np.full_like(x, -100.0)
        # If we shoot rays straight up, then it's easy to predict the intersection
        vx = np.zeros_like(x)
        vy = np.zeros_like(x)
        vz = np.ones_like(x)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        np.testing.assert_allclose(rv.z, -100.0)
        rv2 = batoid.intersect(para, rv.copy(), paraCoordSys)
        assert rv2.coordSys == paraCoordSys
        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(rv2.z, para.sag(x, y)-1, rtol=0, atol=1e-12)

        # Check default intersect coordTransform
        rv2 = rv.copy().toCoordSys(paraCoordSys)
        batoid.intersect(para, rv2)
        assert rv2.coordSys == paraCoordSys
        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(rv2.z, para.sag(x, y)-1, rtol=0, atol=1e-12)


@timer
def test_reflect():
    rng = np.random.default_rng(57721)
    size = 10_000
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)  # negative allowed
        para = batoid.Paraboloid(R)
        x = rng.normal(0.0, 1.0, size=size)
        y = rng.normal(0.0, 1.0, size=size)
        z = np.full_like(x, -100.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.full_like(x, 1)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.reflect(para, rv.copy())
        rvr2 = para.reflect(rv.copy())
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = para.normal(rvr.x, rvr.y)

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
        R = 1./rng.normal(0.0, 0.3)  # negative allowed
        para = batoid.Paraboloid(R)
        m0 = batoid.ConstMedium(rng.normal(1.2, 0.01))
        m1 = batoid.ConstMedium(rng.normal(1.3, 0.01))
        x = rng.normal(0.0, 1.0, size=size)
        y = rng.normal(0.0, 1.0, size=size)
        z = np.full_like(x, -100.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.sqrt(1-vx*vx-vy*vy)/m0.n
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.refract(para, rv.copy(), m0, m1)
        rvr2 = para.refract(rv.copy(), m0, m1)
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = para.normal(rvr.x, rvr.y)

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
def test_reflect_to_focus():
    rng = np.random.default_rng(5772156)
    size = 10_000
    for i in range(100):
        R = rng.normal(0, 3.0)
        para = batoid.Paraboloid(R)
        x = rng.normal(size=size)
        y = rng.normal(size=size)
        rv = batoid.RayVector(x, y, -1000, 0, 0, 1)
        para.reflect(rv)
        # Now, see if rays pass through paraboloid focus at (0, 0, R/2)
        # Solve 0 = x + vx (t - t0) for t, then propagate to that t
        t = rv.t[0] - rv.r[0,0]/rv.v[0,0]
        focus = rv.positionAtTime(t)
        np.testing.assert_allclose(focus[:,0], 0, atol=1e-12)
        np.testing.assert_allclose(focus[:,1], 0, atol=1e-12)
        np.testing.assert_allclose(focus[:,2], R/2, atol=1e-12)


@timer
def test_ne():
    objs = [
        batoid.Paraboloid(1.0),
        batoid.Paraboloid(2.0),
        batoid.Plane()
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    para = batoid.Paraboloid(1.0)
    rv = batoid.RayVector(0, 0, -10, 0, 0.99, np.sqrt(1-0.99**2))  # Too shallow
    rv2 = batoid.intersect(para, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([True]))
    # This one passes
    rv = batoid.RayVector(0, 0, -1, 0, 0, -1)
    rv2 = batoid.intersect(para, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([False]))


if __name__ == '__main__':
    init_gpu()
    test_properties()
    test_sag()
    test_normal()
    test_intersect()
    test_reflect()
    test_refract()
    test_reflect_to_focus()
    test_ne()
    test_fail()
