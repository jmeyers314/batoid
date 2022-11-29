import batoid
import numpy as np
from test_helpers import timer, do_pickle, rays_allclose, all_obj_diff, init_gpu


@timer
def test_properties():
    rng = np.random.default_rng(5)
    for i in range(100):
        s1 = batoid.Sphere(rng.uniform(1, 3))
        s2 = batoid.Paraboloid(rng.uniform(1, 3))
        sum = batoid.Sum([s1, s2])
        do_pickle(sum)

        assert s1 is sum.surfaces[0]
        assert s2 is sum.surfaces[1]

        s3 = batoid.Quadric(rng.uniform(3, 5), rng.uniform(-0.1, 0.1))
        sum2 = batoid.Sum([s1, s2, s3])
        do_pickle(sum2)

        assert s1 is sum2.surfaces[0]
        assert s2 is sum2.surfaces[1]
        assert s3 is sum2.surfaces[2]

        # alternate constructions
        sum3 = batoid.Sum(s1, s2)
        sum4 = s1 + s2
        assert sum == sum3
        assert sum == sum4


@timer
def test_sag():
    rng = np.random.default_rng(57)
    for _ in range(100):
        s1 = batoid.Sphere(rng.uniform(1, 3))
        s2 = batoid.Paraboloid(rng.uniform(1, 3))
        sum = batoid.Sum([s1, s2])

        x = rng.normal(size=5000)
        y = rng.normal(size=5000)

        np.testing.assert_allclose(
            sum.sag(x, y),
            s1.sag(x, y) + s2.sag(x, y),
            rtol=0,
            atol=1e-12
        )

        s3 = batoid.Quadric(rng.uniform(3, 5), rng.uniform(-0.1, 0.1))
        sum2 = batoid.Sum([s1, s2, s3])

        np.testing.assert_allclose(
            sum2.sag(x, y),
            s1.sag(x, y) + s2.sag(x, y) + s3.sag(x, y),
            rtol=0,
            atol=1e-12
        )


@timer
def test_normal():
    rng = np.random.default_rng(577)
    for _ in range(100):
        s1 = batoid.Sphere(rng.uniform(1, 3))
        s2 = batoid.Paraboloid(rng.uniform(1, 3))
        sum = batoid.Sum([s1, s2])

        x = rng.normal(size=5000)
        y = rng.normal(size=5000)

        n1 = s1.normal(x, y)
        n2 = s2.normal(x, y)
        nx = n1[:, 0]/n1[:, 2] + n2[:, 0]/n2[:, 2]
        ny = n1[:, 1]/n1[:, 2] + n2[:, 1]/n2[:, 2]
        nz = 1./np.sqrt(nx*nx + ny*ny + 1)
        nx *= nz
        ny *= nz
        normal = np.array([nx, ny, nz]).T
        np.testing.assert_allclose(
            sum.normal(x, y),
            normal,
            rtol=0,
            atol=1e-12
        )

        s3 = batoid.Quadric(rng.uniform(3, 5), rng.uniform(-0.1, 0.1))
        sum2 = batoid.Sum([s1, s2, s3])
        n3 = s3.normal(x, y)
        nx = n1[:, 0]/n1[:, 2] + n2[:, 0]/n2[:, 2] + n3[:, 0]/n3[:, 2]
        ny = n1[:, 1]/n1[:, 2] + n2[:, 1]/n2[:, 2] + n3[:, 1]/n3[:, 2]
        nz = 1./np.sqrt(nx*nx + ny*ny + 1)
        nx *= nz
        ny *= nz
        normal = np.array([nx, ny, nz]).T

        np.testing.assert_allclose(
            sum2.normal(x, y),
            normal,
            rtol=0,
            atol=1e-12
        )


@timer
def test_intersect():
    rng = np.random.default_rng(5772)
    size = 10_000
    for _ in range(100):
        s1 = batoid.Sphere(1./rng.normal(0., 0.2))
        s2 = batoid.Paraboloid(rng.uniform(1, 3))
        sum = batoid.Sum([s1, s2])
        sumCoordSys = batoid.CoordSys(origin=[0, 0, -1])
        x = rng.uniform(-1, 1, size=size)
        y = rng.uniform(-1, 1, size=size)
        z = np.full_like(x, -10.0)
        # If we shoot rays straight up, then it's easy to predict the intersection
        vx = np.zeros_like(x)
        vy = np.zeros_like(x)
        vz = np.ones_like(x)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        np.testing.assert_allclose(rv.z, -10.0)
        rv2 = batoid.intersect(sum, rv.copy(), sumCoordSys)
        assert rv2.coordSys == sumCoordSys

        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(
            rv2.z, sum.sag(x, y)-1,
            rtol=0, atol=1e-12
        )

        # Check default intersect coordTransform
        rv2 = rv.copy().toCoordSys(sumCoordSys)
        batoid.intersect(sum, rv2)
        assert rv2.coordSys == sumCoordSys
        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(
            rv2.z, sum.sag(x, y)-1,
            rtol=0, atol=1e-12
        )


@timer
def test_reflect():
    rng = np.random.default_rng(57721)
    size = 10_000
    for _ in range(100):
        s1 = batoid.Sphere(1./rng.normal(0., 0.2))
        s2 = batoid.Paraboloid(rng.uniform(1, 3))
        sum = batoid.Sum([s1, s2])
        x = rng.uniform(-1, 1, size=size)
        y = rng.uniform(-1, 1, size=size)
        z = np.full_like(x, -10.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.full_like(x, 1)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.reflect(sum, rv.copy())
        rvr2 = sum.reflect(rv.copy())
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = sum.normal(rvr.x, rvr.y)

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
    for _ in range(100):
        s1 = batoid.Sphere(1./rng.normal(0., 0.2))
        s2 = batoid.Paraboloid(rng.uniform(1, 3))
        sum = batoid.Sum([s1, s2])
        m0 = batoid.ConstMedium(rng.normal(1.2, 0.01))
        m1 = batoid.ConstMedium(rng.normal(1.3, 0.01))
        x = rng.uniform(-1, 1, size=size)
        y = rng.uniform(-1, 1, size=size)
        z = np.full_like(x, -10.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.sqrt(1-vx*vx-vy*vy)/m0.n
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.refract(sum, rv.copy(), m0, m1)
        rvr2 = sum.refract(rv.copy(), m0, m1)
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = sum.normal(rvr.x, rvr.y)

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
def test_add_plane():
    rng = np.random.default_rng(5772156)
    for _ in range(100):
        # Adding a plane should have zero effect on sag or normal vector
        s1 = batoid.Sphere(rng.uniform(1, 3))
        s2 = batoid.Plane()
        sum = batoid.Sum([s1, s2])

        x = rng.normal(size=5000)
        y = rng.normal(size=5000)

        np.testing.assert_allclose(
            sum.sag(x, y),
            s1.sag(x, y),
            rtol=0,
            atol=1e-12
        )

        np.testing.assert_allclose(
            sum.normal(x, y),
            s1.normal(x, y),
            rtol=0,
            atol=1e-12,
        )


@timer
def test_sum_paraboloid():
    # para_sag = r^2/(2*R^2)
    # so two paraboloids yields r^2 * (1/(2*R1) + 1/(2*R2))
    # so (1/(2*R1) + 1/(2*R2)) = 1/(2*R)
    # implies
    # 0.5/(1/(2*R1) + 1/(2*R2)) = R
    rng = np.random.default_rng(57721566)
    for _ in range(100):
        R1 = rng.uniform(1, 2)
        R2 = rng.uniform(2, 3)
        Rsum = 0.5/(1/(2*R1) + 1/(2*R2))

        para1 = batoid.Paraboloid(R1)
        para2 = batoid.Paraboloid(R2)
        paraSum = batoid.Paraboloid(Rsum)
        paraSum2 = batoid.Sum([para1, para2])

        x = rng.normal(size=5000)
        y = rng.normal(size=5000)

        np.testing.assert_allclose(
            paraSum.sag(x, y),
            paraSum2.sag(x, y),
            rtol=0,
            atol=1e-12
        )

        np.testing.assert_allclose(
            paraSum.normal(x, y),
            paraSum2.normal(x, y),
            rtol=0,
            atol=1e-12,
        )


@timer
def test_ne():
    objs = [
        batoid.Sum([batoid.Plane(), batoid.Plane()]),
        batoid.Sum([batoid.Plane(), batoid.Sphere(1.0)]),
        batoid.Sum([batoid.Plane(), batoid.Plane(), batoid.Plane()]),
        batoid.Plane()
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    sum = batoid.Sum([batoid.Plane(), batoid.Sphere(1.0)])
    rv = batoid.RayVector(0, 10, 0, 0, 0, -1)  # Too far to side
    rv2 = batoid.intersect(sum, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([True]))
    # This one passes
    rv = batoid.RayVector(0, 0, -1, 0, 0, +1)
    rv2 = batoid.intersect(sum, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([False]))


if __name__ == '__main__':
    init_gpu()
    test_properties()
    test_sag()
    test_normal()
    test_intersect()
    test_reflect()
    test_refract()
    test_add_plane()
    test_sum_paraboloid()
    test_ne()
    test_fail()
