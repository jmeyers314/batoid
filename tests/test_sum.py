import batoid
import numpy as np
from test_helpers import timer, do_pickle, rays_allclose, all_obj_diff


@timer
def test_properties():
    np.random.seed(5)
    for _ in range(100):
        s1 = batoid.Sphere(np.random.uniform(1, 3))
        s2 = batoid.Paraboloid(np.random.uniform(1, 3))
        sum = batoid.Sum([s1, s2])
        do_pickle(sum)

        assert sum.surfaces[0] == s1
        assert sum.surfaces[1] == s2

        s3 = batoid.Zernike([0]*np.random.randint(1, 55)+[np.random.normal()])
        sum2 = batoid.Sum([s1, s2, s3])
        do_pickle(sum2)

        assert sum2.surfaces[0] == s1
        assert sum2.surfaces[1] == s2
        assert sum2.surfaces[2] == s3

        do_pickle(sum)


@timer
def test_sag():
    np.random.seed(57)
    for _ in range(100):
        s1 = batoid.Sphere(np.random.uniform(1, 3))
        s2 = batoid.Paraboloid(np.random.uniform(1, 3))
        sum = batoid.Sum([s1, s2])

        x = np.random.normal(size=5000)
        y = np.random.normal(size=5000)

        np.testing.assert_allclose(
            sum.sag(x, y),
            s1.sag(x, y) + s2.sag(x, y),
            rtol=1e-12,
            atol=1e-12
        )

        s3 = batoid.Zernike([0]*np.random.randint(1, 55)+[np.random.normal()])
        sum2 = batoid.Sum([s1, s2, s3])

        np.testing.assert_allclose(
            sum2.sag(x, y),
            s1.sag(x, y) + s2.sag(x, y) + s3.sag(x, y),
            rtol=1e-12,
            atol=1e-12
        )


@timer
def test_add_plane():
    np.random.seed(577)
    for _ in range(100):
        # Adding a plane should have zero effect on sag or normal vector
        s1 = batoid.Sphere(np.random.uniform(1, 3))
        s2 = batoid.Plane()
        sum = batoid.Sum([s1, s2])

        x = np.random.normal(size=5000)
        y = np.random.normal(size=5000)

        np.testing.assert_allclose(
            sum.sag(x, y),
            s1.sag(x, y),
            rtol=1e-12,
            atol=1e-12
        )

        for _x, _y in zip(x[:100], y[:100]):
            np.testing.assert_allclose(
                sum.normal(_x, _y),
                s1.normal(_x, _y),
                rtol=1e-12,
                atol=1e-12
            )


@timer
def test_sum_paraboloid():
    # para_sag = r^2/(2*R^2)
    # so two paraboloids yields r^2 * (1/(2*R1) + 1/(2*R2))
    # so (1/(2*R1) + 1/(2*R2)) = 1/(2*R)
    # implies
    # 0.5/(1/(2*R1) + 1/(2*R2)) = R
    np.random.seed(5772)
    for _ in range(100):
        R1 = np.random.uniform(1, 2)
        R2 = np.random.uniform(2, 3)
        Rsum = 0.5/(1/(2*R1) + 1/(2*R2))

        para1 = batoid.Paraboloid(R1)
        para2 = batoid.Paraboloid(R2)
        paraSum = batoid.Paraboloid(Rsum)
        paraSum2 = batoid.Sum([para1, para2])

        x = np.random.normal(size=5000)
        y = np.random.normal(size=5000)

        np.testing.assert_allclose(
            paraSum.sag(x, y),
            paraSum2.sag(x, y),
            rtol=1e-12,
            atol=1e-12
        )

        for _x, _y in zip(x[:100], y[:100]):
            np.testing.assert_allclose(
                paraSum.normal(_x, _y),
                paraSum2.normal(_x, _y),
                rtol=1e-12,
                atol=1e-12
            )


@timer
def test_intersect():
    np.random.seed(57721)
    rv0 = batoid.RayVector([
        batoid.Ray(
            np.random.normal(scale=0.1),
            np.random.normal(scale=0.1),
            10,
            np.random.normal(scale=1e-4),
            np.random.normal(scale=1e-4),
            -1
        )
        for _ in range(100)
    ])
    for _ in range(100):
        s1 = batoid.Sphere(np.random.uniform(3, 10))
        s2 = batoid.Paraboloid(np.random.uniform(3, 10))
        sum = batoid.Sum([s1, s2])

        rv = batoid.RayVector(rv0)
        rv1 = sum.intersect(rv)
        rv2 = batoid.RayVector([sum.intersect(r) for r in rv])
        rv3 = batoid.RayVector(rv)
        sum.intersectInPlace(rv)

        for r in rv3:
            sum.intersectInPlace(r)

        assert rays_allclose(rv1, rv)
        assert rays_allclose(rv2, rv)
        assert rays_allclose(rv3, rv)


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
    sum = batoid.Sum([batoid.Zernike([0,0,0,0,1]), batoid.Sphere(1.0)])
    ray = batoid.Ray([0,0,sum.sag(0,0)-1], [0,0,-1])
    ray = sum.intersect(ray)
    assert ray.failed

    ray = batoid.Ray([0,0,sum.sag(0,0)-1], [0,0,-1])
    sum.intersectInPlace(ray)
    assert ray.failed


if __name__ == '__main__':
    test_properties()
    test_sag()
    test_add_plane()
    test_sum_paraboloid()
    test_intersect()
    test_ne()
    test_fail()
