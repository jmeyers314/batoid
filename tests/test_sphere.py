import batoid
import numpy as np
import math
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_properties():
    import random
    random.seed(5)
    for i in range(100):
        R = random.gauss(0.7, 0.8)
        sphere = batoid.Sphere(R)
        assert sphere.R == R
        do_pickle(sphere)


@timer
def test_sag():
    import random
    random.seed(57)
    for i in range(100):
        R = random.gauss(4.2, 0.3)
        sphere = batoid.Sphere(R)
        for j in range(10):
            x = random.uniform(-0.7*R, 0.7*R)
            y = random.uniform(-0.7*R, 0.7*R)
            result = sphere.sag(x, y)
            np.testing.assert_allclose(result, R*(1-math.sqrt(1.0-(x*x + y*y)/R/R)))
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check vectorization
        x = np.random.uniform(-0.7*R, 0.7*R, size=(10, 10))
        y = np.random.uniform(-0.7*R, 0.7*R, size=(10, 10))
        np.testing.assert_allclose(sphere.sag(x, y), R*(1-np.sqrt(1.0-(x*x + y*y)/R/R)))
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            sphere.sag(x[::5,::2], y[::5,::2]),
            (R*(1-np.sqrt(1.0-(x*x + y*y)/R/R)))[::5, ::2]
        )


@timer
def test_intersect():
    import random
    random.seed(577)
    for i in range(100):
        R = random.gauss(10.0, 0.1)
        sphere = batoid.Sphere(R)
        for j in range(10):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r0 = batoid.Ray((x, y, -1000), (0, 0, 1), 0)
            r = sphere.intersect(r0)
            np.testing.assert_allclose(r.r[0], x)
            np.testing.assert_allclose(r.r[1], y)
            np.testing.assert_allclose(r.r[2], sphere.sag(x, y), rtol=0, atol=1e-9)

    # Check normal for R=0 paraboloid (a plane)
    sphere = batoid.Sphere(0.0)
    np.testing.assert_array_equal(sphere.normal(0.1,0.1), [0,0,1])


@timer
def test_intersect_vectorized():
    import random
    random.seed(5772)
    r0s = [batoid.Ray([random.gauss(0.0, 0.1),
                       random.gauss(0.0, 0.1),
                       random.gauss(10.0, 0.1)],
                      [random.gauss(0.0, 0.1),
                       random.gauss(0.0, 0.1),
                       random.gauss(-1.0, 0.1)],
                      random.gauss(0.0, 0.1))
           for i in range(1000)]
    r0s = batoid.RayVector(r0s)

    for i in range(100):
        R = random.gauss(0.05, 0.01)
        sphere = batoid.Sphere(R)
        r1s = sphere.intersect(r0s)
        r2s = batoid.RayVector([sphere.intersect(r0) for r0 in r0s])
        assert r1s == r2s


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
    ray = batoid.Ray([0,0,-1], [0,0,-1])
    ray = sphere.intersect(ray)
    assert ray.failed

    ray = batoid.Ray([0,0,-1], [0,0,-1])
    sphere.intersectInPlace(ray)
    assert ray.failed


if __name__ == '__main__':
    test_properties()
    test_sag()
    test_intersect()
    test_intersect_vectorized()
    test_ne()
    test_fail()
