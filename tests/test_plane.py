import batoid
import numpy as np
from test_helpers import isclose, timer, do_pickle


@timer
def test_sag():
    import random
    random.seed(57)
    for i in range(100):
        plane = batoid.Plane()
        for j in range(10):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            result = plane.sag(x, y)
            assert isclose(result, 0.0)
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check vectorization
        x = np.random.normal(0.0, 1.0, size=(10, 10))
        y = np.random.normal(0.0, 1.0, size=(10, 10))
        np.testing.assert_allclose(plane.sag(x, y), 0.0)
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(plane.sag(x[::5,::2], y[::5,::2]), 0.0)
        do_pickle(plane)


@timer
def test_intersect():
    import random
    random.seed(577)
    for i in range(100):
        plane = batoid.Plane()
        for j in range(10):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r0 = batoid.Ray(x, y, -1000, 0, 0, 1, 0)
            r = plane.intersect(r0)
            assert isclose(r.p0[0], x)
            assert isclose(r.p0[1], y)
            assert isclose(r.p0[2], plane.sag(x, y), rel_tol=0, abs_tol=1e-9)


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
        plane = batoid.Plane()
        r1s = plane.intersect(r0s)
        r2s = batoid.RayVector([plane.intersect(r0) for r0 in r0s])
        assert r1s == r2s


@timer
def test_fail():
    plane = batoid.Plane()
    ray = batoid.Ray([0,0,-1], [0,0,-1])
    ray = plane.intersect(ray)
    assert ray.failed

    ray = batoid.Ray([0,0,-1], [0,0,-1])
    plane.intersectInPlace(ray)
    assert ray.failed


if __name__ == '__main__':
    test_sag()
    test_intersect()
    test_intersect_vectorized()
    test_fail()
