import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff


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
            np.testing.assert_allclose(result, 0.0)
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
            r0 = batoid.Ray((x, y, -1000), (0, 0, 1), 0)
            r = plane.intersect(r0)
            np.testing.assert_allclose(r.r[0], x)
            np.testing.assert_allclose(r.r[1], y)
            np.testing.assert_allclose(r.r[2], plane.sag(x, y), rtol=0, atol=1e-9)


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
def test_ne():
    objs = [
        batoid.Plane(),
        batoid.Plane(allowReverse=True),
        batoid.Paraboloid(2.0),
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    plane = batoid.Plane()
    assert plane.allowReverse == False
    ray = batoid.Ray([0,0,-1], [0,0,-1])
    ray = plane.intersect(ray)
    assert ray.failed

    ray = batoid.Ray([0,0,-1], [0,0,-1])
    plane.intersectInPlace(ray)
    assert ray.failed

    # These should succeed though if allowReverse is True
    plane = batoid.Plane(allowReverse=True)
    assert plane.allowReverse == True
    ray = batoid.Ray([0,0,-1], [0,0,-1])
    ray = plane.intersect(ray)
    assert not ray.failed

    ray = batoid.Ray([0,0,-1], [0,0,-1])
    plane.intersectInPlace(ray)
    assert not ray.failed

if __name__ == '__main__':
    test_sag()
    test_intersect()
    test_intersect_vectorized()
    test_ne()
    test_fail()
