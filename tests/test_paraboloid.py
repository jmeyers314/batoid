import batoid
import numpy as np
from test_helpers import isclose, timer, do_pickle, all_obj_diff


@timer
def test_properties():
    import random
    random.seed(5)
    for i in range(100):
        R = random.gauss(0.7, 0.8)
        para = batoid.Paraboloid(R)
        assert para.R == R
        do_pickle(para)


@timer
def test_sag():
    import random
    random.seed(57)
    for i in range(100):
        R = random.gauss(0.2, 0.3)
        para = batoid.Paraboloid(R)
        for j in range(10):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            result = para.sag(x, y)
            assert isclose(result, (x*x + y*y)/2/R)
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check vectorization
        x = np.random.normal(0.0, 1.0, size=(10, 10))
        y = np.random.normal(0.0, 1.0, size=(10, 10))
        np.testing.assert_allclose(para.sag(x, y), (x*x + y*y)/2/R)
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(para.sag(x[::5,::2], y[::5,::2]), ((x*x + y*y)/2/R)[::5,::2])


@timer
def test_intersect():
    import random
    random.seed(577)
    for i in range(100):
        R = random.gauss(10.0, 0.1)
        para = batoid.Paraboloid(R)
        for j in range(10):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r0 = batoid.Ray(x, y, -1000, 0, 0, 1, 0)
            r = para.intersect(r0)
            assert isclose(r.p0[0], x)
            assert isclose(r.p0[1], y)
            assert isclose(r.p0[2], para.sag(x, y), rel_tol=0, abs_tol=1e-9)

        # Check normal at vertex
        np.testing.assert_array_equal(para.normal(0,0), [0,0,1])


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
        para = batoid.Paraboloid(R)
        r1s = para.intersect(r0s)
        r2s = batoid.RayVector([para.intersect(r0) for r0 in r0s])
        assert r1s == r2s


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
    ray = batoid.Ray([0,0,-1], [0,0,-1])
    ray = para.intersect(ray)
    assert ray.failed

    ray = batoid.Ray([0,0,-1], [0,0,-1])
    para.intersectInPlace(ray)
    assert ray.failed


if __name__ == '__main__':
    test_properties()
    test_sag()
    test_intersect()
    test_intersect_vectorized()
    test_ne()
    test_fail()
