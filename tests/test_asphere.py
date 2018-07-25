import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_properties():
    import random
    random.seed(5)
    for i in range(100):
        R = random.gauss(0.7, 0.8)
        conic = random.uniform(-2.0, 1.0)
        ncoef = random.randint(0, 4)
        coefs = [random.gauss(0, 1e-10) for i in range(ncoef)]
        asphere = batoid.Asphere(R, conic, coefs)
        assert asphere.R == R
        assert asphere.conic == conic
        assert asphere.coefs == coefs
        do_pickle(asphere)


def py_asphere(R, conic, coefs):
    def f(x, y):
        r2 = x*x + y*y
        den = R*(1+np.sqrt(1-(1+conic)*r2/R/R))
        result = r2/den
        for i, a in enumerate(coefs):
            result += r2**(i+2) * a
        return result
    return f


@timer
def test_sag():
    import random
    random.seed(57)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        conic = random.uniform(-2.0, 1.0)
        ncoefs = random.randint(0, 4)
        coefs = [random.gauss(0, 1e-10) for i in range(ncoefs)]
        asphere = batoid.Asphere(R, conic, coefs)
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            result = asphere.sag(x, y)
            np.testing.assert_allclose(result, py_asphere(R, conic, coefs)(x, y))
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check vectorization
        x = np.random.normal(0.0, 1.0, size=(10,10))
        y = np.random.normal(0.0, 1.0, size=(10,10))
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            asphere.sag(x[::5,::2], y[::5,::2]),
            py_asphere(R, conic, coefs)(x, y)[::5,::2]
        )


@timer
def test_intersect():
    import random
    random.seed(577)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        conic = random.uniform(-2.0, 1.0)
        ncoefs = random.randint(0, 4)
        coefs = [random.gauss(0, 1e-10) for i in range(ncoefs)]
        asphere = batoid.Asphere(R, conic, coefs)
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r0 = batoid.Ray(x, y, -10, 0, 0, 1, 0)
            r = asphere.intersect(r0)
            np.testing.assert_allclose(r.r[0], x)
            np.testing.assert_allclose(r.r[1], y)
            np.testing.assert_allclose(r.r[2], asphere.sag(x, y), rtol=0, atol=1e-9)

    # Check normal for R=0 paraboloid (a plane)
    asphere = batoid.Asphere(0.0, 0.0, [])
    np.testing.assert_array_equal(asphere.normal(0.1, 0.1), [0,0,1])
    # Check that normal parallelizes
    xs = np.random.normal(size=20)
    ys = np.random.normal(size=20)
    np.testing.assert_array_equal(
        asphere.normal(xs, ys),
        np.array([asphere.normal(x, y) for x, y in zip(xs, ys)])
    )
    # Test shape vectorization
    np.testing.assert_array_equal(
        asphere.normal(xs.reshape((10, 2)), ys.reshape((10, 2))),
        np.array([asphere.normal(x, y) for x, y in zip(xs, ys)]).reshape(10, 2, 3)
    )
    np.testing.assert_array_equal(
        asphere.normal(xs.reshape((2, 5, 2)), ys.reshape((2, 5, 2))),
        np.array([asphere.normal(x, y) for x, y in zip(xs, ys)]).reshape(2, 5, 2, 3)
    )
    # Also test non-unit strides on last index
    np.testing.assert_array_equal(
        asphere.normal(xs.reshape((10, 2))[::2], ys.reshape((10, 2))[::2]),
        np.array([asphere.normal(x, y) for x, y in zip(xs, ys)]).reshape(10, 2, 3)[::2]
    )


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
        R = random.gauss(25.0, 0.2)
        conic = random.uniform(-2.0, 1.0)
        ncoefs = random.randint(0, 4)
        coefs = [random.gauss(0, 1e-10) for i in range(ncoefs)]
        asphere = batoid.Asphere(R, conic, coefs)

        r1s = asphere.intersect(r0s)
        r2s = batoid.RayVector([asphere.intersect(r0) for r0 in r0s])
        assert r1s == r2s


def py_poly(coefs):
    def f(x, y):
        r2 = x*x + y*y
        result = 0
        for i, a in enumerate(coefs):
            result += r2**(i+2) * a
        return result
    return f


@timer
def test_quad_plus_poly():
    import random
    random.seed(5772)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        conic = random.uniform(-2.0, 1.0)
        ncoefs = random.randint(0, 4)
        coefs = [random.gauss(0, 1e-10) for i in range(ncoefs)]
        asphere = batoid.Asphere(R, conic, coefs)
        quad = batoid.Quadric(R, conic)
        poly = py_poly(coefs)
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            np.testing.assert_allclose(asphere.sag(x, y), quad.sag(x, y)+poly(x, y))


@timer
def test_ne():
    objs = [
        batoid.Asphere(10.0, 1.0, []),
        batoid.Asphere(10.0, 1.0, [0]),
        batoid.Asphere(10.0, 1.0, [0,1]),
        batoid.Asphere(10.0, 1.0, [1,0]),
        batoid.Asphere(10.0, 1.1, []),
        batoid.Asphere(10.1, 1.0, []),
        batoid.Quadric(10.0, 1.0)
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    asphere = batoid.Asphere(1.0, 1.0, [1.0, 1.0])
    ray = batoid.Ray([0,0,-1], [0,0,-1])
    ray = asphere.intersect(ray)
    assert ray.failed

    ray = batoid.Ray([0,0,-1], [0,0,-1])
    asphere.intersectInPlace(ray)
    assert ray.failed


if __name__ == '__main__':
    test_properties()
    test_sag()
    test_intersect()
    test_intersect_vectorized()
    test_quad_plus_poly()
    test_ne()
    test_fail()
