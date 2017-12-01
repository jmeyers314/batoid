import batoid
import numpy as np
from test_helpers import isclose, timer, do_pickle


@timer
def test_properties():
    import random
    random.seed(5)
    for i in range(100):
        R = random.gauss(0.7, 0.8)
        conic = random.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        assert quad.R == R
        assert quad.conic == conic
        do_pickle(quad)


def quadric(R, conic):
    def f(x, y):
        r2 = x*x + y*y
        den = R*(1+np.sqrt(1-(1+conic)*r2/R/R))
        return r2/den
    return f


@timer
def test_sag():
    import random
    random.seed(57)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        conic = random.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            result = quad.sag(x, y)
            assert isclose(result, quadric(R, conic)(x, y))
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check vectorization
        x = np.random.normal(0.0, 1.0, size=(10, 10))
        y = np.random.normal(0.0, 1.0, size=(10, 10))
        np.testing.assert_allclose(quad.sag(x, y), quadric(R, conic)(x, y))
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            quad.sag(x[::5,::2], y[::5,::2]),
            quadric(R, conic)(x, y)[::5,::2]
        )


@timer
def test_intersect():
    import random
    random.seed(577)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        conic = random.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r = batoid.Ray(x, y, -10, 0, 0, 1, 0)
            isec = quad.intersect(r)
            assert isclose(isec.point.x, x)
            assert isclose(isec.point.y, y)
            assert isclose(isec.point.z, quad.sag(x, y), rel_tol=0, abs_tol=1e-9)

            # We can also check just for mutual consistency of the asphere,
            # ray and intersection.

            vx = random.gauss(0.0, 0.05)
            vy = random.gauss(0.0, 0.05)
            vz = 1.0
            v = batoid.Vec3(vx, vy, vz).UnitVec3()
            r = batoid.Ray(batoid.Vec3(x, y, -10), v, 0)
            isec = quad.intersect(r)
            p1 = r.positionAtTime(isec.t)
            p2 = isec.point
            assert isclose(p1.x, p2.x)
            assert isclose(p1.y, p2.y)
            assert isclose(p1.z, p2.z)
            assert isclose(quad.sag(p2.x, p2.y), p1.z, rel_tol=0, abs_tol=1e-9)


@timer
def test_intersect_vectorized():
    import random
    random.seed(5772)
    rays = [batoid.Ray([random.gauss(0.0, 0.1),
                        random.gauss(0.0, 0.1),
                        random.gauss(10.0, 0.1)],
                       [random.gauss(0.0, 0.1),
                        random.gauss(0.0, 0.1),
                        random.gauss(-1.0, 0.1)],
                       random.gauss(0.0, 0.1))
            for i in range(1000)]
    rays = batoid.RayVector(rays)

    for i in range(100):
        R = random.gauss(25.0, 0.2)
        conic = random.uniform(-2.0, 1.0)
        quad = batoid.Quadric(R, conic)
        intersections = quad.intersect(rays)
        intersections2 = [quad.intersect(ray) for ray in rays]
        intersections2 = batoid.IntersectionVector(intersections2)
        assert intersections == intersections2


if __name__ == '__main__':
    test_properties()
    test_sag()
    test_intersect()
    test_intersect_vectorized()
