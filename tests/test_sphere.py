import batoid
import numpy as np
import math
from test_helpers import isclose, timer, do_pickle


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
            assert isclose(result, R*(1-math.sqrt(1.0-(x*x + y*y)/R/R)))
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
            r = batoid.Ray(x, y, -1000, 0, 0, 1, 0)
            isec = sphere.intersect(r)
            assert isclose(isec.point.x, x)
            assert isclose(isec.point.y, y)
            assert isclose(isec.point.z, sphere.sag(x, y), rel_tol=0, abs_tol=1e-9)

            # We can also check just for mutual consistency of the sphere,
            # ray and intersection.
            vx = random.gauss(0.0, 0.1)
            vy = random.gauss(0.0, 0.1)
            vz = 1.0
            v = batoid.Vec3(vx, vy, vz).UnitVec3()
            r = batoid.Ray(batoid.Vec3(x, y, -10), v, 0)
            isec = sphere.intersect(r)
            p1 = r.positionAtTime(isec.t)
            p2 = isec.point
            assert isclose(p1.x, p2.x)
            assert isclose(p1.y, p2.y)
            assert isclose(p1.z, p2.z)
            assert isclose(sphere.sag(p1.x, p2.y), p1.z, rel_tol=0, abs_tol=1e-6)


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
        R = random.gauss(0.05, 0.01)
        sphere = batoid.Sphere(R)
        intersections = sphere.intersect(rays)
        intersections2 = [sphere.intersect(ray) for ray in rays]
        intersections2 = batoid.IntersectionVector(intersections2)
        assert intersections == intersections2


if __name__ == '__main__':
    test_properties()
    test_sag()
    test_intersect()
    test_intersect_vectorized()
