import jtrace
from test_helpers import isclose


def test_properties():
    import random
    random.seed(5)
    for i in range(100):
        R = random.gauss(0.7, 0.8)
        kappa = random.uniform(-2.0, 1.0)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = jtrace.Asphere(R, kappa, alpha, B)
        assert asphere.R == R
        assert asphere.kappa == kappa
        assert asphere.alpha == alpha
        assert asphere.B == B


def py_asphere(R, kappa, alpha, B):
    def f(x, y):
        import math
        r2 = x*x + y*y
        den = R*(1+math.sqrt(1-(1+kappa)*r2/R/R))
        result = r2/den + B
        for i, a in enumerate(alpha):
            result += r2**(i+2) * a
        return result
    return f


def test_call():
    import random
    random.seed(57)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        kappa = random.uniform(-2.0, 1.0)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = jtrace.Asphere(R, kappa, alpha, B)
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            assert isclose(asphere(x, y), py_asphere(R, kappa, alpha, B)(x, y))


def test_intersect():
    import random
    random.seed(577)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        kappa = random.uniform(-2.0, 1.0)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 0.5)
        asphere = jtrace.Asphere(R, kappa, alpha, B)
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r = jtrace.Ray(x, y, -10, 0, 0, 1, 0)
            isec = asphere.intersect(r)
            assert isclose(isec.point.x, x)
            assert isclose(isec.point.y, y)
            assert isclose(isec.point.z, asphere(x, y), rel_tol=0, abs_tol=1e-9)

            # We can also check just for mutual consistency of the asphere,
            # ray and intersection.

            vx = random.gauss(0.0, 0.01)
            vy = random.gauss(0.0, 0.01)
            vz = 1.0
            v = jtrace.Vec3(vx, vy, vz).UnitVec3()
            r = jtrace.Ray(jtrace.Vec3(x, y, -10), v, 0)
            isec = asphere.intersect(r)
            p1 = r(isec.t)
            p2 = isec.point
            assert isclose(p1.x, p2.x)
            assert isclose(p1.y, p2.y)
            assert isclose(p1.z, p2.z)
            assert isclose(asphere(p1.x, p2.y), p1.z, rel_tol=0, abs_tol=1e-9)


def test_intersect_vectorized():
    import random
    random.seed(5772)
    rays = [jtrace.Ray([random.gauss(0.0, 0.1),
                        random.gauss(0.0, 0.1),
                        random.gauss(10.0, 0.1)],
                       [random.gauss(0.0, 0.1),
                        random.gauss(0.0, 0.1),
                        random.gauss(-1.0, 0.1)],
                       random.gauss(0.0, 0.1))
            for i in range(1000)]
    rays = jtrace.RayVector(rays)

    for i in range(100):
        R = random.gauss(25.0, 0.2)
        kappa = random.uniform(-2.0, 1.0)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 0.5)
        asphere = jtrace.Asphere(R, kappa, alpha, B)
        intersections = asphere.intersect(rays)
        intersections2 = [asphere.intersect(ray) for ray in rays]
        intersections2 = jtrace.IntersectionVector(intersections2)
        assert intersections == intersections2


def py_poly(alpha):
    def f(x, y):
        r2 = x*x + y*y
        result = 0
        for i, a in enumerate(alpha):
            result += r2**(i+2) * a
        return result
    return f


def test_quad_plus_poly():
    import random
    random.seed(5772)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        kappa = random.uniform(-2.0, 1.0)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = jtrace.Asphere(R, kappa, alpha, B)
        quad = jtrace.Quadric(R, kappa, B)
        poly = py_poly(alpha)
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            assert isclose(asphere(x, y), quad(x, y)+poly(x, y))


if __name__ == '__main__':
    test_properties()
    test_call()
    test_intersect()
    test_intersect_vectorized()
    test_quad_plus_poly()
