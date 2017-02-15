import jtrace
from test_helpers import isclose


def test_properties():
    import random
    for i in range(100):
        R = random.gauss(0.7, 0.8)
        kappa = random.uniform(0.9, 1.0)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-2) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = jtrace.Asphere(R, kappa, alpha, B)
        dx = random.gauss(0, 1)
        dy = random.gauss(0, 1)
        dz = random.gauss(0, 1)
        transformed = jtrace.Transformation(asphere, dx, dy, dz)
        assert transformed.dx == dx
        assert transformed.dy == dy
        assert transformed.dz == dz
        # Try other way to effect a shift.
        transformed2 = asphere.shift(dx, dy, dz)
        assert transformed2.dx == dx
        assert transformed2.dy == dy
        assert transformed2.dz == dz
        # and a third way
        dr = jtrace.Vec3(dx, dy, dz)
        transformed3 = asphere.shift(dr)
        assert transformed3.dx == dx
        assert transformed3.dy == dy
        assert transformed3.dz == dz
        assert transformed3.dr == dr


def test_intersect():
    import random
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        kappa = random.uniform(-1.0, -0.9)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-6) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = jtrace.Asphere(R, kappa, alpha, B)
        dx = random.gauss(0, 1)
        dy = random.gauss(0, 1)
        dz = random.gauss(0, 1)
        transformed = asphere.shift(dx, dy, dz).rotX(0.1)
        print(transformed)

        for j in range(10):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r = jtrace.Ray(x, y, -10, 0, 0, 1, 0)
            isec = transformed.intersect(r)
            assert isclose(isec.point.x, x)
            assert isclose(isec.point.y, y)
            assert isclose(isec.point.z, asphere(x-dx, y-dy)+dz, rel_tol=0, abs_tol=1e-9)

            # We can also check just for mutual consistency of the asphere,
            # ray and intersection.

            vx = random.gauss(0.0, 0.01)
            vy = random.gauss(0.0, 0.01)
            vz = 1.0
            v = jtrace.Vec3(vx, vy, vz).UnitVec3()
            r = jtrace.Ray(jtrace.Vec3(x, y, -10), v, 0)
            isec = transformed.intersect(r)
            p1 = r(isec.t)
            p2 = isec.point
            assert isclose(p1.x, p2.x)
            assert isclose(p1.y, p2.y)
            assert isclose(p1.z, p2.z)
            assert isclose(asphere(p1.x-dx, p2.y-dy)+dz, p1.z, rel_tol=0, abs_tol=1e-9)


if __name__ == '__main__':
    test_properties()
    test_intersect()
