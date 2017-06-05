import jtrace
from test_helpers import isclose, timer


@timer
def test_call():
    import random
    random.seed(5)
    for i in range(100):
        x = random.gauss(0.1, 2.3)
        y = random.gauss(2.1, 4.3)
        z = random.gauss(-0.13, 1.3)
        vx = random.gauss(3.1, 6.3)
        vy = random.gauss(5.1, 24.3)
        vz = random.gauss(-1.13, 31.3)
        t0 = random.gauss(0.1, 1.1)
        t = random.gauss(5.5, 1.3)

        # Test both ways of constructing a Ray
        r1 = jtrace.Ray(x, y, z, vx, vy, vz, t0)
        r2 = jtrace.Ray(jtrace.Vec3(x, y, z), jtrace.Vec3(vx, vy, vz), t0)
        for r in [r1, r2]:
            assert isclose(r(t).x, x+vx*(t-t0))
            assert isclose(r(t).y, y+vy*(t-t0))
            assert isclose(r(t).z, z+vz*(t-t0))
        assert r1 == r2


@timer
def test_properties():
    import random
    random.seed(57)
    for i in range(100):
        x = random.gauss(0.1, 2.3)
        y = random.gauss(2.1, 4.3)
        z = random.gauss(-0.13, 1.3)
        vx = random.gauss(3.1, 6.3)
        vy = random.gauss(5.1, 24.3)
        vz = random.gauss(-1.13, 31.3)
        t0 = random.gauss(0.1, 1.1)

        r1 = jtrace.Ray(x, y, z, vx, vy, vz, t0)
        r2 = jtrace.Ray(jtrace.Vec3(x, y, z), jtrace.Vec3(vx, vy, vz), t0)
        for r in [r1, r2]:
            assert r.x0 == x
            assert r.y0 == y
            assert r.z0 == z
            assert r.vx == vx
            assert r.vy == vy
            assert r.vz == vz
            assert r.t0 == t0
        assert r1 == r2


if __name__ == '__main__':
    test_call()
    test_properties()
