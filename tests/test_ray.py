import jtrace


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def test_call():
    import random
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
        # Ray normalizes the "velocity" vector to a unit vector.
        v = jtrace.Vec3(vx, vy, vz).UnitVec3()
        for r in [r1, r2]:
            assert isclose(r(t).x, x+v.x*(t-t0))
            assert isclose(r(t).y, y+v.y*(t-t0))
            assert isclose(r(t).z, z+v.z*(t-t0))


def test_properties():
    import random
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
        v = jtrace.Vec3(vx, vy, vz).UnitVec3()
        for r in [r1, r2]:
            assert r.p0.x == x
            assert r.p0.y == y
            assert r.p0.z == z
            assert r.v.x == v.x
            assert r.v.y == v.y
            assert r.v.z == v.z
            assert r.t0 == t0
