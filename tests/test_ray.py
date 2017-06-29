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
            assert isclose(r.positionAtTime(t).x, x+vx*(t-t0))
            assert isclose(r.positionAtTime(t).y, y+vy*(t-t0))
            assert isclose(r.positionAtTime(t).z, z+vz*(t-t0))
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


@timer
def test_phase():
    import random
    random.seed(577)
    for n in [1.0, 1.3]:  # refractive index
        for i in range(1000):
            x = random.gauss(0.1, 2.3)
            y = random.gauss(2.1, 4.3)
            z = random.gauss(-0.13, 1.3)
            vx = random.gauss(3.1, 6.3)
            vy = random.gauss(5.1, 24.3)
            vz = random.gauss(-1.13, 31.3)
            t0 = random.gauss(0.1, 1.1)
            w = random.uniform(300, 1100)
            p0 = jtrace.Vec3(x, y, z)
            v0 = jtrace.Vec3(vx, vy, vz)
            v0 /= v0.Magnitude()*n
            r = jtrace.Ray(p0, v0, t0, w)

            # Phase is always 0 at current location and time of ray.
            assert r.phase(p0, t0) == 0.0

            # If we move the position forward by an integer multiple of the
            # wavelength, but keep the time the same, the phase should still be
            # 0 (mod 2pi), which we can check for via the amplitude being 1.0
            p1 = r.positionAtTime(t0+5123456789*w*1e-9)
            assert isclose(r.amplitude(p1, t0).real, 1.0,
                           rel_tol=0, abs_tol=1e-9)
            # Let's try a half integer
            p1 = r.positionAtTime(t0+6987654321.5*w*1e-9)
            assert isclose(r.amplitude(p1, t0).real, -1.0,
                           rel_tol=0, abs_tol=1e-9)
            # And a quarter integer
            p1 = r.positionAtTime(t0+7192837465.25*w*1e-9)
            assert isclose(r.amplitude(p1, t0).imag, 1.0,
                           rel_tol=0, abs_tol=1e-9)
            # And a three-quarters integer
            p1 = r.positionAtTime(t0+7182738495.75*w*1e-9)
            assert isclose(r.amplitude(p1, t0).imag, -1.0,
                           rel_tol=0, abs_tol=1e-9)

            # We can also keep the position the same, and change the time in
            # (half/quarter) integer multiples of the period.
            assert isclose(r.amplitude(r.p0, t0 + 5e9*w*1e-9).real, 1.0,
                           rel_tol=0, abs_tol=1e-9)
            assert isclose(r.amplitude(r.p0, t0 + (5e9+5.5)*w*1e-9).real, -1.0,
                           rel_tol=0, abs_tol=1e-9)
            assert isclose(r.amplitude(r.p0, t0 + (5e9+2.25)*w*1e-9).imag, -1.0,
                           rel_tol=0, abs_tol=1e-9)
            assert isclose(r.amplitude(r.p0, t0 + (5e9+1.75)*w*1e-9).imag, 1.0,
                           rel_tol=0, abs_tol=1e-9)

            # If we pick a point anywhere along a vector originating at the Ray
            # position, but orthogonal to its direction of propagation, then we
            # should get phase = 0 (mod 2pi).
            for j in range(10):
                v1 = jtrace.Vec3(
                    random.gauss(0.0, 2.3),
                    random.gauss(0.0, 20.3),
                    random.gauss(0.0, 1.1)
                )
                v1 = jtrace.CrossProduct(v1, r.v)
                p1 = r.p0 + v1
                assert isclose(r.amplitude(p1, t0).real, 1.0,
                               rel_tol=0, abs_tol=1e-9)


if __name__ == '__main__':
    test_call()
    test_properties()
    test_phase()
