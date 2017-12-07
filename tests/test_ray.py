import batoid
from test_helpers import isclose, timer, do_pickle, all_obj_diff


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
        r1 = batoid.Ray(x, y, z, vx, vy, vz, t0)
        r2 = batoid.Ray(batoid.Vec3(x, y, z), batoid.Vec3(vx, vy, vz), t0)
        for r in [r1, r2]:
            assert isclose(r.positionAtTime(t).x, x+vx*(t-t0))
            assert isclose(r.positionAtTime(t).y, y+vy*(t-t0))
            assert isclose(r.positionAtTime(t).z, z+vz*(t-t0))
        assert r1 == r2
        do_pickle(r1)


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

        r1 = batoid.Ray(x, y, z, vx, vy, vz, t0)
        r2 = batoid.Ray(batoid.Vec3(x, y, z), batoid.Vec3(vx, vy, vz), t0)
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
            w = random.uniform(300e-9, 1100e-9)
            p0 = batoid.Vec3(x, y, z)
            v0 = batoid.Vec3(vx, vy, vz)
            v0 /= v0.Magnitude()*n
            r = batoid.Ray(p0, v0, t0, w)

            # Phase is always 0 at current location and time of ray.
            assert r.phase(p0, t0) == 0.0

            # If we move the position forward by an integer multiple of the
            # wavelength, but keep the time the same, the phase should still be
            # 0 (mod 2pi), which we can check for via the amplitude being 1.0
            p1 = r.positionAtTime(t0+5123456789*w)
            assert isclose(r.amplitude(p1, t0).real, 1.0,
                           rel_tol=0, abs_tol=1e-9)
            # Let's try a half integer
            p1 = r.positionAtTime(t0+6987654321.5*w)
            assert isclose(r.amplitude(p1, t0).real, -1.0,
                           rel_tol=0, abs_tol=1e-9)
            # And a quarter integer
            p1 = r.positionAtTime(t0+7192837465.25*w)
            assert isclose(r.amplitude(p1, t0).imag, 1.0,
                           rel_tol=0, abs_tol=1e-9)
            # And a three-quarters integer
            p1 = r.positionAtTime(t0+7182738495.75*w)
            assert isclose(r.amplitude(p1, t0).imag, -1.0,
                           rel_tol=0, abs_tol=1e-9)

            # We can also keep the position the same, and change the time in
            # (half/quarter) integer multiples of the period.
            assert isclose(r.amplitude(r.p0, t0 + 5e9*w).real, 1.0,
                           rel_tol=0, abs_tol=1e-9)
            assert isclose(r.amplitude(r.p0, t0 + (5e9+5.5)*w).real, -1.0,
                           rel_tol=0, abs_tol=1e-9)
            assert isclose(r.amplitude(r.p0, t0 + (5e9+2.25)*w).imag, -1.0,
                           rel_tol=0, abs_tol=1e-9)
            assert isclose(r.amplitude(r.p0, t0 + (5e9+1.75)*w).imag, 1.0,
                           rel_tol=0, abs_tol=1e-9)

            # If we pick a point anywhere along a vector originating at the Ray
            # position, but orthogonal to its direction of propagation, then we
            # should get phase = 0 (mod 2pi).
            for j in range(10):
                v1 = batoid.Vec3(
                    random.gauss(0.0, 2.3),
                    random.gauss(0.0, 20.3),
                    random.gauss(0.0, 1.1)
                )
                v1 = batoid.CrossProduct(v1, r.v)
                p1 = r.p0 + v1
                assert isclose(r.amplitude(p1, t0).real, 1.0,
                               rel_tol=0, abs_tol=1e-9)


@timer
def test_RayVector():
    import random
    import numpy as np
    random.seed(5772)
    rayList = []
    for i in range(1000):
        rayList.append(
            batoid.Ray(
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                True if random.gauss(0.0, 1.0) < 0.0 else False
            )
        )
    rayVector = batoid.RayVector(rayList)
    np.testing.assert_equal(rayVector.x, np.array([r.x0 for r in rayVector]))
    np.testing.assert_equal(rayVector.y, np.array([r.y0 for r in rayVector]))
    np.testing.assert_equal(rayVector.z, np.array([r.z0 for r in rayVector]))
    np.testing.assert_equal(rayVector.vx, np.array([r.vx for r in rayVector]))
    np.testing.assert_equal(rayVector.vy, np.array([r.vy for r in rayVector]))
    np.testing.assert_equal(rayVector.vz, np.array([r.vz for r in rayVector]))
    np.testing.assert_equal(rayVector.t0, np.array([r.t0 for r in rayVector]))
    np.testing.assert_equal(rayVector.wavelength, np.array([r.wavelength for r in rayVector]))
    np.testing.assert_equal(rayVector.isVignetted, np.array([r.isVignetted for r in rayVector]))
    np.testing.assert_equal(rayVector.failed, np.array([r.failed for r in rayVector]))
    np.testing.assert_equal(batoid._batoid.phaseMany(rayVector, batoid.Vec3(1, 2, 3), 4.0),
                            np.array([r.phase(batoid.Vec3(1, 2, 3), 4.0) for r in rayVector]))
    np.testing.assert_equal(batoid._batoid.amplitudeMany(rayVector, batoid.Vec3(1, 2, 3), 4.0),
                            np.array([r.amplitude(batoid.Vec3(1, 2, 3), 4.0) for r in rayVector]))


@timer
def test_rayGrid():
    dist = 10.0
    length = 10.0
    xcos = 0.1
    ycos = 0.2
    nside = 9
    wavelength = 500e-9
    n = 1.2

    rays = batoid.rayGrid(dist, length, xcos, ycos, nside, wavelength, n)
    # Check that all rays are perpendicular to v
    r0 = rays[0]
    for r in rays:
        dr = r.p0 - r0.p0
        dp = batoid.DotProduct(dr, r0.v)
        assert isclose(dp, 0.0, abs_tol=1e-14, rel_tol=0.0)
        assert isclose(r.wavelength, wavelength)
        assert isclose(r.v.Magnitude(), 1/n)
        assert isclose(r.v.x*n, xcos)
        assert isclose(r.v.y*n, ycos)

    # Check that ray that intersects at origin is initially dist away.
    assert isclose((rays[len(rays)//2].p0).Magnitude(), dist)


@timer
def test_circularGrid():
    dist = 10.0
    outer = 4.1
    inner = 0.5
    xcos = 0.1
    ycos = 0.2
    nradii = 5
    naz = 50
    wavelength = 500e-9
    n = 1.2

    rays = batoid.circularGrid(dist, outer, inner, xcos, ycos, nradii, naz, wavelength, n)
    # Check that all rays are perpendicular to v
    r0 = rays[0]
    for r in rays:
        dr = r.p0 - r0.p0
        dp = batoid.DotProduct(dr, r0.v)
        assert isclose(dp, 0.0, abs_tol=1e-14, rel_tol=0.0)
        assert isclose(r.wavelength, wavelength)
        assert isclose(r.v.Magnitude(), 1/n)
        assert isclose(r.v.x*n, xcos)
        assert isclose(r.v.y*n, ycos)


@timer
def test_ne():
    objs = [batoid.Ray(batoid.Vec3(), batoid.Vec3()),
            batoid.Ray(batoid.Vec3(0,0,1), batoid.Vec3()),
            batoid.Ray(batoid.Vec3(0,1,0), batoid.Vec3()),
            batoid.Ray(batoid.Vec3(), batoid.Vec3(), t=1),
            batoid.Ray(batoid.Vec3(), batoid.Vec3(), w=500e-9),
            batoid.Ray(batoid.Vec3(), batoid.Vec3(), isV=True),
            # Should really get a failed Ray to test here...
            batoid.Vec3(),
            batoid.RayVector(),
            batoid.RayVector([
                batoid.Ray(batoid.Vec3(0,0,1), batoid.Vec3()),
                batoid.Ray(batoid.Vec3(), batoid.Vec3())
            ]),
            batoid.RayVector([
                batoid.Ray(batoid.Vec3(), batoid.Vec3()),
                batoid.Ray(batoid.Vec3(0,0,1), batoid.Vec3())
            ]),
            batoid.RayVector([batoid.Ray(batoid.Vec3(), batoid.Vec3())])
    ]
    all_obj_diff(objs)

if __name__ == '__main__':
    test_call()
    test_properties()
    test_phase()
    test_RayVector()
    test_rayGrid()
    test_circularGrid()
    test_ne()
