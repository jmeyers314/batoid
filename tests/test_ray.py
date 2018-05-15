import batoid
import numpy as np
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
        r2 = batoid.Ray([x, y, z], [vx, vy, vz], t0)
        r3 = batoid.Ray((x, y, z), (vx, vy, vz), t0)
        r4 = batoid.Ray(np.array([x, y, z]), np.array([vx, vy, vz]), t0)
        for r in [r1, r2, r3, r4]:
            np.testing.assert_allclose(r.positionAtTime(t)[0], x+vx*(t-t0))
            np.testing.assert_allclose(r.positionAtTime(t)[1], y+vy*(t-t0))
            np.testing.assert_allclose(r.positionAtTime(t)[2], z+vz*(t-t0))
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
        r2 = batoid.Ray([x, y, z], [vx, vy, vz], t0)
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
            p0 = np.array([x, y, z])
            v0 = np.array([vx, vy, vz])
            v0 /= np.linalg.norm(v0)*n
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
                v1 = np.array([
                    random.gauss(0.0, 2.3),
                    random.gauss(0.0, 20.3),
                    random.gauss(0.0, 1.1)
                ])
                v1 = np.cross(v1, r.v)
                p1 = r.p0 + v1
                assert isclose(r.amplitude(p1, t0).real, 1.0,
                               rel_tol=0, abs_tol=1e-9)


@timer
def test_RayVector():
    import random
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
    np.testing.assert_equal(rayVector.phase([1, 2, 3], 4.0),
                            np.array([r.phase([1, 2, 3], 4.0) for r in rayVector]))
    np.testing.assert_equal(rayVector.amplitude([1, 2, 3], 4.0),
                            np.array([r.amplitude([1, 2, 3], 4.0) for r in rayVector]))

    np.testing.assert_equal(rayVector.v, np.array([[r.vx, r.vy, r.vz] for r in rayVector]))
    np.testing.assert_equal(rayVector.p0, np.array([[r.x0, r.y0, r.z0] for r in rayVector]))
    np.testing.assert_equal(rayVector.k, np.array([r.k for r in rayVector]))
    np.testing.assert_equal(rayVector.omega, np.array([r.omega for r in rayVector]))

    np.testing.assert_equal(rayVector.kx, np.array([r.kx for r in rayVector]))
    np.testing.assert_equal(rayVector.ky, np.array([r.ky for r in rayVector]))
    np.testing.assert_equal(rayVector.kz, np.array([r.kz for r in rayVector]))

    # Make sure we really got a view and not a copy
    x = rayVector.x
    x[0] += 1
    assert np.all(x == rayVector.x)
    assert not rayVector.x.flags.owndata

    # What about lifetimes?  What happens to x if rayVector disappears?
    x2 = np.copy(x)
    assert x is not x2
    del rayVector
    assert np.all(x == x2)


@timer
def test_rayGrid():
    dist = 10.0
    length = 10.0
    xcos = 0.1
    ycos = 0.2
    zcos = -np.sqrt(1.0 - xcos**2 - ycos**2)
    nside = 10
    wavelength = 500e-9
    medium = batoid.ConstMedium(1.2)

    rays = batoid.rayGrid(dist, length, xcos, ycos, zcos, nside, wavelength, medium)
    # Check that all rays are perpendicular to v
    r0 = rays[0]
    for r in rays:
        dr = r.p0 - r0.p0
        dp = np.dot(dr, r0.v)
        np.testing.assert_allclose(dp, 0.0, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(r.wavelength, wavelength)
        np.testing.assert_allclose(np.linalg.norm(r.v), 1./1.2)
        np.testing.assert_allclose(r.v[0]*1.2, xcos)
        np.testing.assert_allclose(r.v[1]*1.2, ycos)

    # Check that ray that intersects at origin is initially dist away.
    # Need the ray that is in the middle in both dimensions...
    idx = np.ravel_multi_index((nside//2, nside//2), (nside, nside))
    np.testing.assert_allclose(np.linalg.norm(rays[idx].p0), dist)


@timer
def test_circularGrid():
    dist = 10.0
    outer = 4.1
    inner = 0.5
    xcos = 0.1
    ycos = 0.2
    zcos = -np.sqrt(1.0 - xcos**2 - ycos**2)
    nradii = 5
    naz = 50
    wavelength = 500e-9
    medium = batoid.ConstMedium(1.2)

    rays = batoid.circularGrid(dist, outer, inner, xcos, ycos, zcos, nradii, naz, wavelength, medium)
    # Check that all rays are perpendicular to v
    r0 = rays[0]
    for r in rays:
        dr = r.p0 - r0.p0
        dp = np.dot(dr, r0.v)
        np.testing.assert_allclose(dp, 0.0, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(r.wavelength, wavelength)
        np.testing.assert_allclose(np.linalg.norm(r.v), 1./1.2)
        np.testing.assert_allclose(r.v[0]*1.2, xcos)
        np.testing.assert_allclose(r.v[1]*1.2, ycos)


@timer
def test_ne():
    objs = [batoid.Ray((0,0,0), (0,0,0)),
            batoid.Ray((0,0,1), (0,0,0)),
            batoid.Ray((0,1,0), (0,0,0)),
            batoid.Ray((0,0,0), (0,0,0), t=1),
            batoid.Ray((0,0,0), (0,0,0), w=500e-9),
            batoid.Ray((0,0,0), (0,0,0), isV=True),
            # Should really get a failed Ray to test here...
            (0,0,0),
            batoid.RayVector(),
            batoid.RayVector([
                batoid.Ray((0,0,1), (0,0,0)),
                batoid.Ray((0,0,0), (0,0,0))
            ]),
            batoid.RayVector([
                batoid.Ray((0,0,0), (0,0,0)),
                batoid.Ray((0,0,1), (0,0,0))
            ]),
            batoid.RayVector([batoid.Ray((0,0,0), (0,0,0))])
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    surface = batoid.Sphere(1.0)
    ray = batoid.Ray([0,10,-1], [0,0,1])

    ray = surface.intersect(ray)
    assert ray.failed
    do_pickle(ray)


if __name__ == '__main__':
    test_call()
    test_properties()
    test_phase()
    test_RayVector()
    test_rayGrid()
    test_circularGrid()
    test_ne()
    test_fail()
