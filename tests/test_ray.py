import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff, rays_allclose, checkAngle


@timer
def test_positionAtTime():
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

        # Test different ways of constructing a Ray
        ray1 = batoid.Ray([x, y, z], [vx, vy, vz], t0)
        ray2 = batoid.Ray((x, y, z), (vx, vy, vz), t0)
        ray3 = batoid.Ray(np.array([x, y, z]), np.array([vx, vy, vz]), t0)
        for ray in [ray1, ray2, ray3]:
            np.testing.assert_allclose(ray.positionAtTime(t)[0], x+vx*(t-t0))
            np.testing.assert_allclose(ray.positionAtTime(t)[1], y+vy*(t-t0))
            np.testing.assert_allclose(ray.positionAtTime(t)[2], z+vz*(t-t0))
        assert ray1 == ray2 == ray3
        do_pickle(ray1)


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
        t = random.gauss(0.1, 1.1)
        w = random.gauss(1000.0, 10.0)
        f = random.gauss(1000.0, 10.0)
        v = random.choice([True, False])

        ray = batoid.Ray([x, y, z], [vx, vy, vz], t, w, f, v)
        assert ray.x == x
        assert ray.y == y
        assert ray.z == z
        assert ray.vx == vx
        assert ray.vy == vy
        assert ray.vz == vz
        assert ray.t == t
        assert ray.wavelength == w
        assert ray.vignetted == v


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
            t = random.gauss(0.1, 1.1)
            w = random.uniform(300e-9, 1100e-9)
            r = np.array([x, y, z])
            v0 = np.array([vx, vy, vz])
            v0 /= np.linalg.norm(v0)*n
            ray = batoid.Ray(r, v0, t, w)

            # Phase is always 0 at current location and time of ray.
            assert ray.phase(r, t) == 0.0

            # If we move the position forward by an integer multiple of the
            # wavelength, but keep the time the same, the phase should still be
            # 0 (mod 2pi), which we can check for via the amplitude being 1.0
            r1 = ray.positionAtTime(t+5123456789*w)
            np.testing.assert_allclose(ray.amplitude(r1, t).real, 1.0,
                                       rtol=0, atol=1e-9)
            # Let's try a half integer
            r1 = ray.positionAtTime(t+6987654321.5*w)
            np.testing.assert_allclose(ray.amplitude(r1, t).real, -1.0,
                                       rtol=0, atol=1e-9)
            # And a quarter integer
            r1 = ray.positionAtTime(t+7192837465.25*w)
            np.testing.assert_allclose(ray.amplitude(r1, t).imag, 1.0,
                                       rtol=0, atol=1e-9)
            # And a three-quarters integer
            r1 = ray.positionAtTime(t+7182738495.75*w)
            np.testing.assert_allclose(ray.amplitude(r1, t).imag, -1.0,
                                       rtol=0, atol=1e-9)

            # We can also keep the position the same, and change the time in
            # (half/quarter) integer multiples of the period.
            np.testing.assert_allclose(ray.amplitude(ray.r, t + 5e9*w).real, 1.0,
                                       rtol=0, atol=1e-9)
            np.testing.assert_allclose(ray.amplitude(ray.r, t + (5e9+5.5)*w).real, -1.0,
                                       rtol=0, atol=1e-9)
            np.testing.assert_allclose(ray.amplitude(ray.r, t + (5e9+2.25)*w).imag, -1.0,
                                       rtol=0, atol=1e-9)
            np.testing.assert_allclose(ray.amplitude(ray.r, t + (5e9+1.75)*w).imag, 1.0,
                                       rtol=0, atol=1e-9)

            # If we pick a point anywhere along a vector originating at the Ray
            # position, but orthogonal to its direction of propagation, then we
            # should get phase = 0 (mod 2pi).
            for j in range(10):
                v1 = np.array([
                    random.gauss(0.0, 2.3),
                    random.gauss(0.0, 20.3),
                    random.gauss(0.0, 1.1)
                ])
                v1 = np.cross(v1, ray.v)
                p1 = ray.r + v1
                np.testing.assert_allclose(ray.amplitude(p1, t).real, 1.0,
                                           rtol=0, atol=1e-9)


@timer
def test_RayVector():
    import random
    random.seed(5772)
    rayList = []
    for i in range(1000):
        rayList.append(
            batoid.Ray(
                [random.gauss(0.0, 1.0),  # x
                 random.gauss(0.0, 1.0),  # y
                 random.gauss(0.0, 1.0)],  # z
                [random.gauss(0.0, 1.0),  # vx
                 random.gauss(0.0, 1.0),  # vy
                 random.gauss(0.0, 1.0)],  # vz
                random.gauss(0.0, 1.0),  # t0
                random.gauss(1000.0, 1.0),  # wavelength
                random.gauss(100.0, 1.0),  # flux
                random.choice([True, False])  # vignetted
            )
        )
    rayVector = batoid.RayVector(rayList)
    assert rayVector.monochromatic == False
    do_pickle(rayVector)
    np.testing.assert_equal(rayVector.x, np.array([ray.x for ray in rayVector]))
    np.testing.assert_equal(rayVector.y, np.array([ray.y for ray in rayVector]))
    np.testing.assert_equal(rayVector.z, np.array([ray.z for ray in rayVector]))
    np.testing.assert_equal(rayVector.vx, np.array([ray.vx for ray in rayVector]))
    np.testing.assert_equal(rayVector.vy, np.array([ray.vy for ray in rayVector]))
    np.testing.assert_equal(rayVector.vz, np.array([ray.vz for ray in rayVector]))
    np.testing.assert_equal(rayVector.t, np.array([ray.t for ray in rayVector]))
    np.testing.assert_equal(rayVector.wavelength, np.array([ray.wavelength for ray in rayVector]))
    np.testing.assert_equal(rayVector.flux, np.array([ray.flux for ray in rayVector]))
    np.testing.assert_equal(rayVector.vignetted, np.array([ray.vignetted for ray in rayVector]))
    np.testing.assert_equal(rayVector.failed, np.array([ray.failed for ray in rayVector]))
    np.testing.assert_equal(rayVector.phase([1, 2, 3], 4.0),
                            np.array([ray.phase([1, 2, 3], 4.0) for ray in rayVector]))
    np.testing.assert_equal(rayVector.amplitude([1, 2, 3], 4.0),
                            np.array([ray.amplitude([1, 2, 3], 4.0) for ray in rayVector]))

    np.testing.assert_equal(rayVector.v, np.array([[ray.vx, ray.vy, ray.vz] for ray in rayVector]))
    np.testing.assert_equal(rayVector.r, np.array([[ray.x, ray.y, ray.z] for ray in rayVector]))
    np.testing.assert_equal(rayVector.k, np.array([ray.k for ray in rayVector]))
    np.testing.assert_equal(rayVector.omega, np.array([ray.omega for ray in rayVector]))

    np.testing.assert_equal(rayVector.kx, np.array([ray.kx for ray in rayVector]))
    np.testing.assert_equal(rayVector.ky, np.array([ray.ky for ray in rayVector]))
    np.testing.assert_equal(rayVector.kz, np.array([ray.kz for ray in rayVector]))

    # Try the other ctor
    rayVector2 = batoid.RayVector.fromArrays(
        np.array([ray.x for ray in rayList]),
        np.array([ray.y for ray in rayList]),
        np.array([ray.z for ray in rayList]),
        np.array([ray.vx for ray in rayList]),
        np.array([ray.vy for ray in rayList]),
        np.array([ray.vz for ray in rayList]),
        np.array([ray.t for ray in rayList]),
        np.array([ray.wavelength for ray in rayList]),
        np.array([ray.flux for ray in rayList]),
        np.array([ray.vignetted for ray in rayList])
    )
    assert rayVector == rayVector2
    assert rayVector2.monochromatic == False

    # See if we can make monochromatic True
    rayVector3 = batoid.RayVector.fromArrays(
        np.array([ray.x for ray in rayList]),
        np.array([ray.y for ray in rayList]),
        np.array([ray.z for ray in rayList]),
        np.array([ray.vx for ray in rayList]),
        np.array([ray.vy for ray in rayList]),
        np.array([ray.vz for ray in rayList]),
        np.array([ray.t for ray in rayList]),
        np.array([1.0 for ray in rayList]),  # wavelength
        np.array([ray.flux for ray in rayList]),
        np.array([ray.vignetted for ray in rayList])
    )
    assert rayVector3.monochromatic == True

    # Make sure we really got a view and not a copy
    x = rayVector.x
    x[0] += 1
    assert np.all(x == rayVector.x)
    assert not rayVector.x.flags.owndata

    # What about lifetimes?  What happens to x if rayVector disappears?
    x2 = np.copy(x)
    assert x is not x2
    del rayVector
    assert np.all(x == x2)  # it survives!

    # Test concatenateRayVectors
    rv1 = batoid.RayVector(rayList[0:5])
    rv2 = batoid.RayVector(rayList[5:10])
    rv3 = batoid.RayVector(rayList[10:12])
    rv4 = batoid.RayVector(rayList[12:40])
    rvA = batoid.concatenateRayVectors([rv1, rv2, rv3, rv4])
    rvB = batoid.RayVector(rayList[0:40])
    assert rvA == rvB


@timer
def test_rayGrid():
    dist = 10.0
    length = 10.0
    xcos = 0.1
    ycos = 0.2
    zcos = -np.sqrt(1.0 - xcos**2 - ycos**2)
    nside = 10
    wavelength = 500e-9
    flux = 1.2
    medium = batoid.ConstMedium(1.2)

    rays = batoid.rayGrid(
        dist, length, xcos, ycos, zcos, nside, wavelength, flux, medium,
        lattice=True
    )
    assert rays.monochromatic == True
    # Check that all rays are perpendicular to v
    ray0 = rays[0]
    for ray in rays:
        dr = ray.r - ray0.r
        dp = np.dot(dr, ray0.v)
        np.testing.assert_allclose(dp, 0.0, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(ray.wavelength, wavelength)
        np.testing.assert_allclose(ray.flux, flux)
        np.testing.assert_allclose(np.linalg.norm(ray.v), 1./1.2)
        np.testing.assert_allclose(ray.v[0]*1.2, xcos)
        np.testing.assert_allclose(ray.v[1]*1.2, ycos)

    # Check that ray that intersects at origin is initially dist away.
    # Need the ray that is in the middle in both dimensions...
    idx = np.ravel_multi_index((nside//2, nside//2), (nside, nside))
    rays.propagateInPlace(dist*1.2)
    np.testing.assert_equal(rays[idx].r, [0,0,0])
    # but mean position won't be the origin, since lattice implies off-center
    assert np.linalg.norm(np.mean(rays.r, axis=0)) > 0.5

    # Now try again with lattice flag set to False
    rays = batoid.rayGrid(
        dist, length, xcos, ycos, zcos, nside, wavelength, flux, medium,
        lattice=False
    )
    # "Central" ray will not intersect origin in this case, but average of
    # all rays should be the origin
    idx = np.ravel_multi_index((nside//2, nside//2), (nside, nside))
    rays.propagateInPlace(dist*1.2)
    assert np.linalg.norm(rays[idx].r) > 0.1
    np.testing.assert_allclose(np.linalg.norm(np.mean(rays.r, axis=0)), 0.0, rtol=0, atol=1e-14)

    # If we use an odd nside, then both the central point and the mean will be the origin.
    nside = 11
    rays = batoid.rayGrid(
        dist, length, xcos, ycos, zcos, nside, wavelength, flux, medium,
        lattice=False
    )
    idx = np.ravel_multi_index((nside//2, nside//2), (nside, nside))
    rays.propagateInPlace(dist*1.2)
    np.testing.assert_allclose(rays[idx].r, [0,0,0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(np.mean(rays.r, axis=0)), 0.0, rtol=0, atol=1e-14)


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
    flux = 1.2
    medium = batoid.ConstMedium(1.2)

    rays = batoid.circularGrid(dist, outer, inner, xcos, ycos, zcos, nradii, naz, wavelength, flux, medium)
    assert rays.monochromatic == True
    # Check that all rays are perpendicular to v
    ray0 = rays[0]
    for ray in rays:
        dr = ray.r - ray0.r
        dp = np.dot(dr, ray0.v)
        np.testing.assert_allclose(dp, 0.0, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(ray.wavelength, wavelength)
        np.testing.assert_allclose(ray.flux, flux)
        np.testing.assert_allclose(np.linalg.norm(ray.v), 1./1.2)
        np.testing.assert_allclose(ray.v[0]*1.2, xcos)
        np.testing.assert_allclose(ray.v[1]*1.2, ycos)


@timer
def test_uniformCircularGrid():
    dist = 10.0
    outer = 4.1
    inner = 0.5
    xcos = 0
    ycos = 0
    zcos = -1
    nray = 100000
    wavelength = 500e-9
    flux = 1.
    medium = batoid.ConstMedium(1.)
    seed = 0

    rays = batoid.uniformCircularGrid(dist, outer, inner, xcos, ycos, zcos, nray, wavelength, flux,
                                      medium, seed=seed)
    radius = np.hypot(rays.x, rays.y)
    angle = np.arctan2(rays.y, rays.x)

    np.testing.assert_almost_equal(radius.max(), outer, decimal=4)
    np.testing.assert_almost_equal(radius.min(), inner, decimal=4)
    np.testing.assert_almost_equal(rays.x.mean(), 0, decimal=2)
    np.testing.assert_almost_equal(rays.y.mean(), 0, decimal=2)
    np.testing.assert_almost_equal(angle.mean(), 0, decimal=2)

    # test radial distribution
    for cutoff in np.linspace(inner, outer, 5):
        frac = np.sum(radius < cutoff) / nray
        expected = (cutoff ** 2 - inner ** 2) / (outer ** 2 - inner ** 2)
        np.testing.assert_almost_equal(frac, expected, decimal=1)


    # test seed, reproducibility
    rays2 = batoid.uniformCircularGrid(dist, outer, inner, xcos, ycos, zcos, nray, wavelength, flux,
                                       medium, seed=seed)
    newseed = 666
    rays3 = batoid.uniformCircularGrid(dist, outer, inner, xcos, ycos, zcos, nray, wavelength, flux,
                                       medium, seed=newseed)

    assert np.all(rays.r == rays2.r)
    assert not np.all(rays2.r == rays3.r)


@timer
def test_pointSourceCircularGrid():
    source = [0, 1, 10]
    outer = 0.1
    inner = 0.0
    nradii = 2
    naz = 6
    wavelength = 500e-9
    flux = 1.0
    medium = batoid.ConstMedium(1)
    rays = batoid.pointSourceCircularGrid(
        source, outer, inner, nradii, naz, wavelength, flux, medium)
    # Verify that the central ray is pointed at the origin
    # (last ray is central if inner==0.0)
    centerRay = rays[len(rays)-1]
    centerRay.propagateInPlace(np.sqrt(101))
    np.testing.assert_allclose(centerRay.r, [0,0,0], rtol=0, atol=1e-10)


@timer
def test_ne():
    objs = [batoid.Ray((0,0,0), (0,0,0)),
            batoid.Ray((0,0,1), (0,0,0)),
            batoid.Ray((0,1,0), (0,0,0)),
            batoid.Ray((0,0,0), (0,0,0), t=1),
            batoid.Ray((0,0,0), (0,0,0), wavelength=500e-9),
            batoid.Ray((0,0,0), (0,0,0), wavelength=500e-9, flux=1.2),
            batoid.Ray((0,0,0), (0,0,0), vignetted=True),
            # Should really get a failed Ray to test here...
            (0,0,0),
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


@timer
def test_RVasGrid():
    for _ in range(10):
        dist = np.random.uniform(9.0, 11.0)
        wavelength = np.random.uniform(300e-9, 1100e-9)
        nx = 1
        while (nx%2) == 1:
            nx = np.random.randint(10, 20)
        lx = np.random.uniform(1.0, 10.0)
        dx = lx/(nx-2)
        dirCos = np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-1.2, -0.8),
        ])
        dirCos /= np.sqrt(np.dot(dirCos, dirCos))

        # Some things that should be equivalent
        grid1 = batoid.RayVector.asGrid(
            dist, wavelength,
            nx=nx, lx=lx, dirCos=dirCos
        )
        grid2 = batoid.RayVector.asGrid(
            dist, wavelength,
            nx=nx, dx=dx, dirCos=dirCos
        )
        grid3 = batoid.RayVector.asGrid(
            dist, wavelength,
            dx=dx, lx=lx, dirCos=dirCos
        )
        grid4 = batoid.RayVector.asGrid(
            dist, wavelength,
            nx=nx, lx=(lx, 0.0), dirCos=dirCos
        )
        assert rays_allclose(grid1, grid2)
        assert rays_allclose(grid1, grid3)
        assert rays_allclose(grid1, grid4)

        # Check distance to chief ray
        cridx = (nx//2)*nx+nx//2
        obs_dist = np.sqrt(np.dot(grid1[cridx].r, grid1[cridx].r))
        np.testing.assert_allclose(obs_dist, dist)

        np.testing.assert_allclose(grid1.t, 0)
        np.testing.assert_allclose(grid1.wavelength, wavelength)
        np.testing.assert_allclose(grid1.vignetted, False)
        np.testing.assert_allclose(grid1.failed, False)
        np.testing.assert_allclose(grid1.vx, dirCos[0])
        np.testing.assert_allclose(grid1.vy, dirCos[1])
        np.testing.assert_allclose(grid1.vz, dirCos[2])

        # Check distribution of points propagated to entrance pupil
        pupil = batoid.Plane()
        pupil.intersectInPlace(grid1)
        np.testing.assert_allclose(np.diff(grid1.x)[0], dx)
        np.testing.assert_allclose(np.diff(grid1.y)[0], 0, atol=1e-14)
        np.testing.assert_allclose(np.diff(grid1.x)[nx-1], -dx*(nx-1))
        np.testing.assert_allclose(np.diff(grid1.y)[nx-1], dx)

        # Another set, but with odd nx
    for _ in range(10):
        dist = np.random.uniform(9.0, 11.0)
        wavelength = np.random.uniform(300e-9, 1100e-9)
        while (nx%2) == 0:
            nx = np.random.randint(10, 20)
        lx = np.random.uniform(1.0, 10.0)
        dx = lx/(nx-1)
        dirCos = np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-1.2, -0.8),
        ])
        dirCos /= np.sqrt(np.dot(dirCos, dirCos))

        grid1 = batoid.RayVector.asGrid(
            dist, wavelength,
            nx=nx, lx=lx, dirCos=dirCos
        )
        grid2 = batoid.RayVector.asGrid(
            dist, wavelength,
            nx=nx, dx=dx, dirCos=dirCos
        )
        grid3 = batoid.RayVector.asGrid(
            dist, wavelength,
            nx=nx, lx=(lx, 0), dirCos=dirCos
        )
        # ... but the following is not equivalent, since default is to always
        # infer an even nx and ny
        # grid4 = batoid.RayVector.asGrid(
        #     dist, wavelength,
        #     dx=1/9, lx=1.0, dirCos=dirCos
        # )

        assert rays_allclose(grid1, grid2)
        assert rays_allclose(grid1, grid3)

        cridx = (nx*nx-1)//2
        obs_dist = np.sqrt(np.dot(grid1[cridx].r, grid1[cridx].r))
        np.testing.assert_allclose(obs_dist, dist)

        np.testing.assert_allclose(grid1.t, 0)
        np.testing.assert_allclose(grid1.wavelength, wavelength)
        np.testing.assert_allclose(grid1.vignetted, False)
        np.testing.assert_allclose(grid1.failed, False)
        np.testing.assert_allclose(grid1.vx, dirCos[0])
        np.testing.assert_allclose(grid1.vy, dirCos[1])
        np.testing.assert_allclose(grid1.vz, dirCos[2])

        # Check distribution of points propagated to entrance pupil
        pupil = batoid.Plane()
        pupil.intersectInPlace(grid1)
        np.testing.assert_allclose(np.diff(grid1.x)[0], dx)
        np.testing.assert_allclose(np.diff(grid1.y)[0], 0, atol=1e-14)
        np.testing.assert_allclose(np.diff(grid1.x)[nx-1], -dx*(nx-1))
        np.testing.assert_allclose(np.diff(grid1.y)[nx-1], dx)

    for _ in range(10):
        # Check nrandom
        rays = batoid.RayVector.asGrid(
            dist, wavelength,
            lx=1.0, nx=1,
            nrandom=1000, dirCos=dirCos
        )

        np.testing.assert_allclose(rays.t, 0)
        np.testing.assert_allclose(rays.wavelength, wavelength)
        np.testing.assert_allclose(rays.vignetted, False)
        np.testing.assert_allclose(rays.failed, False)
        np.testing.assert_allclose(rays.vx, dirCos[0])
        np.testing.assert_allclose(rays.vy, dirCos[1])
        np.testing.assert_allclose(rays.vz, dirCos[2])

        # Check that projected points are inside region
        pupil = batoid.Plane()
        pupil.intersectInPlace(rays)
        np.testing.assert_allclose(rays.z, 0.0)
        np.testing.assert_array_less(rays.x, 0.5)
        np.testing.assert_array_less(rays.y, 0.5)
        np.testing.assert_array_less(-0.5, rays.x)
        np.testing.assert_array_less(-0.5, rays.y)
        assert len(rays) == 1000

    # # Other things to test:
    # #   check x vs y cloning
    # #   check that interface intersection works, for nontrivial interfaces.
    # #   medium works

@timer
def test_RVasPolar():
    for _ in range(10):
        dist = np.random.uniform(9.0, 11.0)
        wavelength = np.random.uniform(300e-9, 1100e-9)
        inner = np.random.uniform(1.0, 3.0)
        outer = inner + np.random.uniform(1.0, 3.0)
        nrad = np.random.randint(1, 10)
        naz = np.random.randint(10, 20)
        dirCos = np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-1.2, -0.8),
        ])
        dirCos /= np.sqrt(np.dot(dirCos, dirCos))

        rays = batoid.RayVector.asPolar(
            dist, wavelength,
            outer, inner,
            nrad=nrad, naz=naz,
            dirCos=dirCos
        )

        np.testing.assert_allclose(rays.t, 0)
        np.testing.assert_allclose(rays.wavelength, wavelength)
        np.testing.assert_allclose(rays.vignetted, False)
        np.testing.assert_allclose(rays.failed, False)
        np.testing.assert_allclose(rays.vx, dirCos[0])
        np.testing.assert_allclose(rays.vy, dirCos[1])
        np.testing.assert_allclose(rays.vz, dirCos[2])

        assert len(rays)%6 == 0

        # If we set inner=0, then last ray should
        # intersect the center of the pupil

        inner = 0.0
        rays = batoid.RayVector.asPolar(
            dist, wavelength,
            outer, inner,
            nrad=nrad, naz=naz,
            dirCos=dirCos
        )
        assert len(rays)%6 == 1

        # Check distribution of points propagated to entrance pupil
        pupil = batoid.Plane()
        pupil.intersectInPlace(rays)
        np.testing.assert_allclose(rays[len(rays)-1].x, 0, atol=1e-14)
        np.testing.assert_allclose(rays[len(rays)-1].y, 0, atol=1e-14)
        np.testing.assert_allclose(rays[len(rays)-1].z, 0, atol=1e-14)


@timer
def test_RVasSpokes():
    for _ in range(10):
        dist = np.random.uniform(9.0, 11.0)
        wavelength = np.random.uniform(300e-9, 1100e-9)
        inner = np.random.uniform(1.0, 3.0)
        outer = inner + np.random.uniform(1.0, 3.0)
        rings = np.random.randint(1, 10)
        spokes = np.random.randint(10, 20)
        dirCos = np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-1.2, -0.8),
        ])
        dirCos /= np.sqrt(np.dot(dirCos, dirCos))

        rays = batoid.RayVector.asSpokes(
            dist, wavelength,
            outer=outer, inner=inner,
            spokes=spokes, rings=rings,
            dirCos=dirCos
        )

        np.testing.assert_allclose(rays.t, 0)
        np.testing.assert_allclose(rays.wavelength, wavelength)
        np.testing.assert_allclose(rays.vignetted, False)
        np.testing.assert_allclose(rays.failed, False)
        np.testing.assert_allclose(rays.vx, dirCos[0])
        np.testing.assert_allclose(rays.vy, dirCos[1])
        np.testing.assert_allclose(rays.vz, dirCos[2])

        assert len(rays) == spokes*rings

        pupil = batoid.Plane()
        pupil.intersectInPlace(rays)
        radii = np.hypot(rays.x, rays.y)

        for i in range(spokes):
            np.testing.assert_allclose(
                radii[rings*i:rings*(i+1)],
                np.linspace(inner, outer, rings, endpoint=True)
            )
        ths = np.arctan2(rays.y, rays.x)

        checkAngle(ths[::rings], np.linspace(0, 2*np.pi, spokes, endpoint=False))

    # rays = batoid.RayVector.asSpokes(
    #     dist, wavelength,
    #     outer=outer, inner=inner,
    #     spokes=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rings=rings,
    #     dirCos=dirCos
    # )
    #
    # rays = batoid.RayVector.asSpokes(
    #     dist, wavelength,
    #     outer=outer, inner=inner,
    #     spokes=6, rings=[0, 1, 2],
    #     dirCos=dirCos
    # )
    #
    # rays = batoid.RayVector.asSpokes(
    #     dist, wavelength,
    #     outer=outer, inner=inner,
    #     spokes=[0,1,2], rings=[0, 1, 2],
    #     dirCos=dirCos
    # )
    #
    # # Gaussian Quadrature rays
    # rays = batoid.RayVector.asSpokes(
    #     dist, wavelength,
    #     spacing='GQ',
    #     rings=6,
    #     dirCos=[0,0,-1]
    # )
    #
    # # Point source
    # source = [0., 1.0, 1.0]
    # rays = batoid.RayVector.asSpokes(
    #     dist, wavelength, source=source,
    #     outer=outer, inner=inner,
    #     spokes=[0,1,2], rings=[0, 1, 2],
    # )


if __name__ == '__main__':
    test_positionAtTime()
    test_properties()
    test_phase()
    test_RayVector()
    test_rayGrid()
    test_circularGrid()
    test_uniformCircularGrid()
    test_pointSourceCircularGrid()
    test_ne()
    test_fail()
    test_RVasGrid()
    test_RVasPolar()
    test_RVasSpokes()
