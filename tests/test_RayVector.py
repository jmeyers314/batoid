import batoid
import numpy as np
from test_helpers import timer, init_gpu, rays_allclose, checkAngle, do_pickle


@timer
def test_properties():
    rng = np.random.default_rng(5)
    size = 10
    for i in range(100):
        x = rng.normal(size=size)
        y = rng.normal(size=size)
        z = rng.normal(size=size)
        vx = rng.normal(size=size)
        vy = rng.normal(size=size)
        vz = rng.normal(size=size)
        t = rng.normal(size=size)
        w = rng.normal(size=size)
        fx = rng.normal(size=size)
        vig = rng.choice([True, False], size=size)
        fa = rng.choice([True, False], size=size)
        cs = batoid.CoordSys(
            origin=rng.normal(size=3),
            rot=batoid.RotX(rng.normal())@batoid.RotY(rng.normal())
        )

        rv = batoid.RayVector(x, y, z, vx, vy, vz, t, w, fx, vig, fa, cs)

        np.testing.assert_array_equal(rv.x, x)
        np.testing.assert_array_equal(rv.y, y)
        np.testing.assert_array_equal(rv.z, z)
        np.testing.assert_array_equal(rv.vx, vx)
        np.testing.assert_array_equal(rv.vy, vy)
        np.testing.assert_array_equal(rv.vz, vz)
        np.testing.assert_array_equal(rv.t, t)
        np.testing.assert_array_equal(rv.wavelength, w)
        np.testing.assert_array_equal(rv.flux, fx)
        np.testing.assert_array_equal(rv.vignetted, vig)
        np.testing.assert_array_equal(rv.failed, fa)
        assert rv.coordSys == cs

        rv._syncToDevice()
        do_pickle(rv)


@timer
def test_positionAtTime():
    rng = np.random.default_rng(57)
    size = 10_000
    x = rng.uniform(-1, 1, size=size)
    y = rng.uniform(-1, 1, size=size)
    z = rng.uniform(-0.1, 0.1, size=size)
    vx = rng.uniform(-0.05, 0.05, size=size)
    vy = rng.uniform(-0.05, 0.05, size=size)
    vz = np.sqrt(1.0 - vx*vx - vy*vy)

    # Try with default t=0 first
    rv = batoid.RayVector(x, y, z, vx, vy, vz)
    np.testing.assert_equal(rv.x, x)
    np.testing.assert_equal(rv.y, y)
    np.testing.assert_equal(rv.z, z)
    np.testing.assert_equal(rv.vx, vx)
    np.testing.assert_equal(rv.vy, vy)
    np.testing.assert_equal(rv.vz, vz)
    np.testing.assert_equal(rv.t, 0.0)
    np.testing.assert_equal(rv.wavelength, 500e-9)

    for t1 in [0.0, 1.0, -1.1, 2.5]:
        np.testing.assert_equal(
            rv.positionAtTime(t1),
            rv.r + t1 * rv.v
        )

    # Now add some random t's
    t = rng.uniform(-1.0, 1.0, size=size)
    rv = batoid.RayVector(x, y, z, vx, vy, vz, t)
    np.testing.assert_equal(rv.x, x)
    np.testing.assert_equal(rv.y, y)
    np.testing.assert_equal(rv.z, z)
    np.testing.assert_equal(rv.vx, vx)
    np.testing.assert_equal(rv.vy, vy)
    np.testing.assert_equal(rv.vz, vz)
    np.testing.assert_equal(rv.t, t)
    np.testing.assert_equal(rv.wavelength, 500e-9)

    for t1 in [0.0, 1.4, -1.3, 2.1]:
        np.testing.assert_equal(
            rv.positionAtTime(t1),
            rv.r + rv.v*(t1-rv.t)[:,None]
        )


@timer
def test_propagate():
    rng = np.random.default_rng(577)
    size = 10_000

    x = rng.uniform(-1, 1, size=size)
    y = rng.uniform(-1, 1, size=size)
    z = rng.uniform(-0.1, 0.1, size=size)
    vx = rng.uniform(-0.05, 0.05, size=size)
    vy = rng.uniform(-0.05, 0.05, size=size)
    vz = np.sqrt(1.0 - vx*vx - vy*vy)
    # Try with default t=0 first
    rv = batoid.RayVector(x, y, z, vx, vy, vz)

    for t1 in [0.0, 1.0, -1.1, 2.5]:
        rvcopy = rv.copy()
        r1 = rv.positionAtTime(t1)
        rvcopy.propagate(t1)
        np.testing.assert_equal(
            rvcopy.r,
            r1
        )
        np.testing.assert_equal(
            rvcopy.v,
            rv.v
        )
        np.testing.assert_equal(
            rvcopy.t,
            t1
        )

    # Now add some random t's
    t = rng.uniform(-1.0, 1.0, size=size)
    rv = batoid.RayVector(x, y, z, vx, vy, vz, t)
    for t1 in [0.0, 1.0, -1.1, 2.5]:
        rvcopy = rv.copy()
        r1 = rv.positionAtTime(t1)
        rvcopy.propagate(t1)
        np.testing.assert_equal(
            rvcopy.r,
            r1
        )
        np.testing.assert_equal(
            rvcopy.v,
            rv.v
        )
        np.testing.assert_equal(
            rvcopy.t,
            t1
        )


@timer
def test_phase():
    rng = np.random.default_rng(5772)
    size = 10_000

    for n in [1.0, 1.3]:
        x = rng.uniform(-1, 1, size=size)
        y = rng.uniform(-1, 1, size=size)
        z = rng.uniform(-0.1, 0.1, size=size)
        vx = rng.uniform(-0.05, 0.05, size=size)
        vy = rng.uniform(-0.05, 0.05, size=size)
        vz = np.sqrt(1.0/(n*n) - vx*vx - vy*vy)
        t = rng.uniform(-1.0, 1.0, size=size)
        wavelength = rng.uniform(300e-9, 1100e-9, size=size)
        rv = batoid.RayVector(x, y, z, vx, vy, vz, t, wavelength)

        # First explicitly check that phase is 0 at position and time of individual
        # rays
        for i in rng.choice(size, size=10):
            np.testing.assert_equal(
                rv.phase(rv.r[i], rv.t[i])[i],
                0.0
            )
        # Now use actual formula
        # phi = k.(r-r0) - (t-t0)omega
        # k = 2 pi v / lambda |v|^2
        # omega = 2 pi / lambda
        # |v| = 1 / n
        for r1, t1 in [
            ((0, 0, 0), 0),
            ((0, 1, 2), 3),
            ((-1, 2, 4), -1),
            ((0, 1, -4), -2)
        ]:
            phi = np.einsum("ij,ij->i", rv.v, r1-rv.r)
            phi *= n*n
            phi -= (t1-rv.t)
            phi *= 2*np.pi/wavelength
            np.testing.assert_allclose(
                rv.phase(r1, t1),
                phi,
                rtol=0,
                atol=1e-7
            )
        for i in rng.choice(size, size=10):
            s = slice(i, i+1)
            rvi = batoid.RayVector(
                x[s], y[s], z[s],
                vx[s], vy[s], vz[s],
                t[s].copy(), wavelength[s].copy()
            )
            # Move integer number of wavelengths ahead
            ti = rvi.t[0]
            wi = rvi.wavelength[0]
            r1 = rvi.positionAtTime(ti + 5123456789*wi)[0]
            a = rvi.amplitude(r1, ti)
            np.testing.assert_allclose(a.real, 1.0, rtol=0, atol=2e-5)
            np.testing.assert_allclose(a.imag, 0.0, rtol=0, atol=2e-5)
            # Half wavelength
            r1 = rvi.positionAtTime(ti + 6987654321.5*wi)[0]
            a = rvi.amplitude(r1, ti)
            np.testing.assert_allclose(a.real, -1.0, rtol=0, atol=2e-5)
            np.testing.assert_allclose(a.imag, 0.0, rtol=0, atol=2e-5)
            # Quarter wavelength
            r1 = rvi.positionAtTime(ti + 0.25*wi)[0]
            a = rvi.amplitude(r1, ti)
            np.testing.assert_allclose(a.real, 0.0, rtol=0, atol=2e-5)
            np.testing.assert_allclose(a.imag, 1.0, rtol=0, atol=2e-5)
            # Three-quarters wavelength
            r1 = rvi.positionAtTime(ti + 7182738495.75*wi)[0]
            a = rvi.amplitude(r1, ti)
            np.testing.assert_allclose(a.real, 0.0, rtol=0, atol=2e-5)
            np.testing.assert_allclose(a.imag, -1.0, rtol=0, atol=2e-5)

            # We can also keep the position the same and change the time in
            # half/quarter integer multiples of the period.
            a = rvi.amplitude(rvi.r[0], rvi.t[0]+5e9*wi)
            np.testing.assert_allclose(a.real, 1.0, rtol=0, atol=1e-5)
            np.testing.assert_allclose(a.imag, 0.0, rtol=0, atol=1e-5)
            a = rvi.amplitude(rvi.r[0], rvi.t[0]+(5e9+5.5)*wi)
            np.testing.assert_allclose(a.real, -1.0, rtol=0, atol=1e-5)
            np.testing.assert_allclose(a.imag, 0.0, rtol=0, atol=1e-5)
            a = rvi.amplitude(rvi.r[0], rvi.t[0]+(5e9+2.25)*wi)
            np.testing.assert_allclose(a.real, 0.0, rtol=0, atol=1e-5)
            np.testing.assert_allclose(a.imag, -1.0, rtol=0, atol=1e-5)
            a = rvi.amplitude(rvi.r[0], rvi.t[0]+(5e9+1.75)*wi)
            np.testing.assert_allclose(a.real, 0.0, rtol=0, atol=1e-5)
            np.testing.assert_allclose(a.imag, 1.0, rtol=0, atol=1e-5)

            # If we pick a point anywhere along a vector originating at the ray
            # position, but orthogonal to its direction of propagation, then we
            # should get phase = 0 (mod 2pi).
            v1 = np.array([1.0, 0.0, 0.0])
            v1 = np.cross(rvi.v[0], v1)
            p1 = rvi.r[0] + v1
            a = rvi.amplitude(p1, rvi.t[0])
            np.testing.assert_allclose(a.real, 1.0, rtol=0, atol=1e-5)
            np.testing.assert_allclose(a.imag, 0.0, rtol=0, atol=1e-5)


@timer
def test_sumAmplitude():
    import time
    rng = np.random.default_rng(57721)
    size = 10_000

    for n in [1.0, 1.3]:
        x = rng.uniform(-1, 1, size=size)
        y = rng.uniform(-1, 1, size=size)
        z = rng.uniform(-0.1, 0.1, size=size)
        vx = rng.uniform(-0.05, 0.05, size=size)
        vy = rng.uniform(-0.05, 0.05, size=size)
        vz = np.sqrt(1.0/(n*n) - vx*vx - vy*vy)
        t = rng.uniform(-1.0, 1.0, size=size)
        wavelength = rng.uniform(300e-9, 1100e-9, size=size)
        rv = batoid.RayVector(x, y, z, vx, vy, vz, t, wavelength)
        satime = 0
        atime = 0
        for r1, t1 in [
            ((0, 0, 0), 0),
            ((0, 1, 2), 3),
            ((-1, 2, 4), -1),
            ((0, 1, -4), -2)
        ]:
            at0 = time.time()
            s1 = rv.sumAmplitude(r1, t1)
            at1 = time.time()
            s2 = np.sum(rv.amplitude(r1, t1))
            at2 = time.time()

            np.testing.assert_allclose(s1, s2, rtol=0, atol=1e-11)
            satime += at1-at0
            atime += at2-at1
        # print(f"sumAplitude() time: {satime}")
        # print(f"np.sum(amplitude()) time: {atime}")


@timer
def test_equals():
    import time
    rng = np.random.default_rng(577215)
    size = 10_000

    x = rng.uniform(-1, 1, size=size)
    y = rng.uniform(-1, 1, size=size)
    z = rng.uniform(-0.1, 0.1, size=size)
    vx = rng.uniform(-0.05, 0.05, size=size)
    vy = rng.uniform(-0.05, 0.05, size=size)
    vz = np.sqrt(1.0 - vx*vx - vy*vy)
    t = rng.uniform(-1.0, 1.0, size=size)
    wavelength = rng.uniform(300e-9, 1100e-9, size=size)
    flux = rng.uniform(0.9, 1.1, size=size)
    vignetted = rng.choice([True, False], size=size)
    failed = rng.choice([True, False], size=size)

    args = x, y, z, vx, vy, vz, t, wavelength, flux, vignetted, failed
    rv = batoid.RayVector(*args)
    rv2 = rv.copy()
    assert rv == rv2

    for i in range(len(args)):
        newargs = [args[i].copy() for i in range(len(args))]
        ai = newargs[i]
        if ai.dtype == float:
            ai[0] = 1.2+ai[0]*3.45
        elif ai.dtype == bool:
            ai[0] = not ai[0]
        # else panic!
        rv2 = batoid.RayVector(*newargs)
        assert rv != rv2

    # Repeat, but force comparison on device
    rv2 = rv.copy()
    rv._rv.r.syncToDevice()
    rv._rv.v.syncToDevice()
    rv._rv.t.syncToDevice()
    rv._rv.wavelength.syncToDevice()
    rv._rv.flux.syncToDevice()
    rv._rv.vignetted.syncToDevice()
    rv._rv.failed.syncToDevice()
    assert rv == rv2
    for i in range(len(args)):
        newargs = [args[i].copy() for i in range(len(args))]
        ai = newargs[i]
        if ai.dtype == float:
            ai[0] = 1.2+ai[0]*3.45
        elif ai.dtype == bool:
            ai[0] = not ai[0]
        # else panic!
        rv2 = batoid.RayVector(*newargs)
        assert rv != rv2


@timer
def test_asGrid():
    rng = np.random.default_rng(5772156)
    for _ in range(10):
        backDist = rng.uniform(9.0, 11.0)
        wavelength = rng.uniform(300e-9, 1100e-9)
        nx = 1
        while (nx%2) == 1:
            nx = rng.integers(10, 21)
        lx = rng.uniform(1.0, 10.0)
        dx = lx/(nx-2)
        dirCos = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(-1.2, -0.8),
        ])
        dirCos /= np.sqrt(np.dot(dirCos, dirCos))

        # Some things that should be equivalent
        grid1 = batoid.RayVector.asGrid(
            backDist=backDist, wavelength=wavelength,
            nx=nx, lx=lx, dirCos=dirCos
        )
        grid2 = batoid.RayVector.asGrid(
            backDist=backDist, wavelength=wavelength,
            nx=nx, dx=dx, dirCos=dirCos
        )
        grid3 = batoid.RayVector.asGrid(
            backDist=backDist, wavelength=wavelength,
            dx=dx, lx=lx, dirCos=dirCos
        )
        grid4 = batoid.RayVector.asGrid(
            backDist=backDist, wavelength=wavelength,
            nx=nx, lx=(lx, 0.0), dirCos=dirCos
        )
        theta_x, theta_y = batoid.utils.dirCosToField(*dirCos)
        grid5 = batoid.RayVector.asGrid(
            backDist=backDist, wavelength=wavelength,
            nx=nx, lx=(lx, 0.0), theta_x=theta_x, theta_y=theta_y
        )
        rays_allclose(grid1, grid2)
        rays_allclose(grid1, grid3)
        rays_allclose(grid1, grid4)
        rays_allclose(grid1, grid5)

        # Check distance to chief ray
        cridx = (nx//2)*nx+nx//2
        obs_dist = np.sqrt(np.dot(grid1.r[cridx], grid1.r[cridx]))
        np.testing.assert_allclose(obs_dist, backDist)

        np.testing.assert_allclose(grid1.t, 0)
        np.testing.assert_allclose(grid1.wavelength, wavelength)
        np.testing.assert_allclose(grid1.vignetted, False)
        np.testing.assert_allclose(grid1.failed, False)
        np.testing.assert_allclose(grid1.vx, dirCos[0])
        np.testing.assert_allclose(grid1.vy, dirCos[1])
        np.testing.assert_allclose(grid1.vz, dirCos[2])

        # Check distribution of points propagated to entrance pupil
        pupil = batoid.Plane()
        pupil.intersect(grid1)
        np.testing.assert_allclose(np.diff(grid1.x)[0], dx)
        np.testing.assert_allclose(np.diff(grid1.y)[0], 0, atol=1e-14)
        np.testing.assert_allclose(np.diff(grid1.x)[nx-1], -dx*(nx-1))
        np.testing.assert_allclose(np.diff(grid1.y)[nx-1], dx)

        # Another set, but with odd nx
    for _ in range(10):
        backDist = rng.uniform(9.0, 11.0)
        wavelength = rng.uniform(300e-9, 1100e-9)
        while (nx%2) == 0:
            nx = rng.integers(10, 21)
        lx = rng.uniform(1.0, 10.0)
        dx = lx/(nx-1)
        dirCos = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(-1.2, -0.8),
        ])
        dirCos /= np.sqrt(np.dot(dirCos, dirCos))

        grid1 = batoid.RayVector.asGrid(
            backDist=backDist, wavelength=wavelength,
            nx=nx, lx=lx, dirCos=dirCos
        )
        grid2 = batoid.RayVector.asGrid(
            backDist=backDist, wavelength=wavelength,
            nx=nx, dx=dx, dirCos=dirCos
        )
        grid3 = batoid.RayVector.asGrid(
            backDist=backDist, wavelength=wavelength,
            nx=nx, lx=(lx, 0), dirCos=dirCos
        )
        # ... but the following is not equivalent, since default is to always
        # infer an even nx and ny
        # grid4 = batoid.RayVector.asGrid(
        #     backDist=backDist, wavelength=wavelength,
        #     dx=1/9, lx=1.0, dirCos=dirCos
        # )

        rays_allclose(grid1, grid2)
        rays_allclose(grid1, grid3)

        cridx = (nx*nx-1)//2
        obs_dist = np.sqrt(np.dot(grid1.r[cridx], grid1.r[cridx]))
        np.testing.assert_allclose(obs_dist, backDist)

        np.testing.assert_allclose(grid1.t, 0)
        np.testing.assert_allclose(grid1.wavelength, wavelength)
        np.testing.assert_allclose(grid1.vignetted, False)
        np.testing.assert_allclose(grid1.failed, False)
        np.testing.assert_allclose(grid1.vx, dirCos[0])
        np.testing.assert_allclose(grid1.vy, dirCos[1])
        np.testing.assert_allclose(grid1.vz, dirCos[2])

        # Check distribution of points propagated to entrance pupil
        pupil = batoid.Plane()
        pupil.intersect(grid1)
        np.testing.assert_allclose(np.diff(grid1.x)[0], dx)
        np.testing.assert_allclose(np.diff(grid1.y)[0], 0, atol=1e-14)
        np.testing.assert_allclose(np.diff(grid1.x)[nx-1], -dx*(nx-1))
        np.testing.assert_allclose(np.diff(grid1.y)[nx-1], dx)

    for _ in range(10):
        # Check nrandom
        rays = batoid.RayVector.asGrid(
            backDist=backDist, wavelength=wavelength,
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
        pupil.intersect(rays)
        np.testing.assert_allclose(rays.z, 0.0)
        np.testing.assert_array_less(rays.x, 0.5)
        np.testing.assert_array_less(rays.y, 0.5)
        np.testing.assert_array_less(-0.5, rays.x)
        np.testing.assert_array_less(-0.5, rays.y)
        assert len(rays) == 1000


@timer
def test_asPolar():
    rng = np.random.default_rng(5772156)
    for _ in range(10):
        backDist = rng.uniform(9.0, 11.0)
        wavelength = rng.uniform(300e-9, 1100e-9)
        inner = rng.uniform(1.0, 3.0)
        outer = inner + rng.uniform(1.0, 3.0)
        nrad = rng.integers(1, 11)
        naz = rng.integers(10, 21)
        dirCos = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(-1.2, -0.8),
        ])
        dirCos /= np.sqrt(np.dot(dirCos, dirCos))

        rays = batoid.RayVector.asPolar(
            backDist=backDist, wavelength=wavelength,
            outer=outer, inner=inner,
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
            backDist=backDist, wavelength=wavelength,
            outer=outer, inner=inner,
            nrad=nrad, naz=naz,
            dirCos=dirCos
        )
        assert len(rays)%6 == 1

        pupil = batoid.Plane()
        pupil.intersect(rays)
        np.testing.assert_allclose(rays.x[-1], 0, atol=1e-14)
        np.testing.assert_allclose(rays.y[-1], 0, atol=1e-14)
        np.testing.assert_allclose(rays.z[-1], 0, atol=1e-14)


@timer
def test_asSpokes():
    rng = np.random.default_rng(5772156)
    for _ in range(10):
        backDist = rng.uniform(9.0, 11.0)
        wavelength = rng.uniform(300e-9, 1100e-9)
        inner = rng.uniform(1.0, 3.0)
        outer = inner + rng.uniform(1.0, 3.0)
        rings = rng.integers(1, 11)
        spokes = rng.integers(10, 21)
        dirCos = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(-1.2, -0.8),
        ])
        dirCos /= np.sqrt(np.dot(dirCos, dirCos))

        rays = batoid.RayVector.asSpokes(
            backDist=backDist, wavelength=wavelength,
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
        pupil.intersect(rays)
        radii = np.hypot(rays.x, rays.y)
        ths = np.arctan2(rays.y, rays.x)

        for i in range(spokes):
            np.testing.assert_allclose(
                radii[rings*i:rings*(i+1)],
                np.linspace(inner, outer, rings, endpoint=True)
            )
        for i in range(rings):
            checkAngle(ths[i::rings], np.linspace(0, 2*np.pi, spokes, endpoint=False))

        # Check explicit rings and spokes
        rings = rng.uniform(inner, outer, rings)
        spokes = rng.uniform(0, 2*np.pi, spokes)

        rays = batoid.RayVector.asSpokes(
            backDist=backDist, wavelength=wavelength,
            outer=outer, inner=inner,
            rings=rings, spokes=spokes,
            dirCos=dirCos
        )

        pupil = batoid.Plane()
        pupil.intersect(rays)
        radii = np.hypot(rays.x, rays.y)
        ths = np.arctan2(rays.y, rays.x)

        for i in range(len(spokes)):
            np.testing.assert_allclose(
                radii[len(rings)*i:len(rings)*(i+1)],
                rings
            )
        for i in range(len(rings)):
            checkAngle(
                ths[i::len(rings)],
                spokes
            )

        # Check Gaussian Quadrature
        rings = rng.integers(5, 11)
        spokes = 2*rings+1
        rays = batoid.RayVector.asSpokes(
            backDist=backDist, wavelength=wavelength,
            outer=outer,
            rings=rings,
            spacing='GQ',
            dirCos=dirCos
        )
        assert len(rays) == spokes*rings

        pupil = batoid.Plane()
        pupil.intersect(rays)
        radii = np.hypot(rays.x, rays.y)
        ths = np.arctan2(rays.y, rays.x)

        Li, w = np.polynomial.legendre.leggauss(rings)
        rings = np.sqrt((1+Li)/2)*outer
        flux = w*np.pi/(2*spokes)
        spokes = np.linspace(0, 2*np.pi, spokes, endpoint=False)

        for i in range(len(spokes)):
            np.testing.assert_allclose(
                radii[len(rings)*i:len(rings)*(i+1)],
                rings
            )
            np.testing.assert_allclose(
                rays.flux[len(rings)*i:len(rings)*(i+1)],
                flux
            )

        for i in range(len(rings)):
            checkAngle(
                ths[i::len(rings)],
                spokes
            )

    # Sanity check GQ grids against literature
    # Values from Forbes JOSA Vol. 5, No. 11 (1988) Table 1
    rings = [1, 2, 3, 4, 5, 6]
    rad = [
        [0.70710678],
        [0.45970084, 0.88807383],
        [0.33571069, 0.70710678, 0.94196515],
        [0.26349923, 0.57446451, 0.81852949, 0.96465961],
        [0.21658734, 0.48038042, 0.70710678, 0.87706023, 0.97626324],
        [0.18375321, 0.41157661, 0.61700114, 0.78696226, 0.91137517, 0.98297241]
    ]
    w = [
        [0.5],
        [0.25, 0.25],
        [0.13888889, 0.22222222, 0.13888889],
        [0.08696371, 0.16303629, 0.16303629, 0.08696371],
        [0.05923172, 0.11965717, 0.14222222, 0.11965717, 0.05923172],
        [0.04283112, 0.09019039, 0.11697848, 0.11697848, 0.09019039, 0.04283112]
    ]

    for rings_, rad_, w_ in zip(rings, rad, w):
        rays = batoid.RayVector.asSpokes(
            backDist=backDist, wavelength=wavelength,
            outer=1,
            rings=rings_,
            spacing='GQ',
            dirCos=[0,0,-1]
        )
        spokes = rings_*2+1

        radii = np.hypot(rays.x, rays.y)
        for i in range(spokes):
            np.testing.assert_allclose(
                radii[rings_*i:rings_*(i+1)],
                rad_
            )
            np.testing.assert_allclose(
                rays.flux[rings_*i:rings_*(i+1)]*spokes/(2*np.pi),
                w_
            )


@timer
def test_factory_optic():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")

    grid1 = batoid.RayVector.asGrid(
        optic=telescope, wavelength=500e-9, theta_x=0.1, theta_y=0.1,
        nx=16
    )
    grid2 = batoid.RayVector.asGrid(
        wavelength=500e-9, theta_x=0.1, theta_y=0.1,
        backDist=telescope.backDist, stopSurface=telescope.stopSurface,
        medium=telescope.inMedium, lx=telescope.pupilSize,
        nx=16
    )
    rays_allclose(grid1, grid2)

    grid1 = batoid.RayVector.asPolar(
        optic=telescope, wavelength=500e-9, theta_x=0.1, theta_y=0.1,
        naz=100, nrad=20
    )
    grid2 = batoid.RayVector.asPolar(
        wavelength=500e-9, theta_x=0.1, theta_y=0.1,
        backDist=telescope.backDist, stopSurface=telescope.stopSurface,
        medium=telescope.inMedium, outer=telescope.pupilSize/2,
        naz=100, nrad=20
    )
    rays_allclose(grid1, grid2)

    grid1 = batoid.RayVector.asSpokes(
        optic=telescope, wavelength=500e-9, theta_x=0.1, theta_y=0.1,
        rings=10, spokes=21
    )
    grid2 = batoid.RayVector.asSpokes(
        wavelength=500e-9, theta_x=0.1, theta_y=0.1,
        backDist=telescope.backDist, stopSurface=telescope.stopSurface,
        medium=telescope.inMedium, outer=telescope.pupilSize/2,
        rings=10, spokes=21
    )
    rays_allclose(grid1, grid2)


if __name__ == '__main__':
    init_gpu()
    test_properties()
    test_positionAtTime()
    test_propagate()
    test_phase()
    test_sumAmplitude()
    test_equals()
    test_asGrid()
    test_asPolar()
    test_asSpokes()
    test_factory_optic()

