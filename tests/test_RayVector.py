import batoid
import numpy as np
from test_helpers import timer, init_gpu


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


if __name__ == '__main__':
    init_gpu()
    test_properties()
    test_positionAtTime()
    test_propagate()
    test_phase()
    test_sumAmplitude()
    test_equals()
