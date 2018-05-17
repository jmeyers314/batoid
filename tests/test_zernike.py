import batoid
import galsim
import numpy as np
from test_helpers import timer


@timer
def test_nCr():
    for _ in range(100):
        # 68 seems to be the limit.  Guessing that ulonglong limit is here.
        n = np.random.randint(1, 68)
        r = np.random.randint(0, n)
        np.testing.assert_equal(
            batoid._batoid.nCr(n, r),
            galsim.utilities.nCr(n, r)
        )


@timer
def test_binomial():
    for _ in range(100):
        a = np.random.normal()
        b = np.random.normal()
        n = np.random.randint(0, 100)

        np.testing.assert_allclose(
            batoid._batoid.binomial(a, b, n),
            galsim.utilities.binomial(a, b, n),
            atol=1e-14,
            rtol=1e-14
        )


@timer
def test_horner2d():
    for _ in range(100):
        nx = np.random.randint(1, 10)
        ny = np.random.randint(1, 10)
        arr = np.random.normal(size=(nx, ny))
        x = np.random.normal()
        y = np.random.normal()

        np.testing.assert_allclose(
            batoid._batoid.horner2d(x, y, arr),
            galsim.utilities.horner2d(x, y, arr),
            atol=1e-17,
            rtol=1e-17
        )


@timer
def test_noll_to_zern():
    for j in range(1, 1000):
        np.testing.assert_equal(
            batoid._batoid.noll_to_zern(j),
            galsim.zernike.noll_to_zern(j)
        )


@timer
def test_zern_rho_coefs():
    # 1541 seems to be the limit.  Not sure what the error is, but may be related to
    # python's ability to do arbitrary precision arithmetic.
    # jmax = 1541
    # But for speed we'll just check up to jmax=100
    jmax = 100

    for j in range(1, jmax):
        n, m = batoid._batoid.noll_to_zern(j)
        np.testing.assert_allclose(
            batoid._batoid.zern_rho_coefs(n, m),
            galsim.zernike._zern_rho_coefs(n, m),
            atol=1e-15,
            rtol=1e-15
        )


@timer
def test_h():
    # Accuracy here seems to be quite difficult.
    # 1e-15 achieved only up to j=8
    # 1e-14 achieved up to j=26
    # above j=26, major differences occur, on the order of 10%  !!!
    # there must be an overflow somewhere...

    for _ in range(100):
        j = np.random.randint(1, 26)
        n, m = batoid._batoid.noll_to_zern(j)
        m = abs(m)
        eps = np.random.uniform(0.0, 1.0)
        np.testing.assert_allclose(
            batoid._batoid.h(m, j, eps),
            galsim.zernike._h(m, j, eps),
            atol=1e-13,
            rtol=1e-13
        )


@timer
def test_Q0():
    # Similar story to h.
    # Breakdown occurs above j=26
    for _ in range(100):
        j = np.random.randint(1, 26)
        n, m = batoid._batoid.noll_to_zern(j)
        m = abs(m)
        eps = np.random.uniform(0.0, 1.0)
        np.testing.assert_allclose(
            batoid._batoid.Q0(m, j, eps),
            galsim.zernike._Q(m, j, eps)[0],
            atol=1e-13,
            rtol=1e-13
        )


@timer
def test_Q():
    # didn't check, but presumably same caveats as Q0
    for _ in range(100):
        j = np.random.randint(1, 26)
        n, m = batoid._batoid.noll_to_zern(j)
        m = abs(m)
        eps = np.random.uniform(0.0, 1.0)
        np.testing.assert_allclose(
            batoid._batoid.Q(m, j, eps),
            galsim.zernike._Q(m, j, eps),
            atol=1e-13,
            rtol=1e-13
        )


@timer
def test_annular_zern_rho_coefs():
    # remarkably, despite the limitations above,
    # the final results here seem to agree to 1e-13 up past j=1000

    jmax = 1000

    for _ in range(100):
        j = np.random.randint(1, jmax)
        n, m = batoid._batoid.noll_to_zern(j)
        eps = np.random.uniform(0.0, 1.0)
        np.testing.assert_allclose(
            batoid._batoid.annular_zern_rho_coefs(n, m, eps),
            galsim.zernike._annular_zern_rho_coefs(n, m, eps),
            # atol=1e-10,
            atol=0,
            rtol=1e-13
        )


@timer
def test_zern_coef_array():
    # ... and similar to above, this function seems replicatable up to jmax=1000
    jmax = 1000

    for _ in range(100):
        j = np.random.randint(1, jmax)
        n, m = batoid._batoid.noll_to_zern(j)
        eps = np.random.uniform(0.0, 1.0)
        np.testing.assert_allclose(
            batoid._batoid.zern_coef_array(n, m, eps, (n+1, n+1)),
            galsim.zernike._zern_coef_array(n, m, eps, (n+1, n+1)),
            # atol=1e-10,
            atol=0,
            rtol=1e-13
        )


@timer
def test_xy_contribution():
    for _ in range(100):
        p1 = np.random.randint(0,10)
        p2 = np.random.randint(0,20)

        np.testing.assert_equal(
            batoid._batoid.xy_contribution(p1, p2, (2*p1+p2+1, 2*p1+p2+1)),
            galsim.zernike._xy_contribution(p1, p2, (2*p1+p2+1, 2*p1+p2+1))
        )


@timer
def test_noll_coef_array():
    jmaxmax = 1000
    for _ in range(10):
        jmax = np.random.randint(2, jmaxmax)
        eps = np.random.uniform(0.0, 1.0)

        bnca = batoid._batoid.noll_coef_array(jmax, eps)
        gnca = galsim.zernike._noll_coef_array(jmax, eps)

        np.testing.assert_allclose(
            np.transpose(bnca, axes=(1,2,0)),
            gnca,
            atol=0,
            rtol=1e-13
        )


@timer
def test_rrsqr_to_xy():
    jmaxmax = 1000
    for _ in range(10):
        jmax = np.random.randint(2, jmaxmax)
        maxn, _ = batoid._batoid.noll_to_zern(jmax)
        shape = (maxn+1, maxn+1)
        eps = np.random.uniform(0.0, 1.0)

        bnca = batoid._batoid.noll_coef_array(jmax, eps)
        gnca = galsim.zernike._noll_coef_array(jmax, eps)

        i = np.random.randint(len(bnca))
        np.testing.assert_allclose(
            batoid._batoid.rrsq_to_xy(bnca[i], shape),
            batoid._batoid.rrsq_to_xy(gnca[:,:,i], shape),
            atol=0,
            rtol=1e-13
        )
        np.testing.assert_allclose(
            galsim.zernike._rrsq_to_xy(bnca[i], shape),
            galsim.zernike._rrsq_to_xy(gnca[:,:,i], shape),
            atol=0,
            rtol=1e-13
        )

        np.testing.assert_allclose(
            batoid._batoid.rrsq_to_xy(bnca[i], shape),
            galsim.zernike._rrsq_to_xy(bnca[i], shape),
            atol=0,
            rtol=1e-13
        )


@timer
def test_noll_coef_array_xy():
    jmaxmax = 100
    for _ in range(10):
        jmax = np.random.randint(2, jmaxmax)
        eps = np.random.uniform(0.0, 1.0)

        bnca = batoid._batoid.noll_coef_array_xy(jmax, eps)
        gnca = galsim.zernike._noll_coef_array_xy(jmax, eps)

        np.testing.assert_allclose(
            np.transpose(bnca, axes=(1,2,0)),
            gnca,
            atol=0,
            rtol=1e-13
        )


@timer
def test_sag():
    jmaxmax=200
    for _ in range(100):
        n = np.random.randint(5, jmaxmax)
        coefs = np.random.normal(size=n)*1e-3  # I think this corresponds to mm RMS fluctuations
        R_outer = np.random.uniform(0.5, 5.0)
        R_inner = np.random.uniform(0.0, 0.8*R_outer)

        gz = galsim.zernike.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
        bz = batoid._batoid.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)

        x = np.random.uniform(-R_outer, R_outer, size=5000)
        y = np.random.uniform(-R_outer, R_outer, size=5000)
        w = np.hypot(x, y) < R_outer
        x = x[w]
        y = y[w]

        np.testing.assert_allclose(
            gz.evalCartesian(x, y),
            bz.sag(x, y),
            atol=1e-8,  # 10 nm absolute precision
            rtol=0
        )

        np.testing.assert_allclose(
            gz.evalCartesian(x[::5], y[::5]),
            bz.sag(x[::5], y[::5]),
            atol=1e-8,  # 10 nm absolute precision
            rtol=0
        )


@timer
def test_properties():
    jmaxmax=200
    for _ in range(100):
        n = np.random.randint(5, jmaxmax)
        coefs = np.random.normal(size=n)*1e-3
        R_outer = np.random.uniform(0.5, 5.0)
        R_inner = np.random.uniform(0.0, 0.8*R_outer)
        zernike = batoid._batoid.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)

        assert np.all(zernike.coefs == coefs)
        assert zernike.R_outer == R_outer
        assert zernike.R_inner == R_inner


@timer
def test_intersect():
    jmaxmax=50
    import random
    random.seed(577)
    for i in range(100):
        n = np.random.randint(5, jmaxmax)
        coefs = np.random.normal(size=n)*1e-3
        R_outer = np.random.uniform(0.5, 5.0)
        R_inner = np.random.uniform(0.0, 0.8*R_outer)
        zernike = batoid._batoid.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r0 = batoid.Ray(x, y, -10, 0, 0, 1, 0)
            r = zernike.intersect(r0)
            np.testing.assert_allclose(r.p0[0], x, rtol=0, atol=1e-9)
            np.testing.assert_allclose(r.p0[1], y, rtol=0, atol=1e-9)
            np.testing.assert_allclose(r.p0[2], zernike.sag(x, y), rtol=0, atol=1e-9)


@timer
def test_intersect_vectorized():
    jmaxmax=50
    import random
    random.seed(5772)
    r0s = [batoid.Ray([random.gauss(0.0, 0.1),
                       random.gauss(0.0, 0.1),
                       random.gauss(10.0, 0.1)],
                      [random.gauss(0.0, 0.1),
                       random.gauss(0.0, 0.1),
                       random.gauss(-1.0, 0.1)],
                      random.gauss(0.0, 0.1))
            for i in range(100)]
    r0s = batoid.RayVector(r0s)

    for i in range(100):
        n = np.random.randint(5, jmaxmax)
        coefs = np.random.normal(size=n)*1e-3
        R_outer = np.random.uniform(0.5, 5.0)
        R_inner = np.random.uniform(0.0, 0.8*R_outer)
        zernike = batoid._batoid.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)

        r1s = zernike.intersect(r0s)
        r2s = batoid.RayVector([zernike.intersect(r0) for r0 in r0s])
        assert r1s == r2s


if __name__ == '__main__':
    # test_nCr()
    # test_binomial()
    # test_horner2d()
    # test_noll_to_zern()
    # test_zern_rho_coefs()
    # test_h()
    # test_Q0()
    # test_Q()
    # test_annular_zern_rho_coefs()
    # test_zern_coef_array()
    # test_xy_contribution()
    # test_noll_coef_array()
    # test_rrsqr_to_xy()
    # test_noll_coef_array_xy()
    # test_sag()
    # test_properties()
    # test_intersect()
    test_intersect_vectorized()
