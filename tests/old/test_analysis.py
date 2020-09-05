import pytest
import numpy as np
import batoid
from test_helpers import timer


hasGalSim = True
try:
    import galsim
except ImportError:
    hasGalSim = False


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_zernikeGQ():
    if __name__ == '__main__':
        nx=1024
        rings=10
        tol=1e-4
    else:
        nx=128
        rings=5
        tol=1e-3
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    telescope.clearObscuration()
    telescope['LSST.M1'].obscuration = batoid.ObscNegation(batoid.ObscCircle(4.18))
    zSquare = batoid.analysis.zernike(
        telescope, 0.0, 0.0, 625e-9,
        nx=nx, jmax=28, reference='chief'
    )
    zGQ = batoid.analysis.zernikeGQ(
        telescope, 0.0, 0.0, 625e-9,
        rings=rings, jmax=28, reference='chief'
    )

    np.testing.assert_allclose(
        zSquare, zGQ, rtol=0, atol=tol
    )

    # Repeat with annular Zernikes
    telescope['LSST.M1'].obscuration = batoid.ObscNegation(batoid.ObscAnnulus(0.61*4.18, 4.18))
    zSquare = batoid.analysis.zernike(
        telescope, 0.0, 0.0, 625e-9,
        nx=nx, jmax=28, reference='chief', eps=0.61
    )
    zGQ = batoid.analysis.zernikeGQ(
        telescope, 0.0, 0.0, 625e-9,
        rings=rings, jmax=28, reference='chief', eps=0.61
    )

    np.testing.assert_allclose(
        zSquare, zGQ, rtol=0, atol=tol
    )

    # Try off-axis
    zSquare = batoid.analysis.zernike(
        telescope, np.deg2rad(0.2), np.deg2rad(0.1), 625e-9,
        nx=nx, jmax=28, reference='chief', eps=0.61
    )
    zGQ = batoid.analysis.zernikeGQ(
        telescope, np.deg2rad(0.2), np.deg2rad(0.1), 625e-9,
        rings=rings, jmax=28, reference='chief', eps=0.61
    )

    np.testing.assert_allclose(
        zSquare, zGQ, rtol=0, atol=tol
    )

    # Try reference == mean
    # Try off-axis
    zSquare = batoid.analysis.zernike(
        telescope, np.deg2rad(0.2), np.deg2rad(0.1), 625e-9,
        nx=nx, jmax=28, reference='mean', eps=0.61
    )
    zGQ = batoid.analysis.zernikeGQ(
        telescope, np.deg2rad(0.2), np.deg2rad(0.1), 625e-9,
        rings=rings, jmax=28, reference='mean', eps=0.61
    )
    # Z1-3 less reliable, but mostly uninteresting anyway...
    np.testing.assert_allclose(
        zSquare[4:], zGQ[4:], rtol=0, atol=tol
    )


@timer
def test_huygensPSF():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")

    # Test that we can infer dy from dx properly
    psf1 = batoid.analysis.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=10e-6,
    )
    psf2 = batoid.analysis.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=10e-6,
        dy=10e-6
    )
    assert psf1 == psf2

    # Test vector vs scalar dx,dy
    psf1 = batoid.analysis.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=[10e-6, 0],
        dy=[0, 11e-6]
    )
    psf2 = batoid.analysis.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=10e-6,
        dy=11e-6
    )
    assert psf1 == psf2

    # Should still work with reference = 'chief'
    psf3 = batoid.analysis.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=[10e-6, 0],
        dy=[0, 11e-6],
        reference='chief'
    )
    psf4 = batoid.analysis.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=10e-6,
        dy=11e-6,
        reference='chief'
    )
    assert psf1 != psf3
    assert psf3 == psf4

    # And just cover nx odd
    psf = batoid.analysis.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=63,
    )


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_doubleZernike():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    dz = batoid.analysis.doubleZernike(
        telescope,
        np.deg2rad(1.75),
        625e-9,
        10,
        kmax=28,
        jmax=22
    )

    # Now evaluate DZ a few places and compare with zernikeGQ
    size = 20
    js = np.random.randint(4, 22, size=size)
    thr = np.deg2rad(np.sqrt(np.random.uniform(0, 1.75**2, size=size)))
    thth = np.random.uniform(0, 2*np.pi, size=size)
    thx = thr*np.cos(thth)
    thy = thr*np.sin(thth)

    for j in js:
        Z = galsim.zernike.Zernike(dz[:,j], R_inner=0.0, R_outer=np.deg2rad(1.75))
        for thx_, thy_ in zip(thx, thy):
            zGQ = batoid.analysis.zernikeGQ(
                telescope, thx_, thy_,
                625e-9,
                jmax=22
            )
            np.testing.assert_allclose(Z(thx_, thy_), zGQ[j], rtol=0, atol=1e-4)

    # Check that we get similar results with different number of rings/spokes
    dz2 = batoid.analysis.doubleZernike(
        telescope,
        np.deg2rad(1.75),
        625e-9,
        rings=12,
        spokes=29,
        kmax=28,
        jmax=22
    )
    np.testing.assert_allclose(dz, dz2, rtol=0, atol=1e-2)


if __name__ == '__main__':
    test_zernikeGQ()
    test_huygensPSF()
    test_doubleZernike()
