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


if __name__ == '__main__':
    test_zernikeGQ()
    test_huygensPSF()
