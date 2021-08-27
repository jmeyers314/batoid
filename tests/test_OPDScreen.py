import batoid
import numpy as np
from test_helpers import timer, do_pickle, init_gpu


@timer
def test_prescreen():
    """Add an OPDScreen in front of LSST entrance pupil.  The OPD that comes out
    should be _negative_ the added phase delay by convention.
    """
    lsst = batoid.Optic.fromYaml("LSST_r.yaml")
    wavelength = 620e-9

    z_ref = batoid.analysis.zernikeGQ(
        lsst, 0, 0, wavelength, rings=10, reference='chief', jmax=37
    )
    rng = np.random.default_rng(577)

    for i in range(4, 38):
        amplitude = rng.uniform(-1, 1)
        zern = batoid.Zernike(
            np.array([0]*i+[amplitude])*wavelength,
            R_outer=4.18
        )

        tel = batoid.CompoundOptic(
            (batoid.optic.OPDScreen(
                batoid.Plane(),
                zern,
                name='PS',
                obscuration=batoid.ObscNegation(batoid.ObscCircle(5.0)),
                coordSys=lsst.stopSurface.coordSys
            ),
            *lsst.items
            ),
            name='PS0',
            backDist=lsst.backDist,
            pupilSize=lsst.pupilSize,
            inMedium=lsst.inMedium,
            stopSurface=lsst.stopSurface,
            sphereRadius=lsst.sphereRadius,
            pupilObscuration=lsst.pupilObscuration
        )
        do_pickle(tel)

        z_test = batoid.analysis.zernikeGQ(
            tel, 0, 0, wavelength, rings=10, reference='chief', jmax=37
        )

        z_expect = np.zeros_like(z_test)
        z_expect[i] = -amplitude  # Longer OPL => negative OPD
        np.testing.assert_allclose(
            (z_test-z_ref)[4:], z_expect[4:],
            rtol=0, atol=5e-4
        )


if __name__ == '__main__':
    init_gpu()
    test_prescreen()
