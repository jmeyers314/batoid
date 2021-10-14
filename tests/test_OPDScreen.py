import batoid
import numpy as np
from test_helpers import timer, do_pickle, init_gpu, rays_allclose


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
            (
                batoid.optic.OPDScreen(
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


@timer
def test_zeroscreen():
    """Add a zero phase OPDScreen in front of LSST entrance pupil.  Should have
    _no_ effect.
    """
    lsst = batoid.Optic.fromYaml("LSST_r.yaml")

    screens = [
        batoid.optic.OPDScreen(
            batoid.Plane(),
            batoid.Plane(),
            name='PS',
            coordSys=lsst.stopSurface.coordSys
        ),
        batoid.optic.OPDScreen(
            batoid.Paraboloid(100.0),
            batoid.Plane(),
            name='PS',
            coordSys=lsst.stopSurface.coordSys
        ),
        batoid.optic.OPDScreen(
            batoid.Quadric(11.0, -0.5),
            batoid.Plane(),
            name='PS',
            coordSys=lsst.stopSurface.coordSys
        ),
        batoid.optic.OPDScreen(
            batoid.Zernike([0, 0, 0, 0, 300e-9, 0, 0, 400e-9, -600e-9]),
            batoid.Zernike([0]*22),
            name='PS',
            coordSys=lsst.stopSurface.coordSys
        )
    ]

    for screen in screens:
        tel = batoid.CompoundOptic(
            (screen, *lsst.items),
            name='PS0',
            backDist=lsst.backDist,
            pupilSize=lsst.pupilSize,
            inMedium=lsst.inMedium,
            stopSurface=lsst.stopSurface,
            sphereRadius=lsst.sphereRadius,
            pupilObscuration=lsst.pupilObscuration
        )
        do_pickle(tel)

        rng = np.random.default_rng(57)
        thx = np.deg2rad(rng.uniform(-1, 1))
        thy = np.deg2rad(rng.uniform(-1, 1))
        rays = batoid.RayVector.asPolar(
            optic=tel, wavelength=620e-9,
            theta_x=thx, theta_y=thy,
            nrad=2, naz=6
        )

        tf1 = tel.traceFull(rays)
        tf2 = lsst.traceFull(rays)

        np.testing.assert_allclose(
            tf1['PS']['in'].v,
            tf1['PS']['out'].v,
            rtol=0, atol=1e-14
        )

        for key in tf2:
            rays_allclose(
                tf1[key]['out'],
                tf2[key]['out'],
                atol=1e-13
            )


if __name__ == '__main__':
    init_gpu()
    test_prescreen()
    test_zeroscreen()