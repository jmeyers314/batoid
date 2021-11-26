import batoid
import yaml
import numpy as np
from test_helpers import timer, do_pickle, init_gpu, rays_allclose


@timer
def test_prescreen():
    """Add an OPDScreen in front of LSST entrance pupil.  The OPD that comes out
    should be _negative_ the added phase delay by convention.
    """
    lsst = batoid.Optic.fromYaml("LSST_r.yaml")
    wavelength = 620e-9

    z_ref = batoid.zernikeGQ(
        lsst, 0, 0, wavelength, rings=10, reference='chief', jmax=37, eps=0.61
    )
    rng = np.random.default_rng(577)

    for i in range(4, 38):
        amplitude = rng.uniform(0.1, 0.2)
        zern = batoid.Zernike(
            np.array([0]*i+[amplitude])*wavelength,
            R_outer=4.18, R_inner=0.61*4.18
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

        zGQ = batoid.zernikeGQ(
            tel, 0, 0, wavelength, rings=10, reference='chief', jmax=37, eps=0.61
        )
        zTA = batoid.zernikeTA(
            tel, 0, 0, wavelength, nrad=10, naz=60, reference='chief', jmax=37, eps=0.61
        )

        z_expect = np.zeros_like(zGQ)
        z_expect[i] = -amplitude  # Longer OPL => negative OPD
        np.testing.assert_allclose(
            (zGQ-z_ref)[4:], z_expect[4:],
            rtol=0, atol=5e-4
        )
        # Distortion makes this comparison less precise
        np.testing.assert_allclose(
            zGQ[4:], zTA[4:],
            rtol=0, atol=5e-3
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
            nrad=5, naz=60
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


@timer
def test_z4_focus():
    """Test thin lens approximation
    """
    from scipy.optimize import minimize_scalar

    R = 0.5  # symmetric biconvex surface curvature radii
    d = 0.01  # front/back separation
    n0 = 1.0003
    n1 = 1.46
    # Lens-maker equation for focal length:
    f_inv = (n1-n0)*(2/R + (n1-n0)*d/R**2)
    f = 1/f_inv

    # With R = 0.5, sag is approximately -r^2 / (2 R)
    # So with 2 surfaces, total lens thickness is ~ -r^2 / R
    # With the refractive index difference, total delay is ~ -(n1-n0) r^2 / R
    # Z4 is sqrt(3) (2 r^2 - 1)
    # Ignoring the constant part, we can solve for the Z4 coefficient:
    #   a4 sqrt(3) 2 r^2 = -(n1-n0) r^2 / R
    #   a4 = -(n1-n0) / (2 sqrt(3) R)
    a4 = -(n1-n0) / (2 * np.sqrt(3) * R)

    biconvex_str = f"""
    type: CompoundOptic
    inMedium: {n0}
    backDist: 1.0
    stopSurface:
        type: Interface
        surface:
            type: Plane
        coordSys:
            z: 0.0
    pupilSize: 0.1
    pupilObscuration: 0.0
    items:
        -
            type: RefractiveInterface
            surface:
                type: Sphere
                R: {-R}
            coordSys:
                z: {+d/2}
            inMedium: {n0}
            outMedium: {n1}
            name: 'L1'
        -
            type: RefractiveInterface
            surface:
                type: Sphere
                R: {R}
            coordSys:
                z: {-d/2}
            inMedium: {n1}
            outMedium: {n0}
            name: 'L2'
        -
            type: Detector
            surface:
                type: Plane
            coordSys:
                z: {-f}
            inMedium: {n0}
            name: D
    """
    biconvex = batoid.parse.parse_optic(
        yaml.safe_load(biconvex_str)
    )

    screen_str = f"""
    type: CompoundOptic
    inMedium: {n0}
    backDist: 1.0
    stopSurface:
      type: Interface
      surface:
        type: Plane
      coordSys:
        z: 0.0
    pupilSize: 0.1
    pupilObscuration: 0.0
    items:
      -
        type: OPDScreen
        surface:
          type: Plane
        screen:
          type: Zernike
          coef: [0.0, 0.0, 0.0, 0.0, {a4}]
        inMedium: {n0}
        outMedium: {n0}
        name: screen
      -
        type: Detector
        surface:
          type: Plane
        coordSys:
          z: {-f}
        inMedium: {n0}
        name: D
    """
    screen = batoid.parse.parse_optic(
        yaml.safe_load(screen_str)
    )

    # Merit function to compute RMS spot size under given surface displacement
    def meritSpot(shift, telescope, surface, wavelength, axis=2):
        rays = batoid.RayVector.asPolar(
            optic=telescope,
            wavelength=wavelength,
            theta_x=0, theta_y=0,
            nrad=10, naz=60
        )
        displacement = np.zeros(3)
        displacement[axis] = shift
        perturbed = telescope.withGloballyShiftedOptic(surface, displacement)
        perturbed.trace(rays)
        w = ~rays.vignetted
        return np.sqrt(np.var(rays.x[w]) + np.var(rays.y[w]))  # meters

    x_biconvex = minimize_scalar(
            meritSpot,
            (-0.01, 0.0, 0.01),
            args=(biconvex, "D", 500e-9)
        )

    x_screen = minimize_scalar(
        meritSpot,
        (-0.01, 0.0, 0.01),
        args=(screen, "D", 500e-9)
    )

    np.testing.assert_allclose(x_biconvex.x, 0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(x_screen.x, 0, rtol=0, atol=1e-3)


if __name__ == '__main__':
    init_gpu()
    test_prescreen()
    test_zeroscreen()
    test_z4_focus()
