import batoid
import numpy as np
import pytest
from test_helpers import timer, do_pickle, all_obj_diff, rays_allclose, init_gpu
import time


@timer
def test_optic():
    rays = batoid.RayVector.asGrid(
        backDist=20, lx=12, nx=128,
        theta_x=0.005, theta_y=0.005,
        wavelength=500e-9,
        medium = batoid.ConstMedium(1.0)
    )

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))

    # Do one with full pathname
    import os
    filename = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
    with np.testing.assert_raises(FileNotFoundError):
        telescope = batoid.Optic.fromYaml(filename+".gobbledegook")
    telescope = batoid.Optic.fromYaml(filename)

    # ... and one without
    telescope = batoid.Optic.fromYaml("HSC.yaml")
    do_pickle(telescope)

    telescope.trace(rays)


@timer
def test_traceFull():
    telescope = batoid.Optic.fromYaml("HSC.yaml")
    rays = batoid.RayVector.asPolar(
        optic=telescope,
        nrad=100, naz=100,
        theta_x=0.005, theta_y=0.005,
        wavelength=650e-9
    )

    print("Tracing {} rays.".format(len(rays)))

    tf = telescope.traceFull(rays)
    rays = telescope.trace(rays)

    assert rays == tf['D']['out']


@timer
def test_traceReverse():
    telescope = batoid.Optic.fromYaml("HSC.yaml")

    init_rays = batoid.RayVector.asGrid(
        backDist=20, lx=12, nx=128,
        theta_x=0.005, theta_y=0.005,
        wavelength=500e-9,
        medium = batoid.ConstMedium(1.0)
    )
    forward_rays = telescope.trace(init_rays.copy())

    # Now, turn the result rays around and trace backwards
    forward_rays.propagate(40.0)
    reverse_rays = batoid.RayVector(
        forward_rays.x,
        forward_rays.y,
        forward_rays.z,
        -forward_rays.vx,
        -forward_rays.vy,
        -forward_rays.vz,
        -forward_rays.t.copy(),
        forward_rays.wavelength.copy(),
    )

    final_rays = telescope.trace(reverse_rays.copy(), reverse=True)
    # propagate all the way to t=0
    final_rays.propagate(0.0)
    final_rays.toCoordSys(batoid.globalCoordSys)

    w = ~final_rays.vignetted
    np.testing.assert_allclose(init_rays.x[w], final_rays.x[w])
    np.testing.assert_allclose(init_rays.y[w], final_rays.y[w])
    np.testing.assert_allclose(init_rays.z[w], final_rays.z[w])
    np.testing.assert_allclose(init_rays.vx[w], -final_rays.vx[w])
    np.testing.assert_allclose(init_rays.vy[w], -final_rays.vy[w])
    np.testing.assert_allclose(init_rays.vz[w], -final_rays.vz[w])
    np.testing.assert_allclose(final_rays.t[w], 0)


@timer
def test_withSurface():
    telescope = batoid.Optic.fromYaml("HSC.yaml")
    rays = batoid.RayVector.asPolar(
        telescope,
        wavelength=620e-9,
        theta_x=np.deg2rad(0.1), theta_y=0.0,
        nrad=10, naz=60
    )
    trays = telescope.trace(rays.copy())
    for key, item in telescope.itemDict.items():
        if not isinstance(item, batoid.Interface):
            continue
        # Do a trivial surface replacement
        surf2 = batoid.Sum([batoid.Plane(), item.surface])
        telescope2 = telescope.withSurface(key, surf2)
        assert telescope != telescope2
        trays2 = telescope.trace(rays.copy())
        rays_allclose(trays, trays2)


@timer
def test_shift():
    rng = np.random.default_rng(5)

    telescope = batoid.Optic.fromYaml("HSC.yaml")

    shift = rng.uniform(low=-1, high=1, size=3)
    assert telescope.withGlobalShift(shift).withGlobalShift(-shift) == telescope
    for item in telescope.itemDict:
        shifted = telescope.withGloballyShiftedOptic(item, shift)
        shifted = shifted.withGloballyShiftedOptic(item, -shift)
        assert telescope == shifted
    # Also test a non-fully-qualified name
    shifted = telescope.withGloballyShiftedOptic("G1", shift)
    shifted = shifted.withGloballyShiftedOptic("G1", -shift)
    assert shifted == telescope


@timer
def test_rotXYZ_parsing():
    import os
    import yaml
    fn = os.path.join(batoid.datadir, "DESI", "DESI.yaml")
    config = yaml.safe_load(open(fn))
    # Verify that we can parse the DESI model, which has some rotated lens
    # surfaces.
    telescope = batoid.parse.parse_optic(config['opticalSystem'])
    # Verify that only a single rotation is allowed.
    config = yaml.safe_load(open(fn))
    coordSys = config['opticalSystem']['items'][0]['coordSys']
    coordSys['rotX'] = 1.
    coordSys['rotY'] = 1.
    with pytest.raises(ValueError) as excinfo:
        telescope = batoid.parse.parse_optic(config['opticalSystem'])


# @timer
# def test_rotation():
#     rng = np.random.default_rng(57)

#     telescope = batoid.Optic.fromYaml("HSC.yaml")

#     rot = batoid.RotX(rng.uniform(low=0.0, high=2*np.pi))
#     rot = rot.dot(batoid.RotY(rng.uniform(low=0.0, high=2*np.pi)))
#     rot = rot.dot(batoid.RotZ(rng.uniform(low=0.0, high=2*np.pi)))
#     rotInv = np.linalg.inv(rot)

#     # It's hard to test the two telescopes for equality due to rounding errors,
#     # so we test by comparing zernikes
#     rotTel = telescope.withLocalRotation(rot).withLocalRotation(rotInv)

#     theta_x = rng.uniform(-0.005, 0.005)
#     theta_y = rng.uniform(-0.005, 0.005)
#     wavelength = 750e-9

#     np.testing.assert_allclose(
#         batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
#         batoid.psf.zernike(rotTel, theta_x, theta_y, wavelength),
#         atol=1e-5
#     )

#     for item in telescope.itemDict:
#         rotTel = telescope.withLocallyRotatedOptic(item, rot)
#         rotTel = rotTel.withLocallyRotatedOptic(item, rotInv)
#         rotTel2 = telescope.withLocallyRotatedOptic(item, np.eye(3))
#         theta_x = rng.uniform(-0.005, 0.005)
#         theta_y = rng.uniform(-0.005, 0.005)
#         np.testing.assert_allclose(
#             batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
#             batoid.psf.zernike(rotTel, theta_x, theta_y, wavelength),
#             atol=1e-5
#         )
#         np.testing.assert_allclose(
#             batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
#             batoid.psf.zernike(rotTel2, theta_x, theta_y, wavelength),
#             atol=1e-5
#         )
#     # Test with non-fully-qualified name
#     rotTel = telescope.withLocallyRotatedOptic('G1', rot)
#     rotTel = rotTel.withLocallyRotatedOptic('G1', rotInv)
#     rotTel2 = rotTel.withLocallyRotatedOptic('G1', np.eye(3))
#     np.testing.assert_allclose(
#         batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
#         batoid.psf.zernike(rotTel, theta_x, theta_y, wavelength),
#         atol=1e-5
#     )
#     np.testing.assert_allclose(
#         batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
#         batoid.psf.zernike(rotTel2, theta_x, theta_y, wavelength),
#         atol=1e-5
#     )


@timer
def test_ne():
    objs = [
        batoid.Mirror(batoid.Plane()),
        batoid.Detector(batoid.Plane()),
        batoid.Baffle(batoid.Plane()),
        batoid.RefractiveInterface(batoid.Plane()),
        batoid.Mirror(batoid.Paraboloid(0.1)),
        batoid.Detector(batoid.Paraboloid(0.1)),
        batoid.Baffle(batoid.Paraboloid(0.1)),
        batoid.RefractiveInterface(batoid.Paraboloid(0.1)),
        batoid.Mirror(batoid.Plane(), obscuration=batoid.ObscCircle(0.1)),
        batoid.Mirror(batoid.Plane(), inMedium=batoid.ConstMedium(1.1)),
        batoid.Mirror(batoid.Plane(), outMedium=batoid.ConstMedium(1.1)),
        batoid.Mirror(batoid.Plane(), coordSys=batoid.CoordSys([0,0,1])),
        batoid.CompoundOptic([
            batoid.Mirror(batoid.Plane()),
            batoid.Mirror(batoid.Plane())
        ]),
        batoid.CompoundOptic(
            [batoid.Mirror(batoid.Plane()),
             batoid.Baffle(batoid.Plane())]
        ),
        batoid.CompoundOptic(
            [batoid.RefractiveInterface(batoid.Plane()),
             batoid.RefractiveInterface(batoid.Plane())]
        ),
        batoid.Lens(
            [batoid.RefractiveInterface(batoid.Plane()),
             batoid.RefractiveInterface(batoid.Plane())]
        ),
    ]
    all_obj_diff(objs)


@timer
def test_name():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    for name, surface in telescope.itemDict.items():
        shortName = name.split('.')[-1]
        assert telescope[name] == telescope[shortName]
    # What about partially qualified names?
    for name, surface in telescope.itemDict.items():
        tokens = name.split('.')
        shortName = tokens[-1]
        for token in reversed(tokens[:-1]):
            shortName = '.'.join([token, shortName])
            assert telescope[name] == telescope[shortName]
    # Now introduce a name conflict and verify we get an exception
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    telescope.items[3].items[0].name = 'L2'
    with np.testing.assert_raises(ValueError):
        telescope['L1'].name


if __name__ == '__main__':
    init_gpu()
    test_optic()
    test_traceFull()
    test_traceReverse()
    test_withSurface()
    test_shift()
    test_rotXYZ_parsing()
    # test_rotation()
    test_ne()
    test_name()
