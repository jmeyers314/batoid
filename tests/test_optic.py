import batoid
import numpy as np
import pytest
from test_helpers import timer, do_pickle, all_obj_diff
import time


@timer
def test_optic():
    if __name__ == '__main__':
        nside = 128
    else:
        nside = 32

    rays = batoid.rayGrid(20, 12.0, 0.005, 0.005, -1.0, nside, 500e-9, 1.0, batoid.ConstMedium(1.0))

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))
    t_fast = 0.0
    t_slow = 0.0

    telescope = batoid.Optic.fromYaml("HSC.yaml")
    do_pickle(telescope)

    t0 = time.time()

    rays_fast = telescope.trace(rays.copy())
    t1 = time.time()
    rays_slow = batoid.RayVector([telescope.trace(r.copy()) for r in rays])
    t2 = time.time()

    assert rays_fast == rays_slow
    t_fast = t1 - t0
    t_slow = t2 - t1
    print("Fast trace: {:5.3f} s".format(t_fast))
    print("            {} rays per second".format(int(nrays/t_fast)))
    print("Slow trace: {:5.3f} s".format(t_slow))
    print("            {} rays per second".format(int(nrays/t_slow)))


@timer
def test_traceFull():
    if __name__ == '__main__':
        nside = 128
    else:
        nside = 32

    rays = batoid.rayGrid(20, 12.0, 0.005, 0.005, -1.0, nside, 500e-9, 1.0, batoid.ConstMedium(1.0))

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))

    telescope = batoid.Optic.fromYaml("HSC.yaml")

    tf = telescope.traceFull(rays)
    rays = telescope.trace(rays)

    assert rays == tf['D']['out']


@timer
def test_traceReverse():
    if __name__ == '__main__':
        nside = 128
    else:
        nside = 32

    telescope = batoid.Optic.fromYaml("HSC.yaml")

    init_rays = batoid.rayGrid(20, 12.0, 0.005, 0.005, -1.0, nside, 500e-9, 1.0, batoid.ConstMedium(1.0))
    forward_rays = telescope.trace(init_rays.copy())

    # Now, turn the result rays around and trace backwards
    forward_rays.propagate(40.0)
    reverse_rays = batoid.RayVector(
        [batoid.Ray(r.r, -r.v, -r.t, r.wavelength) for r in forward_rays]
    )

    final_rays = telescope.trace(reverse_rays.copy(), reverse=True)
    # propagate all the way to t=0
    final_rays = final_rays.propagate(0.0)
    final_rays.toCoordSys(batoid.globalCoordSys)

    w = np.where(np.logical_not(final_rays.vignetted))[0]
    for idx in w:
        np.testing.assert_allclose(init_rays[idx].x, final_rays[idx].x)
        np.testing.assert_allclose(init_rays[idx].y, final_rays[idx].y)
        np.testing.assert_allclose(init_rays[idx].z, final_rays[idx].z)
        np.testing.assert_allclose(init_rays[idx].vx, -final_rays[idx].vx)
        np.testing.assert_allclose(init_rays[idx].vy, -final_rays[idx].vy)
        np.testing.assert_allclose(init_rays[idx].vz, -final_rays[idx].vz)
        np.testing.assert_allclose(final_rays[idx].t, 0)


@timer
def test_shift():
    np.random.seed(5)

    telescope = batoid.Optic.fromYaml("HSC.yaml")

    shift = np.random.uniform(low=-1, high=1, size=3)
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
    np.random.seed(5)
    fn = os.path.join(batoid.datadir, "DESI", "DESI.yaml")
    config = yaml.safe_load(open(fn))
    # Verify that we can parse the DESI model, which has some rotated lens surfaces.
    telescope = batoid.parse.parse_optic(config['opticalSystem'])
    # Verify that only a single rotation is allowed.
    config = yaml.safe_load(open(fn))
    coordSys = config['opticalSystem']['items'][0]['coordSys']
    coordSys['rotX'] = 1.
    coordSys['rotY'] = 1.
    with pytest.raises(ValueError) as excinfo:
        telescope = batoid.parse.parse_optic(config['opticalSystem'])


@timer
def test_rotation():
    try:
        import galsim
    except ImportError:
        print("optic rotation test requires GalSim")
        return

    np.random.seed(57)

    telescope = batoid.Optic.fromYaml("HSC.yaml")

    rot = batoid.RotX(np.random.uniform(low=0.0, high=2*np.pi))
    rot = rot.dot(batoid.RotY(np.random.uniform(low=0.0, high=2*np.pi)))
    rot = rot.dot(batoid.RotZ(np.random.uniform(low=0.0, high=2*np.pi)))
    rotInv = np.linalg.inv(rot)

    # It's hard to test the two telescopes for equality due to rounding errors, so we test by
    # comparing zernikes
    rotTel = telescope.withLocalRotation(rot).withLocalRotation(rotInv)

    theta_x = np.random.uniform(-0.005, 0.005)
    theta_y = np.random.uniform(-0.005, 0.005)
    wavelength = 750e-9

    np.testing.assert_allclose(
        batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
        batoid.psf.zernike(rotTel, theta_x, theta_y, wavelength),
        atol=1e-5
    )

    for item in telescope.itemDict:
        rotTel = telescope.withLocallyRotatedOptic(item, rot)
        rotTel = rotTel.withLocallyRotatedOptic(item, rotInv)
        rotTel2 = telescope.withLocallyRotatedOptic(item, np.eye(3))
        theta_x = np.random.uniform(-0.005, 0.005)
        theta_y = np.random.uniform(-0.005, 0.005)
        np.testing.assert_allclose(
            batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
            batoid.psf.zernike(rotTel, theta_x, theta_y, wavelength),
            atol=1e-5
        )
        np.testing.assert_allclose(
            batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
            batoid.psf.zernike(rotTel2, theta_x, theta_y, wavelength),
            atol=1e-5
        )
    # Test with non-fully-qualified name
    rotTel = telescope.withLocallyRotatedOptic('G1', rot)
    rotTel = rotTel.withLocallyRotatedOptic('G1', rotInv)
    rotTel2 = rotTel.withLocallyRotatedOptic('G1', np.eye(3))
    np.testing.assert_allclose(
        batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
        batoid.psf.zernike(rotTel, theta_x, theta_y, wavelength),
        atol=1e-5
    )
    np.testing.assert_allclose(
        batoid.psf.zernike(telescope, theta_x, theta_y, wavelength),
        batoid.psf.zernike(rotTel2, theta_x, theta_y, wavelength),
        atol=1e-5
    )

@timer
def test_thread():
    telescope = batoid.Optic.fromYaml("HSC.yaml")

    rayGrid = batoid.rayGrid(
        telescope.backDist, telescope.pupilSize,
        0.0, 0.0, -1.0,
        32, 750e-9, 1.0, telescope.inMedium)

    batoid._batoid.setNThread(4)
    assert batoid._batoid.getNThread() == 4

    rays4 = telescope.trace(rayGrid)

    batoid._batoid.setNThread(2)
    assert batoid._batoid.getNThread() == 2

    rays2 = telescope.trace(rayGrid)

    batoid._batoid.setNThread(1)
    assert batoid._batoid.getNThread() == 1

    rays1 = telescope.trace(rayGrid)

    assert rays1 == rays2 == rays4


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
    test_optic()
    test_traceFull()
    test_traceReverse()
    test_shift()
    test_rotXYZ_parsing()
    test_rotation()
    test_thread()
    test_ne()
    test_name()
