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

    # Do one with full pathname
    import os
    filename = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
    with np.testing.assert_raises(FileNotFoundError):
        telescope = batoid.Optic.fromYaml(filename+".gobbledegook")
    telescope = batoid.Optic.fromYaml(filename)

    # ... and one without
    telescope = batoid.Optic.fromYaml("HSC.yaml")
    do_pickle(telescope)

    rays1 = telescope.trace(rays.copy())

    # Try tracing but skipping the baffles
    for k, v in telescope.itemDict.items():
        if isinstance(v, batoid.Baffle):
            v.skip = True
    rays2 = telescope.trace(rays.copy())
    # rays2 will have fewer rays vignetted rays
    assert np.sum(rays2.vignetted) < np.sum(rays1.vignetted)
    # every place where rays2 is vignetted, is also vignetted in rays1
    w = rays2.vignetted
    assert np.all(rays1.vignetted[w])
    # and every place rays1 is not vignetted has some coords as in rays2
    w = ~rays1.vignetted
    np.testing.assert_allclose(rays1.r[w], rays2.r[w])
    np.testing.assert_allclose(rays1.v[w], rays2.v[w])
    np.testing.assert_allclose(rays1.t[w], rays2.t[w])


@timer
def test_traceFull():
    telescope = batoid.Optic.fromYaml("HSC.yaml")
    rays = batoid.RayVector.asPolar(
        optic=telescope,
        nrad=100, naz=100,
        theta_x=0.005, theta_y=0.005,
        wavelength=650e-9
    )

    tf = telescope.traceFull(rays)
    rays = telescope.trace(rays)

    assert rays == tf['D']['out']


@timer
def test_traceReverse():
    telescope = batoid.Optic.fromYaml("HSC.yaml")

    init_rays = batoid.RayVector.asGrid(
        # backDist=20, lx=8.3, nx=128,
        backDist=25, lx=8.3, nx=6,
        theta_x=0.005, theta_y=0.005,
        wavelength=500e-9,
        medium = batoid.ConstMedium(1.0)
    )
    forward_rays = telescope.trace(init_rays.copy())

    # Now, turn the result rays around and trace backwards
    forward_rays.propagate(40.0)
    reverse_rays = forward_rays.copy()
    reverse_rays.vx[:] *= -1
    reverse_rays.vy[:] *= -1
    reverse_rays.vz[:] *= -1
    reverse_rays.t[:] *= -1

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
        theta_x=np.deg2rad(0.01), theta_y=0.0,  # TODO: why does this fail for theta_x=np.rad2deg(0.1)?
        nrad=10, naz=60
    )
    trays = telescope.trace(rays.copy())
    for key, item in telescope.itemDict.items():
        if not isinstance(item, batoid.Interface):
            continue
        # Do a trivial surface replacement
        surf2 = item.surface + batoid.Plane()
        telescope2 = telescope.withSurface(key, surf2)
        assert telescope != telescope2
        trays2 = telescope2.trace(rays.copy())
        rays_allclose(trays, trays2, atol=1e-12)
        telescope3 = telescope.withPerturbedSurface(key, batoid.Plane())
        assert telescope != telescope3
        assert telescope2 == telescope3
        trays3 = telescope3.trace(rays.copy())
        rays_allclose(trays, trays3, atol=1e-12)

        # Do a more complicated surface replacement
        telescope4 = telescope.withSurface(key, item.surface+batoid.Sphere(1000.0))
        assert telescope != telescope4
        trays4 = telescope4.trace(rays.copy())
        telescope5 = telescope.withPerturbedSurface(key, batoid.Sphere(1000.0))
        assert telescope != telescope5
        assert telescope4 == telescope5
        trays5 = telescope5.trace(rays.copy())
        rays_allclose(trays4, trays5, atol=1e-12)


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
    # Try local shifts
    assert telescope.withLocalShift(shift).withLocalShift(-shift) == telescope
    assert (
        telescope.withGloballyShiftedOptic(telescope.name, shift)
        == telescope.withGlobalShift(shift)
    )
    for item in telescope.itemDict:
        shifted = telescope.withLocallyShiftedOptic(item, shift)
        shifted = shifted.withLocallyShiftedOptic(item, -shift)
        assert telescope == shifted
    shifted = telescope.withLocallyShiftedOptic("G1", shift)
    shifted = shifted.withLocallyShiftedOptic("G1", -shift)
    assert shifted == telescope
    assert (
        telescope.withLocallyShiftedOptic(telescope.name, shift)
        == telescope.withLocalShift(shift)
    )


@timer
def test_rotXYZ_parsing():
    import os
    import yaml
    fn = os.path.join(batoid.datadir, "DESI", "DESI.yaml")
    with open(fn) as f:
        config = yaml.safe_load(f)
    # Verify that we can parse the DESI model, which has some rotated lens
    # surfaces.
    _ = batoid.parse.parse_optic(config['opticalSystem'])


@timer
def test_rotation():
    rng = np.random.default_rng(57)

    telescope = batoid.Optic.fromYaml("HSC.yaml")

    rot = batoid.RotX(rng.uniform(low=0.0, high=2*np.pi))
    rot = rot.dot(batoid.RotY(rng.uniform(low=0.0, high=2*np.pi)))
    rot = rot.dot(batoid.RotZ(rng.uniform(low=0.0, high=2*np.pi)))
    rotInv = np.linalg.inv(rot)

    # It's hard to test the two telescopes for equality due to rounding errors,
    # so we test by comparing zernikes
    rotTel = telescope.withLocalRotation(rot).withLocalRotation(rotInv)

    theta_x = rng.uniform(-0.005, 0.005)
    theta_y = rng.uniform(-0.005, 0.005)
    wavelength = 750e-9

    for k in telescope.itemDict.keys():
        np.testing.assert_allclose(
            telescope[k].coordSys.origin,
            rotTel[k].coordSys.origin,
            rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            telescope[k].coordSys.rot,
            rotTel[k].coordSys.rot,
            rtol=0, atol=1e-13
        )

    np.testing.assert_allclose(
        batoid.zernikeGQ(telescope, theta_x, theta_y, wavelength),
        batoid.zernikeGQ(rotTel, theta_x, theta_y, wavelength),
        atol=1e-7
    )

    for item in telescope.itemDict:
        rotTel = telescope.withLocallyRotatedOptic(item, rot)
        rotTel = rotTel.withLocallyRotatedOptic(item, rotInv)
        rotTel2 = telescope.withLocallyRotatedOptic(item, np.eye(3))
        theta_x = rng.uniform(-0.005, 0.005)
        theta_y = rng.uniform(-0.005, 0.005)
        np.testing.assert_allclose(
            batoid.zernikeGQ(telescope, theta_x, theta_y, wavelength),
            batoid.zernikeGQ(rotTel, theta_x, theta_y, wavelength),
            atol=1e-7
        )
        np.testing.assert_allclose(
            batoid.zernikeGQ(telescope, theta_x, theta_y, wavelength),
            batoid.zernikeGQ(rotTel2, theta_x, theta_y, wavelength),
            atol=1e-7
        )
    # Test with non-fully-qualified name
    rotTel = telescope.withLocallyRotatedOptic('G1', rot)
    rotTel = rotTel.withLocallyRotatedOptic('G1', rotInv)
    rotTel2 = rotTel.withLocallyRotatedOptic('G1', np.eye(3))
    np.testing.assert_allclose(
        batoid.zernikeGQ(telescope, theta_x, theta_y, wavelength),
        batoid.zernikeGQ(rotTel, theta_x, theta_y, wavelength),
        atol=1e-7
    )
    np.testing.assert_allclose(
        batoid.zernikeGQ(telescope, theta_x, theta_y, wavelength),
        batoid.zernikeGQ(rotTel2, theta_x, theta_y, wavelength),
        atol=1e-7
    )


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


def test_cbp_rotation():
    from scipy.spatial.distance import pdist
    # LSST Collimated Beam Projector has non-trivial rotations in it's design.
    # We test here that we can coherently rotate it.
    # The test criterion is that distances between vertices are retained, as
    # well as distances between coordinate system unit vectors.
    cbp = batoid.Optic.fromYaml("CBP.yaml")

    rotated1 = cbp.withLocalRotation(batoid.RotX(np.deg2rad(30)))
    rotated2 = cbp.withLocalRotation(batoid.RotY(np.deg2rad(30)))
    rotated3 = cbp.withLocalRotation(batoid.RotZ(np.deg2rad(30)))
    rotated4 = rotated1.withLocalRotation(batoid.RotZ(np.deg2rad(70)))
    rotated5 = rotated2.withLocalRotation(batoid.RotZ(np.deg2rad(70)))
    rotated6 = rotated3.withLocalRotation(batoid.RotX(np.deg2rad(10)))
    rotated7 = rotated4.withGlobalRotation(batoid.RotX(np.deg2rad(20)))

    def getCoordSysData(optic):
        origins = np.empty((len(optic.itemDict), 3))
        unitVectors = np.empty((len(optic.itemDict), 3, 3))
        for i, (k,v) in enumerate(optic.itemDict.items()):
            origins[i] = v.coordSys.origin
            unitVectors[i,::3] = v.coordSys.xhat
            unitVectors[i,1::3] = v.coordSys.yhat
            unitVectors[i,2::3] = v.coordSys.zhat
        return origins, unitVectors

    cbp_o, cbp_d = getCoordSysData(cbp)
    rot1_o, rot1_d = getCoordSysData(rotated1)
    rot2_o, rot2_d = getCoordSysData(rotated2)
    rot3_o, rot3_d = getCoordSysData(rotated3)
    rot4_o, rot4_d = getCoordSysData(rotated4)
    rot5_o, rot5_d = getCoordSysData(rotated5)
    rot6_o, rot6_d = getCoordSysData(rotated6)
    rot7_o, rot7_d = getCoordSysData(rotated7)

    np.testing.assert_allclose(pdist(cbp_o), pdist(rot1_o), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_o), pdist(rot2_o), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_o), pdist(rot3_o), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_o), pdist(rot4_o), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_o), pdist(rot5_o), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_o), pdist(rot6_o), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_o), pdist(rot7_o), atol=1e-13)

    np.testing.assert_allclose(pdist(cbp_d.reshape(-1, 3)), pdist(rot1_d.reshape(-1, 3)), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_d.reshape(-1, 3)), pdist(rot2_d.reshape(-1, 3)), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_d.reshape(-1, 3)), pdist(rot3_d.reshape(-1, 3)), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_d.reshape(-1, 3)), pdist(rot4_d.reshape(-1, 3)), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_d.reshape(-1, 3)), pdist(rot5_d.reshape(-1, 3)), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_d.reshape(-1, 3)), pdist(rot6_d.reshape(-1, 3)), atol=1e-13)
    np.testing.assert_allclose(pdist(cbp_d.reshape(-1, 3)), pdist(rot7_d.reshape(-1, 3)), atol=1e-13)


if __name__ == '__main__':
    init_gpu()
    test_optic()
    test_traceFull()
    test_traceReverse()
    test_withSurface()
    test_shift()
    test_rotXYZ_parsing()
    test_rotation()
    test_ne()
    test_name()
    test_cbp_rotation()
