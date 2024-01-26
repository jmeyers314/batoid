import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff, rays_allclose, init_gpu


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
    np.testing.assert_allclose(rays1.r[w], rays2.r[w], atol=1e-12, rtol=0)
    np.testing.assert_allclose(rays1.v[w], rays2.v[w], atol=1e-12, rtol=0)
    np.testing.assert_allclose(rays1.t[w], rays2.t[w], atol=1e-12, rtol=0)


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
        theta_x=np.deg2rad(0.1), theta_y=0.0,
        nrandom=300, rng=np.random.default_rng(5)
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
def test_local_shift_with_rotation():
    # Load HSC for testing
    telescope = batoid.Optic.fromYaml("HSC.yaml")

    # Rotate the entire telescope
    rotFull = telescope.withLocalRotation(batoid.RotX(np.pi/2))

    # Make global and local shifts
    rotFullGlobalShift = rotFull.withGlobalShift([1, 3, 5])
    rotFullLocalShift = rotFull.withLocalShift([1, 5, -3])

    # Check these coordinate systems are the same
    assert np.allclose(
        rotFullGlobalShift.coordSys.origin,
        rotFullLocalShift.coordSys.origin,
    )
    assert np.allclose(
        rotFullGlobalShift.coordSys.rot,
        rotFullLocalShift.coordSys.rot,
    )

    # Now rotate a subitem
    rotCAM = telescope.withLocallyRotatedOptic("CAM", batoid.RotY(np.pi/2))

    # Make global and local shifts
    rotCamGlobalShift = rotCAM.withGloballyShiftedOptic("CAM", [3, 5, 1])
    rotCamLocalShift = rotCAM.withLocallyShiftedOptic("CAM", [-1, 5, 3])

    # Check these coordinate systems are the same
    assert np.allclose(
        rotCamGlobalShift["CAM"].coordSys.origin,
        rotCamLocalShift["CAM"].coordSys.origin,
    )
    assert np.allclose(
        rotCamGlobalShift["CAM"].coordSys.rot,
        rotCamLocalShift["CAM"].coordSys.rot,
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


@timer
def test_insert():
    # Insert an empty phase screen should do nothing.
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    inserted = telescope.withInsertedOptic(
        before="M1",
        item=batoid.OPDScreen(
            name='Screen',
            surface=batoid.Plane(),
            screen=batoid.Plane(),
            coordSys=telescope.stopSurface.coordSys,
        )
    )
    rays = batoid.RayVector.asPolar(
        optic=telescope,
        theta_x=0.0, theta_y=0.0,
        wavelength=622e-9,
        nrandom=1000,
    )
    trays = telescope.trace(rays.copy())
    irays = inserted.trace(rays.copy())
    rays_allclose(trays, irays, atol=1e-13)

    # If we add a constant phase, then get identical positions but different
    # phases.  Easiest way to realize is with Z1 term of Zernike.
    inserted = telescope.withInsertedOptic(
        before="M1",
        item=batoid.OPDScreen(
            name='Screen',
            surface=batoid.Plane(),
            screen=batoid.Zernike([0, 0.3]),
            coordSys=telescope.stopSurface.coordSys,
        )
    )
    irays = inserted.trace(rays.copy())
    # If we subtract 0.3 from irays.t, then should be identical
    irays._t -= 0.3
    rays_allclose(trays, irays, atol=1e-13)

    # But we do get different positions if we add higher order Zernike terms.
    inserted = telescope.withInsertedOptic(
        before="M1",
        item=batoid.OPDScreen(
            name='Screen',
            surface=batoid.Plane(),
            screen=batoid.Zernike([0]*4+[0.1]),
            coordSys=telescope.stopSurface.coordSys,
        )
    )
    irays = inserted.trace(rays.copy())
    assert not np.allclose(trays.x, irays.x)
    assert not np.allclose(trays.y, irays.y)

    # We ought to be able to add a constant phase screen anywhere in the optics
    # and get a constant phase offset result.
    inserted = telescope.withInsertedOptic(
        before="M2",
        item=batoid.OPDScreen(
            name='Screen',
            surface=batoid.Plane(),
            screen=batoid.Zernike([0, 0.3]),
            coordSys=telescope.stopSurface.coordSys,
        )
    )
    irays = inserted.trace(rays.copy())
    # If we subtract 0.3 from irays.t, then should be identical
    irays._t -= 0.3
    rays_allclose(trays, irays, atol=1e-13)

    # Above but with non-trivial coordsys.
    inserted = telescope.withInsertedOptic(
        before="M2",
        item=batoid.OPDScreen(
            name='Screen',
            surface=batoid.Plane(),
            screen=batoid.Zernike([0, 0.3]),
            coordSys=batoid.CoordSys(origin=[0.1, 0.2, 0.3], rot=batoid.RotX(0.001))
        )
    )
    irays = inserted.trace(rays.copy())
    irays._t -= 0.3
    rays_allclose(trays, irays, atol=1e-13)

    # Let's remove M3 and add it back in.
    M3 = telescope['M3']
    removed = telescope.withRemovedOptic('M3')
    reinserted = removed.withInsertedOptic(
        before="LSSTCamera",
        item=M3
    )
    with np.testing.assert_raises(ValueError):
        removed['M3']
    assert telescope == reinserted
    rrays = reinserted.trace(rays.copy())
    rays_allclose(trays, rrays, atol=1e-13)

    with np.testing.assert_raises(ValueError):
        inserted = telescope.withInsertedOptic(
            before="junk",
            item=M3
        )

    with np.testing.assert_raises(ValueError):
        removed = telescope.withRemovedOptic(
            item='junk',
        )


@timer
def test_insert_null_phase():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    thx = 0.01
    thy = 0.01
    wavelength = 622e-9
    zk0 = batoid.zernike(
        telescope,
        theta_x=thx, theta_y=thy,
        wavelength=wavelength,
        eps=0.61,
        jmax=28
    )*wavelength
    inserted = telescope.withInsertedOptic(
        before="M1",
        item=batoid.OPDScreen(
            name='Screen',
            surface=batoid.Plane(),
            screen=batoid.Zernike(zk0, R_outer=4.18, R_inner=0.61*4.18),
            coordSys=telescope.stopSurface.coordSys,
        )
    )
    zk1 = batoid.zernike(
        inserted,
        theta_x=thx, theta_y=thy,
        wavelength=wavelength,
        eps=0.61,
        jmax=28
    )*wavelength

    # I get a ton of PTT here, but the most interesting Zernikes do get nulled.
    np.testing.assert_allclose(zk1[4:]*1e9, 0.0, atol=2e-3)


@timer
def test_insert_middle():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    L1S1 = telescope['L1_entrance']
    removed = telescope.withRemovedOptic('L1_entrance')
    reinserted = removed.withInsertedOptic(
        before="L1_exit",
        item=L1S1
    )
    assert telescope == reinserted

    rays = batoid.RayVector.asPolar(
        optic=telescope,
        theta_x=0.01, theta_y=0.01,
        wavelength=622e-9,
        nrandom=1000,
    )
    trays = telescope.trace(rays.copy())
    rrays = reinserted.trace(rays.copy())
    rays_allclose(trays, rrays, atol=1e-13)


@timer
def test_optic_radii():
    import os
    import yaml

    telescope = batoid.Optic.fromYaml("LSST_r.yaml")

    fn = os.path.join(batoid.datadir, "LSST", "LSST_r.yaml")
    dct = yaml.safe_load(open(fn, 'r'))
    M1_outer = dct['opticalSystem']['items'][0]['obscuration']['outer']
    M1_inner = dct['opticalSystem']['items'][0]['obscuration']['inner']
    M2_outer = dct['opticalSystem']['items'][1]['obscuration']['outer']
    M2_inner = dct['opticalSystem']['items'][1]['obscuration']['inner']
    M3_outer = dct['opticalSystem']['items'][2]['obscuration']['outer']
    M3_inner = dct['opticalSystem']['items'][2]['obscuration']['inner']

    cam_dct = dct['opticalSystem']['items'][-1]
    L1S1_outer = cam_dct['items'][0]['items'][0]['obscuration']['radius']
    L1S1_inner = 0.0

    assert telescope['M1'].R_outer == M1_outer
    assert telescope['M1'].R_inner == M1_inner
    assert telescope['M2'].R_outer == M2_outer
    assert telescope['M2'].R_inner == M2_inner
    assert telescope['M3'].R_outer == M3_outer
    assert telescope['M3'].R_inner == M3_inner
    assert telescope['L1_entrance'].R_outer == L1S1_outer
    assert telescope['L1_entrance'].R_inner == L1S1_inner

    # Try overriding with an explicit value
    dct = yaml.safe_load(open(fn, 'r'))
    dct['opticalSystem']['items'][0]['R_outer'] = 4.2
    optic = batoid.parse.parse_optic(dct['opticalSystem'])
    assert optic['M1'].R_outer == 4.2
    # Still gets R_inner from obscuration
    assert optic['M1'].R_inner == M1_inner
    # unless we override that too
    dct = yaml.safe_load(open(fn, 'r'))
    dct['opticalSystem']['items'][0]['R_outer'] = 4.2
    dct['opticalSystem']['items'][0]['R_inner'] = 2.5
    optic = batoid.parse.parse_optic(dct['opticalSystem'])
    assert optic['M1'].R_outer == 4.2
    assert optic['M1'].R_inner == 2.5

    # If no obscuration or override present, then returns None
    dct = yaml.safe_load(open(fn, 'r'))
    del dct['opticalSystem']['items'][0]['obscuration']
    optic = batoid.parse.parse_optic(dct['opticalSystem'])
    assert optic['M1'].R_outer is None
    assert optic['M1'].R_inner is None



if __name__ == '__main__':
    init_gpu()
    test_optic()
    test_traceFull()
    test_traceReverse()
    test_withSurface()
    test_shift()
    test_rotXYZ_parsing()
    test_rotation()
    test_local_shift_with_rotation()
    test_ne()
    test_name()
    test_cbp_rotation()
    test_insert()
    test_insert_null_phase()
    test_insert_middle()
    test_optic_radii()
