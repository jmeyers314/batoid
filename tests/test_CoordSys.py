import numpy as np
import batoid
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_rot_matrix():
    xhat = np.array([1, 0, 0])
    yhat = np.array([0, 1, 0])
    zhat = np.array([0, 0, 1])

    # small +ve around X means
    #   xhat stays the same,
    #   yhat tilts up towards +zhat
    #   zhat tilts down towards -yhat
    rot = batoid.RotX(0.1)
    np.testing.assert_equal(
        rot@xhat,
        xhat
    )
    np.testing.assert_equal(
        rot@yhat,
        np.array([0, np.cos(0.1), np.sin(0.1)])
    )
    np.testing.assert_equal(
        rot@zhat,
        np.array([0, -np.sin(0.1), np.cos(0.1)])
    )

    # similar small rot around Y
    rot = batoid.RotY(0.1)
    np.testing.assert_equal(
        rot@xhat,
        np.array([np.cos(0.1), 0, -np.sin(0.1)])
    )
    np.testing.assert_equal(
        rot@yhat,
        yhat
    )
    np.testing.assert_equal(
        rot@zhat,
        np.array([np.sin(0.1), 0, np.cos(0.1)])
    )

    # and Z
    rot = batoid.RotZ(0.1)
    np.testing.assert_equal(
        rot@xhat,
        np.array([np.cos(0.1), np.sin(0.1), 0])
    )
    np.testing.assert_equal(
        rot@yhat,
        np.array([-np.sin(0.1), np.cos(0.1), 0])
    )
    np.testing.assert_equal(
        rot@zhat,
        zhat
    )



@timer
def test_params():
    rng = np.random.default_rng(5)
    for _ in range(30):
        origin = rng.uniform(size=3)
        rot = (
            batoid.RotX(rng.uniform())
            @batoid.RotY(rng.uniform())
            @batoid.RotZ(rng.uniform())
        )
        coordSys = batoid.CoordSys(origin, rot)
        np.testing.assert_equal(coordSys.origin, origin)
        np.testing.assert_equal(coordSys.rot, rot)
        np.testing.assert_equal(coordSys.xhat, rot[:,0])
        np.testing.assert_equal(coordSys.yhat, rot[:,1])
        np.testing.assert_equal(coordSys.zhat, rot[:,2])
        np.testing.assert_equal(rot@np.array([1,0,0]), coordSys.xhat)
        np.testing.assert_equal(rot@np.array([0,1,0]), coordSys.yhat)
        np.testing.assert_equal(rot@np.array([0,0,1]), coordSys.zhat)

        coordSys = batoid.CoordSys(origin=origin)
        np.testing.assert_equal(coordSys.origin, origin)
        np.testing.assert_equal(coordSys.rot, np.eye(3))
        np.testing.assert_equal(coordSys.xhat, [1,0,0])
        np.testing.assert_equal(coordSys.yhat, [0,1,0])
        np.testing.assert_equal(coordSys.zhat, [0,0,1])

        coordSys = batoid.CoordSys(rot=rot)
        np.testing.assert_equal(coordSys.origin, np.zeros(3))
        np.testing.assert_equal(coordSys.rot, rot)
        np.testing.assert_equal(coordSys.xhat, rot[:,0])
        np.testing.assert_equal(coordSys.yhat, rot[:,1])
        np.testing.assert_equal(coordSys.zhat, rot[:,2])

        coordSys = batoid.CoordSys()
        np.testing.assert_equal(coordSys.origin, np.zeros(3))
        np.testing.assert_equal(coordSys.rot, np.eye(3))
        np.testing.assert_equal(coordSys.xhat, [1,0,0])
        np.testing.assert_equal(coordSys.yhat, [0,1,0])
        np.testing.assert_equal(coordSys.zhat, [0,0,1])

        coordSys = batoid.CoordSys()
        coordSys = coordSys.rotateGlobal(rot).shiftGlobal(origin)
        np.testing.assert_equal(coordSys.origin, origin)
        np.testing.assert_equal(coordSys.rot, rot)
        np.testing.assert_equal(coordSys.xhat, rot[:,0])
        np.testing.assert_equal(coordSys.yhat, rot[:,1])
        np.testing.assert_equal(coordSys.zhat, rot[:,2])
        np.testing.assert_equal(rot@np.array([1,0,0]), coordSys.xhat)
        np.testing.assert_equal(rot@np.array([0,1,0]), coordSys.yhat)
        np.testing.assert_equal(rot@np.array([0,0,1]), coordSys.zhat)

        coordSys = batoid.CoordSys()
        coordSys = coordSys.rotateLocal(rot).shiftGlobal(origin)
        np.testing.assert_equal(coordSys.origin, origin)
        np.testing.assert_equal(coordSys.rot, rot)
        np.testing.assert_equal(coordSys.xhat, rot[:,0])
        np.testing.assert_equal(coordSys.yhat, rot[:,1])
        np.testing.assert_equal(coordSys.zhat, rot[:,2])
        np.testing.assert_equal(rot@np.array([1,0,0]), coordSys.xhat)
        np.testing.assert_equal(rot@np.array([0,1,0]), coordSys.yhat)
        np.testing.assert_equal(rot@np.array([0,0,1]), coordSys.zhat)

        coordSys = batoid.CoordSys()
        coordSys = coordSys.shiftLocal(origin).rotateLocal(rot)
        np.testing.assert_equal(coordSys.origin, origin)
        np.testing.assert_equal(coordSys.rot, rot)
        np.testing.assert_equal(coordSys.xhat, rot[:,0])
        np.testing.assert_equal(coordSys.yhat, rot[:,1])
        np.testing.assert_equal(coordSys.zhat, rot[:,2])
        np.testing.assert_equal(rot@np.array([1,0,0]), coordSys.xhat)
        np.testing.assert_equal(rot@np.array([0,1,0]), coordSys.yhat)
        np.testing.assert_equal(rot@np.array([0,0,1]), coordSys.zhat)

        coordSys = batoid.CoordSys()
        coordSys = coordSys.shiftGlobal(origin).rotateLocal(rot)
        np.testing.assert_equal(coordSys.origin, origin)
        np.testing.assert_equal(coordSys.rot, rot)
        np.testing.assert_equal(coordSys.xhat, rot[:,0])
        np.testing.assert_equal(coordSys.yhat, rot[:,1])
        np.testing.assert_equal(coordSys.zhat, rot[:,2])
        np.testing.assert_equal(rot@np.array([1,0,0]), coordSys.xhat)
        np.testing.assert_equal(rot@np.array([0,1,0]), coordSys.yhat)
        np.testing.assert_equal(rot@np.array([0,0,1]), coordSys.zhat)

        # Can't simply do a global rotation after a shift, since that will
        # change the origin too.  Works if we manually specify the rotation
        # center though
        coordSys = batoid.CoordSys()
        coordSys1 = coordSys.shiftGlobal(origin)
        coordSys = coordSys1.rotateGlobal(rot, origin, coordSys)
        np.testing.assert_equal(coordSys.origin, origin)
        np.testing.assert_equal(coordSys.rot, rot)
        np.testing.assert_equal(coordSys.xhat, rot[:,0])
        np.testing.assert_equal(coordSys.yhat, rot[:,1])
        np.testing.assert_equal(coordSys.zhat, rot[:,2])
        np.testing.assert_equal(rot@np.array([1,0,0]), coordSys.xhat)
        np.testing.assert_equal(rot@np.array([0,1,0]), coordSys.yhat)
        np.testing.assert_equal(rot@np.array([0,0,1]), coordSys.zhat)


@timer
def test_shift():
    rng = np.random.default_rng(57)
    globalCoordSys = batoid.CoordSys()
    for i in range(30):
        x, y, z = rng.normal(0.1, 2.3, size=3)
        newCoordSys = globalCoordSys.shiftGlobal([x, y, z])
        do_pickle(newCoordSys)
        np.testing.assert_array_equal(newCoordSys.xhat, [1,0,0])
        np.testing.assert_array_equal(newCoordSys.yhat, [0,1,0])
        np.testing.assert_array_equal(newCoordSys.zhat, [0,0,1])
        np.testing.assert_array_equal(newCoordSys.origin, [x,y,z])
        np.testing.assert_array_equal(newCoordSys.rot, np.eye(3))

        for j in range(30):
            x2, y2, z2 = rng.normal(0.1, 2.3, size=3)
            newNewCoordSys = newCoordSys.shiftGlobal([x2, y2, z2])
            np.testing.assert_array_equal(newNewCoordSys.xhat, [1,0,0])
            np.testing.assert_array_equal(newNewCoordSys.yhat, [0,1,0])
            np.testing.assert_array_equal(newNewCoordSys.zhat, [0,0,1])
            np.testing.assert_array_equal(
                newNewCoordSys.origin,
                [x+x2, y+y2, z+z2]
            )
            np.testing.assert_array_equal(newNewCoordSys.rot, np.eye(3))

            newNewCoordSys = newCoordSys.shiftLocal([x2, y2, z2])
            np.testing.assert_array_equal(newNewCoordSys.xhat, [1,0,0])
            np.testing.assert_array_equal(newNewCoordSys.yhat, [0,1,0])
            np.testing.assert_array_equal(newNewCoordSys.zhat, [0,0,1])
            np.testing.assert_array_equal(
                newNewCoordSys.origin,
                [x+x2, y+y2, z+z2]
            )
            np.testing.assert_array_equal(newNewCoordSys.rot, np.eye(3))


def randomCoordSys(rng):
    return batoid.CoordSys(
        rng.uniform(size=3),
        (batoid.RotX(rng.uniform())
         @batoid.RotY(rng.uniform())
         @batoid.RotZ(rng.uniform()))
    )


@timer
def test_rotate_identity():
    rng = np.random.default_rng(577)
    coordSys = randomCoordSys(rng)
    otherCoordSys = randomCoordSys(rng)
    rotOrigin = rng.uniform(size=3)

    newCoordSys = coordSys.rotateGlobal(np.eye(3))
    assert coordSys == newCoordSys

    newCoordSys = coordSys.rotateLocal(np.eye(3))
    assert coordSys == newCoordSys

    newCoordSys = coordSys.rotateGlobal(np.eye(3), rotOrigin, otherCoordSys)
    np.testing.assert_allclose(
        coordSys.origin, newCoordSys.origin, rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(coordSys.rot, newCoordSys.rot)

    newCoordSys = coordSys.rotateLocal(np.eye(3), rotOrigin, otherCoordSys)
    np.testing.assert_allclose(
        coordSys.origin, newCoordSys.origin, rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(coordSys.rot, newCoordSys.rot)


@timer
def test_rotate():
    rng = np.random.default_rng(57)
    for _ in range(10):
        r1 = batoid.RotX(rng.uniform())
        r2 = batoid.RotY(rng.uniform())
        r3 = batoid.RotZ(rng.uniform())
        r4 = batoid.RotX(rng.uniform())
        r5 = batoid.RotY(rng.uniform())
        r6 = batoid.RotZ(rng.uniform())

        rot = r6@r5@r4@r3@r2@r1
        coordSys = batoid.CoordSys().rotateGlobal(rot)
        np.testing.assert_equal(coordSys.xhat, rot[:, 0])
        np.testing.assert_equal(coordSys.yhat, rot[:, 1])
        np.testing.assert_equal(coordSys.zhat, rot[:, 2])
        np.testing.assert_equal(coordSys.origin, 0)

        rot1 = r3@r2@r1
        rot2 = r6@r5@r4

        coordSys = batoid.CoordSys().rotateGlobal(rot1).rotateGlobal(rot2)
        np.testing.assert_allclose(coordSys.xhat, rot[:, 0])
        np.testing.assert_allclose(coordSys.yhat, rot[:, 1])
        np.testing.assert_allclose(coordSys.zhat, rot[:, 2])
        np.testing.assert_equal(coordSys.origin, 0)

        coordSys = batoid.CoordSys(rot=rot1)
        coordSys2 = coordSys.rotateLocal(batoid.RotX(np.pi/2))
        # Since second rotation was about the local X, both should have same
        # xhat
        np.testing.assert_allclose(coordSys.xhat, coordSys2.xhat)
        # 90 degree positive rotation then means y -> -z, z -> y
        np.testing.assert_allclose(coordSys.yhat, -coordSys2.zhat)
        np.testing.assert_allclose(coordSys.zhat, coordSys2.yhat)

        # Try a loop
        coordSys = batoid.CoordSys(rot=rot)
        coordSys = coordSys.rotateGlobal(batoid.RotX(0.1))
        coordSys = coordSys.rotateGlobal(batoid.RotZ(np.pi))
        coordSys = coordSys.rotateGlobal(batoid.RotX(0.1))
        coordSys = coordSys.rotateGlobal(batoid.RotZ(np.pi))
        # Should be back where we started...
        np.testing.assert_allclose(coordSys.rot, rot)

        # Miscentered origins
        origin = rng.uniform(size=3)
        coordSys = batoid.CoordSys(origin=origin, rot=rot)
        np.testing.assert_equal(origin, coordSys.origin)
        coordSys2 = coordSys.rotateGlobal(rot)
        np.testing.assert_equal(rot@origin, coordSys2.origin)
        coordSys3 = coordSys.rotateLocal(rot)
        np.testing.assert_equal(origin, coordSys3.origin)

        # Miscentered rotation axes
        # Global with center specified is same as local
        coordSys = batoid.CoordSys(origin=origin, rot=rot)
        coordSys2 = coordSys.rotateLocal(rot)
        coordSys3 = coordSys.rotateGlobal(rot, origin, batoid.CoordSys())
        np.testing.assert_allclose(coordSys2.origin, origin)
        np.testing.assert_allclose(coordSys2.origin, coordSys3.origin)
        np.testing.assert_allclose(coordSys2.rot, coordSys3.rot)


@timer
def test_combinations():
    # Test some particular combinations of coord sys transformations

    # +90 around x, followed by shift to (1, 0, 0)
    # equivalent to shift first, and then rotation +90 about x
    # latter rotation can be either global or local, since origin is unaffected
    # (1, 0, 0) -> (1, 0, 0)
    # by convention for local rotation
    # and by coincidence (origin is on xaxis) for global rotation
    coordSys = batoid.CoordSys()
    coordSys1 = coordSys.rotateGlobal(batoid.RotX(np.pi/2)).shiftGlobal([1,0,0])
    coordSys2 = coordSys.shiftGlobal([1,0,0]).rotateLocal(batoid.RotX(np.pi/2))
    coordSys3 = coordSys.shiftGlobal([1,0,0]).rotateGlobal(batoid.RotX(np.pi/2))
    np.testing.assert_allclose(coordSys1.origin, coordSys2.origin)
    np.testing.assert_allclose(coordSys1.rot, coordSys2.rot)
    np.testing.assert_allclose(coordSys1.origin, coordSys3.origin)
    np.testing.assert_allclose(coordSys1.rot, coordSys3.rot)

    # +45 around x, followed by sqrt(2) in new y direction, which is the (0, 1, 1)
    # direction in global coords, followed by -45 about x.
    # Should be parallel to global coords, but origin at (0, 1, 1)
    coordSys1 = (
        batoid.CoordSys()
        .rotateGlobal(batoid.RotX(np.pi/4))
        .shiftLocal([0,np.sqrt(2),0])
        .rotateLocal(batoid.RotX(-np.pi/4))
    )
    coordSys2 = batoid.CoordSys().shiftLocal([0,1,1])
    np.testing.assert_allclose(
        coordSys1.origin, coordSys2.origin,
        rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        coordSys1.rot, coordSys2.rot,
        rtol=0, atol=1e-15
    )

    # rotate +90 around point (1, 0, 0) with rot axis parallel to y axis.
    # moves origin from (0, 0, 0) to (1, 0, 1)
    coordSys = batoid.CoordSys()
    coordSys1 = coordSys.rotateGlobal(batoid.RotY(np.pi/2), [1,0,0])
    coordSys2 = coordSys.rotateGlobal(batoid.RotY(np.pi/2)).shiftGlobal([1,0,1])
    # local coords of global (1, 0, 1) are (-1, 0, 1)
    coordSys3 = coordSys.rotateGlobal(batoid.RotY(np.pi/2)).shiftLocal([-1,0,1])
    np.testing.assert_allclose(coordSys1.origin, coordSys2.origin)
    np.testing.assert_allclose(coordSys1.rot, coordSys2.rot)
    np.testing.assert_allclose(coordSys1.origin, coordSys3.origin)
    np.testing.assert_allclose(coordSys1.rot, coordSys3.rot)

    # Rotating LSST and LSSTCamera should commute
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    telescope1 = (telescope
        .withLocalRotation(batoid.RotX(0.2))
        .withLocallyRotatedOptic("LSSTCamera", batoid.RotZ(-0.3))
    )
    telescope2 = (telescope
        .withLocallyRotatedOptic("LSSTCamera", batoid.RotZ(-0.3))
        .withLocalRotation(batoid.RotX(0.2))
    )
    for optic in telescope.itemDict.keys():
        cs1 = telescope1[optic].coordSys
        cs2 = telescope2[optic].coordSys
        np.testing.assert_allclose(cs1.origin, cs2.origin, rtol=0, atol=1e-14)
        np.testing.assert_allclose(cs1.rot, cs2.rot, rtol=0, atol=1e-14)


@timer
def test_ne():
    objs = [
        batoid.CoordTransform(batoid.CoordSys(), batoid.CoordSys()),
        batoid.CoordSys(),
        batoid.CoordSys([0,0,1]),
        batoid.CoordSys([0,1,0]),
        batoid.CoordSys([0,0,1], batoid.RotX(0.1)),
        batoid.CoordSys(rot=batoid.RotX(0.1)),
        batoid.CoordSys(rot=batoid.RotX(0.2)),
    ]
    all_obj_diff(objs)


if __name__ == '__main__':
    test_rot_matrix()
    test_params()
    test_shift()
    test_rotate_identity()
    test_rotate()
    test_combinations()
    test_ne()
