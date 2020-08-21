import numpy as np
import batoid
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_params():
    rng = np.random.default_rng(5)
    for _ in range(30):
        origin = rng.uniform(size=3)
        rot = (
            batoid.RotX(rng.uniform())
            .dot(batoid.RotY(rng.uniform()))
            .dot(batoid.RotZ(rng.uniform()))
        )
        coordSys = batoid.CoordSys(origin, rot)
        np.testing.assert_equal(coordSys.origin, origin)
        np.testing.assert_equal(coordSys.rot, rot)
        np.testing.assert_equal(coordSys.xhat, rot[:,0])
        np.testing.assert_equal(coordSys.yhat, rot[:,1])
        np.testing.assert_equal(coordSys.zhat, rot[:,2])

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
         .dot(batoid.RotY(rng.uniform()))
         .dot(batoid.RotZ(rng.uniform())))
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
def test_combinations():
    # Test some particular combinations of coord sys transformations
    coordSys = batoid.CoordSys()
    coordSys1 = coordSys.rotateGlobal(batoid.RotX(np.pi/2)).shiftGlobal([1,0,0])
    coordSys2 = coordSys.shiftGlobal([1,0,0]).rotateGlobal(batoid.RotX(np.pi/2))
    np.testing.assert_allclose(coordSys1.origin, coordSys2.origin)
    np.testing.assert_allclose(coordSys1.rot, coordSys2.rot)

    coordSys1 = (
        coordSys
        .rotateGlobal(batoid.RotX(np.pi/4))
        .shiftLocal([0,np.sqrt(2),0])
        .rotateGlobal(batoid.RotX(-np.pi/4))
    )
    coordSys2 = coordSys.shiftLocal([0,1,1])
    np.testing.assert_allclose(coordSys1.origin, coordSys2.origin)
    np.testing.assert_allclose(coordSys1.rot, coordSys2.rot)

    coordSys1 = coordSys.rotateGlobal(batoid.RotY(np.pi/2), [1,0,0])
    coordSys2 = coordSys.rotateGlobal(batoid.RotY(np.pi/2)).shiftGlobal([1,0,1])
    coordSys3 = coordSys.rotateGlobal(batoid.RotY(np.pi/2)).shiftLocal([-1,0,1])
    np.testing.assert_allclose(coordSys1.origin, coordSys2.origin)
    np.testing.assert_allclose(coordSys1.rot, coordSys2.rot)
    np.testing.assert_allclose(coordSys1.origin, coordSys3.origin)
    np.testing.assert_allclose(coordSys1.rot, coordSys3.rot)


@timer
def test_ne():
    objs = [
        batoid.CoordSys(),
        batoid.CoordSys([0,0,1]),
        batoid.CoordSys([0,1,0]),
        batoid.CoordSys(rot=batoid.RotX(0.1)),
    ]
    all_obj_diff(objs)


if __name__ == '__main__':
    test_params()
    test_shift()
    test_rotate_identity()
    test_combinations()
    test_ne()
