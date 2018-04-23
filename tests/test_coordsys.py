import numpy as np
import batoid
from test_helpers import isclose, timer, do_pickle, all_obj_diff, vec3_isclose, rot3_isclose


def randomVec3():
    import random
    return np.array([
        random.uniform(0, 1),
        random.uniform(0, 1),
        random.uniform(0, 1)
    ])


def randomCoordSys():
    import random
    return batoid.CoordSys(
        randomVec3(),
        (batoid.RotX(random.uniform(0, 1))
         .dot(batoid.RotY(random.uniform(0, 1)))
         .dot(batoid.RotZ(random.uniform(0, 1))))
    )


@timer
def test_shift():
    import random
    random.seed(5)
    globalCoordSys = batoid.CoordSys()
    for i in range(30):
        x = random.gauss(0.1, 2.3)
        y = random.gauss(0.1, 2.3)
        z = random.gauss(0.1, 2.3)
        newCoordSys = globalCoordSys.shiftGlobal([x, y, z])
        do_pickle(newCoordSys)
        np.testing.assert_array_equal(newCoordSys.xhat, [1,0,0])
        np.testing.assert_array_equal(newCoordSys.yhat, [0,1,0])
        np.testing.assert_array_equal(newCoordSys.zhat, [0,0,1])
        np.testing.assert_array_equal(newCoordSys.origin, [x,y,z])
        np.testing.assert_array_equal(newCoordSys.rot, np.eye(3))

        coordTransform = batoid.CoordTransform(globalCoordSys, newCoordSys)
        do_pickle(coordTransform)

        for j in range(30):
            x2 = random.gauss(0.1, 2.3)
            y2 = random.gauss(0.1, 2.3)
            z2 = random.gauss(0.1, 2.3)
            newNewCoordSys = newCoordSys.shiftGlobal([x2, y2, z2])
            np.testing.assert_array_equal(newNewCoordSys.xhat, [1,0,0])
            np.testing.assert_array_equal(newNewCoordSys.yhat, [0,1,0])
            np.testing.assert_array_equal(newNewCoordSys.zhat, [0,0,1])
            np.testing.assert_array_equal(newNewCoordSys.origin, [x+x2, y+y2, z+z2])
            np.testing.assert_array_equal(newNewCoordSys.rot, np.eye(3))

            newNewCoordSys = newCoordSys.shiftLocal([x2, y2, z2])
            np.testing.assert_array_equal(newNewCoordSys.xhat, [1,0,0])
            np.testing.assert_array_equal(newNewCoordSys.yhat, [0,1,0])
            np.testing.assert_array_equal(newNewCoordSys.zhat, [0,0,1])
            np.testing.assert_array_equal(newNewCoordSys.origin, [x+x2, y+y2, z+z2])
            np.testing.assert_array_equal(newNewCoordSys.rot, np.eye(3))


@timer
def test_rotate_identity():
    import random
    random.seed(57)
    coordSys = randomCoordSys()
    otherCoordSys = randomCoordSys()
    rotOrigin = randomVec3()

    newCoordSys = coordSys.rotateGlobal(np.eye(3))
    assert coordSys == newCoordSys

    newCoordSys = coordSys.rotateLocal(np.eye(3))
    assert coordSys == newCoordSys

    newCoordSys = coordSys.rotateGlobal(np.eye(3), rotOrigin, otherCoordSys)
    np.testing.assert_allclose(coordSys.origin, newCoordSys.origin, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(coordSys.rot, newCoordSys.rot)

    newCoordSys = coordSys.rotateLocal(np.eye(3), rotOrigin, otherCoordSys)
    np.testing.assert_allclose(coordSys.origin, newCoordSys.origin, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(coordSys.rot, newCoordSys.rot)


@timer
def test_ne():
    objs = [
        batoid.CoordSys(),
        batoid.CoordSys([0,0,1]),
        batoid.CoordSys([0,1,0]),
        batoid.CoordSys(batoid.RotX(0.1)),
        batoid.CoordTransform(batoid.CoordSys(), batoid.CoordSys()),
        batoid.CoordTransform(batoid.CoordSys(), batoid.CoordSys([0,0,1])),
        batoid.CoordTransform(batoid.CoordSys(), batoid.CoordSys(batoid.RotX(0.1)))
    ]
    all_obj_diff(objs)


if __name__ == '__main__':
    test_shift()
    test_rotate_identity()
    test_ne()
