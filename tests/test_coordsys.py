import batoid
from test_helpers import isclose, timer, do_pickle, all_obj_diff


@timer
def test_shift():
    import random
    random.seed(5)
    globalCoordSys = batoid.CoordSys()
    for i in range(30):
        x = random.gauss(0.1, 2.3)
        y = random.gauss(0.1, 2.3)
        z = random.gauss(0.1, 2.3)
        newCoordSys = globalCoordSys.shiftGlobal(batoid.Vec3(x, y, z))
        do_pickle(newCoordSys)
        assert newCoordSys.xhat == batoid.Vec3(1,0,0)
        assert newCoordSys.yhat == batoid.Vec3(0,1,0)
        assert newCoordSys.zhat == batoid.Vec3(0,0,1)
        assert newCoordSys.origin == batoid.Vec3(x,y,z)
        assert newCoordSys.rotation == batoid.Rot3()

        coordTransform = batoid.CoordTransform(globalCoordSys, newCoordSys)
        do_pickle(coordTransform)

        for j in range(30):
            x2 = random.gauss(0.1, 2.3)
            y2 = random.gauss(0.1, 2.3)
            z2 = random.gauss(0.1, 2.3)
            newNewCoordSys = newCoordSys.shiftGlobal(batoid.Vec3(x2, y2, z2))
            assert newNewCoordSys.xhat == batoid.Vec3(1,0,0)
            assert newNewCoordSys.yhat == batoid.Vec3(0,1,0)
            assert newNewCoordSys.zhat == batoid.Vec3(0,0,1)
            assert newNewCoordSys.origin == batoid.Vec3(x+x2, y+y2, z+z2)
            assert newNewCoordSys.rotation == batoid.Rot3()

            newNewCoordSys = newCoordSys.shiftLocal(batoid.Vec3(x2, y2, z2))
            assert newNewCoordSys.xhat == batoid.Vec3(1,0,0)
            assert newNewCoordSys.yhat == batoid.Vec3(0,1,0)
            assert newNewCoordSys.zhat == batoid.Vec3(0,0,1)
            assert newNewCoordSys.origin == batoid.Vec3(x+x2, y+y2, z+z2)
            assert newNewCoordSys.rotation == batoid.Rot3()


@timer
def test_ne():
    objs = [
        batoid.CoordSys(),
        batoid.CoordSys(batoid.Vec3(0,0,1)),
        batoid.CoordSys(batoid.Vec3(0,1,0)),
        batoid.CoordSys(batoid.RotX(0.1)),
        batoid.CoordTransform(batoid.CoordSys(), batoid.CoordSys()),
        batoid.CoordTransform(batoid.CoordSys(), batoid.CoordSys(batoid.Vec3(0,0,1))),
        batoid.CoordTransform(batoid.CoordSys(), batoid.CoordSys(batoid.RotX(0.1)))
    ]
    all_obj_diff(objs)


if __name__ == '__main__':
    test_shift()
    test_ne()
