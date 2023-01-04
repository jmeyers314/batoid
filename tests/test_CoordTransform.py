import numpy as np
import batoid
from test_helpers import rays_allclose, timer, do_pickle, init_gpu, all_obj_diff


def randomCoordSys(rng):
    return batoid.CoordSys(
        rng.uniform(size=3),
        (batoid.RotX(rng.uniform())
         .dot(batoid.RotY(rng.uniform()))
         .dot(batoid.RotZ(rng.uniform())))
    )


def randomRayVector(rng, size, coordSys):
    x = rng.uniform(size=size)
    y = rng.uniform(size=size)
    z = rng.uniform(size=size)

    vx = rng.uniform(size=size)
    vy = rng.uniform(size=size)
    vz = rng.uniform(size=size)

    return batoid.RayVector(x, y, z, vx, vy, vz, coordSys=coordSys)


@timer
def test_simple_transform():
    rng = np.random.default_rng(5)
    size = 10_000

    # Worked a few examples out manually
    # Example 1
    coordSys1 = batoid.CoordSys()
    coordSys2 = batoid.CoordSys().shiftGlobal([0,0,1])
    rv = randomRayVector(rng, size, coordSys1)
    transform = batoid.CoordTransform(coordSys1, coordSys2)
    rv2 = batoid.applyForwardTransform(transform, rv.copy())
    np.testing.assert_allclose(rv.x, rv2.x)
    np.testing.assert_allclose(rv.y, rv2.y)
    np.testing.assert_allclose(rv.z-1, rv2.z)
    # Repeat using toCoordSys
    rv3 = rv.copy().toCoordSys(coordSys2)
    np.testing.assert_allclose(rv.x, rv3.x)
    np.testing.assert_allclose(rv.y, rv3.y)
    np.testing.assert_allclose(rv.z-1, rv3.z)
    # Transform of numpy array
    x, y, z = transform.applyForwardArray(rv.x, rv.y, rv.z)
    np.testing.assert_allclose(rv2.x, x)
    np.testing.assert_allclose(rv2.y, y)
    np.testing.assert_allclose(rv2.z, z)

    # Example 2
    # First for a single specific point I worked out
    coordSys1 = batoid.CoordSys()
    coordSys2 = batoid.CoordSys(origin=[1,1,1], rot=batoid.RotY(np.pi/2))
    x = y = z = np.array([2])
    vx = vy = vz = np.array([0])
    rv = batoid.RayVector(x, y, z, vx, vy, vz)
    transform = batoid.CoordTransform(coordSys1, coordSys2)
    rv2 = batoid.applyForwardTransform(transform, rv.copy())
    np.testing.assert_allclose(rv2.r, [[-1, 1, 1]])
    # Transform of numpy array
    x, y, z = transform.applyForwardArray(rv.x, rv.y, rv.z)
    np.testing.assert_allclose(rv2.x, x)
    np.testing.assert_allclose(rv2.y, y)
    np.testing.assert_allclose(rv2.z, z)

    # Here's the generalization
    # Also using alternate syntax for applyForward here.
    rv = randomRayVector(rng, size, coordSys1)
    rv2 = transform.applyForward(rv.copy())
    np.testing.assert_allclose(rv2.x, 1-rv.z)
    np.testing.assert_allclose(rv2.y, rv.y-1)
    np.testing.assert_allclose(rv2.z, rv.x-1)
    rv3 = rv.copy().toCoordSys(coordSys2)
    np.testing.assert_allclose(rv3.x, 1-rv.z)
    np.testing.assert_allclose(rv3.y, rv.y-1)
    np.testing.assert_allclose(rv3.z, rv.x-1)
    # Transform of numpy array
    x, y, z = transform.applyForwardArray(rv.x, rv.y, rv.z)
    np.testing.assert_allclose(rv2.x, x)
    np.testing.assert_allclose(rv2.y, y)
    np.testing.assert_allclose(rv2.z, z)


@timer
def test_roundtrip():
    rng = np.random.default_rng(57)
    size = 10_000

    for i in range(10):
        coordSys1 = randomCoordSys(rng)
        coordSys2 = randomCoordSys(rng)
        transform = batoid.CoordTransform(coordSys1, coordSys2)

        rv0 = randomRayVector(rng, size, coordSys1)
        rv1 = transform.applyForward(rv0.copy())
        rv2 = transform.applyReverse(rv1.copy())
        rv3 = rv0.copy().toCoordSys(coordSys2).toCoordSys(coordSys1)

        assert rv0.coordSys == coordSys1
        assert rv1.coordSys == coordSys2
        assert rv2.coordSys == coordSys1
        assert rv3.coordSys == coordSys1

        rays_allclose(rv0, rv2)
        rays_allclose(rv0, rv3)

        x, y, z = transform.applyForwardArray(rv0.x, rv0.y, rv0.z)
        x, y, z = transform.applyReverseArray(x, y, z)
        np.testing.assert_allclose(x, rv0.x)
        np.testing.assert_allclose(y, rv0.y)
        np.testing.assert_allclose(z, rv0.z)


@timer
def test_composition():
    rng = np.random.default_rng(577)
    size = 10_000

    for i in range(10):
        coordSys1 = randomCoordSys(rng)
        coordSys2 = randomCoordSys(rng)
        coordSys3 = randomCoordSys(rng)

        assert coordSys1 != coordSys2
        assert coordSys1 != coordSys3

        do_pickle(coordSys1)

        transform1to2 = batoid.CoordTransform(coordSys1, coordSys2)
        transform1to3 = batoid.CoordTransform(coordSys1, coordSys3)
        transform2to3 = batoid.CoordTransform(coordSys2, coordSys3)

        do_pickle(transform1to2)

        for j in range(10):
            rv = randomRayVector(rng, size, coordSys1)
            rv_a = transform1to3.applyForward(rv.copy())
            assert rv_a != rv
            assert rv_a.coordSys == coordSys3
            rv_b = transform2to3.applyForward(transform1to2.applyForward(rv.copy()))
            assert rv_b != rv
            assert rv_b.coordSys == coordSys3
            rays_allclose(rv_a, rv_b), "error with composite transform of RayVector"

            rv = randomRayVector(rng, size, coordSys3)
            rv_a = transform1to3.applyReverse(rv.copy())
            assert rv_a != rv
            assert rv_a.coordSys == coordSys1
            rv_b = transform1to2.applyReverse(transform2to3.applyReverse(rv.copy()))
            assert rv_b != rv
            assert rv_b.coordSys == coordSys1
            rays_allclose(rv_a, rv_b), "error with reverse composite transform of RayVector"


@timer
def test_ne():
    objs = [
        batoid.CoordSys(),
        batoid.CoordTransform(batoid.CoordSys(), batoid.CoordSys()),
        batoid.CoordTransform(
            batoid.CoordSys(),
            batoid.CoordSys(origin=(0,0,1))
        )
    ]
    all_obj_diff(objs)


@timer
def test_global_local():
    rng = np.random.default_rng(5772)
    for _ in range(100):
        coordSys = randomCoordSys(rng)
        toLocal = batoid.CoordTransform(batoid.CoordSys(), coordSys)
        toGlobal = batoid.CoordTransform(coordSys, batoid.CoordSys())

        r = rng.uniform(-1, 1, size=3)
        np.testing.assert_allclose(
            coordSys.toGlobal(r),
            toGlobal.applyForwardArray(*r),
            rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            coordSys.toLocal(r),
            toLocal.applyForwardArray(*r),
            rtol=0, atol=1e-15
        )

        r = rng.uniform(-1, 1, size=(10, 3))
        np.testing.assert_allclose(
            coordSys.toGlobal(r),
            toGlobal.applyForwardArray(*r.T).T,
            rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            coordSys.toLocal(r),
            toLocal.applyForwardArray(*r.T).T,
            rtol=0, atol=1e-15
        )


if __name__ == '__main__':
    init_gpu()
    test_simple_transform()
    test_roundtrip()
    test_composition()
    test_ne()
    test_global_local()