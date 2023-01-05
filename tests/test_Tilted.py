import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff, init_gpu, rays_allclose


@timer
def test_sag():
    rng = np.random.default_rng(5)
    for i in range(100):
        tanx = rng.uniform(-0.1, 0.1)
        tany = rng.uniform(-0.1, 0.1)
        tilted = batoid.Tilted(tanx, tany)
        for j in range(10):
            x = rng.normal()
            y = rng.normal()
            result = tilted.sag(x, y)
            np.testing.assert_allclose(
                result,
                x*tanx + y*tany,
                atol=1e-14, rtol=0
            )
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check vectorization
        x = rng.normal(size=(10, 10))
        y = rng.normal(size=(10, 10))
        np.testing.assert_allclose(tilted.sag(x, y), x*tanx + y*tany)
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            tilted.sag(x[::5,::2], y[::5,::2]),
            (x*tanx + y*tany)[::5,::2]
        )
        do_pickle(tilted)


@timer
def test_normal():
    rng = np.random.default_rng(57)
    for i in range(100):
        tanx = rng.uniform(-0.1, 0.1)
        tany = rng.uniform(-0.1, 0.1)
        tilted = batoid.Tilted(tanx, tany)
        for j in range(10):
            x = rng.normal()
            y = rng.normal()
            result = tilted.normal(x, y)
            np.testing.assert_allclose(
                result,
                (-tanx, -tany, np.sqrt(1-tanx*tanx-tany*tany)),
                atol=1e-14, rtol=0
            )
        # Check vectorization
        x = rng.normal(size=(10, 10))
        y = rng.normal(size=(10, 10))
        np.testing.assert_allclose(
            tilted.normal(x, y),
            np.broadcast_to(
                np.array([
                    -tanx, -tany, np.sqrt(1-tanx*tanx-tany*tany)
                ]),
                (10, 10, 3)
            )
        )
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            tilted.normal(x[::5,::2], y[::5,::2]),
            np.broadcast_to(
                np.array([
                    -tanx, -tany, np.sqrt(1-tanx*tanx-tany*tany)
                ]),
                (2, 5, 3)
            )
        )


@timer
def test_intersect():
    rng = np.random.default_rng(577)
    tanx = rng.uniform(-0.1, 0.1)
    tany = rng.uniform(-0.1, 0.1)
    size = 10_000
    tiltedCoordSys = batoid.CoordSys(origin=[0, 0, -1])
    tilted = batoid.Tilted(tanx, tany)
    x = rng.normal(0.0, 1.0, size=size)
    y = rng.normal(0.0, 1.0, size=size)
    z = np.full_like(x, -100.0)
    # If we shoot rays straight up, then it's easy to predict the intersection
    vx = np.zeros_like(x)
    vy = np.zeros_like(x)
    vz = np.ones_like(x)
    rv = batoid.RayVector(x, y, z, vx, vy, vz)
    np.testing.assert_allclose(rv.z, -100.0)
    rv2 = batoid.intersect(tilted, rv.copy(), tiltedCoordSys)
    assert rv2.coordSys == tiltedCoordSys
    rv2 = rv2.toCoordSys(batoid.CoordSys())
    np.testing.assert_allclose(rv2.x, x)
    np.testing.assert_allclose(rv2.y, y)
    np.testing.assert_allclose(rv2.z, tilted.sag(x, y)-1, rtol=0, atol=1e-12)

    # Check default intersect coordTransform
    rv2 = rv.copy().toCoordSys(tiltedCoordSys)
    batoid.intersect(tilted, rv2)
    assert rv2.coordSys == tiltedCoordSys
    rv2 = rv2.toCoordSys(batoid.CoordSys())
    np.testing.assert_allclose(rv2.x, x)
    np.testing.assert_allclose(rv2.y, y)
    np.testing.assert_allclose(rv2.z, tilted.sag(x, y)-1, rtol=0, atol=1e-12)


@timer
def test_reflect():
    rng = np.random.default_rng(5772)
    size = 10_000
    tanx = rng.uniform(-0.1, 0.1)
    tany = rng.uniform(-0.1, 0.1)
    plane = batoid.Tilted(tanx, tany)
    x = rng.normal(0.0, 1.0, size=size)
    y = rng.normal(0.0, 1.0, size=size)
    z = np.full_like(x, -100.0)
    vx = rng.uniform(-1e-5, 1e-5, size=size)
    vy = rng.uniform(-1e-5, 1e-5, size=size)
    vz = np.ones_like(x)
    rv = batoid.RayVector(x, y, z, vx, vy, vz)
    rvr = batoid.reflect(plane, rv.copy())
    rvr2 = plane.reflect(rv.copy())
    rays_allclose(rvr, rvr2)
    # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
    normal = plane.normal(rvr.x, rvr.y)

    # Test law of reflection
    a0 = np.einsum("ad,ad->a", normal, rv.v)[~rvr.failed]
    a1 = np.einsum("ad,ad->a", normal, -rvr.v)[~rvr.failed]
    np.testing.assert_allclose(
        a0, a1,
        rtol=0, atol=1e-12
    )

    # Test that rv.v, rvr.v and normal are all in the same plane
    np.testing.assert_allclose(
        np.einsum(
            "ad,ad->a",
            np.cross(normal, rv.v),
            rv.v
        )[~rvr.failed],
        0.0,
        rtol=0, atol=1e-12
    )


@timer
def test_refract():
    rng = np.random.default_rng(57721)
    size = 10_000
    tanx = rng.uniform(-0.1, 0.1)
    tany = rng.uniform(-0.1, 0.1)
    plane = batoid.Tilted(tanx, tany)
    m0 = batoid.ConstMedium(rng.normal(1.2, 0.01))
    m1 = batoid.ConstMedium(rng.normal(1.3, 0.01))
    x = rng.normal(0.0, 1.0, size=size)
    y = rng.normal(0.0, 1.0, size=size)
    z = np.full_like(x, -100.0)
    vx = rng.uniform(-1e-5, 1e-5, size=size)
    vy = rng.uniform(-1e-5, 1e-5, size=size)
    vz = np.sqrt(1-vx*vx-vy*vy)/m0.n
    rv = batoid.RayVector(x, y, z, vx, vy, vz)
    rvr = batoid.refract(plane, rv.copy(), m0, m1)
    rvr2 = plane.refract(rv.copy(), m0, m1)
    rays_allclose(rvr, rvr2)
    # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
    normal = plane.normal(rvr.x, rvr.y)

    # Test Snell's law
    s0 = np.sum(np.cross(normal, rv.v*m0.n)[~rvr.failed], axis=-1)
    s1 = np.sum(np.cross(normal, rvr.v*m1.n)[~rvr.failed], axis=-1)
    np.testing.assert_allclose(
        m0.n*s0, m1.n*s1,
        rtol=0, atol=1e-9
    )

    # Test that rv.v, rvr.v and normal are all in the same plane
    np.testing.assert_allclose(
        np.einsum(
            "ad,ad->a",
            np.cross(normal, rv.v),
            rv.v
        )[~rvr.failed],
        0.0,
        rtol=0, atol=1e-12
    )


@timer
def test_ne():
    objs = [
        batoid.Plane(),
        batoid.Tilted(0.0, 0.1),
        batoid.Tilted(0.1, 0.1),
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    tilted = batoid.Tilted(0, 0)

    # Fail if vz == 0
    rv = batoid.RayVector(0,0,0, 0,1,0)
    rv2 = batoid.intersect(tilted, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([True]))

    # Otherwise, succeeds
    rv = batoid.RayVector(0,0,0, 0,0,-1)
    rv2 = batoid.intersect(tilted, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([False]))


if __name__ == '__main__':
    init_gpu()
    test_sag()
    test_normal()
    test_intersect()
    test_reflect()
    test_refract()
    test_ne()
    test_fail()
