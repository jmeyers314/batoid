import numpy as np
import batoid
from test_helpers import vec3_isclose, ray_isclose, timer, rays_allclose, do_pickle


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


def randomRay():
    import random
    return batoid.Ray(
        randomVec3(),
        randomVec3(),
        random.uniform(0, 1),
        random.uniform(0, 1),
        random.uniform(0, 1),
    )


def randomRayVector():
    import random
    return batoid.RayVector(
        [randomRay() for i in range(10)]
    )


@timer
def test_composition():
    import random
    random.seed(5)

    for i in range(10):
        coordSys1 = randomCoordSys()
        coordSys2 = randomCoordSys()
        coordSys3 = randomCoordSys()

        assert coordSys1 != coordSys2
        assert coordSys1 != coordSys3

        do_pickle(coordSys1)

        transform1to2 = batoid.CoordTransform(coordSys1, coordSys2)
        transform1to3 = batoid.CoordTransform(coordSys1, coordSys3)
        transform2to3 = batoid.CoordTransform(coordSys2, coordSys3)

        do_pickle(transform1to2)

        for i in range(10):
            vec3 = randomVec3()
            vec3_a = transform1to3.applyForward(vec3)
            vec3_b = transform2to3.applyForward(transform1to2.applyForward(vec3))
            np.testing.assert_allclose(vec3_a, vec3_b)
            vec3_ra = transform1to3.applyReverse(vec3)
            vec3_rb = transform1to2.applyReverse(transform2to3.applyReverse(vec3))
            np.testing.assert_allclose(vec3_ra, vec3_rb)

            ray = randomRay()
            ray_a = transform1to3.applyForward(ray)
            ray_b = transform2to3.applyForward(transform1to2.applyForward(ray))
            assert ray_isclose(ray_a, ray_b), "error with composite transform of Ray"
            ray_ra = transform1to3.applyReverse(ray)
            ray_rb = transform1to2.applyReverse(transform2to3.applyReverse(ray))
            assert ray_isclose(ray_ra, ray_rb), "error with reverse composite transform of Ray"

            rv = randomRayVector()
            rv_a = transform1to3.applyForward(rv)
            rv_b = transform2to3.applyForward(transform1to2.applyForward(rv))
            assert rays_allclose(rv_a, rv_b), "error with composite transform of RayVector"
            rv_ra = transform1to3.applyReverse(rv)
            rv_rb = transform1to2.applyReverse(transform2to3.applyReverse(rv))
            assert rays_allclose(rv_ra, rv_rb), "error with reverse composite transform of RayVector"

            # Test with numpy arrays
            xyz = rv.x, rv.y, rv.z
            xyz_a = transform1to3.applyForward(*xyz)
            xyz_b = transform2to3.applyForward(*transform1to2.applyForward(*xyz))
            xyz_c = [transform2to3.applyForward(transform1to2.applyForward(r.r)) for r in rv]
            np.testing.assert_allclose(xyz_a, xyz_b)
            np.testing.assert_allclose(xyz_a, np.transpose(xyz_c))
            # Should still work if we reshape.
            xyz2 = rv.x.reshape((2, 5)), rv.y.reshape((2, 5)), rv.z.reshape((2, 5))
            xyz2_a = transform1to3.applyForward(*xyz2)
            xyz2_b = transform2to3.applyForward(*transform1to2.applyForward(*xyz2))
            np.testing.assert_allclose(xyz2_a, xyz2_b)

            # And also work if we reverse
            np.testing.assert_allclose(xyz, transform1to3.applyReverse(*xyz_a))
            np.testing.assert_allclose(xyz, transform1to2.applyReverse(*transform2to3.applyReverse(*xyz_b)))

            # Test in-place on Ray
            ray = randomRay()
            ray_copy = batoid.Ray(ray)
            transform1to2.applyForwardInPlace(ray)
            transform2to3.applyForwardInPlace(ray)
            transform1to3.applyForwardInPlace(ray_copy)
            assert ray_isclose(ray, ray_copy)

            # in-place reverse on Ray
            ray = randomRay()
            ray_copy = batoid.Ray(ray)
            transform2to3.applyReverseInPlace(ray)
            transform1to2.applyReverseInPlace(ray)
            transform1to3.applyReverseInPlace(ray_copy)
            assert ray_isclose(ray, ray_copy)

            # Test in-place on RayVector
            rv = randomRayVector()
            rv_copy = batoid.RayVector(rv)
            transform1to2.applyForwardInPlace(rv)
            transform2to3.applyForwardInPlace(rv)
            transform1to3.applyForwardInPlace(rv_copy)
            assert rays_allclose(rv, rv_copy)

            # in-place reverse on RayVector
            rv = randomRayVector()
            rv_copy = batoid.RayVector(rv)
            transform2to3.applyReverseInPlace(rv)
            transform1to2.applyReverseInPlace(rv)
            transform1to3.applyReverseInPlace(rv_copy)
            assert rays_allclose(rv, rv_copy)


if __name__ == '__main__':
    test_composition()
