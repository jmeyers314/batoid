import batoid
from test_helpers import vec3_isclose, ray_isclose, timer, rays_allclose


def randomVec3():
    import random
    return batoid.Vec3(
        random.uniform(0, 1),
        random.uniform(0, 1),
        random.uniform(0, 1)
    )


def randomCoordSys():
    import random
    return batoid.CoordSys(
        randomVec3(),
        (batoid.RotX(random.uniform(0, 1))
         *batoid.RotY(random.uniform(0, 1))
         *batoid.RotZ(random.uniform(0, 1)))
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

        transform1to2 = batoid.CoordTransform(coordSys1, coordSys2)
        transform1to3 = batoid.CoordTransform(coordSys1, coordSys3)
        transform2to3 = batoid.CoordTransform(coordSys2, coordSys3)

        for i in range(10):
            vec3 = randomVec3()
            vec3_a = transform1to3.applyForward(vec3)
            vec3_b = transform2to3.applyForward(transform1to2.applyForward(vec3))
            assert vec3_isclose(vec3_a, vec3_b), "error with composite transform of Vec3"
            vec3_ra = transform1to3.applyReverse(vec3)
            vec3_rb = transform1to2.applyReverse(transform2to3.applyReverse(vec3))
            assert vec3_isclose(vec3_ra, vec3_rb), "error with reverse composite transform of Vec3"

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


if __name__ == '__main__':
    test_composition()
