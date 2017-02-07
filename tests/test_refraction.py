import jtrace


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def test_plane_refraction_plane():
    import random
    plane = jtrace.Plane(10)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(jtrace.Vec3(x, y, 0),
                         jtrace.Vec3(vx, vy, 1).UnitVec3(),
                         0)
        isec = plane.intersect(ray)
        n1 = 1.1
        n2 = 1.3
        rray = isec.refractedRay(ray, n1, n2)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        assert isclose(
            jtrace.DotProduct(
                jtrace.CrossProduct(ray.v, isec.surfaceNormal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)


        # Test Snell's law
        assert isclose(
            n1*jtrace.CrossProduct(ray.v, isec.surfaceNormal).Magnitude(),
            n2*jtrace.CrossProduct(rray.v, isec.surfaceNormal).Magnitude(),
            rel_tol=0, abs_tol=1e-15)


def test_plane_refraction_reversal():
    import random
    plane = jtrace.Plane(10)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(jtrace.Vec3(x, y, 0),
                         jtrace.Vec3(vx, vy, 1).UnitVec3(),
                         0)
        isec = plane.intersect(ray)
        n1 = 1.5
        n2 = 1.2
        rray = isec.refractedRay(ray, n1, n2)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray(rray.t0+0.1)
        return_ray = jtrace.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        risec = plane.intersect(return_ray)
        assert isclose(isec.point.x, risec.point.x, rel_tol=0, abs_tol=1e-10)
        assert isclose(isec.point.y, risec.point.y, rel_tol=0, abs_tol=1e-10)
        assert isclose(isec.point.z, risec.point.z, rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = risec.refractedRay(return_ray, n2, n1)
        cpoint = cray(0)
        assert isclose(cpoint.x, x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.y, y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.z, 0, rel_tol=0, abs_tol=1e-10)


def test_paraboloid_refraction_plane():
    import random
    para = jtrace.Paraboloid(-0.1, 10)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(x, y, 0, vx, vy, 1, 0)
        isec = para.intersect(ray)
        n1 = 1.11
        n2 = 1.32
        rray = isec.refractedRay(ray, n1, n2)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        assert isclose(
            jtrace.DotProduct(
                jtrace.CrossProduct(ray.v, isec.surfaceNormal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            n1*jtrace.CrossProduct(ray.v, isec.surfaceNormal).Magnitude(),
            n2*jtrace.CrossProduct(rray.v, isec.surfaceNormal).Magnitude(),
            rel_tol=0, abs_tol=1e-15)


def test_paraboloid_refraction_reversal():
    import random
    para = jtrace.Paraboloid(-0.1, 10)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(jtrace.Vec3(x, y, 0),
                         jtrace.Vec3(vx, vy, 1).UnitVec3(),
                         0)
        isec = para.intersect(ray)
        n1 = 1.43
        n2 = 1.34
        rray = isec.refractedRay(ray, n1, n2)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray(rray.t0+0.1)
        return_ray = jtrace.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        risec = para.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(isec.point.x, risec.point.x, rel_tol=0, abs_tol=1e-10)
        assert isclose(isec.point.y, risec.point.y, rel_tol=0, abs_tol=1e-10)
        assert isclose(isec.point.z, risec.point.z, rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = risec.refractedRay(return_ray, n2, n1)
        cpoint = cray(0)
        assert isclose(cpoint.x, x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.y, y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.z, 0, rel_tol=0, abs_tol=1e-10)


def test_asphere_refraction_plane():
    import random
    asphere = jtrace.Asphere(25.0, -0.97, [1e-3, 1e-5], 0.1)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(x, y, 0, vx, vy, 1, 0)
        isec = asphere.intersect(ray)
        n1 = 1.7
        n2 = 1.2
        rray = isec.refractedRay(ray, n1, n2)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        assert isclose(
            jtrace.DotProduct(
                jtrace.CrossProduct(ray.v, isec.surfaceNormal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            n1*jtrace.CrossProduct(ray.v, isec.surfaceNormal).Magnitude(),
            n2*jtrace.CrossProduct(rray.v, isec.surfaceNormal).Magnitude(),
            rel_tol=0, abs_tol=1e-15)


def test_asphere_refraction_reversal():
    import random
    asphere = jtrace.Asphere(23.0, -0.97, [1e-5, 1e-6], 0.1)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(jtrace.Vec3(x, y, 0),
                         jtrace.Vec3(vx, vy, 1).UnitVec3(),
                         0)
        isec = asphere.intersect(ray)
        n1 = 1.7
        n2 = 1.9
        rray = isec.refractedRay(ray, n1, n2)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray(rray.t0+0.1)
        return_ray = jtrace.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        risec = asphere.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(isec.point.x, risec.point.x, rel_tol=0, abs_tol=1e-9)
        assert isclose(isec.point.y, risec.point.y, rel_tol=0, abs_tol=1e-9)
        assert isclose(isec.point.z, risec.point.z, rel_tol=0, abs_tol=1e-9)
        # Refract and propagate back to t=0.
        cray = risec.refractedRay(return_ray, n2, n1)
        cpoint = cray(0)
        assert isclose(cpoint.x, x, rel_tol=0, abs_tol=1e-9)
        assert isclose(cpoint.y, y, rel_tol=0, abs_tol=1e-9)
        assert isclose(cpoint.z, 0, rel_tol=0, abs_tol=1e-9)


if __name__ == '__main__':
    test_plane_refraction_plane()
    test_plane_refraction_reversal()
    test_paraboloid_refraction_plane()
    test_paraboloid_refraction_reversal()
    test_asphere_refraction_plane()
    test_asphere_refraction_reversal()
