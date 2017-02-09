import jtrace
from test_helpers import isclose


def test_plane_reflection_plane():
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
        rray = isec.reflectedRay(ray)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        assert isclose(
            jtrace.DotProduct(
                jtrace.CrossProduct(ray.v, isec.surfaceNormal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Actually, reflection off a plane is pretty straigtforward to test
        # directly.
        assert isclose(ray.v.x, rray.v.x)
        assert isclose(ray.v.y, rray.v.y)
        assert isclose(ray.v.z, -rray.v.z)


def test_plane_reflection_reversal():
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
        rray = isec.reflectedRay(ray)

        # Invert the reflected ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray(rray.t0+0.1)
        return_ray = jtrace.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        risec = plane.intersect(return_ray)
        assert isclose(isec.point.x, risec.point.x, rel_tol=0, abs_tol=1e-10)
        assert isclose(isec.point.y, risec.point.y, rel_tol=0, abs_tol=1e-10)
        assert isclose(isec.point.z, risec.point.z, rel_tol=0, abs_tol=1e-10)
        # Reflect and propagate back to t=0.
        cray = risec.reflectedRay(return_ray)
        cray = cray(0)
        assert isclose(cray.x, x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cray.y, y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cray.z, 0, rel_tol=0, abs_tol=1e-10)


def test_paraboloid_reflection_plane():
    import random
    para = jtrace.Paraboloid(-0.1, 10)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(x, y, 0, vx, vy, 1, 0)
        isec = para.intersect(ray)
        rray = isec.reflectedRay(ray)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        assert isclose(
            jtrace.DotProduct(
                jtrace.CrossProduct(ray.v, isec.surfaceNormal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)


def test_paraboloid_reflection_reversal():
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
        rray = isec.reflectedRay(ray)

        # Invert the reflected ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray(rray.t0+0.1)
        return_ray = jtrace.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        risec = para.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(isec.point.x, risec.point.x, rel_tol=0, abs_tol=1e-10)
        assert isclose(isec.point.y, risec.point.y, rel_tol=0, abs_tol=1e-10)
        assert isclose(isec.point.z, risec.point.z, rel_tol=0, abs_tol=1e-10)
        # Reflect and propagate back to t=0.
        cray = risec.reflectedRay(return_ray)
        cray = cray(0)
        assert isclose(cray.x, x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cray.y, y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cray.z, 0, rel_tol=0, abs_tol=1e-10)


def test_paraboloid_reflection_to_focus():
    import random
    for i in range(100):
        A = random.gauss(0, 1)
        para = jtrace.Paraboloid(A, 0)  # z = A(x^2 + y^2)
        for j in range(100):
            x = random.gauss(0, 1)
            y = random.gauss(0, 1)
            ray = jtrace.Ray(x,y,-1000, 0,0,1, 0)
            isec = para.intersect(ray)
            rray = isec.reflectedRay(ray)
            # Now see if rray goes through (0,0,1/(4*A))
            # Solve the x equation: 0 = rray.p0.x + rray.v.x*(t-t0) for t:
            # t = t0 - p0.x/vx
            t = rray.t0 - rray.p0.x/rray.v.x
            focus = rray(t)
            assert isclose(focus.x, 0, abs_tol=1e-12)
            assert isclose(focus.y, 0, abs_tol=1e-12)
            assert isclose(focus.z, 1.0/(4*A), abs_tol=1e-12)


def test_asphere_reflection_plane():
    import random
    asphere = jtrace.Asphere(25.0, -0.97, [1e-3, 1e-5], 0.1)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(x, y, 0, vx, vy, 1, 0)
        isec = asphere.intersect(ray)
        rray = isec.reflectedRay(ray)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        assert isclose(
            jtrace.DotProduct(
                jtrace.CrossProduct(ray.v, isec.surfaceNormal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)


def test_asphere_reflection_reversal():
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
        rray = isec.reflectedRay(ray)

        # Invert the reflected ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray(rray.t0+0.1)
        return_ray = jtrace.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        risec = asphere.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(isec.point.x, risec.point.x, rel_tol=0, abs_tol=1e-9)
        assert isclose(isec.point.y, risec.point.y, rel_tol=0, abs_tol=1e-9)
        assert isclose(isec.point.z, risec.point.z, rel_tol=0, abs_tol=1e-9)
        # Reflect and propagate back to t=0.
        cray = risec.reflectedRay(return_ray)
        cray = cray(0)
        assert isclose(cray.x, x, rel_tol=0, abs_tol=1e-9)
        assert isclose(cray.y, y, rel_tol=0, abs_tol=1e-9)
        assert isclose(cray.z, 0, rel_tol=0, abs_tol=1e-9)


if __name__ == '__main__':
    test_plane_reflection_plane()
    test_plane_reflection_reversal()
    test_paraboloid_reflection_plane()
    test_paraboloid_reflection_reversal()
    test_paraboloid_reflection_to_focus()
    test_asphere_reflection_plane()
    test_asphere_reflection_reversal()
