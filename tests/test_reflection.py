import numpy as np
import batoid
from test_helpers import isclose, timer


@timer
def test_plane_reflection_plane():
    import random
    random.seed(5)
    plane = batoid.Plane()
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = np.array([vx, vy, 1])
        v / np.linalg.norm(v)
        ray = batoid.Ray([x, y, -10], v, 0)
        rray = plane.reflect(ray)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        assert isclose(
            np.dot(np.cross(ray.v, plane.normal(rray.r[0], rray.r[1])), rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Actually, reflection off a plane is pretty straigtforward to test
        # directly.
        assert isclose(ray.v[0], rray.v[0])
        assert isclose(ray.v[1], rray.v[1])
        assert isclose(ray.v[2], -rray.v[2])


@timer
def test_plane_reflection_reversal():
    import random
    random.seed(57)
    plane = batoid.Plane()
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = np.array([vx, vy, 1])
        v / np.linalg.norm(v)
        ray = batoid.Ray([x, y, -10], v, 0)
        rray = plane.reflect(ray)

        # Invert the reflected ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t+0.1))
        riray = plane.intersect(return_ray)
        assert isclose(rray.r[0], riray.r[0], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[1], riray.r[1], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[2], riray.r[2], rel_tol=0, abs_tol=1e-10)
        # Reflect and propagate back to t=0.
        cray = plane.reflect(return_ray)
        cray = cray.positionAtTime(0)
        assert isclose(cray[0], x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cray[1], y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cray[2], -10, rel_tol=0, abs_tol=1e-10)


@timer
def test_paraboloid_reflection_plane():
    import random
    random.seed(577)
    para = batoid.Paraboloid(-0.1)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = batoid.Ray(x, y, -10, vx, vy, 1, 0)
        rray = para.reflect(ray)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        assert isclose(
            np.dot(np.cross(ray.v, para.normal(rray.r[0], rray.r[1])), rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)


@timer
def test_paraboloid_reflection_reversal():
    import random
    random.seed(5772)
    para = batoid.Paraboloid(-5.0)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = np.array([vx, vy, 1])
        v /= np.linalg.norm(v)

        ray = batoid.Ray([x, y, -10], v, 0)
        rray = para.reflect(ray)

        # Invert the reflected ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t+0.1))
        riray = para.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(rray.r[0], riray.r[0], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[1], riray.r[1], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[2], riray.r[2], rel_tol=0, abs_tol=1e-10)
        # Reflect and propagate back to t=0.
        cray = para.reflect(return_ray)
        cray = cray.positionAtTime(0)
        assert isclose(cray[0], x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cray[1], y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cray[2], -10, rel_tol=0, abs_tol=1e-10)


@timer
def test_paraboloid_reflection_to_focus():
    import random
    random.seed(57721)
    for i in range(100):
        R = random.gauss(0, 3.0)
        para = batoid.Paraboloid(R)
        for j in range(100):
            x = random.gauss(0, 1)
            y = random.gauss(0, 1)
            ray = batoid.Ray(x,y,-1000, 0,0,1, 0)
            rray = para.reflect(ray)
            # Now see if rray goes through (0,0,R/2)
            # Solve the x equation: 0 = rray.r[0] + rray.v[0]*(t-t0) for t:
            # t = t0 - p0[0]/vx
            t = rray.t - rray.r[0]/rray.v[0]
            focus = rray.positionAtTime(t)
            assert isclose(focus[0], 0, abs_tol=1e-12)
            assert isclose(focus[1], 0, abs_tol=1e-12)
            assert isclose(focus[2], R/2.0, abs_tol=1e-12)


@timer
def test_asphere_reflection_plane():
    import random
    random.seed(577215)
    asphere = batoid.Asphere(25.0, -0.97, [1e-3, 1e-5])
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = batoid.Ray(x, y, -0.1, vx, vy, 1, 0)
        rray = asphere.reflect(ray)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        assert isclose(
            np.dot(np.cross(ray.v, asphere.normal(rray.r[0], rray.r[1])), rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)


@timer
def test_asphere_reflection_reversal():
    import random
    random.seed(5772156)
    asphere = batoid.Asphere(23.0, -0.97, [1e-5, 1e-6])
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = np.array([vx, vy, 1])
        v /= np.linalg.norm(v)
        ray = batoid.Ray([x, y, -0.1], v, 0)
        rray = asphere.reflect(ray)

        # Invert the reflected ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t+0.1))
        riray = asphere.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(rray.r[0], riray.r[0], rel_tol=0, abs_tol=1e-9)
        assert isclose(rray.r[1], riray.r[1], rel_tol=0, abs_tol=1e-9)
        assert isclose(rray.r[2], riray.r[2], rel_tol=0, abs_tol=1e-9)
        # Reflect and propagate back to t=0.
        cray = asphere.reflect(return_ray)
        cray = cray.positionAtTime(0)
        assert isclose(cray[0], x, rel_tol=0, abs_tol=1e-9)
        assert isclose(cray[1], y, rel_tol=0, abs_tol=1e-9)
        assert isclose(cray[2], -0.1, rel_tol=0, abs_tol=1e-9)


if __name__ == '__main__':
    test_plane_reflection_plane()
    test_plane_reflection_reversal()
    test_paraboloid_reflection_plane()
    test_paraboloid_reflection_reversal()
    test_paraboloid_reflection_to_focus()
    test_asphere_reflection_plane()
    test_asphere_reflection_reversal()
