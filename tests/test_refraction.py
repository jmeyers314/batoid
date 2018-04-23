import os
import numpy as np
import batoid
from test_helpers import isclose, timer


def normalized(v):
    return v/np.linalg.norm(v)


@timer
def test_plane_refraction_plane():
    import random
    random.seed(5)
    plane = batoid.Plane()
    n1 = 1.1
    n2 = 1.3
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = np.array([vx, vy, 1])
        v /= np.linalg.norm(v)
        ray = batoid.Ray([x, y, -10], v/n1, 0)
        iray = plane.intersect(ray)
        rray = batoid._batoid.refract(iray, plane, n1, n2)
        assert isclose(np.linalg.norm(rray.v), 1./n2, rel_tol=1e-15)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        normal = plane.normal(iray.p0[0], iray.p0[1])
        assert isclose(
            np.dot(np.cross(ray.v, normal), rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            n1*np.linalg.norm(np.cross(normalized(ray.v), normal)),
            n2*np.linalg.norm(np.cross(normalized(rray.v), normal)),
            rel_tol=0, abs_tol=1e-15)


@timer
def test_plane_refraction_reversal():
    import random
    random.seed(57)
    plane = batoid.Plane()
    n1 = 1.5
    n2 = 1.2
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = batoid.Ray([x, y, -10],
                         normalized(np.array([vx, vy, 1]))/n1,
                         0)
        iray = plane.intersect(ray)
        rray = batoid._batoid.refract(iray, plane, n1, n2)
        assert isclose(np.linalg.norm(rray.v), 1./n2, rel_tol=1e-15)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t0+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        riray = plane.intersect(return_ray)
        assert isclose(iray.p0[0], riray.p0[0], rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0[1], riray.p0[1], rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0[2], riray.p0[2], rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = batoid._batoid.refract(riray, plane, n2, n1)
        assert isclose(np.linalg.norm(cray.v), 1./n1, rel_tol=1e-15)
        cpoint = cray.positionAtTime(0)
        assert isclose(cpoint[0], x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[1], y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[2], -10, rel_tol=0, abs_tol=1e-10)


@timer
def test_paraboloid_refraction_plane():
    import random
    random.seed(577)
    para = batoid.Paraboloid(-20.0)
    n1 = 1.11
    n2 = 1.32
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = normalized(np.array([vx, vy, 1]))/n1
        ray = batoid.Ray(x, y, -10, v[0], v[1], v[2], 0)
        iray = para.intersect(ray)
        rray = batoid._batoid.refract(iray, para, n1, n2)
        assert isclose(np.linalg.norm(rray.v), 1./n2, rel_tol=1e-15)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        normal = para.normal(iray.p0[0], iray.p0[1])
        assert isclose(
            np.dot(np.cross(ray.v, normal), rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            n1*np.linalg.norm(np.cross(normalized(ray.v), normal)),
            n2*np.linalg.norm(np.cross(normalized(rray.v), normal)),
            rel_tol=0, abs_tol=1e-15)


@timer
def test_paraboloid_refraction_reversal():
    import random
    random.seed(5772)
    para = batoid.Paraboloid(-20.0)
    n1 = 1.43
    n2 = 1.34
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = batoid.Ray([x, y, -10],
                         normalized(np.array([vx, vy, 1]))/n1,
                         0)
        iray = para.intersect(ray)
        rray = batoid._batoid.refract(iray, para, n1, n2)
        assert isclose(np.linalg.norm(rray.v), 1./n2, rel_tol=1e-15)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t0+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        riray = para.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(iray.p0[0], riray.p0[0], rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0[1], riray.p0[1], rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0[2], riray.p0[2], rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = batoid._batoid.refract(riray, para, n2, n1)
        assert isclose(np.linalg.norm(cray.v), 1./n1, rel_tol=1e-15)
        cpoint = cray.positionAtTime(0)
        assert isclose(cpoint[0], x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[1], y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[2], -10, rel_tol=0, abs_tol=1e-10)


@timer
def test_asphere_refraction_plane():
    import random
    random.seed(57721)
    asphere = batoid.Asphere(25.0, -0.97, [1e-3, 1e-5])
    n1 = 1.7
    n2 = 1.2
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = normalized(np.array([vx, vy, 1]))/n1
        ray = batoid.Ray(x, y, -0.1, v[0], v[1], v[2], 0)
        iray = asphere.intersect(ray)
        rray = batoid._batoid.refract(iray, asphere, n1, n2)
        assert isclose(np.linalg.norm(rray.v), 1./n2, rel_tol=1e-15)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        normal = asphere.normal(iray.p0[0], iray.p0[1])
        assert isclose(
            np.dot(np.cross(ray.v, normal), rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            n1*np.linalg.norm(np.cross(normalized(ray.v), normal)),
            n2*np.linalg.norm(np.cross(normalized(rray.v), normal)),
            rel_tol=0, abs_tol=1e-15)


@timer
def test_asphere_refraction_reversal():
    import random
    random.seed(577215)
    asphere = batoid.Asphere(23.0, -0.97, [1e-5, 1e-6])
    n1 = 1.7
    n2 = 1.9
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = batoid.Ray([x, y, -0.1],
                         normalized(np.array([vx, vy, 1]))/n1,
                         0)
        iray = asphere.intersect(ray)
        rray = batoid._batoid.refract(iray, asphere, n1, n2)
        assert isclose(np.linalg.norm(rray.v), 1./n2, rel_tol=1e-15)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t0+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        riray = asphere.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(iray.p0[0], riray.p0[0], rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0[1], riray.p0[1], rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0[2], riray.p0[2], rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = batoid._batoid.refract(riray, asphere, n2, n1)
        assert isclose(np.linalg.norm(cray.v), 1./n1, rel_tol=1e-15)
        cpoint = cray.positionAtTime(0)
        assert isclose(cpoint[0], x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[1], y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[2], -0.1, rel_tol=0, abs_tol=1e-10)


@timer
def test_const_medium_refraction():
    import random
    random.seed(5772156)

    asphere = batoid.Asphere(25.0, -0.97, [1e-3, 1e-5])
    for i in range(10000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = batoid.Ray(x, y, -0.1, vx, vy, 1, 0)
        iray = asphere.intersect(ray)
        n1 = 1.7
        n2 = 1.2
        m1 = batoid.ConstMedium(n1)
        m2 = batoid.ConstMedium(n2)
        rray1 = batoid._batoid.refract(ray, asphere, n1, n2)
        rray2 = batoid._batoid.refract(ray, asphere, m1, m2)

        assert rray1 == rray2


@timer
def test_table_medium_refraction():
    import random
    random.seed(57721566)

    filename = os.path.join(batoid.datadir, "media", "silica_dispersion.txt")
    wave, n = np.genfromtxt(filename).T
    table = batoid.Table(wave, n, batoid.Table.Interpolant.linear)
    silica = batoid.TableMedium(table)
    air = batoid.ConstMedium(1.000277)

    asphere = batoid.Asphere(25.0, -0.97, [1e-3, 1e-5])
    for i in range(10000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        wavelength = random.uniform(0.3, 1.2)
        ray = batoid.Ray(x, y, -0.1, vx, vy, 1, 0, wavelength)
        iray = asphere.intersect(ray)

        n1 = silica.getN(wavelength)
        n2 = air.getN(wavelength)

        rray1 = batoid._batoid.refract(ray, asphere, silica, air)
        rray2 = batoid._batoid.refract(ray, asphere, n1, n2)

        assert rray1 == rray2


if __name__ == '__main__':
    test_plane_refraction_plane()
    test_plane_refraction_reversal()
    test_paraboloid_refraction_plane()
    test_paraboloid_refraction_reversal()
    test_asphere_refraction_plane()
    test_asphere_refraction_reversal()
    test_const_medium_refraction()
    test_table_medium_refraction()
