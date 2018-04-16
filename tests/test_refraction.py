import os
import numpy as np
import batoid
from test_helpers import isclose, timer


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
        ray = batoid.Ray(batoid.Vec3(x, y, -10),
                         batoid.Vec3(vx, vy, 1).UnitVec3()/n1,
                         0)
        iray = plane.intercept(ray)
        rray = batoid._batoid.refract(iray, plane, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        normal = plane.normal(iray.p0.x, iray.p0.y)
        assert isclose(
            batoid.DotProduct(
                batoid.CrossProduct(ray.v, normal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            n1*batoid.CrossProduct(ray.v.UnitVec3(), normal).Magnitude(),
            n2*batoid.CrossProduct(rray.v.UnitVec3(), normal).Magnitude(),
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
        ray = batoid.Ray(batoid.Vec3(x, y, -10),
                         batoid.Vec3(vx, vy, 1).UnitVec3()/n1,
                         0)
        iray = plane.intercept(ray)
        rray = batoid._batoid.refract(iray, plane, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t0+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        riray = plane.intercept(return_ray)
        assert isclose(iray.p0.x, riray.p0.x, rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0.y, riray.p0.y, rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0.z, riray.p0.z, rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = batoid._batoid.refract(riray, plane, n2, n1)
        assert isclose(cray.v.Magnitude(), 1./n1, rel_tol=1e-15)
        cpoint = cray.positionAtTime(0)
        assert isclose(cpoint.x, x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.y, y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.z, -10, rel_tol=0, abs_tol=1e-10)


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
        v = batoid.Vec3(vx, vy, 1).UnitVec3()/n1
        ray = batoid.Ray(x, y, -10, v.x, v.y, v.z, 0)
        iray = para.intercept(ray)
        rray = batoid._batoid.refract(iray, para, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        normal = para.normal(iray.p0.x, iray.p0.y)
        assert isclose(
            batoid.DotProduct(
                batoid.CrossProduct(ray.v, normal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            n1*batoid.CrossProduct(ray.v.UnitVec3(), normal).Magnitude(),
            n2*batoid.CrossProduct(rray.v.UnitVec3(), normal).Magnitude(),
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
        ray = batoid.Ray(batoid.Vec3(x, y, -10),
                         batoid.Vec3(vx, vy, 1).UnitVec3()/n1,
                         0)
        iray = para.intercept(ray)
        rray = batoid._batoid.refract(iray, para, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t0+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        riray = para.intercept(return_ray)
        # First check that we intersected at the same point
        assert isclose(iray.p0.x, riray.p0.x, rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0.y, riray.p0.y, rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0.z, riray.p0.z, rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = batoid._batoid.refract(riray, para, n2, n1)
        assert isclose(cray.v.Magnitude(), 1./n1, rel_tol=1e-15)
        cpoint = cray.positionAtTime(0)
        assert isclose(cpoint.x, x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.y, y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.z, -10, rel_tol=0, abs_tol=1e-10)


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
        v = batoid.Vec3(vx, vy, 1).UnitVec3()/n1
        ray = batoid.Ray(x, y, -0.1, v.x, v.y, v.z, 0)
        iray = asphere.intercept(ray)
        rray = batoid._batoid.refract(iray, asphere, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        normal = asphere.normal(iray.p0.x, iray.p0.y)
        assert isclose(
            batoid.DotProduct(
                batoid.CrossProduct(ray.v, normal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            n1*batoid.CrossProduct(ray.v.UnitVec3(), normal).Magnitude(),
            n2*batoid.CrossProduct(rray.v.UnitVec3(), normal).Magnitude(),
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
        ray = batoid.Ray(batoid.Vec3(x, y, -0.1),
                         batoid.Vec3(vx, vy, 1).UnitVec3()/n1,
                         0)
        iray = asphere.intercept(ray)
        rray = batoid._batoid.refract(iray, asphere, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t0+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t0+0.1))
        riray = asphere.intercept(return_ray)
        # First check that we intersected at the same point
        assert isclose(iray.p0.x, riray.p0.x, rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0.y, riray.p0.y, rel_tol=0, abs_tol=1e-10)
        assert isclose(iray.p0.z, riray.p0.z, rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = batoid._batoid.refract(riray, asphere, n2, n1)
        assert isclose(cray.v.Magnitude(), 1./n1, rel_tol=1e-15)
        cpoint = cray.positionAtTime(0)
        assert isclose(cpoint.x, x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.y, y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.z, -0.1, rel_tol=0, abs_tol=1e-10)


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
        iray = asphere.intercept(ray)
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
        iray = asphere.intercept(ray)

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
