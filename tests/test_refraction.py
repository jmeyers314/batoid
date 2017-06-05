import os
import numpy as np
import jtrace
from test_helpers import isclose, timer


@timer
def test_plane_refraction_plane():
    import random
    random.seed(5)
    plane = jtrace.Plane(10)
    n1 = 1.1
    n2 = 1.3
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(jtrace.Vec3(x, y, 0),
                         jtrace.Vec3(vx, vy, 1).UnitVec3()/n1,
                         0)
        isec = plane.intersect(ray)
        rray = isec.refractedRay(ray, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        assert isclose(
            jtrace.DotProduct(
                jtrace.CrossProduct(ray.v, isec.surfaceNormal),
                rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            n1*jtrace.CrossProduct(ray.v.UnitVec3(), isec.surfaceNormal).Magnitude(),
            n2*jtrace.CrossProduct(rray.v.UnitVec3(), isec.surfaceNormal).Magnitude(),
            rel_tol=0, abs_tol=1e-15)


@timer
def test_plane_refraction_reversal():
    import random
    random.seed(57)
    plane = jtrace.Plane(10)
    n1 = 1.5
    n2 = 1.2
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(jtrace.Vec3(x, y, 0),
                         jtrace.Vec3(vx, vy, 1).UnitVec3()/n1,
                         0)
        isec = plane.intersect(ray)
        rray = isec.refractedRay(ray, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

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
        assert isclose(cray.v.Magnitude(), 1./n1, rel_tol=1e-15)
        cpoint = cray(0)
        assert isclose(cpoint.x, x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.y, y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.z, 0, rel_tol=0, abs_tol=1e-10)


@timer
def test_paraboloid_refraction_plane():
    import random
    random.seed(577)
    para = jtrace.Paraboloid(-0.1, 10)
    n1 = 1.11
    n2 = 1.32
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = jtrace.Vec3(vx, vy, 1).UnitVec3()/n1
        ray = jtrace.Ray(x, y, 0, v.x, v.y, v.z, 0)
        isec = para.intersect(ray)
        rray = isec.refractedRay(ray, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

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
            n1*jtrace.CrossProduct(ray.v.UnitVec3(), isec.surfaceNormal).Magnitude(),
            n2*jtrace.CrossProduct(rray.v.UnitVec3(), isec.surfaceNormal).Magnitude(),
            rel_tol=0, abs_tol=1e-15)


@timer
def test_paraboloid_refraction_reversal():
    import random
    random.seed(5772)
    para = jtrace.Paraboloid(-0.1, 10)
    n1 = 1.43
    n2 = 1.34
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(jtrace.Vec3(x, y, 0),
                         jtrace.Vec3(vx, vy, 1).UnitVec3()/n1,
                         0)
        isec = para.intersect(ray)
        rray = isec.refractedRay(ray, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

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
        assert isclose(cray.v.Magnitude(), 1./n1, rel_tol=1e-15)
        cpoint = cray(0)
        assert isclose(cpoint.x, x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.y, y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint.z, 0, rel_tol=0, abs_tol=1e-10)


@timer
def test_asphere_refraction_plane():
    import random
    random.seed(57721)
    asphere = jtrace.Asphere(25.0, -0.97, [1e-3, 1e-5], 0.1)
    n1 = 1.7
    n2 = 1.2
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = jtrace.Vec3(vx, vy, 1).UnitVec3()/n1
        ray = jtrace.Ray(x, y, 0, v.x, v.y, v.z, 0)
        isec = asphere.intersect(ray)
        rray = isec.refractedRay(ray, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

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
            n1*jtrace.CrossProduct(ray.v.UnitVec3(), isec.surfaceNormal).Magnitude(),
            n2*jtrace.CrossProduct(rray.v.UnitVec3(), isec.surfaceNormal).Magnitude(),
            rel_tol=0, abs_tol=1e-15)


@timer
def test_asphere_refraction_reversal():
    import random
    random.seed(577215)
    asphere = jtrace.Asphere(23.0, -0.97, [1e-5, 1e-6], 0.1)
    n1 = 1.7
    n2 = 1.9
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(jtrace.Vec3(x, y, 0),
                         jtrace.Vec3(vx, vy, 1).UnitVec3()/n1,
                         0)
        isec = asphere.intersect(ray)
        rray = isec.refractedRay(ray, n1, n2)
        assert isclose(rray.v.Magnitude(), 1./n2, rel_tol=1e-15)

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
        assert isclose(cray.v.Magnitude(), 1./n1, rel_tol=1e-15)
        cpoint = cray(0)
        assert isclose(cpoint.x, x, rel_tol=0, abs_tol=1e-9)
        assert isclose(cpoint.y, y, rel_tol=0, abs_tol=1e-9)
        assert isclose(cpoint.z, 0, rel_tol=0, abs_tol=1e-9)


@timer
def test_const_medium_refraction():
    import random
    random.seed(5772156)

    asphere = jtrace.Asphere(25.0, -0.97, [1e-3, 1e-5], 0.1)
    for i in range(10000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = jtrace.Ray(x, y, 0, vx, vy, 1, 0)
        isec = asphere.intersect(ray)
        n1 = 1.7
        n2 = 1.2
        m1 = jtrace.ConstMedium(n1)
        m2 = jtrace.ConstMedium(n2)
        rray1 = isec.refractedRay(ray, n1, n2)
        rray2 = isec.refractedRay(ray, m1, m2)

        assert rray1 == rray2


@timer
def test_table_medium_refraction():
    import random
    random.seed(57721566)

    filename = os.path.join(jtrace.datadir, "media", "silica_dispersion.txt")
    wave, n = np.genfromtxt(filename).T
    table = jtrace.Table(wave, n, jtrace.Table.Interpolant.linear)
    silica = jtrace.TableMedium(table)
    air = jtrace.ConstMedium(1.000277)

    asphere = jtrace.Asphere(25.0, -0.97, [1e-3, 1e-5], 0.1)
    for i in range(10000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        wavelength = random.uniform(0.3, 1.2)
        ray = jtrace.Ray(x, y, 0, vx, vy, 1, 0, wavelength)
        isec = asphere.intersect(ray)

        n1 = silica.getN(wavelength)
        n2 = air.getN(wavelength)

        rray1 = isec.refractedRay(ray, silica, air)
        rray2 = isec.refractedRay(ray, n1, n2)

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
