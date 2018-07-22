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
    wavelength = 500e-9  # arbitrary
    plane = batoid.Plane()
    m1 = batoid.ConstMedium(1.1)
    m2 = batoid.ConstMedium(1.3)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = np.array([vx, vy, 1])
        v /= np.linalg.norm(v)
        ray = batoid.Ray([x, y, -10], v/m1.getN(wavelength), 0)
        rray = plane.refract(ray, m1, m2)
        assert isclose(np.linalg.norm(rray.v), 1./m2.getN(wavelength), rel_tol=1e-15)
        # also check refractInPlace
        rray2 = batoid.Ray(ray)
        plane.refractInPlace(rray2, m1, m2)
        assert rray == rray2

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        normal = plane.normal(rray.r[0], rray.r[1])
        assert isclose(
            np.dot(np.cross(ray.v, normal), rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            m1.getN(wavelength)*np.linalg.norm(np.cross(normalized(ray.v), normal)),
            m2.getN(wavelength)*np.linalg.norm(np.cross(normalized(rray.v), normal)),
            rel_tol=0, abs_tol=1e-15)


@timer
def test_plane_refraction_reversal():
    import random
    random.seed(57)
    wavelength = 500e-9  # arbitrary
    plane = batoid.Plane()
    m1 = batoid.ConstMedium(1.5)
    m2 = batoid.ConstMedium(1.2)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = batoid.Ray([x, y, -10],
                         normalized(np.array([vx, vy, 1]))/m1.getN(wavelength),
                         0)
        rray = plane.refract(ray, m1, m2)
        assert isclose(np.linalg.norm(rray.v), 1./m2.getN(wavelength), rel_tol=1e-15)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t+0.1))
        riray = plane.intersect(return_ray)
        assert isclose(rray.r[0], riray.r[0], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[1], riray.r[1], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[2], riray.r[2], rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = plane.refract(return_ray, m2, m1)
        assert isclose(np.linalg.norm(cray.v), 1./m1.getN(wavelength), rel_tol=1e-15)
        cpoint = cray.positionAtTime(0)
        assert isclose(cpoint[0], x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[1], y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[2], -10, rel_tol=0, abs_tol=1e-10)


@timer
def test_paraboloid_refraction_plane():
    import random
    random.seed(577)
    wavelength = 500e-9  # arbitrary
    para = batoid.Paraboloid(-20.0)
    m1 = batoid.ConstMedium(1.11)
    m2 = batoid.ConstMedium(1.32)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = normalized(np.array([vx, vy, 1]))/m1.getN(wavelength)
        ray = batoid.Ray(x, y, -10, v[0], v[1], v[2], 0)
        rray = para.refract(ray, m1, m2)
        assert isclose(np.linalg.norm(rray.v), 1./m2.getN(wavelength), rel_tol=1e-15)
        # also check refractInPlace
        rray2 = batoid.Ray(ray)
        para.refractInPlace(rray2, m1, m2)
        assert rray == rray2

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        normal = para.normal(rray.r[0], rray.r[1])
        assert isclose(
            np.dot(np.cross(ray.v, normal), rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            m1.getN(wavelength)*np.linalg.norm(np.cross(normalized(ray.v), normal)),
            m2.getN(wavelength)*np.linalg.norm(np.cross(normalized(rray.v), normal)),
            rel_tol=0, abs_tol=1e-15)


@timer
def test_paraboloid_refraction_reversal():
    import random
    random.seed(5772)
    wavelength = 500e-9  # arbitrary
    para = batoid.Paraboloid(-20.0)
    m1 = batoid.ConstMedium(1.43)
    m2 = batoid.ConstMedium(1.34)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = batoid.Ray([x, y, -10],
                         normalized(np.array([vx, vy, 1]))/m1.getN(wavelength),
                         0)
        rray = para.refract(ray, m1, m2)
        assert isclose(np.linalg.norm(rray.v), 1./m2.getN(wavelength), rel_tol=1e-15)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t+0.1))
        riray = para.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(rray.r[0], riray.r[0], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[1], riray.r[1], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[2], riray.r[2], rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = para.refract(return_ray, m2, m1)
        assert isclose(np.linalg.norm(cray.v), 1./m1.getN(wavelength), rel_tol=1e-15)
        cpoint = cray.positionAtTime(0)
        assert isclose(cpoint[0], x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[1], y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[2], -10, rel_tol=0, abs_tol=1e-10)


@timer
def test_asphere_refraction_plane():
    import random
    random.seed(57721)
    wavelength = 500e-9  # arbitrary
    asphere = batoid.Asphere(25.0, -0.97, [1e-3, 1e-5])
    m1 = batoid.ConstMedium(1.7)
    m2 = batoid.ConstMedium(1.2)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        v = normalized(np.array([vx, vy, 1]))/m1.getN(wavelength)
        ray = batoid.Ray(x, y, -0.1, v[0], v[1], v[2], 0)
        rray = asphere.refract(ray, m1, m2)
        assert isclose(np.linalg.norm(rray.v), 1./m2.getN(wavelength), rel_tol=1e-15)
        # also check refractInPlace
        rray2 = batoid.Ray(ray)
        asphere.refractInPlace(rray2, m1, m2)
        assert rray == rray2

        # ray.v, surfaceNormal, and rray.v should all be in the same plane, and
        # hence (ray.v x surfaceNormal) . rray.v should have zero magnitude.
        # magnitude zero.
        normal = asphere.normal(rray.r[0], rray.r[1])
        assert isclose(
            np.dot(np.cross(ray.v, normal), rray.v),
            0.0, rel_tol=0, abs_tol=1e-15)

        # Test Snell's law
        assert isclose(
            m1.getN(wavelength)*np.linalg.norm(np.cross(normalized(ray.v), normal)),
            m2.getN(wavelength)*np.linalg.norm(np.cross(normalized(rray.v), normal)),
            rel_tol=0, abs_tol=1e-15)


@timer
def test_asphere_refraction_reversal():
    import random
    random.seed(577215)
    wavelength = 500e-9  # arbitrary
    asphere = batoid.Asphere(23.0, -0.97, [1e-5, 1e-6])
    m1 = batoid.ConstMedium(1.7)
    m2 = batoid.ConstMedium(1.9)
    for i in range(1000):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        vx = random.gauss(0, 1e-1)
        vy = random.gauss(0, 1e-1)
        ray = batoid.Ray([x, y, -0.1],
                         normalized(np.array([vx, vy, 1]))/m1.getN(wavelength),
                         0)
        rray = asphere.refract(ray, m1, m2)
        assert isclose(np.linalg.norm(rray.v), 1./m2.getN(wavelength), rel_tol=1e-15)

        # Invert the refracted ray, and see that it ends back at the starting
        # point

        # Keep going a bit before turning around though
        turn_around = rray.positionAtTime(rray.t+0.1)
        return_ray = batoid.Ray(turn_around, -rray.v, -(rray.t+0.1))
        riray = asphere.intersect(return_ray)
        # First check that we intersected at the same point
        assert isclose(rray.r[0], riray.r[0], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[1], riray.r[1], rel_tol=0, abs_tol=1e-10)
        assert isclose(rray.r[2], riray.r[2], rel_tol=0, abs_tol=1e-10)
        # Refract and propagate back to t=0.
        cray = asphere.refract(return_ray, m2, m1)
        assert isclose(np.linalg.norm(cray.v), 1./m1.getN(wavelength), rel_tol=1e-15)
        cpoint = cray.positionAtTime(0)
        assert isclose(cpoint[0], x, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[1], y, rel_tol=0, abs_tol=1e-10)
        assert isclose(cpoint[2], -0.1, rel_tol=0, abs_tol=1e-10)


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

        cm1 = batoid.ConstMedium(silica.getN(wavelength))
        cm2 = batoid.ConstMedium(air.getN(wavelength))

        rray1 = asphere.refract(ray, silica, air)
        rray2 = asphere.refract(ray, cm1, cm2)

        assert rray1 == rray2


@timer
def test_refraction_chromatic():
    import random
    random.seed(577215664)
    wavelength1 = 500e-9
    wavelength2 = 600e-9

    plane = batoid.Plane()
    filename = os.path.join(batoid.datadir, "media", "silica_dispersion.txt")
    wave, n = np.genfromtxt(filename).T
    wave *= 1e-6  # micron -> meters
    table = batoid.Table(wave, n, batoid.Table.Interpolant.linear)
    silica = batoid.TableMedium(table)
    air = batoid.Air()

    thx, thy = 0.001, 0.0001
    dirCos = batoid.utils.gnomicToDirCos(thx, thy)
    rv1 = batoid.rayGrid(10.0, 1., dirCos[0], dirCos[1], -dirCos[2], 2, wavelength1, silica)
    rv2 = batoid.rayGrid(10.0, 1., dirCos[0], dirCos[1], -dirCos[2], 2, wavelength2, silica)
    rays = []
    for ray in rv1:
        rays.append(ray)
    for ray in rv2:
        rays.append(ray)
    rvCombined = batoid.RayVector(rays)

    rv1r = plane.refract(rv1, silica, air)
    rv2r = plane.refract(rv2, silica, air)
    assert rv1r != rv2r
    rays = []
    for ray in rv1r:
        rays.append(ray)
    for ray in rv2r:
        rays.append(ray)
    rvrCombined1 = batoid.RayVector(rays)

    rvrCombined2 = plane.refract(rvCombined, silica, air)

    assert rvrCombined1 == rvrCombined2

    # Check in-place
    plane.refractInPlace(rv1, silica, air)
    plane.refractInPlace(rv2, silica, air)
    assert rv1 != rv2
    plane.refractInPlace(rvCombined, silica, air)
    rays = []
    for ray in rv1:
        rays.append(ray)
    for ray in rv2:
        rays.append(ray)
    rvCombined2 = batoid.RayVector(rays)

    assert rvCombined == rvCombined2


if __name__ == '__main__':
    test_plane_refraction_plane()
    test_plane_refraction_reversal()
    test_paraboloid_refraction_plane()
    test_paraboloid_refraction_reversal()
    test_asphere_refraction_plane()
    test_asphere_refraction_reversal()
    test_table_medium_refraction()
    test_refraction_chromatic()
