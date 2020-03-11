import pytest
import time
import numpy as np
import batoid
from test_helpers import timer, rays_allclose, init_gpu

@timer
@pytest.mark.gpu
def test_medium(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)
    wavelength = np.random.uniform(size=Nray)

    mcpus = [
        batoid.ConstMedium(1.1),
        batoid.SellmeierMedium(
            0.6961663, 0.4079426, 0.8974794,
            0.00467914825849, 0.013512063073959999, 97.93400253792099
        ),
        batoid.SumitaMedium(
            0.6961663, 0.4079426, 0.8974794,
            0.00467914825849, 0.013512063073959999, 97.93400253792099
        ),
        batoid.Air()
    ]
    mgpus = [
        batoid.ConstMedium2(1.1),
        batoid.SellmeierMedium2(
            0.6961663, 0.4079426, 0.8974794,
            0.00467914825849, 0.013512063073959999, 97.93400253792099
        ),
        batoid.SumitaMedium2(
            0.6961663, 0.4079426, 0.8974794,
            0.00467914825849, 0.013512063073959999, 97.93400253792099
        ),
        batoid.Air2()
    ]

    for mcpu, mgpu in zip(mcpus, mgpus):
        t0 = time.time()
        for _ in range(Nloop):
            ncpu = mcpu.getN(wavelength)
        t1 = time.time()
        for _ in range(Nloop):
            ngpu = mgpu.getN(wavelength)
        t2 = time.time()

        print("test_medium")
        print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
        print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

        np.testing.assert_array_equal(ncpu, ngpu)


@timer
@pytest.mark.gpu
def test_coordtransform(Nthread=1, Nray=100_000, Nloop=1):
    x = np.random.uniform(size=Nray)
    y = np.random.uniform(size=Nray)+1
    z = np.random.uniform(size=Nray)-200
    vx = np.random.uniform(size=Nray)+3
    vy = np.random.uniform(size=Nray)+4
    vz = np.random.uniform(size=Nray)+5
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )
    print(rv2.r.shape)

    cs1 = batoid.CoordSys(origin=(0,1,2), rot=batoid.RotX(0.1))
    cs2 = batoid.CoordSys(origin=(20,-1,-2), rot=batoid.RotZ(0.1))
    xf1 = batoid.CoordTransform(cs1, cs2)
    xf2 = batoid.CoordTransform2(cs1, cs2)

    t0 = time.time()
    for _ in range(Nloop):
        xf1.applyForwardInPlace(rv)
        xf1.applyReverseInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        xf2.applyForwardInPlace(rv2)
        xf2.applyReverseInPlace(rv2)
    t2 = time.time()
    print("test_coordtransform")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_intersect_plane(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)
    y = np.random.uniform(size=Nray)+1
    z = np.random.uniform(size=Nray)-200
    vx = np.random.uniform(size=Nray)+3
    vy = np.random.uniform(size=Nray)+4
    vz = np.random.uniform(size=Nray)+5
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    plane = batoid.Plane()
    plane2 = batoid.Plane2()

    t0 = time.time()
    for _ in range(Nloop):
        plane.intersectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        plane2.intersectInPlace(rv2)
    t2 = time.time()
    print("test_intersect_plane")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_reflect_plane(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)
    y = np.random.uniform(size=Nray)+1
    z = np.random.uniform(size=Nray)-200
    vx = np.random.uniform(size=Nray)+3
    vy = np.random.uniform(size=Nray)+4
    vz = np.random.uniform(size=Nray)+5
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    plane = batoid.Plane()
    plane2 = batoid.Plane2()

    t0 = time.time()
    for _ in range(Nloop):
        plane.reflectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        plane2.reflectInPlace(rv2)
    t2 = time.time()

    print("test_reflect_plane")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_refract_plane(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)
    y = np.random.uniform(size=Nray)+1
    z = np.random.uniform(size=Nray)-200
    vx = np.random.uniform(size=Nray)+3
    vy = np.random.uniform(size=Nray)+4
    vz = np.random.uniform(size=Nray)+5
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v
    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    plane = batoid.Plane()
    plane2 = batoid.Plane2()

    m1 = batoid.ConstMedium(1.1)
    m2 = batoid.ConstMedium(1.2)
    m1gpu = batoid.ConstMedium2(1.1)
    m2gpu = batoid.ConstMedium2(1.2)

    t0 = time.time()
    for _ in range(Nloop):
        plane.refractInPlace(rv, m1, m2)
    t1 = time.time()
    for _ in range(Nloop):
        plane2.refractInPlace(rv2, m1gpu, m2gpu)
    t2 = time.time()

    print("test_refract_plane")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_intersect_sphere(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*(-1)
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    sphere = batoid.Sphere(21.5)
    sphere2 = batoid.Sphere2(21.5)

    t0 = time.time()
    for _ in range(Nloop):
        sphere.intersectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        sphere2.intersectInPlace(rv2)
    t2 = time.time()
    print("test_intersect_sphere")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_reflect_sphere(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*(-1)
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    sphere = batoid.Sphere(21.5)
    sphere2 = batoid.Sphere2(21.5)

    t0 = time.time()
    for _ in range(Nloop):
        sphere.reflectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        sphere2.reflectInPlace(rv2)
    t2 = time.time()

    print("test_reflect_sphere")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_refract_sphere(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*(-1)
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v
    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    sphere = batoid.Sphere(21.5)
    sphere2 = batoid.Sphere2(21.5)

    m1 = batoid.ConstMedium(1.1)
    m2 = batoid.ConstMedium(1.2)
    m1gpu = batoid.ConstMedium2(1.1)
    m2gpu = batoid.ConstMedium2(1.2)

    t0 = time.time()
    for _ in range(Nloop):
        sphere.refractInPlace(rv, m1, m2)
    t1 = time.time()
    for _ in range(Nloop):
        sphere2.refractInPlace(rv2, m1gpu, m2gpu)
    t2 = time.time()

    print("test_refract_sphere")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_intersect_paraboloid(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*(-1)
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    paraboloid = batoid.Paraboloid(21.5)
    paraboloid2 = batoid.Paraboloid2(21.5)

    t0 = time.time()
    for _ in range(Nloop):
        paraboloid.intersectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        paraboloid2.intersectInPlace(rv2)
    t2 = time.time()
    print("test_intersect_paraboloid")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_reflect_paraboloid(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*(-1)
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    paraboloid = batoid.Paraboloid(21.5)
    paraboloid2 = batoid.Paraboloid2(21.5)

    t0 = time.time()
    for _ in range(Nloop):
        paraboloid.reflectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        paraboloid2.reflectInPlace(rv2)
    t2 = time.time()
    print("test_reflect_paraboloid")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_refract_paraboloid(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*(-1)
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    paraboloid = batoid.Paraboloid(21.5)
    paraboloid2 = batoid.Paraboloid2(21.5)

    m1 = batoid.ConstMedium(1.1)
    m2 = batoid.ConstMedium(1.2)
    m1gpu = batoid.ConstMedium2(1.1)
    m2gpu = batoid.ConstMedium2(1.2)

    t0 = time.time()
    for _ in range(Nloop):
        paraboloid.refractInPlace(rv, m1, m2)
    t1 = time.time()
    for _ in range(Nloop):
        paraboloid2.refractInPlace(rv2, m1gpu, m2gpu)
    t2 = time.time()
    print("test_refract_paraboloid")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_intersect_quadric(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*(-1)
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    quadric = batoid.Quadric(21.5, -0.9)
    quadric2 = batoid.Quadric2(21.5, -0.9)

    t0 = time.time()
    for _ in range(Nloop):
        quadric.intersectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        quadric2.intersectInPlace(rv2)
    t2 = time.time()
    print("test_intersect_quadric")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_reflect_quadric(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*(-1)
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    quadric = batoid.Quadric(21.5, -0.9)
    quadric2 = batoid.Quadric2(21.5, -0.9)

    t0 = time.time()
    for _ in range(Nloop):
        quadric.reflectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        quadric2.reflectInPlace(rv2)
    t2 = time.time()
    print("test_reflect_quadric")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_refract_quadric(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*(-1)
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v
    failed = np.zeros(Nray, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    quadric = batoid.Quadric(21.5, -0.9)
    quadric2 = batoid.Quadric2(21.5, -0.9)

    m1 = batoid.ConstMedium(1.1)
    m2 = batoid.ConstMedium(1.2)
    m1gpu = batoid.ConstMedium2(1.1)
    m2gpu = batoid.ConstMedium2(1.2)

    t0 = time.time()
    for _ in range(Nloop):
        quadric.refractInPlace(rv, m1, m2)
    t1 = time.time()
    for _ in range(Nloop):
        quadric2.refractInPlace(rv2, m1gpu, m2gpu)
    t2 = time.time()
    print("test_refract_quadric")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_intersect_asphere(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*0.02-1
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= v
    vy /= v
    vz /= v

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    asphere = batoid.Asphere(19.835, -1.215, [0.0, -1.381e-9])
    asphere2 = batoid.Asphere2(19.835, -1.215, [0.0, -1.381e-9])

    t0 = time.time()
    for _ in range(Nloop):
        asphere.intersectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        asphere2.intersectInPlace(rv2)
    t2 = time.time()
    print("test_intersect_asphere")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_reflect_asphere(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*0.02-1
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= v
    vy /= v
    vz /= v

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    asphere = batoid.Asphere(19.835, -1.215, [0.0, -1.381e-9])
    asphere2 = batoid.Asphere2(19.835, -1.215, [0.0, -1.381e-9])

    t0 = time.time()
    for _ in range(Nloop):
        asphere.reflectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        asphere2.reflectInPlace(rv2)
    t2 = time.time()
    print("test_reflect_asphere")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_refract_asphere(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*0.02-1
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    asphere = batoid.Asphere(19.835, -1.215, [0.0, -1.381e-9])
    asphere2 = batoid.Asphere2(19.835, -1.215, [0.0, -1.381e-9])

    m1 = batoid.ConstMedium(1.1)
    m2 = batoid.ConstMedium(1.2)
    m1gpu = batoid.ConstMedium2(1.1)
    m2gpu = batoid.ConstMedium2(1.2)

    t0 = time.time()
    for _ in range(Nloop):
        asphere.refractInPlace(rv, m1, m2)
    t1 = time.time()
    for _ in range(Nloop):
        asphere2.refractInPlace(rv2, m1gpu, m2gpu)
    t2 = time.time()
    print("test_refract_asphere")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_bicubic(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(5772156)

    xs = np.linspace(-4.4, 4.4, 101)
    ys = np.linspace(-4.4, 4.4, 101)
    def f(x, y):
        return x**2*y - y**2*x + 3*x - 2
    def dfdx(x, y):
        return 2*x*y - y**2 + 3
    def dfdy(x, y):
        return x**2 - 2*y*x
    def d2fdxdy(x, y):
        return 2*x - 2*y

    zs = f(*np.meshgrid(xs, ys))
    surf = batoid.Bicubic(xs, ys, zs)
    surf2 = batoid.Bicubic2(xs, ys, zs)

    testx = np.random.uniform(-4, 4, size=Nray)
    testy = np.random.uniform(-4, 4, size=Nray)
    sag = surf2.sag(testx, testy)
    np.testing.assert_allclose(sag, f(testx, testy), atol=1e-10, rtol=0)

    norm = surf2.normal(testx, testy)
    arr = np.vstack([
        -dfdx(testx, testy),
        -dfdy(testx, testy),
        np.ones(len(testx))
    ]).T
    arr /= np.sqrt(np.sum(arr**2, axis=1))[:,None]
    np.testing.assert_allclose(norm, arr, atol=1e-10, rtol=0)


@timer
@pytest.mark.gpu
def test_intersect_bicubic(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(5772156)

    # Try some intersection
    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*0.02-1
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    xs = np.linspace(-4.4, 4.4, 101)
    ys = np.linspace(-4.4, 4.4, 101)
    def f(x, y):
        return x**2*y - y**2*x + 3*x - 2

    zs = f(*np.meshgrid(xs, ys))
    surf = batoid.Bicubic(xs, ys, zs)
    surf2 = batoid.Bicubic2(xs, ys, zs)

    t0 = time.time()
    for _ in range(Nloop):
        surf.intersectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        surf2.intersectInPlace(rv2)
    t2 = time.time()
    print("test_intersect_bicubic")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_reflect_bicubic(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(5772156)

    # Try some intersection
    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*0.02-1
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    xs = np.linspace(-4.4, 4.4, 101)
    ys = np.linspace(-4.4, 4.4, 101)
    def f(x, y):
        return 1e-4*(x**2*y - y**2*x + 3*x - 2)

    zs = f(*np.meshgrid(xs, ys))
    surf = batoid.Bicubic(xs, ys, zs)
    surf2 = batoid.Bicubic2(xs, ys, zs)

    t0 = time.time()
    for _ in range(Nloop):
        surf.reflectInPlace(rv)
    t1 = time.time()
    for _ in range(Nloop):
        surf2.reflectInPlace(rv2)
    t2 = time.time()
    print("test_reflect_bicubic")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_refract_bicubic(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(5772156)

    # Try some intersection
    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*0.02-1
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    xs = np.linspace(-4.4, 4.4, 101)
    ys = np.linspace(-4.4, 4.4, 101)
    def f(x, y):
        return x**2*y - y**2*x + 3*x - 2

    zs = f(*np.meshgrid(xs, ys))
    surf = batoid.Bicubic(xs, ys, zs)
    surf2 = batoid.Bicubic2(xs, ys, zs)

    m1 = batoid.ConstMedium(1.1)
    m2 = batoid.ConstMedium(1.2)
    m1gpu = batoid.ConstMedium2(1.1)
    m2gpu = batoid.ConstMedium2(1.2)

    t0 = time.time()
    for _ in range(Nloop):
        surf.refractInPlace(rv, m1, m2)
    t1 = time.time()
    for _ in range(Nloop):
        surf2.refractInPlace(rv2, m1gpu, m2gpu)
    t2 = time.time()

    print("test_refract_bicubic")
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

    if (Nloop == 1):
        np.testing.assert_allclose(rv.r, rv2.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.v, rv2.v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rv.t, rv2.t, rtol=0, atol=1e-13)


@timer
@pytest.mark.gpu
def test_extendedAsphere(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(5772156)

    xs = np.linspace(-4.4, 4.4, 101)
    ys = np.linspace(-4.4, 4.4, 101)
    def f(x, y):
        return x**2*y - y**2*x + 3*x - 2
    def dfdx(x, y):
        return 2*x*y - y**2 + 3
    def dfdy(x, y):
        return x**2 - 2*y*x
    def d2fdxdy(x, y):
        return 2*x - 2*y

    zs = f(*np.meshgrid(xs, ys))
    asphere = batoid.Asphere(19.835, -1.215, [0.0, -1.381e-9])
    bc = batoid.Bicubic(xs, ys, zs)
    surf = batoid.Sum([asphere, bc])

    surf2 = batoid.ExtendedAsphere2(
        19.835, -1.215, [0.0, -1.381e-9],
        xs, ys, zs
    )

    testx = np.random.uniform(-4, 4, size=Nray)
    testy = np.random.uniform(-4, 4, size=Nray)
    np.testing.assert_allclose(
        surf.sag(testx, testy),
        surf2.sag(testx, testy),
        atol=1e-10, rtol=0
    )

    np.testing.assert_allclose(
        surf.normal(testx, testy),
        surf2.normal(testx, testy),
        atol=1e-10, rtol=0
    )


@timer
@pytest.mark.gpu
def test_obsc(Nthread=1, Nray=100_000, Nloop=1):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(5772)

    # Try some intersection
    x = np.random.uniform(size=Nray)-0.5
    y = np.random.uniform(size=Nray)-0.5
    z = np.random.uniform(size=Nray)+5
    vx = np.random.uniform(size=Nray)*0.02-0.01
    vy = np.random.uniform(size=Nray)*0.02-0.01
    vz = np.random.uniform(size=Nray)*0.02-1
    t = np.zeros(Nray)
    w = np.random.uniform(size=Nray)
    flux = np.random.uniform(size=Nray)
    vignetted = np.zeros(Nray, dtype=bool)
    failed = np.zeros(Nray, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v

    rv_0 = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted
    )
    rv2_0 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

    obscs = [
        batoid.ObscCircle(0.25),
        batoid.ObscAnnulus(0.1, 0.3, 0.01, 0.02),
        batoid.ObscNegation(batoid.ObscAnnulus(0.1, 0.3, 0.01, 0.02)),
        batoid.ObscRectangle(0.1, 0.14, 0.1, -0.02, 10.0),
        batoid.ObscRay(0.1, 0.3, 0.1, -0.02),
        batoid.ObscUnion([
            batoid.ObscRectangle(0.1, 0.14, 0.1, -0.02, 10.0),
            batoid.ObscRay(0.1, 0.3, 0.1, -0.02)
        ]),
        batoid.ObscIntersection([
            batoid.ObscRectangle(0.1, 0.14, 0.1, -0.02, 10.0),
            batoid.ObscRay(0.1, 0.3, 0.1, -0.02)
        ])
    ]

    obsc2s = [
        batoid.ObscCircle2(0.25),
        batoid.ObscAnnulus2(0.1, 0.3, 0.01, 0.02),
        batoid.ObscNegation2(batoid.ObscAnnulus2(0.1, 0.3, 0.01, 0.02)),
        batoid.ObscRectangle2(0.1, 0.14, 0.1, -0.02, 10.0),
        batoid.ObscRay2(0.1, 0.3, 0.1, -0.02),
        batoid.ObscUnion2([
            batoid.ObscRectangle2(0.1, 0.14, 0.1, -0.02, 10.0),
            batoid.ObscRay2(0.1, 0.3, 0.1, -0.02)
        ]),
        batoid.ObscIntersection2([
            batoid.ObscRectangle2(0.1, 0.14, 0.1, -0.02, 10.0),
            batoid.ObscRay2(0.1, 0.3, 0.1, -0.02)
        ])
    ]

    for obsc, obsc2 in zip(obscs, obsc2s):
        rv = rv_0.copy()
        rv2 = rv2_0.copy()
        t0 = time.time()
        for _ in range(Nloop):
            obsc.obscureInPlace(rv)
        t1 = time.time()
        for _ in range(Nloop):
            obsc2.obscureInPlace(rv2)
        t2 = time.time()

        print(f"{type(obsc)}")
        print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
        print(f"gpu time = {(t2-t1)*1e3:.1f} ms")

        if (Nloop == 1):
            np.testing.assert_array_equal(rv.failed, rv2.failed)
            np.testing.assert_array_equal(rv.vignetted, rv2.vignetted)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-Nray", type=int, default=3_000_000)
    parser.add_argument("-Nthread", type=int, default=12)
    parser.add_argument("-Nloop", type=int, default=50)
    args = parser.parse_args()
    Nray = args.Nray
    Nthread = args.Nthread
    Nloop = args.Nloop

    init_gpu()
    test_medium(Nthread, Nray, Nloop)
    test_coordtransform(Nthread, Nray, Nloop)
    test_intersect_plane(Nthread, Nray, Nloop)
    test_reflect_plane(Nthread, Nray, Nloop)
    test_refract_plane(Nthread, Nray, Nloop)
    test_intersect_sphere(Nthread, Nray, Nloop)
    test_reflect_sphere(Nthread, Nray, Nloop)
    test_refract_sphere(Nthread, Nray, Nloop)
    test_intersect_paraboloid(Nthread, Nray, Nloop)
    test_reflect_paraboloid(Nthread, Nray, Nloop)
    test_refract_paraboloid(Nthread, Nray, Nloop)
    test_intersect_quadric(Nthread, Nray, Nloop)
    test_reflect_quadric(Nthread, Nray, Nloop)
    test_refract_quadric(Nthread, Nray, Nloop)
    test_intersect_asphere(Nthread, Nray, Nloop)
    test_reflect_asphere(Nthread, Nray, Nloop)
    test_refract_asphere(Nthread, Nray, Nloop)
    test_bicubic(Nthread, Nray, Nloop)
    test_intersect_bicubic(Nthread, Nray, Nloop)
    test_reflect_bicubic(Nthread, Nray, Nloop)
    test_refract_bicubic(Nthread, Nray, Nloop)
    test_extendedAsphere(Nthread, Nray, Nloop)
    test_obsc(Nthread, Nray, Nloop)
