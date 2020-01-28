import time
import numpy as np
import batoid
from test_helpers import timer, rays_allclose, init_gpu


@timer
def test_medium(Nthread, N, Nloop):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)
    wavelength = np.random.uniform(size=N)

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
def test_coordtransform(Nthread, N, Nloop):
    x = np.random.uniform(size=N)
    y = np.random.uniform(size=N)+1
    z = np.random.uniform(size=N)-200
    vx = np.random.uniform(size=N)+3
    vy = np.random.uniform(size=N)+4
    vz = np.random.uniform(size=N)+5
    t = np.zeros(N)
    w = np.random.uniform(size=N)
    flux = np.random.uniform(size=N)
    vignetted = np.zeros(N, dtype=bool)
    failed = np.zeros(N, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )
    rv2 = batoid.RayVector2.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )

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
def test_intersect_plane(Nthread, N, Nloop):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=N)
    y = np.random.uniform(size=N)+1
    z = np.random.uniform(size=N)-200
    vx = np.random.uniform(size=N)+3
    vy = np.random.uniform(size=N)+4
    vz = np.random.uniform(size=N)+5
    t = np.zeros(N)
    w = np.random.uniform(size=N)
    flux = np.random.uniform(size=N)
    vignetted = np.zeros(N, dtype=bool)
    failed = np.zeros(N, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
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
def test_reflect_plane(Nthread, N, Nloop):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=N)
    y = np.random.uniform(size=N)+1
    z = np.random.uniform(size=N)-200
    vx = np.random.uniform(size=N)+3
    vy = np.random.uniform(size=N)+4
    vz = np.random.uniform(size=N)+5
    t = np.zeros(N)
    w = np.random.uniform(size=N)
    flux = np.random.uniform(size=N)
    vignetted = np.zeros(N, dtype=bool)
    failed = np.zeros(N, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
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
def test_refract_plane(Nthread, N, Nloop):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=N)
    y = np.random.uniform(size=N)+1
    z = np.random.uniform(size=N)-200
    vx = np.random.uniform(size=N)+3
    vy = np.random.uniform(size=N)+4
    vz = np.random.uniform(size=N)+5
    t = np.zeros(N)
    w = np.random.uniform(size=N)
    flux = np.random.uniform(size=N)
    vignetted = np.zeros(N, dtype=bool)
    failed = np.zeros(N, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v
    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
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
def test_intersect_sphere(Nthread, N, Nloop):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=N)-0.5
    y = np.random.uniform(size=N)-0.5
    z = np.random.uniform(size=N)+5
    vx = np.random.uniform(size=N)*0.02-0.01
    vy = np.random.uniform(size=N)*0.02-0.01
    vz = np.random.uniform(size=N)*(-1)
    t = np.zeros(N)
    w = np.random.uniform(size=N)
    flux = np.random.uniform(size=N)
    vignetted = np.zeros(N, dtype=bool)
    failed = np.zeros(N, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
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
def test_reflect_sphere(Nthread, N, Nloop):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=N)-0.5
    y = np.random.uniform(size=N)-0.5
    z = np.random.uniform(size=N)+5
    vx = np.random.uniform(size=N)*0.02-0.01
    vy = np.random.uniform(size=N)*0.02-0.01
    vz = np.random.uniform(size=N)*(-1)
    t = np.zeros(N)
    w = np.random.uniform(size=N)
    flux = np.random.uniform(size=N)
    vignetted = np.zeros(N, dtype=bool)
    failed = np.zeros(N, dtype=bool)

    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
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
def test_refract_sphere(Nthread, N, Nloop):
    batoid._batoid.setNThread(Nthread)
    np.random.seed(57721)

    x = np.random.uniform(size=N)-0.5
    y = np.random.uniform(size=N)-0.5
    z = np.random.uniform(size=N)+5
    vx = np.random.uniform(size=N)*0.02-0.01
    vy = np.random.uniform(size=N)*0.02-0.01
    vz = np.random.uniform(size=N)*(-1)
    t = np.zeros(N)
    w = np.random.uniform(size=N)
    flux = np.random.uniform(size=N)
    vignetted = np.zeros(N, dtype=bool)
    failed = np.zeros(N, dtype=bool)
    v = np.sqrt(vx*vx+vy*vy+vz*vz)
    vx /= 1.1*v
    vy /= 1.1*v
    vz /= 1.1*v
    rv = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
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


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-N", type=int, default=3_000_000)
    parser.add_argument("-Nthread", type=int, default=12)
    parser.add_argument("-Nloop", type=int, default=50)
    args = parser.parse_args()
    N = args.N
    Nthread = args.Nthread
    Nloop = args.Nloop

    init_gpu()
    test_medium(Nthread, N, Nloop)
    test_coordtransform(Nthread, N, Nloop)
    test_intersect_plane(Nthread, N, Nloop)
    test_reflect_plane(Nthread, N, Nloop)
    test_refract_plane(Nthread, N, Nloop)
    test_intersect_sphere(Nthread, N, Nloop)
    test_reflect_sphere(Nthread, N, Nloop)
    test_refract_sphere(Nthread, N, Nloop)
