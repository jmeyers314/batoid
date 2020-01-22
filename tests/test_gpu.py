import time
import numpy as np
import batoid
from test_helpers import timer, rays_allclose


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

        print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
        print(f"gpu time = {(t2-t1)*1e3:.1f} ms")
        print()
        np.testing.assert_array_equal(ncpu, ngpu)


@timer
def test_intersect(Nthread, N, Nloop):
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
    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")
    print(rv.x)
    print(rv2.x)
    if (Nloop == 1):
        assert rays_allclose(rv, rv2, atol=1e-13)


@timer
def test_reflect(Nthread, N, Nloop):
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

    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")
    print(rv.vz)
    print(rv2.vz)
    if (Nloop == 1):
        assert rays_allclose(rv, rv2, atol=1e-13)


@timer
def test_refract(Nthread, N, Nloop):
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

    print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
    print(f"gpu time = {(t2-t1)*1e3:.1f} ms")
    print(rv.vx)
    print(rv2.vx)
    if (Nloop == 1):
        assert rays_allclose(rv, rv2, atol=1e-13)


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

    test_medium(Nthread, N, Nloop)
    test_intersect(Nthread, N, Nloop)
    test_reflect(Nthread, N, Nloop)
    test_refract(Nthread, N, Nloop)
