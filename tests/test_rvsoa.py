import time
import numpy as np
import batoid
from test_helpers import timer, rays_allclose, checkAngle


#@timer
def testRV(N, Nthread):
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
    RV = batoid.RayVector.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )
    plane = batoid.Plane()

    t0 = time.time()
    RV.propagateInPlace(10.0)
    # plane.intersectInPlace(RV)
    plane.reflectInPlace(RV)
    t1 = time.time()
    print(f"RV took {t1 - t0} seconds")
    print(RV.r[0], RV.v[0], RV.t[0])



#@timer
def testRV4(N):
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
    RV4 = batoid.RayVector4.fromArrays(
        x, y, z, vx, vy, vz, t, w, flux, vignetted, failed
    )
    # elide copy to device?
    RV4._rv4.r.syncToDevice()
    RV4._rv4.v.syncToDevice()
    RV4._rv4.t.syncToDevice()

    plane = batoid.Plane()

    t0 = time.time()
    RV4._rv4.propagateInPlace(10.0)
    # plane._surface.intersectInPlace(RV4._rv4)
    plane._surface.reflectInPlace(RV4._rv4)
    t1 = time.time()
    print(f"RV4 took {t1 - t0} seconds")
    print(RV4.r[0], RV4.v[0], RV4.t[0])

    m1 = batoid._batoid.CPPSellmeierMedium(
        0.6961663, 0.4079426, 0.8974794,
        0.00467914825849, 0.013512063073959999, 97.93400253792099
    )
    m2 = batoid._batoid.CPPSellmeierMedium2(
        0.6961663, 0.4079426, 0.8974794,
        0.00467914825849, 0.013512063073959999, 97.93400253792099
    )
    plane._surface.refractInPlace(RV4._rv4, m2, m2, None)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-N", type=int, default=1_000_000)
    parser.add_argument("-Nthread", type=int, default=4)
    args = parser.parse_args()
    N = args.N
    Nthread = args.Nthread

    testRV(N, Nthread)
    testRV(N, Nthread)
    testRV4(N)
    testRV4(N)
