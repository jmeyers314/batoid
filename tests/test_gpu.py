import time
import numpy as np
import batoid
from test_helpers import timer, rays_allclose, checkAngle


def test_medium():
    np.random.seed(57721)
    wavelength = np.random.uniform(size=3_000_000)

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
        ncpu = mcpu.getN(wavelength)
        t1 = time.time()
        ngpu = mgpu.getN(wavelength)
        t2 = time.time()

        print(f"cpu time = {(t1-t0)*1e3:.1f} ms")
        print(f"gpu time = {(t2-t1)*1e3:.1f} ms")
        np.testing.assert_array_equal(ncpu, ngpu)


if __name__ == '__main__':
    test_medium()
