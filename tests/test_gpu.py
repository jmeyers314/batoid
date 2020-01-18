import time
import numpy as np
import batoid
from test_helpers import timer, rays_allclose, checkAngle


def test_medium():
    np.random.seed(57721)
    wavelength = np.random.uniform(size=100000)

    t0 = time.time()
    mcpu = batoid.SellmeierMedium(
        0.6961663, 0.4079426, 0.8974794,
        0.00467914825849, 0.013512063073959999, 97.93400253792099
    )
    t1 = time.time()
    mgpu = batoid.SellmeierMedium2(
        0.6961663, 0.4079426, 0.8974794,
        0.00467914825849, 0.013512063073959999, 97.93400253792099
    )
    t2 = time.time()

    print(mcpu.getN(wavelength))
    print(mgpu.getN(wavelength))
    print(t1-t0)
    print(t2-t1)

if __name__ == '__main__':
    test_medium()
