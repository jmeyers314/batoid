import batoid
import numpy as np
from test_helpers import timer


@timer
def test_rSplit():
    for i in range(100):
        R = np.random.normal(0.7, 0.8)
        conic = np.random.uniform(-2.0, 1.0)
        ncoef = np.random.randint(0, 4)
        coefs = [np.random.normal(0, 1e-10) for i in range(ncoef)]
        asphere = batoid.Asphere(R, conic, coefs)

        rays = batoid.rayGrid(10, 2*R, 0.0, 0.0, -1.0, 16, 500e-9, 1.0, batoid.Air())
        coating = batoid.SimpleCoating(0.9, 0.1)
        reflectedRays = asphere.reflect(rays, coating)
        m1 = batoid.Air()
        m2 = batoid.ConstMedium(1.1)
        refractedRays = asphere.refract(rays, m1, m2, coating)
        reflectedRays2, refractedRays2 = asphere.rSplit(rays, m1, m2, coating)

        assert reflectedRays == reflectedRays2
        assert refractedRays == refractedRays2


if __name__ == '__main__':
    test_rSplit()
