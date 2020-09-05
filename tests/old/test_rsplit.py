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
        reflectedRays = asphere.reflect(rays.copy(), coating)
        m1 = batoid.Air()
        m2 = batoid.ConstMedium(1.1)
        refractedRays = asphere.refract(rays.copy(), m1, m2, coating)
        reflectedRays2, refractedRays2 = asphere.rSplit(rays, m1, m2, coating)

        assert reflectedRays == reflectedRays2
        assert refractedRays == refractedRays2


@timer
def test_traceSplit():
    telescope = batoid.Optic.fromYaml("HSC.yaml")
    # Make refractive interfaces partially reflective
    for surface in telescope.itemDict.values():
        if isinstance(surface, batoid.RefractiveInterface):
            surface.forwardCoating = batoid.SimpleCoating(0.02, 0.98)
            surface.reverseCoating = batoid.SimpleCoating(0.02, 0.98)
        if isinstance(surface, batoid.Detector):
            surface.forwardCoating = batoid.SimpleCoating(0.02, 0.98)
    rays = batoid.RayVector.asPolar(
        telescope,
        wavelength=620e-9,
        theta_x=np.deg2rad(0.1), theta_y=0.0,
        nrad=10, naz=60,
    )
    rForward, rReverse = telescope.traceSplit(rays, minFlux=1e-4)

    for r in rForward:
        r2 = telescope.trace(rays.copy(), path=r.path)
        r2.trimVignetted()
        np.testing.assert_array_equal(r.r, r2.r)
        np.testing.assert_array_equal(r.v, r2.v)
        np.testing.assert_array_equal(r.t, r2.t)

        tf = telescope.traceFull(rays.copy(), path=r.path)
        keys = []
        for item in r.path:
            j = 0
            key = f"{item}_{j}"
            while key in keys:
                j += 1
                key = f"{item}_{j}"
            keys.append(key)
        assert keys == [k for k in tf.keys()]

        r3 = tf[keys[-1]]['out']
        r3.trimVignetted()
        np.testing.assert_array_equal(r.r, r3.r)
        np.testing.assert_array_equal(r.v, r3.v)
        np.testing.assert_array_equal(r.t, r3.t)

    for r in rReverse:
        r2 = telescope.trace(rays.copy(), path=r.path)
        r2.trimVignetted()
        np.testing.assert_array_equal(r.r, r2.r)
        np.testing.assert_array_equal(r.v, r2.v)
        np.testing.assert_array_equal(r.t, r2.t)

        tf = telescope.traceFull(rays.copy(), path=r.path)
        keys = []
        for item in r.path:
            j = 0
            key = f"{item}_{j}"
            while key in keys:
                j += 1
                key = f"{item}_{j}"
            keys.append(key)
        assert keys == [k for k in tf.keys()]

        r3 = tf[keys[-1]]['out']
        r3.trimVignetted()
        np.testing.assert_array_equal(r.r, r3.r)
        np.testing.assert_array_equal(r.v, r3.v)
        np.testing.assert_array_equal(r.t, r3.t)



if __name__ == '__main__':
    test_rSplit()
    test_traceSplit()
