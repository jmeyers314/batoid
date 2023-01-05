from batoid.medium import ConstMedium
from batoid.optic import RefractiveInterface
import batoid
import numpy as np
from test_helpers import timer, rays_allclose


@timer
def test_rSplit():
    rng = np.random.default_rng(5)
    for _ in range(100):
        R = rng.normal(0.7, 0.8)
        conic = rng.uniform(-2.0, 1.0)
        ncoef = rng.integers(0, 5)
        coefs = [rng.normal(0, 1e-10) for i in range(ncoef)]
        asphere = batoid.Asphere(R, conic, coefs)

        theta_x = rng.normal(0.0, 1e-8)
        theta_y = rng.normal(0.0, 1e-8)
        rays = batoid.RayVector.asPolar(
            backDist=10.0, medium=batoid.Air(),
            wavelength=500e-9, outer=0.25*R,
            theta_x=theta_x, theta_y=theta_y,
            nrad=10, naz=10
        )
        coating = batoid.SimpleCoating(0.9, 0.1)
        reflectedRays = asphere.reflect(rays.copy(), coating=coating)
        m1 = batoid.Air()
        m2 = batoid.ConstMedium(1.1)
        refractedRays = asphere.refract(rays.copy(), m1, m2, coating=coating)
        refractedRays2, reflectedRays2 = asphere.rSplit(rays, m1, m2, coating)

        rays_allclose(reflectedRays, reflectedRays2)
        rays_allclose(refractedRays, refractedRays2)


@timer
def test_traceSplit_simple():
    telescope = batoid.CompoundOptic(
        name = "simple",
        stopSurface=batoid.Interface(batoid.Plane()),
        items = [
            batoid.Lens(
                name = "L1",
                items = [
                    batoid.RefractiveInterface(
                        batoid.Plane(),
                        name = "L1_entrance",
                        coordSys=batoid.CoordSys(origin=[0,0,0.3]),
                        inMedium=batoid.ConstMedium(1.0),
                        outMedium=batoid.ConstMedium(1.1)
                    ),
                    batoid.RefractiveInterface(
                        batoid.Plane(),
                        name = "L1_exit",
                        coordSys=batoid.CoordSys(origin=[0,0,0.2]),
                        inMedium=batoid.ConstMedium(1.1),
                        outMedium=batoid.ConstMedium(1.0)
                    )
                ]
            ),
            batoid.Mirror(
                batoid.Plane(),
                name="Mirror"
            ),
            batoid.Detector(
                batoid.Plane(),
                name="detector",
                coordSys=batoid.CoordSys(origin=[0, 0, 0.1])
            )
        ],
        pupilSize=1.0,
        backDist=1.0,
        inMedium=batoid.ConstMedium(1.0)
    )
    telescope['L1_entrance'].forwardCoating = batoid.SimpleCoating(0.02, 0.98)
    telescope['L1_entrance'].reverseCoating = batoid.SimpleCoating(0.02, 0.98)
    telescope['L1_exit'].forwardCoating = batoid.SimpleCoating(0.02, 0.98)
    telescope['L1_exit'].reverseCoating = batoid.SimpleCoating(0.02, 0.98)
    telescope['detector'].forwardCoating = batoid.SimpleCoating(0.02, 0.98)
    rays = batoid.RayVector.asPolar(
        telescope,
        wavelength=500e-9,
        theta_x=np.deg2rad(1.0),
        theta_y=0.0,
        nrad=10, naz=60
    )
    rForward, rReverse = telescope.traceSplit(rays.copy(), minFlux=1e-4)

    for r in rForward:
        r2 = telescope.trace(rays.copy(), path=r.path)
        w = ~r2.vignetted
        np.testing.assert_allclose(r.r, r2.r[w], atol=1e-14, rtol=0)
        np.testing.assert_allclose(r.v, r2.v[w], atol=1e-14, rtol=0)
        np.testing.assert_allclose(r.t, r2.t[w], atol=1e-14, rtol=0)

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
        w = ~r3.vignetted
        np.testing.assert_allclose(r.r, r3.r[w], atol=1e-14, rtol=0)
        np.testing.assert_allclose(r.v, r3.v[w], atol=1e-14, rtol=0)
        np.testing.assert_allclose(r.t, r3.t[w], atol=1e-14, rtol=0)

    for r in rReverse:
        r2 = telescope.trace(rays.copy(), path=r.path)
        w = ~r2.vignetted
        np.testing.assert_allclose(r.r, r2.r[w], atol=1e-14, rtol=0)
        np.testing.assert_allclose(r.v, r2.v[w], atol=1e-14, rtol=0)
        np.testing.assert_allclose(r.t, r2.t[w], atol=1e-14, rtol=0)

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
        w = ~r3.vignetted
        np.testing.assert_allclose(r.r, r3.r[w], atol=1e-14, rtol=0)
        np.testing.assert_allclose(r.v, r3.v[w], atol=1e-14, rtol=0)
        np.testing.assert_allclose(r.t, r3.t[w], atol=1e-14, rtol=0)


@timer
def test_traceSplit():
    optics = [
        batoid.Optic.fromYaml("DECam.yaml"),
        batoid.Optic.fromYaml("HSC.yaml"),
        batoid.Optic.fromYaml("LSST_r.yaml"),
        batoid.Optic.fromYaml("DESI.yaml"),
    ]
    minFlux = 1e-4
    if __name__ != '__main__':
        optics = optics[0:1]
        minFlux = 1e-3

    for optic in optics:
        # Make refractive interfaces partially reflective
        for surface in optic.itemDict.values():
            if isinstance(surface, batoid.RefractiveInterface):
                surface.forwardCoating = batoid.SimpleCoating(0.02, 0.98)
                surface.reverseCoating = batoid.SimpleCoating(0.02, 0.98)
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(0.02, 0.98)
        rays = batoid.RayVector.asPolar(
            optic,
            wavelength=620e-9,
            theta_x=np.deg2rad(0.1), theta_y=0.0,
            nrad=15, naz=90,
        )
        rForward, rReverse = optic.traceSplit(rays.copy(), minFlux=minFlux)

        for r in rForward:
            r2 = optic.trace(rays.copy(), path=r.path)
            w = ~r2.vignetted
            np.testing.assert_allclose(r.r, r2.r[w], atol=1e-12, rtol=0)
            np.testing.assert_allclose(r.v, r2.v[w], atol=1e-12, rtol=0)
            np.testing.assert_allclose(r.t, r2.t[w], atol=1e-12, rtol=0)

            tf = optic.traceFull(rays.copy(), path=r.path)
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
            w = ~r3.vignetted
            np.testing.assert_allclose(r.r, r3.r[w], atol=1e-12, rtol=0)
            np.testing.assert_allclose(r.v, r3.v[w], atol=1e-12, rtol=0)
            np.testing.assert_allclose(r.t, r3.t[w], atol=1e-12, rtol=0)

        for r in rReverse:
            r2 = optic.trace(rays.copy(), path=r.path)
            w = ~r2.vignetted
            np.testing.assert_allclose(r.r, r2.r[w], atol=1e-12, rtol=0)
            np.testing.assert_allclose(r.v, r2.v[w], atol=1e-12, rtol=0)
            np.testing.assert_allclose(r.t, r2.t[w], atol=1e-12, rtol=0)

            tf = optic.traceFull(rays.copy(), path=r.path)
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
            w = ~r3.vignetted
            np.testing.assert_allclose(r.r, r3.r[w], atol=1e-12, rtol=0)
            np.testing.assert_allclose(r.v, r3.v[w], atol=1e-12, rtol=0)
            np.testing.assert_allclose(r.t, r3.t[w], atol=1e-12, rtol=0)



if __name__ == '__main__':
    test_rSplit()
    test_traceSplit_simple()
    test_traceSplit()
