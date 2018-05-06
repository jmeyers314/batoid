import batoid
import numpy as np
import os
from test_helpers import timer, do_pickle, all_obj_diff
import time
import yaml


@timer
def test_optic():
    if __name__ == '__main__':
        nside = 128
    else:
        nside = 32

    rays = batoid.rayGrid(20, 12.0, 0.005, 0.005, -1.0, nside, 500e-9, batoid.ConstMedium(1.0))

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))
    t_fast = 0.0
    t_slow = 0.0

    fn = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])
    do_pickle(telescope)

    t0 = time.time()

    rays_fast, _ = telescope.trace(rays)
    t1 = time.time()
    rays_slow = batoid.RayVector([telescope.trace(r)[0] for r in rays])
    t2 = time.time()

    assert rays_fast == rays_slow
    t_fast = t1 - t0
    t_slow = t2 - t1
    print("Fast trace: {:5.3f} s".format(t_fast))
    print("            {} rays per second".format(int(nrays/t_fast)))
    print("Slow trace: {:5.3f} s".format(t_slow))
    print("            {} rays per second".format(int(nrays/t_slow)))


@timer
def test_traceFull():
    if __name__ == '__main__':
        nside = 128
    else:
        nside = 32

    rays = batoid.rayGrid(20, 12.0, 0.005, 0.005, -1.0, nside, 500e-9, batoid.ConstMedium(1.0))

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))

    fn = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    tf = telescope.traceFull(rays)
    rays, _ = telescope.trace(rays)

    assert rays == tf[-1]['out']


@timer
def test_traceReverse():
    if __name__ == '__main__':
        nside = 128
    else:
        nside = 32

    fn = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    init_rays = batoid.rayGrid(20, 12.0, 0.005, 0.005, -1.0, nside, 500e-9, batoid.ConstMedium(1.0))
    forward_rays, _ = telescope.trace(init_rays, outCoordSys=batoid.CoordSys())

    # Now, turn the result rays around and trace backwards
    forward_rays = batoid.propagatedToTimesMany(forward_rays, [40]*len(forward_rays))
    reverse_rays = batoid.RayVector(
        [batoid.Ray(r.p0, -r.v, -r.t0, r.wavelength) for r in forward_rays]
    )

    final_rays, _ = telescope.traceReverse(reverse_rays, outCoordSys=batoid.CoordSys())
    # propagate all the way to t=0
    final_rays = batoid.propagatedToTimesMany(final_rays, [0]*len(final_rays))

    w = np.where(np.logical_not(final_rays.isVignetted))[0]
    for idx in w:
        np.testing.assert_allclose(init_rays[idx].x0, final_rays[idx].x0)
        np.testing.assert_allclose(init_rays[idx].y0, final_rays[idx].y0)
        np.testing.assert_allclose(init_rays[idx].z0, final_rays[idx].z0)
        np.testing.assert_allclose(init_rays[idx].vx, -final_rays[idx].vx)
        np.testing.assert_allclose(init_rays[idx].vy, -final_rays[idx].vy)
        np.testing.assert_allclose(init_rays[idx].vz, -final_rays[idx].vz)
        np.testing.assert_allclose(final_rays[idx].t0, 0)


@timer
def test_ne():
    objs = [
        batoid.Mirror(batoid.Plane()),
        batoid.Detector(batoid.Plane()),
        batoid.Baffle(batoid.Plane()),
        batoid.RefractiveInterface(batoid.Plane()),
        batoid.Mirror(batoid.Paraboloid(0.1)),
        batoid.Detector(batoid.Paraboloid(0.1)),
        batoid.Baffle(batoid.Paraboloid(0.1)),
        batoid.RefractiveInterface(batoid.Paraboloid(0.1)),
        batoid.Mirror(batoid.Plane(), obscuration=batoid.ObscCircle(0.1)),
        batoid.Mirror(batoid.Plane(), inMedium=batoid.ConstMedium(1.1)),
        batoid.Mirror(batoid.Plane(), outMedium=batoid.ConstMedium(1.1)),
        batoid.Mirror(batoid.Plane(), coordSys=batoid.CoordSys([0,0,1])),
        batoid.CompoundOptic([
            batoid.Mirror(batoid.Plane()),
            batoid.Mirror(batoid.Plane())
        ]),
        batoid.CompoundOptic(
            [batoid.Mirror(batoid.Plane()),
             batoid.Baffle(batoid.Plane())]
        ),
        batoid.Lens(
            [batoid.RefractiveInterface(batoid.Plane()),
             batoid.RefractiveInterface(batoid.Plane())],
            batoid.ConstMedium(1.1)
        )
    ]
    all_obj_diff(objs)


if __name__ == '__main__':
    test_optic()
    test_traceFull()
    test_traceReverse()
    test_ne()
