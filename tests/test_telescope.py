import jtrace
import numpy as np
import os
from test_helpers import timer
import time


@timer
def test_telescope():
    if __name__ == '__main__':
        naz = 1024
    else:
        naz = 64
    rays = jtrace.parallelRays(z=20, outer=4.18, inner=2.558,
                               theta_x=1.0*np.pi/180, theta_y=0.1*np.pi/180,
                               nradii=50, naz=naz, medium=jtrace.Air())
    nrays = len(rays)*len("ugrizy")
    print("Tracing {} rays.".format(nrays))
    t_fast = 0.0
    t_slow = 0.0
    for f in "ugrizy":
        fn = os.path.join(jtrace.datadir, "lsst", "LSST_{}.yaml".format(f))
        telescope = jtrace.Telescope.makeFromYAML(fn)
        t0 = time.time()
        rays_fast, isecs_fast = telescope.trace(rays)
        t1 = time.time()
        rays_slow, isecs_slow = zip(*[telescope.trace(r) for r in rays])
        rays_slow = jtrace.RayVector(rays_slow)
        isecs_slow = jtrace.IntersectionVector(isecs_slow)
        t2 = time.time()
        assert rays_fast == rays_slow
        assert isecs_fast == isecs_slow
        t_fast += t1 - t0
        t_slow += t2 - t1
    print("Fast trace: {:5.3f} s".format(t_fast))
    print("            {} rays per second".format(int(nrays/t_fast)))
    print("Slow trace: {:5.3f} s".format(t_slow))
    print("            {} rays per second".format(int(nrays/t_slow)))


if __name__ == '__main__':
    test_telescope()
