import jtrace
import numpy as np
import os
from test_helpers import timer
import time

datadir = os.path.join(os.path.dirname(__file__), 'data', 'lsst')

@timer
def test_telescope():
    rays = jtrace.parallelRays(z=20, outer=4.18, inner=2.558,
                               theta_x=1.0*np.pi/180, theta_y=0.1*np.pi/180,
                               nradii=50, naz=64)
    fast_trace = 0.0
    slow_trace = 0.0
    for i in range(6):
        telescope = jtrace.Telescope(os.path.join(datadir, "optics_{}.txt".format(i)))
        t0 = time.time()
        rays_fast = telescope.traceMany(rays)
        t1 = time.time()
        rays_slow = jtrace.IntersectionVector([telescope.trace(r) for r in rays])
        t2 = time.time()
        fast_trace += t1 - t0
        slow_trace += t2 - t1
        assert rays_fast == rays_slow
    print("Fast trace: {:5.3f} s".format(t1-t0))
    print("Slow trace: {:5.3f} s".format(t2-t1))

if __name__ == '__main__':
    test_telescope()
