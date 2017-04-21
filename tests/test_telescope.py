import jtrace
import numpy as np
import os


datadir = os.path.join(os.path.dirname(__file__), 'data', 'lsst')


def test_telescope():
    # Just check that we can trace rays for now.  Not checking if results are accurate or not.
    rays = jtrace.parallelRays(z=20, outer=4.18, inner=2.558,
                               theta_x=1.0*np.pi/180, theta_y=0.1*np.pi/180)
    for i in range(6):
        telescope = jtrace.Telescope(os.path.join(datadir, "optics_{}.txt".format(i)))
        isecs = telescope.traceMany(rays)


if __name__ == '__main__':
    test_telescope()
