import batoid
import numpy as np
import os
from test_helpers import timer
import time
import yaml


@timer
def test_optic():
    if __name__ == '__main__':
        nside = 256
    else:
        nside = 64

    rays = batoid._batoid.rayGrid(20, 2.0, 0.01, 0.01, nside, 500e-9, 1.0)

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))
    t_fast = 0.0
    t_slow = 0.0

    fn = os.path.join(batoid.datadir, "hsc", "HSC3.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

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
        nside = 256
    else:
        nside = 64

    rays = batoid._batoid.rayGrid(20, 2.0, 0.01, 0.01, nside, 500e-9, 1.0)

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))
    t_fast = 0.0
    t_slow = 0.0

    fn = os.path.join(batoid.datadir, "hsc", "HSC3.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    tf = telescope.traceFull(rays)
    rays, _ = telescope.trace(rays)

    assert rays == tf[-1]['out']

if __name__ == '__main__':
    test_optic()
    test_traceFull()
