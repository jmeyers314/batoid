import batoid
import numpy as np
import os
from test_helpers import timer
import time
import yaml


@timer
def parallel_trace_timing(nside=256):
    # rays = batoid._batoid.rayGrid(20, 8.2, 0.01, 0.01, nside, 500e-9, 1.0)
    rays = batoid._batoid.circularGrid(20, 4.1, 0.5, 0.1, 0.1, nside, nside, 500e-9, 1.0)

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))

    fn = os.path.join(batoid.datadir, "hsc", "HSC3.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    t0 = time.time()
    rays_out, _ = telescope.trace(rays)
    t1 = time.time()
    print("{} rays per second".format(int(nrays/(t1-t0))))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nside", type=int, default=256)
    args = parser.parse_args()
    parallel_trace_timing(args.nside)
