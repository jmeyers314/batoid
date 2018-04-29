import batoid
import numpy as np
import os
from test_helpers import timer
import time
import yaml


@timer
def parallel_trace_timing(nside=512):
    rays = batoid.circularGrid(20, 4.1, 0.5, 0.001, 0.001, -1.0, nside, nside, 500e-9, 1.0)

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))
    print()

    fn = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    print("Immutable trace")
    t0 = time.time()
    rays_out, _ = telescope.trace(rays)
    t1 = time.time()
    print("{} rays per second".format(int(nrays/(t1-t0))))
    print()

    print("Trace in place")
    t0 = time.time()
    telescope.traceInPlace(rays)
    t1 = time.time()
    print("{} rays per second".format(int(nrays/(t1-t0))))

    assert rays == rays_out


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nside", type=int, default=512)
    args = parser.parse_args()

    parallel_trace_timing(args.nside)
