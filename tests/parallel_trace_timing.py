import batoid
import numpy as np
import os
from test_helpers import timer
import time
import yaml


@timer
def parallel_trace_timing(nside=1024, nthread=None, minChunk=None):
    if nthread is not None:
        print("setting to nthread to {}".format(nthread))
        batoid._batoid.setNThread(nthread)
    print("Using {} threads".format(batoid._batoid.getNThread()))

    if minChunk is not None:
        print("setting to minChunk to {}".format(minChunk))
        batoid._batoid.setMinChunk(minChunk)
    print("Using minChunk of {}".format(batoid._batoid.getMinChunk()))

    theta_x = np.deg2rad(0.3)
    theta_y = np.deg2rad(0.3)
    dirCos = np.array([theta_x, theta_y, -1.0])
    dirCos = batoid.utils.normalized(dirCos)
    rays = batoid.circularGrid(20, 4.2, 0.5, dirCos[0], dirCos[1], dirCos[2], nside, nside, 700e-9, 1.0, batoid.ConstMedium(1.0))

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))
    print()

    fn = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    # Optionally perturb the primary mirror using Zernike polynomial
    if args.perturbZ != 0:
        orig = telescope.itemDict['SubaruHSC.PM'].surface
        coefs = np.random.normal(size=args.perturbZ+1)*1e-6 # micron perturbations
        perturbation = batoid.Zernike(coefs, R_outer=8.2)
        telescope.itemDict['SubaruHSC.PM'].surface = batoid.Sum([orig, perturbation])

    # Optionally perturb primary mirror using bicubic spline
    if args.perturbBC != 0:
        orig = telescope.itemDict['SubaruHSC.PM'].surface
        xs = np.linspace(-5, 5, 100)
        ys = np.linspace(-5, 5, 100)
        def f(x, y):
            return args.perturbBC*(np.cos(x) + np.sin(y))
        zs = f(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(xs, ys, zs)
        telescope.itemDict['SubaruHSC.PM'].surface = batoid.Sum([orig, bc])

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

    if args.plot:
        import matplotlib.pyplot as plt
        rays.trimVignettedInPlace()
        x = rays.x
        y = rays.y
        x -= np.mean(x)
        y -= np.mean(y)
        x *= 1e6
        y *= 1e6
        plt.scatter(x, y, s=1, alpha=0.01)
        plt.xlim(np.std(x)*np.r_[-3,3])
        plt.ylim(np.std(y)*np.r_[-3,3])
        plt.xlabel("x (microns)")
        plt.ylabel("y (microns)")
        plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--nthread", type=int, default=None)
    parser.add_argument("--minChunk", type=int, default=None)
    parser.add_argument("--perturbZ", type=int, default=0)
    parser.add_argument("--perturbBC", type=float, default=0.0)
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()

    parallel_trace_timing(args.nside, args.nthread, args.minChunk)
