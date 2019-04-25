import batoid
import numpy as np
import os
from test_helpers import timer
import time
import yaml


@timer
def parallel_trace_timing(nside=1024, nthread=None, minChunk=None):
    if nthread is not None:
        print("setting nthread to {}".format(nthread))
        batoid._batoid.setNThread(nthread)
    print("Using {} threads".format(batoid._batoid.getNThread()))

    if minChunk is not None:
        print("setting minChunk to {}".format(minChunk))
        batoid._batoid.setMinChunk(minChunk)
    print("Using minChunk of {}".format(batoid._batoid.getMinChunk()))

    # 0.3, 0.3 should be in bounds for current wide-field telescopes
    dirCos = batoid.utils.gnomonicToDirCos(np.deg2rad(0.3), np.deg2rad(0.3))

    if args.lsst:
        fn = os.path.join(batoid.datadir, "LSST", "LSST_i.yaml")
        pm = 'LSST.M1'
    elif args.decam:
        fn = os.path.join(batoid.datadir, "DECam", "DECam.yaml")
        pm = 'BlancoDECam.PM'
    else:
        fn = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
        pm = 'SubaruHSC.PM'
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    rays = batoid.circularGrid(
        telescope.dist,
        0.5*telescope.pupilSize,
        0.5*telescope.pupilObscuration*telescope.pupilSize,
        dirCos[0], dirCos[1], -dirCos[2],
        nside, nside, 750e-9, 1.0, telescope.inMedium
    )

    nrays = len(rays)
    print("Tracing {} rays.".format(nrays))
    print()

    # Optionally perturb the primary mirror using Zernike polynomial
    if args.perturbZ != 0:
        orig = telescope.itemDict[pm].surface
        coefs = np.random.normal(size=args.perturbZ+1)*1e-6 # micron perturbations
        perturbation = batoid.Zernike(coefs, R_outer=telescope.pupilSize)
        telescope.itemDict[pm].surface = batoid.Sum([orig, perturbation])

    # Optionally perturb primary mirror using bicubic spline
    if args.perturbBC != 0:
        orig = telescope.itemDict[pm].surface
        rad = telescope.pupilSize/2 * 1.1
        xs = np.linspace(-rad, rad, 100)
        ys = np.linspace(-rad, rad, 100)
        def f(x, y):
            return args.perturbBC*(np.cos(x) + np.sin(y))
        zs = f(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(xs, ys, zs)
        telescope.itemDict[pm].surface = batoid.Sum([orig, bc])

    if args.immutable:
        print("Immutable trace")
        t0 = time.time()

        for _ in range(args.nrepeat):
            rays_in = batoid.RayVector(rays)
            rays_out, _ = telescope.trace(rays_in)

        t1 = time.time()
        print("{} rays per second".format(int(nrays*args.nrepeat/(t1-t0))))
        print()
    else:
        print("Trace in place")
        t0 = time.time()

        for _ in range(args.nrepeat):
            rays_out = batoid.RayVector(rays)
            telescope.traceInPlace(rays_out)

        t1 = time.time()
        print("{} rays per second".format(int(nrays*args.nrepeat/(t1-t0))))

    if args.plot:
        import matplotlib.pyplot as plt
        rays_out.trimVignettedInPlace()
        x = rays_out.x
        y = rays_out.y
        x -= np.mean(x)
        y -= np.mean(y)
        x *= 1e6
        y *= 1e6
        plt.scatter(x, y, s=1, alpha=0.01)
        plt.xlim(np.std(x)*np.r_[-5,5])
        plt.ylim(np.std(y)*np.r_[-5,5])
        plt.xlabel("x (microns)")
        plt.ylabel("y (microns)")
        plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--nthread", type=int, default=None)
    parser.add_argument("--nrepeat", type=int, default=1)
    parser.add_argument("--minChunk", type=int, default=None)
    parser.add_argument("--perturbZ", type=int, default=0)
    parser.add_argument("--perturbBC", type=float, default=0.0)
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--lsst", action='store_true')
    parser.add_argument("--decam", action='store_true')
    parser.add_argument("--immutable", action='store_true')
    args = parser.parse_args()

    parallel_trace_timing(args.nside, args.nthread, args.minChunk)
