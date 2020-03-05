import batoid
import numpy as np
from test_helpers import timer
import time


@timer
def parallel_trace_timing(args):
    if args.nthread is not None:
        print("setting nthread to {}".format(args.nthread))
        batoid._batoid.setNThread(args.nthread)
    print("Using {} threads".format(batoid._batoid.getNThread()))

    if args.minChunk is not None:
        print("setting minChunk to {:_d}".format(args.minChunk))
        batoid._batoid.setMinChunk(args.minChunk)
    print("Using minChunk of {:_d}".format(batoid._batoid.getMinChunk()))
    print("Using nside of {:_d}".format(args.nside))

    # 0.3, 0.3 should be in bounds for current wide-field telescopes
    dirCos = batoid.utils.gnomonicToDirCos(np.deg2rad(0.3), np.deg2rad(0.3))

    if args.gpu:
        telescope = batoid.Optic.fromYaml("LSST_r_noobsc.yaml", gpu=True)
        pm = 'M1'
    elif args.lsst:
        telescope = batoid.Optic.fromYaml("LSST_r.yaml")
        pm = 'M1'
    elif args.decam:
        telescope = batoid.Optic.fromYaml("DECam.yaml")
        pm = 'PM'
    else:
        telescope = batoid.Optic.fromYaml("HSC.yaml")
        pm = 'PM'

    if args.gpu:
        rays = batoid.circularGrid(
            telescope.backDist,
            0.5*telescope.pupilSize,
            0.5*telescope.pupilObscuration*telescope.pupilSize,
            dirCos[0], dirCos[1], dirCos[2],
            args.nside, args.nside, 620e-9, 1.0, batoid.Air()
        )
        # Turn RayVector into RayVector2
        rays = batoid.RayVector2.fromArrays(
            rays.x, rays.y, rays.z, rays.vx, rays.vy, rays.vz, rays.t,
            rays.wavelength, rays.flux,
            rays.vignetted, rays.failed
        )
    else:
        rays = batoid.circularGrid(
            telescope.backDist,
            0.5*telescope.pupilSize,
            0.5*telescope.pupilObscuration*telescope.pupilSize,
            dirCos[0], dirCos[1], dirCos[2],
            args.nside, args.nside, 620e-9, 1.0, telescope.inMedium
        )


    nrays = len(rays)
    print("Tracing {:_d} rays.".format(nrays))
    print()

    # Optionally perturb the primary mirror using Zernike polynomial
    if args.perturbZ != 0:
        orig = telescope[pm].surface
        coefs = np.random.normal(size=args.perturbZ+1)*1e-6 # micron perturbations
        perturbation = batoid.Zernike(coefs, R_outer=telescope.pupilSize)
        telescope[pm].surface = batoid.Sum([orig, perturbation])

    # Optionally perturb primary mirror using bicubic spline
    if args.perturbBC != 0:
        if args.gpu:
            orig = telescope[pm].surface
            rad = telescope.pupilSize/2 * 1.1
            xs = np.linspace(-rad, rad, 100)
            ys = np.linspace(-rad, rad, 100)
            def f(x, y):
                return args.perturbBC*(np.cos(x) + np.sin(y))
            zs = f(*np.meshgrid(xs, ys))
            bc = batoid.Bicubic(xs, ys, zs)
            telescope[pm].surface = batoid.ExtendedAsphere2(
                orig.R,
                orig.conic,
                orig.coefs,
                xs, ys, zs,
            )
        else:
            orig = telescope[pm].surface
            rad = telescope.pupilSize/2 * 1.1
            xs = np.linspace(-rad, rad, 100)
            ys = np.linspace(-rad, rad, 100)
            def f(x, y):
                return args.perturbBC*(np.cos(x) + np.sin(y))
            zs = f(*np.meshgrid(xs, ys))
            bc = batoid.Bicubic(xs, ys, zs)
            telescope[pm].surface = batoid.Sum([orig, bc])

    if args.immutable:
        print("Immutable trace")
        t0 = time.time()

        for _ in range(args.nrepeat):
            rays_in = batoid.RayVector(rays)
            rays_out, _ = telescope.trace(rays_in)

        t1 = time.time()
        print("{:_d} rays per second".format(int(nrays*args.nrepeat/(t1-t0))))
        print()
    else:
        print("Trace in place")
        t0 = time.time()

        for _ in range(args.nrepeat):
            rays_out = rays.copy()
            telescope.traceInPlace(rays_out)
            rays_out.r # force copy back to host
            rays_out.v
            rays_out.t
            rays_out.vignetted
            rays_out.failed
        t1 = time.time()
        print("{:_d} rays per second".format(int(nrays*args.nrepeat/(t1-t0))))

    if args.plot:
        import matplotlib.pyplot as plt
        # rays_out.trimVignettedInPlace()
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
        plt.savefig("parallel_trace_timing.png")
        if not args.gpu:
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
    parser.add_argument("--gpu", action='store_true')
    args = parser.parse_args()

    parallel_trace_timing(args)
