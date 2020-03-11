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

    if args.lsst:
        print("Tracing through LSST optics")
        telescope = batoid.Optic.fromYaml("LSST_r.yaml", gpu=args.gpu)
        pm = 'M1'
    elif args.decam:
        print("Tracing through DECam optics")
        telescope = batoid.Optic.fromYaml("DECam.yaml", gpu=args.gpu)
        pm = 'PM'
    else:
        print("Tracing through HSC optics")
        telescope = batoid.Optic.fromYaml("HSC.yaml", gpu=args.gpu)
        pm = 'PM'

    building = []
    for _ in range(args.nrepeat):
        t0 = time.time()
        if args.gpu:
            rays = batoid.RayVector2.asPolar(
                optic=telescope,
                wavelength=620e-9,
                theta_x=np.deg2rad(0.3),
                theta_y=np.deg2rad(0.3),
                inner=0.5*telescope.pupilSize*telescope.pupilObscuration,
                nrad=args.nside, naz=args.nside
            )
        else:
            rays = batoid.RayVector.asPolar(
                optic=telescope,
                wavelength=620e-9,
                theta_x=np.deg2rad(0.3),
                theta_y=np.deg2rad(0.3),
                inner=0.5*telescope.pupilSize*telescope.pupilObscuration,
                nrad=args.nside, naz=args.nside
            )
        t1 = time.time()
        building.append(t1-t0)
    building = np.array(building)

    nrays = len(rays)
    print("Tracing {:_d} rays.".format(nrays))
    print()
    if args.nrepeat > 1:
        print("Ray generation: {:_} +/- {:_} rays per second".format(
            int(np.mean(nrays/building)),
            int(np.std(nrays/building)/np.sqrt(args.nrepeat))
        ))
    else:
        print("Ray generation: {:_} rays per second".format(int(nrays/building[0])))
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
            coefs = orig.coefs if hasattr(orig, 'coefs') else []
            telescope[pm].surface = batoid.ExtendedAsphere2(
                orig.R,
                orig.conic,
                coefs,
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

    copying = []
    tracing = []
    if args.immutable:
        print("Immutable trace")
        t0 = time.time()

        for _ in range(args.nrepeat):
            t1 = time.time()
            rays_in = batoid.RayVector(rays)
            t2 = time.time()
            rays_out, _ = telescope.trace(rays_in)
            t3 = time.time()
            copying.append(t2-t1)
            tracing.append(t3-t2)
        t4 = time.time()
    else:
        print("Trace in place")
        t0 = time.time()
        copying = []
        tracing = []

        for _ in range(args.nrepeat):
            t1 = time.time()
            rays_out = rays.copy()
            t2 = time.time()
            telescope.traceInPlace(rays_out)
            rays_out.r # force copy back to host
            rays_out.v
            rays_out.t
            rays_out.vignetted
            rays_out.failed
            t3 = time.time()
            copying.append(t2-t1)
            tracing.append(t3-t2)
        t4 = time.time()
    copying = np.array(copying)
    tracing = np.array(tracing)

    if args.nrepeat > 1:
        print("copying: {:_} +/- {:_} rays per second".format(
            int(np.mean(nrays/copying)),
            int(np.std(nrays/copying)/np.sqrt(args.nrepeat))
        ))
        print("tracing: {:_} +/- {:_} rays per second".format(
            int(np.mean(nrays/tracing)),
            int(np.std(nrays/tracing)/np.sqrt(args.nrepeat))
        ))
    else:
        print("copying: {:_} rays per second".format(int(nrays/copying)))
        print("tracing: {:_} rays per second".format(int(nrays/tracing)))
    print("overall:")
    print("{:_} rays per second".format(int(nrays*args.nrepeat/(t4-t0))))

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
