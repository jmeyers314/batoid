import batoid
import numpy as np
from test_helpers import timer
import time


@timer
def parallel_trace_timing(args):
    print("Using nrad of {:_d}".format(args.nrad))

    if args.lsst:
        print("Tracing through LSST optics")
        telescope = batoid.Optic.fromYaml("LSST_r.yaml")
        pm = 'M1'
    elif args.decam:
        print("Tracing through DECam optics")
        telescope = batoid.Optic.fromYaml("DECam.yaml")
        pm = 'PM'
    else:
        print("Tracing through HSC optics")
        telescope = batoid.Optic.fromYaml("HSC.yaml")
        pm = 'PM'

    building = []
    for _ in range(args.nrepeat):
        t0 = time.time()
        rays = batoid.RayVector.asPolar(
            optic=telescope,
            wavelength=620e-9,
            theta_x=np.deg2rad(0.3),
            theta_y=np.deg2rad(0.3),
            inner=0.5*telescope.pupilSize*telescope.pupilObscuration,
            nrad=args.nrad, naz=int(2*np.pi*args.nrad)
        )
        t1 = time.time()
        building.append(t1-t0)
    building = np.array(building)

    nrays = len(rays)
    print("Tracing {:_d} rays.".format(nrays))
    print(f"Minimum CPU RAM: {2*nrays*74/1024**3:.2f} GB")
    print(f"Minimum GPU RAM: {nrays*74/1024**3:.2f} GB")
    print()
    print()
    if args.nrepeat > 1:
        print("Generating: {:_} +/- {:_} rays per second".format(
            int(np.mean(nrays/building)),
            int(np.std(nrays/building)/np.sqrt(args.nrepeat))
        ))
    else:
        print("Generating: {:_} rays per second".format(int(nrays/building[0])))

    # Optionally perturb the primary mirror using Zernike polynomial
    if args.perturbZ != 0:
        orig = telescope[pm].surface
        coefs = np.random.normal(size=args.perturbZ+1)*1e-6 # micron perturbations
        perturbation = batoid.Zernike(coefs, R_outer=telescope.pupilSize)
        telescope[pm].surface = batoid.Sum([orig, perturbation])

    # Optionally perturb primary mirror using bicubic spline
    if args.perturbBC != 0:
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
    syncing = []
    tracing = []
    overall = []

    for _ in range(args.nrepeat):
        t1 = time.time()
        rays_out = rays.copy()
        t2 = time.time()
        rays_out._syncToDevice()
        t3 = time.time()
        telescope.trace(rays_out)
        rays_out.r # force copy back to host if doing gpu
        rays_out.v
        rays_out.t
        rays_out.flux
        rays_out.vignetted
        rays_out.failed
        t4 = time.time()
        copying.append(t2-t1)
        syncing.append(t3-t2)
        tracing.append(t4-t3)
        overall.append(t4-t1)
    copying = np.array(copying)
    syncing = np.array(syncing)
    tracing = np.array(tracing)
    overall = np.array(overall)

    if args.nrepeat > 1:
        print()
        print("copying: {:_} +/- {:_} rays per second".format(
            int(np.mean(nrays/copying)),
            int(np.std(nrays/copying)/np.sqrt(args.nrepeat))
        ))
        print()
        print("syncing: {:_} +/- {:_} rays per second".format(
            int(np.mean(nrays/syncing)),
            int(np.std(nrays/syncing)/np.sqrt(args.nrepeat))
        ))
        print()
        print("tracing: {:_} +/- {:_} rays per second".format(
            int(np.mean(nrays/tracing)),
            int(np.std(nrays/tracing)/np.sqrt(args.nrepeat))
        ))
        print()
        print("overall")
        print("-------")
        print("{:_} +/- {:_} rays per second".format(
            int(np.mean(nrays/overall)),
            int(np.std(nrays/overall)/np.sqrt(args.nrepeat))
        ))
        print()
    else:
        print()
        print("copying: {:_} rays per second".format(int(nrays/copying)))
        print()
        print("syncing: {:_} rays per second".format(int(nrays/syncing)))
        print()
        print("tracing: {:_} rays per second".format(int(nrays/tracing)))
        print()
        print("overall")
        print("-------")
        print("{:_} rays per second".format(int(nrays/overall)))
        print()

    if args.plot or args.show:
        import matplotlib.pyplot as plt
        w = ~rays_out.vignetted
        x = rays_out.x[w]
        y = rays_out.y[w]
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
        if args.show:
            plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nrad", type=int, default=250)
    parser.add_argument("--nrepeat", type=int, default=1)
    parser.add_argument("--perturbZ", type=int, default=0)
    parser.add_argument("--perturbBC", type=float, default=0.0)
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--lsst", action='store_true')
    parser.add_argument("--decam", action='store_true')
    args = parser.parse_args()

    parallel_trace_timing(args)
