import os
import numpy as np
import batoid
import yaml
from test_helpers import timer, isclose


@timer
def test_huygens_psf():
    try:
        import galsim
    except:
        print("Huygens PSF test requires GalSim")
        # Could do the integral directly without GalSim?
        return

    if __name__ == '__main__':
        obscurations = [0.0, 0.25, 0.5, 0.75]
    else:
        obscurations = [0.25]

    print("Testing HuygensPSF")
    # Just do a single parabolic mirror test
    focalLength = 1.5
    diam = 0.3
    R = 2*focalLength
    for obscuration in obscurations:
        telescope = batoid.CompoundOptic(
            items = [
                batoid.Mirror(
                    batoid.Paraboloid(R),
                    name="Mirror",
                    obscuration=batoid.ObscNegation(
                        batoid.ObscAnnulus(0.5*obscuration*diam, 0.5*diam)
                    )
                ),
                batoid.Detector(
                    batoid.Plane(),
                    name="detector",
                    coordSys=batoid.CoordSys().shiftGlobal([0,0,focalLength])
                )
            ]
        )

        airy_size = 1.22*500e-9/diam * 206265
        print()
        print("Airy radius: {:4.2f} arcsec".format(airy_size))

        size = 3.2
        npix = 128
        obj = galsim.Airy(lam=500, diam=diam, obscuration=obscuration)
        im = obj.drawImage(nx=npix, ny=npix, scale=size/npix, method='no_pixel')
        arr = im.array/np.max(im.array)
        gs_mom = galsim.hsm.FindAdaptiveMom(im)

        rays = batoid.circularGrid(10.0, 0.5*diam, 0.5*diam*obscuration, 0.0, 0.0, -1.0, 50, 200, 500e-9, 1.0)
        traced_rays = batoid.RayVector(rays)
        telescope.traceInPlace(traced_rays)
        batoid._batoid.trimVignettedInPlace(traced_rays)

        xs = np.linspace(-size/2, size/2, npix) # arcsec
        xs /= (206265/focalLength) # meters
        xs, ys = np.meshgrid(xs, xs)
        xs += np.mean(traced_rays.x)
        ys += np.mean(traced_rays.y)

        psf = batoid.huygensPSF(telescope, xs=xs, ys=ys, rays=rays)
        psf = psf/np.max(psf)

        psfim = galsim.Image(psf)
        jt_mom = galsim.hsm.FindAdaptiveMom(psfim)

        print("GalSim shape: ", gs_mom.observed_shape)
        print("batoid shape: ", jt_mom.observed_shape)
        print("GalSim centroid:  ", gs_mom.moments_centroid)
        print("batoid centroid:  ", jt_mom.moments_centroid)
        print("GalSim size: ", gs_mom.moments_sigma)
        print("batoid size: ", jt_mom.moments_sigma)
        print("GalSim rho4: ", gs_mom.moments_rho4)
        print("batoid rho4: ", jt_mom.moments_rho4)

        assert isclose(gs_mom.observed_shape.g1, jt_mom.observed_shape.g1, abs_tol=3e-3, rel_tol=0.0)
        assert isclose(gs_mom.observed_shape.g2, jt_mom.observed_shape.g2, abs_tol=3e-3, rel_tol=0.0)
        assert isclose(gs_mom.moments_centroid.x, jt_mom.moments_centroid.x, abs_tol=1e-9, rel_tol=0.0)
        assert isclose(gs_mom.moments_centroid.y, jt_mom.moments_centroid.y, abs_tol=1e-9, rel_tol=0.0)
        assert isclose(gs_mom.moments_sigma, jt_mom.moments_sigma, rel_tol=1e-1) # why not better?!
        assert isclose(gs_mom.moments_rho4, jt_mom.moments_rho4, rel_tol=2e-2)

        if __name__ == '__main__':
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(15, 4))
            ax1 = fig.add_subplot(131)
            im1 = ax1.imshow(np.log10(arr), extent=[-size/2, size/2, -size/2, size/2], vmin=-7, vmax=0)
            plt.colorbar(im1, ax=ax1, label='$\log_{10}$ flux')
            ax1.set_title('GalSim')
            ax1.set_xlabel("arcsec")
            ax1.set_ylabel("arcsec")

            xs *= 1e6 # m -> micron
            ys *= 1e6
            rayx = traced_rays.x*1e6
            rayy = traced_rays.y*1e6
            ax2 = fig.add_subplot(132)
            im2 = ax2.imshow(np.log10(psf), extent=[xs.min(), xs.max(), ys.min(), ys.max()], vmin=-7, vmax=0)
            plt.colorbar(im2, ax=ax2, label='$\log_{10}$ flux')
            ax2.scatter(rayx, rayy, s=1, c='r')
            ax2.set_xlim(xs.min(), xs.max())
            ax2.set_ylim(ys.min(), ys.max())
            ax2.set_title('batoid')
            ax2.set_xlabel("$\mu m$")
            ax2.set_ylabel("$\mu m$")

            ax3 = fig.add_subplot(133)
            im3 = ax3.imshow((psf-arr)/np.max(arr), vmin=-0.01, vmax=0.01, cmap='seismic')
            plt.colorbar(im3, ax=ax3, label="(batoid-GalSim)/max(GalSim)")
            ax3.set_title('resid')
            ax3.set_xlabel("$\mu m$")
            ax3.set_ylabel("$\mu m$")

            fig.tight_layout()

            plt.show()


@timer
def test_lsst_psf():
    # Just testing that doesn't crash for the moment
    fn = os.path.join(batoid.datadir, "LSST", "LSST_r.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # telescope.draw(ax)
    # plt.show()
    # import sys; sys.exit()

    if __name__ == '__main__':
        thetas = [0.0, 1200.0, 3600.0, 6300.0] # arcsec
    else:
        thetas = [6300.0]
    for theta in thetas:
        print(theta/3600.0)
        rays = batoid.circularGrid(10.0, 4.2, 2.55,
                                   np.sin(theta/206265), 0.0, -1.0,
                                   10, 100, 620e-9, batoid.Air())
        traced_rays = batoid.RayVector(rays)
        telescope.traceInPlace(traced_rays)
        batoid._batoid.trimVignettedInPlace(traced_rays)

        nx = 64
        xs = np.linspace(-10e-6, 10e-6, nx) # 2 pixels wide
        ys = np.linspace(-10e-6, 10e-6, nx)
        xs += np.mean(traced_rays.x)
        ys += np.mean(traced_rays.y)
        xs, ys = np.meshgrid(xs, ys)

        psf = batoid.huygensPSF(telescope, xs=xs, ys=ys, rays=rays)

        if __name__ == '__main__':
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.imshow(psf, extent=np.r_[xs.min(), xs.max(), ys.min(), ys.max()]*1e6)
            ax.scatter(traced_rays.x*1e6, traced_rays.y*1e6, s=1, c='r', alpha=0.25)
            ax.set_xlim(xs.min()*1e6, xs.max()*1e6)
            ax.set_ylim(ys.min()*1e6, ys.max()*1e6)
            ax.set_title("LSST PSF field={:5.2f}".format(theta/3600.0))
            ax.set_xlabel("$\mu m$")
            ax.set_ylabel("$\mu m$")

            fig.tight_layout()
            plt.show()


@timer
def test_hsc_psf():
    # Just testing that doesn't crash for the moment
    fn = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    if __name__ == '__main__':
        thetas = [0.0, 1350.0, 2700.0] # arcsec
    else:
        thetas = [2700.0]
    for theta in thetas:
        print(theta/3600.0)
        rays = batoid.circularGrid(10.0, 4.2, 2.55,
                                   np.sin(theta/206265), 0.0, -1.0,
                                   10, 100, 620e-9, 1.0)
        traced_rays = batoid.RayVector(rays)
        telescope.traceInPlace(traced_rays)
        batoid._batoid.trimVignettedInPlace(traced_rays)

        nx = 64
        xs = np.linspace(-27.1e-6, 27.1e-6, nx) # 2 pixels wide
        ys = np.linspace(-27.1e-6, 27.1e-6, nx)
        xs += np.mean(traced_rays.x)
        ys += np.mean(traced_rays.y)
        xs, ys = np.meshgrid(xs, ys)

        psf = batoid.huygensPSF(telescope, xs=xs, ys=ys, rays=rays)

        if __name__ == '__main__':
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.imshow(psf, extent=np.r_[xs.min(), xs.max(), ys.min(), ys.max()]*1e6)
            ax.scatter(traced_rays.x*1e6, traced_rays.y*1e6, s=1, c='r')
            ax.set_xlim(xs.min()*1e6, xs.max()*1e6)
            ax.set_ylim(ys.min()*1e6, ys.max()*1e6)
            ax.set_title("HSC PSF field={:5.2f}".format(theta/3600.0))
            ax.set_xlabel("$\mu m$")
            ax.set_ylabel("$\mu m$")

            fig.tight_layout()
            plt.show()


@timer
def test_decam_psf():
    # Just testing that doesn't crash for the moment
    fn = os.path.join(batoid.datadir, "DECam", "DECam.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # telescope.draw(ax)
    # plt.show()
    # import sys; sys.exit()

    if __name__ == '__main__':
        thetas = [0.0, 1800.0, 3960.0] # arcsec
    else:
        thetas = [3960.0]
    for theta in thetas:
        print(theta/3600.0)
        rays = batoid.circularGrid(10.0, 1.95, 0.5,
                                   np.sin(theta/206265), 0.0, -1.0,
                                   30, 200, 760e-9, batoid.Air())
        traced_rays = batoid.RayVector(rays)
        telescope.traceInPlace(traced_rays)
        batoid._batoid.trimVignettedInPlace(traced_rays)

        nx = 64
        xs = np.linspace(-27.1e-6, 27.1e-6, nx)
        ys = np.linspace(-27.1e-6, 27.1e-6, nx)
        xs += np.mean(traced_rays.x)
        ys += np.mean(traced_rays.y)
        xs, ys = np.meshgrid(xs, ys)

        psf = batoid.huygensPSF(telescope, xs=xs, ys=ys, rays=rays)

        if __name__ == '__main__':
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.imshow(psf, extent=np.r_[xs.min(), xs.max(), ys.min(), ys.max()]*1e6)
            ax.scatter(traced_rays.x*1e6, traced_rays.y*1e6, s=1, c='r')
            ax.set_xlim(xs.min()*1e6, xs.max()*1e6)
            ax.set_ylim(ys.min()*1e6, ys.max()*1e6)
            ax.set_title("DECam PSF field={:5.2f}".format(theta/3600.0))
            ax.set_xlabel("$\mu m$")
            ax.set_ylabel("$\mu m$")

            fig.tight_layout()
            plt.show()


if __name__ == '__main__':
    test_huygens_psf()
    test_lsst_psf()
    test_hsc_psf()
    test_decam_psf()
