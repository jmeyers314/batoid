import os
import numpy as np
import batoid
import yaml
from test_helpers import timer


@timer
def test_huygens_psf():
    try:
        import galsim
    except ImportError:
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
            ],
            pupilSize=diam,
            dist=10.0,
            inMedium=batoid.ConstMedium(1.0)
        )

        airy_size = 1.22*500e-9/diam * 206265
        print()
        print("Airy radius: {:4.2f} arcsec".format(airy_size))

        # Start with the HuygensPSF
        npix = 96
        size = 3.0
        dsize = size/npix  # arcsec
        dsize_X = dsize*focalLength/206265  # meters

        psf = batoid.huygensPSF(telescope, 0.0, 0.0, 500e-9, nx=npix, dx=dsize_X, dy=dsize_X)
        psf.array /= np.max(psf.array)

        scale = np.sqrt(np.abs(np.linalg.det(psf.primitiveVectors)))  # meters
        scale *= 206265/focalLength  # arcsec
        obj = galsim.Airy(lam=500, diam=diam, obscuration=obscuration)
        # Need to shift by half a pixel.
        obj = obj.shift(scale/2, scale/2)
        im = obj.drawImage(nx=npix, ny=npix, scale=scale, method='no_pixel')
        arr = im.array/np.max(im.array)
        gs_mom = galsim.hsm.FindAdaptiveMom(im)

        psfim = galsim.Image(psf.array)
        jt_mom = galsim.hsm.FindAdaptiveMom(psfim)

        print("GalSim shape: ", gs_mom.observed_shape)
        print("batoid shape: ", jt_mom.observed_shape)
        print("GalSim centroid:  ", gs_mom.moments_centroid)
        print("batoid centroid:  ", jt_mom.moments_centroid)
        print("GalSim size: ", gs_mom.moments_sigma)
        print("batoid size: ", jt_mom.moments_sigma)
        print("GalSim rho4: ", gs_mom.moments_rho4)
        print("batoid rho4: ", jt_mom.moments_rho4)

        np.testing.assert_allclose(gs_mom.observed_shape.g1, jt_mom.observed_shape.g1, rtol=0.0, atol=3e-3)
        np.testing.assert_allclose(gs_mom.observed_shape.g2, jt_mom.observed_shape.g2, rtol=0.0, atol=3e-3)
        np.testing.assert_allclose(gs_mom.moments_centroid.x, jt_mom.moments_centroid.x, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(gs_mom.moments_centroid.y, jt_mom.moments_centroid.y, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(gs_mom.moments_sigma, jt_mom.moments_sigma, rtol=1e-2) # why not better?!
        np.testing.assert_allclose(gs_mom.moments_rho4, jt_mom.moments_rho4, rtol=2e-2)

        if __name__ == '__main__':
            size = scale*npix
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(15, 4))
            ax1 = fig.add_subplot(131)
            im1 = ax1.imshow(np.log10(arr), extent=np.r_[-1,1,-1,1]*size/2, vmin=-7, vmax=0)
            plt.colorbar(im1, ax=ax1, label=r'$\log_{10}$ flux')
            ax1.set_title('GalSim')
            ax1.set_xlabel("arcsec")
            ax1.set_ylabel("arcsec")

            sizeX = dsize_X * npix * 1e6  # microns
            ax2 = fig.add_subplot(132)
            im2 = ax2.imshow(np.log10(psf.array), extent=np.r_[-1,1,-1,1]*sizeX/2, vmin=-7, vmax=0)
            plt.colorbar(im2, ax=ax2, label=r'$\log_{10}$ flux')
            ax2.set_title('batoid')
            ax2.set_xlabel(r"$\mu m$")
            ax2.set_ylabel(r"$\mu m$")

            ax3 = fig.add_subplot(133)
            im3 = ax3.imshow((psf.array-arr)/np.max(arr), vmin=-0.01, vmax=0.01, cmap='seismic')
            plt.colorbar(im3, ax=ax3, label="(batoid-GalSim)/max(GalSim)")
            ax3.set_title('resid')
            ax3.set_xlabel(r"$\mu m$")
            ax3.set_ylabel(r"$\mu m$")

            fig.tight_layout()

            plt.show()


@timer
def test_lsst_psf():
    # Just testing that doesn't crash for the moment
    fn = os.path.join(batoid.datadir, "LSST", "LSST_r.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    stampSize = 0.5 # arcsec
    nx = 64
    focalLength = 1.234*8.36 # meters

    if __name__ == '__main__':
        thetas = [0.0, 1200.0, 3600.0, 6300.0] # arcsec
    else:
        thetas = [6300.0]
    for theta in thetas:
        print(theta/3600.0)
        dirCos = batoid.utils.gnomicToDirCos(0.0, theta/206265)
        rays = batoid.circularGrid(10.0, 4.2, 2.55,
                                   dirCos[0], dirCos[1], -dirCos[2],
                                   10, 100, 620e-9, 1.0, batoid.Air())
        telescope.traceInPlace(rays)
        rays.trimVignettedInPlace()
        xs = rays.x - np.mean(rays.x)
        ys = rays.y - np.mean(rays.y)

        xs *= 206265/focalLength
        ys *= 206265/focalLength

        # Need to add half-pixel offset
        xs += stampSize/nx/2
        ys += stampSize/nx/2

        dx = stampSize/nx * focalLength/206265 # meters

        psf = batoid.huygensPSF(telescope, 0.0, theta/206265, 620e-9, nx=64, dx=dx, dy=dx)

        if __name__ == '__main__':
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.imshow(psf.array, extent=np.r_[-1,1,-1,1]*stampSize/2)
            ax.scatter(xs, ys, s=5, c='r', alpha=0.5)
            ax.set_title("LSST PSF field={:5.2f}".format(theta/3600.0))
            ax.set_xlabel("arcsec")
            ax.set_ylabel("arcsec")

            fig.tight_layout()
            plt.show()


@timer
def test_hsc_psf():
    # Just testing that doesn't crash for the moment
    fn = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    stampSize = 0.75  # arcsec
    nx = 64
    focalLength = 15.0  # guess

    if __name__ == '__main__':
        thetas = [0.0, 1350.0, 2700.0] # arcsec
    else:
        thetas = [2700.0]
    for theta in thetas:
        print(theta/3600.0)
        dirCos = batoid.utils.gnomicToDirCos(0.0, theta/206265)
        rays = batoid.circularGrid(20.0, 4.1, 0.9,
                                   dirCos[0], dirCos[1], -dirCos[2],
                                   10, 100, 620e-9, 1.0, batoid.ConstMedium(1.0))
        telescope.traceInPlace(rays)
        rays.trimVignettedInPlace()
        xs = rays.x - np.mean(rays.x)
        ys = rays.y - np.mean(rays.y)

        xs *= 206265/focalLength  # meters to arcsec
        ys *= 206265/focalLength

        # Need to add half-pixel offset
        xs += stampSize/nx/2
        ys += stampSize/nx/2

        dx = stampSize/nx * focalLength/206265 # meters

        psf = batoid.huygensPSF(telescope, 0.0, theta/206265, 620e-9, nx=nx, dx=dx, dy=dx)

        if __name__ == '__main__':
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.imshow(psf.array, extent=np.r_[-1,1,-1,1]*stampSize/2)
            ax.scatter(xs, ys, s=5, c='r', alpha=0.5)
            ax.set_title("HSC PSF field={:5.2f}".format(theta/3600.0))
            ax.set_xlabel("arcsec")
            ax.set_ylabel("arcsec")

            fig.tight_layout()
            plt.show()


@timer
def test_decam_psf():
    # Just testing that doesn't crash for the moment
    fn = os.path.join(batoid.datadir, "DECam", "DECam.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    stampSize = 1.0  # arcsec
    nx = 64
    focalLength = 10.0  # guess

    if __name__ == '__main__':
        thetas = [0.0, 1800.0, 3960.0] # arcsec
    else:
        thetas = [3960.0]
    for theta in thetas:
        print(theta/3600.0)
        dirCos = batoid.utils.gnomicToDirCos(0.0, theta/206265)
        rays = batoid.circularGrid(10.0, 1.95, 0.5,
                                   dirCos[0], dirCos[1], -dirCos[2],
                                   10, 100, 620e-9, 1.0, batoid.Air())
        telescope.traceInPlace(rays)
        rays.trimVignettedInPlace()
        xs = rays.x - np.mean(rays.x)
        ys = rays.y - np.mean(rays.y)

        xs *= 206265/focalLength  # meters to arcsec
        ys *= 206265/focalLength

        # Need to add half-pixel offset
        xs += stampSize/nx/2
        ys += stampSize/nx/2

        dx = stampSize/nx * focalLength/206265 # meters

        psf = batoid.huygensPSF(telescope, 0.0, theta/206265, 620e-9, nx=nx, dx=dx, dy=dx)

        if __name__ == '__main__':
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.imshow(psf.array, extent=np.r_[-1,1,-1,1]*stampSize/2)
            ax.scatter(xs, ys, s=5, c='r', alpha=0.5)
            ax.set_title("DECam PSF field={:5.2f}".format(theta/3600.0))
            ax.set_xlabel("arcsec")
            ax.set_ylabel("arcsec")

            fig.tight_layout()
            plt.show()


if __name__ == '__main__':
    test_huygens_psf()
    test_lsst_psf()
    test_hsc_psf()
    test_decam_psf()
