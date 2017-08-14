import os
import numpy as np
import batoid
from test_helpers import timer, isclose


@timer
def test_huygens_psf():
    try:
        import galsim
    except:
        print("Huygens PSF test requires GalSim")
        # Could do the integral directly without GalSim?
        return

    print("Testing HuygensPSF")
    # Just do a single parabolic mirror test
    focalLength = 1.5
    diam = 0.3
    R = 2*focalLength
    for obscuration in [0.0, 0.25, 0.5, 0.75]:
        surfaces = [
            dict(
                name='Mirror',
                inner=obscuration*diam/2,
                outer=diam/2,
                type='mirror',
                m0=1.0,
                m1=1.0,
                surface=batoid.Paraboloid(R, 0.0, Rin=obscuration*diam/2, Rout=diam/2)
            ),
            dict(
                name='Det',
                inner=0.0,
                outer=0.01,
                type='det',
                m0=1.0,
                m1=1.0,
                surface=batoid.Plane(focalLength, Rout=0.01)
            )
        ]
        telescope = batoid.Telescope(surfaces)

        airy_size = 1.22*500e-9/diam * 206265
        print()
        print("Airy radius: {:4.2f} arcsec".format(airy_size))

        size = 3.2
        npix = 128
        obj = galsim.OpticalPSF(diam=diam, lam=500, obscuration=obscuration)
        im = obj.drawImage(nx=npix, ny=npix, scale=size/npix, method='no_pixel')
        arr = im.array/im.array.sum()
        gs_mom = galsim.hsm.FindAdaptiveMom(im)

        rays = batoid.parallelRays(10, 0.15, 0.0, nradii=20, naz=100, wavelength=500e-9)
        traced_rays = telescope.trace(rays)
        traced_rays = batoid.RayVector([r for r in traced_rays if not r.isVignetted])

        xs = np.linspace(-size/2, size/2, npix) # arcsec
        xs /= (206265/focalLength) # meters
        xs, ys = np.meshgrid(xs, xs)
        xs += np.mean(traced_rays.x)
        ys += np.mean(traced_rays.y)

        psf = telescope.huygensPSF(rays=rays, xs=xs, ys=ys)
        psf = psf/psf.sum()

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
            im1 = ax1.imshow(np.log10(arr), extent=[-size/2, size/2, -size/2, size/2], vmin=-7, vmax=-1)
            plt.colorbar(im1, ax=ax1)
            ax1.set_title('GalSim')
            ax1.set_xlabel("arcsec")
            ax1.set_ylabel("arcsec")

            xs *= 1e6 # m -> micron
            ys *= 1e6
            rayx = traced_rays.x*1e6
            rayy = traced_rays.y*1e6
            ax2 = fig.add_subplot(132)
            im2 = ax2.imshow(np.log10(psf), extent=[xs.min(), xs.max(), ys.min(), ys.max()], vmin=-7, vmax=-1)
            plt.colorbar(im2, ax=ax2)
            ax2.scatter(rayx, rayy, s=1, c='r')
            ax2.set_xlim(xs.min(), xs.max())
            ax2.set_ylim(ys.min(), ys.max())
            ax2.set_title('batoid')
            ax2.set_xlabel("$\mu m$")
            ax2.set_ylabel("$\mu m$")

            ax3 = fig.add_subplot(133)
            im3 = ax3.imshow((psf-arr)/arr, vmin=-0.1, vmax=0.1)
            plt.colorbar(im3, ax=ax3)
            ax3.set_title('resid')
            ax3.set_xlabel("$\mu m$")
            ax3.set_ylabel("$\mu m$")

            fig.tight_layout()

            plt.show()


@timer
def test_lsst_psf():
    # Just testing that doesn't crash for the moment
    fn = os.path.join(batoid.datadir, "lsst", "LSST_r.yaml")
    telescope = batoid.Telescope.makeFromYAML(fn)

    if __name__ == '__main__':
        thetas = [0.0, 1200.0, 3600.0, 6300.0] # arcsec
    else:
        thetas = [6300.0]
    for theta in thetas:
        print(theta/3600.0)
        rays = batoid.parallelRays(10, 4.2, 2.55, theta_x=theta/206265, nradii=10, naz=100,
                                   wavelength=620e-9, medium=batoid.Air())
        traced_rays = telescope.trace(rays)
        traced_rays = batoid.RayVector([r for r in traced_rays if not r.isVignetted])

        nx = 64
        xs = np.linspace(-10e-6, 10e-6, nx) # 2 pixels wide
        ys = np.linspace(-10e-6, 10e-6, nx)
        xs += np.mean(traced_rays.x)
        ys += np.mean(traced_rays.y)
        xs, ys = np.meshgrid(xs, ys)

        psf = telescope.huygensPSF(xs=xs, ys=ys, rays=rays)

        if __name__ == '__main__':
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.imshow(psf, extent=np.r_[xs.min(), xs.max(), ys.min(), ys.max()]*1e6)
            ax.scatter(traced_rays.x*1e6, traced_rays.y*1e6, s=1, c='r')
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
    fn = os.path.join(batoid.datadir, "hsc", "HSC.yaml")
    telescope = batoid.Telescope.makeFromYAML(fn)

    if __name__ == '__main__':
        thetas = [0.0, 1350.0, 2700.0] # arcsec
    else:
        thetas = [2700.0]
    for theta in thetas:
        print(theta/3600.0)
        rays = batoid.parallelRays(10, 4.1, 0.75, theta_y=theta/206265, nradii=10, naz=100,
                                   wavelength=760e-9)
        traced_rays = telescope.trace(rays)
        traced_rays = batoid.RayVector([r for r in traced_rays if not r.isVignetted])

        nx = 64
        xs = np.linspace(-27.1e-6, 27.1e-6, nx) # 2 pixels wide
        ys = np.linspace(-27.1e-6, 27.1e-6, nx)
        xs += np.mean(traced_rays.x)
        ys += np.mean(traced_rays.y)
        xs, ys = np.meshgrid(xs, ys)

        psf = telescope.huygensPSF(xs=xs, ys=ys, rays=rays)

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
    fn = os.path.join(batoid.datadir, "decam", "DECam.yaml")
    telescope = batoid.Telescope.makeFromYAML(fn)

    if __name__ == '__main__':
        thetas = [0.0, 1800.0, 3960.0] # arcsec
    else:
        thetas = [3960.0]
    for theta in thetas:
        print(theta/3600.0)
        rays = batoid.parallelRays(10, 4.1, 0.75, theta_y=theta/206265, nradii=30, naz=200,
                                   wavelength=760e-9, medium=batoid.Air())
        traced_rays = telescope.trace(rays)
        traced_rays = batoid.RayVector([r for r in traced_rays if not r.isVignetted])

        nx = 64
        xs = np.linspace(-27.1e-6, 27.1e-6, nx)
        ys = np.linspace(-27.1e-6, 27.1e-6, nx)
        xs += np.mean(traced_rays.x)
        ys += np.mean(traced_rays.y)
        xs, ys = np.meshgrid(xs, ys)

        psf = telescope.huygensPSF(xs=xs, ys=ys, rays=rays)

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
