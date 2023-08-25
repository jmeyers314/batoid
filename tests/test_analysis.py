import numpy as np
import galsim
import batoid
from test_helpers import timer, init_gpu


@timer
def test_zernikeGQ():
    if __name__ == '__main__':
        nx=1024
        rings=10
        tol=1e-4
    else:
        nx=128
        rings=5
        tol=1e-3
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    telescope.clearObscuration()
    telescope['LSST.M1'].obscuration = batoid.ObscNegation(
        batoid.ObscCircle(4.18)
    )
    zSquare = batoid.zernike(
        telescope, 0.0, 0.0, 625e-9,
        nx=nx, jmax=28, reference='chief'
    )
    zGQ = batoid.zernikeGQ(
        telescope, 0.0, 0.0, 625e-9,
        rings=rings, jmax=28, reference='chief'
    )

    np.testing.assert_allclose(
        zSquare, zGQ, rtol=0, atol=tol
    )

    # Repeat with annular Zernikes
    telescope['LSST.M1'].obscuration = batoid.ObscNegation(
        batoid.ObscAnnulus(0.61*4.18, 4.18)
    )
    zSquare = batoid.zernike(
        telescope, 0.0, 0.0, 625e-9,
        nx=nx, jmax=28, reference='chief', eps=0.61
    )
    zGQ = batoid.zernikeGQ(
        telescope, 0.0, 0.0, 625e-9,
        rings=rings, jmax=28, reference='chief', eps=0.61
    )

    np.testing.assert_allclose(
        zSquare, zGQ, rtol=0, atol=tol
    )

    # Try off-axis
    zSquare = batoid.zernike(
        telescope, np.deg2rad(0.2), np.deg2rad(0.1), 625e-9,
        nx=nx, jmax=28, reference='chief', eps=0.61
    )
    zGQ = batoid.zernikeGQ(
        telescope, np.deg2rad(0.2), np.deg2rad(0.1), 625e-9,
        rings=rings, jmax=28, reference='chief', eps=0.61
    )

    np.testing.assert_allclose(
        zSquare, zGQ, rtol=0, atol=tol
    )

    # Try reference == mean
    # Try off-axis
    zSquare = batoid.zernike(
        telescope, np.deg2rad(0.2), np.deg2rad(0.1), 625e-9,
        nx=nx, jmax=28, reference='mean', eps=0.61
    )
    zGQ = batoid.zernikeGQ(
        telescope, np.deg2rad(0.2), np.deg2rad(0.1), 625e-9,
        rings=rings, jmax=28, reference='mean', eps=0.61
    )
    # Z1-3 less reliable, but mostly uninteresting anyway...
    np.testing.assert_allclose(
        zSquare[4:], zGQ[4:], rtol=0, atol=tol
    )


@timer
def test_huygensPSF():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")

    # Test that we can infer dy from dx properly
    psf1 = batoid.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=10e-6,
        reference='mean'
    )
    psf2 = batoid.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=10e-6,
        dy=10e-6,
        reference='mean'
    )
    assert np.array_equal(psf1.primitiveVectors, psf2.primitiveVectors)
    np.testing.assert_allclose(psf1.array, psf2.array, rtol=1e-14, atol=1e-15)

    # Test vector vs scalar dx,dy
    psf1 = batoid.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=[10e-6, 0],
        dy=[0, 11e-6],
        reference='mean'
    )
    psf2 = batoid.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=10e-6,
        dy=11e-6,
        reference='mean'
    )
    assert np.array_equal(psf1.primitiveVectors, psf2.primitiveVectors)
    np.testing.assert_allclose(psf1.array, psf2.array, rtol=1e-14, atol=1e-15)

    # Should still work with reference = 'chief'
    psf3 = batoid.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=[10e-6, 0],
        dy=[0, 11e-6],
        reference='chief'
    )
    psf4 = batoid.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=64,
        nxOut=32,
        dx=10e-6,
        dy=11e-6,
        reference='chief'
    )
    assert np.array_equal(psf1.primitiveVectors, psf3.primitiveVectors)
    assert not np.allclose(psf1.array, psf3.array, rtol=1e-14, atol=1e-15)

    assert np.array_equal(psf3.primitiveVectors, psf4.primitiveVectors)
    np.testing.assert_allclose(psf3.array, psf4.array, rtol=1e-14, atol=1e-15)

    # And just cover nx odd
    batoid.huygensPSF(
        telescope,
        np.deg2rad(0.1), np.deg2rad(0.1),
        620e-9,
        nx=63,
    )


@timer
def test_doubleZernike():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    dz = batoid.doubleZernike(
        telescope,
        np.deg2rad(1.75),
        625e-9,
        10,
        kmax=28,
        jmax=22
    )

    # Now evaluate DZ a few places and compare with zernikeGQ
    size = 20
    js = np.random.randint(4, 22, size=size)
    thr = np.deg2rad(np.sqrt(np.random.uniform(0, 1.75**2, size=size)))
    thth = np.random.uniform(0, 2*np.pi, size=size)
    thx = thr*np.cos(thth)
    thy = thr*np.sin(thth)

    for j in js:
        Z = galsim.zernike.Zernike(
            dz[:,j],
            R_inner=0.0,
            R_outer=np.deg2rad(1.75)
        )
        for thx_, thy_ in zip(thx, thy):
            zGQ = batoid.zernikeGQ(
                telescope, thx_, thy_,
                625e-9,
                jmax=22
            )
            np.testing.assert_allclose(Z(thx_, thy_), zGQ[j], rtol=0, atol=1e-4)

    # Check that we get similar results with different number of rings/spokes
    dz2 = batoid.doubleZernike(
        telescope,
        np.deg2rad(1.75),
        625e-9,
        rings=12,
        spokes=29,
        kmax=28,
        jmax=22
    )
    np.testing.assert_allclose(dz, dz2, rtol=0, atol=1e-2)


@timer
def test_huygens_paraboloid(plot=False):
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
                    coordSys=batoid.CoordSys(origin=[0, 0, focalLength])
                )
            ],
            pupilSize=diam,
            backDist=10.0,
            inMedium=batoid.ConstMedium(1.0)
        )

        airy_size = 1.22*500e-9/diam * 206265
        print()
        print("Airy radius: {:4.2f} arcsec".format(airy_size))

        # Start with the HuygensPSF
        npix = 96
        size = 3.0 # arcsec
        dsize = size/npix
        dsize_X = dsize*focalLength/206265  # meters

        psf = batoid.huygensPSF(
            telescope, 0.0, 0.0, 500e-9,
            nx=npix, dx=dsize_X, dy=dsize_X
        )
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

        np.testing.assert_allclose(
            gs_mom.observed_shape.g1,
            jt_mom.observed_shape.g1,
            rtol=0.0, atol=3e-3
        )
        np.testing.assert_allclose(
            gs_mom.observed_shape.g2,
            jt_mom.observed_shape.g2,
            rtol=0.0, atol=3e-3
        )
        np.testing.assert_allclose(
            gs_mom.moments_centroid.x,
            jt_mom.moments_centroid.x,
            rtol=0.0, atol=1e-9
        )
        np.testing.assert_allclose(
            gs_mom.moments_centroid.y,
            jt_mom.moments_centroid.y,
            rtol=0.0, atol=1e-9
        )
        np.testing.assert_allclose(
            gs_mom.moments_sigma,
            jt_mom.moments_sigma,
            rtol=1e-2  # why not better?!
        )
        np.testing.assert_allclose(
            gs_mom.moments_rho4,
            jt_mom.moments_rho4,
            rtol=2e-2
        )

        if plot:
            size = scale*npix
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(15, 4))
            ax1 = fig.add_subplot(131)
            im1 = ax1.imshow(
                np.log10(arr),
                extent=np.r_[-1,1,-1,1]*size/2,
                vmin=-7, vmax=0
            )
            plt.colorbar(im1, ax=ax1, label=r'$\log_{10}$ flux')
            ax1.set_title('GalSim')
            ax1.set_xlabel("arcsec")
            ax1.set_ylabel("arcsec")

            sizeX = dsize_X * npix * 1e6  # microns
            ax2 = fig.add_subplot(132)
            im2 = ax2.imshow(
                np.log10(psf.array),
                extent=np.r_[-1,1,-1,1]*sizeX/2,
                vmin=-7, vmax=0
            )
            plt.colorbar(im2, ax=ax2, label=r'$\log_{10}$ flux')
            ax2.set_title('batoid')
            ax2.set_xlabel(r"$\mu m$")
            ax2.set_ylabel(r"$\mu m$")

            ax3 = fig.add_subplot(133)
            im3 = ax3.imshow(
                (psf.array-arr)/np.max(arr),
                vmin=-0.01, vmax=0.01,
                cmap='seismic'
            )
            plt.colorbar(im3, ax=ax3, label="(batoid-GalSim)/max(GalSim)")
            ax3.set_title('resid')
            ax3.set_xlabel(r"$\mu m$")
            ax3.set_ylabel(r"$\mu m$")

            fig.tight_layout()

            plt.show()


@timer
def test_transverse_aberrations():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    thx = np.deg2rad(1.5)
    thy = np.deg2rad(0.9)
    wavelength = 622e-9
    focal_length = 10.31

    zTA = batoid.zernikeTA(
        telescope, thx, thy, wavelength,
        nrad=20, naz=120,
        jmax=66, eps=0.61,
        focal_length=focal_length
    )
    zTA = galsim.zernike.Zernike(
        zTA,
        R_outer=4.18, R_inner=4.18*0.61,
    )
    zX, zY = batoid.zernikeXYAberrations(
        telescope, thx, thy, wavelength,
        nrad=20, naz=120,
        jmax=55, eps=0.61,
    )
    zX = galsim.zernike.Zernike(
        zX,
        R_outer=4.18, R_inner=4.18*0.61,
    )
    zY = galsim.zernike.Zernike(
        zY,
        R_outer=4.18, R_inner=4.18*0.61,
    )
    u = np.linspace(-4.18, 4.18, 50)
    u, v = np.meshgrid(u, u)
    r = np.hypot(u, v)
    w = r < 4.18
    w &= r > 2.558
    u = u[w]
    v = v[w]

    np.testing.assert_allclose(
        -zTA.gradX(u, v)*wavelength*focal_length,
        zX(u, v),
        atol=5e-7, rtol=0  # 0.5 micron = 1/20 pixel
    )

    np.testing.assert_allclose(
        -zTA.gradY(u, v)*wavelength*focal_length,
        zY(u, v),
        atol=5e-7, rtol=0  # 0.5 micron = 1/20 pixel
    )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()

    init_gpu()
    test_zernikeGQ()
    test_huygensPSF()
    test_doubleZernike()
    test_huygens_paraboloid(args.plot)
    test_transverse_aberrations()
