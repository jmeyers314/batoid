# Compare ray by ray tracing to Zemax
import os

import pytest
import galsim
import numpy as np
from scipy.optimize import least_squares

import batoid
from test_helpers import timer, init_gpu

directory = os.path.dirname(__file__)


@timer
def test_HSC_trace():
    telescope = batoid.Optic.fromYaml("HSC_old.yaml")

    # Zemax has a number of virtual surfaces that we don't trace in batoid.
    # Also, the HSC.yaml above includes Baffle surfaces not in Zemax.  The
    # following lists select out the surfaces in common to both models.
    HSC_surfaces = [
        3, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 24, 25, 28, 29,
        31
    ]
    surface_names = ['PM', 'G1_entrance', 'G1_exit', 'G2_entrance', 'G2_exit',
                     'ADC1_entrance', 'ADC1_exit', 'ADC2_entrance', 'ADC2_exit',
                     'G3_entrance', 'G3_exit', 'G4_entrance', 'G4_exit',
                     'G5_entrance', 'G5_exit', 'F_entrance', 'F_exit',
                     'W_entrance', 'W_exit', 'D']

    for fn in [
        "HSC_raytrace_1.txt", "HSC_raytrace_2.txt", "HSC_raytrace_3.txt"
    ]:
        filename = os.path.join(directory, "testdata", fn)
        with open(filename) as f:
            arr = np.loadtxt(f, skiprows=22, usecols=list(range(0, 12)))
        arr0 = arr[0]
        rv = batoid.RayVector(
            arr0[1]/1000, arr0[2]/1000, 16.0,
            arr0[4], arr0[5], -arr0[6],
            t=0, wavelength=750e-9
        )
        tf = telescope.traceFull(rv)

        i = 0
        for name in surface_names:
            surface = tf[name]
            srv = surface['out']

            srv.toCoordSys(batoid.CoordSys())
            bt_isec = np.array([srv.x, srv.y, srv.z-16.0]).T[0]
            zx_isec = arr[HSC_surfaces[i]-1][1:4]/1000
            # nanometer agreement
            np.testing.assert_allclose(bt_isec, zx_isec, rtol=0, atol=1e-9)

            v = srv.v/np.linalg.norm(srv.v)
            bt_angle = v[0]
            zx_angle = arr[HSC_surfaces[i]-1][4:7]
            # direction cosines agree to 1e-9
            np.testing.assert_allclose(bt_angle, zx_angle, rtol=0, atol=1e-9)

            i += 1


@timer
def test_HSC_huygensPSF(plot=False):
    fn = os.path.join(directory, "testdata", "HSC_huygensPSF.txt")
    with open(fn) as f:
        Zarr = np.loadtxt(f, skiprows=21)
    Zarr = Zarr[::-1]  # Need to invert, probably just a Zemax convention...

    telescope = batoid.Optic.fromYaml("HSC_no_obsc.yaml")

    thx = np.deg2rad(0.0)
    thy = np.deg2rad(0.75)
    wavelength = 750e-9
    nx = 128
    dx = 0.25e-6
    print("computing Huygens PSF")
    hPSF = batoid.huygensPSF(
        telescope,
        thx, thy, projection='zemax',
        wavelength=wavelength,
        nx=nx, dx=dx, nxOut=256,
        reference='mean'
    )
    print("Done")

    # Normalize images
    Zarr /= np.sum(Zarr)
    hPSF.array /= np.sum(hPSF.array)
    Zmax = np.max(Zarr)
    Zarr /= Zmax
    hPSF.array /= Zmax

    # Use GalSim InterpolateImage to align and subtract
    ii = galsim.InterpolatedImage(
        galsim.Image(hPSF.array, scale=0.25),
        normalization='sb'
    )

    # Now setup an optimizer to fit for x/y shift
    def modelimg(params, ii=ii):
        dx, dy, dlogflux = params
        model = ii.shift(dx, dy)*np.exp(dlogflux)
        return model.drawImage(method='sb', scale=0.25, nx=256, ny=256)

    def resid(params, ii=ii, Zarr=Zarr):
        img = modelimg(params, ii=ii)
        r = (img.array - Zarr).ravel()
        return r

    kwargs = dict(ii=ii, Zarr=Zarr)
    print("Aligning")
    result = least_squares(resid, np.array([0.0, 0.0, 0.0]), kwargs=kwargs)
    optImg = modelimg(result.x, ii=ii)
    print("Done")

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ncols=3, figsize=(10,3))
        i0 = axes[0].imshow(optImg.array)
        i1 = axes[1].imshow(Zarr)
        i2 = axes[2].imshow(optImg.array-Zarr)
        plt.colorbar(i0, ax=axes[0])
        plt.colorbar(i1, ax=axes[1])
        plt.colorbar(i2, ax=axes[2])
        plt.tight_layout()
        plt.show()


    np.testing.assert_allclose(Zarr, optImg.array, rtol=0, atol=3e-2)
    bmom = galsim.hsm.FindAdaptiveMom(optImg)
    Zmom = galsim.hsm.FindAdaptiveMom(galsim.Image(Zarr, copy=True))
    np.testing.assert_allclose(
        Zmom.observed_shape.g1,
        bmom.observed_shape.g1,
        rtol=0, atol=0.01
    )
    np.testing.assert_allclose(
        Zmom.observed_shape.g2,
        bmom.observed_shape.g2,
        rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        Zmom.moments_sigma,
        bmom.moments_sigma,
        rtol=0, atol=0.1
    )


@timer
def test_HSC_wf():
    fn = os.path.join(directory, "testdata", "HSC_wavefront.txt")
    with open(fn) as f:
        Zwf = np.loadtxt(f, skiprows=17)
    Zwf = Zwf[::-1]  # Need to invert, probably just a Zemax convention...

    telescope = batoid.Optic.fromYaml("HSC_no_obsc.yaml")

    thx = np.deg2rad(0.0)
    thy = np.deg2rad(0.75)
    wavelength = 750e-9
    nx = 512
    bwf = batoid.wavefront(telescope, thx, thy, wavelength, nx=nx)

    Zwf = np.ma.MaskedArray(data=Zwf, mask=Zwf==0)  # Turn Zwf into masked array

    # There are unimportant differences in piston, tip, and tilt terms.  So
    # instead of comparing the wavefront directly, we'll compare Zernike
    # coefficients for j >= 4.
    x = np.linspace(-1, 1, nx, endpoint=False)
    x, y = np.meshgrid(x, x)
    w = ~Zwf.mask  # Use the same mask for both Zemax and batoid
    basis = galsim.zernike.zernikeBasis(37, x[w], y[w])
    Zcoefs, _, _, _ = np.linalg.lstsq(basis.T, Zwf[w], rcond=-1)
    Bcoefs, _, _, _ = np.linalg.lstsq(basis.T, bwf.array[w], rcond=-1)

    for j in range(1, 38):
        print("{:<4d} {:8.4f} {:8.4f}".format(j, Zcoefs[j], Bcoefs[j]))

    np.testing.assert_allclose(Zcoefs[4:], Bcoefs[4:], rtol=0, atol=0.01)
    # higher order Zernikes match even better
    np.testing.assert_allclose(Zcoefs[11:], Bcoefs[11:], rtol=0, atol=0.01)


@timer
def test_HSC_zernike():
    ZZernike = [0]
    with open(os.path.join(directory, "testdata", "HSC_Zernike.txt")) as f:
        for i, line in enumerate(f):
            if i > 38:
                ZZernike.append(float(line[9:20]))
    ZZernike = np.array(ZZernike)

    telescope = batoid.Optic.fromYaml("HSC_no_obsc.yaml")

    thx = np.deg2rad(0.0)
    thy = np.deg2rad(0.75)
    wavelength = 750e-9
    nx = 256

    bZernike = batoid.zernike(
        telescope, thx, thy, wavelength, jmax=37, nx=nx,
        projection='zemax', reference='chief'
    )

    print()
    print("j      Zemax    batoid    diff")
    print("------------------------------")
    for j in range(1, 38):
        print(
            f"{j:<4d} {ZZernike[j]:8.4f} {bZernike[j]:8.4f} "
            f"{ZZernike[j]-bZernike[j]:8.4f}"
        )

    # Don't care about piston, tip, or tilt.
    np.testing.assert_allclose(ZZernike[4:], bZernike[4:], rtol=0, atol=1e-2)
    np.testing.assert_allclose(ZZernike[11:], bZernike[11:], rtol=0, atol=3e-3)


@timer
def test_LSST_wf(plot=False):
    thxs = [0.0, 0.0, 0.0, 1.176]
    thys = [0.0, 1.225, 1.75, 1.176]
    fns = ["LSST_wf_0.0_0.0.txt",
           "LSST_wf_0.0_1.225.txt",
           "LSST_wf_0.0_1.75.txt",
           "LSST_wf_1.176_1.176.txt"]
    for thx, thy, fn in zip(thxs, thys, fns):
        fn = os.path.join(directory, "testdata", fn)
        with open(fn, encoding='utf-16-le') as f:
            Zwf = np.loadtxt(f, skiprows=16)
        Zwf = Zwf[::-1]  # Need to invert, probably just a Zemax convention...

        telescope = batoid.Optic.fromYaml("LSST_g_500.yaml")

        thx = np.deg2rad(thx)
        thy = np.deg2rad(thy)
        wavelength = 500e-9
        nx = 32

        bwf = batoid.wavefront(
            telescope, thx, thy, wavelength, nx=nx,
            reference='chief', projection='zemax'
        )
        # Turn Zwf into masked array
        Zwf = np.ma.MaskedArray(data=Zwf, mask=Zwf==0)

        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(ncols=3, figsize=(10,3))
            i0 = axes[0].imshow(bwf.array)
            i1 = axes[1].imshow(Zwf)
            i2 = axes[2].imshow(bwf.array-Zwf)
            axes[0].set_title("batoid")
            axes[1].set_title("Zemax")
            axes[2].set_title("difference")
            plt.colorbar(i0, ax=axes[0], label='waves')
            plt.colorbar(i1, ax=axes[1], label='waves')
            plt.colorbar(i2, ax=axes[2], label='waves')
            plt.tight_layout()
            plt.show()

        np.testing.assert_allclose(
            Zwf*wavelength,
            bwf.array*wavelength,
            atol=1e-11, rtol=0)  # 10 picometer tolerance!


@timer
def test_LSST_fftPSF(plot=False):
    thxs = [0.0, 0.0, 0.0, 1.176]
    thys = [0.0, 1.225, 1.75, 1.176]
    fns = ["LSST_fftpsf_0.0_0.0.txt",
           "LSST_fftpsf_0.0_1.225.txt",
           "LSST_fftpsf_0.0_1.75.txt",
           "LSST_fftpsf_1.176_1.176.txt"]
    for thx, thy, fn in zip(thxs, thys, fns):
        fn = os.path.join(directory, "testdata", fn)
        with open(fn, encoding='utf-16-le') as f:
            Zpsf = np.loadtxt(f, skiprows=18)
        Zpsf = Zpsf[::-1]  # Need to invert, probably just a Zemax convention...
        Zpsf /= np.max(Zpsf)

        telescope = batoid.Optic.fromYaml("LSST_g_500.yaml")

        thx = np.deg2rad(thx)
        thy = np.deg2rad(thy)
        wavelength = 500e-9
        nx = 32

        bpsf = batoid.fftPSF(
            telescope, thx, thy, wavelength, nx=nx,
            reference='chief', projection='zemax'
        )
        bpsf.array = bpsf.array[::-1,::-1] # b/c primitives are negative
        bpsf.array /= np.max(bpsf.array)

        # Use GalSim InterpolateImage to align and subtract
        ii = galsim.InterpolatedImage(
            galsim.Image(bpsf.array, scale=1.0),
            normalization='sb'
        )

        # Now setup an optimizer to fit for x/y shift
        def modelimg(params, ii=ii):
            dx, dy, dlogflux = params
            model = ii.shift(dx, dy)*np.exp(dlogflux)
            return model.drawImage(method='sb', scale=1.0, nx=64, ny=64)

        def resid(params, ii=ii, Zpsf=Zpsf):
            img = modelimg(params, ii=ii)
            r = (img.array - Zpsf).ravel()
            return r

        kwargs = dict(ii=ii, Zpsf=Zpsf)
        result = least_squares(resid, np.array([0.0, 0.0, 0.0]), kwargs=kwargs)
        optImg = modelimg(result.x, ii=ii)

        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(ncols=3, figsize=(10,3))
            i0 = axes[0].imshow(optImg.array)
            i1 = axes[1].imshow(Zpsf)
            i2 = axes[2].imshow(optImg.array-Zpsf)
            plt.colorbar(i0, ax=axes[0])
            plt.colorbar(i1, ax=axes[1])
            plt.colorbar(i2, ax=axes[2])
            plt.tight_layout()
            plt.show()


@pytest.mark.slow
@timer
def test_LSST_huygensPSF(plot=False):
    thxs = [0.0, 0.0, 0.0, 1.176]
    thys = [0.0, 1.225, 1.75, 1.176]
    fns = ["LSST_hpsf_0.0_0.0.txt",
           "LSST_hpsf_0.0_1.225.txt",
           "LSST_hpsf_0.0_1.75.txt",
           "LSST_hpsf_1.176_1.176.txt"]
    if __name__ != "__main__":
        thxs = thxs[2:3]
        thys = thys[2:3]
        fns = fns[2:3]
    for thx, thy, fn in zip(thxs, thys, fns):
        fn = os.path.join(directory, "testdata", fn)
        with open(fn, encoding='utf-16-le') as f:
            Zpsf = np.loadtxt(f, skiprows=21)
        Zpsf = Zpsf[::-1]  # Need to invert, probably just a Zemax convention...
        Zpsf /= np.max(Zpsf)

        telescope = batoid.Optic.fromYaml("LSST_g_500.yaml")

        thx = np.deg2rad(thx)
        thy = np.deg2rad(thy)
        wavelength = 500e-9

        bpsf = batoid.huygensPSF(
            telescope, thx, thy, wavelength, nx=128,
            # telescope, thx, thy, wavelength, nx=1024,
            reference='chief', projection='zemax',
            dx=0.289e-6, nxOut=64
        )
        bpsf.array /= np.max(bpsf.array)

        # Use GalSim InterpolateImage to align and subtract
        ii = galsim.InterpolatedImage(
            galsim.Image(bpsf.array, scale=1.0),
            normalization='sb'
        )

        # Now setup an optimizer to fit for x/y shift
        def modelimg(params, ii=ii):
            dx, dy, dlogflux = params
            model = ii.shift(dx, dy)*np.exp(dlogflux)
            return model.drawImage(method='sb', scale=1.0, nx=64, ny=64)

        def resid(params, ii=ii, Zpsf=Zpsf):
            img = modelimg(params, ii=ii)
            r = (img.array - Zpsf).ravel()
            return r

        kwargs = dict(ii=ii, Zpsf=Zpsf)
        print("Aligning")
        result = least_squares(resid, np.array([0.0, 0.0, 0.0]), kwargs=kwargs)
        optImg = modelimg(result.x, ii=ii)
        print("Done")

        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(ncols=3, figsize=(10,3))
            i0 = axes[0].imshow(optImg.array)
            i1 = axes[1].imshow(Zpsf)
            i2 = axes[2].imshow(optImg.array-Zpsf)
            plt.colorbar(i0, ax=axes[0])
            plt.colorbar(i1, ax=axes[1])
            plt.colorbar(i2, ax=axes[2])
            plt.tight_layout()
            plt.show()

            if thy not in [0.0, 1.176]:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(optImg.array[:,32], c='g')
                ax.plot(Zpsf[:,32], c='b')
                ax.plot((optImg.array-Zpsf)[:,32], c='r')
                plt.show()


@timer
def test_LSST_trace(verbose=False):
    # The g_500 file uses vacuum instead of air, which is important to match
    # Zemax for this test.
    telescope = batoid.Optic.fromYaml("LSST_g_500.yaml")

    zSurfaces = [4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    for fn in ["LSST_trace_0.txt", "LSST_trace_1.txt", "LSST_trace_2.txt"]:
        filename = os.path.join(directory, "testdata", fn)
        # Get normalized coordinates
        with open(filename, encoding='utf-16-le') as f:
            Hx, Hy, Px, Py = np.genfromtxt(f, skip_header=13, max_rows=4, usecols=(6,))
        with open(filename, encoding='utf-16-le') as f:
            arr = np.genfromtxt(f, skip_header=22, max_rows=18, usecols=list(range(1, 12)))

        ray = batoid.RayVector.fromStop(
            Px*4.18, Py*4.18,
            optic=telescope,
            wavelength=500e-9,
            theta_x=np.deg2rad(Hx*1.75), theta_y=np.deg2rad(Hy*1.75),
            projection='zemax'
        )
        tf = telescope.traceFull(ray)

        for surface, iz in zip(tf.values(), zSurfaces):
            r = surface['out'].toCoordSys(batoid.globalCoordSys)
            n = 1./np.sqrt(np.sum(r.v**2))

            # Note Zemax has different sign convention for z-coordinates and
            # direction cosines.  The important bits to match are x and y, which
            # do match, including the signs.

            if verbose:
                print(surface['name'])
                print('x', r.x[0], arr[iz][0]/1e3, r.x-arr[iz][0]/1e3)
                print('y', r.y[0], arr[iz][1]/1e3, r.y-arr[iz][1]/1e3)
                print('z', r.z[0], arr[iz][2]/1e3, r.z+arr[iz][2]/1e3)
                print('vx', r.vx[0]*n, arr[iz][3], np.abs(r.vx*n) - np.abs(arr[iz][3]))
                print('vy', r.vy[0]*n, arr[iz][4], np.abs(r.vy*n) - np.abs(arr[iz][4]))
                print('vz', r.vz[0]*n, arr[iz][5], np.abs(r.vz*n) - np.abs(arr[iz][5]))
                print()
            np.testing.assert_allclose(r.x, arr[iz][0]/1e3, rtol=0, atol=1e-10)
            np.testing.assert_allclose(r.y, arr[iz][1]/1e3, rtol=0, atol=1e-10)
            np.testing.assert_allclose(
                np.abs(r.z), np.abs(arr[iz][2]/1e3),
                rtol=0, atol=1e-10
            )
            np.testing.assert_allclose(np.abs(r.vx*n), np.abs(arr[iz][3]), rtol=0, atol=1e-10)
            np.testing.assert_allclose(np.abs(r.vy*n), np.abs(arr[iz][4]), rtol=0, atol=1e-10)
            np.testing.assert_allclose(np.abs(r.vz*n), np.abs(arr[iz][5]), rtol=0, atol=1e-10)


@timer
def test_DECam_trace(verbose=False):
    telescope = batoid.Optic.fromYaml("DECam.yaml")
    for fn in ["DECam_raytrace_0.txt", "DECam_raytrace_1.txt", "DECam_raytrace_2.txt"]:
        filename = os.path.join(directory, "testdata", fn)

        # Get normalized coordinates
        with open(filename, encoding='utf-16-le') as f:
            Hx, Hy, Px, Py = np.genfromtxt(f, skip_header=13, max_rows=4, usecols=(6,))

        with open(filename, encoding='utf-16-le') as f:
            arr = np.genfromtxt(f, skip_header=22, max_rows=40, usecols=list(range(1, 12)))

        ray = batoid.RayVector.fromStop(
            Px*2.005, Py*2.005,
            optic=telescope,
            stopSurface=telescope['PM'],
            wavelength=700e-9,
            theta_x=np.deg2rad(Hx*1.1), theta_y=np.deg2rad(Hy*1.1),
            projection='zemax'
        )
        tf = telescope.traceFull(ray)

        zSurfaces = [
            2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28,
            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
        ]

        for surface, iiz in zip(tf.values(), zSurfaces):
            iz = iiz-1
            r = surface['out'].toCoordSys(batoid.globalCoordSys)
            n = 1./np.sqrt(np.sum(r.v**2))

            # Note Zemax has different sign convention for z-coordinates and
            # direction cosines.  The important bits to match are x and y, which
            # do match, including the signs.
            if verbose:
                print(surface['name'])
                print('x', r.x[0], arr[iz][0]/1e3, r.x-arr[iz][0]/1e3)
                print('y', r.y[0], arr[iz][1]/1e3, r.y-arr[iz][1]/1e3)
                print('z', r.z[0], -arr[iz][2]/1e3, r.z+arr[iz][2]/1e3)
                print('vx', r.vx[0]*n, arr[iz][3], np.abs(r.vx*n) - np.abs(arr[iz][3]))
                print('vy', r.vy[0]*n, arr[iz][4], np.abs(r.vy*n) - np.abs(arr[iz][4]))
                print('vz', r.vz[0]*n, arr[iz][5], np.abs(r.vz*n) - np.abs(arr[iz][5]))
                print()
            np.testing.assert_allclose(r.x, arr[iz][0]/1e3, rtol=0, atol=1e-9)
            np.testing.assert_allclose(r.y, arr[iz][1]/1e3, rtol=0, atol=1e-9)
            np.testing.assert_allclose(
                np.abs(r.z), np.abs(arr[iz][2]/1e3),
                rtol=0, atol=1e-9
            )
            np.testing.assert_allclose(np.abs(r.vx*n), np.abs(arr[iz][3]), rtol=0, atol=1e-9)
            np.testing.assert_allclose(np.abs(r.vy*n), np.abs(arr[iz][4]), rtol=0, atol=1e-9)
            np.testing.assert_allclose(np.abs(r.vz*n), np.abs(arr[iz][5]), rtol=0, atol=1e-9)


@timer
def test_DECam_exit_pupil_pos():
    telescope = batoid.Optic.fromYaml("DECam.yaml")
    # From the Optics Prescription report, the exit pupil is
    # -4275.016 mm behind the focal plane.  This is with the stop surface set to
    # the primary mirror though, so make sure to adjust that.
    telescope.stopSurface = telescope['PM']
    np.testing.assert_allclose(
        batoid.exitPupilPos(telescope, wavelength=700e-9),
        (0, 0, telescope['D'].coordSys.origin[2]+4275.016/1e3),
        rtol=0, atol=1e-3
    )


@timer
def test_CBP_trace():
    cbp = batoid.Optic.fromYaml("CBP.yaml")

    indices = {
        'Schmidt_plate_entrance': 1,
        'Schmidt_plate_exit': 2,
        'stop': 3,
        'turning_flat': 5,
        'PM': 7,
        'flat_hole': 9,
        'L1_entrance': 11,
        'L1_exit': 12,
        'L2_entrance': 13,
        'L2_exit': 14,
        'L3_entrance': 15,
        'L3_exit': 16,
        'Detector': 17
    }

    for i in range(1, 5):
    # for i in range(2, 3):
        fn = os.path.join(directory, "testdata", f"CBP_SRT{i}.txt")
        with open(fn, encoding='utf-16-le') as f:
            wavelength = np.genfromtxt(f, skip_header=8, max_rows=1, usecols=(2,))
        with open(fn, encoding='utf-16-le') as f:
            Hx, Hy, Px, Py = np.genfromtxt(f, skip_header=13, max_rows=4, usecols=(6,))
        with open(fn, encoding='utf-16-le') as f:
            idxs, x, y, z, xcos, ycos, zcos, nx, ny, nz, cos, path = np.genfromtxt(
                f, skip_header=22, max_rows=18, usecols=list(range(0, 12)), unpack=True
            )
        # Switch to batoid conventions.  x, y, z -> -x, y, -z
        Hx *= -1
        Px *= -1
        x *= -1
        z *= -1
        xcos *= -1
        zcos *= -1
        nx *= -1
        nz *= -1
        # Wavelength to meters
        wavelength *= 1e-6
        # distances to meters
        x *= 1e-3
        y *= 1e-3
        z *= 1e-3

        # Initial ray conditions
        r = np.r_[x[0], y[0], z[0]]
        v = batoid.utils.zemaxToDirCos(Hx*np.deg2rad(2.05), Hy*np.deg2rad(2.05))

        ray = batoid.RayVector(*r, *v, t=0, wavelength=wavelength)
        ray.propagate(-0.1)  # rewind a bit

        tf = cbp.traceFull(ray)

        for k in tf:
            if k == 'flat_hole':
                continue  # Zemax is weird about the tilt here.  ignore.
            idx = indices[k]
            np.testing.assert_allclose(
                tf[k]['out'].toCoordSys(batoid.globalCoordSys).r[0],
                np.array([x[idx], y[idx], z[idx]]),
                rtol=0, atol=1e-10
            )

            b_v = tf[k]['out'].toCoordSys(batoid.globalCoordSys).v[0]
            b_v /= np.linalg.norm(b_v)
            # Zemax always has zcos > 0
            if b_v[2] > 0:
                b_v *= -1
            np.testing.assert_allclose(
                b_v,
                np.array([xcos[idx], ycos[idx], zcos[idx]]),
                rtol=0, atol=1e-10
            )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--plotWF", action='store_true')
    parser.add_argument("--plotFFT", action='store_true')
    parser.add_argument("--plotHuygens", action='store_true')
    args = parser.parse_args()

    init_gpu()
    test_HSC_trace()
    test_HSC_huygensPSF(args.plotHuygens)
    test_HSC_wf()
    test_HSC_zernike()
    test_LSST_wf(args.plotWF)
    test_LSST_fftPSF(args.plotFFT)
    test_LSST_huygensPSF(args.plotHuygens)
    test_LSST_trace(verbose=False)
    test_DECam_trace(verbose=False)
    test_DECam_exit_pupil_pos()
    test_CBP_trace()