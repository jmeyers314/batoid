# Compare ray by ray tracing to Zemax
import os

import pytest
import numpy as np
import batoid
from test_helpers import timer
import yaml

hasGalSim = True
try:
    import galsim
except ImportError:
    hasGalSim = False

hasLMFit = True
try:
    import lmfit
except ImportError:
    hasLMFit = False


directory = os.path.dirname(__file__)


@timer
def test_HSC_trace():
    fn = os.path.join(batoid.datadir, "HSC", "HSC_old.yaml")
    config = yaml.safe_load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    # Zemax has a number of virtual surfaces that we don't trace in batoid.  Also, the HSC.yaml
    # above includes Baffle surfaces not in Zemax.  The following lists select out the surfaces in
    # common to both models.
    HSC_surfaces = [3, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 24, 25, 28, 29, 31]
    surface_names = ['PM', 'G1_entrance', 'G1_exit', 'G2_entrance', 'G2_exit',
                     'ADC1_entrance', 'ADC1_exit', 'ADC2_entrance', 'ADC2_exit',
                     'G3_entrance', 'G3_exit', 'G4_entrance', 'G4_exit',
                     'G5_entrance', 'G5_exit', 'F_entrance', 'F_exit',
                     'W_entrance', 'W_exit', 'D']

    for fn in ["HSC_raytrace_1.txt", "HSC_raytrace_2.txt", "HSC_raytrace_3.txt"]:
        filename = os.path.join(directory, "testdata", fn)
        with open(filename) as f:
            arr = np.loadtxt(f, skiprows=22, usecols=list(range(0, 12)))
        arr0 = arr[0]
        ray = batoid.Ray((arr0[1]/1000, arr0[2]/1000, 16.0), (arr0[4], arr0[5], -arr0[6]),
                         t=0, wavelength=750e-9)
        tf = telescope.traceFull(ray)

        i = 0
        for surface in tf:
            if surface['name'] != surface_names[i]:
                continue

            s = surface['out']
            v = s.v/np.linalg.norm(s.v)

            transform = batoid.CoordTransform(surface['outCoordSys'], batoid.CoordSys())
            s = transform.applyForward(s)
            bt_isec = np.array([s.x, s.y, s.z-16.0])
            zx_isec = arr[HSC_surfaces[i]-1][1:4]/1000
            np.testing.assert_allclose(bt_isec, zx_isec, rtol=0, atol=1e-9) # nanometer agreement

            bt_angle = np.array([v[0], v[1], v[2]])
            zx_angle = arr[HSC_surfaces[i]-1][4:7]
            # direction cosines agree to 1e-9
            np.testing.assert_allclose(bt_angle, zx_angle, rtol=0, atol=1e-9)

            i += 1


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@pytest.mark.skipif(not hasLMFit, reason="lmfit not found")
@pytest.mark.skipif(__name__ != '__main__', reason="slow test")
@timer
def test_HSC_huygensPSF():
    fn = os.path.join(directory, "testdata", "HSC_huygensPSF.txt")
    with open(fn) as f:
        Zarr = np.loadtxt(f, skiprows=21)
    Zarr = Zarr[::-1]  # Need to invert, probably just a Zemax convention...

    HSC_fn = os.path.join(batoid.datadir, "HSC", "HSC_no_obsc.yaml")
    config = yaml.safe_load(open(HSC_fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    thx = np.deg2rad(0.0)
    thy = np.deg2rad(0.75)
    wavelength = 750e-9
    nx = 512
    dx = 0.25e-6
    print("computing Huygens PSF")
    hPSF = batoid.huygensPSF(telescope, thx, thy, wavelength, nx=nx, projection='zemax',
                             dx=dx, nxOut=256)
    print("Done")

    # Normalize images
    Zarr /= np.sum(Zarr)
    hPSF.array /= np.sum(hPSF.array)
    Zmax = np.max(Zarr)
    Zarr /= Zmax
    hPSF.array /= Zmax

    # Use GalSim InterpolateImage to align and subtract
    ii = galsim.InterpolatedImage(galsim.Image(hPSF.array, scale=0.25), normalization='sb')

    # Now setup an optimizer to fit for x/y shift
    def resid(params):
        p = params.valuesdict()
        model = ii.shift(p['dx'], p['dy'])*np.exp(p['dlogflux'])
        img = model.drawImage(method='sb', scale=0.25, nx=256, ny=256)
        r = (img.array - Zarr).ravel()
        return r
    params = lmfit.Parameters()
    params.add('dx', value=0.0)
    params.add('dy', value=0.0)
    params.add('dlogflux', value=0.0)
    print("Aligning")
    opt = lmfit.minimize(resid, params)
    print("Done")

    p = opt.params.valuesdict()
    model = ii.shift(p['dx'], p['dy'])*np.exp(p['dlogflux'])
    optImg = model.drawImage(method='sb', scale=0.25, nx=256, ny=256)

    np.testing.assert_allclose(Zarr, optImg.array, rtol=0, atol=3e-2)
    Zmom = galsim.hsm.FindAdaptiveMom(galsim.Image(Zarr, scale=0.25))
    bmom = galsim.hsm.FindAdaptiveMom(optImg)
    np.testing.assert_allclose(Zmom.observed_shape.g1, bmom.observed_shape.g1, rtol=0, atol=0.01)
    np.testing.assert_allclose(Zmom.observed_shape.g2, bmom.observed_shape.g2, rtol=0, atol=1e-7)
    np.testing.assert_allclose(Zmom.moments_sigma, bmom.moments_sigma, rtol=0, atol=0.1)


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_HSC_wf():
    fn = os.path.join(directory, "testdata", "HSC_wavefront.txt")
    with open(fn) as f:
        Zwf = np.loadtxt(f, skiprows=17)
    Zwf = Zwf[::-1]  # Need to invert, probably just a Zemax convention...

    HSC_fn = os.path.join(batoid.datadir, "HSC", "HSC_no_obsc.yaml")
    config = yaml.safe_load(open(HSC_fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    thx = np.deg2rad(0.0)
    thy = np.deg2rad(0.75)
    wavelength = 750e-9
    nx = 512
    bwf = batoid.wavefront(telescope, thx, thy, wavelength, nx=nx)

    Zwf = np.ma.MaskedArray(data=Zwf, mask=Zwf==0)  # Turn Zwf into masked array

    # There are unimportant differences in piston, tip, and tilt terms.  So instead of comparing
    # the wavefront directly, we'll compare Zernike coefficients for j >= 4.
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


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_HSC_zernike():
    ZZernike = [0]
    with open(os.path.join(directory, "testdata", "HSC_Zernike.txt")) as f:
        for i, line in enumerate(f):
            if i > 38:
                ZZernike.append(float(line[9:20]))
    ZZernike = np.array(ZZernike)

    HSC_fn = os.path.join(batoid.datadir, "HSC", "HSC_no_obsc.yaml")
    config = yaml.safe_load(open(HSC_fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    thx = np.deg2rad(0.0)
    thy = np.deg2rad(0.75)
    wavelength = 750e-9
    nx = 256

    bZernike = batoid.zernike(
        telescope, thx, thy, wavelength, jmax=37, nx=nx,
        projection='gnomonic')
    # revisit this with projection='zemax' once we're referencing the wavefront
    # to the chief ray...


    print()
    print("j      Zemax    batoid")
    print("----------------------")
    for j in range(1, 38):
        print("{:<4d} {:8.4f} {:8.4f}".format(j, ZZernike[j], bZernike[j]))

    # Don't care about piston, tip, or tilt.
    np.testing.assert_allclose(ZZernike[4:], bZernike[4:], rtol=0, atol=1e-3)
    np.testing.assert_allclose(ZZernike[11:], bZernike[11:], rtol=0, atol=2e-4)


@pytest.mark.skipif(not hasGalSim, reason="galsim not found")
@timer
def test_LSST_wf():
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

        LSST_fn = os.path.join(batoid.datadir, "LSST", "LSST_g_500.yaml")
        config = yaml.safe_load(open(LSST_fn))
        telescope = batoid.parse.parse_optic(config['opticalSystem'])

        thx = np.deg2rad(thx)
        thy = np.deg2rad(thy)
        wavelength = 500e-9
        nx = 32

        bwf = batoid.psf.newWavefront(
            telescope, thx, thy, wavelength, nx=nx,
            reference='chief', projection='zemax'
        )
        Zwf = np.ma.MaskedArray(data=Zwf, mask=Zwf==0)  # Turn Zwf into masked array

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(ncols=3)
        # i0 = axes[0].imshow(bwf.array)
        # i1 = axes[1].imshow(Zwf)
        # i2 = axes[2].imshow(bwf.array-Zwf)
        # plt.colorbar(i0, ax=axes[0])
        # plt.colorbar(i1, ax=axes[1])
        # plt.colorbar(i2, ax=axes[2])
        # plt.show()

        np.testing.assert_allclose(
            Zwf*wavelength,
            bwf.array*wavelength,
            atol=1e-11, rtol=0)  # 10 picometer tolerance!


if __name__ == '__main__':
    test_HSC_trace()
    test_HSC_huygensPSF()
    test_HSC_wf()
    test_HSC_zernike()
    test_LSST_wf()
