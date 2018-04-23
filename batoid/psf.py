import numpy as np
import batoid


def huygensPSF(optic, xs=None, ys=None, zs=None, rays=None, saveRays=False):
    if zs is None:
        zs = np.zeros_like(xs)
    if saveRays:
        rays = batoid.RayVector(rays)  # Make a copy
    rays, outCoordSys = optic.traceInPlace(rays)
    batoid._batoid.trimVignettedInPlace(rays)
    # transform = batoid.CoordTransform(outCoordSys, batoid.CoordSys())
    # transform.applyForwardInPlace(rays)
    points = np.concatenate([aux[..., None] for aux in (xs, ys, zs)], axis=-1)
    time = rays[0].t0
    amplitudes = np.empty(xs.shape, dtype=np.complex128)
    for (i, j) in np.ndindex(xs.shape):
        amplitudes[i, j] = np.sum(
            batoid._batoid.amplitudeMany(
                rays,
                points[i, j],
                time
            )
        )
    return np.abs(amplitudes)**2


def wavefront(optic, wavelength, theta_x=0, theta_y=0, nx=32, rays=None, saveRays=False,
              sphereRadius=None):
    if rays is None:
        xcos = np.sin(theta_x*np.pi/180)
        ycos = np.sin(theta_y*np.pi/180)
        zcos = -np.sqrt(1.0 - xcos**2 - ycos**2)

        rays = batoid.rayGrid(
                optic.dist, optic.pupil_size, xcos, ycos, zcos, nx, wavelength, optic.inMedium)
    if saveRays:
        rays = batoid.RayVector(rays)
    if sphereRadius is None:
        sphereRadius = optic.sphereRadius

    outCoordSys = batoid.CoordSys()
    optic.traceInPlace(rays, outCoordSys=outCoordSys)
    goodRays = batoid._batoid.trimVignetted(rays)
    point = np.array([np.mean(goodRays.x), np.mean(goodRays.y), np.mean(goodRays.z)])

    # We want to place the vertex of the reference sphere one radius length away from the
    # intersection point.  So transform our rays into that coordinate system.
    transform = batoid.CoordTransform(
            outCoordSys, batoid.CoordSys(point+np.array([0,0,sphereRadius])))
    transform.applyForwardInPlace(rays)

    sphere = batoid.Sphere(-sphereRadius)
    sphere.intersectInPlace(rays)
    goodRays = batoid._batoid.trimVignetted(rays)
    # Should potentially try to make the reference time w.r.t. the chief ray instead of the mean
    # of the good (unvignetted) rays.
    t0 = np.mean(goodRays.t0)

    ts = rays.t0[:]
    isV = rays.isVignetted[:]
    ts -= t0
    ts /= wavelength
    wf = np.ma.masked_array(ts, mask=isV)
    return wf


def fftPSF(optic, wavelength, theta_x, theta_y, nx=32, pad_factor=2):
    L = optic.pupil_size*pad_factor
    im_dtheta = wavelength / L
    wf = wavefront(optic, wavelength, theta_x, theta_y, nx).reshape(nx, nx)
    pad_size = nx*pad_factor
    expwf = np.zeros((pad_size, pad_size), dtype=np.complex128)
    start = pad_size//2-nx//2
    stop = pad_size//2+nx//2
    expwf[start:stop, start:stop][~wf.mask] = np.exp(2j*np.pi*wf[~wf.mask])
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(expwf))))**2
    return im_dtheta, psf


def zernike(optic, wavelength, theta_x, theta_y, jmax=22, nx=32, eps=0.0):
    import galsim.zernike as zern

    xcos = np.sin(theta_x*np.pi/180)
    ycos = np.sin(theta_y*np.pi/180)
    zcos = -np.sqrt(1.0 - xcos**2 - ycos**2)

    rays = batoid.rayGrid(
            optic.dist, optic.pupil_size, xcos, ycos, zcos, nx, wavelength, optic.inMedium)

    orig_x = rays.x[:]
    orig_y = rays.y[:]

    wf = wavefront(optic, wavelength, rays=rays)

    w = ~wf.mask

    basis = zern.zernikeBasis(
            jmax, orig_x[w], orig_y[w],
            R_outer=optic.pupil_size/2, R_inner=optic.pupil_size/2*eps
    )
    coefs, _, _, _ = np.linalg.lstsq(basis.T, wf[w])

    return coefs
