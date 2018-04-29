import numpy as np
import batoid
from .utils import bivariate_fit, gnomicToDirCos, dirCosToGnomic


def huygensPSF(optic, xs, ys, zs=None, rays=None, saveRays=False):
    """Compute a PSF via the Huygens construction.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    xs, ys : ndarray
        Coordinates at which to evaluate the PSF in the optic.outCoordSys system.
    zs : ndarray, optional
        Optional z coordinates at which to evaluate the PSF.  Default: 0.
    rays : RayVector
        Input rays to optical system.
    saveRays : bool, optional
        Whether or not to preserve input rays or overwrite.  Default: False

    Returns
    -------
    psf : ndarray
        The PSF

    Notes
    -----
    The Huygens construction is to evaluate the PSF as

    I(r) \propto \Sum_u exp(i phi(u)) exp(i k(u).r)

    The u are assumed to uniformly sample the entrance pupil.  The phis are the phases of the exit
    rays evaluated at a single arbitrary time.  The k(u) indicates the conversion of the uniform
    entrance pupil samples into nearly (though not exactly) uniform samples in k-space of the output
    rays.
    """
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
        amplitudes[i, j] = batoid._batoid.sumAmplitudeMany(
            rays,
            points[i, j],
            time
        )
    return np.abs(amplitudes)**2


def drdth(optic, theta_x, theta_y, wavelength, nx=16):
    """Calculate derivative of focal plane coord with respect to field angle.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians (gnomic tangent plane projection)
    wavelength : float
        Wavelength in meters
    nx : int, optional
        Size of ray grid to use.

    Returns
    -------
    drdth : (2, 2), ndarray
        Jacobian transformation matrix for converting between (theta_x, theta_y)
        and (x, y) on the focal plane.

    Notes
    -----
        This is the Jacobian of pixels -> tangent plane, (and importantly, not pixels -> ra/dec).
        It should be *close* to the inverse plate scale though, especially near the center of the
        tangent plane projection.
    """
    # We just use a finite difference approach here.
    dth = 1e-5

    # Make direction cosine vectors
    nominalCos = gnomicToDirCos(theta_x, theta_y)
    dthxCos = gnomicToDirCos(theta_x + dth, theta_y)
    dthyCos = gnomicToDirCos(theta_x, theta_y+ dth)

    # Flip the dirCos z-components so rays are headed downwards
    rays = batoid.rayGrid(optic.dist, optic.pupilSize,
        nominalCos[0], nominalCos[1], -nominalCos[2],
        nx, wavelength=wavelength, medium=optic.inMedium)

    rays_x = batoid.rayGrid(optic.dist, optic.pupilSize,
        dthxCos[0], dthxCos[1], -dthxCos[2],
        nx, wavelength=wavelength, medium=optic.inMedium)

    rays_y = batoid.rayGrid(optic.dist, optic.pupilSize,
        dthyCos[0], dthyCos[1], -dthyCos[2],
        nx, wavelength=wavelength, medium=optic.inMedium)

    optic.traceInPlace(rays)
    optic.traceInPlace(rays_x)
    optic.traceInPlace(rays_y)

    batoid.trimVignettedInPlace(rays)
    batoid.trimVignettedInPlace(rays_x)
    batoid.trimVignettedInPlace(rays_y)

    # meters / radian
    drx_dthx = (np.mean(rays_x.x) - np.mean(rays.x))/dth
    drx_dthy = (np.mean(rays_y.x) - np.mean(rays.x))/dth
    dry_dthx = (np.mean(rays_x.y) - np.mean(rays.y))/dth
    dry_dthy = (np.mean(rays_y.y) - np.mean(rays.y))/dth

    return np.array([[drx_dthx, dry_dthx], [drx_dthy, dry_dthy]])


def dthdr(optic, theta_x, theta_y, wavelength, nx=16):
    """Calculate derivative of field angle with respect to focal plane coordinate.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians (gnomic tangent plane projection)
    wavelength : float
        Wavelength in meters
    nx : int, optional
        Size of ray grid to use.

    Returns
    -------
    dthdr : (2, 2), ndarray
        Jacobian transformation matrix for converting between (x, y) on the focal plane and
        field angle (theta_x, theta_y).

    Notes
    -----
        This is the Jacobian of tangen plane -> pixels, (and importantly, not ra/dec -> pixels).
        It should be *close* to the plate scale though, especially near the center of the tangent
        plane projection.
    """
    return np.linalg.inv(drdth(optic, theta_x, theta_y, wavelength, nx=nx))


def dkdu(optic, theta_x, theta_y, wavelength, nx=16):
    """Calculate derivative of outgoing ray k-vector with respect to incoming ray
    pupil coordinate.


    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians (gnomic tangent plane projection)
    wavelength : float
        Wavelength in meters
    nx : int, optional
        Size of ray grid to use.

    Returns
    -------
    dkdu : (2, 2), ndarray
        Jacobian transformation matrix for converting between (kx, ky) of rays impacting the focal
        plane and initial field angle (gnomic tangent plane projection).
    """
    dirCos = gnomicToDirCos(theta_x, theta_y)
    rays = batoid.rayGrid(
        optic.dist, optic.pupilSize,
        dirCos[0], dirCos[1], -dirCos[2],
        nx, wavelength, optic.inMedium
    )

    pupilRays = batoid._batoid.propagatedToTimesMany(rays, np.zeros_like(rays.x))
    ux = np.array(pupilRays.x)
    uy = np.array(pupilRays.y)

    optic.traceInPlace(rays)
    w = np.where(1-rays.isVignetted)[0]
    ux = ux[w]
    uy = uy[w]

    kx = rays.kx[w]
    ky = rays.ky[w]

    soln = bivariate_fit(ux, uy, kx, ky)
    return soln[1:]


def wavefront(optic, theta_x, theta_y, wavelength, nx=32, sphereRadius=None):
    dirCos = gnomicToDirCos(theta_x, theta_y)
    rays = batoid.rayGrid(
        optic.dist, optic.pupilSize,
        dirCos[0], dirCos[1], -dirCos[2],
        nx, wavelength, optic.inMedium
    )

    if sphereRadius is None:
        sphereRadius = optic.sphereRadius

    outCoordSys = batoid.CoordSys()
    optic.traceInPlace(rays, outCoordSys=outCoordSys)
    w = np.where(1-rays.isVignetted)[0]
    point = np.mean(rays.p0[w], axis=0)

    # We want to place the vertex of the reference sphere one radius length away from the
    # intersection point.  So transform our rays into that coordinate system.
    transform = batoid.CoordTransform(
            outCoordSys, batoid.CoordSys(point+np.array([0,0,sphereRadius])))
    transform.applyForwardInPlace(rays)

    sphere = batoid.Sphere(-sphereRadius)
    sphere.intersectInPlace(rays)

    w = np.where(1-rays.isVignetted)[0]
    # Should potentially try to make the reference time w.r.t. the chief ray instead of the mean
    # of the good (unvignetted) rays.
    t0 = np.mean(rays.t0[w])

    return np.ma.masked_array((rays.t0-t0)/wavelength, mask=rays.isVignetted).reshape(nx, nx)


def fftPSF(optic, theta_x, theta_y, wavelength, nx=32, pad_factor=2):
    L = optic.pupilSize*pad_factor
    im_dtheta = wavelength / L
    wf = wavefront(optic, theta_x, theta_y, wavelength, nx)
    pad_size = nx*pad_factor
    expwf = np.zeros((pad_size, pad_size), dtype=np.complex128)
    start = pad_size//2-nx//2
    stop = pad_size//2+nx//2
    expwf[start:stop, start:stop][~wf.mask] = np.exp(2j*np.pi*wf[~wf.mask])
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(expwf))))**2
    return im_dtheta, psf


def zernike(optic, theta_x, theta_y, wavelength, nx=32, jmax=22, eps=0.0):
    import galsim.zernike as zern

    dirCos = gnomicToDirCos(theta_x, theta_y)
    rays = batoid.rayGrid(
        optic.dist, optic.pupilSize,
        dirCos[0], dirCos[1], -dirCos[2],
        nx, wavelength, optic.inMedium
    )

    batoid.propagateInPlaceMany(rays, np.zeros_like(rays.x))

    orig_x = np.array(rays.x).reshape(nx,nx)
    orig_y = np.array(rays.y).reshape(nx,nx)

    wf = wavefront(optic, theta_x, theta_y, wavelength, nx=nx)
    w = ~wf.mask

    basis = zern.zernikeBasis(
            jmax, orig_x[w], orig_y[w],
            R_outer=optic.pupilSize/2, R_inner=optic.pupilSize/2*eps
    )
    coefs, _, _, _ = np.linalg.lstsq(basis.T, wf[w], rcond=-1)

    return coefs
