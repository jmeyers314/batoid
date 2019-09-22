import numpy as np
import batoid
from .utils import bilinear_fit, fieldToDirCos


def huygensPSF(optic, theta_x=None, theta_y=None, wavelength=None, nx=None,
               projection='postel', dx=None, dy=None, nxOut=None):
    r"""Compute a PSF via the Huygens construction.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float, optional
        Wavelength in meters
    nx : int, optional
        Size of ray grid to use.
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.
    dx, dy : float, optional
        Lattice scales to use for PSF evaluation locations.  Default, use fftPSF lattice.

    Returns
    -------
    psf : batoid.Lattice
        The PSF.

    Notes
    -----
    The Huygens construction is to evaluate the PSF as

    I(x) \propto \Sum_u exp(i phi(u)) exp(i k(u).r)

    The u are assumed to uniformly sample the entrance pupil, but not include any rays that get
    vignetted before they reach the focal plane.  The phis are the phases of the exit rays evaluated
    at a single arbitrary time.  The k(u) indicates the conversion of the uniform entrance pupil
    samples into nearly (though not exactly) uniform samples in k-space of the output rays.

    The output locations where the PSF is evaluated are governed by dx, dy and nx.  If dx and dy are
    None, then the same lattice as in fftPSF will be used.  If dx and dy are scalars, then a lattice
    with primitive vectors [dx, 0] and [0, dy] will be used.  If dx and dy are 2-vectors, then those
    will be the primitive vectors of the output lattice.
    """
    from numbers import Real

    if dx is None:
        primitiveU = np.array([[optic.pupilSize/nx,0], [0, optic.pupilSize/nx]])
        primitiveK = dkdu(optic, theta_x, theta_y, wavelength, projection=projection).dot(primitiveU)
        pad_factor = 2
        primitiveX = np.vstack(
            reciprocalLatticeVectors(primitiveK[0], primitiveK[1], pad_factor*nx)
        )
    elif isinstance(dx, Real):
        if dy is None:
            dy = dx
        primitiveX = np.vstack([[dx, 0], [0, dy]])
        pad_factor = 1
    else:
        primitiveX = np.vstack([dx, dy])
        pad_factor = 1

    if nxOut is None:
        nxOut = nx

    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    rays = batoid.rayGrid(optic.backDist, optic.pupilSize,
        dirCos[0], dirCos[1], dirCos[2],
        nx, wavelength=wavelength, flux=1, medium=optic.inMedium,
        lattice=True)

    amplitudes = np.zeros((nxOut*pad_factor, nxOut*pad_factor), dtype=np.complex128)
    out = batoid.Lattice(np.zeros((nxOut*pad_factor, nxOut*pad_factor), dtype=float), primitiveX)

    rays = optic.traceInPlace(rays)
    rays.trimVignettedInPlace()
    # Need transpose to conform to numpy [y,x] ordering convention
    xs = out.coords[..., 0].T + np.mean(rays.x)
    ys = out.coords[..., 1].T + np.mean(rays.y)
    zs = np.zeros_like(xs)

    points = np.concatenate([aux[..., None] for aux in (xs, ys, zs)], axis=-1)
    time = rays[0].t
    for idx in np.ndindex(amplitudes.shape):
        amplitudes[idx] = rays.sumAmplitude(points[idx], time)
    out.array = np.abs(amplitudes)**2
    return out


def drdth(optic, theta_x, theta_y, wavelength, nx=16, projection='postel'):
    """Calculate derivative of focal plane coord with respect to field angle.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    nx : int, optional
        Size of ray grid to use.
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.

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
    nominalCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    dthxCos = fieldToDirCos(theta_x + dth, theta_y, projection=projection)
    dthyCos = fieldToDirCos(theta_x, theta_y + dth, projection=projection)

    rays = batoid.rayGrid(optic.backDist, optic.pupilSize,
        nominalCos[0], nominalCos[1], nominalCos[2],
        nx, wavelength=wavelength, flux=1, medium=optic.inMedium)

    rays_x = batoid.rayGrid(optic.backDist, optic.pupilSize,
        dthxCos[0], dthxCos[1], dthxCos[2],
        nx, wavelength=wavelength, flux=1, medium=optic.inMedium)

    rays_y = batoid.rayGrid(optic.backDist, optic.pupilSize,
        dthyCos[0], dthyCos[1], dthyCos[2],
        nx, wavelength=wavelength, flux=1, medium=optic.inMedium)

    optic.traceInPlace(rays)
    optic.traceInPlace(rays_x)
    optic.traceInPlace(rays_y)

    rays.trimVignettedInPlace()
    rays_x.trimVignettedInPlace()
    rays_y.trimVignettedInPlace()

    # meters / radian
    drx_dthx = (np.mean(rays_x.x) - np.mean(rays.x))/dth
    drx_dthy = (np.mean(rays_y.x) - np.mean(rays.x))/dth
    dry_dthx = (np.mean(rays_x.y) - np.mean(rays.y))/dth
    dry_dthy = (np.mean(rays_y.y) - np.mean(rays.y))/dth

    return np.array([[drx_dthx, dry_dthx], [drx_dthy, dry_dthy]])


def dthdr(optic, theta_x, theta_y, wavelength, nx=16, projection='postel'):
    """Calculate derivative of field angle with respect to focal plane coordinate.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    nx : int, optional
        Size of ray grid to use.
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.

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
    return np.linalg.inv(drdth(optic, theta_x, theta_y, wavelength, nx=nx, projection=projection))


def dkdu(optic, theta_x, theta_y, wavelength, nx=16, projection='postel'):
    """Calculate derivative of outgoing ray k-vector with respect to incoming ray
    pupil coordinate.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    nx : int, optional
        Size of ray grid to use.
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.

    Returns
    -------
    dkdu : (2, 2), ndarray
        Jacobian transformation matrix for converting between (kx, ky) of rays impacting the focal
        plane and initial field angle (gnomonic tangent plane projection).
    """
    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    rays = batoid.rayGrid(
        optic.backDist, optic.pupilSize,
        dirCos[0], dirCos[1], dirCos[2],
        nx, wavelength, 1.0, optic.inMedium
    )

    pupilRays = rays.propagatedToTime(0.0)
    ux = np.array(pupilRays.x)
    uy = np.array(pupilRays.y)

    optic.traceInPlace(rays)
    w = np.where(1-rays.vignetted)[0]
    ux = ux[w]
    uy = uy[w]

    kx = rays.kx[w]
    ky = rays.ky[w]

    soln = bilinear_fit(ux, uy, kx, ky)
    return soln[1:]


def wavefront(optic, theta_x, theta_y, wavelength, nx=32, projection='postel', sphereRadius=None,
              lattice=False, _addedWF=None):
    """Compute wavefront.

    Parameters
    ----------
    optic : batoid.Optic
        Optic for which to compute wavefront.
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float, optional
        Wavelength in meters
    nx : int, optional
        Size of ray grid to generate to compute wavefront.  Default: 32
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.
    sphereRadius : float, optional
        Radius of reference sphere in meters.  If None, then use optic.sphereRadius.
    lattice : bool, optional
        If true, then decenter the grid so it spans (-N/2, N/2+1), as appropriate
        for Fourier transforms.

    Returns
    -------
    wavefront : batoid.Lattice
        A batoid.Lattice object containing the wavefront values in waves and
        the primitive lattice vectors of the entrance pupil grid in meters.
    """
    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    rays = batoid.rayGrid(
        optic.backDist, optic.pupilSize,
        dirCos[0], dirCos[1], dirCos[2],
        nx, wavelength, 1.0, optic.inMedium,
        lattice=lattice
    )

    if sphereRadius is None:
        sphereRadius = optic.sphereRadius

    optic.traceInPlace(rays)
    w = np.where(1-rays.vignetted)[0]
    point = np.mean(rays.r[w], axis=0)

    # We want to place the vertex of the reference sphere one radius length away from the
    # intersection point.  So transform our rays into that coordinate system.
    targetCoordSys = rays.coordSys.shiftLocal(
        point+np.array([0,0,sphereRadius])
    )
    rays.toCoordSysInPlace(targetCoordSys)

    sphere = batoid.Sphere(-sphereRadius)
    sphere.intersectInPlace(rays)

    w = np.where(1-rays.vignetted)[0]
    # Should potentially try to make the reference time w.r.t. the chief ray instead of the mean
    # of the good (unvignetted) rays.
    t0 = np.mean(rays.t[w])

    arr = np.ma.masked_array((t0-rays.t)/wavelength, mask=rays.vignetted).reshape(nx, nx)
    if _addedWF is not None:
        arr += _addedWF
    primitiveVectors = np.vstack([[optic.pupilSize/nx, 0], [0, optic.pupilSize/nx]])
    return batoid.Lattice(arr, primitiveVectors)


def reciprocalLatticeVectors(a1, a2, N):
    norm = 2*np.pi/(a1[0]*a2[1] - a1[1]*a2[0])/N
    b1 = norm*np.array([a2[1], a2[0]])
    b2 = norm*np.array([a1[1], a1[0]])
    return b1, b2


def fftPSF(optic, theta_x, theta_y, wavelength, nx=32, projection='postel', pad_factor=2,
           _addedWF=None):
    """Compute PSF using FFT.

    Parameters
    ----------
    optic : batoid.Optic
        Optic for which to compute wavefront.
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float, optional
        Wavelength in meters
    nx : int, optional
        Size of ray grid to generate to compute wavefront.  Default: 32
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.
    pad_factor : int, optional
        Factor by which to pad pupil array.  Default: 2

    Returns
    -------
    psf : batoid.Lattice
        A batoid.Lattice object containing the relative PSF values and
        the primitive lattice vectors of the focal plane grid.
    """
    L = optic.pupilSize*pad_factor
    # im_dtheta = wavelength / L
    wf = wavefront(optic, theta_x, theta_y, wavelength, nx, projection=projection, lattice=True,
                   _addedWF=_addedWF)
    wfarr = wf.array
    pad_size = nx*pad_factor
    expwf = np.zeros((pad_size, pad_size), dtype=np.complex128)
    start = pad_size//2-nx//2
    stop = pad_size//2+nx//2
    expwf[start:stop, start:stop][~wfarr.mask] = np.exp(2j*np.pi*wfarr[~wfarr.mask])
    psf = np.abs(np.fft.fftshift(np.fft.fft2(expwf)))**2

    primitiveU = wf.primitiveVectors
    primitiveK = dkdu(optic, theta_x, theta_y, wavelength, projection=projection).dot(primitiveU)
    primitiveX = np.vstack(reciprocalLatticeVectors(primitiveK[0], primitiveK[1], pad_size))

    return batoid.Lattice(psf, primitiveX)


def zernike(optic, theta_x, theta_y, wavelength, nx=32, projection='postel', jmax=22, eps=0.0,
            sphereRadius=None, _addedWF=None):
    import galsim

    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    rays = batoid.rayGrid(
        optic.backDist, optic.pupilSize,
        dirCos[0], dirCos[1], dirCos[2],
        nx, wavelength, 1.0, optic.inMedium
    )

    # Propagate to t=optic.backDist where rays are close to being
    # in the entrance pupil.  (TODO: actually intersect EP here?)
    rays.propagateInPlace(optic.backDist)
    orig_x = np.array(rays.x).reshape(nx,nx)
    orig_y = np.array(rays.y).reshape(nx,nx)

    wf = wavefront(optic, theta_x, theta_y, wavelength, nx=nx, projection=projection,
                   sphereRadius=sphereRadius, _addedWF=_addedWF)
    wfarr = wf.array
    w = ~wfarr.mask

    basis = galsim.zernike.zernikeBasis(
            jmax, orig_x[w], orig_y[w],
            R_outer=optic.pupilSize/2, R_inner=optic.pupilSize/2*eps
    )
    coefs, _, _, _ = np.linalg.lstsq(basis.T, wfarr[w], rcond=-1)

    return np.array(coefs)


def fpPosition(optic, theta_x, theta_y, wavelength, nx=32, projection='postel'):
    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    rays = batoid.rayGrid(
        optic.backDist, optic.pupilSize,
        dirCos[0], dirCos[1], dirCos[2],
        nx, wavelength, 1.0, optic.inMedium
    )
    optic.traceInPlace(rays)
    rays.trimVignettedInPlace()
    return np.mean(rays.x), np.mean(rays.y)
