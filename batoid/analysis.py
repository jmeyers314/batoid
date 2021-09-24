import numpy as np
import batoid

from .utils import bilinear_fit, fieldToDirCos


def _reciprocalLatticeVectors(a1, a2, N):
    norm = 2*np.pi/(a1[0]*a2[1] - a1[1]*a2[0])/N
    b1 = norm*np.array([a2[1], a2[0]])
    b2 = norm*np.array([a1[1], a1[0]])
    return b1, b2


def dkdu(
    optic, theta_x, theta_y, wavelength, nrad=16, naz=16, projection='postel'
):
    """Calculate derivative of outgoing ray k-vector with respect to incoming
    ray pupil coordinate.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    nrad : int, optional
        Number of ray radii to use.  (see RayVector.asPolar())
    naz : int, optional
        Approximate number of azimuthal angles in outermost ring.  (see
        RayVector.asPolar())
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.

    Returns
    -------
    dkdu : (2, 2), ndarray
        Jacobian transformation matrix for converting between (kx, ky) of rays
        impacting the focal plane and initial pupil plane coordinate.
    """

    rays = batoid.RayVector.asPolar(
        optic=optic, theta_x=theta_x, theta_y=theta_y, wavelength=wavelength,
        projection=projection, nrad=nrad, naz=naz
    )
    ux = np.array(rays.x)
    uy = np.array(rays.y)

    optic.trace(rays)
    w = ~rays.vignetted
    soln = bilinear_fit(ux[w], uy[w], rays.kx[w], rays.ky[w])
    return soln[1:]


def drdth(
    optic, theta_x, theta_y, wavelength, nrad=50, naz=300, projection='postel'
):
    """Calculate derivative of focal plane coord with respect to field angle.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    nrad : int, optional
        Number of ray radii to use.  (see RayVector.asPolar())
    naz : int, optional
        Approximate number of azimuthal angles in outermost ring.  (see
        RayVector.asPolar())
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.

    Returns
    -------
    drdth : (2, 2), ndarray
        Jacobian transformation matrix for converting between (theta_x, theta_y)
        and (x, y) on the focal plane.

    Notes
    -----
        This is the Jacobian of pixels -> tangent plane, (and importantly, not
        pixels -> ra/dec).  It should be *close* to the inverse plate scale
        though, especially near the center of the tangent plane projection.
    """
    # We just use a finite difference approach here.
    dth = 1e-5

    # Make direction cosine vectors
    nominalCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    dthxCos = fieldToDirCos(theta_x + dth, theta_y, projection=projection)
    dthyCos = fieldToDirCos(theta_x, theta_y + dth, projection=projection)

    rays = batoid.RayVector.asPolar(
        optic=optic, wavelength=wavelength,
        dirCos=nominalCos,
        nrad=nrad, naz=naz
    )
    rays_x = batoid.RayVector.asPolar(
        optic=optic, wavelength=wavelength,
        dirCos=dthxCos,
        nrad=nrad, naz=naz
    )
    rays_y = batoid.RayVector.asPolar(
        optic=optic, wavelength=wavelength,
        dirCos=dthyCos,
        nrad=nrad, naz=naz
    )

    optic.trace(rays)
    optic.trace(rays_x)
    optic.trace(rays_y)

    w = ~rays.vignetted
    mx = np.mean(rays.x[w])
    my = np.mean(rays.y[w])

    # meters / radian
    drx_dthx = (np.mean(rays_x.x[w]) - mx)/dth
    drx_dthy = (np.mean(rays_y.x[w]) - mx)/dth
    dry_dthx = (np.mean(rays_x.y[w]) - my)/dth
    dry_dthy = (np.mean(rays_y.y[w]) - my)/dth

    return np.array([[drx_dthx, drx_dthy], [dry_dthx, dry_dthy]])


def dthdr(
    optic, theta_x, theta_y, wavelength, nrad=50, naz=300, projection='postel'
):
    """Calculate derivative of field angle with respect to focal plane
    coordinate.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    nrad : int, optional
        Number of ray radii to use.  (see RayVector.asPolar())
    naz : int, optional
        Approximate number of azimuthal angles in outermost ring.  (see
        RayVector.asPolar())
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.

    Returns
    -------
    dthdr : (2, 2), ndarray
        Jacobian transformation matrix for converting between (x, y) on the
        focal plane and field angle (theta_x, theta_y).

    Notes
    -----
        This is the Jacobian of tangent plane -> pixels, (and importantly, not
        ra/dec -> pixels). It should be *close* to the plate scale though,
        especially near the center of the tangent plane projection.
    """
    return np.linalg.inv(
        drdth(
            optic, theta_x, theta_y, wavelength,
            nrad=nrad, naz=naz, projection=projection
        )
    )


def huygensPSF(
    optic, theta_x, theta_y, wavelength,
    projection='postel', nx=None, dx=None, dy=None,
    nxOut=None, reference='mean'
):
    r"""Compute a PSF via the Huygens construction.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.
    nx : int, optional
        Size of ray grid to use.
    dx, dy : float, optional
        Lattice scales to use for PSF evaluation locations.  Default, use
        fftPSF lattice.
    nxOut : int, optional
        Size of the output lattice.  Default is to use nx.
    reference : {'chief', 'mean'}
        If 'chief', then center the output lattice where the chief ray
        intersects the focal plane.  If 'mean', then center at the mean
        non-vignetted ray intersection.

    Returns
    -------
    psf : batoid.Lattice
        The PSF.

    Notes
    -----
    The Huygens construction is to evaluate the PSF as

    .. math::

        I(x) \propto \sum_u \exp(i \phi(u)) \exp(i k(u) \cdot r)

    The :math:`u` are assumed to uniformly sample the entrance pupil, but not
    include any rays that get vignetted before they reach the focal plane.  The
    :math:`\phi` s are the phases of the exit rays evaluated at a single
    arbitrary time.  The :math:`k(u)` indicates the conversion of the uniform
    entrance pupil samples into nearly (though not exactly) uniform samples in
    k-space of the output rays.

    The output locations where the PSF is evaluated are governed by ``dx``,
    ``dy``, and ``nx``.  If ``dx`` and ``dy`` are None, then the same lattice
    as in fftPSF will be used.  If ``dx`` and ``dy`` are scalars, then a
    lattice with primitive vectors ``[dx, 0]`` and ``[0, dy]`` will be used.
    If ``dx`` and ``dy`` are 2-vectors, then those will be the primitive
    vectors of the output lattice.
    """
    from numbers import Real

    if dx is None:
        if (nx%2) == 0:
            primitiveU = np.array(
                [[optic.pupilSize/(nx-2),0],
                 [0, optic.pupilSize/(nx-2)]]
            )
        else:
            primitiveU = np.array(
                [[optic.pupilSize/(nx-1),0],
                 [0, optic.pupilSize/(nx-1)]]
            )
        primitiveK = dkdu(
            optic, theta_x, theta_y, wavelength,
            projection=projection
        ).dot(primitiveU)
        pad_factor = 2
        primitiveX = np.vstack(
            _reciprocalLatticeVectors(
                primitiveK[0], primitiveK[1], pad_factor*nx
            )
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

    rays = batoid.RayVector.asGrid(
        optic=optic, wavelength=wavelength,
        dirCos=dirCos, nx=nx
    )

    amplitudes = np.zeros(
        (nxOut*pad_factor, nxOut*pad_factor),
        dtype=np.complex128
    )
    out = batoid.Lattice(
        np.zeros((nxOut*pad_factor, nxOut*pad_factor), dtype=float),
        primitiveX
    )

    optic.trace(rays)
    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        point = np.mean(rays.r[w], axis=0)
    elif reference == 'chief':
        cridx = (nx//2)*nx+nx//2 if (nx%2)==0 else (nx*nx-1)//2
        point = rays.r[cridx]
    # Need transpose to conform to numpy [y,x] ordering convention
    xs = out.coords[..., 0].T + point[0]
    ys = out.coords[..., 1].T + point[1]
    zs = np.zeros_like(xs)

    points = np.concatenate([aux[..., None] for aux in (xs, ys, zs)], axis=-1)
    time = rays.t[0]
    for idx in np.ndindex(amplitudes.shape):
        amplitudes[idx] = rays.sumAmplitude(points[idx], time)
    out.array = np.abs(amplitudes)**2
    return out


def wavefront(
    optic, theta_x, theta_y, wavelength,
    projection='postel', nx=32,
    sphereRadius=None, reference='mean'
):
    """Compute wavefront.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.
    nx : int, optional
        Size of ray grid to use.
    sphereRadius : float, optional
        The radius of the reference sphere.  Nominally this should be set to
        the distance to the exit pupil, though the calculation is usually not
        very sensitive to this.  Many of the telescopes that come with batoid
        have values for this set in their yaml files, which will be used if
        this is None.
    reference : {'chief', 'mean'}
        If 'chief', then center the output lattice where the chief ray
        intersects the focal plane.  If 'mean', then center at the mean
        non-vignetted ray intersection.

    Returns
    -------
    wavefront : batoid.Lattice
        A batoid.Lattice object containing the wavefront values in waves and
        the primitive lattice vectors of the entrance pupil grid in meters.
    """
    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    rays = batoid.RayVector.asGrid(
        optic=optic, wavelength=wavelength,
        nx=nx, dirCos=dirCos
    )
    if sphereRadius is None:
        sphereRadius = optic.sphereRadius

    optic.trace(rays)
    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        point = np.mean(rays.r[w], axis=0)
    elif reference == 'chief':
        cridx = (nx//2)*nx+nx//2 if (nx%2)==0 else (nx*nx-1)//2
        point = rays.r[cridx]
    # Place vertex of reference sphere one radius length away from the
    # intersection point.  So transform our rays into that coordinate system.
    targetCoordSys = rays.coordSys.shiftLocal(
        point+np.array([0,0,sphereRadius])
    )
    rays.toCoordSys(targetCoordSys)

    sphere = batoid.Sphere(-sphereRadius)
    sphere.intersect(rays)

    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        t0 = np.mean(rays.t[w])
    elif reference == 'chief':
        t0 = rays.t[cridx]
    arr = np.ma.masked_array(
        (t0-rays.t)/wavelength,
        mask=rays.vignetted
    ).reshape(nx, nx)
    if (nx%2) == 0:
        primitiveU = np.vstack(
            [[optic.pupilSize/(nx-2), 0],
             [0, optic.pupilSize/(nx-2)]]
        )
    else:
        primitiveU = np.vstack(
            [[optic.pupilSize/(nx-1), 0],
             [0, optic.pupilSize/(nx-1)]]
        )
    return batoid.Lattice(arr, primitiveU)


def spot(
    optic, theta_x, theta_y, wavelength,
    projection='postel', nx=32, reference='mean'
):
    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    rays = batoid.RayVector.asGrid(
        optic=optic, wavelength=wavelength,
        nx=nx, dirCos=dirCos
    )
    optic.trace(rays)
    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        point = np.mean(rays.r[w], axis=0)
    elif reference == 'chief':
        cridx = (nx//2)*nx+nx//2 if (nx%2)==0 else (nx*nx-1)//2
        point = rays[cridx].r
    else:
        point = [0,0,0]
    targetCoordSys = rays.coordSys.shiftLocal(point)
    rays.toCoordSys(targetCoordSys)

    w = ~rays.vignetted
    return rays.x[w], rays.y[w]

def fftPSF(
    optic, theta_x, theta_y, wavelength,
    projection='postel', nx=32, pad_factor=2,
    sphereRadius=None, reference='mean'
):
    """Compute PSF using FFT.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.
    nx : int, optional
        Size of ray grid to use.
    pad_factor : int, optional
        Factor by which to pad pupil array.  Default: 2
    sphereRadius : float, optional
        The radius of the reference sphere.  Nominally this should be set to
        the distance to the exit pupil, though the calculation is usually not
        very sensitive to this.  Many of the telescopes that come with batoid
        have values for this set in their yaml files, which will be used if
        this is None.
    reference : {'chief', 'mean'}
        If 'chief', then center the output lattice where the chief ray
        intersects the focal plane.  If 'mean', then center at the mean
        non-vignetted ray intersection.

    Returns
    -------
    psf : batoid.Lattice
        A batoid.Lattice object containing the relative PSF values and
        the primitive lattice vectors of the focal plane grid.
    """
    wf = wavefront(
        optic, theta_x, theta_y, wavelength,
        nx=nx, projection=projection,
        sphereRadius=sphereRadius, reference=reference
    )
    wfarr = wf.array
    pad_size = nx*pad_factor
    expwf = np.zeros((pad_size, pad_size), dtype=np.complex128)
    start = pad_size//2-nx//2
    stop = pad_size//2+nx//2
    expwf[start:stop, start:stop][~wfarr.mask] = \
        np.exp(2j*np.pi*wfarr[~wfarr.mask])
    psf = np.abs(np.fft.fftshift(np.fft.fft2(expwf)))**2
    primitiveU = wf.primitiveVectors
    primitiveK = dkdu(
        optic, theta_x, theta_y, wavelength,
        projection=projection
    ).dot(primitiveU)
    primitiveX = np.vstack(
        _reciprocalLatticeVectors(primitiveK[0], primitiveK[1], pad_size)
    )

    return batoid.Lattice(psf, primitiveX)


def zernike(
    optic, theta_x, theta_y, wavelength,
    projection='postel', nx=32,
    sphereRadius=None, reference='mean', jmax=22, eps=0.0
):
    """Compute Zernike polynomial decomposition of the wavefront.

    This calculation propagates a square grid of rays to the exit pupil
    reference sphere where the wavefront is computed.  The optical path
    differences of non-vignetted rays are then fit to Zernike polynomial
    coefficients numerically.


    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.
    nx : int, optional
        Size of ray grid to use.
    sphereRadius : float, optional
        The radius of the reference sphere.  Nominally this should be set to
        the distance to the exit pupil, though the calculation is usually not
        very sensitive to this.  Many of the telescopes that come with batoid
        have values for this set in their yaml files, which will be used if
        this is None.
    reference : {'chief', 'mean'}
        If 'chief', then center the output lattice where the chief ray
        intersects the focal plane.  If 'mean', then center at the mean
        non-vignetted ray intersection.
    jmax : int, optional
        Number of coefficients to compute.  Default: 12.
    eps : float, optional
        Use annular Zernike polynomials with this fractional inner radius.
        Default: 0.0.

    Returns
    -------
    zernikes : array
        Zernike polynomial coefficients.

    Notes
    -----
    Zernike coefficients are indexed following the Noll convention.
    Additionally, since python lists start at 0, but the Noll convention starts
    at 1, the 0-th index of the returned array is meaningless.  I.e.,
    zernikes[1] is piston, zernikes[4] is defocus, and so on...

    Also, since Zernike polynomials are orthogonal over a circle or annulus,
    but vignetting may make the actual wavefront region of support something
    different, the values of fit coefficients can depend on the total number of
    coefficients being fit.  For example, the j=4 (defocus) coefficient may
    depend on whether jmax=11 and jmax=21 (or some other value).  See the
    zernikeGQ function for an alternative algorithm that is independent of
    jmax.
    """
    import galsim

    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    rays = batoid.RayVector.asGrid(
        optic=optic, wavelength=wavelength,
        nx=nx, dirCos=dirCos
    )
    # Propagate to entrance pupil to get positions
    epRays = rays.toCoordSys(optic.stopSurface.coordSys)
    optic.stopSurface.surface.intersect(epRays)
    orig_x = np.array(epRays.x).reshape(nx, nx)
    orig_y = np.array(epRays.y).reshape(nx, nx)

    wf = wavefront(optic, theta_x, theta_y, wavelength, nx=nx,
                   projection=projection, sphereRadius=sphereRadius,
                   reference=reference)
    wfarr = wf.array
    w = ~wfarr.mask

    basis = galsim.zernike.zernikeBasis(
        jmax, orig_x[w], orig_y[w],
        R_outer=optic.pupilSize/2, R_inner=optic.pupilSize/2*eps
    )
    coefs, _, _, _ = np.linalg.lstsq(basis.T, wfarr[w], rcond=-1)
    # coefs[0] is meaningless, so always set to 0.0 for comparison consistency
    coefs[0] = 0.0
    return np.array(coefs)


def zernikeGQ(
    optic, theta_x, theta_y, wavelength,
    projection='postel', rings=6, spokes=None,
    sphereRadius=None, reference='mean',
    jmax=22, eps=0.0
):
    r"""Compute Zernike polynomial decomposition of the wavefront.

    This calculation uses Gaussian Quadrature points and weights to compute the
    Zernike decomposition of the wavefront.  The wavefront values at the GQ
    points are determined by tracing from the entrance pupil to the exit pupil
    reference sphere, and evaluating the optical path
    differences on this sphere.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    theta_x, theta_y : float
        Field angle in radians
    wavelength : float
        Wavelength in meters
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.
    rings : int, optional
        Number of Gaussian quadrature rings to use.  Default: 6.
    spokes : int, optional
        Number of Gaussian quadrature spokes to use.  Default: 2*rings + 1
    sphereRadius : float, optional
        The radius of the reference sphere.  Nominally this should be set to
        the distance to the exit pupil, though the calculation is usually not
        very sensitive to this.  Many of the telescopes that come with batoid
        have values for this set in their yaml files, which will be used if
        this is None.
    reference : {'chief', 'mean'}
        If 'chief', then center the output lattice where the chief ray
        intersects the focal plane.  If 'mean', then center at the mean
        non-vignetted ray intersection.
    jmax : int, optional
        Number of coefficients to compute.  Default: 22.
    eps : float, optional
        Use annular Zernike polynomials with this fractional inner radius.
        Default: 0.0.

    Returns
    -------
    zernikes : array
        Zernike polynomial coefficients.

    Notes
    -----
    Zernike coefficients are indexed following the Noll convention.
    Additionally, since python lists start at 0, but the Noll convention starts
    at 1, the 0-th index of the returned array is meaningless.  I.e.,
    zernikes[1] is piston, zernikes[4] is defocus, and so on...

    This algorithm takes advantage of the fact that Zernike polynomials are
    orthogonal on a circle or annulus to compute the value of individual
    coefficients via an integral:

    .. math::

        a_j \propto \int Z_j(x, y) W(x, y) dx dy

    This integral is approximated using Gaussian quadrature.  Since the above
    integral depends on orthogonality, the wavefront must be decomposed over a
    circular or annular region of support.  As such, this algorithm is
    unaffected by vignetting.  It is required that no rays fail to be traced,
    even in vignetted regions.
    """
    import galsim
    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)

    inner = eps*optic.pupilSize/2
    rays = batoid.RayVector.asSpokes(
        optic=optic, wavelength=wavelength,
        inner=inner,
        dirCos=dirCos,
        rings=rings,
        spokes=spokes,
        spacing='GQ',
    )

    # Trace to stopSurface to get points at which to evaluate Zernikes
    epRays = rays.copy().toCoordSys(optic.stopSurface.coordSys)
    optic.stopSurface.surface.intersect(epRays)

    basis = galsim.zernike.zernikeBasis(
        jmax, epRays.x, epRays.y,
        R_outer=optic.pupilSize/2,
        R_inner=inner
    )

    if sphereRadius is None:
        sphereRadius = optic.sphereRadius

    optic.trace(rays)

    if np.any(rays.failed):
        raise ValueError(
            "Cannot compute zernike with Gaussian Quadrature with failed rays."
        )
    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        point = np.mean(rays.r[w], axis=0)
    elif reference == 'chief':
        chiefRay = batoid.RayVector.fromStop(
            0.0, 0.0,
            backDist=optic.backDist, wavelength=wavelength,
            dirCos=dirCos,
            medium=optic.inMedium,
            stopSurface=optic.stopSurface
        )
        optic.trace(chiefRay)
        point = chiefRay.r[0]

    # Place vertex of reference sphere one radius length away from the
    # intersection point.  So transform our rays into that coordinate system.
    targetCoordSys = rays.coordSys.shiftLocal(
        point+np.array([0,0,sphereRadius])
    )
    rays.toCoordSys(targetCoordSys)

    sphere = batoid.Sphere(-sphereRadius)
    sphere.intersect(rays)

    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        t0 = np.mean(rays.t[w])
    elif reference == 'chief':
        chiefRay.toCoordSys(targetCoordSys)
        sphere.intersect(chiefRay)
        t0 = chiefRay.t

    # Zernike coefficients are flux-weighted dot products of relative phases
    # with basis.
    area = np.pi*(1.-eps**2)
    return np.dot(basis, (t0-rays.t)/wavelength*rays.flux)/area


def _dZernikeBasis(jmax, x, y, R_outer=1.0, R_inner=0.0):
    import galsim
    xout = np.zeros(tuple((jmax+1,)+x.shape), dtype=float)
    yout = np.zeros_like(xout)
    for j in range(2, 1+jmax):
        xout[j,:] = galsim.zernike.Zernike(
            [0]*j+[1], R_outer=R_outer, R_inner=R_inner
        ).gradX(x, y)
        yout[j,:] = galsim.zernike.Zernike(
            [0]*j+[1], R_outer=R_outer, R_inner=R_inner
        ).gradY(x, y)
    return np.array([xout, yout])


def zernikeTransverseAberration(
    optic, theta_x, theta_y, wavelength,
    projection='postel', nrad=10, naz=60,
    reference='mean', jmax=22, eps=0.0
):
    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    rays = batoid.RayVector.asPolar(
        optic=optic,
        wavelength=wavelength,
        dirCos=dirCos,
        nrad=nrad, naz=naz
    )
    # Propagate to entrance pupil to get positions
    epRays = rays.copy().toCoordSys(optic.stopSurface.coordSys)
    optic.stopSurface.surface.intersect(epRays)
    u = np.array(epRays.x)
    v = np.array(epRays.y)

    rays = optic.trace(rays)
    w = ~rays.vignetted
    if reference == 'mean':
        point = np.mean(rays.r[w], axis=0)
    elif reference == 'chief':
        chief = batoid.RayVector.fromStop(
            0, 0, optic, wavelength=wavelength,
            dirCos=dirCos
        )
        optic.trace(chief)
        point = chief.r[0]
    x = rays.x - point[0]
    y = rays.y - point[1]

    # We may wish to revisit defining/using the focal length this way in the
    # future.
    focalLength = np.sqrt(np.linalg.det(drdth(
        optic, theta_x, theta_y, wavelength, projection=projection
    )))

    dzb = _dZernikeBasis(
        jmax, u[w], v[w], optic.pupilSize/2, eps*optic.pupilSize/2
    )
    a = np.hstack(dzb).T
    b = np.hstack([x[w], y[w]])
    r, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return -r/focalLength/wavelength


def doubleZernike(
    optic, field, wavelength, rings=6, spokes=None, kmax=22,
    **kwargs
):
    r"""Compute double Zernike polynomial decomposition of the wavefront.

    The double Zernike decomposition describes both the focal and pupil
    variation of the wavefront.  More specifically:

    .. math::

        W(u, \theta) = \sum_{jk} a_{jk} Z_j(u), Z_k(\theta)

    where :math:`u` indicates the pupil coordinate and :math:`\theta` indicates
    the field angle coordinate.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    field : float
        Outer field angle radius in radians.
    wavelength : float
        Wavelength in meters
    rings : int, optional
        Number of Gaussian quadrature rings to use.  Default: 6.
    spokes : int, optional
        Number of Gaussian quadrature spokes to use.  Default: 2*rings + 1
    kmax : int, optional
        Number of focal coefficients to compute.  Default: 22.
    **kwargs : dict
        Keyword arguments to pass to `zernikeGQ`.

    Returns
    -------
    dzs : array
        Double Zernike polynomial coefficients.
    """
    if spokes is None:
        spokes = 2*rings+1
    Li, w = np.polynomial.legendre.leggauss(rings)
    radii = np.sqrt((1+Li)/2)*field
    w *= np.pi/(2*spokes)
    azs = np.linspace(0, 2*np.pi, spokes, endpoint=False)
    radii, azs = np.meshgrid(radii, azs)
    w = np.broadcast_to(w, radii.shape)
    radii = radii.ravel()
    azs = azs.ravel()
    w = w.ravel()

    thx = radii * np.cos(azs)
    thy = radii * np.sin(azs)
    coefs = []
    for thx_, thy_ in zip(thx, thy):
        coefs.append(zernikeGQ(
            optic, thx_, thy_, wavelength,
            rings=rings, spokes=spokes,
            **kwargs
        ))
    coefs = np.array(coefs)

    import galsim
    basis = galsim.zernike.zernikeBasis(
        kmax, thx, thy, R_outer=field
    )
    dzs = np.dot(basis, coefs*w[:,None])/np.pi
    return dzs


def _closestApproach(P, u, Q, v):
    """Compute position along line P + u t of closest approach to line Q + v t

    Parameters
    ----------
    P, Q : ndarray
        Points on lines
    u, v : ndarray
        Direction cosines of lines

    Returns
    -------
    Pc : ndarray
        Closest approach point.
    """
    # Follows http://geomalgorithms.com/a07-_distance.html
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    w0 = P - Q
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    den = a*c - b*b
    if den == 0:
        raise ValueError("Lines are parallel")
    sc = (b*e - c*d)/den
    return P + sc*u


def exitPupilPos(optic, wavelength, smallAngle=np.deg2rad(1./3600)):
    """Compute position of the exit pupil.

    Traces a collection of small angle chief rays into object space, and then
    finds the mean of their closest approaches to the optic axis.  Possibly only
    accurate for axially symmetric optical systems.

    Parameters
    ----------
    optic : batoid.Optic
        Optical system
    wavelength : float
        Wavelength in meters
    smallAngle : float, optional
        Angle in radians from which to search for the exit pupil position.

    Returns
    -------
    Location of exit pupil in global coordinates.
    """
    thx = np.array([0, 0, smallAngle, -smallAngle])
    thy = np.array([smallAngle, -smallAngle, 0, 0])
    rays = batoid.RayVector.fromFieldAngles(
        thx, thy,
        optic=optic,
        wavelength=wavelength
    )
    optic.trace(rays)
    rays.toCoordSys(batoid.globalCoordSys)
    # Assume last intersection places rays into "object" space.
    # Now find closest approach to optic axis.
    ps = []
    for ray in rays:
        ps.append(_closestApproach(
            ray.r[0],
            ray.v[0],
            np.array([0., 0., 0.]),
            np.array([0., 0., 1.0])
        ))
    return np.mean(ps, axis=0)
