import numpy as np
import batoid

from .utils import bilinear_fit, fieldToDirCos
from .psf import dkdu, reciprocalLatticeVectors


def huygensPSF(optic, theta_x, theta_y, wavelength,
               projection='postel', nx=None, dx=None, dy=None,
               nxOut=None, reference='mean'):
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
            reciprocalLatticeVectors(
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

    optic.traceInPlace(rays)
    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        point = np.mean(rays.r[w], axis=0)
    elif reference == 'chief':
        cridx = (nx//2)*nx+nx//2 if (nx%2)==0 else (nx*nx-1)//2
        point = rays[cridx].r
    rays.trimVignettedInPlace()
    # Need transpose to conform to numpy [y,x] ordering convention
    xs = out.coords[..., 0].T + point[0]
    ys = out.coords[..., 1].T + point[1]
    zs = np.zeros_like(xs)

    points = np.concatenate([aux[..., None] for aux in (xs, ys, zs)], axis=-1)
    time = rays[0].t
    for idx in np.ndindex(amplitudes.shape):
        amplitudes[idx] = rays.sumAmplitude(points[idx], time)
    out.array = np.abs(amplitudes)**2
    return out


def wavefront(optic, theta_x, theta_y, wavelength,
              projection='postel', nx=32,
              sphereRadius=None, reference='mean'):
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

    optic.traceInPlace(rays, outCoordSys=batoid.globalCoordSys)
    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        point = np.mean(rays.r[w], axis=0)
    elif reference == 'chief':
        cridx = (nx//2)*nx+nx//2 if (nx%2)==0 else (nx*nx-1)//2
        point = rays[cridx].r

    # Place vertex of reference sphere one radius length away from the
    # intersection point.  So transform our rays into that coordinate system.
    transform = batoid.CoordTransform(
        batoid.globalCoordSys,
        batoid.CoordSys(point+np.array([0,0,sphereRadius]))
    )
    transform.applyForwardInPlace(rays)

    sphere = batoid.Sphere(-sphereRadius)
    sphere.intersectInPlace(rays)

    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        t0 = np.mean(rays.t[w])
    elif reference == 'chief':
        t0 = rays[cridx].t
    arr = np.ma.masked_array(
        (t0-rays.t)/wavelength,
        mask=rays.vignetted
    ).reshape(nx, nx)
    if (nx%2) == 0:
        primitiveVectors = np.vstack(
            [[optic.pupilSize/(nx-2), 0],
             [0, optic.pupilSize/(nx-2)]]
        )
    else:
        primitiveVectors = np.vstack(
            [[optic.pupilSize/(nx-1), 0],
             [0, optic.pupilSize/(nx-1)]]
        )
    return batoid.Lattice(arr, primitiveVectors)


def fftPSF(optic, theta_x, theta_y, wavelength,
           projection='postel', nx=32, pad_factor=2,
           sphereRadius=None, reference='mean', _addedWF=None):
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
        reciprocalLatticeVectors(primitiveK[0], primitiveK[1], pad_size)
    )

    return batoid.Lattice(psf, primitiveX)


def zernike(optic, theta_x, theta_y, wavelength,
            projection='postel', nx=32,
            sphereRadius=None, reference='mean', jmax=22, eps=0.0):
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
    transform = batoid.CoordTransform(
        batoid.globalCoordSys, optic.stopSurface.coordSys
    )
    epRays = transform.applyForward(rays)
    optic.stopSurface.surface.intersectInPlace(epRays)
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


def zernikeGQ(optic, theta_x, theta_y, wavelength,
              projection='postel', rings=6, spokes=None,
              sphereRadius=None, reference='mean',
              jmax=22, eps=None):
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

    inner = 0.0 if eps is None else eps*optic.pupilSize/2
    rays = batoid.RayVector.asSpokes(
        optic=optic, wavelength=wavelength,
        inner=inner,
        dirCos=dirCos,
        rings=rings,
        spokes=spokes,
        spacing='GQ',
    )

    # Trace to stopSurface to get points at which to evalue Zernikes
    transform = batoid.CoordTransform(
        batoid.globalCoordSys, optic.stopSurface.coordSys
    )
    epRays = transform.applyForward(rays)
    optic.stopSurface.surface.intersectInPlace(epRays)

    basis = galsim.zernike.zernikeBasis(
        jmax, epRays.x, epRays.y,
        R_outer=optic.pupilSize/2,
        R_inner=inner
    )

    if sphereRadius is None:
        sphereRadius = optic.sphereRadius

    optic.traceInPlace(rays, outCoordSys=batoid.globalCoordSys)

    if np.any(rays.failed):
        raise ValueError(
            "Cannot compute zernike with Gaussian Quadrature with failed rays."
        )
    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        point = np.mean(rays.r[w], axis=0)
    elif reference == 'chief':
        chiefRay = batoid.Ray.fromStop(
            0.0, 0.0,
            backDist=optic.backDist, wavelength=wavelength,
            dirCos=dirCos,
            medium=optic.inMedium,
            stopSurface=optic.stopSurface
        )
        optic.traceInPlace(chiefRay, outCoordSys=batoid.globalCoordSys)
        point = chiefRay.r

    # Place vertex of reference sphere one radius length away from the
    # intersection point.  So transform our rays into that coordinate system.
    transform = batoid.CoordTransform(
        batoid.globalCoordSys,
        batoid.CoordSys(point+np.array([0,0,sphereRadius]))
    )
    transform.applyForwardInPlace(rays)

    sphere = batoid.Sphere(-sphereRadius)
    sphere.intersectInPlace(rays)

    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        t0 = np.mean(rays.t[w])
    elif reference == 'chief':
        transform.applyForwardInPlace(chiefRay)
        sphere.intersectInPlace(chiefRay)
        t0 = chiefRay.t

    # Zernike coefficients are flux-weighted dot products of relative phases
    # with basis.
    return np.dot(basis, (t0-rays.t)/wavelength*rays.flux)/np.pi
