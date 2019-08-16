import numpy as np
import batoid

from .utils import bilinear_fit, fieldToDirCos
from .psf import dkdu, reciprocalLatticeVectors


def huygensPSF(optic, theta_x=None, theta_y=None, wavelength=None, nx=None,
               projection='postel', dx=None, dy=None, nxOut=None, reference='mean'):
    from numbers import Real

    if dx is None:
        if (nx%2) == 0:
            primitiveU = np.array([[optic.pupilSize/(nx-2),0], [0, optic.pupilSize/(nx-2)]])
        else:
            primitiveU = np.array([[optic.pupilSize/(nx-1),0], [0, optic.pupilSize/(nx-1)]])
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
    dirCos = dirCos[0:2] + (-dirCos[2],)

    rays = batoid.RayVector.asGrid(
        optic.dist, wavelength,
        dirCos=dirCos, nx=nx, lx=optic.pupilSize,
        medium=optic.inMedium,
        interface=optic.entrancePupil
    )

    amplitudes = np.zeros((nxOut*pad_factor, nxOut*pad_factor), dtype=np.complex128)
    out = batoid.Lattice(np.zeros((nxOut*pad_factor, nxOut*pad_factor), dtype=float), primitiveX)

    rays, outCoordSys = optic.traceInPlace(rays)
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
    return batoid.Lattice(np.abs(amplitudes)**2, primitiveX)


def wavefront(optic, theta_x, theta_y, wavelength, nx=32, projection='postel',
              sphereRadius=None, reference='mean'):
    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    dirCos = dirCos[0:2]+(-dirCos[2],)
    rays = batoid.RayVector.asGrid(
        optic.dist, wavelength,
        nx=nx, lx=optic.pupilSize,
        dirCos=dirCos,
        medium=optic.inMedium,
        interface=optic.entrancePupil
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

    # Place vertex of reference sphere one radius length away from the intersection point.
    # So transform our rays into that coordinate system.
    transform = batoid.CoordTransform(
        batoid.globalCoordSys, batoid.CoordSys(point+np.array([0,0,sphereRadius]))
    )
    transform.applyForwardInPlace(rays)

    sphere = batoid.Sphere(-sphereRadius)
    sphere.intersectInPlace(rays)

    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        t0 = np.mean(rays.t[w])
    elif reference == 'chief':
        t0 = rays[cridx].t
    arr = np.ma.masked_array((t0-rays.t)/wavelength, mask=rays.vignetted).reshape(nx, nx)
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


def fftPSF(optic, theta_x, theta_y, wavelength, nx=32, projection='postel', pad_factor=2,
           sphereRadius=None, reference='mean', _addedWF=None):
    wf = wavefront(optic, theta_x, theta_y, wavelength, nx=nx, projection=projection,
                   sphereRadius=sphereRadius, reference=reference)
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


def zernike(optic, theta_x, theta_y, wavelength, nx=32, projection='postel',
            sphereRadius=None, lattice=False, reference='mean', jmax=22, eps=0.0):
    import galsim

    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    dirCos = dirCos[0:2]+(-dirCos[2],)
    rays = batoid.RayVector.asGrid(
        optic.dist, wavelength,
        nx=nx, lx=optic.pupilSize,
        dirCos=dirCos,
        medium=optic.inMedium,
        interface=optic.entrancePupil
    )

    # Propagate to entrance pupil to get positions
    transform = batoid.CoordTransform(batoid.globalCoordSys, optic.entrancePupil.coordSys)
    epRays = transform.applyForward(rays)
    optic.entrancePupil.surface.intersectInPlace(epRays)
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

    return np.array(coefs)


def zernikeGQ(optic, theta_x, theta_y, wavelength, nrings=6, nspokes=None,
              projection='postel', jmax=22, sphereRadius=None,
              reference='mean'):
    import galsim
    dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    dirCos = (dirCos[0], dirCos[1], -dirCos[2])
    rays = batoid.RayVector.asSpokes(
        optic.dist, wavelength,
        outer=optic.pupilSize/2,
        dirCos=dirCos,
        rings=nrings,
        spokes=nspokes,
        spacing='GQ',
        medium=optic.inMedium,
        interface=optic.entrancePupil
    )

    # Trace to entrancePupil to get points at which to evalue Zernikes
    transform = batoid.CoordTransform(batoid.globalCoordSys, optic.entrancePupil.coordSys)
    epRays = transform.applyForward(rays)
    optic.entrancePupil.surface.intersectInPlace(epRays)

    basis = galsim.zernike.zernikeBasis(
        jmax, epRays.x, epRays.y,
        R_outer=optic.pupilSize/2
    )

    if sphereRadius is None:
        sphereRadius = optic.sphereRadius

    optic.traceInPlace(rays, outCoordSys=batoid.globalCoordSys)
    if reference == 'mean':
        w = np.where(1-rays.vignetted)[0]
        point = np.mean(rays.r[w], axis=0)
    elif reference == 'chief':
        chiefRay = batoid.Ray.fromPupil(
            0.0, 0.0,
            optic.dist, wavelength,
            dirCos=dirCos,
            medium=optic.inMedium,
            interface=optic.entrancePupil
        )
        optic.traceInPlace(chiefRay, outCoordSys=batoid.globalCoordSys)
        point = chiefRay.r

    # Place vertex of reference sphere one radius length away from the intersection point.
    # So transform our rays into that coordinate system.
    transform = batoid.CoordTransform(
        batoid.globalCoordSys, batoid.CoordSys(point+np.array([0,0,sphereRadius]))
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

    # Zernike coefficients are flux-weighted dot products of relative phases with basis.
    return np.dot(basis, (t0-rays.t)/wavelength*rays.flux)/np.pi
