import numpy as np


def normalized(*args):
    if len(args) == 1:
        args = np.array(*args)
    return args/np.linalg.norm(args)


def bilinear_fit(ux, uy, kx, ky):
    a = np.empty((len(ux), 3), dtype=float)
    a[:,0] = 1
    a[:,1] = ux
    a[:,2] = uy
    b = np.empty((len(ux), 2), dtype=float)
    b[:,0] = kx
    b[:,1] = ky
    x, _, _, _ = np.linalg.lstsq(a, b, rcond=-1)
    return x


def gnomicToDirCos(u, v):
    """Convert gnomic tangent plane projection u,v to direction cosines.

    Parameters
    ----------
    u, v : float
        Gnomic tangent plane coordinates in radians.

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0) and (alpha, beta, gamma) = (0,0,1),
    and u.x > 0, u.y=0, v.x=0, v.y > 0.
    """
    gamma = 1/np.sqrt(1.0 + u*u + v*v)
    alpha = u*gamma
    beta = v*gamma

    return alpha, beta, gamma


def dirCosToGnomic(alpha, beta, gamma):
    """Convert direction cosines to gnomic tangent plane projection.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Returns
    -------
    u, v : float
        Gnomic tangent plane coordinates in radians.

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0) and (alpha, beta, gamma) = (0,0,1)
    and u.x > 0, u.y=0, v.x=0, v.y > 0.
    """
    u = alpha / gamma
    v = beta / gamma

    return u, v


def gnomicToSpherical(u, v):
    """Convert gnomic tangent plane projection u, v to spherical coordinates.

    Parameters
    ----------
    u, v : float
        Gnomic tangent plane coordinates in radians.

    Returns
    -------
    phi : float
        Polar angle in radians
    theta : float
        Azimuthal angle in radians (always in [-pi, pi], and 0 by convention when phi=0)

    Notes
    -----
    The azimuthal angle is measured from +u through +v (CCW).
    """
    phi = np.arctan(np.sqrt(u*u + v*v))
    theta = np.arctan2(v, u)

    return phi, theta


def sphericalToGnomic(phi, theta):
    """Convert spherical coordiantes to gnomic tangent plane projection.

    Parameters
    ----------
    phi : float
        Polar angle in radians
    theta : float
        Azimuthal angle in radians

    Returns
    -------
    u, v : float
        Gnomic tangent plane coordinates in radians.

    Notes
    -----
    The azimuthal angle is measured from +u through +v (CCW).
    """
    tph = np.tan(phi)
    u = tph * np.cos(theta)
    v = tph * np.sin(theta)

    return u, v


def dirCosToSpherical(alpha, beta, gamma):
    """Convert direction cosines into spherical coordinates.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Returns
    -------
    phi : float
        Polar angle in radians
    theta : float
        Azimuthal angle in radians

    Notes
    -----
    The azimuthal angle is measured from the +alpha axis through the +beta axis (CCW).
    """
    phi = np.arccos(gamma)
    theta = np.arctan2(beta, alpha)

    return phi, theta


def sphericalToDirCos(phi, theta):
    """Convert spherical coordinates into direction cosines.

    Parameters
    ----------
    phi : float
        Polar angle in radians
    theta : float
        Azimuthal angle in radians

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The azimuthal angle is measured from the +alpha axis through the +beta axis (CCW).
    """
    r = np.sin(phi)
    alpha = r * np.cos(theta)
    beta = r * np.sin(theta)
    gamma = np.cos(phi)

    return alpha, beta, gamma


def dSphericalDGnomic(u, v):
    """Compute Jacobian of transformation from gnomic tangent plane coordinates to spherical
    coordinates.

    Parameters
    ----------
    u, v : float
        Gnomic tangent plane coordinates in radians.

    Returns
    -------
    jac : (2, 2) ndarray
        [[dphi/du, dphi/dv],
         [sin(phi) dtheta/du, sin(phi) dtheta/dv]]
    """
    rsqr = u*u + v*v
    r = np.sqrt(rsqr)
    den = (1+rsqr)*r
    sph = r/np.sqrt(1+rsqr)

    dphdu = u/den
    dphdv = v/den

    dthdu = -v/rsqr
    dthdv = u/rsqr

    return np.array([[dphdu, dphdv], [sph*dthdu, sph*dthdv]])


def dGnomicDSpherical(phi, theta):
    """Compute Jacobian of transformation from spherical coordinates to gnomic tangent plane
    coordinates.

    Parameters
    ----------
    phi : float
        Polar angle in radians
    theta : float
        Azimuthal angle in radians

    Returns
    -------
    jac : (2, 2) ndarray
        [[du/dphi, csc(phi) du/dtheta],
         [dv/dphi, csc(phi) dv/dtheta]]
    """
    sc2ph = np.cos(phi)**(-2)
    tph = np.tan(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cscph = 1/np.sin(phi)

    dudph = sc2ph * cth
    dvdph = sc2ph * sth

    dudth = -tph * sth
    dvdth = tph * cth

    return np.array([[dudph, cscph*dudth], [dvdph, cscph*dvdth]])


# http://stackoverflow.com/a/6849299
class lazy_property(object):
    """
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value
