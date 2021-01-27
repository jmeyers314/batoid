import warnings
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


def gnomonicToDirCos(u, v):
    """Convert gnomonic tangent plane projection u,v to direction cosines.

    Parameters
    ----------
    u, v : float
        Gnomonic tangent plane coordinates in radians.

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    gamma = 1/np.sqrt(1.0 + u*u + v*v)
    alpha = u*gamma
    beta = v*gamma

    return alpha, beta, -gamma


def dirCosToGnomonic(alpha, beta, gamma):
    """Convert direction cosines to gnomonic tangent plane projection.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Returns
    -------
    u, v : float
        Gnomonic tangent plane coordinates in radians.

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    u = -alpha / gamma
    v = -beta / gamma

    return u, v


def postelToDirCos(u, v):
    """Convert Postel azimuthal equidistant tangent plane projection u,v to
    direction cosines.

    Parameters
    ----------
    u, v : float
        Postel tangent plane coordinates in radians.

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    rho = np.sqrt(u*u + v*v)
    wZero = (rho == 0.0)
    try:
        if wZero:
            return 0.0, 0.0, -1.0
    except ValueError:
        pass
    srho = np.sin(rho)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha = u/rho*srho
        beta = v/rho*srho
    gamma = -np.cos(rho)
    if np.any(wZero):
        alpha[wZero] = 0.0
        beta[wZero] = 0.0
        gamma[wZero] = -1.0

    return alpha, beta, gamma


def dirCosToPostel(alpha, beta, gamma):
    """Convert direction cosines to Postel azimuthal equidistant tangent plane
    projection.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Returns
    -------
    u, v : float
        Postel tangent plane coordinates in radians.

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    wZero = (gamma == -1)
    try:
        if wZero:
            return 0.0, 0.0
    except ValueError:
        pass
    rho = np.arccos(-gamma)
    srho = np.sin(rho)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = alpha*rho/srho
        v = beta*rho/srho
    if np.any(wZero):
        u[wZero] = 0.0
        v[wZero] = 0.0
    return u, v


def zemaxToDirCos(u, v):
    """Convert Zemax field angles u,v to direction cosines.

    Parameters
    ----------
    u, v : float
        Zemax field angles in radians.

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.

    The Zemax field angle convention is not rotationally invariant.  The
    z-direction cosine for (u, v) = (0, 1) does not equal the z-direction
    cosine for (u, v) = (0.6, 0.8).
    """
    tanu = np.tan(u)
    tanv = np.tan(v)
    norm = np.sqrt(1 + tanu*tanu + tanv*tanv)
    return tanu/norm, tanv/norm, -1/norm


def dirCosToZemax(alpha, beta, gamma):
    """Convert direction cosines to Postel azimuthal equidistant tangent plane
    projection.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Returns
    -------
    u, v : float
        Postel tangent plane coordinates in radians.

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.

    The Zemax field angle convention is not rotationally invariant.  The
    z-direction cosine for (u, v) = (0, 1) does not equal the z-direction
    cosine for (u, v) = (0.6, 0.8).
    """
    return np.arctan(-alpha/gamma), np.arctan(-beta/gamma)


def stereographicToDirCos(u, v):
    """Convert stereographic tangent plane projection u,v to direction cosines.

    Parameters
    ----------
    u, v : float
        Stereographic tangent plane coordinates in radians.

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    rho = np.sqrt(u*u + v*v)
    wZero = (rho == 0.0)
    try:
        if wZero:
            return 0.0, 0.0, -1.0
    except ValueError:
        pass
    theta = 2*np.arctan(rho/2)
    stheta = np.sin(theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha = u/rho*stheta
        beta = v/rho*stheta
    gamma = -np.cos(theta)
    if np.any(wZero):
        alpha[wZero] = 0.0
        beta[wZero] = 0.0
        gamma[wZero] = -1.0
    return alpha, beta, gamma


def dirCosToStereographic(alpha, beta, gamma):
    """Convert direction cosines to stereographic tangent plane projection.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Returns
    -------
    u, v : float
        Stereographic tangent plane coordinates in radians.

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    wZero = (gamma == -1)
    try:
        if wZero:
            return 0.0, 0.0
    except ValueError:
        pass
    theta = np.arccos(-gamma)
    rho = 2*np.tan(theta/2)
    stheta = np.sin(theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = alpha*rho/stheta
        v = beta*rho/stheta
    if np.any(wZero):
        u[wZero] = 0.0
        v[wZero] = 0.0
    return u, v


def orthographicToDirCos(u, v):
    """Convert orthographic tangent plane projection u,v to direction cosines.

    Parameters
    ----------
    u, v : float
        Orthographic tangent plane coordinates in radians.

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    rho = np.sqrt(u*u + v*v)
    theta = np.arcsin(rho)
    gamma = np.cos(theta)
    alpha = u
    beta = v
    return alpha, beta, -gamma


def dirCosToOrthographic(alpha, beta, gamma):
    """Convert direction cosines to orthographic tangent plane projection.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Returns
    -------
    u, v : float
        Orthographic tangent plane coordinates in radians.

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    u = alpha
    v = beta
    return u, v


def lambertToDirCos(u, v):
    """Convert Lambert azimuthal equal-area tangent plane projection u,v to
    direction cosines.

    Parameters
    ----------
    u, v : float
        Lambert tangent plane coordinates in radians.

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    rhosqr = u*u + v*v
    wZero = rhosqr == 0.0
    try:
        if wZero:
            return 0.0, 0.0, -1.0
    except ValueError:
        pass
    rho = np.sqrt(rhosqr)
    gamma = (2-rhosqr)/2
    r = np.sqrt(1-gamma*gamma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha = u * r/rho
        beta = v * r/rho
    if np.any(wZero):
        alpha[wZero] = 0.0
        beta[wZero] = 0.0
        gamma[wZero] = 1.0
    return alpha, beta, -gamma


def dirCosToLambert(alpha, beta, gamma):
    """Convert direction cosines to Lambert azimuthal equal-area tangent plane
    projection.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Returns
    -------
    u, v : float
        Lambert tangent plane coordinates in radians.

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    wZero = (gamma == -1)
    try:
        if wZero:
            return 0.0, 0.0
    except ValueError:
        pass
    rho = np.sqrt(2+2*gamma)
    norm = np.sqrt(1-gamma*gamma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = alpha*rho/norm
        v = beta*rho/norm
    if np.any(wZero):
        u[wZero] = 0.0
        v[wZero] = 0.0
    return u, v


def fieldToDirCos(u, v, projection='postel'):
    """Convert field angle to direction cosines using specified projection.

    Parameters
    ----------
    u, v : float
        Tangent plane coordinates in radians.
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert field angle to direction cosines.

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    if projection == 'postel':
        return postelToDirCos(u, v)
    elif projection == 'zemax':
        return zemaxToDirCos(u, v)
    elif projection == 'gnomonic':
        return gnomonicToDirCos(u, v)
    elif projection == 'stereographic':
        return stereographicToDirCos(u, v)
    elif projection == 'lambert':
        return lambertToDirCos(u, v)
    elif projection == 'orthographic':
        return orthographicToDirCos(u, v)
    else:
        raise ValueError("Bad projection: {}".format(projection))


def dirCosToField(alpha, beta, gamma, projection='postel'):
    """Convert direction cosines to field angle using specified projection.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)
    projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}
        Projection used to convert direction cosines to field angle.

    Returns
    -------
    u, v : float
        Tangent plane coordinates in radians.

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0), which corresponds to
    (alpha, beta, gamma) = (0, 0, -1) (a ray coming directly from above).  The
    orientation is such that vx (vy) is positive when u (v) is positive.
    """
    if projection == 'postel':
        return dirCosToPostel(alpha, beta, gamma)
    elif projection == 'zemax':
        return dirCosToZemax(alpha, beta, gamma)
    elif projection == 'gnomonic':
        return dirCosToGnomonic(alpha, beta, gamma)
    elif projection == 'stereographic':
        return dirCosToStereographic(alpha, beta, gamma)
    elif projection == 'lambert':
        return dirCosToLambert(alpha, beta, gamma)
    elif projection == 'orthographic':
        return dirCosToOrthographic(alpha, beta, gamma)
    else:
        raise ValueError("Bad projection: {}".format(projection))


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
