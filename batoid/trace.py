from . import _batoid
from .coordSys import CoordSys
from .coordTransform import CoordTransform


def intersect(surface, rv, coordSys=None):
    """Calculate intersection of rays with surface.

    Parameters
    ----------
    surface: Surface
        Surface to intersect.
    rv : RayVector
        Ray(s) to intersect.
    coordSys : CoordSys, optional
        Transform rays into this coordinate system before computing
        intersection.

    Returns
    -------
    out : RayVector
        Reference to transformed input ray vector, which has been modified in
        place.
    """
    if coordSys is None:
        coordSys = rv.coordSys
    ct = CoordTransform(rv.coordSys, coordSys)

    _batoid.intersect(
        surface._surface,
        ct.dr, ct.drot.ravel(),
        rv._rv,
    )
    rv.coordSys = coordSys
    return rv


def applyForwardTransform(ct, rv):
    _batoid.applyForwardTransform(ct.dr, ct.drot.ravel(), rv._rv)
    rv.coordSys = ct.toSys
    return rv

def applyReverseTransform(ct, rv):
    _batoid.applyReverseTransform(ct.dr, ct.drot.ravel(), rv._rv)
    rv.coordSys = ct.fromSys
    return rv

def obscure(obsc, rv):
    _batoid.obscure(obsc._obsc, rv._rv);
    return rv
