from . import _batoid
from .coordTransform import CoordTransform


def intersect(surface, rv, ct=None):
    """Calculate intersection of rays with surface.

    Parameters
    ----------
    surface: Surface
        Surface to intersect.
    rv : RayVector
        Ray(s) to intersect.
    ct : CoordTransform, optional
        Coordinate transform from rv sys to surface coordsys.

    Returns
    -------
    out : RayVector
        Reference to transformed input ray vector, which has been modified in
        place.
    """
    if ct is None:
        ct = CoordTransform(rv.coordSys, rv.coordSys)

    _batoid.intersect(
        surface._surface,
        rv._rv,
        ct._coordTransform
    )
    rv.coordSys = ct.toSys
    return rv
