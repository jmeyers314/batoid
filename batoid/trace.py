from . import _batoid
from .coordTransform import CoordTransform


def applyForwardTransform(ct, rv):
    from .global_vars import _batoid_max_threads
    _batoid.applyForwardTransform(
        ct.dr, ct.drot.ravel(), rv._rv, _batoid_max_threads
    )
    rv.coordSys = ct.toSys
    return rv


def applyReverseTransform(ct, rv):
    from .global_vars import _batoid_max_threads
    _batoid.applyReverseTransform(
        ct.dr, ct.drot.ravel(), rv._rv, _batoid_max_threads
    )
    rv.coordSys = ct.fromSys
    return rv


def applyForwardTransformArrays(ct, x, y, z):
    from .global_vars import _batoid_max_threads
    _batoid.applyForwardTransformArrays(
        ct.dr, ct.drot.ravel(),
        x.ctypes.data, y.ctypes.data, z.ctypes.data,
        len(x), _batoid_max_threads
    )


def applyReverseTransformArrays(ct, x, y, z):
    from .global_vars import _batoid_max_threads
    _batoid.applyReverseTransformArrays(
        ct.dr, ct.drot.ravel(),
        x.ctypes.data, y.ctypes.data, z.ctypes.data,
        len(x), _batoid_max_threads
    )


def obscure(obsc, rv):
    from .global_vars import _batoid_max_threads
    _batoid.obscure(obsc._obsc, rv._rv, _batoid_max_threads)
    return rv


def intersect(surface, rv, coordSys=None, coating=None):
    """Calculate intersection of rays with surface.

    Parameters
    ----------
    surface: Surface
        Surface to intersect.
    rv : RayVector
        Rays to intersect.
    coordSys : CoordSys, optional
        Transform rays into this coordinate system before computing
        intersection.
    coating : Coating, optional
        Apply this coating upon surface intersection.

    Returns
    -------
    out : RayVector
        Reference to transformed input ray vector, which has been modified in
        place.
    """
    from .global_vars import _batoid_max_threads, _batoid_niter
    if coordSys is None:
        coordSys = rv.coordSys
    ct = CoordTransform(rv.coordSys, coordSys)
    _coating = coating._coating if coating else None

    _batoid.intersect(
        surface._surface,
        ct.dr, ct.drot.ravel(),
        rv._rv, _coating,
        _batoid_max_threads, _batoid_niter
    )
    rv.coordSys = coordSys
    return rv


def reflect(surface, rv, coordSys=None, coating=None):
    from .global_vars import _batoid_max_threads, _batoid_niter
    if coordSys is None:
        coordSys = rv.coordSys
    ct = CoordTransform(rv.coordSys, coordSys)
    _coating = coating._coating if coating else None

    _batoid.reflect(
        surface._surface,
        ct.dr, ct.drot.ravel(),
        rv._rv, _coating,
        _batoid_max_threads, _batoid_niter
    )
    rv.coordSys = coordSys
    return rv


def refract(surface, rv, m1, m2, coordSys=None, coating=None):
    from .global_vars import _batoid_max_threads, _batoid_niter
    if coordSys is None:
        coordSys = rv.coordSys
    ct = CoordTransform(rv.coordSys, coordSys)
    _coating = coating._coating if coating else None

    _batoid.refract(
        surface._surface,
        ct.dr, ct.drot.ravel(),
        m1._medium, m2._medium,
        rv._rv, _coating,
        _batoid_max_threads, _batoid_niter
    )
    rv.coordSys = coordSys
    return rv


def rSplit(surface, rv, inMedium, outMedium, coating, coordSys=None):
    from .global_vars import _batoid_max_threads, _batoid_niter
    if coordSys is None:
        coordSys = rv.coordSys
    ct = CoordTransform(rv.coordSys, coordSys)

    rvSplit = rv.copy()
    _batoid.rSplit(
        surface._surface,
        ct.dr, ct.drot.ravel(),
        inMedium._medium, outMedium._medium,
        coating._coating,
        rv._rv, rvSplit._rv,
        _batoid_max_threads, _batoid_niter
    )
    rv.coordSys = coordSys
    rvSplit.coordSys = coordSys
    return rv, rvSplit


def refractScreen(surface, rv, screen, coordSys=None):
    from .global_vars import _batoid_max_threads, _batoid_niter
    if coordSys is None:
        coordSys = rv.coordSys
    ct = CoordTransform(rv.coordSys, coordSys)

    _batoid.refractScreen(
        surface._surface,
        ct.dr, ct.drot.ravel(),
        screen._surface,
        rv._rv,
        _batoid_max_threads, _batoid_niter
    )
    rv.coordSys = coordSys
    return rv
