import numpy as np
from . import _batoid
from .utils import lazy_property


class Surface2:
    def intersectInPlace(self, r, coordSys=None):
        if coordSys is not None:
            coordSys = coordSys._coordSys
        self._surface.intersectInPlace(r._rv, coordSys)

    def reflectInPlace(self, r, coordSys=None):
        if coordSys is not None:
            coordSys = coordSys._coordSys
        self._surface.reflectInPlace(r._rv, coordSys)

    def refractInPlace(self, r, m1, m2, coordSys=None):
        if coordSys is not None:
            coordSys = coordSys._coordSys
        self._surface.refractInPlace(r._rv, m1._medium, m2._medium, coordSys)


class Plane2(Surface2):
    def __init__(self, allowReverse=False):
        self._allowReverse = allowReverse

    @property
    def allowReverse(self):
        return self._allowReverse

    @lazy_property
    def _surface(self):
        return _batoid.CPPPlane2(self._allowReverse)


class Sphere2(Surface2):
    def __init__(self, R):
        self._R = R

    @property
    def R(self):
        return self._R

    @lazy_property
    def _surface(self):
        return _batoid.CPPSphere2(self._R)


class Paraboloid2(Surface2):
    def __init__(self, R):
        self._R = R

    @property
    def R(self):
        return self._R

    @lazy_property
    def _surface(self):
        return _batoid.CPPParaboloid2(self._R)


class Quadric2(Surface2):
    def __init__(self, R, conic):
        self._R = R
        self._conic = conic

    @property
    def R(self):
        return self._R

    @property
    def conic(self):
        return self._conic

    @lazy_property
    def _surface(self):
        return _batoid.CPPQuadric2(self._R, self._conic)


class Asphere2(Surface2):
    def __init__(self, R, conic, coefs):
        self._R = R
        self._conic = conic
        self._coefs = coefs

    @property
    def R(self):
        return self._R

    @property
    def conic(self):
        return self._conic

    @property
    def coefs(self):
        return self._coefs

    @lazy_property
    def _surface(self):
        coefs = [0.0]*10
        coefs[:len(self._coefs)] = self._coefs
        return _batoid.CPPAsphere2(self._R, self._conic, coefs)
