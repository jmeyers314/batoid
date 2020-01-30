from . import _batoid
from .utils import lazy_property


class Surface2:
    def intersectInPlace(self, r):
        self._surface.intersectInPlace(r._rv)

    def reflectInPlace(self, r):
        self._surface.reflectInPlace(r._rv)

    def refractInPlace(self, r, m1, m2):
        self._surface.refractInPlace(r._rv, m1._medium, m2._medium)


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
