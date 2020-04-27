import numpy as np
from . import _batoid
from .utils import lazy_property


class Surface2:
    def intersect(self, r, coordSys=None):
        if coordSys is not None:
            coordSys = coordSys._coordSys
        self._surface.intersectInPlace(r._rv, coordSys)

    def reflect(self, r, coordSys=None):
        if coordSys is not None:
            coordSys = coordSys._coordSys
        self._surface.reflectInPlace(r._rv, coordSys)

    def refract(self, r, m1, m2, coordSys=None):
        if coordSys is not None:
            coordSys = coordSys._coordSys
        self._surface.refractInPlace(r._rv, m1._medium, m2._medium, coordSys)

    def sag(self, x, y):
        x = np.ascontiguousarray(x, dtype=float)
        y = np.ascontiguousarray(y, dtype=float)
        out = np.empty_like(x)
        self._surface.sag(
            x.ctypes.data,
            y.ctypes.data,
            len(x),
            out.ctypes.data
        )
        return out

    def normal(self, x, y):
        x = np.ascontiguousarray(x, dtype=float)
        y = np.ascontiguousarray(y, dtype=float)
        out = np.empty((3, len(x)), dtype=float).T
        self._surface.normal(
            x.ctypes.data,
            y.ctypes.data,
            len(x),
            out.ctypes.data
        )
        return out

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
        self._coefs = np.ascontiguousarray(coefs)

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
        return _batoid.CPPAsphere2(
            self._R, self._conic, self._coefs.ctypes.data, len(self._coefs)
        )


class Bilinear2(Surface2):
    def __init__(self, xs, ys, zs):
        self._xs = xs
        self._ys = ys
        self._x0 = xs[0]
        self._y0 = ys[0]
        self._dx = (xs[-1]-xs[0])/len(xs)
        self._dy = (ys[-1]-ys[0])/len(ys)
        self._zs = np.ascontiguousarray(zs)

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def zs(self):
        return self._zs

    @lazy_property
    def _surface(self):
        return _batoid.CPPBilinear2(
            self._x0, self._y0, self._dx, self._dy,
            self._zs.ctypes.data,
            self._zs.shape[0]
        )


class Bicubic2(Surface2):
    def __init__(
        self, xs, ys, zs,
        dzdxs=None, dzdys=None, d2zdxdys=None
    ):
        self._xs = xs
        self._ys = ys
        self._x0 = xs[0]
        self._y0 = ys[0]
        self._dx = (xs[-1]-xs[0])/(len(xs)-1)
        self._dy = (ys[-1]-ys[0])/(len(ys)-1)
        self._zs = np.ascontiguousarray(zs)

        if dzdxs is None:
            dx = self._dx
            dy = self._dy
            dzdys = np.empty_like(self._zs)
            dzdys[1:-1, :] = (self._zs[2:, :] - self._zs[:-2, :])/(2*dy)
            dzdys[0, :] = (self._zs[1, :] - self._zs[0, :])/dy
            dzdys[-1, :] = (self._zs[-1, :] - self._zs[-2, :])/dy

            dzdxs = np.empty_like(self._zs)
            dzdxs[:, 1:-1] = (self._zs[:, 2:] - self._zs[:, :-2])/(2*dx)
            dzdxs[:, 0] = (self._zs[:, 1] - self._zs[:, 0])/dx
            dzdxs[:, -1] = (self._zs[:, -1] - self._zs[:, -2])/dx

            d2zdxdys = np.empty_like(self._zs)
            d2zdxdys[:, 1:-1] = (dzdys[:, 2:] - dzdys[:, :-2])/(2*dx)
            d2zdxdys[:, 0] = (dzdys[:, 1] - dzdys[:, 0])/dx
            d2zdxdys[:, -1] = (dzdys[:, -1] - dzdys[:, -2])/dx

        self._dzdxs = np.ascontiguousarray(dzdxs)
        self._dzdys = np.ascontiguousarray(dzdys)
        self._d2zdxdys = np.ascontiguousarray(d2zdxdys)

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def xs(self):
        return self._xs

    @property
    def ys(self):
        return self._ys

    @property
    def zs(self):
        return self._zs

    @property
    def dzdxs(self):
        return self._dzdxs

    @property
    def dzdys(self):
        return self._dzdys

    @property
    def d2zdxdys(self):
        return self._d2zdxdys

    @lazy_property
    def _surface(self):
        return _batoid.CPPBicubic2(
            self._x0, self._y0, self._dx, self._dy,
            self._zs.ctypes.data,
            self._dzdxs.ctypes.data,
            self._dzdys.ctypes.data,
            self._d2zdxdys.ctypes.data,
            self._zs.shape[0]
        )


class ExtendedAsphere2(Surface2):
    def __init__(
        self, R, conic, coefs,
        xs, ys, zs,
        dzdxs=None, dzdys=None, d2zdxdys=None
    ):
        # Asphere part
        self._R = R
        self._conic = conic
        self._coefs = np.ascontiguousarray(coefs)

        # Bicubic part
        self._xs = xs
        self._ys = ys
        self._x0 = xs[0]
        self._y0 = ys[0]
        self._dx = (xs[-1]-xs[0])/(len(xs)-1)
        self._dy = (ys[-1]-ys[0])/(len(ys)-1)
        self._zs = np.ascontiguousarray(zs)

        if dzdxs is None:
            dx = self._dx
            dy = self._dy
            dzdys = np.empty_like(self._zs)
            dzdys[1:-1, :] = (self._zs[2:, :] - self._zs[:-2, :])/(2*dy)
            dzdys[0, :] = (self._zs[1, :] - self._zs[0, :])/dy
            dzdys[-1, :] = (self._zs[-1, :] - self._zs[-2, :])/dy

            dzdxs = np.empty_like(self._zs)
            dzdxs[:, 1:-1] = (self._zs[:, 2:] - self._zs[:, :-2])/(2*dx)
            dzdxs[:, 0] = (self._zs[:, 1] - self._zs[:, 0])/dx
            dzdxs[:, -1] = (self._zs[:, -1] - self._zs[:, -2])/dx

            d2zdxdys = np.empty_like(self._zs)
            d2zdxdys[:, 1:-1] = (dzdys[:, 2:] - dzdys[:, :-2])/(2*dx)
            d2zdxdys[:, 0] = (dzdys[:, 1] - dzdys[:, 0])/dx
            d2zdxdys[:, -1] = (dzdys[:, -1] - dzdys[:, -2])/dx

        self._dzdxs = np.ascontiguousarray(dzdxs)
        self._dzdys = np.ascontiguousarray(dzdys)
        self._d2zdxdys = np.ascontiguousarray(d2zdxdys)

    @lazy_property
    def _surface(self):
        return _batoid.CPPExtendedAsphere2(
            self._R, self._conic, self._coefs.ctypes.data, len(self._coefs),
            self._x0, self._y0, self._dx, self._dy,
            self._zs.ctypes.data,
            self._dzdxs.ctypes.data,
            self._dzdys.ctypes.data,
            self._d2zdxdys.ctypes.data,
            self._zs.shape[0]
        )
