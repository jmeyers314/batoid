from . import _batoid
import numpy as np
from abc import ABC, abstractmethod


class Surface(ABC):
    def sag(self, x, y):
        return self._surface.sag(x, y)

    def normal(self, x, y):
        return self._surface.normal(x, y)

    def intersect(self, r):
        return self._surface.intersect(r)

    def intersectInPlace(self, r):
        return self._surface.intersectInPlace(r)

    def reflect(self, r):
        return self._surface.reflect(r)

    def reflectInPlace(self, r):
        self._surface.reflectInPlace(r)

    def refract(self, r, inMedium, outMedium):
        return self._surface.refract(r, inMedium, outMedium)

    def refractInPlace(self, r, inMedium, outMedium):
        self._surface.refractInPlace(r, inMedium, outMedium)

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __setstate__(self, state):
        pass

    @abstractmethod
    def __getstate__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __eq__(self, rhs):
        pass

    def __ne__(self, rhs):
        return not (self == rhs)


class Plane(Surface):
    def __init__(self):
        self._surface = _batoid.Plane()

    def __hash__(self):
        return hash("batoid.Plane")

    def __setstate__(self, state):
        self._surface = _batoid.Plane()

    def __getstate__(self):
        pass

    def __eq__(self, rhs):
        return isinstance(rhs, Plane)

    def __repr__(self):
        return "Plane()"


class Paraboloid(Surface):
    def __init__(self, R):
        self._surface = _batoid.Paraboloid(R)

    @property
    def R(self):
        return self._surface.R

    def __hash__(self):
        return hash(("batoid.Paraboloid", self.R))

    def __setstate__(self, R):
        self._surface = _batoid.Paraboloid(R)

    def __getstate__(self):
        return self.R

    def __eq__(self, rhs):
        if not isinstance(rhs, Paraboloid): return False
        return self.R == rhs.R

    def __repr__(self):
        return "Paraboloid({})".format(self.R)


class Sphere(Surface):
    def __init__(self, R):
        self._surface = _batoid.Sphere(R)

    @property
    def R(self):
        return self._surface.R

    def __hash__(self):
        return hash(("batoid.Sphere", self.R))

    def __setstate__(self, R):
        self._surface = _batoid.Sphere(R)

    def __getstate__(self):
        return self.R

    def __eq__(self, rhs):
        if not isinstance(rhs, Sphere): return False
        return self.R == rhs.R

    def __repr__(self):
        return "Sphere({})".format(self.R)


class Quadric(Surface):
    def __init__(self, R, conic):
        self._surface = _batoid.Quadric(R, conic)

    @property
    def R(self):
        return self._surface.R

    @property
    def conic(self):
        return self._surface.conic

    def __hash__(self):
        return hash(("batoid.Quadric", self.R, self.conic))

    def __setstate__(self, args):
        self._surface = _batoid.Quadric(*args)

    def __getstate__(self):
        return (self.R, self.conic)

    def __eq__(self, rhs):
        if not isinstance(rhs, Quadric): return False
        return (self.R == rhs.R and
                self.conic == rhs.conic)

    def __repr__(self):
        return "Quadric({}, {})".format(self.R, self.conic)


class Asphere(Surface):
    def __init__(self, R, conic, coefs):
        self._surface = _batoid.Asphere(R, conic, coefs)

    @property
    def R(self):
        return self._surface.R

    @property
    def conic(self):
        return self._surface.conic

    @property
    def coefs(self):
        return self._surface.coefs

    def __hash__(self):
        return hash(("batoid.Asphere", self.R, self.conic, tuple(self.coefs)))

    def __setstate__(self, args):
        self._surface = _batoid.Asphere(*args)

    def __getstate__(self):
        return self.R, self.conic, self.coefs

    def __eq__(self, rhs):
        if not isinstance(rhs, Asphere): return False
        return (self.R == rhs.R and
                self.conic == rhs.conic and
                self.coefs == rhs.coefs)

    def __repr__(self):
        return "Asphere({}, {}, {})".format(self.R, self.conic, self.coefs)


class Zernike(Surface):
    def __init__(self, coef, R_outer=1.0, R_inner=0.0):
        import galsim

        self._coef = np.asarray(coef)
        self._R_outer = float(R_outer)
        self._R_inner = float(R_inner)
        self.Z = galsim.zernike.Zernike(coef, R_outer, R_inner)
        pcoef = self.Z._coef_array_xy
        self._surface = _batoid.PolynomialSurface(pcoef)

    @property
    def coef(self):
        return self._coef

    @property
    def R_outer(self):
        return self._R_outer

    @property
    def R_inner(self):
        return self._R_inner

    @property
    def gradX(self):
        return Zernike(self.Z.gradX.coef, self.R_outer, self.R_inner)

    @property
    def gradY(self):
        return Zernike(self.Z.gradY.coef, self.R_outer, self.R_inner)

    def __hash__(self):
        return hash(("batoid.Zernike", tuple(self.coef), self.R_outer, self.R_inner))

    def __setstate__(self, args):
        self.__init__(*args)

    def __getstate__(self):
        return self.coef, self.R_outer, self.R_inner

    def __eq__(self, rhs):
        if not isinstance(rhs, Zernike): return False
        return (np.array_equal(self.coef, rhs.coef) and
                self.R_outer == rhs.R_outer and
                self.R_inner == rhs.R_inner)

    def __repr__(self):
        return "Zernike({!r}, {!r}, {!r})".format(self.coef, self.R_outer, self.R_inner)


class Sum(Surface):
    def __init__(self, surfaces):
        self._surfaces = sorted(surfaces, key=repr)
        self._surface = _batoid.Sum([s._surface for s in surfaces])

    @property
    def surfaces(self):
        return self._surfaces

    def __hash__(self):
        return hash(("batoid.Sum", tuple(self.surfaces)))

    def __setstate__(self, args):
        self._surfaces = args
        self._surface = _batoid.Sum([s._surface for s in args])

    def __getstate__(self):
        return self.surfaces

    def __eq__(self, rhs):
        if not isinstance(rhs, Sum): return False
        return self.surfaces == rhs.surfaces

    def __repr__(self):
        return "Sum({})".format(self.surfaces)
