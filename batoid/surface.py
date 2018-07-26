from . import _batoid
import numpy as np
from abc import ABC, abstractmethod


class Surface(ABC):
    """Abstract base class representing a 2D geometric surface.
    """
    def sag(self, x, y):
        """The function defining the surface; z(x, y).

        Parameters
        ----------
        x, y : array_like
            Positions at which to evaluate the surface.

        Returns
        -------
        z : array_like
            Surface height.
        """
        return self._surface.sag(x, y)

    def normal(self, x, y):
        """The normal vector to the surface at (x, y, z(x, y)).

        Parameters
        ----------
        x, y : array_like
            Positions at which to evaluate the surface.

        Returns
        -------
        normal : array_like
            Surface normals.
        """
        return self._surface.normal(x, y)

    def intersect(self, r):
        """Calculate intersection of a ray or rays with this surface.  If the intersection is in the
        past, then set the ray.fail flag.  If the ray intersects at an obscured point, then set the
        ray.vignetted flag.

        Parameters
        ----------
        r : Ray or RayVector
            Input ray(s) to intersect

        Returns
        -------
        outRays : Ray or RayVector
            New object corresponding to original ray(s) propagated to the intersection point.
        """
        return self._surface.intersect(r)

    def intersectInPlace(self, r):
        """Calculate intersection of ray or rays with this surface.  Same as `intersect`, but
        operates on the input argument in place.

        Parameters
        ----------
        r : Ray or RayVector
            Ray(s) to manipulate in place.
        """
        return self._surface.intersectInPlace(r)

    def reflect(self, r):
        """Calculate intersection of ray(s) with this surface, and immediately reflect the ray(s) at
        the point(s) of intersection.

        Parameters
        ----------
        r : Ray or RayVector
            Ray(s) to reflect.

        Returns
        -------
        outRays : Ray or RayVector
            New object corresponding to original ray(s) propagated and reflected.
        """
        return self._surface.reflect(r)

    def reflectInPlace(self, r):
        """Calculate intersection of ray(s) with this surface, and immediately reflect the ray(s) at
        the point(s) of intersection.  Same as `reflect`, but manipulates the input ray(s) in place.

        Parameters
        ----------
        r : Ray or RayVector
            Ray(s) to reflect in place.
        """
        self._surface.reflectInPlace(r)

    def refract(self, r, inMedium, outMedium):
        """Calculate intersection of ray(s) with this surface, and immediately refract the ray(s)
        through the surface at the point(s) of intersection.

        Parameters
        ----------
        r : Ray or RayVector
            Ray(s) to refract.
        inMedium : Medium
            Refractive medium on the incoming side of the surface.
        outMedium : Medium
            Refractive medium on the outgoing side of the surface.

        Returns
        -------
        outRays : Ray or RayVector
            New object corresponding to original ray(s) propagated and refracted.
        """
        return self._surface.refract(r, inMedium, outMedium)

    def refractInPlace(self, r, inMedium, outMedium):
        """Calculate intersection of ray(s) with this surface, and immediately refract the ray(s)
        through the surface at the point(s) of intersection.  Same as `refract`, but manipulates the
        input ray(s) in place.

        Parameters
        ----------
        r : Ray or RayVector
            Ray(s) to refract in place.
        inMedium : Medium
            Refractive medium on the incoming side of the surface.
        outMedium : Medium
            Refractive medium on the outgoing side of the surface.
        """
        self._surface.refractInPlace(r, inMedium, outMedium)

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

    @abstractmethod
    def __setstate__(self, state):
        raise NotImplementedError

    @abstractmethod
    def __getstate__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, rhs):
        raise NotImplementedError

    def __ne__(self, rhs):
        return not (self == rhs)


class Plane(Surface):
    """Planar surface.  Represents the equation

    z(x, y) = 0.

    """
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
    """Surface of revolution with parabolic cross-section, and where the axis of revolution is along
    the axis of the parabola.  Represents the equation

    z(x, y) = z(r) = r^2 / (2 R)

    where r = sqrt(x^2 + y^2) and `R` is the radius of curvature at the paraboloid vertex.

    Parameters
    ----------
    R : float
        Radius of curvature at paraboloid vertex.

    """
    def __init__(self, R):
        self._surface = _batoid.Paraboloid(R)

    @property
    def R(self):
        """Radius of curvature at paraboloid vertex.
        """
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
    """Spherical surface.  Represents the equation

    z(x, y) = z(r) = R (1 - sqrt(1-r^2/R^2))

    where r = sqrt(x^2 + y^2) and `R` is the radius the sphere.  Note that the center of the sphere
    is a distance `R` above the vertex.

    Parameters
    ----------
    R : float
        Sphere radius.

    """
    def __init__(self, R):
        self._surface = _batoid.Sphere(R)

    @property
    def R(self):
        """Sphere radius.
        """
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
    """Surface of revolution where the cross section is a conic section.  Represents the equation

    z(x, y) = z(r) = r^2 / (R (1 + sqrt(1 -r^2/R^2 (1 + conic))))

    where r = sqrt(x^2 + y^2), `R` is the radius of curvature at the surface vertex, and `conic` is
    the conic constant.  Different ranges of `conic` produce different categories of surfaces:

        conic > 0        =>  oblate ellipsoid
        conic == 0       =>  sphere
        -1 < conic < 0   =>  prolate ellipsoid
        conic = -1       =>  paraboloid
        conic < -1       =>  hyperboloid

    Parameters
    ----------
    R : float
        Radius of curvature at vertex.
    conic : float
        Conic constant.

    """
    def __init__(self, R, conic):
        self._surface = _batoid.Quadric(R, conic)

    @property
    def R(self):
        """Radius of curvature at quadric vertex.
        """
        return self._surface.R

    @property
    def conic(self):
        """Conic constant.
        """
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
    """Surface of revolution where the cross section is a conic section plus an even polynomial.
    Represents the equation

    z(x, y) = z(r) = r^2 / (R (1 + sqrt(1 -r^2/R^2 (1 + conic)))) + sum_i alpha_i r^(2 i)

    where r = sqrt(x^2 + y^2), `R` is the radius of curvature at the surface vertex, and `conic` is
    the conic constant, and `alpha` are the even polynomial coefficients.  Different ranges of
    `conic` produce different categories of surfaces (where alpha==0):

        conic > 0        =>  oblate ellipsoid
        conic == 0       =>  sphere
        -1 < conic < 0   =>  prolate ellipsoid
        conic = -1       =>  paraboloid
        conic < -1       =>  hyperboloid

    Parameters
    ----------
    R : float
        Radius of curvature at vertex.
    conic : float
        Conic constant.
    alpha : list of float
        Even polynomial coefficients.
    """
    def __init__(self, R, conic, coefs):
        self._surface = _batoid.Asphere(R, conic, coefs)

    @property
    def R(self):
        """Radius of curvature at asphere vertex.
        """
        return self._surface.R

    @property
    def conic(self):
        """Conic constant.
        """
        return self._surface.conic

    @property
    def coefs(self):
        """Even polynomial coefficients.
        """
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
    """Surface defined by Zernike polynomials.  Represents the equation

    z(x, y) = sum_j coef_j Z_j(R_inner/R_outer; x/R_outer, y/R_outer)

    where Z_j(eps, u, v) are the annular Zernike polynomials (Mahajan) with central obscuration
    `eps` indexed by the Noll (1976) convention, `R_outer` is the outer radius of the annulus,
    `R_inner` is the inner radius of the annulus, and `coef_j` are corresponding annular
    coefficients.

    Note that the Noll convention starts at j=1, so the coef_0 = coef[0] value has no effect on the
    surface.

    Parameters
    ----------
    coef : list of float
        Annular Zernike polynomial coefficients.
    R_outer : float
        Outer radius of annulus.
    R_inner : float
        Inner radius of annulus.
    """
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
        """Annular Zernike polynomial coefficients.
        """
        return self._coef

    @property
    def R_outer(self):
        """Outer radius of annulus.
        """
        return self._R_outer

    @property
    def R_inner(self):
        """Outer radius of annulus.
        """
        return self._R_inner

    @property
    def gradX(self):
        """Gradient of Zernike surface in the x direction as a new Zernike object.
        """
        return Zernike(self.Z.gradX.coef, self.R_outer, self.R_inner)

    @property
    def gradY(self):
        """Gradient of Zernike surface in the y direction as a new Zernike object.
        """
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
    """Composite surface combining two or more other Surfaces through addition.  Represents the
    equation

    z(x, y) = sum_i S_i(x, y)

    where S_i is the ith input Surface.

    Parameters
    ----------
    surfaces : list of Surface
        `Surface`s to add together.
    """
    def __init__(self, surfaces):
        self._surfaces = sorted(surfaces, key=repr)
        self._surface = _batoid.Sum([s._surface for s in surfaces])

    @property
    def surfaces(self):
        """List of constituent `Surface`s.
        """
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
