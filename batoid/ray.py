from collections.abc import Sequence

import numpy as np

from . import _batoid
from .constants import vacuum, globalCoordSys
from .coordsys import CoordSys
from .coordtransform import CoordTransform
from .utils import fieldToDirCos

class Ray:
    """A geometric ray to trace through an optical system.  May also be thought
    of as a monochromatic propagating plane wave.

    Parameters
    ----------
    r : ndarray of float, shape (3,)
        Position of ray in meters.
    v : ndarray of float, shape (3,)
        Velocity vector in units of the speed of light in vacuum.  Note this
        may have magnitude < 1 if the Ray is inside a refractive medium.
    t : float
        Reference time (divided by the speed of light in vacuum) in units of
        meters.  Equivalent to the optical path length.
    wavelength : float
        Vacuum wavelength in meters.
    flux : float
        Flux in arbitrary units.
    vignetted : bool
        Whether Ray has been vignetted or not.
    failed : bool
        Whether Ray is in failed state or not, which may happen if an
        intersection with a surface is requested but cannot be found.
    coordSys : CoordSys
        Coordinate system in which this ray is expressed.  Default: the global
        coordinate system.
    """
    def __init__(self, r=None, v=None, t=0.0, wavelength=0.0, flux=1.0,
                 vignetted=False, failed=False, coordSys=globalCoordSys):
        if failed:
            self._r = _batoid.CPPRayVector([_batoid.CPPRay(failed=True)], coordSys._coordSys)
        elif r is None and v is None:
            self._r = _batoid.CPPRayVector([_batoid.CPPRay()], coordSys._coordSys)
        else:
            self._r = _batoid.CPPRayVector([_batoid.CPPRay(
                r, v, t, wavelength, flux, vignetted
            )], coordSys._coordSys)
        self.coordSys = coordSys

    @classmethod
    def _fromCPPRay(cls, _r, coordSys=globalCoordSys):
        """Turn a c++ Ray into a python Ray."""
        ret = cls.__new__(cls)
        ret._r = _batoid.CPPRayVector([_r], coordSys._coordSys)
        ret.coordSys = coordSys
        return ret

    @classmethod
    def fromStop(
        cls, x, y,
        optic=None, backDist=None, medium=None, stopSurface=None,
        wavelength=None,
        source=None, dirCos=None,
        theta_x=None, theta_y=None, projection='postel',
        flux=1, coordSys=globalCoordSys
    ):
        """Create a Ray that intersects the "stop" surface at a given point.

        The algorithm used here starts by placing the ray on the "stop"
        surface, and then backing it up such that it is in front of any
        surfaces of the optic it's intended to trace.

        The stop surface of most large telescopes is the plane perpendicular to
        the optic axis and flush with the rim of the primary mirror.  This
        plane is usually also the entrance pupil since there are no earlier
        refractive or reflective surfaces.  However, since this plane is a bit
        difficult to locate automatically, the default stop surface in batoid
        is the global x-y plane.

        If a telescope has an stopSurface attribute in its yaml file, then this
        is usually a good choice to use in this function.  Using a curved
        surface for the stop surface is allowed, but is usually a bad idea as
        this may lead to a non-uniformly illuminated pupil and is inconsistent
        with, say, an incoming uniform spherical wave or uniform plane wave.

        Parameters
        ----------
        x, y : float
            X/Y coordinates on the stop surface where the ray would intersect
            if not refracted or reflected first.
        optic : `batoid.Optic`, optional
            If present, then try to extract values for ``backDist``,
            ``medium``, and ``stopSurface`` from the Optic.  Note that values
            explicitly passed here as keyword arguments override those
            extracted from ``optic``.
        backDist : float, optional
            Map ray backwards from the stop surface to the plane that is
            perpendicular to the ray and ``backDist`` meters from the point
            (0, 0, z(0,0)) on the stop surface.  This should generally be set
            large enough that any obscurations or phantom surfaces occuring
            before the stop surface are now "in front" of the ray.  If this
            keyword is set to ``None`` and the ``optic`` keyword is set, then
            infer a value from ``optic.backDist``.  If both this keyword and
            ``optic`` are ``None``, then use a default of 40 meters, which
            should be sufficiently large for foreseeable telescopes.
        medium : `batoid.Medium`, optional
            Initial medium of ray.  If this keyword is set to ``None`` and
            the ``optic`` keyword is set, then infer a value from
            ``optic.inMedium``.  If both this keyword and ``optic`` are
            ``None``, then use a default of vacuum.
        stopSurface : batoid.Interface, optional
            Surface defining the system stop.  If this keyword is set to
            ``None`` and the ``optic`` keyword is set, then infer a value from
            ``optic.stopSurface``.  If both this keyword and ``optic`` are
            ``None``, then use a default ``Interface(Plane())``, which is the
            global x-y plane.
        wavelength : float
            Vacuum wavelength of ray in meters.
        source : None or ndarray of float, shape (3,), optional
            Where the ray originates.  If None, then the ray originates an
            infinite distance away, in which case the ``dirCos`` kwarg must also
            be specified to set the direction of ray propagation.  If an
            ndarray, then the ray originates from this point in global
            coordinates and the ``dirCos`` kwarg is ignored.
        dirCos : ndarray of float, shape (3,), optional
            If source is None, then indicates the direction of ray propagation.
            If source is not None, then this is ignored.
        theta_x, theta_y : float, optional
            Field angle in radians.  If source is None, then this indicates the
            initial direction of propagation of the ray.  If source is not
            None, then this is ignored.  Uses `utils.fieldToDirCos` to convert
            to direction cosines.  Also see ``dirCos`` as an alternative to
            this keyword.
        projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}, optional
            Projection used to convert field angle to direction cosines.
        flux : float, optional
            Flux of ray.  Default is 1.0.
        coordSys : CoordSys
            Coordinate system in which ray is expressed.  Default: the global
            coordinate system.
        """
        from .optic import Interface
        from .surface import Plane

        if optic is not None:
            if backDist is None:
                backDist = optic.backDist
            if medium is None:
                medium = optic.inMedium
            if stopSurface is None:
                stopSurface = optic.stopSurface

        if backDist is None:
            backDist = 40.0
        if stopSurface is None:
            stopSurface = Interface(Plane())
        if medium is None:
            medium = vacuum

        if dirCos is None and source is None:
            dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)

        if wavelength is None:
            raise ValueError("Missing wavelength keyword")

        z = stopSurface.surface.sag(x, y)
        transform = CoordTransform(stopSurface.coordSys, globalCoordSys)
        x, y, z = transform.applyForward(x, y, z)

        t = 0.0
        n = medium.getN(wavelength)
        if source is None:
            v = np.array(dirCos, dtype=float)
            v /= n*np.sqrt(np.dot(v, v))
            ray = Ray((x, y, z), -v, t, wavelength, flux)
            zhat = -n*v
            xhat = np.cross(np.array([1.0, 0.0, 0.0]), zhat)
            xhat /= np.sqrt(np.dot(xhat, xhat))
            yhat = np.cross(xhat, zhat)
            origin = zhat*backDist
            coordSys = CoordSys(origin, np.stack([xhat, yhat, zhat]).T)
            transform = CoordTransform(globalCoordSys, coordSys)
            transform.applyForwardInPlace(ray)
            plane = Plane()
            plane.intersectInPlace(ray)
            transform.applyReverseInPlace(ray)
            return Ray(
                (ray.x, ray.y, ray.z),
                v, t, wavelength, flux,
                coordSys=globalCoordSys
            )
        else:
            vx = x - source[0]
            vy = y - source[1]
            vz = z - source[2]
            v = np.array([vx, vy, vz])
            v /= n*np.sqrt(np.dot(v, v))
            return Ray((x, y, z), v, t, w, flux, coordSys=globalCoordSys)

    def copy(self):
        """Return a copy of this Ray."""
        return Ray._fromCPPRay(self._r[0], self.coordSys)

    def __repr__(self):
        return repr(self._r[0])

    def amplitude(self, r, t):
        """Calculate (scalar) complex electric-field amplitude at given
        position and time.

        Parameters
        ----------
        r : ndarray of float, shape (3,)
            Position in meters.
        t : float
            Time (over vacuum speed of light; in meters).

        Returns
        -------
        complex
        """
        return self._r.amplitude(r, t)[0]

    def positionAtTime(self, t):
        """Calculate the position of the Ray at a given time.

        Parameters
        ----------
        t : float
            Time (over vacuum speed of light; in meters).

        Returns
        -------
        ndarray of float, shape (3,)
            Position in meters.
        """
        return self._r.positionAtTime(t)[0]

    def propagatedToTime(self, t):
        """Return a Ray propagated to given time.

        Parameters
        ----------
        t : float
            Time (over vacuum speed of light; in meters).

        Returns
        -------
        Ray
        """
        return self._r.propagatedToTime(t)[0]

    def propagateInPlace(self, t):
        """Propagate Ray to given time.

        Parameters
        ----------
        t : float
            Time (over vacuum speed of light; in meters).
        """
        self._r.propagateInPlace(t)

    def phase(self, r, t):
        """Calculate plane wave phase at given position and time.

        Parameters
        ----------
        r : ndarray of float, shape (3,)
            Position in meters at which to compute phase
        t : float
            Time (over vacuum speed of light; in meters).

        Returns
        -------
        float
        """
        return self._r.phase(r, t)[0]

    def toCoordSys(self, coordSys):
        """Transform ray into new coordinate system.

        Parameters
        ----------
        coordSys: batoid.CoordSys
            Destination coordinate system.

        Returns
        -------
        Ray
        """
        transform = CoordTransform(self.coordSys, coordSys)
        return transform.applyForward(self)

    def toCoordSysInPlace(self, coordSys):
        """Transform ray into new coordinate system in place.

        Parameters
        ----------
        coordSys: batoid.CoordSys
            Destination coordinate system.
        """
        transform = CoordTransform(self.coordSys, coordSys)
        transform.applyForwardInPlace(self)

    @property
    def r(self):
        """ndarray of float, shape (3,): Position of ray in meters."""
        return self._r.r[0]

    @property
    def v(self):
        """ndarray of float, shape (3,): Velocity of ray in units of the speed
        of light in vacuum. Note that this may have magnitude < 1 if the ray is
        inside a refractive medium.
        """
        return self._r.v[0]

    @property
    def t(self):
        """Reference time (divided by the speed of light in vacuum) in units of
        meters, also known as the optical path length.
        """
        return self._r.t[0]

    @property
    def wavelength(self):
        """Vacuum wavelength in meters."""
        return self._r.wavelength[0]

    @property
    def flux(self):
        """Ray flux in arbitrary units."""
        return self._r.flux[0]

    @property
    def vignetted(self):
        """True if ray has been vignetted."""
        return self._r.vignetted[0]

    @property
    def failed(self):
        """True if ray is in a failed state.  This may occur, for example, if
        batoid failed to find the intersection of a ray with a surface.
        """
        return self._r.failed[0]

    @property
    def x(self):
        """The x component of the ray position in meters."""
        return self._r.x[0]

    @property
    def y(self):
        """The y component of the ray position in meters."""
        return self._r.y[0]

    @property
    def z(self):
        """The z component of the ray position in meters."""
        return self._r.z[0]

    @property
    def vx(self):
        """The x component of the ray velocity in units of the vacuum speed of
        light.
        """
        return self._r.vx[0]

    @property
    def vy(self):
        """The y component of the ray velocity in units of the vacuum speed of
        light.
        """
        return self._r.vy[0]

    @property
    def vz(self):
        """The z component of the ray velocity in units of the vacuum speed of
        light.
        """
        return self._r.vz[0]

    @property
    def k(self):
        r"""ndarray of float, shape (3,): Wavevector of plane wave in units of
        radians per meter.  The magnitude of the wavevector is equal to
        :math:`2 \pi n / \lambda`, where :math:`n` is the refractive index and
        :math:`\lambda` is the wavelength.
        """
        return self._r.k[0]

    @property
    def kx(self):
        """The x component of the ray wave vector in radians per meter."""
        return self._r.kx[0]

    @property
    def ky(self):
        """The y component of the ray wave vector in radians per meter."""
        return self._r.ky[0]

    @property
    def kz(self):
        """The z component of the ray wave vector in radians per meter."""
        return self._r.kz[0]

    @property
    def omega(self):
        r"""The temporal angular frequency of the plane wave divided by the
        vacuum speed of light in units of radians per meter.  Equals
        :math:`2 \pi / \lambda`.
        """
        return self._r.omega[0]

    def __eq__(self, rhs):
        if not isinstance(rhs, Ray): return False
        return self._r == rhs._r and self.coordSys == rhs.coordSys

    def __ne__(self, rhs):
        return not (self == rhs)
