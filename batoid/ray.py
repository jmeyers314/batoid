from collections.abc import Sequence

import numpy as np

from . import _batoid
from .constants import vacuum, globalCoordSys
from .coordsys import CoordSys, CoordTransform

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
    """
    def __init__(self, r=None, v=None, t=0.0, wavelength=0.0, flux=1.0,
                 vignetted=False, failed=False):
        if failed:
            self._r = _batoid.Ray(failed=True)
        elif r is None and v is None:
            self._r = _batoid.Ray()
        elif isinstance(r, Ray):
            self._r = _batoid.Ray(r._r)
        else:
            self._r = _batoid.Ray(r, v, t, wavelength, flux, vignetted)

    @classmethod
    def _fromRay(cls, _r):
        """Turn a c++ Ray into a python Ray."""
        ret = cls.__new__(cls)
        ret._r = _r
        return ret

    @classmethod
    def fromStop(cls, x, y,
                 dist, wavelength,
                 source=None, dirCos=None,
                 flux=1, medium=vacuum,
                 stopSurface=None):
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
        dist : float
            Map ray backwards from the stop surface to the plane that is
            perpendicular to the ray velocity and ``dist`` meters from the
            point (0, 0, z(0,0)) on the stop surface.  This should generally be
            set large enough that any obscurations or phantom surfaces occuring
            before the stop surface are "in front" of the ray.
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
        flux : float, optional
            Flux of ray.  Default is 1.0.
        medium : batoid.Medium, optional
            Initial medium of Ray.  Default is vacuum.
        stopSurface : batoid.Interface, optional
            Surface defining the system stop.  Default: ``Interface(Plane())``.
        """
        from .optic import Interface
        from .surface import Plane

        if stopSurface is None:
            stopSurface = Interface(Plane())

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
            origin = zhat*dist
            coordSys = CoordSys(origin, np.stack([xhat, yhat, zhat]).T)
            transform = CoordTransform(globalCoordSys, coordSys)
            transform.applyForwardInPlace(ray)
            plane = Plane()
            plane.intersectInPlace(ray)
            transform.applyReverseInPlace(ray)
            return Ray((ray.x, ray.y, ray.z), v, t, wavelength, flux)
        else:
            vx = x - source[0]
            vy = y - source[1]
            vz = z - source[2]
            v = np.array([vx, vy, vz])
            v /= n*np.sqrt(np.dot(v, v))
            return Ray((x, y, z), v, t, w, flux)

    def __repr__(self):
        return repr(self._r)

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
        return self._r.amplitude(r, t)

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
        return self._r.positionAtTime(t)

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
        return self._r.propagatedToTime(t)

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
        return self._r.phase(r, t)

    @property
    def r(self):
        """ndarray of float, shape (3,): Position of ray in meters."""
        return self._r.r

    @property
    def v(self):
        """ndarray of float, shape (3,): Velocity of ray in units of the speed
        of light in vacuum. Note that this may have magnitude < 1 if the ray is
        inside a refractive medium.
        """
        return self._r.v

    @property
    def t(self):
        """Reference time (divided by the speed of light in vacuum) in units of
        meters, also known as the optical path length.
        """
        return self._r.t

    @property
    def wavelength(self):
        """Vacuum wavelength in meters."""
        return self._r.wavelength

    @property
    def flux(self):
        """Ray flux in arbitrary units."""
        return self._r.flux

    @property
    def vignetted(self):
        """True if ray has been vignetted."""
        return self._r.vignetted

    @property
    def failed(self):
        """True if ray is in a failed state.  This may occur, for example, if
        batoid failed to find the intersection of a ray with a surface.
        """
        return self._r.failed

    @property
    def x(self):
        """The x component of the ray position in meters."""
        return self._r.x

    @property
    def y(self):
        """The y component of the ray position in meters."""
        return self._r.y

    @property
    def z(self):
        """The z component of the ray position in meters."""
        return self._r.z

    @property
    def vx(self):
        """The x component of the ray velocity in units of the vacuum speed of
        light.
        """
        return self._r.vx

    @property
    def vy(self):
        """The y component of the ray velocity in units of the vacuum speed of
        light.
        """
        return self._r.vy

    @property
    def vz(self):
        """The z component of the ray velocity in units of the vacuum speed of
        light.
        """
        return self._r.vz

    @property
    def k(self):
        r"""ndarray of float, shape (3,): Wavevector of plane wave in units of
        radians per meter.  The magnitude of the wavevector is equal to
        :math:`2 \pi n / \lambda`, where :math:`n` is the refractive index and
        :math:`\lambda` is the wavelength.
        """
        return self._r.k

    @property
    def kx(self):
        """The x component of the ray wave vector in radians per meter."""
        return self._r.kx

    @property
    def ky(self):
        """The y component of the ray wave vector in radians per meter."""
        return self._r.ky

    @property
    def kz(self):
        """The z component of the ray wave vector in radians per meter."""
        return self._r.kz

    @property
    def omega(self):
        r"""The temporal angular frequency of the plane wave divided by the
        vacuum speed of light in units of radians per meter.  Equals
        :math:`2 \pi / \lambda`.
        """
        return self._r.omega

    def __eq__(self, rhs):
        if not isinstance(rhs, Ray): return False
        return self._r == rhs._r

    def __ne__(self, rhs):
        return not (self == rhs)
