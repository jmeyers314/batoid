from collections.abc import Sequence

import numpy as np

from . import _batoid
from .constants import vacuum, globalCoordSys
from .coordsys import CoordSys, CoordTransform

class Ray:
    r"""A geometric ray to trace through an optical system.  May also be
    thought of as a monochromatic propagating plane wave.

    Parameters
    ----------
    r : (3,) array of float
        Position of ray in meters.
    v : (3,) array of float
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

    Attributes
    ----------
    r
    v
    t
    wavelength
    flux
    vignetted
    failed
    x, y, z : float
        Alternate access to ray position.
    vx, vy, vz : float
        Alternate access to ray velocity.
    k : (3,) array of float
        Wavevector of plane wave in units of radians per meter.  The magnitude
        of the wavevector is equal to :math:`2 \pi n / \lambda`, where
        :math:`n` is the refractive index and :math:`\lambda` is the
        wavelength.
    kx, ky, kz : float
        Alternative access to wavevector.
    omega : float
        Temporal frequency of the plane wave over the speed of light.  Units
        are inverse meters.  Equals :math:`2 \pi / \lambda`.

    Methods
    -------
    positionAtTime(t)
        Calculate position of ray at time `t`.
    propagatedToTime(t)
        Return new ray propagated to time `t`.
    propagatedInPlace(t)
        Propagated current ray to time `t`.
    phase(r, t)
        Return plane wave phase at position and time.
    amplitude(r, t)
        Return plane wave complex E-field amplitude at position and time.
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
        ret = cls.__new__(cls)
        ret._r = _r
        return ret

    @classmethod
    def fromPupil(cls, x, y,
                  dist, wavelength,
                  source=None, dirCos=None,
                  flux=1, medium=vacuum,
                  interface=None):
        from .optic import Interface
        from .surface import Plane

        if interface is None:
            interface = Interface(Plane())

        z = interface.surface.sag(x, y)
        transform = CoordTransform(interface.coordSys, globalCoordSys)
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
        r : (3,), array of float
            Position in meters at which to compute phase.
        t : float
            Time (over the speed of light; in meters) at which to compute
            phase.

        Returns
        -------
        amplitude : complex
        """
        return self._r.amplitude(r, t)

    def positionAtTime(self, t):
        """Calculate the position of the Ray at a given time.

        Parameters
        ----------
        t : float
            Time (over the speed of light; in meters) at which to compute
            position.

        Returns
        -------
        position : (3,), array of float
           Position in meters.
        """
        return self._r.positionAtTime(t)

    def propagatedToTime(self, t):
        """Return a Ray propagated to given time.

        Parameters
        ----------
        t : float
            Time (over the speed of light; in meters) to which to propagate
            ray.

        Returns
        -------
        propagatedRay : Ray
        """
        return self._r.propagatedToTime(t)

    def propagateInPlace(self, t):
        """Propagate Ray to given time.

        Parameters
        ----------
        t : float
            Time (over the speed of light; in meters) to which to propagate
            ray.
        """
        self._r.propagateInPlace(t)

    def phase(self, r, t):
        """Calculate plane wave phase at given position and time.

        Parameters
        ----------
        r : (3,), array of float
            Position at which to compute phase
        t : float
            Time (over the speed of light; in meters) at which to compute
            phase.

        Returns
        -------
        phase : float
        """
        return self._r.phase(r, t)

    @property
    def r(self):
        return self._r.r

    @property
    def v(self):
        return self._r.v

    @property
    def t(self):
        return self._r.t

    @property
    def wavelength(self):
        return self._r.wavelength

    @property
    def flux(self):
        return self._r.flux

    @property
    def vignetted(self):
        return self._r.vignetted

    @property
    def failed(self):
        return self._r.failed

    @property
    def x(self):
        return self._r.x

    @property
    def y(self):
        return self._r.y

    @property
    def z(self):
        return self._r.z

    @property
    def vx(self):
        return self._r.vx

    @property
    def vy(self):
        return self._r.vy

    @property
    def vz(self):
        return self._r.vz

    @property
    def k(self):
        return self._r.k

    @property
    def kx(self):
        return self._r.kx

    @property
    def ky(self):
        return self._r.ky

    @property
    def kz(self):
        return self._r.kz

    @property
    def omega(self):
        return self._r.omega

    def __eq__(self, rhs):
        if not isinstance(rhs, Ray): return False
        return self._r == rhs._r

    def __ne__(self, rhs):
        return not (self == rhs)
