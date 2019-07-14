from . import _batoid
from collections.abc import Sequence

class Ray:
    """
    """
    def __init__(self, r=None, v=None, t=0.0, wavelength=0.0, flux=1.0, vignetted=False, failed=False):
        if r is None and v is None:
            self._r = _batoid.Ray(failed=failed)
        elif isinstance(r, Ray):
            self._r = _batoid.Ray(r._r)
        else:
            self._r = _batoid.Ray(r, v, t, wavelength, flux, vignetted)

    @classmethod
    def createFailed(cls):
        ret = self.__class__.__new__(self.__class__)
        ret._r = _batoid.Ray(False)
        return ret

    @classmethod
    def _fromRay(cls, _r):
        ret = cls.__new__(cls)
        ret._r = _r
        return ret

    def copy(self):
        return Ray._fromRay(self._r)

    def __repr__(self):
        return repr(self._r)

    def amplitude(self, r, t):
        return self._r.amplitude(r, t)

    def positionAtTime(self, t):
        return self._r.positionAtTime(t)

    def propagatedToTime(self, t):
        return self._r.propagatedToTime(t)

    def propagateInPlace(self, t):
        self._r.propagateInPlace(t)

    def phase(self, r, t):
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
