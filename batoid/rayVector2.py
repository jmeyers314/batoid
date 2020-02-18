import numpy as np
from .utils import lazy_property

from . import _batoid
from .constants import globalCoordSys
from .coordtransform2 import CoordTransform2

class RayVector2:
    @classmethod
    def fromArrays(cls, x, y, z, vx, vy, vz, t, w, flux, vignetted, failed,
                   coordSys=globalCoordSys):
        """Create RayVector2 from 1d parameter arrays.  Always makes a copy
        of input arrays.

        Parameters
        ----------
        x, y, z : ndarray of float, shape (n,)
            Positions of rays in meters.
        vx, vy, vz : ndarray of float, shape (n,)
            Velocities of rays in units of the speed of light in vacuum.
        t : ndarray of float, shape (n,)
            Reference times (divided by the speed of light in vacuum) in units
            of meters.
        wavelength : ndarray of float, shape (n,)
            Vacuum wavelengths in meters.
        flux : ndarray of float, shape (n,)
            Fluxes in arbitrary units.
        vignetted : ndarray of bool, shape (n,)
            True where rays have been vignetted.
        coordSys : CoordSys
            Coordinate system in which this ray is expressed.  Default: the
            global coordinate system.
        """
        ret = cls.__new__(cls)
        ret._r = np.ascontiguousarray([x, y, z]).T
        ret._v = np.ascontiguousarray([vx, vy, vz]).T
        ret._t = np.ascontiguousarray(t)
        ret._wavelength = np.ascontiguousarray(w)
        ret._flux = np.ascontiguousarray(flux)
        ret._vignetted = np.ascontiguousarray(vignetted)
        ret._failed = np.ascontiguousarray(vignetted)
        ret.coordSys = coordSys
        return ret

    @property
    def r(self):
        self._rv.r.syncToHost()
        return self._r

    @property
    def x(self):
        self._rv.r.syncToHost()
        return self._r[:, 0]

    @property
    def y(self):
        self._rv.r.syncToHost()
        return self._r[:, 1]

    @property
    def z(self):
        self._rv.r.syncToHost()
        return self._r[:, 2]

    @property
    def v(self):
        self._rv.v.syncToHost()
        return self._v

    @property
    def vx(self):
        self._rv.v.syncToHost()
        return self._v[:, 0]

    @property
    def vy(self):
        self._rv.v.syncToHost()
        return self._v[:, 1]

    @property
    def vz(self):
        self._rv.v.syncToHost()
        return self._v[:, 2]

    @property
    def t(self):
        self._rv.t.syncToHost()
        return self._t

    @property
    def wavelength(self):
        # wavelength is constant, so no need to synchronize
        return self._wavelength

    @property
    def flux(self):
        self._rv.flux.syncToHost()
        return self._flux

    @property
    def vignetted(self):
        self._rv.vignetted.syncToHost()
        return self._vignetted

    @property
    def failed(self):
        self._rv.failed.syncToHost()
        return self._failed

    @lazy_property
    def _rv(self):
        return _batoid.CPPRayVector2(
            self._r.ctypes.data, self._v.ctypes.data, self._t.ctypes.data,
            self._wavelength.ctypes.data, self._flux.ctypes.data,
            self._vignetted.ctypes.data, self._failed.ctypes.data,
            len(self._wavelength), self.coordSys._coordSys
        )

    def copy(self):
        # copy on host side for now...
        return self.fromArrays(
            self.x.copy(), self.y.copy(), self.z.copy(),
            self.vx.copy(), self.vy.copy(), self.vz.copy(),
            self.t.copy(), self.wavelength.copy(), self.flux.copy(),
            self.vignetted.copy(), self.failed.copy()
        )

    def toCoordSysInPlace(self, coordSys):
        transform = CoordTransform2(self.coordSys, coordSys)
        transform.applyForwardInPlace(self)

    def __len__(self):
        return self._rv.t.size;
