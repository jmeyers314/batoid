from numbers import Real, Integral

import numpy as np

from . import _batoid
from .constants import globalCoordSys
from .coordSys import CoordSys
from .coordTransform import CoordTransform
from .utils import lazy_property


class RayVector:
    def __init__(
        self, x, y, z, vx, vy, vz, t=0, wavelength=500e-9, flux=1,
        vignetted=False, failed=False, coordSys=globalCoordSys
    ):
        """Create RayVector from 1d parameter arrays.  Always makes a copy
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
        n = len(x)
        self._r = np.ascontiguousarray([x, y, z], dtype=float).T
        self._v = np.ascontiguousarray([vx, vy, vz], dtype=float).T

        if isinstance(t, Real):
            self._t = np.empty_like(x)
            self._t.fill(t)
        else:
            self._t = np.ascontiguousarray(t)

        if isinstance(wavelength, Real):
            self._wavelength = np.empty_like(x)
            self._wavelength.fill(wavelength)
        else:
            self._wavelength = np.ascontiguousarray(wavelength)

        if isinstance(flux, Real):
            self._flux = np.empty_like(x)
            self._flux.fill(flux)
        else:
            self._flux = np.ascontiguousarray(flux)

        if isinstance(vignetted, bool):
            self._vignetted = np.empty_like(x, dtype=bool)
            self._vignetted.fill(vignetted)
        else:
            self._vignetted = np.ascontiguousarray(vignetted)

        if isinstance(failed, bool):
            self._failed = np.empty_like(x, dtype=bool)
            self._failed.fill(failed)
        else:
            self._failed = np.ascontiguousarray(failed)

        self.coordSys = coordSys

    def positionAtTime(self, t):
        out = np.empty_like(self._r)
        self._rv.positionAtTime(t, out.ctypes.data)
        return out

    def propagate(self, t):
        self._rv.propagateInPlace(t)

    def phase(self, r, t):
        out = np.empty_like(self._t)
        self._rv.phase(r[0], r[1], r[2], t, out.ctypes.data)
        return out

    def amplitude(self, r, t):
        out = np.empty_like(self._t, dtype=np.complex128)
        self._rv.amplitude(r[0], r[1], r[2], t, out.ctypes.data)
        return out

    def sumAmplitude(self, r, t):
        return self._rv.sumAmplitude(r[0], r[1], r[2], t)

    # @classmethod
    # def asPolar(
    #     cls,
    #     optic=None, backDist=None, medium=None, stopSurface=None,
    #     wavelength=None,
    #     outer=None, inner=0.0,
    #     source=None, dirCos=None,
    #     theta_x=None, theta_y=None, projection='postel',
    #     nrad=None, naz=None,
    #     flux=1,
    #     nrandom=None
    # ):
    #     from .optic import Interface
    #     from .surface2 import Plane2
    #     from .constants import vacuum2
    #
    #     if optic is not None:
    #         if backDist is None:
    #             backDist = optic.backDist
    #         if medium is None:
    #             medium = optic.inMedium
    #         if stopSurface is None:
    #             stopSurface = optic.stopSurface
    #         if outer is None:
    #             outer = optic.pupilSize/2
    #
    #     if backDist is None:
    #         backDist = 40.0
    #     if stopSurface is None:
    #         stopSurface = Interface(Plane2())
    #     if medium is None:
    #         medium = vacuum2
    #
    #     if dirCos is None and source is None:
    #         dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)
    #
    #     if wavelength is None:
    #         raise ValueError("Missing wavelength keyword")
    #
    #     if nrandom is None:
    #         ths = []
    #         rs = []
    #         for r in np.linspace(outer, inner, nrad):
    #             if r == 0:
    #                 break
    #             nphi = int((naz*r/outer)//6)*6
    #             if nphi == 0:
    #                 nphi = 6
    #             ths.append(np.linspace(0, 2*np.pi, nphi, endpoint=False))
    #             rs.append(np.ones(nphi)*r)
    #         # Point in center is a special case
    #         if inner == 0.0:
    #             ths[-1] = np.array([0.0])
    #             rs[-1] = np.array([0.0])
    #         r = np.concatenate(rs)
    #         th = np.concatenate(ths)
    #     else:
    #         r = np.sqrt(np.random.uniform(inner**2, outer**2, size=nrandom))
    #         th = np.random.uniform(0, 2*np.pi, size=nrandom)
    #     x = r*np.cos(th)
    #     y = r*np.sin(th)
    #     z = stopSurface.surface.sag(x, y)
    #     transform = CoordTransform(stopSurface.coordSys, globalCoordSys)
    #     x, y, z = transform.applyForward(x, y, z)
    #     t = np.zeros_like(x)
    #     w = np.empty_like(x)
    #     w.fill(wavelength)
    #     n = medium.getN(wavelength)
    #
    #     return cls._finish(backDist, source, dirCos, n, x, y, z, t, w, flux)
    #
    # @classmethod
    # def _finish(cls, backDist, source, dirCos, n, x, y, z, t, w, flux):
    #     """Map rays backwards to their source position."""
    #     from .surface2 import Plane2
    #     if source is None:
    #         v = np.array(dirCos, dtype=float)
    #         v /= n*np.sqrt(np.dot(v, v))
    #         vx = np.empty_like(x)
    #         vx.fill(v[0])
    #         vy = np.empty_like(x)
    #         vy.fill(v[1])
    #         vz = np.empty_like(x)
    #         vz.fill(v[2])
    #         # Now need to raytrace backwards to the plane dist units away.
    #         rays = RayVector2.fromArrays(
    #             x, y, z, -vx, -vy, -vz, t, w, flux=flux
    #         )
    #
    #         zhat = -n*v
    #         xhat = np.cross(np.array([1.0, 0.0, 0.0]), zhat)
    #         xhat /= np.sqrt(np.dot(xhat, xhat))
    #         yhat = np.cross(xhat, zhat)
    #         origin = zhat*backDist
    #         cs = CoordSys(origin, np.stack([xhat, yhat, zhat]).T)
    #         transform = CoordTransform2(globalCoordSys, cs)
    #         transform.applyForward(rays)
    #         plane = Plane2()
    #         plane.intersect(rays)
    #         transform.applyReverse(rays)
    #         return RayVector2.fromArrays(
    #             rays.x, rays.y, rays.z, vx, vy, vz, t, w, flux=flux
    #         )
    #     else:
    #         vx = x - source[0]
    #         vy = y - source[1]
    #         vz = z - source[2]
    #         v = np.stack([vx, vy, vz])
    #         v /= n*np.einsum('ab,ab->b', v, v)
    #         x.fill(source[0])
    #         y.fill(source[1])
    #         z.fill(source[2])
    #         return RayVector2.fromArrays(
    #             x, y, z, v[0], v[1], v[2], t, w, flux=flux
    #         )

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
        return _batoid.CPPRayVector(
            self._r.ctypes.data, self._v.ctypes.data, self._t.ctypes.data,
            self._wavelength.ctypes.data, self._flux.ctypes.data,
            self._vignetted.ctypes.data, self._failed.ctypes.data,
            len(self._wavelength)
        )

    def _syncToHost(self):
        self._rv.r.syncToHost()
        self._rv.v.syncToHost()
        self._rv.t.syncToHost()
        self._rv.wavelength.syncToHost()
        self._rv.flux.syncToHost()
        self._rv.vignetted.syncToHost()
        self._rv.failed.syncToHost()

    def _syncToDevice(self):
        self._rv.r.syncToDevice()
        self._rv.v.syncToDevice()
        self._rv.t.syncToDevice()
        self._rv.wavelength.syncToDevice()
        self._rv.flux.syncToDevice()
        self._rv.vignetted.syncToDevice()
        self._rv.failed.syncToDevice()

    def copy(self):
        # copy on host side for now...
        self._syncToHost()
        x = self._r[:, 0].copy()
        y = self._r[:, 1].copy()
        z = self._r[:, 2].copy()
        vx = self._v[:, 0].copy()
        vy = self._v[:, 1].copy()
        vz = self._v[:, 2].copy()
        t = self._t.copy()
        wavelength = self._wavelength.copy()
        flux = self._flux.copy()
        vignetted = self._vignetted.copy()
        failed = self._failed.copy()

        return RayVector(
            x, y, z,
            vx, vy, vz,
            t, wavelength, flux, vignetted, failed, self.coordSys.copy()
        )

    def toCoordSys(self, coordSys):
        transform = CoordTransform(self.coordSys, coordSys)
        transform.applyForward(self)
        return self

    def __len__(self):
        return self._rv.t.size;

    def __eq__(self, rhs):
        return self._rv == rhs._rv

    def __ne__(self, rhs):
        return self._rv != rhs._rv
