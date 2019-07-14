from . import _batoid
from .ray import Ray
from collections.abc import Sequence
from numbers import Real

class RayVector:
    """
    """
    def __init__(self, *args, wavelength=None):
        if len(args) == 0:
            self._r = _batoid.RayVector()
        elif len(args) == 1:
            args = args[0]
            if isinstance(args, _batoid.RayVector):
                self._r = args
            elif isinstance(args, RayVector):
                self._r = _batoid.RayVector(args._r)
            elif isinstance(args, Sequence):
                if wavelength is not None:
                    self._r = _batoid.RayVector([a._r for a in args], wavelength)
                else:
                    self._r = _batoid.RayVector([a._r for a in args])
        elif len(args) == 10:
            self._r = _batoid.RayVector(*args)
        else:
            raise ValueError("Wrong arguments to RayVector()")

    @classmethod
    def fromArrays(cls, x, y, z, vx, vy, vz, t, w, flux=1, vignetted=None):
        n = len(x)
        if isinstance(flux, Real):
            tmp = flux
            flux = np.empty_like(x)
            flux.fill(tmp)
        if isinstance(vignetted, bool):
            tmp = vignetted
            vignetted = np.empty_like(x, dtype=bool)
            vignetted.fill(tmp)
        ret = cls.__new__(cls)
        ret._r = _batoid.RayVector(x, y, z, vx, vy, vz, t, w, flux, vignetted)
        return ret

    @classmethod
    def _fromRayVector(cls, _r):
        ret = cls.__new__(cls)
        ret._r = _r
        return ret

    def __repr__(self):
        return repr(self._r)

    def amplitude(self, r, t):
        return self._r.amplitude(r, t)

    def sumAmplitude(self, r, t):
        return self._r.sumAmplitude(r, t)

    def phase(self, r, t):
        return self._r.phase(r, t)

    def positionAtTime(self, t):
        return self._r.positionAtTime(t)

    def propagatedToTime(self, t):
        return RayVector(self._r.propagatedToTime(t))

    def propagateInPlace(self, t):
        self._r.propagateInPlace(t)

    def trimVignetted(self, minflux=0.0):
        return RayVector(self._r.trimVignetted(minflux))

    def trimVignettedInPlace(self, minflux=0.0):
        self._r.trimVignettedInPlace(minflux)

    @property
    def monochromatic(self):
        return self._r.monochromatic

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
    def r(self):
        return self._r.r

    @property
    def v(self):
        return self._r.v

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

    def __getitem__(self, idx):
        return Ray._fromRay(self._r[idx])

    def __iter__(self):
        self._iter = iter(self._r)
        return self

    def __next__(self):
        return Ray._fromRay(next(self._iter))

    def __len__(self):
        return len(self._r)

    def __eq__(self, rhs):
        if not isinstance(rhs, RayVector): return False
        return self._r == rhs._r

    def __ne__(self, rhs):
        return not (self == rhs)

def concatenateRayVectors(args):
    _r = _batoid.concatenateRayVectors([a._r for a in args])
    return RayVector(_r)


def rayGrid(zdist, length, xcos, ycos, zcos, nside, wavelength, flux, medium, lattice=False):
    return RayVector._fromRayVector(
        _batoid.rayGrid(zdist, length, xcos, ycos, zcos, nside, wavelength, flux, medium, lattice)
    )

def circularGrid(zdist, outer, inner, xcos, ycos, zcos, nradii, naz, wavelength, flux, medium):
    return RayVector._fromRayVector(
        _batoid.circularGrid(zdist, outer, inner, xcos, ycos, zcos, nradii, naz, wavelength, flux, medium)
    )

def uniformCircularGrid(zdist, outer, inner, xcos, ycos, zcos, nrays, wavelength, flux, medium, seed=0):
    return RayVector._fromRayVector(
        _batoid.uniformCircularGrid(zdist, outer, inner, xcos, ycos, zcos, nrays, wavelength, flux, medium, seed)
    )

def pointSourceCircularGrid(source, outer, inner, nradii, naz, wavelength, flux, medium):
    return RayVector._fromRayVector(
        _batoid.pointSourceCircularGrid(source, outer, inner, nradii, naz, wavelength, flux, medium)
    )
