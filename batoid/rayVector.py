from numbers import Real, Integral
from collections.abc import Sequence

import numpy as np

from . import _batoid
from .constants import vacuum, globalCoordSys
from .coordsys import CoordSys, CoordTransform

class RayVector:
    """A sequence of `Ray`s.

    Parameters
    ----------
    rays : list of Ray

    Methods
    -------
    fromArrays(x, y, z, vx, vy, vz, t, w, flux=1, vignetted=None)
        Factory function alternate constructor.
    positionAtTime(t)
        Calculate positions of rays at time `t`.
    propagatedToTime(t)
        Propagate ray to time `t`.
    propagatedInPlace(t)
        Propagate ray to time `t` in place.
    phase(r, t)
        Calculate ray phases at position `r` and time `t`.
    amplitude(r, t)
        Calculate ray amplitudes at position `r` and time `t`.
    sumAmplitude(r, t)
        Calculate sum of ray amplitudes at position `r` and time `t`.
    trimVignetted(minFlux)
        Remove vignetted rays or rays with flux below threshold.
    trimVignettedInPlace(minFlux)
        Remove vignetted rays or rays with flux below threshold in place.

    Attributes
    ----------
    r : (n, 3), array of float
        Ray positions in meters.
    x, y, z : array of float
        X, Y, Z coordinates of ray positions.
    v : (n, 3), array of float
        Ray velocities in meters per second.
    vx, vy, vz : array of float
        X, Y, Z coordinates of ray velocities.
    t : array of float
        Reference time (divided by the speed of light in vacuum) for rays in
        units of meters.
    wavelength : array of float
        Vacuum wavelength of rays in meters.
    flux : array of float
        Flux of rays in arbitrary units.
    vignetted : array of bool
        Whether rays are vignetted or not.
    failed : array of bool
        Whether rays have failed or not.
    k : (n, 3) array of float
        Wavevectors of Rays in radians per meter.
    kx, ky, kz : array of float
        X, Y, Z components of wavevectors.
    omega : array of float
        Temporal frequency of plane waves (divided by the speed of light in
        vacuum) in inverse meters.
    monochromatic : bool
        True only if all rays are same wavelength.
    """
    def __init__(self, rays, wavelength=float("nan")):
        if len(rays) < 1:
            raise ValueError("No Rays from which to create RayVector")
        if isinstance(rays, RayVector):
            self._r = _batoid.RayVector(rays._r)
        elif isinstance(rays, Sequence):
            self._r = _batoid.RayVector([ray._r for ray in rays], wavelength)
        else:
            raise ValueError("Wrong arguments to RayVector")

    @classmethod
    def fromArrays(cls, x, y, z, vx, vy, vz, t, w, flux=1, vignetted=False):
        """Create RayVector from 1d parameter arrays.

        Parameters
        ----------
        x, y, z : array of float
            Positions of rays in meters.
        vx, vy, vz : array of float
            Velocities of rays in units of the speed of light in vacuum.
        t : Reference times (divided by the speed of light in vacuum) in units
            of meters.
        wavelength : array of float
            Vacuum wavelengths in meters.
        flux : array of float
            Fluxes in arbitrary units.
        vignetted : array of bool
            Which rays have been vignetted.
        """
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
    def asGrid(cls, source, wavelength,
               nx=None, ny=None,
               dx=None, dy=None,
               lx=None, ly=None,
               flux=1,
               medium=vacuum,
               dirCos=None,
               nrandom=None,
               interface=None,
               doCenter=False):
        """
        Create RayVector on a parallelogram shaped region.

        Parameters
        ----------
        source : float or (3,) array of float
            If float, then distance of central Ray from (0,0,z(0,0)),
            If array of float, then rays originate from this point.
        wavelength : float
            Vacuum wavelength of all rays in meters.
        nx, ny : int, optional
            Number of Rays on a side.
        dx, dy : float or (2,) array of float, optional
            Differential separation of ray intersections in x and y directions.
        lx, ly : float, optional
            Length of each side of ray grid.
        flux : float
            Flux to assign each Ray.
        dirCos : (3,) array of float
            If source is float, then direction cosines of ray velocities.
            If source is array of float, then unused.
        medium : batoid.Medium
            Initial medium of each Ray.
        nrandom : int
            If not None, then uniformly sample this many rays from
            parallelogram region instead of sampling on a regular grid.
        interface : batoid.Interface, optional
            Interface from which grid is projected.
        doCenter : bool
            If True, then center the returned RayVector around (0,0).  If
            False, then follow Fourier conventions for centering.
        """
        from .optic import Interface
        from .surface import Plane

        # To determine the parallelogram, exactly 2 of nx, dx, lx must be set.
        if sum(a is not None for a in [nx, dx, lx]) != 2:
            raise ValueError("Exactly 2 of nx, dx, lx must be specified")

        if interface is None:
            interface = Interface(Plane())

        if nx is not None and ny is None:
            ny = nx
        if dx is not None and dy is None:
            dy = dx
        if lx is not None and ly is None:
            if isinstance(lx, Real):
                ly = lx
            else:
                ly = np.dot(np.array([[0, -1], [1, 0]]), lx)

        if nx is not None and dx is not None:
            if (nx%2) == 0:
                lx = dx*(nx-2)
            else:
                lx = dx*(nx-1)
            if (ny%2) == 0:
                ly = dy*(ny-2)
            else:
                ly = dy*(ny-1)
        elif lx is not None and dx is not None:
            # adjust dx in this case
            # always infer an even n (since even and odd are degenerate given
            # only lx, dx)
            nx = int(lx/dx//2)*2+2
            ny = int(ly/dy//2)*2+2
            dx = lx/(nx-2)
            dy = ly/(ny-2)

        if isinstance(lx, Real):
            lx = (lx, 0.0)
        if isinstance(ly, Real):
            ly = (0.0, ly)

        if nrandom is not None:
            x = np.random.uniform(-0.5, 0.5, size=nrandom)
            y = np.random.uniform(-0.5, 0.5, size=nrandom)
        else:
            x_d = (nx-(2 if (nx%2) == 0 else 1))/nx
            y_d = (ny-(2 if (ny%2) == 0 else 1))/ny
            x = np.fft.fftshift(np.fft.fftfreq(nx, x_d))
            y = np.fft.fftshift(np.fft.fftfreq(ny, y_d))
            x, y = np.meshgrid(x, y)
            x = x.ravel()
            y = y.ravel()
        stack = np.stack([x, y])
        x = np.dot(lx, stack)
        y = np.dot(ly, stack)
        z = interface.surface.sag(x, y)
        transform = CoordTransform(interface.coordSys, globalCoordSys)
        x, y, z = transform.applyForward(x, y, z)

        t = np.zeros_like(x)
        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)

        return cls._finish(source, dirCos, n, x, y, z, t, w, flux)

    @classmethod
    def asPolar(cls, source, wavelength,
                outer, inner=0.0,
                nrad=None, naz=None,
                flux=1,
                medium=vacuum,
                dirCos=None,
                nrandom=None,
                interface=None):
        """
        """
        from .optic import Interface
        from .surface import Plane

        if interface is None:
            interface = Interface(Plane())

        if nrandom is None:
            ths = []
            rs = []
            for r in np.linspace(outer, inner, nrad):
                nphi = int(naz*r/outer//6)*6
                ths.append(np.linspace(0, 2*np.pi, nphi, endpoint=False))
                rs.append(np.ones(nphi)*r)
            # Point in center is a special case
            if inner == 0.0:
                ths[-1] = np.array([0.0])
                rs[-1] = np.array([0.0])
            r = np.concatenate(rs)
            th = np.concatenate(ths)
        else:
            r = np.sqrt(np.random.uniform(inner**2, outer**2, size=nrandom))
            th = np.random.uniform(0, 2*np.pi, size=nrandom)
        x = r*np.cos(th)
        y = r*np.sin(th)
        z = interface.surface.sag(x, y)
        transform = CoordTransform(interface.coordSys, globalCoordSys)
        x, y, z = transform.applyForward(x, y, z)
        t = np.zeros_like(x)
        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)

        return cls._finish(source, dirCos, n, x, y, z, t, w, flux)

    @classmethod
    def asSpokes(cls, source, wavelength,
                 outer=None,
                 inner=0.0,
                 spokes=None,
                 rings=None,
                 spacing='uniform',
                 flux=1,
                 medium=vacuum,
                 dirCos=None,
                 interface=None):
        """
        """
        from .optic import Interface
        from .surface import Plane

        if interface is None:
            interface = Interface(Plane())

        if isinstance(rings, Integral):
            if spacing == 'uniform':
                rings = np.linspace(inner, outer, rings)
            elif spacing == 'GQ':
                if spokes is None:
                    spokes = 2*rings+1
                Li, w = np.polynomial.legendre.leggauss(rings)
                rings = np.sqrt((1+Li)/2)
                flux = w*(2*np.pi)/(4*spokes)
            if isinstance(spokes, Integral):
                spokes = np.linspace(0, 2*np.pi, spokes, endpoint=False)
        rings, spokes = np.meshgrid(rings, spokes)
        flux = np.broadcast_to(flux, rings.shape)
        rings = rings.ravel()
        spokes = spokes.ravel()
        flux = flux.ravel()

        x = rings*np.cos(spokes)
        y = rings*np.sin(spokes)
        z = interface.surface.sag(x, y)
        transform = CoordTransform(interface.coordSys, globalCoordSys)
        x, y, z = transform.applyForward(x, y, z)
        t = np.zeros_like(x)
        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)

        return cls._finish(source, dirCos, n, x, y, z, t, w, flux)

    @classmethod
    def _finish(cls, source, dirCos, n, x, y, z, t, w, flux):
        """Map rays backwards to their source position.
        """
        from .surface import Plane
        if dirCos is not None:
            v = np.array(dirCos, dtype=float)
            v /= n*np.sqrt(np.dot(v, v))
            vx = np.empty_like(x)
            vx.fill(v[0])
            vy = np.empty_like(x)
            vy.fill(v[1])
            vz = np.empty_like(x)
            vz.fill(v[2])
            # Now need to raytrace backwards to the plane source units away.
            rays = RayVector.fromArrays(x, y, z, -vx, -vy, -vz, t, w, flux=flux)

            zhat = -n*v
            xhat = np.cross(np.array([1.0, 0.0, 0.0]), zhat)
            xhat /= np.sqrt(np.dot(xhat, xhat))
            yhat = np.cross(xhat, zhat)
            origin = zhat*source
            coordSys = CoordSys(origin, np.stack([xhat, yhat, zhat]).T)
            transform = CoordTransform(globalCoordSys, coordSys)
            transform.applyForwardInPlace(rays)
            plane = Plane()
            plane.intersectInPlace(rays)
            transform.applyReverseInPlace(rays)
            return RayVector.fromArrays(rays.x, rays.y, rays.z, vx, vy, vz, t, w, flux=flux)
        else:
            vx = x - source[0]
            vy = y - source[1]
            vz = z - source[2]
            v = np.stack([vx, vy, vz])
            v /= n*np.einsum('ab,ab->b', v, v)
            x.fill(source[0])
            y.fill(source[1])
            z.fill(source[2])
            return RayVector.fromArrays(x, y, z, v[0], v[1], v[2], t, w, flux=flux)

    @classmethod
    def _fromRayVector(cls, _r):
        ret = cls.__new__(cls)
        ret._r = _r
        return ret

    def __repr__(self):
        return repr(self._r)

    def amplitude(self, r, t):
        """Calculate (scalar) complex electric-field amplitudes at given
        position and time.

        Parameters
        ----------
        position : (3,), array of float
            Position in meters at which to compute phases.
        time : float
            Time (over the speed of light; in meters) at which to compute
            phases.

        Returns
        -------
        amplitude : array of complex
        """
        return self._r.amplitude(r, t)

    def sumAmplitude(self, r, t):
        """Calculate the sum of (scalar) complex electric-field amplitudes of
        all rays at given position and time.

        Parameters
        ----------
        position : (3,), array of float
            Position in meters at which to compute phases.
        time : float
            Time (over the speed of light; in meters) at which to compute
            phases.

        Returns
        -------
        amplitude : complex
        """
        return self._r.sumAmplitude(r, t)

    def phase(self, r, t):
        """Calculate plane wave phases at given position and time.

        Parameters
        ----------
        position : (3,), array of float
            Position at which to compute phases
        time : float
            Time (over the speed of light; in meters) at which to compute
            phases.

        Returns
        -------
        phases : array of float
        """
        return self._r.phase(r, t)

    def positionAtTime(self, t):
        """Calculate the positions of the rays at a given time.

        Parameters
        ----------
        time : float
            Time (over the speed of light; in meters) at which to compute
            positions.

        Returns
        -------
        position : (N,3), array of float
            Positions in meters.
        """
        return self._r.positionAtTime(t)

    def propagatedToTime(self, t):
        """Return a RayVector propagated to given time.

        Parameters
        ----------
        time : float
            Time (over the speed of light; in meters) to which to propagate
            rays.

        Returns
        -------
        propagatedRays : RayVector
        """
        return RayVector._fromRayVector(self._r.propagatedToTime(t))

    def propagateInPlace(self, t):
        """Propagate RayVector to given time.

        Parameters
        ----------
        t : float
            Time (over the speed of light; in meters) to which to propagate
            rays.
        """
        self._r.propagateInPlace(t)

    def trimVignetted(self, minflux=0.0):
        """Return new RayVector with vignetted rays or rays with flux below
        given threshold removed.

        Parameters
        ----------
        minFlux : float
            Minimum flux value to not remove.

        Returns
        -------
        trimmedRays : RayVector
        """
        return RayVector._fromRayVector(self._r.trimVignetted(minflux))

    def trimVignettedInPlace(self, minflux=0.0):
        """Remove vignetted rays and rays with flux below a given threshold.

        Parameters
        ----------
        minFlux : float
            Minimum flux value to not remove.
        """
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
        from .ray import Ray
        return Ray._fromRay(self._r[idx])

    def __iter__(self):
        self._iter = iter(self._r)
        return self

    def __next__(self):
        from .ray import Ray
        return Ray._fromRay(next(self._iter))

    def __len__(self):
        return len(self._r)

    def __eq__(self, rhs):
        if not isinstance(rhs, RayVector): return False
        return self._r == rhs._r

    def __ne__(self, rhs):
        return not (self == rhs)

def concatenateRayVectors(rvs):
    """Concatenates two or more RayVectors together.

    Parameters
    ----------
    rvs : list of RayVector
        RayVectors to concatenate.

    Returns
    -------
    concatenatedRayVector : RayVector
    """
    _r = _batoid.concatenateRayVectors([rv._r for rv in rvs])
    return RayVector._fromRayVector(_r)


def rayGrid(zdist, length, xcos, ycos, zcos, nside, wavelength, flux, medium,
            lattice=False):
    """Construct a parallel square grid of rays in a given direction.

    Parameters
    ----------
    zdist : float
        Distance of central Ray from origin.
    length : float
        Length of one side of square grid in meters.
    xcos, ycos, zcos : float
        Direction cosines of rays.
    nside : int
        Number of rays on a side.
    wavelength : float
        Vacuum wavelength of rays in meters.
    flux : float
        Flux of rays in arbitrary units.
    medium : batoid.Medium
        Medium containing rays.
    lattice : bool
        Whether to center grid as a batoid.Lattice or not.

    Returns
    -------
    grid : RayVector
        The grid of rays.
    """
    return RayVector._fromRayVector(
        _batoid.rayGrid(
            zdist, length, xcos, ycos, zcos, nside,
            wavelength, flux, medium, lattice
        )
    )

def circularGrid(zdist, outer, inner, xcos, ycos, zcos, nradii, naz,
                 wavelength, flux, medium):
    """Construct a hexapolar grid of rays in a given direction.

    Parameters
    ----------
    zdist : float
        Distance of central Ray from origin.
    outer : float
        Outer radius of grid in meters.
    inner : float
        Inner radius of grid in meters.
    xcos, ycos, zcos : float
        Direction cosines of rays.
    nradii : int
        Number of radii (rings) in hexapolar grid.
    naz : int
        Number of azimuthal positions along outermost ring.
    wavelength : float
        Vacuum wavelength of rays in meters.
    flux : float
        Flux of rays in arbitrary units.
    medium : batoid.Medium
        Medium containing rays.

    Returns
    -------
    hexgrid : RayVector
        The hexapolar grid of rays.
    """
    return RayVector._fromRayVector(
        _batoid.circularGrid(
            zdist, outer, inner, xcos, ycos, zcos, nradii, naz, wavelength,
            flux, medium
        )
    )

def uniformCircularGrid(zdist, outer, inner, xcos, ycos, zcos, nrays,
                        wavelength, flux, medium, seed=0):
    """Uniformly sample ray positions from an annulus, assign all the same
    direction.

    Parameters
    ----------
    zdist : float
        Distance of central Ray from origin.
    outer : float
        Outer radius of grid in meters.
    inner : float
        Inner radius of grid in meters.
    xcos, ycos, zcos : float
        Direction cosines of rays.
    nrays : int
        Number of rays to create.
    wavelength : float
        Vacuum wavelength of rays in meters.
    flux : float
        Flux of rays in arbitrary units.
    medium : batoid.Medium
        Medium containing rays.

    Returns
    -------
    hexgrid : RayVector
        The hexapolar grid of rays.
    """
    return RayVector._fromRayVector(
        _batoid.uniformCircularGrid(
            zdist, outer, inner, xcos, ycos, zcos, nrays, wavelength, flux,
            medium, seed
        )
    )

def pointSourceCircularGrid(source, outer, inner, nradii, naz, wavelength,
                            flux, medium):
    """Construct grid of rays all emanating from the same source location but
    with a hexapolar grid in direction cosines.

    Parameters
    ----------
    source : (3,) array of float
        Source position of rays.
    outer : float
        Outer radius of rays at intersection with plane perpendicular to
        source-origin line.
    inner : float
        Inner radius of rays at intersection with plane perpendicular to
        source-origin line.
    nradii : int
        Number of radii (rings) in hexapolar grid.
    naz : int
        Number of azimuthal positions along outermost ring.
    wavelength : float
        Vacuum wavelength of rays in meters.
    flux : float
        Flux of rays in arbitrary units.
    medium : batoid.Medium
        Medium containing rays.

    Returns
    -------
    hexgrid : RayVector
        The hexapolar grid of rays.
    """
    return RayVector._fromRayVector(
        _batoid.pointSourceCircularGrid(source, outer, inner, nradii, naz, wavelength, flux, medium)
    )
