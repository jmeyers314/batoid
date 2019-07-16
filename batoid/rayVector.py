from . import _batoid
from .ray import Ray
from collections.abc import Sequence
from numbers import Real

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
    def fromArrays(cls, x, y, z, vx, vy, vz, t, w, flux=1, vignetted=None):
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
