from functools import cached_property
from numbers import Real, Integral

import numpy as np

from . import _batoid
from .constants import globalCoordSys, vacuum
from .coordTransform import CoordTransform
from .trace import applyForwardTransform, applyForwardTransformArrays
from .utils import fieldToDirCos, hexapolar
from .surface import Plane


def _reshape_arrays(arrays, shape, dtype=float):
    for i in range(len(arrays)):
        array = arrays[i]
        if not hasattr(array, 'shape') or array.shape != shape:
            arrays[i] = np.array(np.broadcast_to(array, shape))
        arrays[i] = np.ascontiguousarray(arrays[i], dtype=dtype)
    return arrays


class RayVector:
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
    def __init__(
        self, x, y, z, vx, vy, vz, t=0.0, wavelength=0.0, flux=1.0,
        vignetted=False, failed=False, coordSys=globalCoordSys
    ):
        shape = np.broadcast(
            x, y, z, vx, vy, vz, t, wavelength, flux, vignetted, failed
        ).shape
        x, y, z, vx, vy, vz, t, wavelength, flux = _reshape_arrays(
            [x, y, z, vx, vy, vz, t, wavelength, flux],
            shape
        )
        vignetted, failed = _reshape_arrays(
            [vignetted, failed],
            shape,
            bool
        )

        self._x = x
        self._y = y
        self._z = z
        self._vx = vx
        self._vy = vy
        self._vz = vz
        self._t = t
        self._wavelength = wavelength
        self._flux = flux
        self._vignetted = vignetted
        self._failed = failed

        self.coordSys = coordSys

    @staticmethod
    def _directInit(
        x, y, z, vx, vy, vz, t, wavelength, flux, vignetted, failed, coordSys
    ):
        ret = RayVector.__new__(RayVector)
        ret._x = x
        ret._y = y
        ret._z = z
        ret._vx = vx
        ret._vy = vy
        ret._vz = vz
        ret._t = t
        ret._wavelength = wavelength
        ret._flux = flux
        ret._vignetted = vignetted
        ret._failed = failed
        ret.coordSys = coordSys
        return ret

    def _hash(self):
        # Don't implement as __hash__ since RayVector is mutable.
        return hash((
            tuple(self.x.tolist()),
            tuple(self.y.tolist()),
            tuple(self.z.tolist()),
            tuple(self.vx.tolist()),
            tuple(self.vy.tolist()),
            tuple(self.vz.tolist()),
            tuple(self.t.tolist()),
            tuple(self.wavelength.tolist()),
            tuple(self.flux.tolist()),
            tuple(self.vignetted.tolist()),
            tuple(self.failed.tolist()),
            self.coordSys
        ))

    def positionAtTime(self, t):
        """Calculate the positions of the rays at a given time.

        Parameters
        ----------
        t : float
            Time (over vacuum speed of light; in meters).

        Returns
        -------
        ndarray of float, shape (n, 3)
            Positions in meters.
        """
        from .global_vars import _batoid_max_threads
        x = np.empty(len(self._x))
        y = np.empty(len(self._x))
        z = np.empty(len(self._x))
        self._rv.positionAtTime(
            t,
            x.ctypes.data, y.ctypes.data, z.ctypes.data,
            _batoid_max_threads
        )
        return np.array([x, y, z]).T

    def propagate(self, t):
        """Propagate this RayVector to given time.

        Parameters
        ----------
        t : float
            Time (over vacuum speed of light; in meters).

        Returns
        -------
        RayVector
            Reference to self, no copy is made.
        """
        from .global_vars import _batoid_max_threads
        self._rv.propagateInPlace(t, _batoid_max_threads)
        return self

    def phase(self, r, t):
        """Calculate plane wave phases at given position and time.

        Parameters
        ----------
        r : ndarray of float, shape (3,)
            Position in meters at which to compute phase
        t : float
            Time (over vacuum speed of light; in meters).

        Returns
        -------
        ndarray of float, shape(n,)
        """
        from .global_vars import _batoid_max_threads
        out = np.empty_like(self._t)
        self._rv.phase(
            r[0], r[1], r[2], t, out.ctypes.data, _batoid_max_threads
        )
        return out

    def amplitude(self, r, t):
        """Calculate (scalar) complex electric-field amplitudes at given
        position and time.

        Parameters
        ----------
        r : ndarray of float, shape (3,)
            Position in meters.
        t : float
            Time (over vacuum speed of light; in meters).

        Returns
        -------
        ndarray of complex, shape (n,)
        """
        from .global_vars import _batoid_max_threads
        out = np.empty_like(self._t, dtype=np.complex128)
        self._rv.amplitude(
            r[0], r[1], r[2], t, out.ctypes.data, _batoid_max_threads
        )
        return out

    def sumAmplitude(self, r, t, ignoreVignetted=True):
        """Calculate the sum of (scalar) complex electric-field amplitudes of
        all rays at given position and time.

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
        from .global_vars import _batoid_max_threads
        return self._rv.sumAmplitude(
            r[0], r[1], r[2], t, ignoreVignetted, _batoid_max_threads
        )

    @classmethod
    def asGrid(
        cls,
        optic=None, backDist=None, medium=None, stopSurface=None,
        coordSys=None,
        wavelength=None,
        source=None, dirCos=None,
        theta_x=None, theta_y=None, projection='postel',
        nx=None, ny=None,
        dx=None, dy=None,
        lx=None, ly=None,
        flux=1,
        nrandom=None, rng=None
    ):
        """Create RayVector on a parallelogram shaped region.

        This function will often be used to create a grid of rays on a square
        grid, but is flexible enough to also create grids on an arbitrary
        parallelogram, or even randomly distributed across an arbitrary
        parallelogram-shaped region.

        The algorithm starts by placing rays on the "stop" surface, and then
        backing them up such that they are in front of any surfaces of the
        optic they're intended to trace.

        The stop surface of most large telescopes is the plane perpendicular to
        the optic axis and flush with the rim of the primary mirror.  This
        plane is usually also the entrance pupil since there are no earlier
        refractive or reflective surfaces.  However, since this plane is a bit
        difficult to locate automatically, the default stop surface in batoid
        is the global x-y plane.

        If a telescope has a stopSurface attribute in its yaml file, then this
        is usually a good choice to use in this function.  Using a curved
        surface for the stop surface is allowed, but is usually a bad idea as
        this may lead to a non-uniformly illuminated pupil and is inconsistent
        with, say, an incoming uniform spherical wave or uniform plane wave.

        Parameters
        ----------
        optic : `batoid.Optic`, optional
            If present, then try to extract values for ``backDist``,
            ``medium``, ``stopSurface``, and ``lx`` from the Optic.  Note that
            values explicitly passed to `asGrid` as keyword arguments override
            those extracted from ``optic``.
        backDist : float, optional
            Map rays backwards from the stop surface to the plane that is
            perpendicular to the rays and ``backDist`` meters from the point
            (0, 0, z(0,0)) on the stop surface.  This should generally be set
            large enough that any obscurations or phantom surfaces occuring
            before the stop surface are now "in front" of the ray.  If this
            keyword is set to ``None`` and the ``optic`` keyword is set, then
            infer a value from ``optic.backDist``.  If both this keyword and
            ``optic`` are ``None``, then use a default of 40 meters, which
            should be sufficiently large for foreseeable telescopes.
        medium : `batoid.Medium`, optional
            Initial medium of each ray.  If this keyword is set to ``None`` and
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
            Vacuum wavelength of rays in meters.
        source : None or ndarray of float, shape (3,), optional
            Where rays originate.  If None, then rays originate an infinite
            distance away, in which case the ``dirCos`` kwarg must also be
            specified to set the direction of ray propagation.  If an ndarray,
            then the rays originate from this point in global coordinates and
            the ``dirCos`` kwarg is ignored.
        dirCos : ndarray of float, shape (3,), optional
            If source is None, then this indicates the initial direction of
            propagation of the rays.  If source is not None, then this is
            ignored.  Also see ``theta_x``, ``theta_y`` as an alternative to
            this keyword.
        theta_x, theta_y : float, optional
            Field angle in radians.  If source is None, then this indicates the
            initial direction of propagation of the rays.  If source is not
            None, then this is ignored.  Uses `utils.fieldToDirCos` to convert
            to direction cosines.  Also see ``dirCos`` as an alternative to
            this keyword.
        projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}, optional
            Projection used to convert field angle to direction cosines.
        nx, ny : int, optional
            Number of rays on each side of grid.
        dx, dy : float or (2,) array of float, optional
            Separation in meters between adjacent rays in grid.  If scalars,
            then the separations are exactly along the x and y directions.  If
            arrays, then these are interpretted as the primitive vectors for
            the first and second dimensions of the grid.  If only dx is
            explicitly specified, then dy will be inferred as a 90-degree
            rotation from dx with the same length as dx.
        lx, ly : float or (2,) array of float, optional
            Length of each side of ray grid.  If scalars, then these are
            measured along the x and y directions.  If arrays, then these also
            indicate the primitive vectors orientation of the grid.  If only
            lx is specified, then ly will be inferred as a 90-degree rotation
            from lx with the same length as lx.  If lx is ``None``, then first
            infer a value from ``nx`` and ``dx``, and if that doesn't work,
            infer a value from ``optic.pupilSize``.
        flux : float, optional
            Flux to assign each ray.  Default is 1.0.
        nrandom : None or int, optional
            If not None, then uniformly sample this many rays from
            parallelogram region instead of sampling on a regular grid.
        rng : None or int or `numpy.random.Generator`, optional
            Random number generator or seed to use for random sampling.
        """
        from .optic import Interface
        from .surface import Plane

        if optic is not None:
            if backDist is None:
                backDist = optic.backDist
            if medium is None:
                medium = optic.inMedium
            if stopSurface is None:
                try:
                    stopSurface = optic.stopSurface
                except AttributeError:
                    stopSurface = None
            if lx is None:
                # If nx and dx are both present, then let lx get inferred from
                # them.  Otherwise, infer from optic.
                if nx is None or dx is None:
                    lx = optic.pupilSize
            if coordSys is None:
                coordSys = optic.coordSys

        if backDist is None:
            backDist = 40.0
        if stopSurface is None:
            stopSurface = Interface(Plane())
        if medium is None:
            medium = vacuum
        if coordSys is None:
            coordSys = globalCoordSys

        if dirCos is None and source is None:
            dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)

        if wavelength is None:
            raise ValueError("Missing wavelength keyword")

        # To determine the parallelogram, exactly 2 of nx, dx, lx must be set.
        if sum(a is not None for a in [nx, dx, lx]) != 2:
            raise ValueError("Exactly 2 of nx, dx, lx must be specified")

        if nx is not None and ny is None:
            ny = nx
        if dx is not None and dy is None:
            dy = dx
        if lx is not None and ly is None:
            if isinstance(lx, Real):
                ly = lx
            else:
                ly = np.dot(np.array([[0, -1], [1, 0]]), lx)

        # We need lx, ly, nx, ny for below, so construct these from other
        # arguments if they're not already available.
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
            # only lx, dx).
            slop = 0.1  # prevent 3.9999 -> 3, e.g.
            nx = int((lx/dx+slop)//2)*2+2
            ny = int((ly/dy+slop)//2)*2+2
            # These are the real dx, dy; which may be different from what was
            # passed in order to force an integer for nx/ny.  We don't actually
            # need them after this point though.
            # dx = lx/(nx-2)
            # dy = ly/(ny-2)

        if isinstance(lx, Real):
            lx = (lx, 0.0)
        if isinstance(ly, Real):
            ly = (0.0, ly)

        if nrandom is not None:
            if rng is None:
                rng = np.random.default_rng()
            elif isinstance(rng, int):
                rng = np.random.default_rng(rng)
            xx = rng.uniform(-0.5, 0.5, size=nrandom)
            yy = rng.uniform(-0.5, 0.5, size=nrandom)
        else:
            if nx <= 2:
                x_d = 1.
            else:
                x_d = (nx-(2 if (nx%2) == 0 else 1))/nx
            if ny <= 2:
                y_d = 1.
            else:
                y_d = (ny-(2 if (ny%2) == 0 else 1))/ny
            xx = np.fft.fftshift(np.fft.fftfreq(nx, x_d))
            yy = np.fft.fftshift(np.fft.fftfreq(ny, y_d))
            xx, yy = np.meshgrid(xx, yy)
            xx = xx.ravel()
            yy = yy.ravel()
        stack = np.stack([xx, yy])
        x = np.dot(lx, stack)
        y = np.dot(ly, stack)
        del xx, yy, stack
        z = stopSurface.surface.sag(x, y)
        transform = CoordTransform(stopSurface.coordSys, coordSys)
        applyForwardTransformArrays(transform, x, y, z)
        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)

        return cls._finish(
            backDist, source, dirCos, n, x, y, z, w, flux, coordSys
        )

    @classmethod
    def asFan(
        cls,
        nx=None, ny=None,
        **kwargs
    ):
        rvs = []
        if nx > 1:
            rvs.append(RayVector.asGrid(nx=nx, ny=1, **kwargs))
        if ny > 1:
            rvs.append(RayVector.asGrid(nx=1, ny=ny, **kwargs))
        return concatenateRayVectors(rvs)

    @classmethod
    def asPolar(
        cls,
        optic=None, backDist=None, medium=None, stopSurface=None,
        coordSys=None,
        wavelength=None,
        outer=None, inner=None,
        source=None, dirCos=None,
        theta_x=None, theta_y=None, projection='postel',
        nrad=None, naz=None, kfold=6,
        flux=1,
        nrandom=None, rng=None
    ):
        """Create RayVector on an annular region using a hexapolar grid.  (Note
        that hexapolar is the default, but other values of kfold besides 6 are
        allowed.)

        This function can be used to regularly sample the entrance pupil of a
        telescope using polar symmetry (really, hexagonal symmetry).  Rings of
        different radii are used, with the number of samples on each ring
        restricted to a multiple of 6 (with the exception of a potential
        central "ring" of radius 0, which is only ever sampled once).  This may
        be more efficient than using a square grid since more of the rays
        generated may avoid vignetting.

        This function is also used to generate rays uniformly randomly sampled
        from a given annular region.

        The algorithm used here starts by placing rays on the "stop" surface,
        and then backing them up such that they are in front of any surfaces of
        the optic they're intended to trace.

        The stop surface of most large telescopes is the plane perpendicular to
        the optic axis and flush with the rim of the primary mirror.  This
        plane is usually also the entrance pupil since there are no earlier
        refractive or reflective surfaces.  However, since this plane is a bit
        difficult to locate automatically, the default stop surface in batoid
        is the global x-y plane.

        If a telescope has a stopSurface attribute in its yaml file, then this
        is usually a good choice to use in this function.  Using a curved
        surface for the stop surface is allowed, but is usually a bad idea as
        this may lead to a non-uniformly illuminated pupil and is inconsistent
        with, say, an incoming uniform spherical wave or uniform plane wave.

        Parameters
        ----------
        optic : `batoid.Optic`, optional
            If present, then try to extract values for ``backDist``,
            ``medium``, ``stopSurface``, and ``outer`` from the Optic.  Note
            that values explicitly passed to `asPolar` as keyword arguments
            override those extracted from ``optic``.
        backDist : float, optional
            Map rays backwards from the stop surface to the plane that is
            perpendicular to the ray and ``backDist`` meters from the point
            (0, 0, z(0,0)) on the stop surface.  This should generally be set
            large enough that any obscurations or phantom surfaces occuring
            before the stop surface are now "in front" of the ray.  If this
            keyword is set to ``None`` and the ``optic`` keyword is set, then
            infer a value from ``optic.backDist``.  If both this keyword and
            ``optic`` are ``None``, then use a default of 40 meters, which
            should be sufficiently large for foreseeable telescopes.
        medium : `batoid.Medium`, optional
            Initial medium of each ray.  If this keyword is set to ``None`` and
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
            Vacuum wavelength of rays in meters.
        outer : float
            Outer radius of annulus in meters.
        inner : float, optional
            Inner radius of annulus in meters.  Default is 0.0.
        source : None or ndarray of float, shape (3,), optional
            Where rays originate.  If None, then rays originate an infinite
            distance away, in which case the ``dirCos`` kwarg must also be
            specified to set the direction of ray propagation.  If an ndarray,
            then the rays originate from this point in global coordinates and
            the ``dirCos`` kwarg is ignored.
        dirCos : ndarray of float, shape (3,), optional
            If source is None, then this indicates the initial direction of
            propagation of the rays.  If source is not None, then this is
            ignored.  Also see ``theta_x``, ``theta_y`` as an alternative to
            this keyword.
        theta_x, theta_y : float, optional
            Field angle in radians.  If source is None, then this indicates the
            initial direction of propagation of the rays.  If source is not
            None, then this is ignored.  Uses `utils.fieldToDirCos` to convert
            to direction cosines.  Also see ``dirCos`` as an alternative to
            this keyword.
        projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}, optional
            Projection used to convert field angle to direction cosines.
        nrad : int
            Number of radii on which create rays.
        naz : int
            Approximate number of azimuthal angles uniformly spaced along the
            outermost ring.  Each ring is constrained to have a multiple of
            kfold azimuths, so the realized value may be slightly different
            than the input value here.  Inner rings will have fewer azimuths in
            proportion to their radius, but will still be constrained to a
            multiple of kfold.  (If the innermost ring has radius 0, then exactly
            1 ray, with azimuth undefined, will be used on that "ring".)
        kfold : int, optional
            Each ring will have a multiple of this many azimuths.  Default: 6.
        flux : float, optional
            Flux to assign each ray.  Default is 1.0.
        nrandom : int, optional
            If not None, then uniformly sample this many rays from annular
            region instead of sampling on a hexapolar grid.
        rng : None or int or `numpy.random.Generator`, optional
            Random number generator or seed to use for random sampling.
        """
        from .optic import Interface

        if optic is not None:
            if backDist is None:
                backDist = optic.backDist
            if medium is None:
                medium = optic.inMedium
            if stopSurface is None:
                stopSurface = optic.stopSurface
            if outer is None:
                outer = optic.pupilSize/2
            if inner is None:
                if hasattr(optic, 'pupilObscuration'):
                    inner = optic.pupilSize*optic.pupilObscuration/2
                else:
                    inner = 0.0
            if coordSys is None:
                coordSys = optic.coordSys

        else:
            if inner is None:
                inner = 0.0

        if backDist is None:
            backDist = 40.0
        if stopSurface is None:
            stopSurface = Interface(Plane())
        if medium is None:
            medium = vacuum
        if coordSys is None:
            coordSys = globalCoordSys

        if dirCos is None and source is None:
            dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)

        if wavelength is None:
            raise ValueError("Missing wavelength keyword")

        if nrandom is None:
            rr, th = hexapolar(
                outer=outer,
                inner=inner,
                nrad=nrad,
                naz=naz,
                kfold=kfold,
                rth=True
            )
        else:
            if rng is None:
                rng = np.random.default_rng()
            elif isinstance(rng, int):
                rng = np.random.default_rng(rng)
            rr = np.sqrt(rng.uniform(inner**2, outer**2, size=nrandom))
            th = rng.uniform(0, 2*np.pi, size=nrandom)
        x = rr*np.cos(th)
        y = rr*np.sin(th)
        del rr, th
        z = stopSurface.surface.sag(x, y)
        transform = CoordTransform(stopSurface.coordSys, coordSys)
        applyForwardTransformArrays(transform, x, y, z)
        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)

        return cls._finish(
            backDist, source, dirCos, n, x, y, z, w, flux, coordSys
        )

    @classmethod
    def asSpokes(
        cls,
        optic=None, backDist=None, medium=None, stopSurface=None,
        coordSys=None,
        wavelength=None,
        outer=None, inner=0.0,
        source=None, dirCos=None,
        theta_x=None, theta_y=None, projection='postel',
        spokes=None, rings=None,
        spacing='uniform',
        flux=1
    ):
        """Create RayVector on an annular region using a spokes pattern.

        The function generates rays on a rings-and-spokes pattern, with a fixed
        number of radii for each azimuth and a fixed number of azimuths for
        each radius.  Its main use is for decomposing functions in pupil space
        into Zernike components using Gaussian Quadrature integration on
        annuli.  For more general purpose annular sampling, RayVector.asPolar()
        is often a better choice since it samples the pupil more uniformly.

        The algorithm used here starts by placing rays on the "stop" surface,
        and then backing them up such that they are in front of any surfaces of
        the optic they're intended to trace.

        The stop surface of most large telescopes is the plane perpendicular to
        the optic axis and flush with the rim of the primary mirror.  This
        plane is usually also the entrance pupil since there are no earlier
        refractive or reflective surfaces.  However, since this plane is a bit
        difficult to locate automatically, the default stop surface in batoid
        is the global x-y plane.

        If a telescope has a stopSurface attribute in its yaml file, then this
        is usually a good choice to use in this function.  Using a curved
        surface for the stop surface is allowed, but is usually a bad idea as
        this may lead to a non-uniformly illuminated pupil and is inconsistent
        with, say, an incoming uniform spherical wave or uniform plane wave.

        Parameters
        ----------
        optic : `batoid.Optic`, optional
            If present, then try to extract values for ``backDist``,
            ``medium``, ``stopSurface``, and ``outer`` from the Optic.  Note
            that values explicitly passed to `asSpokes` as keyword arguments
            override those extracted from ``optic``.
        backDist : float, optional
            Map rays backwards from the stop surface to the plane that is
            perpendicular to the ray and ``backDist`` meters from the point
            (0, 0, z(0,0)) on the stop surface.  This should generally be set
            large enough that any obscurations or phantom surfaces occuring
            before the stop surface are now "in front" of the ray.  If this
            keyword is set to ``None`` and the ``optic`` keyword is set, then
            infer a value from ``optic.backDist``.  If both this keyword and
            ``optic`` are ``None``, then use a default of 40 meters, which
            should be sufficiently large for foreseeable telescopes.
        medium : `batoid.Medium`, optional
            Initial medium of each ray.  If this keyword is set to ``None`` and
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
            Vacuum wavelength of rays in meters.
        outer : float
            Outer radius of annulus in meters.
        inner : float, optional
            Inner radius of annulus in meters.  Default is 0.0.
        source : None or ndarray of float, shape (3,), optional
            Where rays originate.  If None, then rays originate an infinite
            distance away, in which case the ``dirCos`` kwarg must also be
            specified to set the direction of ray propagation.  If an ndarray,
            then the rays originate from this point in global coordinates and
            the ``dirCos`` kwarg is ignored.
        dirCos : ndarray of float, shape (3,), optional
            If source is None, then this indicates the initial direction of
            propagation of the rays.  If source is not None, then this is
            ignored.  Also see ``theta_x``, ``theta_y`` as an alternative to
            this keyword.
        theta_x, theta_y : float, optional
            Field angle in radians.  If source is None, then this indicates the
            initial direction of propagation of the rays.  If source is not
            None, then this is ignored.  Uses `utils.fieldToDirCos` to convert
            to direction cosines.  Also see ``dirCos`` as an alternative to
            this keyword.
        projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}, optional
            Projection used to convert field angle to direction cosines.
        spokes : int or ndarray of float
            If int, then number of spokes to use.
            If ndarray, then the values of the spokes azimuthal angles in
            radians.
        rings : int or ndarray of float
            If int, then number of rings to use.
            If array, then the values of the ring radii to use in meters.
        spacing : {'uniform', 'GQ'}
            If uniform, assign ring radii uniformly between ``inner`` and
            ``outer``.
            If GQ, then assign ring radii as the Gaussian Quadrature points
            for integration on an annulus.  In this case, the ray fluxes will
            be set to the Gaussian Quadrature weights (and the ``flux`` kwarg
            will be ignored).
        flux : float, optional
            Flux to assign each ray.  Default is 1.0.
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
            if outer is None:
                outer = optic.pupilSize/2
            if coordSys is None:
                coordSys = optic.coordSys

        if backDist is None:
            backDist = 40.0
        if stopSurface is None:
            stopSurface = Interface(Plane())
        if medium is None:
            medium = vacuum
        if coordSys is None:
            coordSys = globalCoordSys

        if dirCos is None and source is None:
            dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)

        if wavelength is None:
            raise ValueError("Missing wavelength keyword")

        if isinstance(rings, Integral):
            if spacing == 'uniform':
                rings = np.linspace(inner, outer, rings)
            elif spacing == 'GQ':
                if spokes is None:
                    spokes = 2*rings+1
                Li, w = np.polynomial.legendre.leggauss(rings)
                eps = inner/outer
                area = np.pi*(1-eps**2)
                rings = np.sqrt(eps**2 + (1+Li)*(1-eps**2)/2)*outer
                flux = w*area/(2*spokes)
            if isinstance(spokes, Integral):
                spokes = np.linspace(0, 2*np.pi, spokes, endpoint=False)
        rings, spokes = np.meshgrid(rings, spokes)
        flux = np.broadcast_to(flux, rings.shape)
        rings = rings.ravel()
        spokes = spokes.ravel()
        flux = flux.ravel()

        x = rings*np.cos(spokes)
        y = rings*np.sin(spokes)
        del rings, spokes
        z = stopSurface.surface.sag(x, y)
        transform = CoordTransform(stopSurface.coordSys, coordSys)
        applyForwardTransformArrays(transform, x, y, z)
        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)
        return cls._finish(
            backDist, source, dirCos, n, x, y, z, w, flux, coordSys
        )

    @classmethod
    def _finish(
        cls, backDist, source, dirCos, n, x, y, z, w, flux, coordSys
    ):
        """Map rays backwards to their source position."""
        if isinstance(flux, Real):
            flux = np.full(len(x), float(flux))
        if source is None:
            vv = np.array(dirCos, dtype=float)
            vv /= n*np.sqrt(np.dot(vv, vv))
            zhat = -n*vv
            xhat = np.cross(np.array([1.0, 0.0, 0.0]), zhat)
            xhat /= np.sqrt(np.dot(xhat, xhat))
            yhat = np.cross(xhat, zhat)
            origin = zhat*backDist
            rot = np.stack([xhat, yhat, zhat]).T
            _batoid.finishParallel(
                origin, rot.ravel(), vv,
                x.ctypes.data, y.ctypes.data, z.ctypes.data,
                len(x)
            )
            vx = np.full_like(x, vv[0])
            vy = np.full_like(y, vv[1])
            vz = np.full_like(z, vv[2])
            t = np.zeros(len(x), dtype=float)
            vignetted = np.zeros(len(x), dtype=bool)
            failed = np.zeros(len(x), dtype=bool)
            return RayVector._directInit(
                x, y, z, vx, vy, vz, t, w,
                flux, vignetted, failed, coordSys
            )
        else:
            pass
            # v = np.copy(r)
            # v -= source
            # v /= n*np.einsum('ab,ab->b', v, v)
            # r[:] = source
            # t = np.zeros(len(r), dtype=float)
            # vignetted = np.zeros(len(r), dtype=bool)
            # failed = np.zeros(len(r), dtype=bool)
            # return RayVector._directInit(
            #     r, v, t, w, flux, vignetted, failed, coordSys
            # )

    @classmethod
    def fromStop(
        cls, x, y,
        optic=None, backDist=None, medium=None, stopSurface=None,
        coordSys=None,
        wavelength=None,
        source=None, dirCos=None,
        theta_x=None, theta_y=None, projection='postel',
        flux=1
    ):
        """Create rays that intersects the "stop" surface at given points.

        The algorithm used here starts by placing the rays on the "stop"
        surface, and then backing them up such that they are in front of any
        surfaces of the optic they're intended to trace.

        The stop surface of most large telescopes is the plane perpendicular to
        the optic axis and flush with the rim of the primary mirror.  This
        plane is usually also the entrance pupil since there are no earlier
        refractive or reflective surfaces.  However, since this plane is a bit
        difficult to locate automatically, the default stop surface in batoid
        is the global x-y plane.

        If a telescope has a stopSurface attribute in its yaml file, then this
        is usually a good choice to use in this function.  Using a curved
        surface for the stop surface is allowed, but is usually a bad idea as
        this may lead to a non-uniformly illuminated pupil and is inconsistent
        with, say, an incoming uniform spherical wave or uniform plane wave.

        Parameters
        ----------
        x, y : ndarray
            X/Y coordinates on the stop surface where the rays would intersect
            if not refracted or reflected first.
        optic : `batoid.Optic`, optional
            If present, then try to extract values for ``backDist``,
            ``medium``, and ``stopSurface`` from the Optic.  Note that values
            explicitly passed here as keyword arguments override those
            extracted from ``optic``.
        backDist : float, optional
            Map rays backwards from the stop surface to the plane that is
            perpendicular to the rays and ``backDist`` meters from the point
            (0, 0, z(0,0)) on the stop surface.  This should generally be set
            large enough that any obscurations or phantom surfaces occuring
            before the stop surface are now "in front" of the ray.  If this
            keyword is set to ``None`` and the ``optic`` keyword is set, then
            infer a value from ``optic.backDist``.  If both this keyword and
            ``optic`` are ``None``, then use a default of 40 meters, which
            should be sufficiently large for foreseeable telescopes.
        medium : `batoid.Medium`, optional
            Initial medium of rays.  If this keyword is set to ``None`` and
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
            Vacuum wavelength of rays in meters.
        source : None or ndarray of float, shape (3,), optional
            Where the rays originate.  If None, then the rays originate an
            infinite distance away, in which case the ``dirCos`` kwarg must also
            be specified to set the direction of ray propagation.  If an
            ndarray, then the rays originates from this point in global
            coordinates and the ``dirCos`` kwarg is ignored.
        dirCos : ndarray of float, shape (3,), optional
            If source is None, then indicates the direction of ray propagation.
            If source is not None, then this is ignored.
        theta_x, theta_y : float, optional
            Field angle in radians.  If source is None, then this indicates the
            initial direction of propagation of the rays.  If source is not
            None, then this is ignored.  Uses `utils.fieldToDirCos` to convert
            to direction cosines.  Also see ``dirCos`` as an alternative to
            this keyword.
        projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}, optional
            Projection used to convert field angle to direction cosines.
        flux : float, optional
            Flux of rays.  Default is 1.0.
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
            if coordSys is None:
                coordSys = optic.coordSys

        if backDist is None:
            backDist = 40.0
        if stopSurface is None:
            stopSurface = Interface(Plane())
        if medium is None:
            medium = vacuum
        if coordSys is None:
            coordSys = globalCoordSys

        if dirCos is None and source is None:
            dirCos = fieldToDirCos(theta_x, theta_y, projection=projection)

        if wavelength is None:
            raise ValueError("Missing wavelength keyword")

        x = np.atleast_1d(x).astype(float, copy=False)
        y = np.atleast_1d(y).astype(float, copy=False)
        z = stopSurface.surface.sag(x, y)
        transform = CoordTransform(stopSurface.coordSys, coordSys)
        applyForwardTransformArrays(transform, x, y, z)

        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)

        return cls._finish(
            backDist, source, dirCos, n, x, y, z, w, flux, coordSys
        )

    @classmethod
    def fromFieldAngles(
        cls, theta_x, theta_y, projection='postel',
        optic=None, backDist=None, medium=None, stopSurface=None,
        wavelength=None,
        x=0, y=0,
        flux=1
    ):
        """Create RayVector with one stop surface point but many field angles.

        This method is similar to `fromStop` but broadcasts over ``theta_x``
        and ``theta_y`` instead of over ``x`` and ``y``.  There is less
        currently less effort paid to synchronizing the ``t`` values of the
        created rays, as they don't correspond to points on a physical incoming
        wavefront in this case.  The primary intended use case is to map chief
        rays (``x`` = ``y`` = 0) from incoming field angle to focal plane
        position.

        Parameters
        ----------
        theta_x, theta_y : ndarray
            Field angles in radians.
        projection : {'postel', 'zemax', 'gnomonic', 'stereographic', 'lambert', 'orthographic'}, optional
            Projection used to convert field angle to direction cosines.
        optic : `batoid.Optic`, optional
            If present, then try to extract values for ``backDist``,
            ``medium``, and ``stopSurface`` from the Optic.  Note that values
            explicitly passed here as keyword arguments override those
            extracted from ``optic``.
        backDist : float, optional
            Map rays backwards from the stop surface this far.  This should
            generally be set large enough that any obscurations or phantom
            surfaces occuring before the stop surface are now "in front" of the
            rays.  If this keyword is set to ``None`` and the ``optic`` keyword
            is set, then infer a value from ``optic.backDist``.  If both this
            keyword and ``optic`` are ``None``, then use a default of 40 meters,
            which should be sufficiently large for foreseeable telescopes.
        medium : `batoid.Medium`, optional
            Initial medium of rays.  If this keyword is set to ``None`` and
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
            Vacuum wavelength of rays in meters.
        x, y : float
            X/Y coordinates on the stop surface where the rays would intersect
            if not refracted or reflected first.
        flux : float, optional
            Flux of rays.  Default is 1.0.
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

        if wavelength is None:
            raise ValueError("Missing wavelength keyword")

        vx, vy, vz = fieldToDirCos(theta_x, theta_y, projection=projection)
        n = medium.getN(wavelength)
        vx /= n
        vy /= n
        vz /= n

        z = stopSurface.surface.sag(x, y)
        x = np.full_like(vx, x)
        y = np.full_like(vx, y)
        z = np.full_like(vx, z)
        t = np.zeros_like(vx)

        rv = RayVector(
            x, y, z,
            vx, vy, vz,
            t, wavelength, flux,
            coordSys=stopSurface.coordSys
        )
        rv.propagate(-backDist*n)

        return rv

    @property
    def r(self):
        """ndarray of float, shape (n, 3): Positions of rays in meters."""
        self._rv.x.syncToHost()
        self._rv.y.syncToHost()
        self._rv.z.syncToHost()
        return np.array([self._x, self._y, self._z]).T

    @property
    def x(self):
        """The x components of ray positions in meters."""
        self._rv.x.syncToHost()
        return self._x

    @property
    def y(self):
        """The y components of ray positions in meters."""
        self._rv.y.syncToHost()
        return self._y

    @property
    def z(self):
        """The z components of ray positions in meters."""
        self._rv.z.syncToHost()
        return self._z

    @property
    def v(self):
        """ndarray of float, shape (n, 3): Velocities of rays in units of the
        speed of light in vacuum.  Note that these may have magnitudes < 1 if
        the rays are inside a refractive medium.
        """
        self._rv.vx.syncToHost()
        self._rv.vy.syncToHost()
        self._rv.vz.syncToHost()
        return np.array([self._vx, self._vy, self._vz]).T

    @property
    def vx(self):
        """The x components of ray velocities units of the vacuum speed of
        light.
        """
        self._rv.vx.syncToHost()
        return self._vx

    @property
    def vy(self):
        """The y components of ray velocities units of the vacuum speed of
        light.
        """
        self._rv.vy.syncToHost()
        return self._vy

    @property
    def vz(self):
        """The z components of ray velocities units of the vacuum speed of
        light.
        """
        self._rv.vz.syncToHost()
        return self._vz

    @property
    def t(self):
        """Reference times (divided by the speed of light in vacuum) in units
        of meters, also known as the optical path lengths.
        """
        self._rv.t.syncToHost()
        return self._t

    @property
    def wavelength(self):
        """Vacuum wavelengths in meters."""
        # wavelength is constant, so no need to synchronize
        return self._wavelength

    @property
    def flux(self):
        """Fluxes in arbitrary units."""
        self._rv.flux.syncToHost()
        return self._flux

    @property
    def vignetted(self):
        """True for rays that have been vignetted."""
        self._rv.vignetted.syncToHost()
        return self._vignetted

    @property
    def failed(self):
        """True for rays that have failed.  This may occur, for example, if
        batoid failed to find the intersection of a ray wiht a surface.
        """
        self._rv.failed.syncToHost()
        return self._failed

    @property
    def k(self):
        r"""ndarray of float, shape (n, 3): Wavevectors of plane waves in units
        of radians per meter.  The magnitude of each wavevector is equal to
        :math:`2 \pi n / \lambda`, where :math:`n` is the refractive index and
        :math:`\lambda` is the wavelength.
        """
        v = self.v
        out = 2*np.pi*v
        out /= self.wavelength[:, None]
        out /= np.sum(v*v, axis=-1)[:, None]
        return out

    @property
    def kx(self):
        """The x component of each ray wavevector in radians per meter."""
        return self.k[:,0]

    @property
    def ky(self):
        """The y component of each ray wavevector in radians per meter."""
        return self.k[:,1]

    @property
    def kz(self):
        """The z component of each ray wavevector in radians per meter."""
        return self.k[:,2]

    @property
    def omega(self):
        r"""The temporal angular frequency of each plane wave divided by the
        vacuum speed of light in units of radians per meter.  Equals
        :math:`2 \pi / \lambda`.
        """
        return 2*np.pi/self.wavelength

    @cached_property
    def _rv(self):
        return _batoid.CPPRayVector(
            self._x.ctypes.data, self._y.ctypes.data, self._z.ctypes.data,
            self._vx.ctypes.data, self._vy.ctypes.data, self._vz.ctypes.data,
            self._t.ctypes.data,
            self._wavelength.ctypes.data, self._flux.ctypes.data,
            self._vignetted.ctypes.data, self._failed.ctypes.data,
            len(self._wavelength)
        )

    def _syncToHost(self):
        if "_rv" not in self.__dict__:
            # Was never copied to device, so still synchronized.
            return
        self._rv.x.syncToHost()
        self._rv.y.syncToHost()
        self._rv.z.syncToHost()
        self._rv.vx.syncToHost()
        self._rv.vy.syncToHost()
        self._rv.vz.syncToHost()
        self._rv.t.syncToHost()
        self._rv.wavelength.syncToHost()
        self._rv.flux.syncToHost()
        self._rv.vignetted.syncToHost()
        self._rv.failed.syncToHost()

    def _syncToDevice(self):
        self._rv.x.syncToDevice()
        self._rv.y.syncToDevice()
        self._rv.z.syncToDevice()
        self._rv.vx.syncToDevice()
        self._rv.vy.syncToDevice()
        self._rv.vz.syncToDevice()
        self._rv.t.syncToDevice()
        self._rv.wavelength.syncToDevice()
        self._rv.flux.syncToDevice()
        self._rv.vignetted.syncToDevice()
        self._rv.failed.syncToDevice()

    def copy(self):
        # copy on host side for now...
        self._syncToHost()
        ret = RayVector.__new__(RayVector)
        ret._x = np.copy(self._x)
        ret._y = np.copy(self._y)
        ret._z = np.copy(self._z)
        ret._vx = np.copy(self._vx)
        ret._vy = np.copy(self._vy)
        ret._vz = np.copy(self._vz)
        ret._t = np.copy(self._t)
        ret._wavelength = np.copy(self._wavelength)
        ret._flux = np.copy(self._flux)
        ret._vignetted = np.copy(self._vignetted)
        ret._failed = np.copy(self._failed)
        ret.coordSys = self.coordSys.copy()
        return ret

    def toCoordSys(self, coordSys):
        """Transform this RayVector into a new coordinate system.

        Parameters
        ----------
        coordSys: batoid.CoordSys
            Destination coordinate system.

        Returns
        -------
        RayVector
            Reference to self, no copy is made.
        """
        transform = CoordTransform(self.coordSys, coordSys)
        applyForwardTransform(transform, self)
        return self

    def __len__(self):
        return self._t.size

    def __eq__(self, rhs):
        return self._rv == rhs._rv

    def __ne__(self, rhs):
        return self._rv != rhs._rv

    def __repr__(self):
        out = f"RayVector({self.x!r}, {self.y!r}, {self.z!r}"
        out += f", {self.vx!r}, {self.vy!r}, {self.vz!r}"
        out += f", {self.t!r}, {self.wavelength!r}, {self.flux!r}"
        out += f", {self.vignetted!r}, {self.failed!r}, {self.coordSys!r})"
        return out

    def __getstate__(self):
        return (
            self.x, self.y, self.z,
            self.vx, self.vy, self.vz,
            self.t,
            self.wavelength, self.flux,
            self.vignetted, self.failed, self.coordSys
        )

    def __setstate__(self, args):
        (self._x, self._y, self._z,
         self._vx, self._vy, self._vz, self._t,
         self._wavelength, self._flux, self._vignetted,
         self._failed, self.coordSys) = args

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx >= 0:
                if idx >= self._rv.t.size:
                    msg = "index {} is out of bounds for axis 0 with size {}"
                    msg = msg.format(idx, self._rv.t.size)
                    raise IndexError(msg)
                idx = slice(idx, idx+1)
            else:
                if idx < -self._rv.t.size:
                    msg = "index {} is out of bounds for axis 0 with size {}"
                    msg = msg.format(idx, self._rv.t.size)
                    raise IndexError(msg)
                idx = slice(self._rv.t.size+idx, self._rv.t.size-idx+1)

        self._syncToHost()
        return RayVector._directInit(
            np.copy(self._x[idx]),
            np.copy(self._y[idx]),
            np.copy(self._z[idx]),
            np.copy(self._vx[idx]),
            np.copy(self._vy[idx]),
            np.copy(self._vz[idx]),
            np.copy(self._t[idx]),
            np.copy(self._wavelength[idx]),
            np.copy(self._flux[idx]),
            np.copy(self._vignetted[idx]),
            np.copy(self._failed[idx]),
            self.coordSys
        )

def concatenateRayVectors(rvs):
    return RayVector(
        np.hstack([rv.x for rv in rvs]),
        np.hstack([rv.y for rv in rvs]),
        np.hstack([rv.z for rv in rvs]),
        np.hstack([rv.vx for rv in rvs]),
        np.hstack([rv.vy for rv in rvs]),
        np.hstack([rv.vz for rv in rvs]),
        np.hstack([rv.t for rv in rvs]),
        np.hstack([rv.wavelength for rv in rvs]),
        np.hstack([rv.flux for rv in rvs]),
        np.hstack([rv.vignetted for rv in rvs]),
        np.hstack([rv.failed for rv in rvs]),
        rvs[0].coordSys
    )
