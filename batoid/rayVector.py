from numbers import Real, Integral
from collections.abc import Sequence

import numpy as np

from . import _batoid
from .constants import vacuum, globalCoordSys
from .coordsys import CoordSys, CoordTransform
from .ray import Ray
from .utils import fieldToDirCos


class RayVector:
    """A sequence of `Ray` s.

    Parameters
    ----------
    rays : list of Ray
        The Rays to assemble into a RayVector.  Note that all Rays must have
        the same coordSys.
    """
    def __init__(self, rays):
        if len(rays) < 1:
            raise ValueError("No Rays from which to create RayVector")
        if isinstance(rays, RayVector):
            self._r = _batoid.RayVector(rays._r)
        elif isinstance(rays, Sequence):
            wavelength = rays[0].wavelength
            for r in rays:
                if r.wavelength != wavelength:
                    wavelength = float("nan")
                    break
            self._r = _batoid.RayVector([ray._r for ray in rays], wavelength)
        else:
            raise ValueError("Wrong arguments to RayVector")
        self.coordSys = rays[0].coordSys

    @classmethod
    def fromArrays(cls, x, y, z, vx, vy, vz, t, w, flux=1, vignetted=False,
                   coordSys=globalCoordSys):
        """Create RayVector from 1d parameter arrays.

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
        ret.coordSys = coordSys
        return ret

    @classmethod
    def asGrid(
        cls,
        optic=None, backDist=None, medium=None, stopSurface=None,
        wavelength=None,
        source=None, dirCos=None,
        theta_x=None, theta_y=None, projection='postel',
        nx=None, ny=None,
        dx=None, dy=None,
        lx=None, ly=None,
        flux=1,
        nrandom=None
    ):
        """Create RayVector on a parallelogram shaped region.

        This function will often be used to create a grid of rays on a square
        grid, but is flexible enough to also create gris on an arbitrary
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

        If a telescope has an stopSurface attribute in its yaml file, then this
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
        z = stopSurface.surface.sag(x, y)
        transform = CoordTransform(stopSurface.coordSys, globalCoordSys)
        x, y, z = transform.applyForward(x, y, z)

        t = np.zeros_like(x)
        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)

        return cls._finish(backDist, source, dirCos, n, x, y, z, t, w, flux)

    @classmethod
    def asPolar(
        cls,
        optic=None, backDist=None, medium=None, stopSurface=None,
        wavelength=None,
        outer=None, inner=0.0,
        source=None, dirCos=None,
        theta_x=None, theta_y=None, projection='postel',
        nrad=None, naz=None,
        flux=1,
        nrandom=None
    ):
        """Create RayVector on an annular region using a hexapolar grid.

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

        If a telescope has an stopSurface attribute in its yaml file, then this
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
            outermost ring.  Each ring is constrained to have a multiple of 6
            azimuths, so the realized value may be slightly different than
            the input value here.  Inner rings will have fewer azimuths in
            proportion to their radius, but will still be constrained to a
            multiple of 6.  (If the innermost ring has radius 0, then exactly
            1 ray, with azimuth undefined, will be used on that "ring".)
        flux : float, optional
            Flux to assign each ray.  Default is 1.0.
        nrandom : int, optional
            If not None, then uniformly sample this many rays from annular
            region instead of sampling on a hexapolar grid.
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
            if outer is None:
                outer = optic.pupilSize/2

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
        z = stopSurface.surface.sag(x, y)
        transform = CoordTransform(stopSurface.coordSys, globalCoordSys)
        x, y, z = transform.applyForward(x, y, z)
        t = np.zeros_like(x)
        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)

        return cls._finish(backDist, source, dirCos, n, x, y, z, t, w, flux)

    @classmethod
    def asSpokes(
        cls,
        optic=None, backDist=None, medium=None, stopSurface=None,
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

        If a telescope has an stopSurface attribute in its yaml file, then this
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

        if isinstance(rings, Integral):
            if spacing == 'uniform':
                rings = np.linspace(inner, outer, rings)
            elif spacing == 'GQ':
                if spokes is None:
                    spokes = 2*rings+1
                Li, w = np.polynomial.legendre.leggauss(rings)
                eps = inner/outer
                rings = np.sqrt(eps**2 + (1+Li)*(1-eps**2)/2)*outer
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
        z = stopSurface.surface.sag(x, y)
        transform = CoordTransform(stopSurface.coordSys, globalCoordSys)
        x, y, z = transform.applyForward(x, y, z)
        t = np.zeros_like(x)
        w = np.empty_like(x)
        w.fill(wavelength)
        n = medium.getN(wavelength)

        return cls._finish(backDist, source, dirCos, n, x, y, z, t, w, flux)

    @classmethod
    def _finish(cls, backDist, source, dirCos, n, x, y, z, t, w, flux):
        """Map rays backwards to their source position."""
        from .surface import Plane
        if source is None:
            v = np.array(dirCos, dtype=float)
            v /= n*np.sqrt(np.dot(v, v))
            vx = np.empty_like(x)
            vx.fill(v[0])
            vy = np.empty_like(x)
            vy.fill(v[1])
            vz = np.empty_like(x)
            vz.fill(v[2])
            # Now need to raytrace backwards to the plane dist units away.
            rays = RayVector.fromArrays(x, y, z, -vx, -vy, -vz, t, w, flux=flux)

            zhat = -n*v
            xhat = np.cross(np.array([1.0, 0.0, 0.0]), zhat)
            xhat /= np.sqrt(np.dot(xhat, xhat))
            yhat = np.cross(xhat, zhat)
            origin = zhat*backDist
            cs = CoordSys(origin, np.stack([xhat, yhat, zhat]).T)
            transform = CoordTransform(globalCoordSys, cs)
            transform.applyForwardInPlace(rays)
            plane = Plane()
            plane.intersectInPlace(rays)
            transform.applyReverseInPlace(rays)
            return RayVector.fromArrays(
                rays.x, rays.y, rays.z, vx, vy, vz, t, w, flux=flux
            )
        else:
            vx = x - source[0]
            vy = y - source[1]
            vz = z - source[2]
            v = np.stack([vx, vy, vz])
            v /= n*np.einsum('ab,ab->b', v, v)
            x.fill(source[0])
            y.fill(source[1])
            z.fill(source[2])
            return RayVector.fromArrays(
                x, y, z, v[0], v[1], v[2], t, w, flux=flux
            )

    @classmethod
    def _fromRayVector(cls, _r, coordSys=globalCoordSys):
        """Turn a c++ RayVector into a python RayVector."""
        ret = cls.__new__(cls)
        ret._r = _r
        ret.coordSys=coordSys
        return ret

    def __repr__(self):
        return repr(self._r)

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
        return self._r.amplitude(r, t)

    def sumAmplitude(self, r, t):
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
        return self._r.sumAmplitude(r, t)

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
        return self._r.phase(r, t)

    def toCoordSys(self, coordSys):
        transform = CoordTransform(self.coordSys, coordSys)
        return transform.applyForward(self)

    def toCoordSysInPlace(self, coordSys):
        transform = CoordTransform(self.coordSys, coordSys)
        transform.applyForwardInPlace(self)

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
        return self._r.positionAtTime(t)

    def propagatedToTime(self, t):
        """Return a RayVector propagated to given time.

        Parameters
        ----------
        t : float
            Time (over vacuum speed of light; in meters).

        Returns
        -------
        RayVector
        """
        return RayVector._fromRayVector(self._r.propagatedToTime(t))

    def propagateInPlace(self, t):
        """Propagate RayVector to given time.

        Parameters
        ----------
        t : float
            Time (over vacuum speed of light; in meters).
        """
        self._r.propagateInPlace(t)

    def trimVignetted(self, minflux=0.0):
        """Return new RayVector with vignetted rays or rays with flux below
        given threshold removed.

        Parameters
        ----------
        minflux : float
            Minimum flux value to not remove.

        Returns
        -------
        RayVector
        """
        return RayVector._fromRayVector(
            self._r.trimVignetted(minflux), self.coordSys
        )

    def trimVignettedInPlace(self, minflux=0.0):
        """Remove vignetted rays and rays with flux below a given threshold.

        Parameters
        ----------
        minflux : float
            Minimum flux value to not remove.
        """
        self._r.trimVignettedInPlace(minflux)

    @property
    def monochromatic(self):
        """True if all rays have same wavelength."""
        return self._r.monochromatic

    @property
    def x(self):
        """The x components of ray positions in meters."""
        return self._r.x

    @property
    def y(self):
        """The y components of ray positions in meters."""
        return self._r.y

    @property
    def z(self):
        """The z components of ray positions in meters."""
        return self._r.z

    @property
    def vx(self):
        """The x components of ray velocities units of the vacuum speed of
        light.
        """
        return self._r.vx

    @property
    def vy(self):
        """The y components of ray velocities units of the vacuum speed of
        light.
        """
        return self._r.vy

    @property
    def vz(self):
        """The z components of ray velocities units of the vacuum speed of
        light.
        """
        return self._r.vz

    @property
    def t(self):
        """Reference times (divided by the speed of light in vacuum) in units
        of meters, also known as the optical path lengths.
        """
        return self._r.t

    @property
    def wavelength(self):
        """Vacuum wavelengths in meters."""
        return self._r.wavelength

    @property
    def flux(self):
        """Fluxes in arbitrary units."""
        return self._r.flux

    @property
    def vignetted(self):
        """True for rays that have been vignetted."""
        return self._r.vignetted

    @property
    def failed(self):
        """True for rays that have failed.  This may occur, for example, if
        batoid failed to find the intersection of a ray wiht a surface.
        """
        return self._r.failed

    @property
    def r(self):
        """ndarray of float, shape (n, 3): Positions of rays in meters."""
        return self._r.r

    @property
    def v(self):
        """ndarray of float, shape (n, 3): Velocities of rays in units of the
        speed of light in vacuum.  Note that these may have magnitudes < 1 if
        the rays are inside a refractive medium.
        """
        return self._r.v

    @property
    def k(self):
        r"""ndarray of float, shape (n, 3): Wavevectors of plane waves in units
        of radians per meter.  The magnitude of each wavevector is equal to
        :math:`2 \pi n / \lambda`, where :math:`n` is the refractive index and
        :math:`\lambda` is the wavelength.
        """
        return self._r.k

    @property
    def kx(self):
        """The x component of each ray wavevector in radians per meter."""
        return self._r.kx

    @property
    def ky(self):
        """The y component of each ray wavevector in radians per meter."""
        return self._r.ky

    @property
    def kz(self):
        """The z component of each ray wavevector in radians per meter."""
        return self._r.kz

    @property
    def omega(self):
        r"""The temporal angular frequency of each plane wave divided by the
        vacuum speed of light in units of radians per meter.  Equals
        :math:`2 \pi / \lambda`.
        """
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
    if len(rvs) == 0:
        return RayVector._fromRayVector(_batoid.RayVector())
    coordSys = rvs[0].coordSys
    for rv in rvs[1:]:
        if rv.coordSys != coordSys:
            raise ValueError(
                "Cannot concatenate RayVectors with different coordinate systems"
            )
    _r = _batoid.concatenateRayVectors([rv._r for rv in rvs])
    return RayVector._fromRayVector(_r, coordSys)


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
            wavelength, flux, medium._medium, lattice
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
            flux, medium._medium
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
            medium._medium, seed
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
        _batoid.pointSourceCircularGrid(
            source, outer, inner, nradii, naz, wavelength, flux, medium._medium
        )
    )
