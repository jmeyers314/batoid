from collections import OrderedDict

import numpy as np

from .coating import SimpleCoating
from .obscuration import ObscNegation, ObscCircle, ObscAnnulus
from .constants import globalCoordSys, vacuum
from .coordsys import CoordTransform
from .rayVector import concatenateRayVectors
from .utils import lazy_property


class Optic:
    """The base class for all varieties of batoid optics and optical systems.

    An `Optic` can include anything from a single reflective or refractive
    surface to an entire telescope including multiple mirrors and/or lenses.

    Parameters
    ----------
    name : str, optional
        An name for this optic
    inMedium : `batoid.Medium`, optional
        Medium in which approaching rays reside.  Default: vacuum.
    outMedium : `batoid.Medium`, optional
        Medium in which rays will leave this optic.  Default: vacuum.
    coordSys : `batoid.CoordSys`, optional
        Coordinate system indicating the position and orientation of this
        optic's vertex with respect to the global coordinate system.  Default:
        the global coordinate system.
    skip : bool, optional
        Whether or not to skip this optic when tracing.  This can be useful if
        you want to trace only through part of a compound optic.  Default:
        False.
    **kwargs : other
        Other attributes to add as object attributes.
    """
    def __init__(self, name=None, coordSys=globalCoordSys, inMedium=vacuum,
                 outMedium=vacuum, skip=False, **kwargs):
        self.name = name
        self.coordSys = coordSys
        self.inMedium = inMedium
        self.outMedium = outMedium
        self.skip = False
        kwargs.pop('itemDict', None)
        self.__dict__.update(**kwargs)

    def _repr_helper(self):
        out = ""
        if self.name is not None:
            out += ", name={!r}".format(self.name)
        out += ", coordSys={!r}".format(self.coordSys)
        if self.inMedium != vacuum:
            out += ", inMedium={!r}".format(self.inMedium)
        if self.outMedium != vacuum:
            out += ", outMedium={!r}".format(self.outMedium)
        return out

    @classmethod
    def fromYaml(cls, filename):
        """Load an `Optic` (commonly a complete telescope) from the given yaml
        file.  This is the most common way to create an `Optic`.  If the file
        is not initially found, then look in the ``batoid.datadir`` directory
        and subdirectories for the first matching filename and use that.

        Look in ``batoid.datadir`` for examples of how to format a batoid optic
        yaml file.

        Parameters
        ----------
        filename : str
            Name of yaml file to load

        Returns
        -------
        `Optic`
        """
        import os
        import yaml
        try:
            with open(filename) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            import glob
            from . import datadir
            filenames = glob.glob(os.path.join(datadir, "**", "*.yaml"))
            for candidate in filenames:
                if os.path.basename(candidate) == filename:
                    with open(candidate) as f:
                        config = yaml.safe_load(f)
                    break
            else:
                raise FileNotFoundError(filename)
        from .parse import parse_optic
        return parse_optic(config['opticalSystem'])


class Interface(Optic):
    """An `Optic` representing a single surface.  Almost always one of the
    concrete subclasses (`Mirror`, `RefractiveInterface`, `Baffle`, `Detector`)
    should be instantiated, depending on whether rays should reflect, refract,
    vignette/pass-through, or stop at this surface.

    Parameters
    ----------
    surface : `batoid.Surface`
        The surface instance for this Interface.
    obscuration : `batoid.Obscuration`, optional
        `batoid.Obscuration` instance indicating which x,y coordinates are
        obscured/unobscured for rays intersecting this optic.  Default: None,
        which means don't apply any ray obscuration.
    **kwargs :
        Other parameters to forward to ``Optic.__init__``
    """
    def __init__(self, surface, obscuration=None, **kwargs):
        Optic.__init__(self, **kwargs)

        self.surface = surface
        self.obscuration = obscuration

        # Stealing inRadius and outRadius from self.obscuration.  These are
        # required for the draw methods.
        self.inRadius = 0.0
        self.outRadius = None
        if self.obscuration is not None:
            if isinstance(self.obscuration, ObscNegation):
                if isinstance(self.obscuration.original, ObscCircle):
                    self.outRadius = self.obscuration.original.radius
                elif isinstance(self.obscuration.original, ObscAnnulus):
                    self.outRadius = self.obscuration.original.outer
                    self.inRadius = self.obscuration.original.inner
            elif isinstance(self.obscuration, ObscCircle):
                self.outRadius = self.obscuration.radius
            elif isinstance(self.obscuration, ObscAnnulus):
                self.outRadius = self.obscuration.outer
                self.inRadius = self.obscuration.inner

    def __hash__(self):
        return hash((self.__class__.__name__, self.surface, self.obscuration,
                     self.name, self.inMedium, self.outMedium, self.coordSys))

    def draw3d(self, ax, **kwargs):
        """Draw this interface on a mplot3d axis.

        Parameters
        ----------
        ax : mplot3d.Axis
            Axis on which to draw this optic.
        """
        if self.outRadius is None:
            return
        transform = CoordTransform(self.coordSys, globalCoordSys)
        # Going to draw 4 objects here: inner circle, outer circle, sag along
        # x=0, sag along y=0 inner circle
        if self.inRadius != 0.0:
            th = np.linspace(0, 2*np.pi, 100)
            cth, sth = np.cos(th), np.sin(th)
            x = self.inRadius * cth
            y = self.inRadius * sth
            z = self.surface.sag(x, y)
            x, y, z = transform.applyForward(x, y, z)
            ax.plot(x, y, z, **kwargs)

        #outer circle
        th = np.linspace(0, 2*np.pi, 100)
        cth, sth = np.cos(th), np.sin(th)
        x = self.outRadius * cth
        y = self.outRadius * sth
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)

        #next, a line at X=0
        y = np.linspace(-self.outRadius, -self.inRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)
        y = np.linspace(self.inRadius, self.outRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)

        #next, a line at Y=0
        x = np.linspace(-self.outRadius, -self.inRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)
        x = np.linspace(self.inRadius, self.outRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)

    def getXZSlice(self, nslice=0):
        """Calculate global coordinates for an (x,z) slice through this
        interface.

        The calculation is split into two half slices: xlocal <= 0 and xlocal
        >= 0.  When the inner radius is zero, these half slices are merged into
        one.  Otherwise, the two half slices are returned separately.

        If the local coordinate system involves any rotation the resulting
        slice may not be calculated correctly since we are really slicing in
        (xlocal, zlocal) then transforming these to (xglobal, zglobal).

        Parameters
        ----------
        nslice : int
            Use the specified number of points on each half slice. When zero,
            the value will be calculated automatically (and will be 2 for
            planar surfaces).

        Returns
        -------
        tuple
            Tuple (xz1, xz2) of 1D arrays where xz1=[x1, z1] is the xlocal <= 0
            half slice and xz2=[x2, z2] is the xlocal >= 0 half slice.
        """
        from .surface import Plane
        slice = []
        if self.outRadius is None:
            return slice
        if nslice <= 0:
            if isinstance(self.surface, Plane):
                nslice = 2
            else:
                nslice = 50
        # Calculate (x,z) slice in local coordinates for x <= 0.
        x = np.linspace(-self.outRadius, -self.inRadius, nslice)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        # Transform slice to global coordinates.
        transform = CoordTransform(self.coordSys, globalCoordSys)
        xneg, yneg, zneg = transform.applyForward(x, y, z)
        if np.any(yneg != 0):
            print('WARNING: getXZSlice used for rotated surface "{0}".'
                .format(self.name)
            )
        # Calculate (x,z) slice in local coordinates for x >= 0.
        x *= -1
        x = x[::-1]
        z[:] = self.surface.sag(x, y)
        # Transform slice to global coordinates.
        xpos, ypos, zpos = transform.applyForward(x, y, z)
        if np.any(ypos != 0):
            print('WARNING: getXZSlice used for rotated surface "{0}".'
                .format(self.name)
            )
        slice.append(np.stack((xpos, zpos), axis=0))
        # Combine x <= 0 and x >= 0 half slices when inner = 0.
        if self.inRadius == 0:
            assert xneg[-1] == xpos[0] and zneg[-1] == zpos[0]
            return (
                np.stack((
                    np.hstack((xneg, xpos[1:])),
                    np.hstack((zneg, zpos[1:]))
                ), axis=0),
            )
        else:
            return (
                np.stack((xneg, zneg), axis=0),
                np.stack((xpos, zpos), axis=0)
            )

    def draw2d(self, ax, **kwargs):
        """Draw this interface on a 2d matplotlib axis.
        May not work if elements are non-circular or not axis-aligned.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis on which to draw this optic.
        """
        slice = self.getXZSlice()
        for x, z in slice:
            ax.plot(x, z, **kwargs)

    def trace(self, r):
        """Trace ray(s) through this optical element.

        Parameters
        ----------
        r : `batoid.Ray` or `batoid.RayVector`
            Input ray(s) to trace

        Returns
        -------
        `batoid.Ray` or `batoid.RayVector`
            Output ray(s)

        Notes
        -----
        Returned rays will be expressed in the local coordinate system of the
        Optic.  See `Ray.toCoordSys` or `RayVector.toCoordSys` to express rays
        in a different coordinate system.
        """
        if self.skip:
            return r
        r = r.toCoordSys(self.coordSys)

        # refract, reflect, pass-through - depending on subclass
        r = self.interact(r)

        if self.obscuration is not None:
            r = self.obscuration.obscure(r)

        return r

    def traceFull(self, r):
        """Trace ray(s) through this optical element, returning a full history
        of all surface intersections.

        Parameters
        ----------
        r : `batoid.Ray` or `batoid.RayVector`
            Input ray(s) to trace

        Returns
        -------
        OrderedDict of dict
            There will be one key-value pair for every Interface traced
            through (which for this class, is just a single Interface).  The
            values will be dicts with key-value pairs:

            ``'name'``
                name of Interface (str)
            ``'in'``
                the incoming ray(s) to that Interface (Ray or RayVector)
            ``'out'``
                the outgoing ray(s) from that Interface (Ray or RayVector)

        Notes
        -----
        Pay careful attention to the coordinate systems of the returned rays.
        These will generally differ from the original input coordinate system.
        To transform to another coordinate system, see `Ray.toCoordSys` or
        `RayVector.toCoordSys`.
        """
        result = OrderedDict()
        if not self.skip:
            result[self.name] = {
                'name':self.name,
                'in':r,
                'out':self.trace(r)
            }
        return result

    def traceInPlace(self, r):
        """Trace ray(s) through this optical element in place (result replaces
        input Ray or RayVector)

        Parameters
        ----------
        r : batoid.Ray or batoid.RayVector
            Input ray(s) to trace

        Returns
        -------
        Ray or RayVector

        Notes
        -----
        The return Ray or RayVector is present for convenience, but is actually
        an alias for the input Ray or RayVector that has had its values
        replaced.

        Returned rays will be expressed in the local coordinate system of the
        Optic.  See `Ray.toCoordSys` or `RayVector.toCoordSys` to express rays
        in a different coordinate system.
        """
        if self.skip:
            return r
        r.toCoordSysInPlace(self.coordSys)

        # refract, reflect, pass-through - depending on subclass
        self.interactInPlace(r)

        if self.obscuration is not None:
            self.obscuration.obscureInPlace(r)

        return r

    def traceReverse(self, r):
        """Trace ray(s) through this optical element in reverse.

        Parameters
        ----------
        r : batoid.Ray or batoid.RayVector
            Input ray(s) to trace

        Returns
        -------
        Ray or RayVector

        Notes
        -----
        You may need to reverse the directions of rays before using this
        method!

        Returned rays will be expressed in the local coordinate system of the
        Optic.  See `Ray.toCoordSys` or `RayVector.toCoordSys` to express rays
        in a different coordinate system.
        """
        if self.skip:
            return r
        r.toCoordSysInPlace(self.coordSys)

        r = self.interactReverse(r)

        if self.obscuration is not None:
            r = self.obscuration.obscure(r)

        return r

    def traceSplit(self, r, minFlux=1e-3, _verbose=False):
        """Trace ray(s) through this optical element, splitting the return
        values into rays that continue propagating in the "forward" direction,
        and those that were reflected into the "reverse" direction.  Fluxes of
        output rays are proportional to reflection/transmission coefficients of
        the interface (which may depend on wavelength and incidence angle).

        Parameters
        ----------
        r : batoid.Ray or batoid.RayVector
            Input ray(s) to trace
        minFlux : float
            Minimum flux value of ray(s) to continue propagating.
            Default: 1e-3.

        Returns
        -------
        forwardRays : batoid.Ray or batoid.RayVector
            Ray(s) propagating in the forward direction.
        reverseRays : batoid.Ray or batoid.RayVector
            Ray(s) propagating in the reverse direction.

        Notes
        -----
        Returned rays will be expressed in the local coordinate system of the
        Optic.  See `Ray.toCoordSys` or `RayVector.toCoordSys` to express rays
        in a different coordinate system.
        """
        if _verbose:
            strtemplate = ("traceSplit        {:15s} "
                           "flux = {:18.8f}   nphot = {:10d}")
            print(strtemplate.format(self.name, np.sum(r.flux), len(r)))
        if self.skip:
            return r, None
        r = r.toCoordSys(self.coordSys)

        rForward, rReverse = self.rSplit(r)

        # For now, apply obscuration equally forwards and backwards
        if self.obscuration is not None:
            self.obscuration.obscureInPlace(rForward)
            self.obscuration.obscureInPlace(rReverse)

        return rForward, rReverse

    def traceSplitReverse(self, r, minFlux=1e-3,_verbose=False):
        """Trace ray(s) through this optical element, splitting the return
        values into rays that propagate in the "forward" direction, and those
        that propagate in the "reverse" direction.  Incoming rays are assumed
        to be propagating in the reverse direction. Fluxes of output rays are
        proportional to reflection/transmission coefficients of the  interface
        (which may depend on wavelength and incidence angle).

        Parameters
        ----------
        r : batoid.Ray or batoid.RayVector
            Input ray(s) to trace
        minFlux : float
            Minimum flux value of ray(s) to continue propagating.
            Default: 1e-3.

        Returns
        -------
        forwardRays : batoid.Ray or batoid.RayVector
            Ray(s) propagating in the forward direction.
        reverseRays : batoid.Ray or batoid.RayVector
            Ray(s) propagating in the reverse direction.

        Notes
        -----
        Returned rays will be expressed in the local coordinate system of the
        Optic.  See `Ray.toCoordSys` or `RayVector.toCoordSys` to express rays
        in a different coordinate system.
        """
        if _verbose:
            strtemplate = ("traceSplitReverse {:15s} "
                           "flux = {:18.8f}   nphot = {:10d}")
            print(strtemplate.format(self.name, np.sum(r.flux), len(r)))
        if self.skip:
            return r, None
        r = r.toCoordSys(self.coordSys)

        rForward, rReverse = self.rSplitReverse(r)

        # For now, apply obscuration equally forwards and backwards
        if self.obscuration is not None:
            self.obscuration.obscureInPlace(rForward)
            self.obscuration.obscureInPlace(rReverse)

        return rForward, rReverse

    def clearObscuration(self, unless=()):
        if self.name not in unless:
            self.obscuration = None

    def interact(self, r):
        return self.surface.intersect(r)

    def interactReverse(self, r):
        return self.surface.intersect(r)

    def interactInPlace(self, r):
        self.surface.intersectInPlace(r)

    def __eq__(self, other):
        if not self.__class__ == other.__class__:
            return False
        return (self.surface == other.surface and
                self.obscuration == other.obscuration and
                self.name == other.name and
                self.inMedium == other.inMedium and
                self.outMedium == other.outMedium and
                self.coordSys == other.coordSys)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        out = "{!s}({!r}".format(self.__class__.__name__, self.surface)
        if self.obscuration is not None:
            out += ", obscuration={!r}".format(self.obscuration)
        out += Optic._repr_helper(self)
        out +=")"
        return out

    def withGlobalShift(self, shift):
        """Return a new `Interface` with its coordinate system shifted.

        Parameters
        ----------
        shift : array (3,)
            The coordinate shift, relative to the global coordinate system, to
            apply to self.coordSys

        Returns
        -------
        `batoid.Interface`
            Shifted interface.
        """
        ret = self.__class__.__new__(self.__class__)
        newDict = dict(self.__dict__)
        newDict['coordSys'] = self.coordSys.shiftGlobal(shift)
        del newDict['surface']
        ret.__init__(
            self.surface,
            **newDict
        )
        return ret

    def withLocalRotation(self, rot, rotOrigin=None, coordSys=None):
        """Return a new `Interface` with its coordinate system rotated.

        Parameters
        ----------
        rot : array (3,3)
            Rotation matrix wrt to the local coordinate system to apply.
        rotOrigin : array (3,)
            Origin of rotation.  Default: None means use [0,0,0]
        coordSys : `batoid.CoordSys`
            Coordinate system of rotOrigin above.  Default: None means use
            self.coordSys.

        Returns
        -------
        `batoid.Interface`
            Rotated interface.
        """
        if rotOrigin is None and coordSys is None:
            coordSys = self.coordSys
            rotOrigin = [0,0,0]
        ret = self.__class__.__new__(self.__class__)
        newDict = dict(self.__dict__)
        newDict['coordSys'] = self.coordSys.rotateLocal(
            rot, rotOrigin, coordSys
        )
        del newDict['surface']
        ret.__init__(
            self.surface,
            **newDict
        )
        return ret

    def withSurface(self, surface):
        """Return a new `Interface` with its surface attribute replaced.

        Parameters
        ----------
        surface : `batoid.Surface`
            New replacement surface.

        Returns
        -------
        `batoid.Interface`
            Interface with new surface.
        """
        ret = self.__class__.__new__(self.__class__)
        newDict = dict(self.__dict__)
        del newDict['surface']
        ret.__init__(surface, **newDict)
        return ret


class RefractiveInterface(Interface):
    """Specialization for refractive interfaces.

    Rays will interact with this surface by refracting through it.
    """
    def __init__(self, *args, **kwargs):
        Interface.__init__(self, *args, **kwargs)
        self.forwardCoating = SimpleCoating(
            reflectivity=0.0, transmissivity=1.0
        )
        self.reverseCoating = SimpleCoating(
            reflectivity=0.0, transmissivity=1.0
        )

    def interact(self, r):
        return self.surface.refract(r, self.inMedium, self.outMedium)

    def interactReverse(self, r):
        return self.surface.refract(r, self.outMedium, self.inMedium)

    def interactInPlace(self, r):
        self.surface.refractInPlace(r, self.inMedium, self.outMedium)

    def rSplit(self, r):
        reflectedR, refractedR = self.surface.rSplit(
            r, self.inMedium, self.outMedium, self.forwardCoating
        )
        return refractedR, reflectedR

    def rSplitReverse(self, r):
        reflectedR, refractedR = self.surface.rSplit(
            r, self.outMedium, self.inMedium, self.reverseCoating
        )
        # rays coming into a refractive interface from reverse direction,
        # means that the refracted rays are going in the reverse direction,
        # and the reflected rays are going in the forward direction.
        # so return reflected (forward) first.
        return reflectedR, refractedR


class Mirror(Interface):
    """Specialization for reflective interfaces.

    Rays will interact with this surface by reflecting off of it.
    """
    def __init__(self, *args, **kwargs):
        Interface.__init__(self, *args, **kwargs)
        self.forwardCoating = SimpleCoating(
            reflectivity=1.0, transmissivity=0.0
        )
        self.reverseCoating = SimpleCoating(
            reflectivity=1.0, transmissivity=0.0
        )

    def interact(self, r):
        return self.surface.reflect(r)

    def interactReverse(self, r):
        return self.surface.reflect(r)

    def interactInPlace(self, r):
        self.surface.reflectInPlace(r)

    def rSplit(self, r):
        reflectedR, refractedR = self.surface.rSplit(
            r, self.inMedium, self.outMedium, self.forwardCoating
        )
        return reflectedR, refractedR

    def rSplitReverse(self, r):
        reflectedR, refractedR = self.surface.rSplit(
            r, self.outMedium, self.inMedium, self.reverseCoating
        )
        return refractedR, reflectedR


class Detector(Interface):
    """Specialization for detector interfaces.

    Rays will interact with this surface by passing through it.  Usually,
    however, this is the last `Optic` in a `CompoundOptic`, so tracing will
    stop with this `Interface`.
    """
    def __init__(self, *args, **kwargs):
        Interface.__init__(self, *args, **kwargs)
        self.forwardCoating = SimpleCoating(
            reflectivity=0.0, transmissivity=1.0
        )
        self.reverseCoating = None

    def rSplit(self, r):
        reflectedR, refractedR = self.surface.rSplit(
            r, self.inMedium, self.outMedium, self.forwardCoating
        )
        return refractedR, reflectedR

    def rSplitReverse(self, r):
        reflectedR, refractedR = self.surface.rSplit(
            r, self.outMedium, self.inMedium, self.reverseCoating
        )
        return reflectedR, refractedR


class Baffle(Interface):
    """Specialization for baffle interfaces.

    Rays will interact with this optic by passing straight through.  However,
    the vignetting calculation will still be applied at this `Interface`.
    """
    def __init__(self, *args, **kwargs):
        Interface.__init__(self, *args, **kwargs)
        self.forwardCoating = SimpleCoating(
            reflectivity=0.0, transmissivity=1.0
        )
        self.reverseCoating = SimpleCoating(
            reflectivity=0.0, transmissivity=1.0
        )

    def rSplit(self, r):
        reflectedR, refractedR = self.surface.rSplit(
            r, self.inMedium, self.outMedium, self.forwardCoating
        )
        return refractedR, reflectedR

    def rSplitReverse(self, r):
        reflectedR, refractedR = self.surface.rSplit(
            r, self.outMedium, self.inMedium, self.reverseCoating
        )
        return reflectedR, refractedR


class CompoundOptic(Optic):
    """An `Optic` containing two or more `Optic` s as subitems.

    Ray traces will be carried out sequentially for the subitems.

    Parameters
    ----------
    items : list of `Optic`
        Subitems making up this compound optic.
    **kwargs :
        Other parameters to forward to Optic.__init__
    """
    def __init__(self, items, **kwargs):
        Optic.__init__(self, **kwargs)
        self.items = tuple(items)

    @lazy_property
    def itemDict(self):
        """Dictionary access of the entire hierarchy of subitems of this
        `CompoundOptic`.

        The key for the first level is just the name of the `CompoundOptic`,
        e.g., ``optic.itemDict['SubaruHSC']``.  The next level is accessed by
        appending a ``.``, e.g., ``optic.itemDict['SubaruHSC.HSC']`` and so on:
        ``optic.itemDict['SubaruHSC.HSC.ADC']``
        ``optic.itemDict['SubaruHSC.HSC.ADC.ADC1']``
        ``optic.itemDict['SubaruHSC.HSC.ADC.ADC1.ADC1_entrance']``

        Note: It's also possible to access subitems using the [] operator
        directly: ``optic['SubaruHSC.PM']``
        """
        out = {}
        out[self.name] = self
        for item in self.items:
            out[self.name+'.'+item.name] = item
            if hasattr(item, 'itemDict'):
                for k, v in item.itemDict.items():
                    out[self.name+'.'+k] = v
        return out

    def __getitem__(self, key):
        """Dictionary access to the entire hierarchy of subitems of this
        `CompoundOptic`.

        Either access through the fully-qualified name
        (``optic['LSST.LSSTCamera.L1']``) or by partially-qualified name
        (``optic['LSSTCamera.L1']`` or even ``optic['L1']``).  Note that
        partially-qualified name access is only available for unique
        partially-qualified names.
        """
        try:
            item = self.itemDict[key]
        except KeyError:
            # try accessing by local name instead of fully-qualified name.
            try:
                item = self.itemDict[self._names[key]]
            except KeyError:
                raise ValueError("Cannot find item {}".format(key))
        return item

    @lazy_property
    def _names(self):
        nameDict = {}
        duplicates = set()
        for k, v in self.itemDict.items():
            tokens = k.split('.')
            shortNames = [tokens[-1]]
            for token in reversed(tokens[:-1]):
                shortNames.append('.'.join([token, shortNames[-1]]))
            for shortName in shortNames:
                if shortName in nameDict:
                    duplicates.add(shortName)
                else:
                    nameDict[shortName] = k
        for name in duplicates:
            del nameDict[name]
        return nameDict

    def trace(self, r):
        """Recursively trace through all subitems of this `CompoundOptic`.

        Parameters
        ----------
        r : `batoid.Ray` or `batoid.RayVector`
            Input ray(s) to trace

        Returns
        -------
        `batoid.Ray` or `batoid.RayVector`
            Output ray(s)

        Notes
        -----
        Returned rays will be expressed in the local coordinate system of the
        last element of the CompoundOptic.  See `Ray.toCoordSys` or
        `RayVector.toCoordSys` to express rays in a different coordinate
        system.
        """
        if self.skip:
            return r  # Should probably make a copy()?
        for item in self.items:
            if not item.skip:
                r = item.trace(r)
        return r

    def traceFull(self, r):
        """Recursively trace ray(s) through this `CompoundOptic`, returning a
        full history of all surface intersections.

        Parameters
        ----------
        r : `batoid.Ray` or `batoid.RayVector`
            Input ray(s) to trace

        Returns
        -------
        OrderedDict of dict
            There will be one key-value pair for every Interface traced
            through.  The values will be dicts with key-value pairs:

            ``'name'``
                name of Interface (str)
            ``'in'``
                the incoming ray(s) to that Interface (Ray or RayVector)
            ``'out'``
                the outgoing ray(s) from that Interface (Ray or RayVector)
        Notes
        -----
        Pay careful attention to the coordinate systems of the returned rays.
        These will generally differ from the original input coordinate system.
        To transform to another coordinate system, see `Ray.toCoordSys` or
        `RayVector.toCoordSys`.
        """
        result = OrderedDict()
        if not self.skip:
            r_in = r
            for item in self.items:
                tf = item.traceFull(r_in)
                for k, v in tf.items():
                    result[k] = v
                    r_in = v['out']
        return result

    def traceInPlace(self, r):
        """Recursively trace ray(s) through this `CompoundOptic` in place
        (result replaces input `Ray` or `RayVector`)

        Parameters
        ----------
        r : `batoid.Ray` or `batoid.RayVector`
            Input ray(s) to trace

        Returns
        -------
        `Ray` or `RayVector`

        Notes
        -----
        The return `Ray` or `RayVector` is present for convenience, but is
        actually an alias for the input `Ray` or `RayVector` that has had its
        values replaced.

        Returned rays will be expressed in the local coordinate system of the
        last element of the CompoundOptic.  See `Ray.toCoordSys` or
        `RayVector.toCoordSys` to express rays in a different coordinate
        system.
        """
        if self.skip:
            return r
        for item in self.items:
            r = item.traceInPlace(r)
        return r

    def traceReverse(self, r):
        """Recursively trace ray(s) through this `CompoundOptic` in reverse.

        Parameters
        ----------
        r : `batoid.Ray` or `batoid.RayVector`
            Input ray(s) to trace

        Returns
        -------
        `Ray` or `RayVector`

        Notes
        -----
        You may need to reverse the directions of rays before using this
        method!

        Returned rays will be expressed in the local coordinate system of the
        first element of the CompoundOptic.  See `Ray.toCoordSys` or
        `RayVector.toCoordSys` to express rays in a different coordinate
        system.
        """
        if self.skip:
            return r
        for item in reversed(self.items):
            if not item.skip:
                r = item.traceReverse(r)
        return r

    def traceSplit(self, r, minFlux=1e-3, _verbose=False):
        """Recursively trace ray(s) through this `CompoundOptic`, splitting at
        each surface.

        The return values are rays that continuing in the "forward" direction,
        and those reflected into the "reverse" direction.  Fluxes of output
        rays are proportional to reflection/transmission coefficients of the
        interface (which may depend on wavelength and incidence angle).  Note
        that traceSplit is applied recursively, so a single call on a
        `CompoundOptic` may result in many combinations of
        reflections/refractions being applied internally before rays are either
        deleted by falling below the minFlux or reach the entrace/exit of the
        `CompoundOptic`.

        Parameters
        ----------
        r : `batoid.Ray` or `batoid.RayVector`
            Input ray(s) to trace
        minFlux : float
            Minimum flux value of ray(s) to continue propagating.
            Default: 1e-3.

        Returns
        -------
        forwardRays : `batoid.Ray` or `batoid.RayVector`
            Ray(s) propagating in the forward direction.
        reverseRays : `batoid.Ray` or `batoid.RayVector`
            Ray(s) propagating in the reverse direction.

        Notes
        -----
        Returned forward (reverse) rays will be expressed in the local
        coordinate system of the last (first) element of the CompoundOptic.
        See `Ray.toCoordSys` or `RayVector.toCoordSys` to express rays in a
        different coordinate system.
        """
        if _verbose:
            strtemplate = ("traceSplit        {:15s} "
                           "flux = {:18.8f}   nphot = {:10d}")
            print(strtemplate.format(self.name, np.sum(r.flux), len(r)))
        if self.skip:
            return r, None

        workQueue = [(r, "forward", 0)]

        outRForward = []
        outRReverse = []

        while workQueue:
            rays, direction, itemIndex = workQueue.pop()
            item = self.items[itemIndex]
            if direction == "forward":
                rForward, rReverse = item.traceSplit(
                    rays, minFlux=minFlux, _verbose=_verbose
                )
            elif direction == "reverse":
                rForward, rReverse = item.traceSplitReverse(
                    rays, minFlux=minFlux, _verbose=_verbose
                )
            else:
                raise RuntimeError("Shouldn't get here!")

            rForward.trimVignettedInPlace(minFlux)
            rReverse.trimVignettedInPlace(minFlux)

            if len(rReverse) > 0:
                if itemIndex == 0:
                    outRReverse.append(rReverse)
                else:
                    workQueue.append((rReverse, "reverse", itemIndex-1))

            if len(rForward) > 0:
                if itemIndex == len(self.items)-1:
                    outRForward.append(rForward)
                else:
                    workQueue.append((rForward, "forward", itemIndex+1))

        rForward = concatenateRayVectors(outRForward)
        rReverse = concatenateRayVectors(outRReverse)
        return rForward, rReverse

    def traceSplitReverse(self, r, minFlux=1e-3, _verbose=False):
        """Recursively trace ray(s) through this `CompoundOptic` in reverse,
        splitting at each surface.

        The return values are both rays continuing in the "forward" direction,
        and those reflected into the "reverse" direction.  Initially incoming
        rays are assumed to be propagating in the reverse direction.  Fluxes of
        output rays are proportional to reflection/transmission coefficients of
        the interface (which may depend on wavelength and incidence angle).
        Note that traceSplitReverse is applied recursively, so a single call on
        a `CompoundOptic` may result in many combinations of
        reflections/refractions being applied internally before rays are either
        deleted by falling below the minFlux or reach the entrace/exit of the
        `CompoundOptic`.

        Parameters
        ----------
        r : `batoid.Ray` or `batoid.RayVector`
            Input ray(s) to trace
        minFlux : float
            Minimum flux value of ray(s) to continue propagating.
            Default: 1e-3.

        Returns
        -------
        forwardRays : `batoid.Ray` or `batoid.RayVector`
            Ray(s) propagating in the forward direction.
        reverseRays : `batoid.Ray` or `batoid.RayVector`
            Ray(s) propagating in the reverse direction.

        Notes
        -----
        Returned forward (reverse) rays will be expressed in the local
        coordinate system of the first (last) element of the CompoundOptic.
        See `Ray.toCoordSys` or `RayVector.toCoordSys` to express rays in a
        different coordinate system.
        """
        if _verbose:
            strtemplate = ("traceSplitReverse {:15s} "
                           "flux = {:18.8f}   nphot = {:10d}")
            print(strtemplate.format(self.name, np.sum(r.flux), len(r)))
        if self.skip:
            return r, None

        workQueue = [(r, "reverse", len(self.items)-1)]

        outRForward = []
        outRReverse = []

        while workQueue:
            rays, direction, itemIndex = workQueue.pop()
            item = self.items[itemIndex]
            if direction == "forward":
                rForward, rReverse = item.traceSplit(
                    rays, minFlux=minFlux, _verbose=_verbose
                )
            elif direction == "reverse":
                rForward, rReverse = item.traceSplitReverse(
                    rays, minFlux=minFlux, _verbose=_verbose
                )
            else:
                raise RuntimeError("Shouldn't get here!")

            rForward.trimVignettedInPlace(minFlux)
            rReverse.trimVignettedInPlace(minFlux)

            if len(rReverse) > 0:
                if itemIndex == 0:
                    outRReverse.append(rReverse)
                else:
                    workQueue.append((rReverse, "reverse", itemIndex-1))

            if len(rForward) > 0:
                if itemIndex == len(self.items)-1:
                    outRForward.append(rForward)
                else:
                    workQueue.append((rForward, "forward", itemIndex+1))

        rForward = concatenateRayVectors(outRForward)
        rReverse = concatenateRayVectors(outRReverse)
        return rForward, rReverse

    def draw3d(self, ax, **kwargs):
        """Recursively draw this `CompoundOptic` on a mplot3d axis by drawing
        all subitems.

        Parameters
        ----------
        ax : mplot3d.Axis
            Axis on which to draw this optic.
        """
        for item in self.items:
            item.draw3d(ax, **kwargs)

    def draw2d(self, ax, **kwargs):
        """Draw a 2D slice of this `CompoundOptic` in the (x,z) plane.

        Calls draw2d recursively on each of our items, with the actual
        drawing taking place in Interface and (optionally) Lens instances.

        The kwargs are passed to the drawing commands, except for the
        optional keyword 'only' which restricts drawing to only instances
        that are subclasses of a specified type or types.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis on which to draw this optic.
        only : list
            A list of types to draw, e.g., `batoid.Mirror`, or `batoid.Lens`.
            Default: None means to draw all types.
        """
        only = kwargs.pop('only', None)
        for item in self.items:
            item_class = item.__class__
            if issubclass(item_class, CompoundOptic):
                item.draw2d(ax, only=only, **kwargs)
            elif only is None or issubclass(item_class, only):
                item.draw2d(ax, **kwargs)

    def clearObscuration(self, unless=()):
        for item in self.items:
            item.clearObscuration(unless=unless)

    def __eq__(self, other):
        if not self.__class__ == other.__class__:
            return False
        return (self.items == other.items and
                self.name == other.name and
                self.inMedium == other.inMedium and
                self.outMedium == other.outMedium and
                self.coordSys == other.coordSys)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        out = "{!s}([".format(self.__class__.__name__)
        for item in self.items[:-1]:
            out += "{!r}, ".format(item)
        out += "{!r}]".format(self.items[-1])
        out += Optic._repr_helper(self)
        out += ")"
        return out

    def __hash__(self):
        return hash((self.__class__.__name__, self.items,
                     self.name, self.inMedium, self.outMedium, self.coordSys))

    def withGlobalShift(self, shift):
        """Return a new `CompoundOptic` with its coordinate system shifted (and
        the coordinate systems of all subitems)

        Parameters
        ----------
        shift : array (3,)
            The coordinate shift, relative to the global coordinate system, to
            apply to self.coordSys

        Returns
        -------
        `CompoundOptic`
            Shifted optic.
        """
        newItems = [item.withGlobalShift(shift) for item in self.items]
        ret = self.__class__.__new__(self.__class__)

        newDict = dict(self.__dict__)
        newDict['coordSys'] = self.coordSys.shiftGlobal(shift)
        del newDict['items']
        ret.__init__(
            newItems,
            **newDict
        )
        return ret

    def withGloballyShiftedOptic(self, name, shift):
        """Return a new `CompoundOptic` with the coordinate system of one of
        its subitems shifted.

        Parameters
        ----------
        name : str
            The subitem to shift.
        shift : array (3,)
            The coordinate shift, relative to the global coordinate system, to
            apply to the subitem's coordSys.

        Returns
        -------
        `batoid.CompoundOptic`
            Optic with shifted subitem.
        """
        # If name is one of items.names, the we use withGlobalShift, and we're
        # done.  If not, then we need to recurse down to whichever item
        # contains name.  Verify that name is in self.itemDict, but first
        # convert partially qualified name to fully qualified.
        if name in self._names:
            name = self._names[name]
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        if name == self.name:
            return self.withGlobalShift(shift)
        # Clip off leading token
        assert name[:len(self.name)+1] == \
            self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
        name = name[len(self.name)+1:]
        newItems = []
        newDict = dict(self.__dict__)
        del newDict['items']
        for i, item in enumerate(self.items):
            if name.startswith(item.name):
                if name == item.name:
                    newItems.append(item.withGlobalShift(shift))
                else:
                    newItems.append(item.withGloballyShiftedOptic(name, shift))
                newItems.extend(self.items[i+1:])
                return self.__class__(
                    newItems,
                    **newDict
                )
            newItems.append(item)
        raise RuntimeError(
            "Error in withGloballyShiftedOptic!, Shouldn't get here!"
        )

    def withLocalRotation(self, rot, rotOrigin=None, coordSys=None):
        """Return a new `CompoundOptic` with its coordinate system rotated.

        Parameters
        ----------
        rot : array (3,3)
            Rotation matrix wrt to the local coordinate system to apply.
        rotOrigin : array (3,)
            Origin of rotation.  Default: None means use [0,0,0]
        coordSys : `batoid.CoordSys`
            Coordinate system of rotOrigin above.  Default: None means use
            self.coordSys.

        Returns
        -------
        `batoid.CompoundOptic`
            Rotated optic.
        """
        if rotOrigin is None and coordSys is None:
            coordSys = self.coordSys
            rotOrigin = [0,0,0]
        newItems = [item.withLocalRotation(rot, rotOrigin, coordSys)
                    for item in self.items]
        ret = self.__class__.__new__(self.__class__)
        newDict = dict(self.__dict__)
        newDict['coordSys'] = self.coordSys.rotateLocal(
            rot, rotOrigin, coordSys
        )
        del newDict['items']
        ret.__init__(
            newItems,
            **newDict
        )
        return ret

    def withLocallyRotatedOptic(self, name, rot, rotOrigin=None,
                                coordSys=None):
        """Return a new `CompoundOptic` with the coordinate system of one of
        its subitems rotated.

        Parameters
        ----------
        name : str
            The subitem to rotate.
        rot : array (3,3)
            Rotation matrix wrt to the subitem's local coordinate system to
            apply.
        rotOrigin : array (3,)
            Origin of rotation.  Default: None means use [0,0,0]
        coordSys : `batoid.CoordSys`
            Coordinate system of rotOrigin above.  Default: None means use the
            coordinate system of the subitem being rotated.

        Returns
        -------
        `batoid.CompoundOptic`
            Optic with rotated subitem.
        """
        # If name is one of items.names, the we use withLocalRotation, and
        # we're done.  If not, then we need to recurse down to whichever item
        # contains name.  Verify that name is in self.itemDict, but first
        # convert partially qualified name to fully qualified.
        if name in self._names:
            name = self._names[name]
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        if name == self.name:
            return self.withLocalRotation(rot, rotOrigin, coordSys)
        if rotOrigin is None and coordSys is None:
            coordSys = self.itemDict[name].coordSys
            rotOrigin = [0,0,0]
        # Clip off leading token
        assert name[:len(self.name)+1] == \
            self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
        name = name[len(self.name)+1:]
        newItems = []
        newDict = dict(self.__dict__)
        del newDict['items']
        for i, item in enumerate(self.items):
            if name.startswith(item.name):
                if name == item.name:
                    newItems.append(item.withLocalRotation(
                        rot, rotOrigin, coordSys
                    ))
                else:
                    newItems.append(item.withLocallyRotatedOptic(
                        name, rot, rotOrigin, coordSys
                    ))
                newItems.extend(self.items[i+1:])
                return self.__class__(
                    newItems,
                    **newDict
                )
            newItems.append(item)
        raise RuntimeError(
            "Error in withLocallyRotatedOptic!, Shouldn't get here!"
        )

    def withSurface(self, name, surface):
        """Return a new `CompoundOptic` with one of its subitem's surfaces
        attribute replaced.

        Parameters
        ----------
        name : str
            Which subitem's surface to replace.
        surface : `batoid.Surface`
            New replacement surface.

        Returns
        -------
        `batoid.CompoundOptic`
            Optic with new surface.
        """
        if name in self._names:
            name = self._names[name]
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        # name is fully qualified, so clip off leading token
        assert name[:len(self.name)+1] == \
            self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
        name = name[len(self.name)+1:]
        newItems = []
        newDict = dict(self.__dict__)
        del newDict['items']
        for i, item in enumerate(self.items):
            if name.startswith(item.name):
                if name == item.name:
                    newItems.append(item.withSurface(surface))
                else:
                    newItems.append(item.withSurface(name, surface))
                newItems.extend(self.items[i+1:])
                return self.__class__(
                    newItems,
                    **newDict
                )
            newItems.append(item)
        raise RuntimeError("Error in withSurface.  Shouldn't get here!")


class Lens(CompoundOptic):
    """An lens with two `Interface` s as front and back surfaces.

    Parameters
    ----------
    items : list of `batoid.Optic`, len (2,)
        Subitems making up this compound optic.
    medium : `batoid.Medium`
        The refractive index medium internal to the lens.
    **kwargs :
        Other parameters to forward to Optic.__init__
    """
    def __init__(self, items, medium, **kwargs):
        Optic.__init__(self, **kwargs)
        self.items = tuple(items)
        self.medium = medium

    def __eq__(self, other):
        if not CompoundOptic.__eq__(self, other):
            return False
        return self.medium == other.medium

    def __repr__(self):
        out = ("{!s}([{!r}, {!r}], {!r}".format(
            self.__class__.__name__, self.items[0], self.items[1], self.medium
        ))
        out += Optic._repr_helper(self)
        out += ")"
        return out

    def __hash__(self):
        return hash((self.medium, CompoundOptic.__hash__(self)))

    def draw2d(self, ax, **kwargs):
        """Specialized draw2d for `Lens` instances.

        If the optional keyword 'only' equals `batoid.Lens`, then fill the area
        between the lens refractive interfaces using the remaining specified
        kwargs (fc, ec, alpha, ...)

        Otherwise, call draw2d on each of our refractive interfaces.

        The optional labelpos and fontdict kwargs are used to draw a label at
        the specified x position (in global coords), using the specified font
        properties.
        """
        only = kwargs.pop('only', None)
        if only == Lens:
            labelpos = kwargs.pop('labelpos', None)
            fontdict = kwargs.pop('fontdict', None)
            if len(self.items) != 2:
                raise RuntimeError(
                    'Cannot draw lens "{0}" with {1} surfaces.'.format(
                        self.name, len(self.items)
                    )
                )
            # Calculate the global coordinates of slices through our two
            # interfaces.
            slice1 = self.items[0].getXZSlice()
            slice2 = self.items[1].getXZSlice()
            # Fill the area between these slices.
            all_z = []
            for (x1, z1), (x2, z2) in zip(slice1, slice2):
                x = np.hstack((x1, x2[::-1], x1[:1]))
                z = np.hstack((z1, z2[::-1], z1[:1]))
                all_z.append(z)
                ax.fill(x, z, **kwargs)
            # Draw an optional label for this lens.
            if labelpos is not None:
                zlabel = np.mean(all_z)
                ax.text(
                    labelpos, zlabel, self.name, fontdict=fontdict,
                    horizontalalignment='center', verticalalignment='center')
        else:
            super(Lens, self).draw2d(ax, only=only, **kwargs)

    def withGlobalShift(self, shift):
        """Return a new `Lens` with its coordinate system shifted (and the
        coordinate systems of all subitems)

        Parameters
        ----------
        shift : array (3,)
            The coordinate shift, relative to the global coordinate system, to
            apply to self.coordSys

        Returns
        -------
        `Lens`
            Shifted optic.
        """
        newItems = [item.withGlobalShift(shift) for item in self.items]
        ret = self.__class__.__new__(self.__class__)
        newDict = dict(self.__dict__)
        newDict['coordSys'] = self.coordSys.shiftGlobal(shift)
        del newDict['items']
        del newDict['medium']
        ret.__init__(
            newItems, self.medium,
            **newDict
        )
        return ret

    def withGloballyShiftedOptic(self, name, shift):
        """Return a new `Lens` with the coordinate system of one of its
        subitems shifted.

        Parameters
        ----------
        name : str
            The subitem to shift.
        shift : array (3,)
            The coordinate shift, relative to the global coordinate system, to
            apply to the subitem's coordSys.

        Returns
        -------
        `batoid.Lens`
            Lens with shifted surface.
        """
        # If name is one of items.names, the we use withGlobalShift, and we're
        # done.  If not, then we need to recurse down to whicever item contains
        # name.  First verify that name is in self.itemDict
        if name in self._names:
            name = self._names[name]
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        if name == self.name:
            return self.withGlobalShift(shift)
        # Clip off leading token
        assert name[:len(self.name)+1] == \
            self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
        name = name[len(self.name)+1:]
        newItems = []
        newDict = dict(self.__dict__)
        del newDict['items']
        del newDict['medium']
        for i, item in enumerate(self.items):
            if name.startswith(item.name):
                if name == item.name:
                    newItems.append(item.withGlobalShift(shift))
                else:
                    newItems.append(item.withGloballyShiftedOptic(name, shift))
                newItems.extend(self.items[i+1:])
                return self.__class__(
                    newItems, self.medium,
                    **newDict
                )
            newItems.append(item)
        raise RuntimeError(
            "Error in withGloballyShiftedOptic!, Shouldn't get here!"
        )

    def withLocalRotation(self, rot, rotOrigin=None, coordSys=None):
        """Return a new `Lens` with its coordinate system rotated.

        Parameters
        ----------
        rot : array (3,3)
            Rotation matrix wrt to the local coordinate system to apply.
        rotOrigin : array (3,)
            Origin of rotation.  Default: None means use [0,0,0]
        coordSys : `batoid.CoordSys`
            Coordinate system of rotOrigin above.  Default: None means use
            self.coordSys.

        Returns
        -------
        `batoid.Lens`
            Rotated lens.
        """
        if rotOrigin is None and coordSys is None:
            coordSys = self.coordSys
            rotOrigin = [0,0,0]
        newItems = [item.withLocalRotation(rot, rotOrigin, coordSys)
                    for item in self.items]
        newDict = dict(self.__dict__)
        del newDict['items']
        del newDict['medium']
        ret = self.__class__.__new__(self.__class__)
        ret.__init__(
            newItems, self.medium,
            **newDict
        )
        return ret

    def withLocallyRotatedOptic(self, name, rot, rotOrigin=None, coordSys=None):
        """Return a new `Lens` with the coordinate system of one of its
        surfaces rotated.

        Parameters
        ----------
        name : str
            The subitem to rotate.
        rot : array (3,3)
            Rotation matrix wrt to the subitem's local coordinate system to
            apply.
        rotOrigin : array (3,)
            Origin of rotation.  Default: None means use [0,0,0]
        coordSys : `batoid.CoordSys`
            Coordinate system of rotOrigin above.  Default: None means use the
            coordinate system of the subitem being rotated.

        Returns
        -------
        `batoid.Lens`
            Lens with rotated surface.
        """
        # If name is one of items.names, the we use withLocalRotation, and
        # we're done.  If not, then we need to recurse down to whichever item
        # contains name.  First verify that name is in self.itemDict
        if name in self._names:
            name = self._names[name]
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        if name == self.name:
            return self.withLocalRotation(rot, rotOrigin, coordSys)
        if rotOrigin is None and coordSys is None:
            coordSys = self.itemDict[name].coordSys
            rotOrigin = [0,0,0]
        # Clip off leading token
        assert name[:len(self.name)+1] == \
            self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
        name = name[len(self.name)+1:]
        newItems = []
        newDict = dict(self.__dict__)
        del newDict['items']
        del newDict['medium']
        for i, item in enumerate(self.items):
            if name.startswith(item.name):
                if name == item.name:
                    newItems.append(item.withLocalRotation(
                        rot, rotOrigin, coordSys
                    ))
                else:
                    newItems.append(item.withLocallyRotatedOptic(
                        name, rot, rotOrigin, coordSys
                    ))
                newItems.extend(self.items[i+1:])
                return self.__class__(
                    newItems, self.medium,
                    **newDict
                )
            newItems.append(item)
        raise RuntimeError(
            "Error in withLocallyRotatedOptic!, Shouldn't get here!"
        )

    def withSurface(self, name, surface):
        """Return a new `Lens` with one of its surfaces replaced.

        Parameters
        ----------
        name : str
            Which surface to replace.
        surface : `batoid.Surface`
            New replacement surface.

        Returns
        -------
        `batoid.Lens`
            Lens with new surface.
        """
        if name in self._names:
            name = self._names[name]
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        # name is fully qualified, so clip off leading token
        assert name[:len(self.name)+1] == \
            self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
        name = name[len(self.name)+1:]
        newItems = []
        newDict = dict(self.__dict__)
        del newDict['items']
        for i, item in enumerate(self.items):
            if name.startswith(item.name):
                if name == item.name:
                    newItems.append(item.withSurface(surface))
                else:
                    newItems.append(item.withSurface(name, surface))
                newItems.extend(self.items[i+1:])
                return self.__class__(
                    newItems,
                    **newDict
                )
            newItems.append(item)
        raise RuntimeError("Error in withSurface.  Shouldn't get here!")


def getGlobalRays(traceFull, start=None, end=None, globalSys=globalCoordSys):
    """Calculate an array of ray vertices in global coordinates.

    Parameters
    ----------
    traceFull : OrderedDict
        Array of per-surface ray-tracing output from traceFull()
    start : str or None
        Name of the first surface to include in the output, or use the first
        surface in the model when None.
    end : str or None
        Name of the last surface to include in the output, or use the last
        surface in the model when None.
    globalSys : `batoid.CoordSys`
        Global coordinate system to use.

    Returns
    -------
    tuple
        Tuple (xyz, raylen) of arrays with shapes (nray, 3, nsurf + 1) and
        (nray,).  The xyz array contains the global coordinates of each
        ray vertex, with raylen giving the number of visible (not vignetted)
        vertices for each ray.
    """
    names = [trace['name'] for trace in traceFull.values()]
    if start is None:
        start = names[0]
    if end is None:
        end = names[-1]
    try:
        istart = names.index(start)
    except ValueError:
        raise ValueError('No such start surface "{0}".'.format(start))
    try:
        iend = names.index(end)
    except ValueError:
        raise ValueError('No such end surface "{0}".'.format(end))
    nsurf = iend - istart + 1
    if nsurf <= 0:
        raise ValueError('Expected start < end.')
    nray = len(traceFull[start]['in'])
    # Allocate an array for all ray vertices in global coords.
    xyz = np.empty((nray, 3, nsurf + 1))
    # First point on each ray is where it enters the start surface.
    transform = CoordTransform(traceFull[start]['in'].coordSys, globalSys)
    xyz[:, :, 0] = np.stack(
        transform.applyForward(*traceFull[start]['in'].r.T), axis=1
    )
    # Keep track of the number of visible points on each ray.
    raylen = np.ones(nray, dtype=int)
    for i, name in enumerate(names[istart:iend+1]):
        surface = traceFull[name]
        # Add a point for where each ray leaves this surface.
        transform = CoordTransform(surface['out'].coordSys, globalSys)
        xyz[:, :, i + 1] = np.stack(
            transform.applyForward(*surface['out'].r.T), axis=1
        )
        # Keep track of rays which are still visible.
        visible = ~surface['out'].vignetted
        raylen[visible] += 1
    return xyz, raylen


def drawTrace3d(ax, traceFull, start=None, end=None, **kwargs):
    """Draw 3D rays in global coordinates on the specified axis.

    Parameters
    ----------
    ax : mplot3d.Axis
        Axis on which to draw rays.
    traceFull : OrderedDict
        Array of per-surface ray-tracing output from traceFull()
    start : str or None
        Name of the first surface to include in the output, or use the first
        surface in the model when None.
    end : str or None
        Name of the last surface to include in the output, or use the last
        surface in the model when None.
    globalSys : `batoid.CoordSys`
        Global coordinate system to use.
    """
    xyz, raylen = getGlobalRays(traceFull, start, end)
    lines = []
    for line, nline in zip(xyz, raylen):
        ax.plot(line[0, :nline], line[1, :nline], line[2, :nline], **kwargs)


def drawTrace2d(ax, traceFull, start=None, end=None, **kwargs):
    """Draw 2D rays in global coordinates on the specified axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw rays.
    traceFull : OrderedDict
        Array of per-surface ray-tracing output from traceFull()
    start : str or None
        Name of the first surface to include in the output, or use the first
        surface in the model when None.
    end : str or None
        Name of the last surface to include in the output, or use the last
        surface in the model when None.
    globalSys : `batoid.CoordSys`
        Global coordinate system to use.
    """
    xyz, raylen = getGlobalRays(traceFull, start, end)
    lines = []
    for line, nline in zip(xyz, raylen):
        lines.extend([line[0, :nline], line[2, :nline]])
    ax.plot(*lines, **kwargs)
