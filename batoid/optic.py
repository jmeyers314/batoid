from collections import OrderedDict

import numpy as np

from .coating import SimpleCoating
from .obscuration import ObscNegation, ObscCircle, ObscAnnulus
from .constants import globalCoordSys, vacuum
from .coordTransform import CoordTransform
from .utils import lazy_property
from .rayVector import RayVector


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
        self.path = [self.name]
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
            x, y, z = transform.applyForwardArray(x, y, z)
            ax.plot(x, y, z, **kwargs)

        #outer circle
        th = np.linspace(0, 2*np.pi, 100)
        cth, sth = np.cos(th), np.sin(th)
        x = self.outRadius * cth
        y = self.outRadius * sth
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForwardArray(x, y, z)
        ax.plot(x, y, z, **kwargs)

        #next, a line at X=0
        y = np.linspace(-self.outRadius, -self.inRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForwardArray(x, y, z)
        ax.plot(x, y, z, **kwargs)
        y = np.linspace(self.inRadius, self.outRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForwardArray(x, y, z)
        ax.plot(x, y, z, **kwargs)

        #next, a line at Y=0
        x = np.linspace(-self.outRadius, -self.inRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForwardArray(x, y, z)
        ax.plot(x, y, z, **kwargs)
        x = np.linspace(self.inRadius, self.outRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        x, y, z = transform.applyForwardArray(x, y, z)
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
        xneg, yneg, zneg = transform.applyForwardArray(x, y, z)
        if np.any(yneg != 0):
            print('WARNING: getXZSlice used for rotated surface "{0}".'
                .format(self.name)
            )
        # Calculate (x,z) slice in local coordinates for x >= 0.
        x *= -1
        x = x[::-1]
        z[:] = self.surface.sag(x, y)
        # Transform slice to global coordinates.
        xpos, ypos, zpos = transform.applyForwardArray(x, y, z)
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

    def trace(self, rv, reverse=False):
        """Trace ray through this optical element.

        Parameters
        ----------
        rv : `batoid.RayVector`
            Input rays to trace, transforming in place.
        reverse : bool
            Trace through optical element in reverse?  Default: False

        Returns
        -------
        `batoid.RayVector`
            Reference to transformed input rays.

        Notes
        -----
        This operation is performed in place; the return value is a reference to
        the transformed input `RayVector`.

        The transformed rays will be expressed in the local coordinate system of
        the Optic.  See `RayVector.toCoordSys` to express rays in a different
        coordinate system.

        Also, you may need to reverse the directions of rays if using this
        method with ``reverse=True``.
        """
        # refract, reflect, pass-through - depending on subclass
        self.interact(rv, reverse=reverse)

        if self.obscuration is not None:
            self.obscuration.obscure(rv)

        return rv

    def traceFull(self, rv, reverse=False):
        """Trace rays through this optical element, returning a full history of
        all surface intersections.

        Parameters
        ----------
        rv : `batoid.RayVector`
            Input rays to trace
        reverse : bool
            Trace through optical element in reverse?  Default: False

        Returns
        -------
        OrderedDict of dict
            There will be one key-value pair for every Interface traced
            through (which for this class, is just a single Interface).  The
            values will be dicts with key-value pairs:

            ``'name'``
                name of Interface (str)
            ``'in'``
                the incoming rays to that Interface (RayVector)
            ``'out'``
                the outgoing rays from that Interface (RayVector)

        Notes
        -----
        Pay careful attention to the coordinate systems of the returned rays.
        These will generally differ from the original input coordinate system.
        To transform to another coordinate system, see `RayVector.toCoordSys`.
        """
        result = OrderedDict()
        if not self.skip:
            result[self.name] = {
                'name':self.name,
                'in':rv,
                'out':self.trace(rv.copy(), reverse=reverse)
            }
        return result

    def traceSplit(self, rv, minFlux=1e-3, reverse=False, _verbose=False):
        """Trace rays through this optical element, splitting the return values
        into rays that continue propagating in the "forward" direction, and
        those that were reflected into the "reverse" direction.  Fluxes of
        output rays are proportional to reflection/transmission coefficients of
        the interface (which may depend on wavelength and incidence angle).

        Parameters
        ----------
        rv : batoid.RayVector
            Input rays to trace
        minFlux : float
            Minimum flux value of rays to continue propagating.
            Default: 1e-3.
        reverse : bool
            Trace through optic in reverse?  Default: False.

        Returns
        -------
        forwardRays : list of batoid.RayVector.
            Each item in list comes from one distinct path through the optic
            exiting in the forward direction.  The exact path traversed is
            accessible from the ``.path`` attribute of the item.
        reverseRays : list of batoid.RayVector.
            Each item in list comes from one distinct path through the optic
            exiting in the reverse direction.  The exact path traversed is
            accessible from the ``.path`` attribute of the item.

        Notes
        -----
        Returned rays will be expressed in the local coordinate system of the
        Optic.  See `RayVector.toCoordSys` to express rays in a different
        coordinate system.
        """
        if _verbose:
            s = "forward" if not reverse else "reverse"
            strtemplate = ("traceSplit {}       {:30s} "
                           "flux = {:18.8f}   nphot = {:10d}")
            print(strtemplate.format(s, self.name, np.sum(rv.flux), len(rv)))
        if self.skip:
            if reverse:
                return None, rv
            else:
                return rv, None
        refracted, reflected = self.rSplit(rv, reverse=reverse)

        # For now, apply obscuration equally forwards and backwards
        if self.obscuration is not None:
            if refracted is not None:
                self.obscuration.obscure(refracted)
            if reflected is not None:
                self.obscuration.obscure(reflected)
        if not hasattr(rv, 'path'):
            if refracted is not None:
                refracted.path = [self.name]
            if reflected is not None:
                reflected.path = [self.name]
        else:
            currentPath = list(rv.path)
            if refracted is not None:
                refracted.path = currentPath+[self.name]
            if reflected is not None:
                reflected.path = currentPath+[self.name]

        rForward, rReverse = [refracted], [reflected]
        if reverse:
            rForward, rReverse = rReverse, rForward
        if isinstance(self, Mirror):
            rForward, rReverse = rReverse, rForward
        return rForward, rReverse

    def clearObscuration(self, unless=()):
        if self.name not in unless:
            self.obscuration = None

    def interact(self, rv, reverse=False):
        # intersect independent of `reverse`
        return self.surface.intersect(rv, coordSys=self.coordSys)

    def __eq__(self, other):
        if not self.__class__ == other.__class__:
            return False
        return (
            self.surface == other.surface and
            self.obscuration == other.obscuration and
            self.name == other.name and
            self.inMedium == other.inMedium and
            self.outMedium == other.outMedium and
            self.coordSys == other.coordSys
        )

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
        `Interface`
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

    def withLocalShift(self, shift):
        """Return a new `Interface` with its coordinate system shifted.

        Parameters
        ----------
        shift : array (3,)
            The coordinate shift, relative to the local coordinate system, to
            apply to self.coordSys

        Returns
        -------
        `Interface`
            Shifted interface.
        """
        # Clearer to apply the rotated global shift.
        return self.withGlobalShift(np.dot(self.coordSys.rot, shift))

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
        `Interface`
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
        `Interface`
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

    def interact(self, rv, reverse=False):
        if reverse:
            m1, m2 = self.outMedium, self.inMedium
        else:
            m1, m2 = self.inMedium, self.outMedium
        return self.surface.refract(rv, m1, m2, coordSys=self.coordSys)

    def rSplit(self, rv, reverse=False):
        # always return in order: refracted, reflected
        if reverse:
            m1 = self.outMedium
            m2 = self.inMedium
            coating = self.forwardCoating
        else:
            m1 = self.inMedium
            m2 = self.outMedium
            coating = self.reverseCoating

        return self.surface.rSplit(rv, m1, m2, coating, coordSys=self.coordSys)


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

    def interact(self, rv, reverse=False):
        # reflect is independent of reverse
        return self.surface.reflect(rv, coordSys=self.coordSys)

    def rSplit(self, rv, reverse=False):
        # reflect is independent of reverse
        return None, self.surface.reflect(rv, coordSys=self.coordSys)


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

    def rSplit(self, rv, reverse=False):
        assert reverse == False
        return self.surface.rSplit(
            rv, self.inMedium, self.outMedium, self.forwardCoating,
            coordSys=self.coordSys
        )


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

    def rSplit(self, rv, reverse=False):
        return self.surface.intersect(rv, self.coordSys), None


class OPDScreen(Interface):
    """An `Optic` defined by a surface and an Optical Path Difference function.

    Notes
    -----
    We use a `batoid.Surface` object for both the intersection surface and the
    OPD.  For the OPD, the `surface.sag` method should return the desired
    optical path difference (in meters) to add to traversing rays.

    Rays traversing an OPD screen are deflected according to:

    cosx2 = cosx1 + d(OPD)/dx
    cosy2 = cosy1 + d(OPD)/dy

    Where cosx1, cosx2 are direction cosines relative to the surface normal.

    Parameters
    ----------
    surface : `batoid.Surface`
        The intersection surface for the phase screen.
    screen : `batoid.Surface`
        The OPD to add upon traversal, in meters.
    **kwargs : dict
        Other keyword arguments to `Interface`.
    """
    def __init__(self, surface, screen, **kwargs):
        Interface.__init__(self, surface, **kwargs)
        self.screen = screen
        assert self.inMedium == self.outMedium
        # Not coatings available.

    def __repr__(self):
        out = "OPDScreen({!r}, {!r}".format(self.surface, self.screen)
        if self.obscuration is not None:
            out += ", obscuration={!r}".format(self.obscuration)
        out += Optic._repr_helper(self)
        out +=")"
        return out

    def __hash__(self):
        return hash((
            Interface.__hash__(self),
            self.screen
        ))

    def __eq__(self, rhs):
        if Interface.__eq__(self, rhs):
            return self.screen == rhs.screen
        return False

    def interact(self, rv, reverse=False):
        # Should reverse be different somehow?
        return self.surface.refractScreen(
            rv,
            self.screen,
            coordSys=self.coordSys
        )

    def rSplit(self, rv, reverse=False):
        # Can only go forward for now.
        return self.surface.interact(rv), None


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
        self.path = [name for item in self.items for name in item.path]

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
        for k in self.itemDict.keys():
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

    def trace(self, rv, reverse=False, path=None):
        """Recursively trace through all subitems of this `CompoundOptic`.

        Parameters
        ----------
        rv : `batoid.RayVector`
            Input rays to trace, transforming in place.
        reverse : bool
            Trace through optical element in reverse?  Default: False
        path : list of names of Interfaces.
            Trace through the optical system in this order, as opposed to the
            natural order.  Useful for investigating particular ghost images.
            Note that items included in path will not be skipped, even if their
            skip attribute is true.

        Returns
        -------
        `batoid.RayVector`
            Reference to transformed input rays.

        Notes
        -----
        This operation is performed in place; the return value is a reference to
        the transformed input `RayVector`.

        Returned rays will be expressed in the local coordinate system of the
        last element of the CompoundOptic.  See `RayVector.toCoordSys` to
        express rays in a different coordinate system.

        Also, you may need to reverse the directions of rays if using this
        method with ``reverse=True``.
        """
        if path is None:
            items = self.items if not reverse else reversed(self.items)
            for item in items:
                if not item.skip:
                    item.trace(rv, reverse=reverse)
        else:
            # establish nominal order of elements by building dict
            # of name -> order
            i = 0
            nominalOrder = {}
            for name in path:
                if name not in nominalOrder.keys():
                    nominalOrder[name] = i
                    i += 1
            direction = "forward"
            for i in range(len(path)):
            # for i in range(len(path)-1):
                currentName = path[i]
                item = self[currentName]
                # logic to decide when to reverse direction
                if i == len(path)-1:
                    if nominalOrder[path[i]] == 0:
                        nextDirection = "reverse"
                    else:
                        nextDirection = direction
                else:
                    nextName = path[i+1]
                    if nominalOrder[nextName] < nominalOrder[currentName]:
                        nextDirection = "reverse"
                    else:
                        nextDirection = "forward"
                if direction == nextDirection:
                    item.trace(rv, reverse=(direction=="reverse"))
                else:
                    direction = nextDirection
                    item.surface.reflect(rv, coordSys=item.coordSys)
                    if item.obscuration:
                        item.obscuration.obscure(rv)
        return rv

    def traceFull(self, rv, reverse=False, path=None):
        """Recursively trace rays through this `CompoundOptic`, returning a full
        history of all surface intersections.

        Parameters
        ----------
        rv : `batoid.RayVector`
            Input rays to trace
        reverse : bool
            Trace through optical element in reverse?  Default: False
        path : list of names of Interfaces.
            Trace through the optical system in this order, as opposed to the
            natural order.  Useful for investigating particular ghost images.
            Note that items included in path will not be skipped, even if their
            skip attribute is true.

        Returns
        -------
        OrderedDict of dict
            There will be one key-value pair for every Interface traced
            through.  The values will be dicts with key-value pairs:

            ``'name'``
                name of Interface (str)
            ``'in'``
                the incoming rays to that Interface (RayVector)
            ``'out'``
                the outgoing rays from that Interface (RayVector)
        Notes
        -----
        Pay careful attention to the coordinate systems of the returned rays.
        These will generally differ from the original input coordinate system.
        To transform to another coordinate system, see `RayVector.toCoordSys`.
        """
        result = OrderedDict()
        if path is None:
            if not self.skip:
                rv_in = rv
                items = self.items if not reverse else reversed(self.items)
                for item in items:
                    tf = item.traceFull(rv_in, reverse=reverse)
                    for k, v in tf.items():
                        result[k] = v
                        rv_in = v['out']
        else:
            # establish nominal order of elements by building dict
            # of name -> order
            i = 0
            nominalOrder = {}
            for name in path:
                if name not in nominalOrder.keys():
                    nominalOrder[name] = i
                    i += 1
            direction = "forward"
            rv_in = rv
            # Do the actual tracing here.
            for i in range(len(path)):
                currentName = path[i]
                item = self[currentName]
                # logic to decide when to reverse direction
                if i == len(path)-1:
                    if nominalOrder[path[i]] == 0:
                        nextDirection = "reverse"
                    else:
                        nextDirection = direction
                else:
                    nextName = path[i+1]
                    if nominalOrder[nextName] < nominalOrder[currentName]:
                        nextDirection = "reverse"
                    else:
                        nextDirection = "forward"
                # trace
                if direction == nextDirection:
                    rv_out = item.trace(
                        rv_in.copy(),
                        reverse=(direction=="reverse")
                    )
                else:
                    direction = nextDirection
                    rv_out = item.surface.reflect(
                        rv_in.copy(), coordSys=item.coordSys
                    )
                    if item.obscuration:
                        item.obscuration.obscure(rv_out)
                # determine output key
                key = item.name+'_0'
                j = 1
                while key in result:
                    key = item.name+'_{}'.format(j)
                    j += 1
                # output
                result[key] = {
                    'name':item.name,
                    'in':rv_in.copy(),
                    'out':rv_out.copy()
                }
                rv_in = rv_out
        return result

    def traceSplit(self, rv, minFlux=1e-3, reverse=False, _verbose=False):
        """Recursively trace rays through this `CompoundOptic`, splitting at
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
        rv : `batoid.RayVector`
            Input rays to trace
        minFlux : float
            Minimum flux value of rays to continue propagating.
            Default: 1e-3.
        reverse : bool
            Trace through optic in reverse?  Default: False.

        Returns
        -------
        forwardRays : list of batoid.RayVector.
            Each item in list comes from one distinct path through the optic
            exiting in the forward direction.  The exact path traversed is
            accessible from the ``.path`` attribute of the item.
        reverseRays : list of batoid.RayVector.
            Each item in list comes from one distinct path through the optic
            exiting in the reverse direction.  The exact path traversed is
            accessible from the ``.path`` attribute of the item.

        Notes
        -----
        Returned forward (reverse) rays will be expressed in the local
        coordinate system of the last (first) element of the CompoundOptic.
        See `RayVector.toCoordSys to express rays in a different coordinate system.
        """
        if _verbose:
            s = "reverse" if reverse else "forward"
            strtemplate = ("traceSplit {}       {:30s} "
                           "flux = {:18.8f}   nphot = {:10d}")
            print(strtemplate.format(s, self.name, np.sum(rv.flux), len(rv)))
        if self.skip:
            if reverse:
                return None, rv
            else:
                return rv, None

        if not reverse:
            workQueue = [(rv, "forward", 0)]
        else:
            workQueue = [(rv, "reverse", len(self.items)-1)]

        outRForward = []
        outRReverse = []

        while workQueue:
            rays, direction, opticIndex = workQueue.pop()
            optic = self.items[opticIndex]
            rForward, rReverse = optic.traceSplit(
                rays, minFlux=minFlux, reverse=(direction=="reverse"),
                _verbose=_verbose
            )
            #  Clear away any None's returned
            rForward = [i for i in rForward if i]
            rReverse = [i for i in rReverse if i]

            # Remove vignetted rays and rays with flux below threshold
            for rList in [rForward, rReverse]:
                for i, rr in enumerate(rList):
                    w = ~rr.vignetted & (rr.flux >= minFlux)
                    rList[i] = RayVector(
                        rr.x[w], rr.y[w], rr.z[w],
                        rr.vx[w], rr.vy[w], rr.vz[w],
                        rr.t[w], rr.wavelength[w], rr.flux[w],
                        rr.vignetted[w], rr.failed[w],
                        rr.coordSys
                    )
                    rList[i].path = rr.path

            for rr in rForward:
                if len(rr) > 0:
                    if opticIndex == len(self.items)-1:
                        outRForward.append(rr)
                    else:
                        workQueue.append((rr, "forward", opticIndex+1))
            for rr in rReverse:
                if len(rr) > 0:
                    if opticIndex == 0:
                        outRReverse.append(rr)
                    else:
                        workQueue.append((rr, "reverse", opticIndex-1))
        return outRForward, outRReverse

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
            A list of types to draw, e.g., `Mirror`, or `Lens`.
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

    def withLocalShift(self, shift):
        """Return a new `CompoundOptic` with its coordinate system shifted.

        Parameters
        ----------
        shift : array (3,)
            The coordinate shift, relative to the local coordinate system, to
            apply to self.coordSys

        Returns
        -------
        `CompoundOptic`
            Shifted interface.
        """
        # Clearer to apply the rotated global shift.
        return self.withGlobalShift(np.dot(self.coordSys.rot, shift))

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
        `CompoundOptic`
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

    def withLocallyShiftedOptic(self, name, shift):
        """Return a new `CompoundOptic` with the coordinate system of one of
        its subitems shifted.

        Parameters
        ----------
        name : str
            The subitem to shift.
        shift : array (3,)
            The coordinate shift, relative to the local coordinate system of
            that subitem, to apply to the subitem's coordSys.

        Returns
        -------
        `CompoundOptic`
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
            return self.withLocalShift(shift)
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
                    newItems.append(item.withLocalShift(shift))
                else:
                    newItems.append(item.withLocallyShiftedOptic(name, shift))
                newItems.extend(self.items[i+1:])
                return self.__class__(
                    newItems,
                    **newDict
                )
            newItems.append(item)
        raise RuntimeError(
            "Error in withLocallyShiftedOptic!, Shouldn't get here!"
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
        `CompoundOptic`
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
        `CompoundOptic`
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
        `CompoundOptic`
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
    **kwargs :
        Other parameters to forward to CompoundOptic.__init__
    """
    def __init__(self, items, **kwargs):
        CompoundOptic.__init__(self, items, **kwargs)
        assert len(items) == 2
        assert items[0].outMedium == items[1].inMedium

    def draw2d(self, ax, **kwargs):
        """Specialized draw2d for `Lens` instances.

        If the optional keyword 'only' equals `Lens`, then fill the area
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
    names = list(traceFull.keys())
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
        transform.applyForwardArray(*traceFull[start]['in'].r.T), axis=1
    )
    # Keep track of the number of visible points on each ray.
    raylen = np.ones(nray, dtype=int)
    for i, name in enumerate(names[istart:iend+1]):
        surface = traceFull[name]
        # Add a point for where each ray leaves this surface.
        transform = CoordTransform(surface['out'].coordSys, globalSys)
        xyz[:, :, i + 1] = np.stack(
            transform.applyForwardArray(*surface['out'].r.T), axis=1
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
