import batoid
import numpy as np


class Optic:
    """This is the most general category of batoid optical system.  It can include
    a single reflective or refractive surface, a lens consisting of two surfaces, or
    an entire telescope including multiple mirrors and/or surfaces.

    Parameters
    ----------
        name : str or None
            An optional name for this optic

        inMedium : batoid.Medium
            Medium in which approaching rays reside

        outMedium : batoid.Medium
            Medium in which rays will leave this optic

        coordSys : batoid.CoordSys
            Coordinate system indicating the position and orientation of this optic's vertex
            with respect to it's parent's coordinate system.  Optics can be nested, i.e., a camera
            Optic may contain several lens Optics, and may itself be part of a larger telescope
            Optic.

        skip : bool
            Whether or not to skip this optic when tracing.

        **kwargs : other
            Other attributes to add as object attributes.
    """
    def __init__(self, name=None, coordSys=batoid.CoordSys(), inMedium=batoid.ConstMedium(1.0),
                 outMedium=batoid.ConstMedium(1.0), skip=False, **kwargs):
        self.name = name
        self.coordSys = coordSys
        self.inMedium = inMedium
        self.outMedium = outMedium
        self.skip = False
        kwargs.pop('_itemDict', None)
        self.__dict__.update(**kwargs)

    def _repr_helper(self):
        out = ""
        if self.name is not None:
            out += ", name={!r}".format(self.name)
        out += ", coordSys={!r}".format(self.coordSys)
        if self.inMedium != batoid.ConstMedium(1.0):
            out += ", inMedium={!r}".format(self.inMedium)
        if self.outMedium != batoid.ConstMedium(1.0):
            out += ", outMedium={!r}".format(self.outMedium)
        return out


class Interface(Optic):
    """The most basic category of Optic representing a single surface.  Almost always one of the
    concrete subclasses should be instantiated, depending on whether rays should reflect, refract,
    or simply pass through this surface.

    Parameters
    ----------
        surface : batoid.Surface
            The surface instance for this Interface.

        obscuration : batoid.Obscuration or None
            Batoid.Obscuration instance indicating which x,y coordinates are obscured/unobscured
            for rays intersecting this optic.

        **kwargs :
            Other parameters to forward to Optic.__init__.
    """
    def __init__(self, surface, obscuration=None, **kwargs):
        Optic.__init__(self, **kwargs)

        self.surface = surface
        self.obscuration = obscuration

        # Stealing inRadius and outRadius from self.obscuration.  These are required for the draw
        # methods.  Only works at the moment if obscuration is a negated circle or negated annulus.
        self.inRadius = 0.0
        self.outRadius = None
        if self.obscuration is not None:
            if isinstance(self.obscuration, batoid.ObscNegation):
                if isinstance(self.obscuration.original, batoid.ObscCircle):
                    self.outRadius = self.obscuration.original.radius
                elif isinstance(self.obscuration.original, batoid.ObscAnnulus):
                    self.outRadius = self.obscuration.original.outer
                    self.inRadius = self.obscuration.original.inner

    def __hash__(self):
        return hash((self.__class__.__name__, self.surface, self.obscuration, self.name,
                     self.inMedium, self.outMedium, self.coordSys))

    def draw3d(self, ax, **kwargs):
        """ Draw this interface on a mplot3d axis.

        Parameters
        ----------

            ax : mplot3d.Axis
                Axis on which to draw this optic.
        """
        if self.outRadius is None:
            return
        # Going to draw 4 objects here: inner circle, outer circle, sag along x=0, sag along y=0
        # inner circle
        if self.inRadius != 0.0:
            th = np.linspace(0, 2*np.pi, 100)
            cth, sth = np.cos(th), np.sin(th)
            x = self.inRadius * cth
            y = self.inRadius * sth
            z = self.surface.sag(x, y)
            transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
            x, y, z = transform.applyForward(x, y, z)
            ax.plot(x, y, z, **kwargs)

        #outer circle
        th = np.linspace(0, 2*np.pi, 100)
        cth, sth = np.cos(th), np.sin(th)
        x = self.outRadius * cth
        y = self.outRadius * sth
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)

        #next, a line at X=0
        y = np.linspace(-self.outRadius, -self.inRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)
        y = np.linspace(self.inRadius, self.outRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)

        #next, a line at Y=0
        x = np.linspace(-self.outRadius, -self.inRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)
        x = np.linspace(self.inRadius, self.outRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, **kwargs)

    def draw2d(self, ax, **kwargs):
        """ Draw this interface on a 2d matplotlib axis.
        May not work if elements are non-circular or not axis-aligned.
        """
        if self.outRadius is None:
            return
        # Drawing in the x-z plane.
        x = np.linspace(-self.outRadius, -self.inRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, z, **kwargs)

        x = np.linspace(self.inRadius, self.outRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, z, **kwargs)

    def trace(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        """ Trace a ray through this optical element.

        Parameters
        ----------

        r : batoid.Ray or batoid.RayVector
            input ray to trace

        inCoordSys : batoid.CoordSys
            Coordinate system of incoming ray(s).  Default: the global coordinate system.

        outCoordSys : batoid.CoordSys
            Coordinate system into which to project the output ray(s).  Default: None,
            which means use the coordinate system of the optic.

        Returns
        -------
            Ray or RayVector, output CoordSys.

        """
        if self.skip:
            return r, inCoordSys
        transform = batoid.CoordTransform(inCoordSys, self.coordSys)
        r = transform.applyForward(r)

        # refract, reflect, pass-through - depending on subclass
        r = self.interact(r)

        if self.obscuration is not None:
            r = self.obscuration.obscure(r)

        if outCoordSys is None:
            return r, self.coordSys
        else:
            transform = batoid.CoordTransform(self.coordSys, outCoordSys)
            return transform.applyForward(r), outCoordSys

    def traceFull(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        if self.skip:
            return []
        result = [{'name':self.name, 'in':r, 'inCoordSys':inCoordSys}]
        r, outCoordSys = self.trace(r, inCoordSys=inCoordSys, outCoordSys=outCoordSys)
        result[0]['out'] = r
        result[0]['outCoordSys'] = outCoordSys
        return result

    def traceInPlace(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        if self.skip:
            return r, inCoordSys
        transform = batoid.CoordTransform(inCoordSys, self.coordSys)
        transform.applyForwardInPlace(r)

        # refract, reflect, pass-through - depending on subclass
        self.interactInPlace(r)

        if self.obscuration is not None:
            self.obscuration.obscureInPlace(r)

        if outCoordSys is None:
            return r, self.coordSys
        else:
            transform = batoid.CoordTransform(self.coordSys, outCoordSys)
            transform.applyForwardInPlace(r)
            return r, outCoordSys

    def traceReverse(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        """Trace through optic(s) in reverse.  Note, you may need to reverse the direction
        of rays for this to work.
        """
        if self.skip:
            return r, inCoordSys
        transform = batoid.CoordTransform(inCoordSys, self.coordSys)
        r = transform.applyForward(r)

        r = self.interactReverse(r)

        if self.obscuration is not None:
            r = self.obscuration.obscure(r)

        if outCoordSys is None:
            return r, self.coordSys
        else:
            transform = batoid.CoordTransform(self.coordSys, outCoordSys)
            return transform.applyForward(r), outCoordSys

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
        if rotOrigin is None and coordSys is None:
            coordSys = self.coordSys
            rotOrigin = [0,0,0]
        ret = self.__class__.__new__(self.__class__)
        newDict = dict(self.__dict__)
        newDict['coordSys'] = self.coordSys.rotateLocal(rot, rotOrigin, coordSys)
        del newDict['surface']
        ret.__init__(
            self.surface,
            **newDict
        )
        return ret


class RefractiveInterface(Interface):
    """Specialization for refractive interfaces.
    """
    def interact(self, r):
        return self.surface.refract(r, self.inMedium, self.outMedium)

    def interactReverse(self, r):
        return self.surface.refract(r, self.outMedium, self.inMedium)

    def interactInPlace(self, r):
        self.surface.refractInPlace(r, self.inMedium, self.outMedium)


class Mirror(Interface):
    """Specialization for reflective interfaces.
    """
    def interact(self, r):
        return self.surface.reflect(r)

    def interactReverse(self, r):
        return self.surface.reflect(r)

    def interactInPlace(self, r):
        self.surface.reflectInPlace(r)


class Detector(Interface):
    """Specialization for detector interfaces.
    """
    def interact(self, r):
        return self.surface.intersect(r)

    def interactReverse(self, r):
        return self.surface.intersect(r)

    def interactInPlace(self, r):
        self.surface.intersectInPlace(r)


class Baffle(Interface):
    """Specialization for baffle interfaces.  Rays will be reported here, but pass through in
    straight lines.
    """
    def interact(self, r):
        return self.surface.intersect(r)

    def interactReverse(self, r):
        return self.surface.intersect(r)

    def interactInPlace(self, r):
        self.surface.intersectInPlace(r)


class CompoundOptic(Optic):
    """A Optic containing two or more Optics as subitems.
    """
    def __init__(self, items, **kwargs):
        Optic.__init__(self, **kwargs)
        self.items = tuple(items)

    @property
    def itemDict(self):
        """A dictionary providing convenient access to the entire hierarchy of this CompoundOptic's
        constituent components.  The key for the first level is just the name of the CompoundOptic,
        e.g., `optic.itemDict['SubaruHSC']`.  The next level is accessed by appending a `.`, e.g.,
        `optic.itemDict['SubaruHSC.HSC']` and so on:
        `optic.itemDict['SubaruHSC.HSC.ADC']`
        `optic.itemDict['SubaruHSC.HSC.ADC.ADC1']`
        `optic.itemDict['SubaruHSC.HSC.ADC.ADC1.ADC1_entrance']`
        """
        if not hasattr(self, '_itemDict'):
            self._itemDict = {}
            self._itemDict[self.name] = self
            for item in self.items:
                self._itemDict[self.name+'.'+item.name] = item
                if hasattr(item, 'itemDict'):
                    for k, v in item.itemDict.items():
                        self._itemDict[self.name+'.'+k] = v
        return self._itemDict

    def trace(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        """ Recursively trace through this Optic by successively tracing through all subitems.
        """
        if self.skip:
            return r, inCoordSys # should maybe make a copy of r here?
        coordSys = inCoordSys
        for item in self.items[:-1]:
            if not item.skip:
                r, coordSys = item.trace(r, inCoordSys=coordSys)
        return self.items[-1].trace(r, inCoordSys=coordSys, outCoordSys=outCoordSys)

    def traceInPlace(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        if self.skip:
            return r, inCoordSys
        coordSys = inCoordSys
        for item in self.items[:-1]:
            r, coordSys = item.traceInPlace(r, inCoordSys=coordSys)
        return self.items[-1].traceInPlace(r, inCoordSys=coordSys, outCoordSys=outCoordSys)

    def traceFull(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        """ Recursively trace through this Optic by successively tracing through all subitems.

        The return value will contain the incoming and outgoing rays for each Interface.
        """
        if self.skip:
            return []
        result = []
        r_in = r
        coordSys = inCoordSys
        for item in self.items[:-1]:
            result.extend(item.traceFull(r_in, inCoordSys=coordSys))
            r_in = result[-1]['out']
            coordSys = result[-1]['outCoordSys']
        result.extend(self.items[-1].traceFull(r_in, inCoordSys=coordSys, outCoordSys=outCoordSys))
        return result

    def traceReverse(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        """Trace through optic(s) in reverse.  Note, you may need to reverse the direction
        of rays for this to work.
        """
        if self.skip:
            return r, inCoordSys
        coordSys = inCoordSys
        for item in reversed(self.items[1:]):
            if not item.skip:
                r, coordSys = item.traceReverse(r, inCoordSys=coordSys)
        return self.items[0].traceReverse(r, inCoordSys=coordSys, outCoordSys=outCoordSys)

    def draw3d(self, ax, **kwargs):
        """ Recursively draw this Optic by successively drawing all subitems.
        """
        for item in self.items:
            item.draw3d(ax, **kwargs)

    def draw2d(self, ax, **kwargs):
        for item in self.items:
            item.draw2d(ax, **kwargs)

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
        """Shift the origin of this optic by `shift`.
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
        """Shift the origin of a subitem given by name (see itemDict docstring for the name format)
        by `shift`.
        """
        # If name is one of items.names, the we use withGlobalShift, and we're done.
        # If not, then we need to recurse down to whichever item contains name.
        # First verify that name is in self.itemDict
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        if name == self.name:
            return self.withGlobalShift(shift)
        # Clip off leading token
        assert name[:len(self.name)+1] == self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
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
        raise RuntimeError("Error in withGloballyShiftedOptic!, Shouldn't get here!")

    def withLocalRotation(self, rot, rotOrigin=None, coordSys=None):
        """Rotate optic by `rot`.
        """
        if rotOrigin is None and coordSys is None:
            coordSys = self.coordSys
            rotOrigin = [0,0,0]
        newItems = [item.withLocalRotation(rot, rotOrigin, coordSys) for item in self.items]
        ret = self.__class__.__new__(self.__class__)
        newDict = dict(self.__dict__)
        newDict['coordSys'] = self.coordSys.rotateLocal(rot, rotOrigin, coordSys)
        del newDict['items']
        ret.__init__(
            newItems,
            **newDict
        )
        return ret

    def withLocallyRotatedOptic(self, name, rot, rotOrigin=None, coordSys=None):
        """Rotate the subitem by `rot`.
        """
        # If name is one of items.names, the we use withLocalRotation, and we're done.
        # If not, then we need to recurse down to whichever item contains name.
        # First verify that name is in self.itemDict
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        if name == self.name:
            return self.withLocalRotation(rot, rotOrigin, coordSys)
        if rotOrigin is None and coordSys is None:
            coordSys = self.itemDict[name].coordSys
            rotOrigin = [0,0,0]
        # Clip off leading token
        assert name[:len(self.name)+1] == self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
        name = name[len(self.name)+1:]
        newItems = []
        newDict = dict(self.__dict__)
        del newDict['items']
        for i, item in enumerate(self.items):
            if name.startswith(item.name):
                if name == item.name:
                    newItems.append(item.withLocalRotation(rot, rotOrigin, coordSys))
                else:
                    newItems.append(item.withLocallyRotatedOptic(name, rot, rotOrigin, coordSys))
                newItems.extend(self.items[i+1:])
                return self.__class__(
                    newItems,
                    **newDict
                )
            newItems.append(item)
        raise RuntimeError("Error in withLocallyRotatedOptic!, Shouldn't get here!")


class Lens(CompoundOptic):
    def __init__(self, items, medium, **kwargs):
        Optic.__init__(self, **kwargs)
        self.items = tuple(items)
        self.medium = medium

    def __eq__(self, other):
        if not CompoundOptic.__eq__(self, other):
            return False
        return self.medium == other.medium

    def __repr__(self):
        out = ("{!s}([{!r}, {!r}], {!r}"
               .format(self.__class__.__name__, self.items[0], self.items[1], self.medium))
        out += Optic._repr_helper(self)
        out += ")"
        return out

    def __hash__(self):
        return hash((self.medium, CompoundOptic.__hash__(self)))

    def withGlobalShift(self, shift):
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
        """Shift the origin of a subitem given by name (see itemDict docstring for the name format)
        by `shift`.
        """
        # If name is one of items.names, the we use withGlobalShift, and we're done.
        # If not, then we need to recurse down to whicever item contains name.
        # First verify that name is in self.itemDict
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        if name == self.name:
            return self.withGlobalShift(shift)
        # Clip off leading token
        assert name[:len(self.name)+1] == self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
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
        raise RuntimeError("Error in withGloballyShiftedOptic!, Shouldn't get here!")

    def withLocalRotation(self, rot, rotOrigin=None, coordSys=None):
        """Rotate optic by `rot`.
        """
        if rotOrigin is None and coordSys is None:
            coordSys = self.coordSys
            rotOrigin = [0,0,0]
        newItems = [item.withLocalRotation(rot, rotOrigin, coordSys) for item in self.items]
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
        """Rotate the subitem by `rot`.
        """
        # If name is one of items.names, the we use withLocalRotation, and we're done.
        # If not, then we need to recurse down to whichever item contains name.
        # First verify that name is in self.itemDict
        if name not in self.itemDict:
            raise ValueError("Optic {} not found".format(name))
        if name == self.name:
            return self.withLocalRotation(rot, rotOrigin, coordSys)
        if rotOrigin is None and coordSys is None:
            coordSys = self.itemDict[name].coordSys
            rotOrigin = [0,0,0]
        # Clip off leading token
        assert name[:len(self.name)+1] == self.name+".", name[:len(self.name)+1]+" != "+self.name+"."
        name = name[len(self.name)+1:]
        newItems = []
        newDict = dict(self.__dict__)
        del newDict['items']
        del newDict['medium']
        for i, item in enumerate(self.items):
            if name.startswith(item.name):
                if name == item.name:
                    newItems.append(item.withLocalRotation(rot, rotOrigin, coordSys))
                else:
                    newItems.append(item.withLocallyRotatedOptic(name, rot, rotOrigin, coordSys))
                newItems.extend(self.items[i+1:])
                return self.__class__(
                    newItems, self.medium,
                    **newDict
                )
            newItems.append(item)
        raise RuntimeError("Error in withLocallyRotatedOptic!, Shouldn't get here!")


def drawTrace3d(ax, traceFull, start=None, end=None, **kwargs):
    if start is None:
        start = traceFull[0]['name']
    if end is None:
        end = traceFull[-1]['name']
    doPlot = False
    for surface in traceFull:
        if surface['name'] == start:
            doPlot = True
        if doPlot:
            inTransform = batoid.CoordTransform(surface['inCoordSys'], batoid.CoordSys())
            outTransform = batoid.CoordTransform(surface['outCoordSys'], batoid.CoordSys())
            for inray, outray in zip(surface['in'], surface['out']):
                if not outray.vignetted:
                    inray = inTransform.applyForward(inray)
                    outray = outTransform.applyForward(outray)
                    ax.plot(
                        [inray.x, outray.x],
                        [inray.y, outray.y],
                        [inray.z, outray.z],
                        **kwargs
                    )
        if surface['name'] == end:
            break


def drawTrace2d(ax, traceFull, start=None, end=None, **kwargs):
    if start is None:
        start = traceFull[0]['name']
    if end is None:
        end = traceFull[-1]['name']
    doPlot = False
    for surface in traceFull:
        if surface['name'] == start:
            doPlot = True
        if doPlot:
            inTransform = batoid.CoordTransform(surface['inCoordSys'], batoid.CoordSys())
            outTransform = batoid.CoordTransform(surface['outCoordSys'], batoid.CoordSys())
            for inray, outray in zip(surface['in'], surface['out']):
                if not outray.vignetted:
                    inray = inTransform.applyForward(inray)
                    outray = outTransform.applyForward(outray)
                    ax.plot(
                        [inray.x, outray.x],
                        [inray.z, outray.z],
                        **kwargs
                    )
        if surface['name'] == end:
            break


# Should pythonize RayVector so can identify when all the wavelengths are the same
# or not, and elide medium.getN() in the inner loop when possible.

# Generic trace looks like
# 1) Change coords from previous element to current local coords.
# 2) Find intersections
# 3) Mark vignetted rays
# 4) Compute reflection/refraction.
#
# Maybe start by converting global -> local -> global for every interface.  Then try
# a more optimized approach?
