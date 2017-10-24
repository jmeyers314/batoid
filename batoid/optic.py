import batoid
import numpy as np

class Optic(object):
    """This is the most general category of batoid optical system.  It can include
    a single reflective or refractive surface, a lens consisting of two surfaces, or
    an entire telescope including multiple mirrors and/or surfaces.
    """
    def __init__(self, name=None, coordSys=None, obscuration=None,
                 inMedium=batoid.ConstMedium(1.0), outMedium=batoid.ConstMedium(1.0)):
        self.name = name
        self.inMedium = inMedium
        self.outMedium = outMedium
        if coordSys is None:
            raise ValueError("coordSys required for optic")
        self.coordSys = coordSys
        self.obscuration = obscuration

class Interface(Optic):
    def __init__(self, surface, **kwargs):
        Optic.__init__(self, **kwargs)

        self.surface = surface
        self.inRadius = 0.0
        self.outRadius = None

        # Stealing inRadius and outRadius from self.obscuration.  Only works at the moment if
        # obscuration is a negated circle.  Should at least extend this to negated annulus.
        if self.obscuration is not None:
            if isinstance(self.obscuration, batoid._batoid.ObscNegation):
                if isinstance(self.obscuration.original, batoid._batoid.ObscCircle):
                    self.outRadius = self.obscuration.original.radius

    def draw(self, ax):
        if self.outRadius is None:
            return
        print("drawing {}".format(self.name))
        # Going to draw 4 objects here: inner circle, outer circle, sag along x=0, sag along y=0
        # inner circle
        if self.inRadius != 0.0:
            th = np.linspace(0, 2*np.pi, 100)
            cth, sth = np.cos(th), np.sin(th)
            x = self.inRadius * cth
            y = self.inRadius * sth
            z = self.surface.sag(x, y)
            # ax.plot(x, y, z, c='k')
            transform = batoid._batoid.CoordTransform(self.coordSys, batoid._batoid.CoordSys())
            x, y, z = transform.applyForward(x, y, z)
            ax.plot(x, y, z, c='k')

        #outer circle
        th = np.linspace(0, 2*np.pi, 100)
        cth, sth = np.cos(th), np.sin(th)
        x = self.outRadius * cth
        y = self.outRadius * sth
        z = self.surface.sag(x, y)
        transform = batoid._batoid.CoordTransform(self.coordSys, batoid._batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')

        #next, a line at X=0
        y = np.linspace(-self.outRadius, -self.inRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        transform = batoid._batoid.CoordTransform(self.coordSys, batoid._batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')
        y = np.linspace(self.inRadius, self.outRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        transform = batoid._batoid.CoordTransform(self.coordSys, batoid._batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')

        #next, a line at Y=0
        x = np.linspace(-self.outRadius, -self.inRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        transform = batoid._batoid.CoordTransform(self.coordSys, batoid._batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')
        x = np.linspace(self.inRadius, self.outRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        transform = batoid._batoid.CoordTransform(self.coordSys, batoid._batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')

class RefractiveInterface(Interface):
    def trace(self, r, inCoordSys=batoid._batoid.CoordSys(), outCoordSys=None):
        transform = batoid._batoid.CoordTransform(inCoordSys, self.coordSys)
        r = transform.applyForward(r)
        r = self.surface.intercept(r)
        # r = self.obscuration.obscure(r)
        r = batoid._batoid.refract(r, self.surface, self.inMedium, self.outMedium)
        if outCoordSys is None:
            return r, self.coordSys
        else:
            transform = batoid._batoid.CoordTransform(self.coordSys, outCoordSys)
            return transform.applyForward(r), outCoordSys

class Mirror(Interface):
    def trace(self, r, inCoordSys=batoid._batoid.CoordSys(), outCoordSys=None):
        transform = batoid._batoid.CoordTransform(inCoordSys, self.coordSys)
        r = transform.applyForward(r)
        r = self.surface.intercept(r)
        # r = self.obscuration.obscure(r)
        r = batoid._batoid.reflect(r, self.surface)
        if outCoordSys is None:
            return r, self.coordSys
        else:
            transform = batoid._batoid.CoordTransform(self.coordSys, outCoordSys)
            return transform.applyForward(r), outCoordSys

class CompoundOptic(Optic):
    def __init__(self, items, **kwargs):
        Optic.__init__(self, **kwargs)
        self.items = items

    def trace(self, r, inCoordSys=batoid._batoid.CoordSys(), outCoordSys=None):
        coordSys = inCoordSys
        for item in self.items:
            r, coordSys = item.trace(r, inCoordSys=coordSys)
        if outCoordSys is None:
            return r, coordSys
        else:
            transform = batoid._batoid.CoordTransform(coordSys, outCoordSys)
            return transform.applyForward(r), coordSys
    def draw(self, ax):
        for item in self.items:
            item.draw(ax)

class Lens(CompoundOptic):
    def __init__(self, items, medium, **kwargs):
        Optic.__init__(self, **kwargs)
        self.items = items
        self.medium = medium

class Detector(Interface):
    pass

class Baffle(Interface):
    pass

class Phantom(Interface):
    pass

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
