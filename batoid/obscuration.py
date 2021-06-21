import numpy as np
from . import _batoid
from .trace import obscure

class Obscuration:
    """An `Obscuration` instance is used to mark as vignetted (i.e., obscured)
    if their x/y coordinates lie in a particular region.

    `Obscuration` s are useful for modeling pupils, clear apertures of optical
    elements, struts, or other physical obstructions in an optical system.
    Note that only the x and y local coordinates of rays are considered; the z
    coordinate is ignored.
    """
    def contains(self, x, y):
        """Return True if the point (x,y) is obscured.

        Parameters
        ----------
        x, y : float
            X/Y coordinates of point in meters.

        Returns
        -------
        obscured : bool
            True if point is obscured.  False otherwise.
        """
        return self._obsc.contains(x, y)

    def obscure(self, rv):
        """Mark rays for potential vignetting.

        Parameters
        ----------
        rv : `RayVector`
            Rays to analyze.

        Returns
        -------
        out : `RayVector`
            Returned object will have appropriate elements marked as vignetted.
        """
        obscure(self, rv)

    def __ne__(self, rhs):
        return not (self == rhs)


class ObscCircle(Obscuration):
    """A circular obscuration.

    Parameters
    ----------
    radius : float
        Radius of circle in meters.
    x, y : float, optional
        Coordinates of circle center in meters.  [default: 0.0]
    """
    def __init__(self, radius, x=0.0, y=0.0):
        self.radius = radius
        self.x = x
        self.y = y
        self._obsc = _batoid.CPPObscCircle(radius, x, y)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return (
                self.radius == rhs.radius
                and self.x == rhs.x
                and self.y == rhs.y
            )
        return False

    def __getstate__(self):
        return self.radius, self.x, self.y

    def __setstate__(self, args):
        self.radius, self.x, self.y = args
        self._obsc = _batoid.CPPObscCircle(*args)

    def __hash__(self):
        return hash(("batoid.ObscCircle", self.radius, self.x, self.y))

    def __repr__(self):
        out = f"ObscCircle({self.radius}"
        if self.x != 0 or self.y != 0:
            out += f", {self.x}, {self.y}"
        out += ")"
        return out


class ObscAnnulus(Obscuration):
    """An annular obscuration.

    Parameters
    ----------
    inner : float
        Inner radius of annulus in meters.
    outer : float
        Outer radius of annulus in meters.
    x, y : float, optional
        Coordinates of annulus center in meters.  [default: 0.0]
    """
    def __init__(self, inner, outer, x=0.0, y=0.0):
        self.inner = inner
        self.outer = outer
        self.x = x
        self.y = y
        self._obsc = _batoid.CPPObscAnnulus(inner, outer, x, y)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return (
                self.inner == rhs.inner
                and self.outer == rhs.outer
                and self.x == rhs.x
                and self.y == rhs.y
            )
        return False

    def __getstate__(self):
        return self.inner, self.outer, self.x, self.y

    def __setstate__(self, args):
        self.inner, self.outer, self.x, self.y = args
        self._obsc = _batoid.CPPObscAnnulus(*args)

    def __hash__(self):
        return hash((
            "batoid.ObscAnnulus", self.inner, self.outer, self.x, self.y
        ))

    def __repr__(self):
        out = f"ObscAnnulus({self.inner}, {self.outer}"
        if self.x != 0 or self.y != 0:
            out += f", {self.x}, {self.y}"
        out += ")"
        return out


class ObscRectangle(Obscuration):
    """A rectangular obscuration.

    Parameters
    ----------
    width : float
        Width (X-extent) of rectangle in meters.
    height : float
        Height (Y-extent) of rectangle in meters.
    x, y : float, optional
        Coordinates of rectangle center in meters.  [default: 0.0]
    theta : float, optional
        Counter-clockwise rotation of rectangle in radians.  [default: 0.0]
    """
    def __init__(self, width, height, x=0.0, y=0.0, theta=0.0):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.theta = theta
        self._obsc = _batoid.CPPObscRectangle(width, height, x, y, theta)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return (
                self.width == rhs.width
                and self.height == rhs.height
                and self.x == rhs.x
                and self.y == rhs.y
                and self.theta == rhs.theta
            )
        return False

    def __getstate__(self):
        return self.width, self.height, self.x, self.y, self.theta

    def __setstate__(self, args):
        self.width, self.height, self.x, self.y, self.theta = args
        self._obsc = _batoid.CPPObscRectangle(*args)

    def __hash__(self):
        return hash((
            "batoid.ObscRectangle",
            self.width, self.height, self.x, self.y, self.theta
        ))

    def __repr__(self):
        out = f"ObscRectangle({self.width}, {self.height}"
        if self.x != 0.0 or self.y != 0.0:
            out += f", {self.x}, {self.y}"
        if self.theta != 0.0:
            out += f", theta={self.theta}"
        out += ")"
        return out


class ObscRay(Obscuration):
    """A finite-width ray-like obscuration.

    (Like a rectangle, but infinitely long in one direction.)

    Parameters
    ----------
    width : float
        Width of ray obscuration in meters.
    theta : float
        Rotation angle of ray in radians.
    x, y : float, optional
        Coordinates of ray origin in meters.  [default: 0.0]
    """
    def __init__(self, width, theta, x=0.0, y=0.0):
        self.width = width
        self.theta = theta
        self.x = x
        self.y = y
        self._obsc = _batoid.CPPObscRay(width, theta, x, y)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return (
                self.width == rhs.width
                and self.theta == rhs.theta
                and self.x == rhs.x
                and self.y == rhs.y
            )
        return False

    def __getstate__(self):
        return self.width, self.theta, self.x, self.y

    def __setstate__(self, args):
        self.width, self.theta, self.x, self.y = args
        self._obsc = _batoid.CPPObscRay(*args)

    def __hash__(self):
        return hash((
            "batoid.ObscRay",
            self.width, self.theta, self.x, self.y
        ))

    def __repr__(self):
        out = f"ObscRay({self.width}, {self.theta}"
        if self.x != 0.0 or self.y != 0.0:
            out += f", {self.x}, {self.y}"
        out += ")"
        return out


class ObscPolygon(Obscuration):
    """A an arbitrary polygon shaped obscuration.

    Parameters
    ----------
    xs : list of float
        x-coordinates of polygon vertices (in order)
    ys : list of float
        y-coordinates of polygon vertices (in order)
    """

    def __init__(self, xs, ys):
        self.xs = np.ascontiguousarray(xs, dtype=float)
        self.ys = np.ascontiguousarray(ys, dtype=float)
        self._obsc = _batoid.CPPObscPolygon(
            self.xs.ctypes.data,
            self.ys.ctypes.data,
            len(self.xs)
        )

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return (
                np.all(self.xs == rhs.xs)
                and np.all(self.ys == rhs.ys)
            )
        return False

    def __getstate__(self):
        return self.xs, self.ys

    def __setstate__(self, args):
        self.__init__(*args)

    def __hash__(self):
        return hash((
            "batoid.ObscPolygon",
            tuple(self.xs),
            tuple(self.ys)
        ))

    def __repr__(self):
        return f"ObscPolygon({self.xs!r}, {self.ys!r})"

    def containsGrid(self, xgrid, ygrid):
        nx = len(xgrid)
        ny = len(ygrid)
        out = np.empty((ny, nx), dtype=bool)
        self._obsc.containsGrid(
            xgrid.ctypes.data, ygrid.ctypes.data, out.ctypes.data, nx, ny
        )
        return out


class ObscNegation(Obscuration):
    """A negated obscuration.

    The originally obscured regions become clear, and the originally clear
    regions become obscured.

    Parameters
    ----------
    original : `Obscuration`
        The obscuration to negate.
    """
    def __init__(self, original):
        self.original = original
        self._obsc = _batoid.CPPObscNegation(original._obsc)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return self.original == rhs.original
        return False

    def __getstate__(self):
        return self.original

    def __setstate__(self, original):
        self.__init__(original)

    def __hash__(self):
        return hash((
            "batoid.ObscNegation",
            self.original
        ))

    def __repr__(self):
        return f"ObscNegation({self.original!r})"


class ObscUnion(Obscuration):
    """A union of `Obscuration` s.

    Parameters
    ----------
    *items : `Obscuration` s
        The `Obscuration` s to unionize.

    Examples
    --------
    Though not very useful, one could in principle unionize a circle and an
    annulus to make a larger circle:

    >>> small_circle = batoid.ObscCircle(1.0)
    >>> annulus = batoid.ObscAnnulus(1.0, 2.0)
    >>> big_circle = batoid.ObscCircle(2.0)
    >>> alternate_big_circle = batoid.ObscUnion(small_circle, annulus)

    Using a list or tuple is also okay with `ObscUnion`.

    >>> other_alternate_big_circle = batoid.ObscUnion([small_circle, annulus])

    Although ``big_circle`` and ``alternate_big_circle`` will not compare equal
    above, their behavior with respect to obscuring rays is the same:

    >>> rays = batoid.RayVector.asGrid(
            backDist=10.0, wavelength=500e-9, nx=10, lx=4.0, dirCos=(0,0,1)
        )
    >>> obsc1 = big_circle.obscure(rays)
    >>> obsc2 = alternate_big_circle.obscure(rays)
    >>> obsc1 == obsc2
    True
    """
    def __init__(self, *items):
        if len(items) == 0:
            raise ValueError("Not enough items")
        elif len(items) == 1:
            if isinstance(items, (list, tuple)):
                items = items[0]
        self.items = sorted(items, key=repr)
        self._obsc = _batoid.CPPObscUnion([item._obsc for item in items])

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return self.items == rhs.items
        return False

    def __getstate__(self):
        return self.items

    def __setstate__(self, items):
        self.__init__(items)

    def __hash__(self):
        return hash((
            "batoid.ObscUnion",
            tuple(self.items)
        ))

    def __repr__(self):
        out = f"ObscUnion({self.items[0]!r}"
        for item in self.items[1:]:
            out += f", {item!r}"
        out += ")"
        return out


class ObscIntersection(Obscuration):
    """An intersection of `Obscuration` s.

    Parameters
    ----------
    *items : `Obscuration` s
        The `Obscuration` s to intersect.

    Examples
    --------
    An alternate way to create an annulus is the intersection of a large circle
    and a negated small circle.

    >>> big_circle = batoid.ObscCircle(2.0)
    >>> small_circle = batoid.ObscCircle(1.0)
    >>> annulus = batoid.ObscAnnulus(1.0, 2.0)
    >>> alternate_annulus = batoid.ObscIntersection(
            batoid.ObscNegation(small_circle),
            big_circle
        )

    Using a list or tuple is also okay with `ObscIntersection`.

    >>> other_alternate_annulus = batoid.ObscIntersection([
            batoid.ObscNegation(small_circle),
            big_circle
        ])

    Although ``annulus`` and ``alternate_annulus`` will not compare equal
    above, their behavior with respect to obscuring rays is the same:

    >>> rays = batoid.RayVector.asGrid(
            backDist=10.0, wavelength=500e-9, nx=10, lx=4.0, dirCos=(0,0,1)
        )
    >>> obsc1 = annulus.obscure(rays)
    >>> obsc2 = alternate_annulus.obscure(rays)
    >>> obsc1 == obsc2
    True
    """
    def __init__(self, *items):
        if len(items) == 0:
            raise ValueError("Not enough items")
        elif len(items) == 1:
            if isinstance(items, (list, tuple)):
                items = items[0]
        self.items = sorted(items, key=repr)
        self._obsc = _batoid.CPPObscIntersection([item._obsc for item in items])

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return self.items == rhs.items
        return False

    def __getstate__(self):
        return self.items

    def __setstate__(self, items):
        self.__init__(items)

    def __hash__(self):
        return hash((
            "batoid.ObscIntersection",
            tuple(self.items)
        ))

    def __repr__(self):
        out = f"ObscIntersection({self.items[0]!r}"
        for item in self.items[1:]:
            out += f", {item!r}"
        out += ")"
        return out
