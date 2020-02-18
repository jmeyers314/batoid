from . import _batoid


class Obscuration:
    """An `Obscuration` instance is used to mark `Ray` s (potentially in
    a `RayVector`) as vignetted (i.e., obscured) if their x/y coordinates lie
    in a particular region.

    `Obscuration` s are useful for modeling pupils, clear apertures of optical
    elements, struts, or other physical obstructions in an optical system.
    Note that only the x and y local coordinates of a `Ray` are considered; the
    z coordinate is ignored.
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

    def obscure(self, r):
        """Mark a `Ray` or `RayVector` for potential vignetting.

        Parameters
        ----------
        r : `Ray` or `RayVector`
            Rays to analyze.

        Returns
        -------
        out : `Ray` or `RayVector`
            Returned object will have appropriate elements marked as vignetted.
        """
        from .ray import Ray
        from .rayVector import RayVector
        _rv = self._obsc.obscure(r._rv)
        if isinstance(r, Ray):
            return Ray._fromCPPRayVector(_rv)
        else:
            return RayVector._fromCPPRayVector(_rv)

    def obscureInPlace(self, r):
        """Mark a `Ray` or `RayVector` for potential vignetting in place.

        Parameters
        ----------
        r : `Ray` or `RayVector`
            Rays to analyze and vignette in place.
        """
        self._obsc.obscureInPlace(r._rv)

    def __eq__(self, rhs):
        return (type(self) == type(rhs)
                and self._obsc == rhs._obsc)

    def __ne__(self, rhs):
        return not (self == rhs)

    def __hash__(self):
        return hash((type(self), self._obsc))

    def __repr__(self):
        return repr(self._obsc)


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
        self._obsc = _batoid.CPPObscCircle(radius, x, y)

    @property
    def radius(self):
        """Radius of circle in meters."""
        return self._obsc.radius

    @property
    def x(self):
        """X coordinate of circle center in meters."""
        return self._obsc.x

    @property
    def y(self):
        """Y coordinate of circle center in meters."""
        return self._obsc.y


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
        self._obsc = _batoid.CPPObscAnnulus(inner, outer, x, y)

    @property
    def inner(self):
        """Inner radius of annulus in meters."""
        return self._obsc.inner

    @property
    def outer(self):
        """Outer radius of annulus in meters."""
        return self._obsc.outer

    @property
    def x(self):
        """X coordinate of annulus center in meters."""
        return self._obsc.x

    @property
    def y(self):
        """Y coordinate of annulus center in meters."""
        return self._obsc.y


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
        self._obsc = _batoid.CPPObscRectangle(width, height, x, y, theta)

    @property
    def width(self):
        """Width (X-center) of rectangle in meters."""
        return self._obsc.width

    @property
    def height(self):
        """Height (Y-center) of rectangle in meters."""
        return self._obsc.height

    @property
    def x(self):
        """X coordinate of rectangle center in meters."""
        return self._obsc.x

    @property
    def y(self):
        """Y coordinate of rectangle center in meters."""
        return self._obsc.y

    @property
    def theta(self):
        """Counter-clockwise rotation of rectangle in radians."""
        return self._obsc.theta


class ObscRay(Obscuration):
    """A finite-width ray-like obscuration.

    (Like a rectangle, but infinitely long in one direction.)

    Parameters
    ----------
    width : float
        Width of ray in meters.
    theta : float
        Rotation angle of ray in radians.
    x, y : float, optional
        Coordinates of ray origin in meters.  [default: 0.0]
    """
    def __init__(self, width, theta, x=0.0, y=0.0):
        self._obsc = _batoid.CPPObscRay(width, theta, x, y)

    @property
    def width(self):
        """Width of ray in meters."""
        return self._obsc.width

    @property
    def theta(self):
        """Rotation angle of ray in radians."""
        return self._obsc.theta

    @property
    def x(self):
        """X coordinate of ray origin in meters."""
        return self._obsc.x

    @property
    def y(self):
        """Y coordinate of ray origin in meters."""
        return self._obsc.y


class ObscNegation(Obscuration):
    """A negated obscuration.

    The originally obscured regions become clear, and the original clear
    regions become obscured.

    Parameters
    ----------
    original : `Obscuration`
        The obscuration to negate.
    """
    def __init__(self, original):
        self._original = original
        self._obsc = _batoid.CPPObscNegation(original._obsc)

    @property
    def original(self):
        """The negated original `Obscuration`."""
        return self._original


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
        self._items = items
        self._obsc = _batoid.CPPObscUnion([item._obsc for item in items])

    @property
    def items(self):
        """List of `Obscuration` : unionized `Obscuration` s."""
        return self._items


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
        self._items = items
        self._obsc = _batoid.CPPObscIntersection([item._obsc for item in items])

    @property
    def items(self):
        """List of `Obscuration` : intersected `Obscuration` s."""
        return self._items
