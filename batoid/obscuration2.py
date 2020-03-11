from . import _batoid


class Obscuration2:
    def contains(self, x, y):
        return self._obsc.contains(x, y)

    def obscureInPlace(self, r):
        self._obsc.obscureInPlace(r._rv)


class ObscCircle2(Obscuration2):
    def __init__(self, radius, x=0.0, y=0.0):
        self._obsc = _batoid.CPPObscCircle2(radius, x, y)

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


class ObscAnnulus2(Obscuration2):
    def __init__(self, inner, outer, x=0.0, y=0.0):
        self._obsc = _batoid.CPPObscAnnulus2(inner, outer, x, y)

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


class ObscRectangle2(Obscuration2):
    def __init__(self, width, height, x=0.0, y=0.0, theta=0.0):
        self._obsc = _batoid.CPPObscRectangle2(width, height, x, y, theta)

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


class ObscRay2(Obscuration2):
    def __init__(self, width, theta, x=0.0, y=0.0):
        self._obsc = _batoid.CPPObscRay2(width, theta, x, y)

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


class ObscNegation2(Obscuration2):
    def __init__(self, original):
        self._original = original
        self._obsc = _batoid.CPPObscNegation2(self._original._obsc)

    @property
    def original(self):
        return self._original


class ObscUnion2(Obscuration2):
    def __init__(self, *items):
        if len(items) == 0:
            raise ValueError("Not enough items")
        elif len(items) == 1:
            if isinstance(items, (list, tuple)):
                items = items[0]
        self._items = items
        self._obsc = _batoid.CPPObscUnion2([item._obsc for item in items])

    @property
    def items(self):
        """List of `Obscuration` : unionized `Obscuration` s."""
        return self._items


class ObscIntersection2(Obscuration2):
    def __init__(self, *items):
        if len(items) == 0:
            raise ValueError("Not enough items")
        elif len(items) == 1:
            if isinstance(items, (list, tuple)):
                items = items[0]
        self._items = items
        self._obsc = _batoid.CPPObscIntersection2([item._obsc for item in items])

    @property
    def items(self):
        """List of `Obscuration` : intersected `Obscuration` s."""
        return self._items
