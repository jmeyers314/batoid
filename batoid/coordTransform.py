from . import _batoid
from .coordSys import CoordSys
import numpy as np


class CoordTransform:
    """Transformation between two coordinate systems.

    Parameters
    ----------
    fromSys : CoordSys
        Origin coordinate systems.
    toSys : CoordSys
        Destination coordinate systems.
    """
    def __init__(self, fromSys, toSys):
        self.fromSys = fromSys
        self.toSys = toSys
        self.dr = fromSys.rot.T@(toSys.origin - fromSys.origin)
        self.drot = fromSys.rot.T@toSys.rot

    def __getstate__(self):
        return self.fromSys, self.toSys

    def __setstate__(self, d):
        self.__init__(*d)

    def __eq__(self, rhs):
        if not isinstance(rhs, CoordTransform): return False
        return (
            self.fromSys == rhs.fromSys and
            self.toSys == rhs.toSys
        )

    def __ne__(self, rhs):
        return not (self == rhs)

    def applyForward(self, rv):
        from .trace import applyForwardTransform
        return applyForwardTransform(self, rv)

    def applyReverse(self, rv):
        from .trace import applyReverseTransform
        return applyReverseTransform(self, rv)

    def applyForwardArray(self, x, y, z):
        r = np.array([x, y, z]).T
        r -= self.dr
        return self.drot.T@r.T

    def applyReverseArray(self, x, y, z):
        r = np.array([x, y, z])
        r = (self.drot@r).T
        r += self.dr
        return r.T

    def __repr__(self):
        return f"CoordTransform({self.fromSys!r}, {self.toSys!r})"

    def __hash__(self):
        return hash(("CoordTransform", self.fromSys, self.toSys))
