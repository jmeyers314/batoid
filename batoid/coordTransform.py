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
        self._coordTransform = _batoid.CPPCoordTransform(
            fromSys._coordSys,
            toSys._coordSys
        )

    @property
    def fromSys(self):
        return CoordSys._fromCoordSys(self._coordTransform.source)

    @property
    def toSys(self):
        return CoordSys._fromCoordSys(self._coordTransform.destination)

    @property
    def dr(self):
        return np.array([self._coordTransform.dr])

    @property
    def rot(self):
        return np.array([self._coordTransform.rot]).reshape(3, 3)

    def __getstate__(self):
        return self.fromSys, self.toSys

    def __setstate__(self, d):
        self._coordTransform = _batoid.CPPCoordTransform(
            d[0]._coordSys,
            d[1]._coordSys
        )

    def applyForward(self, rv):
        """Apply forward-direction transformation.

        Parameters
        ----------
        arg : RayVector
            Object to transform.

        Returns
        -------
        transformed : RayVector
            Result of transformation.
        """
        assert self.fromSys == rv.coordSys
        self._coordTransform.applyForwardInPlace(rv._rv)
        return rv

    def applyReverse(self, rv):
        """Apply reverse-direction transformation.

        Parameters
        ----------
        arg : RayVector
            Object to transform.

        Returns
        -------
        transformed : RayVector
            Result of transformation.
        """
        assert self.toSys == rv.coordSys
        self._coordTransform.applyReverseInPlace(rv._rv)
        return rv

    def __eq__(self, rhs):
        if not isinstance(rhs, CoordTransform): return False
        return (
            self.fromSys == rhs.fromSys and
            self.toSys == rhs.toSys
        )

    def __ne__(self, rhs):
        return not (self == rhs)

    # def __repr__(self):
    #     return repr(self._coordTransform)
    #
    # def __hash__(self):
    #     return hash(self._coordTransform)
