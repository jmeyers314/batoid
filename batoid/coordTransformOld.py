from . import _batoid
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
        self._coordTransform = _batoid.CPPCoordTransform(
            fromSys._coordSys,
            toSys._coordSys
        )

    def applyForward(self, arg1, arg2=None, arg3=None):
        """Apply forward-direction transformation.

        Parameters
        ----------
        arg : Ray or RayVector or array
            Object to transform.  Return type is the same as the input type.

        Returns
        -------
        transformed : Ray or RayVector or array
            Result of transformation.  Type is the same as the input type.
        """
        if arg2 is not None:  # numpy arrays to transform (not-in-place)
            return self._coordTransform.applyForward(arg1, arg2, arg3)
        elif isinstance(arg1, np.ndarray):  # single np array
            return self._coordTransform.applyForward(arg1)
        else: # Ray or RayVector
            self._coordTransform.applyForwardInPlace(arg1._rv)
            return arg1

    def applyReverse(self, arg1, arg2=None, arg3=None):
        """Apply reverse-direction transformation.

        Parameters
        ----------
        arg : Ray or RayVector or array
            Object to transform.  Return type is the same as the input type.

        Returns
        -------
        transformed : Ray or RayVector or array
            Result of transformation.  Type is the same as the input type.
        """
        if arg2 is not None:  # numpy arrays to transform (not-in-place)
            return self._coordTransform.applyReverse(arg1, arg2, arg3)
        elif isinstance(arg1, np.ndarray):  # single np array
            return self._coordTransform.applyReverse(arg1)
        else: # Ray or RayVector
            self._coordTransform.applyReverseInPlace(arg1._rv)
            return arg1

    def __eq__(self, rhs):
        if not isinstance(rhs, CoordTransform): return False
        return self._coordTransform == rhs._coordTransform

    def __ne__(self, rhs):
        return not (self == rhs)

    def __repr__(self):
        return repr(self._coordTransform)

    def __hash__(self):
        return hash(self._coordTransform)
