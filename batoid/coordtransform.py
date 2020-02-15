from . import _batoid


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
        from .ray import Ray
        from .rayVector import RayVector
        if arg2 is not None:
            return self._coordTransform.applyForward(arg1, arg2, arg3)
        elif isinstance(arg1, Ray):
            return Ray._fromCPPRay(
                self._coordTransform.applyForward(arg1._rv)[0],
                self.toSys
            )
        elif isinstance(arg1, RayVector):
            return RayVector._fromCPPRayVector(
                self._coordTransform.applyForward(arg1._rv),
            )
        else:
            return self._coordTransform.applyForward(arg1)

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
        from .ray import Ray
        from .rayVector import RayVector
        if arg2 is not None:
            return self._coordTransform.applyReverse(arg1, arg2, arg3)
        elif isinstance(arg1, Ray):
            return Ray._fromCPPRay(
                self._coordTransform.applyReverse(arg1._rv)[0],
                self.fromSys
            )
        elif isinstance(arg1, RayVector):
            return RayVector._fromCPPRayVector(
                self._coordTransform.applyReverse(arg1._rv),
            )
        else:
            return self._coordTransform.applyReverse(arg1)

    def applyForwardInPlace(self, r):
        """Apply forward-direction transformation in place.

        Parameters
        ----------
        arg : Ray or RayVector
            Object to transform.  Return type is the same as the input type.

        Returns
        -------
        transformed : Ray or RayVector
            Result of transformation.  Type is the same as the input type.
        """
        self._coordTransform.applyForwardInPlace(r._rv)

    def applyReverseInPlace(self, r):
        """Apply reverse-direction transformation in place.

        Parameters
        ----------
        arg : Ray or RayVector
            Object to transform.  Return type is the same as the input type.

        Returns
        -------
        transformed : Ray or RayVector
            Result of transformation.  Type is the same as the input type.
        """
        self._coordTransform.applyReverseInPlace(r._rv)

    def __eq__(self, rhs):
        if not isinstance(rhs, CoordTransform): return False
        return self._coordTransform == rhs._coordTransform

    def __ne__(self, rhs):
        return not (self == rhs)

    def __repr__(self):
        return repr(self._coordTransform)

    def __hash__(self):
        return hash(self._coordTransform)
