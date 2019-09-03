from . import _batoid
import numpy as np

def RotX(th):
    sth, cth = np.sin(th), np.cos(th)
    return np.array([[1,0,0],[0,cth,-sth],[0,sth,cth]])

def RotY(th):
    sth, cth = np.sin(th), np.cos(th)
    return np.array([[cth,0,sth],[0,1,0],[-sth,0,cth]])

def RotZ(th):
    sth, cth = np.sin(th), np.cos(th)
    return np.array([[cth,-sth,0],[sth,cth,0],[0,0,1]])


class CoordSys:
    """A coordinate system against which to measure Surfaces or Rays.

    Coordinate systems consist of an origin and a rotation.  The `origin`
    attribute specifies where in 3D space the current coordinate system's
    origin lands in the global coordinate system.  The rotation `rot` specifies
    the 3D rotation matrix to apply to the global coordinate axes to yield the
    axes of the this coordinate system.


    Parameters
    ----------
    origin : ndarray of float, shape (3,)
        Origin of coordinate system in global coordinates.
    rot : ndarray of float, shape (3, 3)
        Rotation matrix taking global axes into current system axes.
    """
    def __init__(self, origin=None, rot=None):
        if origin is None:
            if rot is None:
                self._coordSys = _batoid.CoordSys()
            else:
                self._coordSys = _batoid.CoordSys(rot)
        else:
            if rot is None:
                self._coordSys = _batoid.CoordSys(origin)
            else:
                self._coordSys = _batoid.CoordSys(origin, rot)

    @classmethod
    def _fromCoordSys(cls, _coordSys):
        ret = cls.__new__(cls)
        ret._coordSys = _coordSys
        return ret

    @property
    def origin(self):
        """ndarray of float, shape (3,): Origin of coordinate system in global
        coordinates.
        """
        return self._coordSys.origin

    @property
    def rot(self):
        """ndarray of float, shape (3, 3): Rotation matrix taking global axes
        into current system axes.
        """
        return self._coordSys.rot

    @property
    def xhat(self):
        """ndarray of float, shape (3,): Orientation of local x vector in
        global coordinates.
        """
        return self._coordSys.xhat

    @property
    def yhat(self):
        """ndarray of float, shape (3,): Orientation of local y vector in
        global coordinates.
        """
        return self._coordSys.yhat

    @property
    def zhat(self):
        """ndarray of float, shape (3,): Orientation of local z vector in
        global coordinates.
        """
        return self._coordSys.zhat

    def __repr__(self):
        return repr(self._coordSys)

    def shiftGlobal(self, dr):
        """Return new CoordSys with origin shifted along global axes.

        Parameters
        ----------
        dr : ndarray of float, shape (3,)
            Amount to shift in meters.

        Returns
        -------
        CoordSys
        """
        return CoordSys._fromCoordSys(self._coordSys.shiftGlobal(dr))

    def shiftLocal(self, dr):
        """Return new CoordSys with origin shifted along local axes.

        Parameters
        ----------
        dr : ndarray of float, shape (3,)
            Amount to shift in meters.

        Returns
        -------
        CoordSys
        """
        return CoordSys._fromCoordSys(self._coordSys.shiftLocal(dr))

    def rotateGlobal(self, rot, rotCenter=(0,0,0), coordSys=None):
        """Return new CoordSys rotated with respect to global axes.

        Parameters
        ----------
        rot : ndarray of float, shape (3, 3)
            Rotation matrix to apply.
        rotCenter : ndarray of float, shape (3,)
            Point about which to rotate.
        coordSys : CoordSys
            Coordinate system in which rotCenter is specified.

        Returns
        -------
        CoordSys
        """
        if coordSys is None:
            coordSys = self
        return CoordSys._fromCoordSys(
            self._coordSys.rotateGlobal(rot, rotCenter, coordSys._coordSys)
        )

    def rotateLocal(self, rot, rotCenter=(0,0,0), coordSys=None):
        """Return new CoordSys rotated with respect to local axes.

        Parameters
        ----------
        rot : ndarray of float, shape (3, 3)
            Rotation matrix to apply.
        rotCenter : ndarray of float, shape (3,)
            Point about which to rotate.
        coordSys : CoordSys
            Coordinate system in which rotCenter is specified.

        Returns
        -------
        CoordSys
        """
        if coordSys is None:
            coordSys = self
        return CoordSys._fromCoordSys(
            self._coordSys.rotateLocal(rot, rotCenter, coordSys._coordSys)
        )

    def __eq__(self, rhs):
        if not isinstance(rhs, CoordSys): return False
        return self._coordSys == rhs._coordSys

    def __ne__(self, rhs):
        return not (self == rhs)

    def __hash__(self):
        return hash(self._coordSys)


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
        self._coordTransform = _batoid.CoordTransform(
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
            return Ray._fromRay(self._coordTransform.applyForward(arg1._r))
        elif isinstance(arg1, RayVector):
            return RayVector._fromRayVector(
                self._coordTransform.applyForward(arg1._r)
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
            return Ray._fromRay(self._coordTransform.applyReverse(arg1._r))
        elif isinstance(arg1, RayVector):
            return RayVector._fromRayVector(
                self._coordTransform.applyReverse(arg1._r)
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
        self._coordTransform.applyForwardInPlace(r._r)

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
        self._coordTransform.applyReverseInPlace(r._r)

    def __eq__(self, rhs):
        if not isinstance(rhs, CoordTransform): return False
        return self._coordTransform == rhs._coordTransform

    def __ne__(self, rhs):
        return not (self == rhs)

    def __repr__(self):
        return repr(self._coordTransform)

    def __hash__(self):
        return hash(self._coordTransform)
