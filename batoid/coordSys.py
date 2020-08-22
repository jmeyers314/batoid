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
                self._coordSys = _batoid.CPPCoordSys()
            else:
                self._coordSys = _batoid.CPPCoordSys(rot.ravel())
        else:
            if rot is None:
                self._coordSys = _batoid.CPPCoordSys(origin)
            else:
                self._coordSys = _batoid.CPPCoordSys(origin, rot.ravel())

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
        return np.array(self._coordSys.origin)

    @property
    def rot(self):
        """ndarray of float, shape (3, 3): Rotation matrix taking global axes
        into current system axes.
        """
        return np.array(self._coordSys.rot).reshape(3, 3)

    @property
    def xhat(self):
        """ndarray of float, shape (3,): Orientation of local x vector in
        global coordinates.
        """
        return np.array(self._coordSys.xhat)

    @property
    def yhat(self):
        """ndarray of float, shape (3,): Orientation of local y vector in
        global coordinates.
        """
        return np.array(self._coordSys.yhat)

    @property
    def zhat(self):
        """ndarray of float, shape (3,): Orientation of local z vector in
        global coordinates.
        """
        return np.array(self._coordSys.zhat)

    def __repr__(self):
        return f"CoordSys(\n{self.origin}, \n{self.rot})\n"

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
            coordSys = CoordSys()
        return CoordSys._fromCoordSys(
            self._coordSys.rotateGlobal(
                rot.ravel(), rotCenter, coordSys._coordSys
            )
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
            self._coordSys.rotateLocal(
                rot.ravel(), rotCenter, coordSys._coordSys
            )
        )

    def __getstate__(self):
        return self.origin, self.rot

    def __setstate__(self, d):
        origin, rot = d
        self._coordSys = _batoid.CPPCoordSys(origin, rot.ravel())

    def __eq__(self, rhs):
        if not isinstance(rhs, CoordSys): return False
        return (
            np.array_equal(self.origin, rhs.origin) and
            np.array_equal(self.rot, rhs.rot)
        )

    def __ne__(self, rhs):
        return not (self == rhs)

    def __hash__(self):
        return hash((
            "batoid.CoordSys",
            tuple(self.origin.tolist()),
            tuple(self.rot.ravel().tolist())
        ))

    def copy(self):
        return CoordSys(self.origin, self.rot)
