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
    """A coordinate system against which to measure surfaces or rays.

    Coordinate systems consist of an origin and a rotation.  The ``origin``
    attribute specifies where in 3D space the current coordinate system's
    origin lands in the global coordinate system.  The rotation ``rot``
    specifies the 3D rotation matrix to apply to the global coordinate axes to
    yield the axes of the this coordinate system.

    Parameters
    ----------
    origin : ndarray of float, shape (3,)
        Origin of coordinate system in global coordinates.
    rot : ndarray of float, shape (3, 3)
        Rotation matrix taking global axes into current system axes.
    """
    def __init__(self, origin=None, rot=None):
        if origin is None:
            origin = np.zeros(3, dtype=float)
        if rot is None:
            rot = np.eye(3, dtype=float)
        self.origin = np.array(origin)
        self.rot = np.array(rot)

    @property
    def xhat(self):
        """ndarray of float, shape (3,): Orientation of local x vector in
        global coordinates.
        """
        return self.rot[:, 0]

    @property
    def yhat(self):
        """ndarray of float, shape (3,): Orientation of local y vector in
        global coordinates.
        """
        return self.rot[:, 1]

    @property
    def zhat(self):
        """ndarray of float, shape (3,): Orientation of local z vector in
        global coordinates.
        """
        return self.rot[:, 2]

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
        return CoordSys(self.origin+dr, self.rot)

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
        # Rotate the shift into global coordinates, then do the shift globally
        return self.shiftGlobal(self.rot@dr)

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
        # Find rot center in global coordinates
        globalRotCenter = coordSys.rot@rotCenter + coordSys.origin
        # Then rotate about this center
        return CoordSys(
            rot@(self.origin-globalRotCenter)+globalRotCenter,
            rot@self.rot
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
        # Find rot center in global coordinates
        globalRotCenter = coordSys.rot@rotCenter + coordSys.origin
        # first rotate rot into global coords: (self.rot rot self.rot.T),
        # then apply that: (self.rot rot self.rot.T) self.rot = self.rot rot
        rTmp = self.rot@rot
        return CoordSys(
            rTmp@self.rot.T@(self.origin-globalRotCenter)+globalRotCenter,
            rTmp
        )

    def __getstate__(self):
        return self.origin, self.rot

    def __setstate__(self, d):
        self.origin, self.rot = d

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

    def __repr__(self):
        rotstr = np.array2string(self.rot, separator=', ').replace('\n', '')
        return f"CoordSys({self.origin!r}, array({rotstr}))"
