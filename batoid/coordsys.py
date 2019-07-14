from . import _batoid
from .ray import Ray
from .rayVector import RayVector
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
        return self._coordSys.origin

    @property
    def rot(self):
        return self._coordSys.rot

    @property
    def xhat(self):
        return self._coordSys.xhat

    @property
    def yhat(self):
        return self._coordSys.yhat

    @property
    def zhat(self):
        return self._coordSys.zhat

    def __repr__(self):
        return repr(self._coordSys)

    def shiftGlobal(self, dr):
        return CoordSys._fromCoordSys(self._coordSys.shiftGlobal(dr))

    def shiftLocal(self, dr):
        return CoordSys._fromCoordSys(self._coordSys.shiftLocal(dr))

    def rotateGlobal(self, rot, rotCenter=(0,0,0), coordSys=None):
        if coordSys is None:
            coordSys = self
        return CoordSys._fromCoordSys(self._coordSys.rotateGlobal(rot, rotCenter, coordSys._coordSys))

    def rotateLocal(self, rot, rotCenter=(0,0,0), coordSys=None):
        if coordSys is None:
            coordSys = self
        return CoordSys._fromCoordSys(self._coordSys.rotateLocal(rot, rotCenter, coordSys._coordSys))

    def __eq__(self, rhs):
        if not isinstance(rhs, CoordSys): return False
        return self._coordSys == rhs._coordSys

    def __ne__(self, rhs):
        return not (self == rhs)

    def __hash__(self):
        return hash(self._coordSys)


class CoordTransform:
    def __init__(self, fromSys, toSys):
        self._coordTransform = _batoid.CoordTransform(fromSys._coordSys, toSys._coordSys)

    def applyForward(self, arg1, arg2=None, arg3=None):
        if arg2 is not None:
            return self._coordTransform.applyForward(arg1, arg2, arg3)
        elif isinstance(arg1, Ray):
            return Ray._fromRay(self._coordTransform.applyForward(arg1._r))
        elif isinstance(arg1, RayVector):
            return RayVector._fromRayVector(self._coordTransform.applyForward(arg1._r))
        else:
            return self._coordTransform.applyForward(arg1)

    def applyReverse(self, arg1, arg2=None, arg3=None):
        if arg2 is not None:
            return self._coordTransform.applyReverse(arg1, arg2, arg3)
        elif isinstance(arg1, Ray):
            return Ray._fromRay(self._coordTransform.applyReverse(arg1._r))
        elif isinstance(arg1, RayVector):
            return RayVector._fromRayVector(self._coordTransform.applyReverse(arg1._r))
        else:
            return self._coordTransform.applyReverse(arg1)

    def applyForwardInPlace(self, r):
        self._coordTransform.applyForwardInPlace(r._r)

    def applyReverseInPlace(self, r):
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
