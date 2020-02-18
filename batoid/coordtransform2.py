from . import _batoid

class CoordTransform2:
    def __init__(self, fromSys, toSys):
        self.fromSys = fromSys
        self.toSys = toSys
        self._coordTransform = _batoid.CPPCoordTransform2(
            fromSys._coordSys,
            toSys._coordSys
        )

    def applyForwardInPlace(self, r):
        self._coordTransform.applyForwardInPlace(r._rv)
        r.coordSys = self.toSys

    def applyReverseInPlace(self, r):
        self._coordTransform.applyReverseInPlace(r._rv)
        r.coordSys = self.fromSys
