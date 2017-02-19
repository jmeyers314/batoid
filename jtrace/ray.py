from . import _jtrace

class Ray(object):
    """A Ray to trace.

    :param origin:     A 3-element sequence indicating the x,y,z origin of the
                       ray.
    :param direction:  A 3-element sequence indicating the x,y,z vector along
                       which the ray is heading.
    :param t:          The time coordinate of the ray.
    """
    def __init__(self, origin, direction, t):
        self._ray = _jtrace.Ray(_jtrace.Vec3(origin), _jtrace.Vec3(direction), t)
