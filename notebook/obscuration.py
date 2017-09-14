import numpy as np


class Circle:
    def __init__(self, x0, y0, radius):
        self.x0 = x0
        self.y0 = y0
        self.radius = radius

    def contains(self, x, y):
        h = np.hypot(x-self.x0, y-self.y0)
        return h < self.radius


class Rectangle:
    def __init__(self, x0, y0, width, height, theta):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.theta = theta

        # Calculate 3 points on the rectangle...
        A = np.array([-width/2, -height/2])
        B = np.array([-width/2, +height/2])
        C = np.array([+width/2, +height/2])
        sth, cth = np.sin(theta), np.cos(theta)
        R = np.array([[cth, -sth], [sth, cth]])
        A = np.dot(R, A)
        B = np.dot(R, B)
        C = np.dot(R, C)

        self._A = A + np.array([x0, y0])
        self._B = B + np.array([x0, y0])
        self._C = C + np.array([x0, y0])

        self._AB = self._B - self._A
        self._ABAB = np.dot(self._AB, self._AB)
        self._BC = self._C - self._B
        self._BCBC = np.dot(self._BC, self._BC)

    def contains(self, x, y):
        if isinstance(x, np.ndarray):
            shape = x.shape
            M = np.vstack([x.ravel(), y.ravel()]).T
        else:
            M = np.array([x, y])
        AM = M - self._A
        BM = M - self._B
        ABAM = np.dot(AM, self._AB)
        BCBM = np.dot(BM, self._BC)
        contained = (0 <= ABAM)
        contained &= ABAM <= self._ABAB
        contained &= 0 <= BCBM
        contained &= BCBM <= self._BCBC
        if isinstance(x, np.ndarray):
            return contained.reshape(shape)
        else:
            return contained


class Ray:
    def __init__(self, x0, y0, width, theta):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.theta = theta

        # Just use a large-ish rectangle for this...
        _height = width
        _width = 100
        _x0 = x0 + _width*np.cos(theta)/2
        _y0 = y0 + _width*np.sin(theta)/2

        self._rect = Rectangle(_x0, _y0, _width, _height, theta)

    def contains(self, x, y):
        return self._rect.contains(x, y)


class Union:
    def __init__(self, *args):
        self.args = args

    def contains(self, x, y):
        contained = self.args[0].contains(x, y)
        for a in self.args[1:]:
            contained |= a.contains(x, y)
        return contained


class Intersection:
    def __init__(self, *args):
        self.args = args

    def contains(self, x, y):
        contained = self.args[0].contains(x, y)
        for a in self.args[1:]:
            contained &= a.contains(x, y)
        return contained


class InvertShape:
    def __init__(self, shape):
        self.shape = shape

    def contains(self, x, y):
        return np.logical_not(self.shape.contains(x, y))
