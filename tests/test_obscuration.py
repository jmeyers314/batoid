import batoid
import numpy as np
from test_helpers import timer, do_pickle


@timer
def test_ObscCircle():
    import random
    random.seed(5)

    for i in range(100):
        cx = random.gauss(0.0, 1.0)
        cy = random.gauss(0.0, 1.0)
        r = random.uniform(0.5, 1.5)

        obsc = batoid._batoid.ObscCircle(r, cx, cy)
        for i in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            assert obsc.contains(x, y) == (np.hypot(x-cx, y-cy) <= r)
        do_pickle(obsc)


@timer
def test_ObscAnnulus():
    import random
    random.seed(57)

    for i in range(100):
        cx = random.gauss(0.0, 1.0)
        cy = random.gauss(0.0, 1.0)
        inner = random.uniform(0.5, 1.5)
        outer = random.uniform(1.6, 1.9)

        obsc = batoid._batoid.ObscAnnulus(inner, outer, cx, cy)
        for i in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            assert obsc.contains(x, y) == (inner <= np.hypot(x-cx, y-cy) < outer)
        do_pickle(obsc)


@timer
def test_ObscRectangle():
    import random
    random.seed(577)

    for i in range(100):
        cx = random.gauss(0.0, 1.0)
        cy = random.gauss(0.0, 1.0)
        w = random.uniform(0.5, 2.5)
        h = random.uniform(0.5, 2.5)

        obsc = batoid._batoid.ObscRectangle(w, h, cx, cy, 0.0)

        for i in range(100):
            x = random.gauss(0.0, 2.0)
            y = random.gauss(0.0, 2.0)
            assert obsc.contains(x, y) == (x > cx-w/2 and x < cx+w/2 and y > cy-h/2 and y < cy+h/2)

        th = random.uniform(0.0, np.pi/2)
        obsc = batoid._batoid.ObscRectangle(w, h, cx, cy, th)
        for i in range(100):
            x = random.gauss(0.0, 2.0)
            y = random.gauss(0.0, 2.0)
            xp = (x-cx)*np.cos(-th) - (y-cy)*np.sin(-th) + cx
            yp = (x-cx)*np.sin(-th) + (y-cy)*np.cos(-th) + cy
            assert obsc.contains(x, y) == (xp > cx-w/2 and xp < cx+w/2 and yp > cy-h/2 and yp < cy+h/2)
        do_pickle(obsc)


def test_ObscRay():
    # Wait for ObscRay to be better defined (not as a rectangle!)
    pass


@timer
def test_ObscNegation():
    import random
    random.seed(5772)

    for i in range(100):
        cx = random.gauss(0.0, 1.0)
        cy = random.gauss(0.0, 1.0)
        r = random.uniform(0.5, 1.5)

        obsc = batoid._batoid.ObscCircle(r, cx, cy)
        obsc = batoid._batoid.ObscNegation(obsc)
        do_pickle(obsc)
        for i in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            assert obsc.contains(x, y) == (np.hypot(x-cx, y-cy) > r)
        # also test config parsing of "ClearCircle"
        config = {'type':'ClearCircle', 'x':cx, 'y':cy, 'radius':r}
        obsc = batoid.parse.parse_obscuration(config)
        do_pickle(obsc)
        for i in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            assert obsc.contains(x, y) == (np.hypot(x-cx, y-cy) > r)


@timer
def test_ObscCompound():
    import random
    random.seed(57721)

    for i in range(100):
        rx = random.gauss(0.0, 1.0)
        ry = random.gauss(0.0, 1.0)
        w = random.uniform(0.5, 2.5)
        h = random.uniform(0.5, 2.5)
        th = random.uniform(0.0, np.pi)
        rect = batoid._batoid.ObscRectangle(w, h, rx, ry, th)

        cx = random.gauss(0.0, 1.0)
        cy = random.gauss(0.0, 1.0)
        r = random.uniform(0.5, 1.5)
        circ = batoid._batoid.ObscCircle(r, cx, cy)

        union = batoid._batoid.ObscUnion([rect, circ])
        do_pickle(union)
        intersection = batoid._batoid.ObscIntersection([rect, circ])

        for i in range(100):
            x = random.gauss(0.0, 2.0)
            y = random.gauss(0.0, 2.0)
            assert union.contains(x, y) == (rect.contains(x, y) or circ.contains(x, y))
            assert intersection.contains(x, y) == (rect.contains(x, y) and circ.contains(x, y))


if __name__ == '__main__':
    test_ObscCircle()
    test_ObscAnnulus()
    test_ObscRectangle()
    test_ObscNegation()
    test_ObscCompound()
