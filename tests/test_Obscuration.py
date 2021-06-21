import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_ObscCircle():
    rng = np.random.default_rng(5)
    size = 10_000

    for i in range(100):
        cx = rng.normal(0.0, 1.0)
        cy = rng.normal(0.0, 1.0)
        r = rng.uniform(0.5, 1.5)
        obsc = batoid.ObscCircle(r, cx, cy)

        for i in range(100):
            x = rng.normal(0.0, 1.0)
            y = rng.normal(0.0, 1.0)
            assert obsc.contains(x, y) == (np.hypot(x-cx, y-cy) <= r)

        x = rng.normal(0.0, 1.0, size=size)
        y = rng.normal(0.0, 1.0, size=size)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            np.hypot(x-cx, y-cy) <= r
        )

        do_pickle(obsc)

        rv = batoid.RayVector(x, y, 0.0, 0.0, 0.0, 0.0)
        batoid.obscure(obsc, rv)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            rv.vignetted
        )

        # Check method syntax too
        rv = batoid.RayVector(x, y, 0.0, 0.0, 0.0, 0.0)
        obsc.obscure(rv)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            rv.vignetted
        )


@timer
def test_ObscAnnulus():
    rng = np.random.default_rng(57)
    size = 10_000

    for i in range(100):
        cx = rng.normal(0.0, 1.0)
        cy = rng.normal(0.0, 1.0)
        inner = rng.uniform(0.5, 1.5)
        outer = rng.uniform(1.6, 1.9)
        obsc = batoid.ObscAnnulus(inner, outer, cx, cy)

        for i in range(100):
            x = rng.normal(0.0, 1.0)
            y = rng.normal(0.0, 1.0)
            assert obsc.contains(x, y) == (
                inner <= np.hypot(x-cx, y-cy) < outer
            )

        x = rng.normal(0.0, 1.0, size=size)
        y = rng.normal(0.0, 1.0, size=size)
        r = np.hypot(x-cx, y-cy)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            (inner <= r) & (r < outer)
        )

        do_pickle(obsc)

        rv = batoid.RayVector(x, y, 0.0, 0.0, 0.0, 0.0)
        batoid.obscure(obsc, rv)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            rv.vignetted
        )


@timer
def test_ObscRectangle():
    rng = np.random.default_rng(577)
    size = 10_000

    for i in range(100):
        cx = rng.normal(0.0, 1.0)
        cy = rng.normal(0.0, 1.0)
        w = rng.uniform(0.5, 2.5)
        h = rng.uniform(0.5, 2.5)
        obsc = batoid.ObscRectangle(w, h, cx, cy)

        for i in range(100):
            x = rng.normal(0.0, 2.0)
            y = rng.normal(0.0, 2.0)
            assert obsc.contains(x, y) == (x > cx-w/2 and x < cx+w/2 and y > cy-h/2 and y < cy+h/2)

        th = rng.uniform(0.0, np.pi/2)
        obsc = batoid.ObscRectangle(w, h, cx, cy, th)

        for i in range(100):
            x = rng.normal(0.0, 2.0)
            y = rng.normal(0.0, 2.0)
            xp = (x-cx)*np.cos(-th) - (y-cy)*np.sin(-th) + cx
            yp = (x-cx)*np.sin(-th) + (y-cy)*np.cos(-th) + cy
            assert obsc.contains(x, y) == (xp > cx-w/2 and xp < cx+w/2 and yp > cy-h/2 and yp < cy+h/2)

        x = rng.normal(0.0, 2.0, size=size)
        y = rng.normal(0.0, 2.0, size=size)
        xp = (x-cx)*np.cos(-th) - (y-cy)*np.sin(-th) + cx
        yp = (x-cx)*np.sin(-th) + (y-cy)*np.cos(-th) + cy
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            (xp > cx-w/2) & (xp < cx+w/2) & (yp > cy-h/2) & (yp < cy+h/2)
        )

        do_pickle(obsc)

        rv = batoid.RayVector(x, y, 0.0, 0.0, 0.0, 0.0)
        batoid.obscure(obsc, rv)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            rv.vignetted
        )


@timer
def test_ObscRay():
    rng = np.random.default_rng(5772)
    size = 10_000

    for i in range(100):
        cx = rng.normal(0.0, 1.0)
        cy = rng.normal(0.0, 1.0)
        w = rng.uniform(0.5, 2.5)
        th = rng.uniform(0.0, np.pi/2)
        obsc = batoid.ObscRay(w, th, cx, cy)

        for i in range(100):
            x = rng.normal(0.0, 2.0)
            y = rng.normal(0.0, 2.0)
            xp = (x-cx)*np.cos(-th) - (y-cy)*np.sin(-th)
            yp = (x-cx)*np.sin(-th) + (y-cy)*np.cos(-th)
            assert obsc.contains(x, y) == (xp > 0.0 and yp > -w/2 and yp < w/2)

        x = rng.normal(0.0, 2.0, size=size)
        y = rng.normal(0.0, 2.0, size=size)
        xp = (x-cx)*np.cos(-th) - (y-cy)*np.sin(-th)
        yp = (x-cx)*np.sin(-th) + (y-cy)*np.cos(-th)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            (xp > 0.0) & (yp > -w/2) & (yp < w/2)
        )

        do_pickle(obsc)

        rv = batoid.RayVector(x, y, 0.0, 0.0, 0.0, 0.0)
        batoid.obscure(obsc, rv)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            rv.vignetted
        )


@timer
def test_ObscPolygon():
    rng = np.random.default_rng(57721)
    size = 10_000

    # Test equivalency with ObscRectangle
    for i in range(100):
        cx = rng.normal(0.0, 1.0)
        cy = rng.normal(0.0, 1.0)
        w = rng.uniform(0.5, 2.5)
        h = rng.uniform(0.5, 2.5)

        xs = [cx-w/2, cx+w/2, cx+w/2, cx-w/2]
        ys = [cy-h/2, cy-h/2, cy+h/2, cy+h/2]
        obscPoly = batoid.ObscPolygon(xs, ys)
        obscRect = batoid.ObscRectangle(w, h, cx, cy, 0.0)

        for i in range(100):
            x = rng.normal(0.0, 2.0)
            y = rng.normal(0.0, 2.0)
            assert obscPoly.contains(x, y) == obscRect.contains(x, y)

        x = rng.normal(0.0, 2.0, size=size)
        y = rng.normal(0.0, 2.0, size=size)
        np.testing.assert_array_equal(
            obscPoly.contains(x, y),
            obscRect.contains(x, y)
        )

        do_pickle(obscPoly)

        rv = batoid.RayVector(x, y, 0.0, 0.0, 0.0, 0.0)
        batoid.obscure(obscPoly, rv)
        np.testing.assert_array_equal(
            obscPoly.contains(x, y),
            rv.vignetted
        )

    # Try Union of two Rectangles equal to Polygon.
    # Center of both rectangles at (1, 2)
    # One is width=4, height=2
    # Other is width=2, height=4
    r1 = batoid.ObscRectangle(4, 2, 1, 2)
    r2 = batoid.ObscRectangle(2, 4, 1, 2)
    o1 = batoid.ObscUnion([r1, r2])
    xs = [-2, -1, -1, 1, 1, 2,  2,  1,  1, -1, -1, -2, -2]
    ys = [ 1,  1,  2, 2, 1, 1, -1, -1, -2, -2, -1, -1,  1]
    o2 = batoid.ObscPolygon(np.array(xs)+1, np.array(ys)+2)

    x = rng.normal(0.0, 2.0, size=size)
    y = rng.normal(0.0, 2.0, size=size)
    np.testing.assert_array_equal(
        o1.contains(x, y),
        o2.contains(x, y)
    )

    # Check containsGrid
    x = np.linspace(-10.0, 10.0, 25)
    y = np.linspace(-10.0, 10.0, 25)
    xx, yy = np.meshgrid(x, y)
    np.testing.assert_array_equal(
        o2.contains(xx, yy),
        o2.containsGrid(x, y)
    )


@timer
def test_ObscNegation():
    rng = np.random.default_rng(577215)
    size = 10_000

    for i in range(100):
        cx = rng.normal(0.0, 1.0)
        cy = rng.normal(0.0, 1.0)
        r = rng.uniform(0.5, 1.5)

        original = batoid.ObscCircle(r, cx, cy)
        obsc = batoid.ObscNegation(original)
        do_pickle(obsc)
        for i in range(100):
            x = rng.normal(0.0, 1.0)
            y = rng.normal(0.0, 1.0)
            assert obsc.contains(x, y) != original.contains(x, y)

        x = rng.normal(0.0, 1.0, size=size)
        y = rng.normal(0.0, 1.0, size=size)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            ~original.contains(x, y)
        )

        do_pickle(obsc)

        rv = batoid.RayVector(x, y, 0.0, 0.0, 0.0, 0.0)
        batoid.obscure(obsc, rv)
        np.testing.assert_array_equal(
            obsc.contains(x, y),
            rv.vignetted
        )

        # also test config parsing of "ClearCircle"
        config = {'type':'ClearCircle', 'x':cx, 'y':cy, 'radius':r}
        obsc = batoid.parse.parse_obscuration(config)
        do_pickle(obsc)
        for i in range(100):
            x = rng.normal(0.0, 1.0)
            y = rng.normal(0.0, 1.0)
            assert obsc.contains(x, y) == (np.hypot(x-cx, y-cy) > r)


@timer
def test_ObscCompound():
    rng = np.random.default_rng(577215)
    size = 10_000

    for i in range(100):
        rx = rng.normal(0.0, 1.0)
        ry = rng.normal(0.0, 1.0)
        w = rng.uniform(0.5, 2.5)
        h = rng.uniform(0.5, 2.5)
        th = rng.uniform(0.0, np.pi)
        rect = batoid.ObscRectangle(w, h, rx, ry, th)

        cx = rng.normal(0.0, 1.0)
        cy = rng.normal(0.0, 1.0)
        r = rng.uniform(0.5, 1.5)
        circ = batoid.ObscCircle(r, cx, cy)

        union = batoid.ObscUnion([rect, circ])
        do_pickle(union)
        union2 = batoid.ObscUnion([circ, rect])
        assert union == union2  # commutative!
        assert hash(union) == hash(union2)
        intersection = batoid.ObscIntersection([rect, circ])
        do_pickle(intersection)
        intersection2 = batoid.ObscIntersection([circ, rect])
        assert intersection == intersection2
        assert hash(intersection) == hash(intersection2)

        for i in range(100):
            x = rng.normal(0.0, 2.0)
            y = rng.normal(0.0, 2.0)
            assert (union.contains(x, y) == union2.contains(x, y)
                    == (rect.contains(x, y) or circ.contains(x, y)))
            assert (intersection.contains(x, y) == intersection2.contains(x, y)
                    == (rect.contains(x, y) and circ.contains(x, y)))

        x = rng.normal(0.0, 2.0, size=size)
        y = rng.normal(0.0, 2.0, size=size)
        np.testing.assert_array_equal(
            union.contains(x, y),
            union2.contains(x, y)
        )
        np.testing.assert_array_equal(
            union.contains(x, y),
            rect.contains(x, y) | circ.contains(x, y)
        )
        np.testing.assert_array_equal(
            intersection.contains(x, y),
            intersection2.contains(x, y)
        )
        np.testing.assert_array_equal(
            intersection.contains(x, y),
            rect.contains(x, y) & circ.contains(x, y)
        )

        rv = batoid.RayVector(x, y, 0.0, 0.0, 0.0, 0.0)
        batoid.obscure(union, rv)
        np.testing.assert_array_equal(
            union.contains(x, y),
            rv.vignetted
        )

        rv = batoid.RayVector(x, y, 0.0, 0.0, 0.0, 0.0)
        batoid.obscure(intersection, rv)
        np.testing.assert_array_equal(
            intersection.contains(x, y),
            rv.vignetted
        )

    with np.testing.assert_raises(ValueError):
        batoid.ObscUnion()
    with np.testing.assert_raises(ValueError):
        batoid.ObscIntersection()


@timer
def test_ne():
    objs = [
        batoid.ObscCircle(1.0),
        batoid.ObscCircle(2.0),
        batoid.ObscCircle(1.0, 0.1, 0.1),
        batoid.ObscAnnulus(0.0, 1.0),
        batoid.ObscAnnulus(0.1, 1.0),
        batoid.ObscAnnulus(0.1, 1.0, 0.1, 0.1),
        batoid.ObscRectangle(1.0, 2.0),
        batoid.ObscRectangle(1.0, 2.0, 0.1, 0.1),
        batoid.ObscRectangle(1.0, 2.0, 0.1, 0.1, 1.0),
        batoid.ObscRay(1.0, 2.0),
        batoid.ObscRay(1.0, 2.0, 0.1, 0.1),
        batoid.ObscNegation(batoid.ObscCircle(1.0)),
        batoid.ObscPolygon([0,1,1,0],[0,0,1,1]),
        batoid.ObscUnion([batoid.ObscCircle(1.0)]),
        batoid.ObscUnion([
            batoid.ObscCircle(1.0),
            batoid.ObscCircle(2.0)
        ]),
        batoid.ObscUnion([
            batoid.ObscCircle(1.0),
            batoid.ObscCircle(2.2)
        ]),
        batoid.ObscUnion([
            batoid.ObscCircle(1.0),
            batoid.ObscCircle(2.2),
            batoid.ObscAnnulus(1.0, 2.0)
        ]),
        batoid.ObscIntersection([batoid.ObscCircle(1.0)]),
        batoid.ObscIntersection([
            batoid.ObscCircle(1.0),
            batoid.ObscCircle(2.0)
        ]),
        batoid.ObscIntersection([
            batoid.ObscCircle(1.0),
            batoid.ObscCircle(2.2)
        ]),
        batoid.ObscIntersection([
            batoid.ObscCircle(1.0),
            batoid.ObscCircle(2.2),
            batoid.ObscAnnulus(1.0, 2.0)
        ]),
    ]
    all_obj_diff(objs)


if __name__ == '__main__':
    test_ObscCircle()
    test_ObscAnnulus()
    test_ObscRectangle()
    test_ObscRay()
    test_ObscPolygon()
    test_ObscNegation()
    test_ObscCompound()
    test_ne()
