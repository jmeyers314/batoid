import math
import batoid
from test_helpers import isclose, timer, test_pickle


@timer
def test_DotProduct():
    import random
    random.seed(5)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)

        x2 = random.gauss(mu=0.13, sigma=3.3)
        y2 = random.gauss(mu=-0.5, sigma=1.21)

        vec1 = batoid.Vec2(x1, y1)
        vec2 = batoid.Vec2(x2, y2)

        assert isclose(batoid.DotProduct(vec1, vec2),
                       x1*x2 + y1*y2)
        assert isclose(batoid.DotProduct(vec1, vec2),
                       vec1.x*vec2.x + vec1.y*vec2.y)

        test_pickle(vec1)


@timer
def test_Magnitude():
    import random
    random.seed(57)
    for i in range(100):
        x = random.gauss(mu=0.1, sigma=0.3)
        y = random.gauss(mu=-0.3, sigma=1.1)

        vec = batoid.Vec2(x, y)
        assert isclose(vec.Magnitude(), math.sqrt(x*x + y*y))
        assert isclose(vec.MagnitudeSquared(), x*x + y*y)
        assert isclose(vec.UnitVec2().Magnitude(), 1.0)


@timer
def test_add():
    import random
    random.seed(577)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)

        x2 = random.gauss(mu=0.13, sigma=3.3)
        y2 = random.gauss(mu=-0.5, sigma=1.21)

        vec1 = batoid.Vec2(x1, y1)
        vec2 = batoid.Vec2(x2, y2)

        Vec2 = vec1 + vec2
        assert isclose(vec1.x+vec2.x, Vec2.x)
        assert isclose(vec1.y+vec2.y, Vec2.y)

        vec1 += vec2
        assert isclose(x1+x2, vec1.x)
        assert isclose(y1+y2, vec1.y)


@timer
def test_sub():
    import random
    random.seed(5772)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)

        x2 = random.gauss(mu=0.13, sigma=3.3)
        y2 = random.gauss(mu=-0.5, sigma=1.21)

        vec1 = batoid.Vec2(x1, y1)
        vec2 = batoid.Vec2(x2, y2)

        Vec2 = vec1 - vec2
        assert isclose(vec1.x-vec2.x, Vec2.x)
        assert isclose(vec1.y-vec2.y, Vec2.y)

        vec1 -= vec2
        assert isclose(x1-x2, vec1.x)
        assert isclose(y1-y2, vec1.y)


@timer
def test_mul():
    import random
    random.seed(57721)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)

        m = random.gauss(mu=2.3, sigma=0.31)

        vec1 = batoid.Vec2(x1, y1)
        vec2 = vec1 * m
        assert isclose(vec1.x*m, vec2.x, rel_tol=1e-7)
        assert isclose(vec1.y*m, vec2.y, rel_tol=1e-7)

        vec1 *= m
        assert isclose(x1*m, vec1.x, rel_tol=1e-7)
        assert isclose(y1*m, vec1.y, rel_tol=1e-7)


@timer
def test_div():
    import random
    random.seed(577215)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)

        m = random.gauss(mu=2.3, sigma=0.31)

        vec1 = batoid.Vec2(x1, y1)
        vec2 = vec1/m
        assert isclose(vec1.x/m, vec2.x, rel_tol=1e-7)
        assert isclose(vec1.y/m, vec2.y, rel_tol=1e-7)

        vec1 /= m
        assert isclose(x1/m, vec1.x, rel_tol=1e-7)
        assert isclose(y1/m, vec1.y, rel_tol=1e-7)


@timer
def test_eq():
    import random
    random.seed(5772156)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)

        vec1 = batoid.Vec2(x1, y1)
        vec2 = batoid.Vec2(x1, y1)

        assert vec1 == vec2


@timer
def test_ne():
    import random
    random.seed(57721566)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)

        vec1 = batoid.Vec2(x1, y1)
        vec2 = batoid.Vec2(x1+1e-12, y1)
        vec3 = batoid.Vec2(x1, y1+1e-12)

        assert vec1 != vec2
        assert vec1 != vec3


@timer
def testRotVec():
    import random
    import math
    random.seed(577215664)
    for i in range(1000):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)

        # Check identity rotation
        ident = batoid.Rot2()
        vec1 = batoid.Vec2(x1, y1)
        vec2 = batoid.RotVec(ident, vec1)
        vec3 = batoid.UnRotVec(ident, vec1)
        assert vec1 == vec2
        assert vec1 == vec3
        assert ident.determinant() == 1.0

        # Check random rotation
        th = random.uniform(0, 2*math.pi)
        sth, cth = math.sin(th), math.cos(th)
        R = batoid.Rot2([cth, -sth, sth, cth])
        vec2 = batoid.RotVec(R, vec1)

        assert isclose(vec1.Magnitude(), vec2.Magnitude())
        # Unrotate...
        vec3 = batoid.UnRotVec(R, vec2)
        assert isclose(vec1.x, vec3.x)
        assert isclose(vec1.y, vec3.y)

        test_pickle(R)


@timer
def test_determinant():
    import random
    random.seed(5772156649)
    import numpy as np

    for i in range(1000):
        R = batoid.Rot2([random.gauss(0,1) for i in range(4)])
        assert isclose(np.linalg.det(R), R.determinant())


if __name__ == '__main__':
    test_DotProduct()
    test_Magnitude()
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_eq()
    test_ne()
    testRotVec()
    test_determinant()
