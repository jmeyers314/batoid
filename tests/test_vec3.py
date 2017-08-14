import math
import batoid
from test_helpers import isclose, timer


@timer
def test_DotProduct():
    import random
    random.seed(5)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)
        z1 = random.gauss(mu=10.34, sigma=13.0)

        x2 = random.gauss(mu=0.13, sigma=3.3)
        y2 = random.gauss(mu=-0.5, sigma=1.21)
        z2 = random.gauss(mu=1.34, sigma=3.01)

        vec1 = batoid.Vec3(x1, y1, z1)
        vec2 = batoid.Vec3(x2, y2, z2)

        assert isclose(batoid.DotProduct(vec1, vec2),
                       x1*x2 + y1*y2 + z1*z2)
        assert isclose(batoid.DotProduct(vec1, vec2),
                       vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z)


@timer
def test_CrossProduct():
    import random
    random.seed(57)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)
        z1 = random.gauss(mu=10.34, sigma=13.0)

        x2 = random.gauss(mu=0.13, sigma=3.3)
        y2 = random.gauss(mu=-0.5, sigma=1.21)
        z2 = random.gauss(mu=1.34, sigma=3.01)

        vec1 = batoid.Vec3(x1, y1, z1)
        vec2 = batoid.Vec3(x2, y2, z2)

        # Cross product is orthogonal to input vectors
        assert isclose(
                batoid.DotProduct(batoid.CrossProduct(vec1, vec2), vec1),
                0.0,
                abs_tol=1e-12)
        assert isclose(
                batoid.DotProduct(batoid.CrossProduct(vec1, vec2), vec2),
                0.0,
                abs_tol=1e-12)

        # Auto cross product is "zero"
        assert isclose(
                batoid.CrossProduct(vec1, vec1).Magnitude(), 0.0,
                abs_tol=1e-12)

        # Magnitude of cross product of orthogonal vectors is product of magnitudes
        vec3 = batoid.CrossProduct(vec1, vec2)
        assert isclose(
                batoid.CrossProduct(vec1, vec3).Magnitude(),
                vec1.Magnitude() * vec3.Magnitude())
        assert isclose(
                batoid.CrossProduct(vec2, vec3).Magnitude(),
                vec2.Magnitude() * vec3.Magnitude())


@timer
def test_Magnitude():
    import random
    random.seed(577)
    for i in range(100):
        x = random.gauss(mu=0.1, sigma=0.3)
        y = random.gauss(mu=-0.3, sigma=1.1)
        z = random.gauss(mu=10.34, sigma=13.0)

        vec = batoid.Vec3(x, y, z)
        assert isclose(vec.Magnitude(), math.sqrt(x*x + y*y + z*z))
        assert isclose(vec.MagnitudeSquared(), x*x + y*y + z*z)
        assert isclose(vec.UnitVec3().Magnitude(), 1.0)


@timer
def test_add():
    import random
    random.seed(5772)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)
        z1 = random.gauss(mu=10.34, sigma=13.0)

        x2 = random.gauss(mu=0.13, sigma=3.3)
        y2 = random.gauss(mu=-0.5, sigma=1.21)
        z2 = random.gauss(mu=1.34, sigma=3.01)

        vec1 = batoid.Vec3(x1, y1, z1)
        vec2 = batoid.Vec3(x2, y2, z2)

        vec3 = vec1 + vec2
        assert isclose(vec1.x+vec2.x, vec3.x)
        assert isclose(vec1.y+vec2.y, vec3.y)
        assert isclose(vec1.z+vec2.z, vec3.z)

        vec1 += vec2
        assert isclose(x1+x2, vec1.x)
        assert isclose(y1+y2, vec1.y)
        assert isclose(z1+z2, vec1.z)


@timer
def test_sub():
    import random
    random.seed(57721)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)
        z1 = random.gauss(mu=10.34, sigma=13.0)

        x2 = random.gauss(mu=0.13, sigma=3.3)
        y2 = random.gauss(mu=-0.5, sigma=1.21)
        z2 = random.gauss(mu=1.34, sigma=3.01)

        vec1 = batoid.Vec3(x1, y1, z1)
        vec2 = batoid.Vec3(x2, y2, z2)

        vec3 = vec1 - vec2
        assert isclose(vec1.x-vec2.x, vec3.x)
        assert isclose(vec1.y-vec2.y, vec3.y)
        assert isclose(vec1.z-vec2.z, vec3.z)

        vec1 -= vec2
        assert isclose(x1-x2, vec1.x)
        assert isclose(y1-y2, vec1.y)
        assert isclose(z1-z2, vec1.z)


@timer
def test_mul():
    import random
    random.seed(577215)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)
        z1 = random.gauss(mu=10.34, sigma=13.0)

        m = random.gauss(mu=2.3, sigma=0.31)

        vec1 = batoid.Vec3(x1, y1, z1)
        vec2 = vec1 * m
        assert isclose(vec1.x*m, vec2.x, rel_tol=1e-7)
        assert isclose(vec1.y*m, vec2.y, rel_tol=1e-7)
        assert isclose(vec1.z*m, vec2.z, rel_tol=1e-7)

        vec1 *= m
        assert isclose(x1*m, vec1.x, rel_tol=1e-7)
        assert isclose(y1*m, vec1.y, rel_tol=1e-7)
        assert isclose(z1*m, vec1.z, rel_tol=1e-7)


@timer
def test_div():
    import random
    random.seed(5772156)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)
        z1 = random.gauss(mu=10.34, sigma=13.0)

        m = random.gauss(mu=2.3, sigma=0.31)

        vec1 = batoid.Vec3(x1, y1, z1)
        vec2 = vec1/m
        assert isclose(vec1.x/m, vec2.x, rel_tol=1e-7)
        assert isclose(vec1.y/m, vec2.y, rel_tol=1e-7)
        assert isclose(vec1.z/m, vec2.z, rel_tol=1e-7)

        vec1 /= m
        assert isclose(x1/m, vec1.x, rel_tol=1e-7)
        assert isclose(y1/m, vec1.y, rel_tol=1e-7)
        assert isclose(z1/m, vec1.z, rel_tol=1e-7)


@timer
def test_eq():
    import random
    random.seed(57721566)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)
        z1 = random.gauss(mu=10.34, sigma=13.0)

        vec1 = batoid.Vec3(x1, y1, z1)
        vec2 = batoid.Vec3(x1, y1, z1)

        assert vec1 == vec2


@timer
def test_ne():
    import random
    random.seed(577215664)
    for i in range(100):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)
        z1 = random.gauss(mu=10.34, sigma=13.0)

        vec1 = batoid.Vec3(x1, y1, z1)
        vec2 = batoid.Vec3(x1+1e-12, y1, z1)
        vec3 = batoid.Vec3(x1, y1+1e-12, z1)
        vec4 = batoid.Vec3(x1, y1, z1+1e-12)

        assert vec1 != vec2
        assert vec1 != vec3
        assert vec1 != vec4


@timer
def testRotVec():
    import random
    import math
    random.seed(5772156649)
    for i in range(1000):
        x1 = random.gauss(mu=0.1, sigma=0.3)
        y1 = random.gauss(mu=-0.3, sigma=1.1)
        z1 = random.gauss(mu=10.34, sigma=13.0)

        # Check identity rotation
        ident = batoid.Rot3()
        vec1 = batoid.Vec3(x1, y1, z1)
        vec2 = batoid.RotVec(ident, vec1)
        vec3 = batoid.UnRotVec(ident, vec1)
        assert vec1 == vec2
        assert vec1 == vec3
        assert ident.determinant() == 1.0

        # Check random rotation
        th1 = random.uniform(0, 2*math.pi)
        th2 = random.uniform(0, 2*math.pi)
        th3 = random.uniform(0, 2*math.pi)
        sth1, cth1 = math.sin(th1), math.cos(th1)
        sth2, cth2 = math.sin(th2), math.cos(th2)
        sth3, cth3 = math.sin(th3), math.cos(th3)
        R1 = batoid.Rot3([1, 0, 0, 0, cth1, -sth1, 0, sth1, cth1])
        R2 = batoid.Rot3([cth2, 0, sth2, 0, 1, 0, -sth2, 0, cth2])
        R3 = batoid.Rot3([cth3, -sth3, 0, sth3, cth3, 0, 0, 0, 1])
        vec2 = batoid.RotVec(R1, batoid.RotVec(R2, batoid.RotVec(R3, vec1)))

        assert isclose(vec1.Magnitude(), vec2.Magnitude())
        # Unrotate...
        vec3 = batoid.UnRotVec(R3, batoid.UnRotVec(R2, batoid.UnRotVec(R1, vec2)))
        assert isclose(vec1.x, vec3.x)
        assert isclose(vec1.y, vec3.y)
        assert isclose(vec1.z, vec3.z)


@timer
def test_determinant():
    import random
    random.seed(57721566490)
    import numpy as np

    for i in range(1000):
        R = batoid.Rot3([random.gauss(0,1) for i in range(9)])
        assert isclose(np.linalg.det(R), R.determinant())


if __name__ == '__main__':
    test_DotProduct()
    test_CrossProduct()
    test_Magnitude()
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_eq()
    test_ne()
    testRotVec()
    test_determinant()
