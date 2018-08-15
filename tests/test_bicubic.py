import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_properties():
    np.random.seed(5)
    for _ in range(100):
        xs = np.arange(10)
        ys = np.arange(10)
        zs = np.random.uniform(0, 1, size=(10, 10))
        bc = batoid.Bicubic(xs, ys, zs)
        np.testing.assert_array_equal(xs, bc.xs)
        np.testing.assert_array_equal(ys, bc.ys)
        np.testing.assert_array_equal(zs, bc.zs)
        do_pickle(bc)


@timer
def test_sag():
    np.random.seed(57)
    # Make some functions that should be trivially interpolatable.
    def f1(x, y):
        return x+y
    def f2(x, y):
        return x-y
    def f3(x, y):
        return x**2
    def f4(x, y):
        return x*y + x - y + 2
    def f5(x, y):
        return x**2 + y**2 + x*y - x - y - 3
    def f6(x, y):
        return x**2*y - y**2*x + 3*x
    xs = np.linspace(0, 10, 1000)
    ys = np.linspace(0, 10, 1000)

    # Ought to be able to interpolate anywhere in [1,8]x[1,8]
    xtest = np.random.uniform(1, 8, size=1000)
    ytest = np.random.uniform(1, 8, size=1000)

    for f in [f1, f2, f3, f4, f5, f6]:
        zs = f(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(xs, ys, zs)

        np.testing.assert_allclose(
            f(xtest, ytest),
            bc.sag(xtest, ytest),
            atol=0, rtol=1e-10
        )


@timer
def test_normal():
    np.random.seed(577)
    # Work out some normals that are trivially interpolatable
    def f1(x, y):
        return x+y
    def df1dx(x, y):
        return np.ones_like(x)
    def df1dy(x, y):
        return np.ones_like(x)

    def f2(x, y):
        return x-y
    def df2dx(x, y):
        return np.ones_like(x)
    def df2dy(x, y):
        return -np.ones_like(x)

    def f3(x, y):
        return x**2
    def df3dx(x, y):
        return 2*x
    def df3dy(x, y):
        return np.zeros_like(x)

    def f4(x, y):
        return x**2*y + 2*y
    def df4dx(x, y):
        return 2*x*y
    def df4dy(x, y):
        return x**2 + 2

    def f5(x, y):
        return x**2*y - y**2*x + 3*x - 2
    def df5dx(x, y):
        return 2*x*y - y**2 + 3
    def df5dy(x, y):
        return x**2 - 2*y*x

    xs = np.linspace(0, 10, 1000)
    ys = np.linspace(0, 10, 1000)

    xtest = np.random.uniform(1, 8, size=1000)
    ytest = np.random.uniform(1, 8, size=1000)

    for f, dfdx, dfdy in zip(
        [f1, f2, f3, f4, f5],
        [df1dx, df2dx, df3dx, df4dx, df5dx],
        [df1dy, df2dy, df3dy, df4dy, df5dy]):

        zs = f(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(xs, ys, zs)
        bcn = bc.normal(xtest, ytest)

        arr = np.vstack([
            -dfdx(xtest, ytest),
            -dfdy(xtest, ytest),
            np.ones(len(xtest))
        ]).T
        arr /= np.sqrt(np.sum(arr**2, axis=1))[:,None]

        np.testing.assert_allclose(
            bcn,
            arr,
            atol=1e-12, rtol=0
        )


@timer
def test_intersect():
    np.random.seed(5772)
    def f(x, y):
        return x**2*y - y**2*x + 3*x - 2 + np.sin(y)*np.cos(x)**2

    xs = np.linspace(0, 1, 1000)
    ys = np.linspace(0, 1, 1000)

    zs = f(*np.meshgrid(xs, ys))
    bc = batoid.Bicubic(xs, ys, zs)

    for _ in range(1000):
        # If we shoot rays straight up, then it's easy to predict the
        # intersection points.
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)

        r0 = batoid.Ray(x, y, -10, 0, 0, 1, 0)
        r = bc.intersect(r0)

        np.testing.assert_allclose(r.r[0], x)
        np.testing.assert_allclose(r.r[1], y)
        np.testing.assert_allclose(r.r[2], bc.sag(x, y), rtol=0, atol=1e-9)


@timer
def test_ne():
    xs1 = np.linspace(0, 1, 10)
    xs2 = np.linspace(0, 1, 11)
    ys1 = np.linspace(0.1, 1, 10)
    ys2 = np.linspace(0.1, 0.9, 11)

    def f1(x, y):
        return x+y
    def f2(x, y):
        return x-y

    zs1 = f1(*np.meshgrid(xs1, ys1))
    zs2 = f2(*np.meshgrid(xs1, ys1))

    objs = [
        batoid.Bicubic(xs1, ys1, zs1),
        batoid.Bicubic(xs1, ys1, zs2),
        batoid.Bicubic(xs2, ys1, zs1),
        batoid.Bicubic(xs1, ys2, zs1),
    ]
    all_obj_diff(objs)


if __name__ == '__main__':
    test_properties()
    test_sag()
    test_normal()
    test_intersect()
    test_ne()
