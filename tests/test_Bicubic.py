import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff, rays_allclose, init_gpu


@timer
def test_properties():
    rng = np.random.default_rng(5)

    for _ in range(10):
        xs = np.arange(10)
        ys = np.arange(10)
        zs = rng.uniform(0, 1, size=(10, 10))
        bc = batoid.Bicubic(xs, ys, zs)
        np.testing.assert_array_equal(xs, bc.xs)
        np.testing.assert_array_equal(ys, bc.ys)
        np.testing.assert_array_equal(zs, bc.zs)
        do_pickle(bc)

    with np.testing.assert_raises(AssertionError):
        batoid.Bicubic(xs, ys, zs, nanpolicy='Giraffe')

    # Check reprability here for zero nanpolicy
    do_pickle(batoid.Bicubic(xs, ys, zs, nanpolicy='ZERO'))

@timer
def test_sag():
    rng = np.random.default_rng(57)
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
    xtest = rng.uniform(1, 8, size=1000)
    ytest = rng.uniform(1, 8, size=1000)

    for f in [f1, f2, f3, f4, f5, f6]:
        zs = f(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(xs, ys, zs)
        np.testing.assert_allclose(
            f(xtest, ytest),
            bc.sag(xtest, ytest),
            atol=0, rtol=1e-10
        )
        assert bc.sag(xtest[0], ytest[0]) == bc.sag(xtest, ytest)[0]
        assert np.array_equal(
            bc.sag(xtest[::5], ytest[::5]),
            bc.sag(xtest, ytest)[::5]
        )

    # sag returns nan outside of grid domain
    assert np.isnan(bc.sag(-1, -1))
    # or zero if nanpolicy is set to 'zero'
    bc = batoid.Bicubic(xs, ys, zs, nanpolicy='ZERO')
    np.testing.assert_equal(bc.sag(-1, -1), 0.0)


@timer
def test_normal():
    rng = np.random.default_rng(577)
    # Work out some normals that are trivially interpolatable
    def f1(x, y):
        return x+y
    def df1dx(x, y):
        return np.ones_like(x)
    def df1dy(x, y):
        return np.ones_like(x)
    def d2f1dxdy(x, y):
        return np.zeros_like(x)

    def f2(x, y):
        return x-y
    def df2dx(x, y):
        return np.ones_like(x)
    def df2dy(x, y):
        return -np.ones_like(x)
    def d2f2dxdy(x, y):
        return np.zeros_like(x)

    def f3(x, y):
        return x**2
    def df3dx(x, y):
        return 2*x
    def df3dy(x, y):
        return np.zeros_like(x)
    def d2f3dxdy(x, y):
        return np.zeros_like(x)

    def f4(x, y):
        return x**2*y + 2*y
    def df4dx(x, y):
        return 2*x*y
    def df4dy(x, y):
        return x**2 + 2
    def d2f4dxdy(x, y):
        return 2*x

    def f5(x, y):
        return x**2*y - y**2*x + 3*x - 2
    def df5dx(x, y):
        return 2*x*y - y**2 + 3
    def df5dy(x, y):
        return x**2 - 2*y*x
    def d2f5dxdy(x, y):
        return 2*x - 2*y

    xs = np.linspace(0, 10, 1000)
    ys = np.linspace(0, 10, 1000)

    xtest = rng.uniform(1, 8, size=1000)
    ytest = rng.uniform(1, 8, size=1000)

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
        assert np.array_equal(
            bc.normal(xtest[0], ytest[0]),
            bc.normal(xtest, ytest)[0]
        )
        assert np.array_equal(
            bc.normal(xtest[::5], ytest[::5]),
            bc.normal(xtest, ytest)[::5]
        )


    # Ought to be able to interpolate cubics if asserting derivatives
    def f6(x, y):
        return x**3*y - y**3*x + 3*x - 2
    def df6dx(x, y):
        return 3*x**2*y - y**3 + 3
    def df6dy(x, y):
        return x**3 - 3*y**2*x
    def d2f6dxdy(x, y):
        return 3*x**2 - 3*y**2

    for f, dfdx, dfdy, d2fdxdy in zip(
        [f1, f2, f3, f4, f5, f6],
        [df1dx, df2dx, df3dx, df4dx, df5dx, df6dx],
        [df1dy, df2dy, df3dy, df4dy, df5dy, df6dy],
        [d2f1dxdy, d2f2dxdy, d2f3dxdy, d2f4dxdy, d2f5dxdy, d2f6dxdy]):

        zs = f(*np.meshgrid(xs, ys))
        dzdxs = dfdx(*np.meshgrid(xs, ys))
        dzdys = dfdy(*np.meshgrid(xs, ys))
        d2zdxdys = d2fdxdy(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(
            xs, ys, zs,
            dzdxs=dzdxs, dzdys=dzdys, d2zdxdys=d2zdxdys
        )
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

    # normal returns (nan, nan, nan) outside of grid domain
    out = bc.normal(-1, -1)
    assert all(np.isnan(o) for o in out)
    # unless nanpolicy == 'ZERO'
    bc = batoid.Bicubic(
        xs, ys, zs,
        dzdxs=dzdxs, dzdys=dzdys, d2zdxdys=d2zdxdys,
        nanpolicy='ZERO'
    )
    np.testing.assert_equal(
        bc.normal(-1, -1),
        np.array([0.0, 0.0, 1.0])
    )


@timer
def test_intersect():
    rng = np.random.default_rng(5772)
    size = 10_000

    for _ in range(10):
        def f(x, y):
            a = rng.uniform(size=5)
            return (
                a[0]*x**2*y - a[1]*y**2*x + a[2]*3*x - a[3]
                + a[4]*np.sin(y)*np.cos(x)**2
            )

        xs = np.linspace(0, 1, 1000)
        ys = np.linspace(0, 1, 1000)

        zs = f(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(xs, ys, zs)

        bcCoordSys = batoid.CoordSys(origin=[0, 0, -1])
        x = rng.uniform(0.1, 0.9, size=size)
        y = rng.uniform(0.1, 0.9, size=size)
        z = np.full_like(x, -10.0)
        # If we shoot rays straight up, then it's easy to predict the intersection
        vx = np.zeros_like(x)
        vy = np.zeros_like(x)
        vz = np.ones_like(x)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        np.testing.assert_allclose(rv.z, -10.0)
        rv2 = batoid.intersect(bc, rv.copy(), bcCoordSys)
        assert rv2.coordSys == bcCoordSys

        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(
            rv2.z, bc.sag(x, y)-1,
            rtol=0, atol=1e-12
        )

        # Check default intersect coordTransform
        rv2 = rv.copy().toCoordSys(bcCoordSys)
        batoid.intersect(bc, rv2)
        assert rv2.coordSys == bcCoordSys
        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(
            rv2.z, bc.sag(x, y)-1,
            rtol=0, atol=1e-12
        )


@timer
def test_reflect():
    rng = np.random.default_rng(57721)
    size = 10_000

    for _ in range(10):
        def f(x, y):
            a = rng.uniform(size=5)
            return (
                a[0]*x**2*y - a[1]*y**2*x + a[2]*3*x - a[3]
                + a[4]*np.sin(y)*np.cos(x)**2
            )

        xs = np.linspace(0, 1, 1000)
        ys = np.linspace(0, 1, 1000)

        zs = f(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(xs, ys, zs)

        x = rng.uniform(0.1, 0.9, size=size)
        y = rng.uniform(0.1, 0.9, size=size)
        z = np.full_like(x, -10.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.full_like(x, 1)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.reflect(bc, rv.copy())
        rvr2 = bc.reflect(rv.copy())
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = bc.normal(rvr.x, rvr.y)

        # Test law of reflection
        a0 = np.einsum("ad,ad->a", normal, rv.v)[~rvr.failed]
        a1 = np.einsum("ad,ad->a", normal, -rvr.v)[~rvr.failed]
        np.testing.assert_allclose(
            a0, a1,
            rtol=0, atol=1e-12
        )

        # Test that rv.v, rvr.v and normal are all in the same plane
        np.testing.assert_allclose(
            np.einsum(
                "ad,ad->a",
                np.cross(normal, rv.v),
                rv.v
            )[~rvr.failed],
            0.0,
            rtol=0, atol=1e-12
        )


@timer
def test_refract():
    rng = np.random.default_rng(577215)
    size = 10_000

    for _ in range(10):
        def f(x, y):
            a = rng.uniform(size=5)
            return (
                a[0]*x**2*y - a[1]*y**2*x + a[2]*3*x - a[3]
                + a[4]*np.sin(y)*np.cos(x)**2
            )

        xs = np.linspace(0, 1, 1000)
        ys = np.linspace(0, 1, 1000)

        zs = f(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(xs, ys, zs)

        m0 = batoid.ConstMedium(rng.normal(1.2, 0.01))
        m1 = batoid.ConstMedium(rng.normal(1.3, 0.01))

        x = rng.uniform(0.1, 0.9, size=size)
        y = rng.uniform(0.1, 0.9, size=size)
        z = np.full_like(x, -10.0)
        vx = rng.uniform(-1e-5, 1e-5, size=size)
        vy = rng.uniform(-1e-5, 1e-5, size=size)
        vz = np.sqrt(1-vx*vx-vy*vy)/m0.n
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        rvr = batoid.refract(bc, rv.copy(), m0, m1)
        rvr2 = bc.refract(rv.copy(), m0, m1)
        np.testing.assert_array_equal(rvr.failed, rvr2.failed)
        rays_allclose(rvr, rvr2)
        # print(f"{np.sum(rvr.failed)/len(rvr)*100:.2f}% failed")
        normal = bc.normal(rvr.x, rvr.y)

        # Test Snell's law
        s0 = np.sum(np.cross(normal, rv.v*m0.n)[~rvr.failed], axis=-1)
        s1 = np.sum(np.cross(normal, rvr.v*m1.n)[~rvr.failed], axis=-1)
        np.testing.assert_allclose(
            m0.n*s0, m1.n*s1,
            rtol=0, atol=1e-9
        )

        # Test that rv.v, rvr.v and normal are all in the same plane
        np.testing.assert_allclose(
            np.einsum(
                "ad,ad->a",
                np.cross(normal, rv.v),
                rv.v
            )[~rvr.failed],
            0.0,
            rtol=0, atol=1e-12
        )


@timer
def test_asphere_approximation():
    rng = np.random.default_rng(5772156)

    xs = np.linspace(-1, 1, 1000)
    ys = np.linspace(-1, 1, 1000)
    xtest = np.random.uniform(-0.9, 0.9, size=1000)
    ytest = np.random.uniform(-0.9, 0.9, size=1000)

    for i in range(10):
        R = rng.normal(20.0, 1.0)
        conic = rng.uniform(-2.0, 1.0)
        ncoef = rng.choice(4)
        coefs = [rng.normal(0, 1e-10) for i in range(ncoef)]
        asphere = batoid.Asphere(R, conic, coefs)
        zs = asphere.sag(*np.meshgrid(xs, ys))
        bc = batoid.Bicubic(xs, ys, zs)

        np.testing.assert_allclose(
            asphere.sag(xtest, ytest),
            bc.sag(xtest, ytest),
            atol=1e-12, rtol=0.0
        )

        np.testing.assert_allclose(
            asphere.normal(xtest, ytest),
            bc.normal(xtest, ytest),
            atol=1e-9, rtol=0
        )


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
        batoid.Bicubic(xs1, ys2, zs1, zs1, zs1, zs1),
        batoid.Bicubic(xs1, ys2, zs1, zs1, zs1, zs2),
        batoid.Bicubic(xs1, ys2, zs1, zs1, zs2, zs2),
        batoid.Bicubic(xs1, ys2, zs1, zs2, zs2, zs2),
        batoid.Bicubic(xs1, ys2, zs1, zs2, zs2, zs2, nanpolicy='ZERO')
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    xs = np.linspace(-1, 1, 10)
    ys = np.linspace(-1, 1, 10)
    def f(x, y):
        return x+y
    zs = f(*np.meshgrid(xs, ys))
    bc = batoid.Bicubic(xs, ys, zs)

    rv = batoid.RayVector(0, 10, 0, 0, 0, -1)  # Too far to side
    rv2 = batoid.intersect(bc, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([True]))
    # This one passes
    rv = batoid.RayVector(0, 0, 0, 0, 0, -1)
    rv2 = batoid.intersect(bc, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([False]))


if __name__ == '__main__':
    init_gpu()
    test_properties()
    test_sag()
    test_normal()
    test_intersect()
    test_reflect()
    test_refract()
    test_asphere_approximation()
    test_ne()
    test_fail()
