import numpy as np
import batoid
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_table():
    import random
    random.seed(5)

    for j in range(10):
        m = random.uniform(3,4)
        b = random.uniform(10, 20)
        f = lambda x: m*x + b

        x = np.array([random.uniform(0, 1) for i in range(10)])
        x.sort()
        y = f(x)
        table = batoid.Table(x, y, batoid.Table.Interpolant.linear)

        xtest = np.array([random.uniform(min(x), max(x)) for i in range(10)])
        ytruth = f(xtest)
        yinterp = np.array([table(xt) for xt in xtest])

        np.testing.assert_allclose(x, table.args, rtol=0, atol=1e-14)
        np.testing.assert_allclose(y, table.vals, rtol=0, atol=1e-14)
        np.testing.assert_allclose(ytruth, yinterp, rtol=0, atol=1e-14)
        assert table.interp == batoid.Table.Interpolant.linear

        do_pickle(table)


@timer
def test_ne():
    objs = [
        batoid.Table([0,1], [1,2], batoid.Table.Interpolant.linear),
        batoid.Table([0,1], [1,2], batoid.Table.Interpolant.ceil),
        batoid.Table([0,1], [2,1], batoid.Table.Interpolant.linear),
        batoid.Table([], [], batoid.Table.Interpolant.linear)
    ]
    all_obj_diff(objs)

if __name__ == '__main__':
    test_table()
    test_ne()
