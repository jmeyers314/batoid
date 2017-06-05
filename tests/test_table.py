import numpy as np
import jtrace
from test_helpers import timer


@timer
def test_table():
    import random
    random.seed(5)

    f = lambda x: 3*x+6

    x = np.array([random.uniform(0, 1) for i in range(10)])
    x.sort()
    y = f(x)
    table = jtrace.Table(x, y, jtrace.Table.Interpolant.linear)
    xtest = np.array([random.uniform(min(x), max(x)) for i in range(10)])
    ytruth = f(xtest)
    yinterp = np.array([table(xt) for xt in xtest])

    np.testing.assert_allclose(x, table.args, rtol=0, atol=1e-14)
    np.testing.assert_allclose(y, table.vals, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ytruth, yinterp, rtol=0, atol=1e-14)
    assert table.interp == jtrace.Table.Interpolant.linear

if __name__ == '__main__':
    test_table()
