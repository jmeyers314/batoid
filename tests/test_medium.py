import os
import jtrace
import numpy as np

datadir = os.path.join(os.path.dirname(__file__), 'data', 'media')

def test_const_medium():
    import random
    random.seed(5)

    n = 1.4
    const_medium = jtrace.ConstMedium(n)
    for i in range(100):
        w = random.uniform(500.0, 1000.0)
        assert const_medium.getN(w) == n

def test_table_medium():
    import random
    random.seed(57)

    filename = os.path.join(datadir, "silica_dispersion.txt")
    wave, n = np.genfromtxt(filename).T
    table = jtrace.Table(wave, n, jtrace.Table.Interpolant.linear)
    table_medium = jtrace.TableMedium(table)
    for i in range(100):
        w = random.uniform(0.3, 1.2)
        assert table_medium.getN(w) == table(w)


if __name__ == '__main__':
    test_const_medium()
    test_table_medium()
