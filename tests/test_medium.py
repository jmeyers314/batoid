import os
import jtrace
import numpy as np
from test_helpers import isclose, timer


@timer
def test_const_medium():
    import random
    random.seed(5)

    n = 1.4
    const_medium = jtrace.ConstMedium(n)
    for i in range(100):
        w = random.uniform(0.5, 1.0)
        assert const_medium.getN(w) == n


@timer
def test_table_medium():
    import random
    random.seed(57)

    # File from phosim
    filename = os.path.join(jtrace.datadir, "media", "silica_dispersion.txt")
    wave, n = np.genfromtxt(filename).T
    wave *= 1e-6    # microns -> meters
    table = jtrace.Table(wave, n, jtrace.Table.Interpolant.linear)
    table_medium = jtrace.TableMedium(table)
    for i in range(100):
        w = random.uniform(0.3e-6, 1.2e-6)
        assert table_medium.getN(w) == table(w)


@timer
def test_silica_sellmeier_table():
    import random
    import time
    random.seed(577)

    # File from phosim
    filename = os.path.join(jtrace.datadir, "media", "silica_dispersion.txt")
    wave, n = np.genfromtxt(filename).T
    wave *= 1e-6    # microns -> meters
    table = jtrace.Table(wave, n, jtrace.Table.Interpolant.linear)
    table_silica = jtrace.TableMedium(table)

    # Coefficients from
    # https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
    sellmeier_silica = jtrace.SellmeierMedium(
            0.6961663, 0.4079426, 0.8974794,
            0.0684043**2, 0.1162414**2, 9.896161**2)

    # Making this a timing test too for fun
    ws = []
    for i in range(100000):
        ws.append(random.uniform(0.3e-6, 1.2e-6))
    table_n = []
    sellmeier_n = []
    t0 = time.time()
    for w in ws:
        table_n.append(table_silica.getN(w))
    t1 = time.time()
    for w in ws:
        sellmeier_n.append(sellmeier_silica.getN(w))
    t2 = time.time()
    print("TableMedium took {} s".format(t1-t0))
    print("SellmeierMedium took {} s".format(t2-t1))
    np.testing.assert_allclose(table_n, sellmeier_n, atol=1e-6, rtol=0)


@timer
def test_air():
    # Just spot check some comparisons with GalSim.
    ws = [0.3, 0.5, 0.7, 0.9, 1.1]
    gsn = [ 1.00019563,  1.00018713,  1.00018498,  1.00018412,  1.00018369]
    air = jtrace.Air()
    for w, n in zip(ws, gsn):
        assert isclose(n, air.getN(w*1e-6), abs_tol=1e-8, rel_tol=0)


if __name__ == '__main__':
    test_const_medium()
    test_table_medium()
    test_silica_sellmeier_table()
    test_air()
