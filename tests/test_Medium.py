import os
import numpy as np
import batoid
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_ConstMedium():
    rng = np.random.default_rng(5)
    for i in range(100):
        n = rng.uniform(1.0, 2.0)
        const_medium = batoid.ConstMedium(n)
        wavelengths = rng.uniform(300e-9, 1100e-9, size=100)
        for w in wavelengths:
            assert const_medium.getN(w) == n
        np.testing.assert_array_equal(
            const_medium.getN(wavelengths),
            np.full_like(wavelengths, n)
        )
    do_pickle(const_medium)


@timer
def test_SellmeierMedium():
    rng = np.random.default_rng(57)
    # Silica coefficients
    # https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794
    C1 = 0.00467914825849
    C2 = 0.013512063073959999
    C3 = 97.93400253792099
    silica = batoid.SellmeierMedium([B1, B2, B3, C1, C2, C3])

    wavelengths = rng.uniform(300e-9, 1100e-9, size=1000)
    indices = silica.getN(wavelengths)
    for w, index in zip(wavelengths, indices):
        assert silica.getN(w) == index

    # CSV also from refractiveindex.info
    filename = os.path.join(
        os.path.dirname(__file__),
        "testdata",
        "silicaDispersion.csv"
    )
    wmicron, ncsv = np.loadtxt(filename, delimiter=',').T
    n = silica.getN(wmicron*1e-6)
    np.testing.assert_allclose(n, ncsv, atol=0, rtol=1e-13)


# @timer
# def test_table_medium():
#     import random
#     random.seed(57)
#
#     # File from phosim
#     filename = os.path.join(batoid.datadir, "media", "silica_dispersion.txt")
#     wave, n = np.genfromtxt(filename).T
#     wave *= 1e-6    # microns -> meters
#     table = batoid.Table(wave, n, 'linear')
#     table_medium = batoid.TableMedium(table)
#     for i in range(100):
#         w = random.uniform(0.3e-6, 1.2e-6)
#         assert table_medium.getN(w) == table(w)
#     assert table_medium.table == table
#     do_pickle(table_medium)
#
#
# @timer
# def test_silica_sellmeier_table():
#     import random
#     import time
#     random.seed(577)
#
#     # File from phosim
#     filename = os.path.join(batoid.datadir, "media", "silica_dispersion.txt")
#     wave, n = np.genfromtxt(filename).T
#     wave *= 1e-6    # microns -> meters
#     table = batoid.Table(wave, n, 'linear')
#     table_silica = batoid.TableMedium(table)
#
#     # Coefficients from
#     # https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
#     sellmeier_silica = batoid.SellmeierMedium(
#             0.6961663, 0.4079426, 0.8974794,
#             0.0684043**2, 0.1162414**2, 9.896161**2)
#
#     # Making this a timing test too for fun
#     ws = []
#     for i in range(100000):
#         ws.append(random.uniform(0.3e-6, 1.2e-6))
#     table_n = []
#     sellmeier_n = []
#     t0 = time.time()
#     for w in ws:
#         table_n.append(table_silica.getN(w))
#     t1 = time.time()
#     for w in ws:
#         sellmeier_n.append(sellmeier_silica.getN(w))
#     t2 = time.time()
#     print("TableMedium took {} s".format(t1-t0))
#     print("SellmeierMedium took {} s".format(t2-t1))
#     np.testing.assert_allclose(table_n, sellmeier_n, atol=1e-6, rtol=0)
#     do_pickle(sellmeier_silica)
#
#
# @timer
# def test_air():
#     # Just spot check some comparisons with GalSim.
#     ws = [0.3, 0.5, 0.7, 0.9, 1.1]
#     gsn = [ 1.00019563,  1.00018713,  1.00018498,  1.00018412,  1.00018369]
#     air = batoid.Air()
#     for w, n in zip(ws, gsn):
#         np.testing.assert_allclose(n, air.getN(w*1e-6), rtol=0, atol=1e-8)
#     do_pickle(air)
#
#
# @timer
# def test_ne():
#     filename = os.path.join(batoid.datadir, "media", "silica_dispersion.txt")
#     wave, n = np.genfromtxt(filename).T
#     wave *= 1e-6    # microns -> meters
#     table = batoid.Table(wave, n, 'linear')
#     table2 = batoid.Table(wave*1.01, n, 'linear')
#
#     objs = [
#         batoid.ConstMedium(1.0),
#         batoid.ConstMedium(1.1),
#         batoid.TableMedium(table),
#         batoid.TableMedium(table2),
#         batoid.SellmeierMedium(
#             0.6961663, 0.4079426, 0.8974794,
#             0.0684043**2, 0.1162414**2, 9.896161**2),
#         batoid.SellmeierMedium(
#             0.4079426, 0.6961663, 0.8974794,
#             0.0684043**2, 0.1162414**2, 9.896161**2),
#         batoid.Air(),
#         batoid.Air(pressure=100)
#     ]
#     all_obj_diff(objs)


if __name__ == '__main__':
    test_ConstMedium()
    test_SellmeierMedium()
    # test_table_medium()
    # test_silica_sellmeier_table()
    # test_air()
    # test_ne()
