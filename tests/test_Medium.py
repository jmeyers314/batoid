from batoid.medium import SumitaMedium, TableMedium
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
def test_TableMedium():
    rng = np.random.default_rng(57)
    for i in range(100):
        ws = np.linspace(300e-9, 1100e-9, 6)
        ns = rng.uniform(1.0, 1.5, size=6)
        table_medium = batoid.TableMedium(ws, ns)
        testws = rng.uniform(300e-9, 1100e-9, size=100)
        for w in testws:
            np.testing.assert_allclose(
                table_medium.getN(w),
                np.interp(w, ws, ns),
                rtol=0, atol=1e-14
            )
        np.testing.assert_allclose(
            table_medium.getN(testws),
            np.interp(testws, ws, ns),
            rtol=0, atol=1e-14
        )
    do_pickle(table_medium)

    # Test load from file
    filename = os.path.join(
        os.path.dirname(__file__),
        "testdata",
        "silicaDispersion.csv"
    )
    medium1 = batoid.TableMedium.fromTxt(
        filename, delimiter=','
    )
    wmicron, ncsv = np.loadtxt(filename, delimiter=',').T
    medium2 = batoid.TableMedium(wmicron*1e-6, ncsv)
    assert medium1 == medium2

    # Load from datadir
    medium3 = batoid.TableMedium.fromTxt("silica_dispersion.txt")

    # Raises on bad input
    with np.testing.assert_raises(FileNotFoundError):
        medium4 = batoid.TableMedium.fromTxt("blah.txt")


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
    silica2 = batoid.SellmeierMedium(B1, B2, B3, C1, C2, C3)
    silica3 = batoid.SellmeierMedium(C1=C1, C2=C2, C3=C3, B1=B1, B2=B2, B3=B3)
    assert silica == silica2
    assert silica == silica3
    with np.testing.assert_raises(TypeError):
        batoid.SellmeierMedium(B1)
    with np.testing.assert_raises(ValueError):
        batoid.SellmeierMedium(B1, B2, B3, C1, C2, C3, B1)

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
    do_pickle(silica)


@timer
def test_SumitaMedium():
    rng = np.random.default_rng(57)
    # K-BK-7 coefficients
    # https://refractiveindex.info/?shelf=glass&book=SUMITA-BK&page=K-BK7
    A0 = 2.2705778
    A1 = -0.010059376
    A2 = 0.010414999
    A3 = 0.00028872517
    A4 = -2.2214495e-5
    A5 = 1.4258559e-6
    kbk7 = batoid.SumitaMedium([A0, A1, A2, A3, A4, A5])
    kbk7_2 = batoid.SumitaMedium(A0, A1, A2, A3, A4, A5)
    kbk7_3 = batoid.SumitaMedium(A0=A0, A1=A1, A2=A2, A3=A3, A4=A4, A5=A5)
    assert kbk7 == kbk7_2
    assert kbk7 == kbk7_3
    with np.testing.assert_raises(TypeError):
        batoid.SumitaMedium(A0)
    with np.testing.assert_raises(ValueError):
        batoid.SumitaMedium(A0, A1, A2, A3, A4, A5, A0)

    wavelengths = rng.uniform(300e-9, 1100e-9, size=1000)
    indices = kbk7.getN(wavelengths)
    for w, index in zip(wavelengths, indices):
        assert kbk7.getN(w) == index

    # CSV also from refractiveindex.info
    filename = os.path.join(
        os.path.dirname(__file__),
        "testdata",
        "kbk7.csv"
    )
    wmicron, ncsv = np.loadtxt(filename, delimiter=',').T
    n = kbk7.getN(wmicron*1e-6)
    np.testing.assert_allclose(n, ncsv, atol=0, rtol=1e-13)
    do_pickle(kbk7)


@timer
def test_air():
    # Just spot check some comparisons with GalSim.
    ws = [0.3, 0.5, 0.7, 0.9, 1.1]
    gsn = [ 1.00019563,  1.00018713,  1.00018498,  1.00018412,  1.00018369]
    air = batoid.Air()
    for w, n in zip(ws, gsn):
        np.testing.assert_allclose(n, air.getN(w*1e-6), rtol=0, atol=1e-8)
    n = air.getN(np.array(ws)*1e-6)
    np.testing.assert_allclose(n, gsn, atol=1e-8)
    do_pickle(air)


@timer
def test_ne():
    objs = [
        batoid.ConstMedium(1.0),
        batoid.ConstMedium(1.1),
        batoid.TableMedium([1, 2, 3], [3, 4, 3]),
        batoid.SellmeierMedium([
            0.6961663, 0.4079426, 0.8974794,
            0.0684043**2, 0.1162414**2, 9.896161**2]),
        batoid.SellmeierMedium([
            0.4079426, 0.6961663, 0.8974794,
            0.0684043**2, 0.1162414**2, 9.896161**2]),
        batoid.SumitaMedium([
            2.2705778, -0.010059376, 0.010414999,
            0.00028872517, -2.2214495e-5, 1.4258559e-6
        ]),
        batoid.SumitaMedium([
            -0.010059376, 2.2705778, 0.010414999,
            0.00028872517, -2.2214495e-5, 1.4258559e-6
        ]),
        batoid.Air(),
        batoid.Air(pressure=100)
    ]
    all_obj_diff(objs)


if __name__ == '__main__':
    test_ConstMedium()
    test_TableMedium()
    test_SellmeierMedium()
    test_SumitaMedium()
    test_air()
    test_ne()
