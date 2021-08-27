import numpy as np
import batoid


def rays_allclose(rv1, rv2, atol=1e-14):
    np.testing.assert_allclose(
        rv1.r, rv2.r, rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        rv1.v, rv2.v, rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        rv1.t, rv2.t, rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        rv1.wavelength, rv2.wavelength, rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        rv1.flux, rv2.flux, rtol=0, atol=atol
    )
    np.testing.assert_array_equal(
        rv1.vignetted, rv2.vignetted
    )
    np.testing.assert_array_equal(
        rv1.failed, rv2.failed
    )


def checkAngle(a, b, rtol=0, atol=1e-14):
    diff = (a-b)%(2*np.pi)
    absdiff = np.min([np.abs(diff), np.abs(2*np.pi-diff)], axis=0)
    np.testing.assert_allclose(absdiff, 0, rtol=0, atol=atol)


def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2


def do_pickle(obj, reprable=True):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    import copy

    pick = pickle.dumps(obj, -1)
    obj2 = pickle.loads(pick)

    assert obj == obj2, "{} != {}".format(obj, obj2)

    from collections.abc import Hashable
    if isinstance(obj, Hashable):
        assert hash(obj) == hash(obj2)

    # try out copy too
    obj3 = copy.copy(obj)
    assert obj == obj3

    obj4 = copy.deepcopy(obj)
    assert obj == obj4

    if reprable:
        from numpy import (
            array, uint16, uint32, int16, int32, float32, float64, complex64,
            complex128, ndarray
        )
        from batoid import (
            RayVector,
            Plane, Paraboloid, Sphere, Quadric, Asphere,
            Bicubic, Sum, Tilted, Zernike,
            ConstMedium, TableMedium, SellmeierMedium, SumitaMedium, Air,
            ObscCircle, ObscAnnulus, ObscRectangle, ObscRay, ObscPolygon,
            ObscNegation, ObscUnion, ObscIntersection,
            CoordSys, CoordTransform,
            SimpleCoating,
            Optic, CompoundOptic, Baffle, Mirror, Lens, RefractiveInterface,
            OPDScreen, Detector,
            Lattice
        )
        # While eval(repr(obj)) == obj is the python repr gold standard, it can
        # be pretty ugly for exact reproduction of doubles.  Here, we strive for
        # a lesser goal:
        #      repr(eval(repr(obj))) == repr(obj).
        # I.e., it's okay to lose precision, as long as it only happens once.
        try:
            obj5 = eval(repr(obj))
        except SyntaxError:
            raise RuntimeError("Failed to eval(repr(obj)) for {!r}".format(obj))
        assert repr(obj) == repr(obj5)


def all_obj_diff(objs):
    """ Helper function that verifies that each element in `objs` is unique and,
    if hashable, produces a unique hash.
    """

    from collections.abc import Hashable
    # Check that all objects are unique.
    # Would like to use `assert len(objs) == len(set(objs))` here, but this
    # requires that the elements of objs are hashable (and that they have unique
    # hashes!, which is what we're trying to test!.  So instead, we just loop
    # over all combinations.
    for i, obji in enumerate(objs):
        # Could probably start the next loop at `i+1`, but we start at 0 for
        # completeness (and to verify a != b implies b != a)
        for j, objj in enumerate(objs):
            if i == j:
                continue
            assert obji != objj, (
                "Found equivalent objects {0} == {1} at indices {2} and {3}"
                .format(obji, objj, i, j)
            )

    # Now check that all hashes are unique (if the items are hashable).
    if not isinstance(objs[0], Hashable):
        return
    hashes = [hash(obj) for obj in objs]
    assert len(hashes) == len(set(hashes))


def init_gpu():
    arr = np.zeros(100)
    _arr = batoid._batoid.CPPDualViewDouble(arr.ctypes.data, len(arr))
    _arr.syncToDevice()
