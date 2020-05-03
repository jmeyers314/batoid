import numpy as np


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def ray_isclose(ray1, ray2, atol=1e-14):
    return (
        np.allclose(ray1.r, ray2.r, rtol=0, atol=atol)
        and np.allclose(ray1.v, ray2.v, rtol=0, atol=atol)
        and np.allclose(ray1.t, ray2.t, rtol=0, atol=atol)
        and np.allclose(ray1.wavelength, ray2.wavelength, rtol=0, atol=atol)
        and ray1.vignetted == ray2.vignetted
        and ray1.failed == ray2.failed
    )


def rays_allclose(rv1, rv2, atol=1e-14):
    return (
        np.allclose(rv1.r, rv2.r, rtol=0, atol=atol)
        and np.allclose(rv1.v, rv2.v, rtol=0, atol=atol)
        and np.allclose(rv1.t, rv2.t, rtol=0, atol=atol)
        and np.allclose(rv1.wavelength, rv2.wavelength, rtol=0, atol=atol)
        and np.all(rv1.vignetted == rv2.vignetted)
        and np.all(rv1.failed == rv2.failed)
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
        from numpy import array, uint16, uint32, int16, int32, float32, float64, complex64, complex128, ndarray
        from batoid import Ray, RayVector
        from batoid import Plane, Paraboloid, Sphere, Quadric, Asphere
        from batoid import Bicubic, Zernike, Sum
        from batoid import Table
        from batoid import ConstMedium, TableMedium, SellmeierMedium, Air
        from batoid import ObscCircle, ObscAnnulus, ObscRectangle, ObscRay
        from batoid import ObscPolygon
        from batoid import ObscNegation, ObscUnion, ObscIntersection
        from batoid import CoordSys, CoordTransform
        from batoid import CompoundOptic, Lens
        from batoid import RefractiveInterface, Mirror, Detector, Baffle
        from batoid import SimpleCoating
        # While eval(repr(obj)) == obj is the python repr gold standard, it can be pretty ugly for
        # exact reproduction of doubles.  Here, we strive for a lesser goal:
        #      repr(eval(repr(obj))) == repr(obj).
        # I.e., it's okay to lose precision, as long as it only happens once.
        try:
            obj5 = eval(repr(obj))
        except SyntaxError:
            raise RuntimeError("Failed to eval(repr(obj)) for {!r}".format(obj))
        assert repr(obj) == repr(obj5)


def all_obj_diff(objs):
    """ Helper function that verifies that each element in `objs` is unique and, if hashable,
    produces a unique hash."""

    from collections.abc import Hashable
    # Check that all objects are unique.
    # Would like to use `assert len(objs) == len(set(objs))` here, but this requires that the
    # elements of objs are hashable (and that they have unique hashes!, which is what we're trying
    # to test!.  So instead, we just loop over all combinations.
    for i, obji in enumerate(objs):
        # Could probably start the next loop at `i+1`, but we start at 0 for completeness
        # (and to verify a != b implies b != a)
        for j, objj in enumerate(objs):
            if i == j:
                continue
            assert obji != objj, ("Found equivalent objects {0} == {1} at indices {2} and {3}"
                                  .format(obji, objj, i, j))

    # Now check that all hashes are unique (if the items are hashable).
    if not isinstance(objs[0], Hashable):
        return
    hashes = [hash(obj) for obj in objs]
    try:
        assert len(hashes) == len(set(hashes))
    except AssertionError as e:
        try:
            # Only valid in 2.7, but only needed if we have an error to provide more information.
            from collections.abc import Counter
        except ImportError:
            raise e
        for k, v in Counter(hashes).items():
            if v <= 1:
                continue
            print("Found multiple equivalent object hashes:")
            for i, obj in enumerate(objs):
                if hash(obj) == k:
                    print(i, repr(obj))
        raise e
