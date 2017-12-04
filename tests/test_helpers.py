import numpy as np


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def ray_isclose(r1, r2, abs_tol=1e-14):
    return (isclose(r1.x0, r2.x0, rel_tol=0, abs_tol=abs_tol)
            and isclose(r1.y0, r2.y0, rel_tol=0, abs_tol=abs_tol)
            and isclose(r1.z0, r2.z0, rel_tol=0, abs_tol=abs_tol)
            and isclose(r1.vx, r2.vx, rel_tol=0, abs_tol=abs_tol)
            and isclose(r1.vy, r2.vy, rel_tol=0, abs_tol=abs_tol)
            and isclose(r1.vz, r2.vz, rel_tol=0, abs_tol=abs_tol)
            and isclose(r1.wavelength, r2.wavelength, rel_tol=0, abs_tol=abs_tol)
            and isclose(r1.t0, r2.t0, rel_tol=0, abs_tol=abs_tol)
            and r1.isVignetted == r2.isVignetted
            and r1.failed == r2.failed)


def rays_allclose(rv1, rv2, abs_tol=1e-14):
    return (np.allclose(rv1.x, rv2.x, rtol=0, atol=abs_tol)
            and np.allclose(rv1.y, rv2.y, rtol=0, atol=abs_tol)
            and np.allclose(rv1.z, rv2.z, rtol=0, atol=abs_tol)
            and np.allclose(rv1.vx, rv2.vx, rtol=0, atol=abs_tol)
            and np.allclose(rv1.vy, rv2.vy, rtol=0, atol=abs_tol)
            and np.allclose(rv1.vz, rv2.vz, rtol=0, atol=abs_tol)
            and np.allclose(rv1.t0, rv2.t0, rtol=0, atol=abs_tol)
            and np.allclose(rv1.wavelength, rv2.wavelength, rtol=0, atol=abs_tol)
            and np.all(rv1.isVignetted == rv2.isVignetted)
            and np.all(rv1.failed == rv2.failed))


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


def do_pickle(obj):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    import copy

    pick = pickle.dumps(obj)
    obj2 = pickle.loads(pick)

    assert obj == obj2, "{} != {}".format(obj, obj2)

    from collections import Hashable
    if isinstance(obj, Hashable):
        assert hash(obj) == hash(obj2)

    # try out copy too
    obj3 = copy.copy(obj)
    assert obj == obj3

    obj4 = copy.deepcopy(obj)
    assert obj == obj4
