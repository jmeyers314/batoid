import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_SimpleCoating():
    rng = np.random.default_rng(5)
    for i in range(1000):
        reflectivity = rng.uniform(0, 1)
        transmissivity = rng.uniform(0, 1)
        sc = batoid.SimpleCoating(reflectivity, transmissivity)
        assert sc.reflectivity == reflectivity == sc.getReflect(0.1, 0.2) == sc.getReflect(0.3, 0.4)
        assert sc.transmissivity == transmissivity == sc.getTransmit(0.1, 0.2) == sc.getTransmit(0.3, 0.4)
        r, t = sc.getCoefs(0.3, 0.6)
        assert r == reflectivity
        assert t == transmissivity
        do_pickle(sc)


@timer
def test_intersect():
    rng = np.random.default_rng(57)
    for i in range(1000):
        R = rng.uniform(1.0, 2.0)
        sphere = batoid.Sphere(R)

        reflectivity = rng.uniform(0, 1)
        transmissivity = rng.uniform(0, 1)
        coating = batoid.SimpleCoating(reflectivity, transmissivity)

        rv = batoid.RayVector(0, 0, 10, 0, 0, -1, 0, 500e-9, 1.0)

        sphere.intersect(rv, coating=coating)

        assert(rv.flux[0] == transmissivity)


@timer
def test_reflect():
    rng = np.random.default_rng(577)
    for i in range(1000):
        R = rng.uniform(1.0, 2.0)
        sphere = batoid.Sphere(R)

        reflectivity = rng.uniform(0, 1)
        transmissivity = rng.uniform(0, 1)
        coating = batoid.SimpleCoating(reflectivity, transmissivity)

        rv = batoid.RayVector(0, 0, 10, 0, 0, -1, 0, 500e-9, 1.0)

        sphere.reflect(rv, coating=coating)

        assert(rv.flux[0] == reflectivity)


@timer
def test_refract():
    rng = np.random.default_rng(5772)
    for i in range(1000):
        R = rng.uniform(1.0, 2.0)
        sphere = batoid.Sphere(R)

        reflectivity = rng.uniform(0, 1)
        transmissivity = rng.uniform(0, 1)
        coating = batoid.SimpleCoating(reflectivity, transmissivity)

        m1 = batoid.ConstMedium(rng.uniform(1.1, 1.2))
        m2 = batoid.ConstMedium(rng.uniform(1.2, 1.3))

        rv = batoid.RayVector(0, 0, 10, 0, 0, -1, 0, 500e-9, 1.0)

        sphere.refract(rv, m1, m2, coating=coating)

        assert(rv.flux[0] == transmissivity)


@timer
def test_ne():
    objs = [
        batoid.SimpleCoating(0.0, 1.0),
        batoid.SimpleCoating(0.0, 0.1),
        batoid.SimpleCoating(0.1, 0.1),
        batoid.CoordSys()
    ]
    all_obj_diff(objs)


if __name__ == '__main__':
    test_SimpleCoating()
    test_intersect()
    test_reflect()
    test_refract()
    test_ne()
