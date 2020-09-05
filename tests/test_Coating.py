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
    test_ne()