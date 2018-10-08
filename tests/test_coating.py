import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff


@timer
def test_SimpleCoating():
    np.random.seed(5)
    for i in range(1000):
        reflectivity = np.random.uniform(0, 1)
        transmissivity = np.random.uniform(0, 1)
        sc = batoid.SimpleCoating(reflectivity, transmissivity)
        assert sc.reflectivity == reflectivity
        assert sc.transmissivity == transmissivity
        do_pickle(sc)


if __name__ == '__main__':
    test_SimpleCoating()
