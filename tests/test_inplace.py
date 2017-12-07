import batoid
import numpy as np
import time
from test_helpers import timer

@timer
def test_inplace():
    rays = batoid.rayGrid(10.0, 10.0, 0.1, 0.1, 1024, 500e-9, 1.2)
    print("propagating {} rays.".format(len(rays)))
    ts = np.random.uniform(size=len(rays))

    rays2 = batoid.RayVector(rays)

    t0 = time.time()
    outrays = batoid._batoid.propagatedToTimesMany(rays, ts)
    t1 = time.time()
    batoid._batoid.propagateInPlaceMany(rays2, ts)
    t2 = time.time()

    print("immutable propagation took {:6.3f} seconds.".format(t1-t0))
    print("in-place propagation took  {:6.3f} seconds.".format(t2-t1))

    assert outrays == rays2

if __name__ == '__main__':
    test_inplace()
