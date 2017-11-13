import batoid
import numpy as np
import time
from test_helpers import timer

@timer
def test_inplace():
    rays = batoid.parallelRays(10, 5, nradii=500, naz=2000)
    print(len(rays))
    ts = np.random.uniform(size=len(rays))

    rays2 = batoid.RayVector([r for r in rays])

    t0 = time.time()
    outrays = batoid._batoid.propagatedToTimesMany(rays, ts)
    t1 = time.time()
    batoid._batoid.propagateInPlaceMany(rays2, ts)
    t2 = time.time()

    print(t1-t0)
    print(t2-t1)

    assert outrays == rays2

if __name__ == '__main__':
    test_inplace()
