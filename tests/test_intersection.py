import jtrace
from test_helpers import isclose, timer


@timer
def test_intersectionVector():
    import random
    import numpy as np
    random.seed(5)
    intersectionList = []
    for i in range(1000):
        intersectionList.append(
            jtrace.Intersection(
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
                random.gauss(0.0, 1.0),
            )
        )
    isecs = jtrace.IntersectionVector(intersectionList)
    np.testing.assert_equal(isecs.t, np.array([isec.t for isec in isecs]))
    np.testing.assert_equal(isecs.x, np.array([isec.point.x for isec in isecs]))
    np.testing.assert_equal(isecs.y, np.array([isec.point.y for isec in isecs]))
    np.testing.assert_equal(isecs.z, np.array([isec.point.z for isec in isecs]))
    np.testing.assert_equal(isecs.nx, np.array([isec.surfaceNormal.x for isec in isecs]))
    np.testing.assert_equal(isecs.ny, np.array([isec.surfaceNormal.y for isec in isecs]))
    np.testing.assert_equal(isecs.nz, np.array([isec.surfaceNormal.z for isec in isecs]))
    np.testing.assert_equal(isecs.isVignetted, np.array([isec.isVignetted for isec in isecs]))
    np.testing.assert_equal(isecs.failed, np.array([isec.failed for isec in isecs]))

if __name__ == '__main__':
    test_intersectionVector()
