import batoid
from test_helpers import timer


@timer
def test_properties():
    s1 = batoid.Sphere(3.0)
    s2 = batoid.Paraboloid(4.0)
    sum = batoid._batoid.Sum([s1, s2])

    assert sum.surfaces[0] == s1
    assert sum.surfaces[1] == s2

    s3 = batoid._batoid.Zernike([0]*4+[1])
    sum2 = batoid._batoid.Sum([s1, s2, s3])

    assert sum2.surfaces[0] == s1
    assert sum2.surfaces[1] == s2
    assert sum2.surfaces[2] == s3


if __name__ == '__main__':
    test_properties()
