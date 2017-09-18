import batoid
import numpy as np
from test_helpers import timer, ray_isclose, rays_allclose


@timer
def test_ident():
    import random
    random.seed(5)

    R = batoid.Rot3()
    dr = batoid.Vec3(0, 0, 0)
    ident = batoid._batoid.XForm(R, dr)
    inverse = ident.inverse()

    rays = []
    for i in range(1000):
        x = random.gauss(0.0, 1.0)
        y = random.gauss(0.0, 1.0)
        z = random.gauss(0.0, 1.0)
        vx = random.gauss(0.0, 1.0)
        vy = random.gauss(0.0, 1.0)
        vz = random.gauss(0.0, 1.0)
        w = random.uniform(300e-9, 1100e-9)
        t = random.uniform(0, 10)
        ray = batoid.Ray(x, y, z, vx, vy, vz, w, t)
        rays.append(ray)
        assert ray == ident.forward(ray)
        assert ray == ident.reverse(ray)
        assert ray == inverse.forward(ray)
        assert ray == inverse.reverse(ray)
    rays = batoid.RayVector(rays)
    assert rays == ident.forward(rays)
    assert rays == ident.reverse(rays)
    assert rays == inverse.forward(rays)
    assert rays == inverse.reverse(rays)


@timer
def test_inverse():
    import random
    random.seed(57)

    th1 = random.uniform(0, np.pi)
    th2 = random.uniform(0, np.pi)
    th3 = random.uniform(0, np.pi)
    R1 = batoid.Rot3([np.cos(th1), -np.sin(th1), 0,
                      np.sin(th1), np.cos(th1), 0,
                      0, 0, 1])
    R2 = batoid.Rot3([np.cos(th2), 0, -np.sin(th2),
                      0, 1, 0,
                      np.sin(th2), 0, np.cos(th2)])
    R3 = batoid.Rot3([np.cos(th3), -np.sin(th3), 0,
                      np.sin(th3), np.cos(th3), 0,
                      0, 0, 1])
    R = R1*R2*R3
    dr = batoid.Vec3(random.gauss(0.0, 1.0), random.gauss(0.0, 1.0), random.gauss(0.0, 1.0))

    xform = batoid._batoid.XForm(R, dr)
    inverse = xform.inverse()

    rays = []
    for i in range(1000):
        x = random.gauss(0.0, 1.0)
        y = random.gauss(0.0, 1.0)
        z = random.gauss(0.0, 1.0)
        vx = random.gauss(0.0, 1.0)
        vy = random.gauss(0.0, 1.0)
        vz = random.gauss(0.0, 1.0)
        w = random.uniform(300e-9, 1100e-9)
        t = random.uniform(0, 10)
        ray = batoid.Ray(x, y, z, vx, vy, vz, t, w)
        rays.append(ray)

        assert ray_isclose(ray, xform.reverse(xform.forward(ray)))
        assert ray_isclose(ray, xform.forward(xform.reverse(ray)))
        assert ray_isclose(ray, inverse.forward(xform.forward(ray)))
        assert ray_isclose(ray, inverse.reverse(xform.reverse(ray)))
        assert ray_isclose(xform.forward(ray), inverse.reverse(ray))
        assert ray_isclose(xform.reverse(ray), inverse.forward(ray))

    rays = batoid.RayVector(rays)

    assert rays_allclose(rays, xform.reverse(xform.forward(rays)))
    assert rays_allclose(rays, xform.forward(xform.reverse(rays)))
    assert rays_allclose(rays, inverse.forward(xform.forward(rays)))
    assert rays_allclose(rays, inverse.reverse(xform.reverse(rays)))
    assert rays_allclose(xform.forward(rays), inverse.reverse(rays))
    assert rays_allclose(xform.reverse(rays), inverse.forward(rays))


@timer
def test_compose():
    import random
    random.seed(577)

    for i in range(10):
        th1 = random.uniform(0, np.pi)
        th2 = random.uniform(0, np.pi)
        th3 = random.uniform(0, np.pi)
        R1 = batoid.Rot3([np.cos(th1), -np.sin(th1), 0,
                          np.sin(th1), np.cos(th1), 0,
                          0, 0, 1])
        R2 = batoid.Rot3([np.cos(th2), 0, -np.sin(th2),
                          0, 1, 0,
                          np.sin(th2), 0, np.cos(th2)])
        R3 = batoid.Rot3([np.cos(th3), -np.sin(th3), 0,
                          np.sin(th3), np.cos(th3), 0,
                          0, 0, 1])
        R = R1*R2*R3
        dr = batoid.Vec3(random.gauss(0.0, 1.0), random.gauss(0.0, 1.0), random.gauss(0.0, 1.0))

        xform1 = batoid._batoid.XForm(R, dr)

        th1 = random.uniform(0, np.pi)
        th2 = random.uniform(0, np.pi)
        th3 = random.uniform(0, np.pi)
        R1 = batoid.Rot3([np.cos(th1), -np.sin(th1), 0,
                          np.sin(th1), np.cos(th1), 0,
                          0, 0, 1])
        R2 = batoid.Rot3([np.cos(th2), 0, -np.sin(th2),
                          0, 1, 0,
                          np.sin(th2), 0, np.cos(th2)])
        R3 = batoid.Rot3([np.cos(th3), -np.sin(th3), 0,
                          np.sin(th3), np.cos(th3), 0,
                          0, 0, 1])
        R = R1*R2*R3
        dr = batoid.Vec3(random.gauss(0.0, 1.0), random.gauss(0.0, 1.0), random.gauss(0.0, 1.0))

        xform2 = batoid._batoid.XForm(R, dr)

        xform21 = xform2*xform1
        xform12 = xform1*xform2

        rays = []
        for j in range(100):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)
            z = random.gauss(0.0, 1.0)
            vx = random.gauss(0.0, 1.0)
            vy = random.gauss(0.0, 1.0)
            vz = random.gauss(0.0, 1.0)
            w = random.uniform(300e-9, 1100e-9)
            t = random.uniform(0, 10)
            ray = batoid.Ray(x, y, z, vx, vy, vz, t, w)
            rays.append(ray)

            assert ray_isclose(xform12.forward(ray), xform1.forward(xform2.forward(ray)))
            assert ray_isclose(xform21.forward(ray), xform2.forward(xform1.forward(ray)))
            assert ray_isclose(xform12.reverse(ray), xform2.reverse(xform1.reverse(ray)))
            assert ray_isclose(xform21.reverse(ray), xform1.reverse(xform2.reverse(ray)))
        rays = batoid.RayVector(rays)

        assert rays_allclose(xform12.forward(rays), xform1.forward(xform2.forward(rays)))
        assert rays_allclose(xform21.forward(rays), xform2.forward(xform1.forward(rays)))
        assert rays_allclose(xform12.reverse(rays), xform2.reverse(xform1.reverse(rays)))
        assert rays_allclose(xform21.reverse(rays), xform1.reverse(xform2.reverse(rays)))


if __name__ == '__main__':
    test_ident()
    test_inverse()
    test_compose()
