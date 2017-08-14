import batoid
from test_helpers import isclose, timer


@timer
def test_properties():
    import random
    random.seed(5)
    for i in range(100):
        R = random.gauss(0.7, 0.8)
        kappa = random.uniform(0.9, 1.0)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = batoid.Asphere(R, kappa, alpha, B)
        dx = random.gauss(0, 1)
        dy = random.gauss(0, 1)
        dz = random.gauss(0, 1)
        transformed = batoid.Transformation(asphere, dx, dy, dz)
        assert transformed.dx == dx
        assert transformed.dy == dy
        assert transformed.dz == dz
        # Try other way to effect a shift.
        transformed2 = asphere.shift(dx, dy, dz)
        assert transformed2.dx == dx
        assert transformed2.dy == dy
        assert transformed2.dz == dz
        # and a third way
        dr = batoid.Vec3(dx, dy, dz)
        transformed3 = asphere.shift(dr)
        assert transformed3.dx == dx
        assert transformed3.dy == dy
        assert transformed3.dz == dz
        assert transformed3.dr == dr


@timer
def test_shift():
    import random
    random.seed(57)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        kappa = random.uniform(-1.0, -0.9)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = batoid.Asphere(R, kappa, alpha, B)
        dx = random.gauss(0, 1)
        dy = random.gauss(0, 1)
        dz = random.gauss(0, 1)
        shifted = asphere.shift(dx, dy, dz)

        for j in range(10):
            x = random.gauss(0.0, 1.0)
            y = random.gauss(0.0, 1.0)

            # If we shoot rays straight up, then it's easy to predict the
            # intersection points.
            r = batoid.Ray(x, y, -10, 0, 0, 1, 0)
            isec = shifted.intersect(r)
            assert isclose(isec.point.x, x)
            assert isclose(isec.point.y, y)
            assert isclose(isec.point.z, asphere.sag(x-dx, y-dy)+dz, rel_tol=0, abs_tol=1e-9)

            # We can also check just for mutual consistency of the asphere,
            # ray and intersection.
            vx = random.gauss(0.0, 0.01)
            vy = random.gauss(0.0, 0.01)
            vz = 1.0
            v = batoid.Vec3(vx, vy, vz).UnitVec3()
            r = batoid.Ray(batoid.Vec3(x, y, -10), v, 0)
            isec = shifted.intersect(r)
            p1 = r.positionAtTime(isec.t)
            p2 = isec.point
            assert isclose(p1.x, p2.x)
            assert isclose(p1.y, p2.y)
            assert isclose(p1.z, p2.z)
            assert isclose(asphere.sag(p1.x-dx, p2.y-dy)+dz, p1.z, rel_tol=0, abs_tol=1e-9)


@timer
def test_shift_vectorized():
    import random
    random.seed(577)
    rays = [batoid.Ray([random.gauss(0.0, 0.1),
                        random.gauss(0.0, 0.1),
                        random.gauss(10.0, 0.1)],
                       [random.gauss(0.0, 0.1),
                        random.gauss(0.0, 0.1),
                        random.gauss(-1.0, 0.1)],
                       random.gauss(0.0, 0.1))
            for i in range(1000)]
    rays = batoid.RayVector(rays)

    for i in range(100):
        R = random.gauss(25.0, 0.2)
        kappa = random.uniform(-1.0, -0.9)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = batoid.Asphere(R, kappa, alpha, B)
        dx = random.gauss(0, 1)
        dy = random.gauss(0, 1)
        dz = random.gauss(0, 1)
        shifted = asphere.shift(dx, dy, dz)
        intersections = shifted.intersect(rays)
        intersections2 = [shifted.intersect(ray) for ray in rays]
        intersections2 = batoid.IntersectionVector(intersections2)
        assert intersections == intersections2


@timer
def test_rotate():
    import random
    import math
    random.seed(5772)
    for i in range(100):
        R = random.gauss(25.0, 0.2)
        kappa = random.uniform(-2.0, 2.0)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = batoid.Asphere(R, kappa, alpha, B)
        # We're going to rain down some photons, rotate the camera and rotate the position/vectors
        # of the photons, and see if we get the same spot pattern.
        for j in range(10):
            x = random.gauss(0, 1.0)
            y = random.gauss(0, 1.0)
            z = 10
            vx = vy = 0
            vz = -1
            r = batoid.Ray(x, y, z, vx, vy, vz, 0)
            isec = asphere.intersect(r)

            theta = random.gauss(0, 0.1)
            st, ct = math.sin(theta), math.cos(theta)
            rotated = asphere.rotX(theta)
            xp = x
            yp = y * ct - z * st
            zp = y * st + z * ct
            vxp = vx
            vyp = vy * ct - vz * st
            vzp = vy * st + vz * ct
            rp = batoid.Ray(xp, yp, zp, vxp, vyp, vzp, 0)
            isecp = rotated.intersect(rp)

            # Now rotate intersection point and normal back to original frame.
            thetap = -theta
            stp, ctp = math.sin(thetap), math.cos(thetap)
            px = isecp.x0
            py = isecp.y0 * ctp - isecp.z0 * stp
            pz = isecp.y0 * stp + isecp.z0 * ctp
            nx = isecp.nx
            ny = isecp.ny * ctp - isecp.nz * stp
            nz = isecp.ny * stp + isecp.nz * ctp
            assert not isclose(isecp.y0, isec.y0)
            assert not isclose(isecp.z0, isec.z0)
            assert not isclose(isecp.ny, isec.ny)
            assert not isclose(isecp.nz, isec.nz)
            assert isclose(px, isec.x0)
            assert isclose(py, isec.y0)
            assert isclose(pz, isec.z0)
            assert isclose(nx, isec.nx)
            assert isclose(ny, isec.ny)
            assert isclose(nz, isec.nz)
            assert isclose(isec.t, isecp.t)

            # And now repeat for RotY
            theta = random.gauss(0, 0.1)
            st, ct = math.sin(theta), math.cos(theta)
            rotated = asphere.rotY(theta)
            xp = x * ct + z * st
            yp = y
            zp = -x * st + z * ct
            vxp = vx * ct + vz * st
            vyp = vy
            vzp = -vx * st + vz * ct
            rp = batoid.Ray(xp, yp, zp, vxp, vyp, vzp, 0)
            isecp = rotated.intersect(rp)

            # Now rotate intersection point and normal back to original frame.
            thetap = -theta
            stp, ctp = math.sin(thetap), math.cos(thetap)
            px = isecp.x0 * ctp + isecp.z0 * stp
            py = isecp.y0
            pz = -isecp.x0 * stp + isecp.z0 * ctp
            nx = isecp.nx * ctp + isecp.nz * stp
            ny = isecp.ny
            nz = -isecp.nx * stp + isecp.nz * ctp
            assert not isclose(isecp.x0, isec.x0)
            assert not isclose(isecp.z0, isec.z0)
            assert not isclose(isecp.nx, isec.nx)
            assert not isclose(isecp.nz, isec.nz)
            assert isclose(px, isec.x0)
            assert isclose(py, isec.y0)
            assert isclose(pz, isec.z0)
            assert isclose(nx, isec.nx)
            assert isclose(ny, isec.ny)
            assert isclose(nz, isec.nz)
            assert isclose(isec.t, isecp.t)

            # And now repeat for RotZ, which as the optic axis, shouldn't actually
            # change any of the intersection points.
            theta = random.gauss(0, 0.1)
            st, ct = math.sin(theta), math.cos(theta)
            rotated = asphere.rotZ(theta)
            isecz = rotated.intersect(r)
            assert isclose(isecz.x0, isec.x0)
            assert isclose(isecz.y0, isec.y0)
            assert isclose(isecz.z0, isec.z0)
            assert isclose(isecz.nx, isec.nx)
            assert isclose(isecz.ny, isec.ny)
            assert isclose(isecz.nz, isec.nz)
            assert isclose(isecz.t, isec.t)


@timer
def test_rotate_vectorized():
    import random
    import math
    random.seed(57721)
    rays = [batoid.Ray([random.gauss(0.0, 0.1),
                        random.gauss(0.0, 0.1),
                        random.gauss(10.0, 0.1)],
                       [random.gauss(0.0, 0.1),
                        random.gauss(0.0, 0.1),
                        random.gauss(-1.0, 0.1)],
                       random.gauss(0.0, 0.1))
            for i in range(1000)]
    rays = batoid.RayVector(rays)

    for i in range(100):
        R = random.gauss(25.0, 0.2)
        kappa = random.uniform(-2.0, 2.0)
        nalpha = random.randint(0, 4)
        alpha = [random.gauss(0, 1e-10) for i in range(nalpha)]
        B = random.gauss(0, 1.1)
        asphere = batoid.Asphere(R, kappa, alpha, B)
        theta = random.gauss(0, 0.1)
        rotated = asphere.rotX(theta)
        phi = random.gauss(0, 0.1)
        rotated = rotated.rotY(phi)
        intersections = rotated.intersect(rays)
        intersections2 = [rotated.intersect(ray) for ray in rays]
        intersections2 = batoid.IntersectionVector(intersections2)
        assert intersections == intersections2


if __name__ == '__main__':
    test_properties()
    test_shift()
    test_shift_vectorized()
    test_rotate()
    test_rotate_vectorized()
