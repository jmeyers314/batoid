import sys
import os
d = os.path.dirname(__file__)
sys.path.append(os.path.join(d, '../'))

import jtrace
# para = jtrace.Paraboloid(0.0, 0.0)
# print(para.A)
# print(para.B)
# vec = jtrace.Vec3()
# print(vec.MagnitudeSquared())
# vec = jtrace.Vec3(1, 2, 3)
# print(vec.MagnitudeSquared())
# unitvec = vec.UnitVec3()
# print(unitvec.Magnitude())
# ray = jtrace.Ray(jtrace.Vec3(), jtrace.Vec3(0,0,1))
# print(ray)
# print(ray(1.0))
# print(ray(1.3))

ray = jtrace.Ray(jtrace.Vec3(0,0.5,0), jtrace.Vec3(0,0,1))
para = jtrace.Paraboloid(1, 1)
print(para.intersect(ray))

asphere = jtrace.Asphere(1.0, -1.0, [0.0, 0.001], 0.0)
print(asphere)
print(asphere.alpha)
isec = asphere.intersect(ray)
print(isec)
print(asphere(isec.point.x, isec.point.y))
print(ray(isec.t))
