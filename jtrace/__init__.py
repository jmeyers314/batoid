from ._jtrace import Vec3, Rot3, RotVec, UnRotVec, Ray
from ._jtrace import DotProduct, CrossProduct
from ._jtrace import Ray, Intersection, RayVector, IntersectionVector
from ._jtrace import Surface, Plane, Paraboloid, Quadric, Asphere
from ._jtrace import Transformation
from ._jtrace import Table
from ._jtrace import Medium, ConstMedium, TableMedium, SellmeierMedium, Air
from .telescope import Telescope
from .rays import parallelRays

import os
datadir = os.path.join(os.path.dirname(__file__), "data")
