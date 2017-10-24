from ._batoid import Vec3, Rot3, RotVec, UnRotVec, Ray
from ._batoid import DotProduct, CrossProduct
from ._batoid import Ray, Intersection, RayVector, IntersectionVector
from ._batoid import Surface, Plane, Paraboloid, Sphere, Quadric, Asphere
from ._batoid import Transformation
from ._batoid import Table
from ._batoid import Medium, ConstMedium, TableMedium, SellmeierMedium, Air
from .telescope import Telescope
from .rays import parallelRays, rayGrid
from .coordsys import RotX, RotY, RotZ
from . import optic

import os
datadir = os.path.join(os.path.dirname(__file__), "data")
