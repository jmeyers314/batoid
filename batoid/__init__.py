from ._batoid import Vec2, Rot2, Vec3, Rot3, RotVec, UnRotVec
from ._batoid import DotProduct, CrossProduct
from ._batoid import Ray, Intersection, RayVector, IntersectionVector
from ._batoid import Surface, Plane, Paraboloid, Sphere, Quadric, Asphere
from ._batoid import Transformation
from ._batoid import Table

from ._batoid import Medium, ConstMedium, TableMedium, SellmeierMedium, Air

from ._batoid import Obscuration, ObscCircle, ObscAnnulus, ObscRectangle, ObscRay
from ._batoid import ObscUnion, ObscIntersection, ObscNegation

from ._batoid import CoordSys, CoordTransform

from ._batoid import rayGrid, circularGrid
from ._batoid import trimVignetted

from .psf import huygensPSF

# from .telescope import Telescope

# from .rays import parallelRays, rayGrid

from .coordsys import RotX, RotY, RotZ

from .optic import Optic, CompoundOptic, Lens
from .optic import Interface, RefractiveInterface, Mirror, Detector, Baffle

from . import parse

import os
datadir = os.path.join(os.path.dirname(__file__), "data")
