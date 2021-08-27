from ._version import __version__, __version_info__

from .rayVector import RayVector, concatenateRayVectors

from .coordSys import CoordSys, RotX, RotY, RotZ
from .coordTransform import CoordTransform

from .surface import (
    Surface, Plane, Paraboloid, Sphere, Quadric, Asphere, Bicubic, Sum, Tilted,
    Zernike
)

from .trace import (
    intersect, applyForwardTransform, applyReverseTransform, obscure,
    reflect, refract, rSplit
)

from .medium import (
    Medium, ConstMedium, TableMedium, SellmeierMedium, SumitaMedium, Air
)

from .obscuration import (
    Obscuration, ObscCircle, ObscAnnulus, ObscRectangle, ObscRay, ObscPolygon,
    ObscNegation, ObscUnion, ObscIntersection
)

from .coating import Coating, SimpleCoating

from .optic import Optic, CompoundOptic, Lens
from .optic import Interface, RefractiveInterface, Mirror, Detector, Baffle
from .optic import OPDScreen
from .optic import drawTrace2d, drawTrace3d

from .lattice import Lattice

import os
datadir = os.path.join(os.path.dirname(__file__), "data")

from . import parse
from . import utils
from .constants import globalCoordSys, vacuum

from .analysis import (
    huygensPSF, fftPSF, wavefront, zernike, zernikeGQ, drdth, dthdr, spot,
    exitPupilPos
)
from . import plotUtils
