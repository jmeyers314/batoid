from ._version import __version__, __version_info__

from .rayVector import RayVector

from .coordSys import CoordSys, RotX, RotY, RotZ
from .coordTransform import CoordTransform

from .surface import (
    Surface, Plane, Paraboloid, Sphere, Quadric, Asphere, Bicubic, Sum,
    Zernike
)

from .trace import (
    intersect, applyForwardTransform, applyReverseTransform, obscure,
    reflect, refract
)

from .medium import (
    Medium, ConstMedium, TableMedium, SellmeierMedium, SumitaMedium, Air
)

from .obscuration import (
    ObscCircle, ObscAnnulus, ObscRectangle, ObscRay, ObscPolygon, ObscNegation,
    ObscUnion, ObscIntersection
)

from .coating import SimpleCoating

from .optic import Optic, CompoundOptic, Lens
from .optic import Interface, RefractiveInterface, Mirror, Detector, Baffle
from .optic import drawTrace2d, drawTrace3d

from .lattice import Lattice

import os
datadir = os.path.join(os.path.dirname(__file__), "data")

# from .psf import huygensPSF, wavefront, fftPSF, zernike, fpPosition

from . import parse
from . import utils
from .constants import globalCoordSys, vacuum

# from . import analysis
# from . import plotUtils
#
# __all__ = []
# __all__ += ["Ray", "RayVector"]
# __all__ += ["Surface", "Plane", "Paraboloid", "Sphere", "Quadric", "Asphere", "Zernike", "Sum"]
# __all__ += ["Table"]
# __all__ += ["Medium", "ConstMedium", "TableMedium", "SellmeierMedium", "Air"]
# __all__ += ["Obscuration", "ObscCircle", "ObscAnnulus", "ObscRectangle", "ObscRay"]
# __all__ += ["ObscUnion", "ObscIntersection", "ObscNegation"]
# __all__ += ["CoordSys", "CoordTransform"]
# __all__ += ["rayGrid", "circularGrid"]
# __all__ += ["huygensPSF", "wavefront", "fftPSF", "zernike"]
# __all__ += ["RotX", "RotY", "RotZ"]
# __all__ += ["Optic", "CompoundOptic", "Lens"]
# __all__ += ["Interface", "RefractiveInterface", "Mirror", "Detector", "Baffle"]
# __all__ += ["drawTrace2d", "drawTrace3d"]
# __all__ += ["parse"]
# __all__ += ["setNThread", "getNThread", "setMinChunk", "getMinChunk"]
