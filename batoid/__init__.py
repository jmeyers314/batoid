from ._version import __version__, __version_info__

from .ray import Ray
from .rayVector import RayVector, concatenateRayVectors, rayGrid, circularGrid
from .rayVector import uniformCircularGrid, pointSourceCircularGrid
from .surface import Surface, Plane, Paraboloid, Sphere, Quadric, Asphere
from .surface import Bicubic, Zernike, Sum
from .table import Table

from .medium import Medium, ConstMedium, TableMedium
from .medium import SellmeierMedium, SumitaMedium, Air

from .obscuration import Obscuration, ObscCircle, ObscAnnulus, ObscRectangle, ObscRay
from .obscuration import ObscUnion, ObscIntersection, ObscNegation

from .coordsys import CoordSys
from .coordtransform import CoordTransform

from .coating import Coating, SimpleCoating

from .psf import huygensPSF, wavefront, fftPSF, zernike, fpPosition

from .coordsys import RotX, RotY, RotZ

from .optic import Optic, CompoundOptic, Lens
from .optic import Interface, RefractiveInterface, Mirror, Detector, Baffle
from .optic import drawTrace2d, drawTrace3d

from .lattice import Lattice

from . import parse
from . import utils
from . import analysis
from . import plotUtils
from .utils import normalized

from .constants import globalCoordSys, vacuum

from ._batoid import setNThread, getNThread, setMinChunk, getMinChunk

import os
datadir = os.path.join(os.path.dirname(__file__), "data")

__all__ = []
__all__ += ["Ray", "RayVector"]
__all__ += ["Surface", "Plane", "Paraboloid", "Sphere", "Quadric", "Asphere", "Zernike", "Sum"]
__all__ += ["Table"]
__all__ += ["Medium", "ConstMedium", "TableMedium", "SellmeierMedium", "Air"]
__all__ += ["Obscuration", "ObscCircle", "ObscAnnulus", "ObscRectangle", "ObscRay"]
__all__ += ["ObscUnion", "ObscIntersection", "ObscNegation"]
__all__ += ["CoordSys", "CoordTransform"]
__all__ += ["rayGrid", "circularGrid"]
__all__ += ["huygensPSF", "wavefront", "fftPSF", "zernike"]
__all__ += ["RotX", "RotY", "RotZ"]
__all__ += ["Optic", "CompoundOptic", "Lens"]
__all__ += ["Interface", "RefractiveInterface", "Mirror", "Detector", "Baffle"]
__all__ += ["drawTrace2d", "drawTrace3d"]
__all__ += ["parse"]
__all__ += ["setNThread", "getNThread", "setMinChunk", "getMinChunk"]
