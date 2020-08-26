from ._version import __version__, __version_info__

from .rayVector import RayVector

from .coordSys import CoordSys, RotX, RotY, RotZ
from .coordTransform import CoordTransform

from .surface import Surface, Plane, Paraboloid

from .trace import intersect, applyForwardTransform, applyReverseTransform

# from .ray import Ray
# from .rayVector import RayVector, concatenateRayVectors, rayGrid, circularGrid
# from .rayVector import uniformCircularGrid, pointSourceCircularGrid
# from .surface import Surface, Plane, Paraboloid, Sphere, Quadric, Asphere
# from .surface import Bicubic, Zernike, Sum
# from .table import Table
#
# from .medium import Medium, ConstMedium, TableMedium
# from .medium import SellmeierMedium, SumitaMedium, Air
#
# from .obscuration import Obscuration, ObscCircle, ObscAnnulus, ObscRectangle
# from .obscuration import ObscRay, ObscPolygon
# from .obscuration import ObscUnion, ObscIntersection, ObscNegation
#
# from .coordsys import CoordSys
# from .coordtransform import CoordTransform
#
# from .coating import Coating, SimpleCoating
#
# from .psf import huygensPSF, wavefront, fftPSF, zernike, fpPosition
#
# from .coordsys import RotX, RotY, RotZ
#
# from .optic import Optic, CompoundOptic, Lens
# from .optic import Interface, RefractiveInterface, Mirror, Detector, Baffle
# from .optic import drawTrace2d, drawTrace3d
#
# from .lattice import Lattice
#
# # GPU experimentation
# _has_gpu = True
# try:
#     from ._batoid import CPPRayVector2
# except:
#     _has_gpu = False
#
# if _has_gpu:
#     from .rayVector2 import RayVector2
#     from .coordtransform2 import CoordTransform2
#     from .medium2 import Medium2, ConstMedium2
#     from .medium2 import SellmeierMedium2, SumitaMedium2, Air2
#     from .surface2 import Surface2, Plane2, Sphere2, Paraboloid2, Quadric2, Asphere2
#     from .surface2 import Bilinear2, Bicubic2, ExtendedAsphere2
#     from .obscuration2 import Obscuration2, ObscCircle2, ObscAnnulus2, ObscRay2
#     from .obscuration2 import ObscRectangle2, ObscNegation2, ObscUnion2
#     from .obscuration2 import ObscIntersection2
#
# from . import parse
# from . import utils
# from . import analysis
# from . import plotUtils
# from .utils import normalized
#
# from .constants import globalCoordSys, vacuum
#
# from ._batoid import setNThread, getNThread, setMinChunk, getMinChunk
#
# import os
# datadir = os.path.join(os.path.dirname(__file__), "data")
#
#
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
