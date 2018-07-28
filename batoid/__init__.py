from ._version import __version__, __version_info__

from ._batoid import Ray, RayVector
from .surface import Surface, Plane, Paraboloid, Sphere, Quadric, Asphere, Zernike, Sum
from ._batoid import Table

from ._batoid import Medium, ConstMedium, TableMedium, SellmeierMedium, SumitaMedium, Air

from ._batoid import Obscuration, ObscCircle, ObscAnnulus, ObscRectangle, ObscRay
from ._batoid import ObscUnion, ObscIntersection, ObscNegation

from ._batoid import CoordSys, CoordTransform

from ._batoid import rayGrid, circularGrid

from .psf import huygensPSF, wavefront, fftPSF, zernike

from .coordsys import RotX, RotY, RotZ

from .optic import Optic, CompoundOptic, Lens
from .optic import Interface, RefractiveInterface, Mirror, Detector, Baffle
from .optic import drawTrace2d, drawTrace3d

from .lattice import Lattice

from . import parse
from . import utils
from .utils import normalized

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
