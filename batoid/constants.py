from .coordsys import CoordSys
from .medium import ConstMedium
from .medium2 import ConstMedium2

globalCoordSys = CoordSys()
vacuum = ConstMedium(1.0)
vacuum2 = ConstMedium2(1.0)
