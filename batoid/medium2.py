import numpy as np

from . import _batoid

class Medium2:
    def getN(self, wavelength):
        import numbers
        if isinstance(wavelength, numbers.Real):
            return self._medium.getN(wavelength)
        else:
            wavelength = np.ascontiguousarray(wavelength)
            out = np.empty_like(wavelength)
            self._medium.getNMany(
                len(wavelength),
                wavelength.ctypes.data,
                out.ctypes.data
            )
            return out


class ConstMedium2(Medium2):
    def __init__(self, n):
        self._medium = _batoid.CPPConstMedium2(n)


# class TableMedium(Medium):
#     def __init__(self, table):
#         self.table = table
#         self._medium = _batoid.CPPTableMedium(self.table._table)
#
#     def __repr__(self):
#         return "TableMedium({!r})".format(self.table)


class SellmeierMedium2(Medium2):
    def __init__(self, B1, B2, B3, C1, C2, C3):
        self._medium = _batoid.CPPSellmeierMedium2(B1, B2, B3, C1, C2, C3)


class SumitaMedium2(Medium2):
    def __init__(self, A0, A1, A2, A3, A4, A5):
        self._medium = _batoid.CPPSumitaMedium2(A0, A1, A2, A3, A4, A5)


class Air2(Medium2):
    def __init__(self, pressure=69.328, temperature=293.15, h2o_pressure=1.067):
        self._medium = _batoid.CPPAir2(pressure, temperature, h2o_pressure)
