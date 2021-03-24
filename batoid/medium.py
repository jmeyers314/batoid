import numpy as np
from . import _batoid


class Medium:
    """The `Medium` class is used to model a refractive medium.

    Media have essentially one function: to return their refractive index given
    a wavelength.
    """
    def getN(self, wavelength):
        """Return refractive index.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength in meters.

        Returns
        -------
        n : float
            Refractive index.
        """
        return self._medium.getN(wavelength)

    def __ne__(self, rhs):
        return not (self == rhs)


class ConstMedium(Medium):
    """A `Medium` with wavelength-independent refractive index.

    Parameters
    ----------
    n : float
        The refractive index.
    """
    def __init__(self, n):
        self.n = n
        self._medium = _batoid.CPPConstMedium(n)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return self.n == rhs.n
        return False

    def __getstate__(self):
        return self.n

    def __setstate__(self, n):
        self.n = n
        self._medium = _batoid.CPPConstMedium(n)

    def __hash__(self):
        return hash(("batoid.ConstMedium", self.n))

    def __repr__(self):
        return f"ConstMedium({self.n})"


class TableMedium(Medium):
    """A `Medium` with refractive index defined via a lookup table.

    Parameters
    ----------
    wavelengths : array of float
        Wavelengths in meters.
    ns : array of float
        Refractive indices.
    """
    def __init__(self, wavelengths, ns):
        self.wavelengths = np.array(wavelengths)
        self.ns = np.array(ns)
        self._medium = _batoid.CPPTableMedium(
            self.wavelengths.ctypes.data,
            self.ns.ctypes.data,
            len(self.wavelengths)
        )

    @classmethod
    def fromTxt(cls, filename, **kwargs):
        """Load a text file with refractive index information in it.
        The file should have two columns, the first with wavelength in microns,
        and the second with the corresponding refractive indices.
        """
        import os
        try:
            wavelength, n = np.loadtxt(filename, unpack=True, **kwargs)
        except IOError:
            import glob
            from . import datadir
            filenames = glob.glob(os.path.join(datadir, "**", "*.txt"))
            for candidate in filenames:
                if os.path.basename(candidate) == filename:
                    wavelength, n = np.loadtxt(candidate, unpack=True, **kwargs)
                    break
            else:
                raise FileNotFoundError(filename)
        return TableMedium(wavelength*1e-6, n)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return (
                np.array_equal(self.wavelengths, rhs.wavelengths)
                and np.array_equal(self.ns, rhs.ns)
            )
        return False

    def __getstate__(self):
        return self.wavelengths, self.ns

    def __setstate__(self, args):
        self.__init__(*args)

    def __hash__(self):
        return hash((
            "batoid.TableMedium", tuple(self.wavelengths), tuple(self.ns)
        ))

    def __repr__(self):
        return f"TableMedium({self.wavelengths!r}, {self.ns!r})"


class SellmeierMedium(Medium):
    r"""A `Medium` with Sellmeier dispersion formula.

    The Sellmeier formula is

    .. math::

        n = \sqrt{1 + \sum_{i=1}^3 \frac{B_i \lambda^2}{\lambda^2 - C_i}}

    where :math:`\lambda` is the vacuum wavelength in microns.

    Parameters
    ----------
    coefs: array of float
        Sellmeier coefficients (B1, B2, B3, C1, C2, C3)
    """
    def __init__(self, *args, **kwargs):
        if len(args) == 6:
            coefs = tuple(args)
        elif len(args) == 1:
            coefs = tuple(args[0])
        elif kwargs:
            coefs = tuple([
                kwargs[k] for k in ['B1', 'B2', 'B3', 'C1', 'C2', 'C3']
            ])
        else:
            raise ValueError("Incorrect number of arguments")
        self.coefs = coefs
        self._medium = _batoid.CPPSellmeierMedium(*coefs)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return self.coefs == rhs.coefs
        return False

    def __getstate__(self):
        return self.coefs

    def __setstate__(self, coefs):
        self.coefs = coefs
        self._medium = _batoid.CPPSellmeierMedium(*coefs)

    def __hash__(self):
        return hash(("batoid.SellmeierMedium", self.coefs))

    def __repr__(self):
        return f"SellmeierMedium({self.coefs})"


class SumitaMedium(Medium):
    r"""A `Medium` with Sumita dispersion formula, also known as the Schott
    dispersion formula.

    The Sumita formula is

    .. math::

        n = \sqrt{A_0 + A_1 \lambda^2 + \sum_{i=2}^5 A_i \lambda^{-2 (i-1)}}

    where :math:`\lambda` is the vacuum wavelength in microns.

    Parameters
    ----------
    coefs: array of float
        Sumita coefficients (A0, A1, A2, A3, A4, A5)
    """
    def __init__(self, *args, **kwargs):
        if len(args) == 6:
            coefs = tuple(args)
        elif len(args) == 1:
            coefs = tuple(args[0])
        elif kwargs:
            coefs = tuple([
                kwargs[k] for k in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
            ])
        else:
            raise ValueError("Incorrect number of arguments")
        self.coefs = coefs
        self._medium = _batoid.CPPSumitaMedium(*coefs)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return self.coefs == rhs.coefs
        return False

    def __getstate__(self):
        return self.coefs

    def __setstate__(self, coefs):
        self.coefs = coefs
        self._medium = _batoid.CPPSumitaMedium(*coefs)

    def __hash__(self):
        return hash(("batoid.SumitaMedium", self.coefs))

    def __repr__(self):
        return f"SumitaMedium({self.coefs})"


class Air(Medium):
    """A `Medium` for air.

    Parameters
    ----------
    pressure : float, optional
        Atmospheric pressure in kiloPascals.  [default: 69.328]
    temperature : float, optional
        Temperature in Kelvin.  [default: 293.15]
    h2o_pressure : float, optional
        Water vapor pressure in kiloPascals.  [default: 1.067]

    Notes
    -----
    Uses the formulae given in Filippenko (1982), which appear to come from
    Edlen (1953), and Coleman, Bozman, and Meggers (1960).  The default values
    for temperature, pressure and water vapor pressure are expected to be
    appropriate for LSST at Cerro Pachon, Chile, but they are broadly
    reasonable for most observatories.
    """
    def __init__(self, pressure=69.328, temperature=293.15, h2o_pressure=1.067):
        self.pressure = pressure
        self.temperature = temperature
        self.h2o_pressure = h2o_pressure
        self._medium = _batoid.CPPAir(pressure, temperature, h2o_pressure)

    def __eq__(self, rhs):
        if type(rhs) == type(self):
            return (
                self.pressure == rhs.pressure
                and self.temperature == rhs.temperature
                and self.h2o_pressure == rhs.h2o_pressure
            )
        return False

    def __getstate__(self):
        return self.pressure, self.temperature, self.h2o_pressure

    def __setstate__(self, args):
        self.pressure, self.temperature, self.h2o_pressure = args
        self._medium = _batoid.CPPAir(
            self.pressure, self.temperature, self.h2o_pressure
        )

    def __hash__(self):
        return hash((
            "batoid.Air", self.pressure, self.temperature, self.h2o_pressure
        ))

    def __repr__(self):
        return f"Air({self.pressure}, {self.temperature}, {self.h2o_pressure})"
