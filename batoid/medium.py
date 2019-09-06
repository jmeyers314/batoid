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

    def __repr__(self):
        return repr(self._medium)

    def __eq__(self, rhs):
        return (type(self) == type(rhs)
                and self._medium == rhs._medium)

    def __ne__(self, rhs):
        return not (self == rhs)

    def __hash__(self):
        return hash((type(self), self._medium))

    def __repr__(self):
        return repr(self._medium)


class ConstMedium(Medium):
    """A `Medium` with wavelength-independent refractive index.

    Parameters
    ----------
    n : float
        The refractive index.
    """
    def __init__(self, n):
        self._medium = _batoid.ConstMedium(n)


class TableMedium(Medium):
    """A `Medium` with refractive index defined via a lookup table.

    Parameters
    ----------
    table : `batoid.Table`
        Lookup table for refractive index.
    """
    def __init__(self, table):
        self._table = table
        self._medium = _batoid.TableMedium(table)

    @property
    def table(self):
        """Lookup table for refractive index."""
        return self._table


class SellmeierMedium(Medium):
    r"""A `Medium` with Sellmeier dispersion formula.

    The Sellmeier formula is

    .. math::

        n = \sqrt{1 + \sum_{i=1}^3 \frac{B_i \lambda^2}{\lambda^2 - C_i}}

    where :math:`\lambda` is the vacuum wavelength in microns.

    Parameters
    ----------
    B1, B2, B3, C1, C2, C3: float
        Sellmeier coefficients.
    """
    def __init__(self, B1, B2, B3, C1, C2, C3):
        self._medium = _batoid.SellmeierMedium(B1, B2, B3, C1, C2, C3)

    @property
    def coefs(self):
        """array of float, shape (6,) : The Sellmeier dispersion formula
        coefficients [B1, B2, B3, C1, C2, C3].
        """
        return self._medium.getCoefs()


class SumitaMedium(Medium):
    r"""A `Medium` with Sumita dispersion formula.

    The Sumita formula is

    .. math::

        n = \sqrt{A_0 + A_1 \lambda^2 + \sum_{i=2}^5 A_i \lambda^{-2 (i-1)}}

    where :math:`\lambda` is the vacuum wavelength in microns.

    Parameters
    ----------
    A0, A1, A2, A3, A4, A5 : float
        Sumita coefficients.
    """
    def __init__(self, A0, A1, A2, A3, A4, A5):
        self._medium = _batoid.SumitaMedium(A0, A1, A2, A3, A4, A5)

    @property
    def coefs(self):
        """array of float, shape (6,) : The Sumita dispersion formula
        coefficients [A0, A1, A2, A3, A4, A5].
        """
        return self._medium.getCoefs()


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
        self._medium = _batoid.Air(pressure, temperature, h2o_pressure)

    @property
    def pressure(self):
        return self._medium.getPressure()

    @property
    def temperature(self):
        return self._medium.getTemperature()

    @property
    def h2o_pressure(self):
        return self._medium.getH2OPressure()
