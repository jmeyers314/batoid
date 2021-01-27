from batoid.trace import reflect
from . import _batoid


class Coating:
    """Class to control ray reflection/transmission at an `Interface`.

    Coatings can be used to split a ray into reflected/refracted components
    using `Surface.rSplit`, or control the transmission or reflection
    efficiency using `Surface.refract` or `Surface.reflect` (or variations
    thereof).

    In general, the reflection and transmission coefficients may depend on both
    wavelength and the cosine of the incidence angle, which is the angle
    between the incoming ray and the surface normal.
    """
    def getCoefs(self, wavelength, cosIncidenceAngle):
        """Return reflection and transmission coefficients.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength in meters.
        cosIncidenceAngle : float
            Cosine of the incidence angle.

        Returns
        -------
        reflect : float
        transmit : float
        """
        return self._coating.getCoefs(wavelength, cosIncidenceAngle)

    def getReflect(self, wavelength, cosIncidenceAngle):
        """Return reflection coefficient.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength in meters.
        cosIncidenceAngle : float
            Cosine of the incidence angle.

        Returns
        -------
        reflect : float
        """
        return self._coating.getReflect(wavelength, cosIncidenceAngle)

    def getTransmit(self, wavelength, cosIncidenceAngle):
        """Return transmission coefficient.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength in meters.
        cosIncidenceAngle : float
            Cosine of the incidence angle.

        Returns
        -------
        transmit : float
        """
        return self._coating.getTransmit(wavelength, cosIncidenceAngle)

    def __ne__(self, rhs):
        return not (self == rhs)


class SimpleCoating(Coating):
    """Coating with reflectivity and transmissivity that are both constant with
    wavelength and incidence angle.

    Parameters
    ----------
    reflectivity : float
        Reflection coefficient
    transmissivity : float
        Transmission coefficient
    """
    def __init__(self, reflectivity, transmissivity):
        self.reflectivity = reflectivity
        self.transmissivity = transmissivity
        self._coating = _batoid.CPPSimpleCoating(
            reflectivity, transmissivity
        )

    def __eq__(self, rhs):
        return (
            isinstance(rhs, SimpleCoating)
            and self.reflectivity == rhs.reflectivity
            and self.transmissivity == rhs.transmissivity
        )

    def __getstate__(self):
        return self.reflectivity, self.transmissivity

    def __setstate__(self, args):
        self.__init__(*args)

    def __hash__(self):
        return hash(("SimpleCoating", self.reflectivity, self.transmissivity))

    def __repr__(self):
        return f"SimpleCoating({self.reflectivity}, {self.transmissivity})"
