import numpy as np
import jtrace

def parallelRays(z, outer, inner=0, theta_x=0, theta_y=0, nradii=10, naz=64,
                 wavelength=500, medium=jtrace.ConstMedium(1.0)):
    """Create a RayVector of parallel Rays aimed toward the origin.

    Parameters
    ----------
    z : float
        distance of center of Ray origins to origin in meters
    outer : float
        Outer radius of distribution of rays in meters
    inner : float
        Inner radius of distribution of rays in meters [default: 0.0]
    theta_x : float
        x-axis field angle in radians [default: 0.0]
    theta_y : float
        y-axis field angle in radians [default: 0.0]
    nradii : int
        Number of radii to use [default: 10]
    naz : int
        Number of azimuthal rays to use on the outer ring. [default: 64]
    wavelength : float
        Wavelength of rays in nm. [default: 500.0]
    medium : jtrace.Medium
        Medium rays are in. [default: jtrace.ConstMedium(1.0)]
    """
    # Only approximately correct for the moment.
    radii = np.linspace(inner, outer, nradii)

    dx = z * np.tan(theta_x)
    dy = z * np.tan(theta_y)
    rays = jtrace.RayVector()

    n = medium.getN(wavelength)

    for r in radii:
        phis = np.linspace(0, 2*np.pi, int(naz*r/outer), endpoint=False)
        for phi in phis:
            p0 = jtrace.Vec3(r*np.cos(phi)+dx, r*np.sin(phi)+dy, z)
            v = jtrace.Vec3(-np.tan(theta_x), -np.tan(theta_y), -1)
            v *= 1./(n*v.Magnitude())
            rays.append(jtrace.Ray(p0, v, 0, wavelength))

    return rays
