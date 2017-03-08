import numpy as np
import jtrace

def parallelRays(z, outer, inner=0, theta_x=0, theta_y=0, nradii=10, naz=64):
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

    """
    # Only approximately correct for the moment.
    radii = np.linspace(inner, outer, nradii)

    dx = z * np.tan(theta_x)
    dy = z * np.tan(theta_y)
    rays = jtrace.RayVector()

    for r in radii:
        phis = np.linspace(0, 2*np.pi, int(naz*r/outer), endpoint=False)
        for phi in phis:
            rays.append(jtrace.Ray(jtrace.Vec3(r*np.cos(phi)+dx, r*np.sin(phi)+dy, z),
                                   jtrace.Vec3(-np.tan(theta_x), -np.tan(theta_y), -1),
                                   0))
    return rays
