import numpy as np
import yaml
from collections import OrderedDict
from past.builtins import basestring
from numbers import Integral

# https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

# # usage example:
# ordered_load(stream, yaml.SafeLoader)


def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        def represent_mapping(self, tag, mapping, flow_style=False):
            return yaml.Dumper.represent_mapping(self, tag, mapping, flow_style)
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

# # usage:
# ordered_dump(data, Dumper=yaml.SafeDumper)


class ListDict(OrderedDict):
    # Like an ordered dict, but you can access items by number instead of just by key.  So it's
    # a sequence and a mapping.
    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        if any(isinstance(k, Integral) for k in self):
            raise ValueError

    def _getKeyFromIndex(self, idx):
        return list(self.keys())[idx]

    def __setitem__(self, key, value):
        # If key is Integral, access that item in order.  Cannot create a new item using Integral
        # key.  If key is not Integral though, can modify or create as needed.
        if isinstance(key, Integral):
            if key >= len(self):
                raise KeyError
            key = self._getKeyFromIndex(key)
        OrderedDict.__setitem__(self, key, value)

    def __getitem__(self, key):
        if isinstance(key, (Integral, slice)):
            # Should really probably return a new ListDict on slice access, but this works for now.
            return list(self.values())[key]
        return OrderedDict.__getitem__(self, key)

    def __delitem__(self, key):
        if isinstance(key, Integral):
            key = self._getKeyFromIndex(key)
        OrderedDict.__delitem__(self, key)


def bivariate_fit(ux, uy, kx, ky):
    a = np.empty((len(ux), 3), dtype=float)
    a[:,0] = 1
    a[:,1] = ux
    a[:,2] = uy
    b = np.empty((len(ux), 2), dtype=float)
    b[:,0] = kx
    b[:,1] = ky
    x, _, _, _ = np.linalg.lstsq(a, b, rcond=-1)
    return x


def gnomicToDirCos(u, v):
    """Convert gnomic tangent plane projection u,v to direction cosines.

    Parameters
    ----------
    u, v : float
        Gnomic tangent plane coordinates in radians.

    Returns
    -------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0) and (alpha, beta, gamma) = (0,0,1),
    and u.x > 0, u.y=0, v.x=0, v.y > 0.
    """
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)

    gamma = 1/np.sqrt(1.0 + u*u + v*v)
    alpha = u*gamma
    beta = v*gamma

    return alpha, beta, gamma

def dirCosToGnomic(alpha, beta, gamma):
    """Convert direction cosines to gnomic tangent plane projection.

    Parameters
    ----------
    alpha, beta, gamma : float
        Direction cosines (unit vector projected onto x, y, z in order)

    Returns
    -------
    u, v : float
        Gnomic tangent plane coordinates in radians.

    Notes
    -----
    The tangent plane reference is at (u,v) = (0,0) and (alpha, beta, gamma) = (0,0,1)
    and u.x > 0, u.y=0, v.x=0, v.y > 0.
    """
    alpha = np.atleast_1d(alpha)
    beta = np.atleast_1d(beta)
    gamma = np.atleast_1d(gamma)

    u = alpha / gamma
    v = beta / gamma

    return u, v


def gnomicToSpherical(u, v):
    """Convert gnomic tangent plane projection u, v to spherical coordinates.

    Parameters
    ----------
    u, v : float
        Gnomic tangent plane coordinates in radians.

    Returns
    -------
    phi : float
        Polar angle in radians
    theta : float
        Azimuthal angle in radians (always in [-pi, pi], and 0 by convention when phi=0)

    Notes
    -----
    The azimuthal angle is measured from +u through +v (CCW).
    """
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)

    phi = np.arctan(np.sqrt(u*u + v*v))
    theta = np.arctan2(v, u)

    return phi, theta


def sphericalToGnomic(phi, theta):
    """Convert spherical coordiantes to gnomic tangent plane projection.

    Parameters
    ----------
    phi : float
        Polar angle in radians
    theta : float
        Azimuthal angle in radians

    Returns
    -------
    u, v : float
        Gnomic tangent plane coordinates in radians.

    Notes
    -----
    The azimuthal angle is measured from +u through +v (CCW).
    """
    phi = np.atleast_1d(phi)
    theta = np.atleast_1d(theta)

    tph = np.tan(phi)
    u = tph * np.cos(theta)
    v = tph * np.sin(theta)

    return u, v
