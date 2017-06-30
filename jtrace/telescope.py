import numpy as np
from collections import OrderedDict
import numbers
import jtrace
from .utils import ordered_load


def media_catalog(media_str):
    # This works for LSST, together with jtrace.Air()
    silica = jtrace.SellmeierMedium(
        0.6961663, 0.4079426, 0.8974794,
        0.0684043**2, 0.1162414**2, 9.896161**2)

    # For HSC, we interpolate between values in pdf description
    w = [0.4, 0.6, 0.75, 0.9, 1.1]
    w = np.array(w)*1e-6
    silica_n = [1.47009272, 1.45801158, 1.45421013, 1.45172729, 1.44917721]
    bsl7y_n = [1.53123287, 1.51671428, 1.51225242, 1.50939738, 1.50653251]
    pbl1y_n = [1.57046066, 1.54784671, 1.54157395, 1.53789058, 1.53457169]

    hsc_silica = jtrace.TableMedium(
        jtrace.Table(w, silica_n, jtrace.Table.Interpolant.linear))
    hsc_bsl7y = jtrace.TableMedium(
        jtrace.Table(w, bsl7y_n, jtrace.Table.Interpolant.linear))
    hsc_pbl1y = jtrace.TableMedium(
        jtrace.Table(w, pbl1y_n, jtrace.Table.Interpolant.linear))

    if media_str == 'air':
        return jtrace.Air()
    elif media_str == 'silica':
        return silica
    elif media_str == 'hsc_air':
        return jtrace.ConstMedium(1.0)
    elif media_str == 'hsc_silica':
        return hsc_silica
    elif media_str == 'hsc_bsl7y':
        return hsc_bsl7y
    elif media_str == 'hsc_pbl1y':
        return hsc_pbl1y
    else:
        raise RuntimeError("Unknown medium {}".format(media_str))

class Telescope(object):
    @classmethod
    def makeFromYAML(cls, infn):
        out = cls.__new__(cls)
        with open(infn, 'r') as infile:
            data = ordered_load(infile)
            # First, parse the optics
            m0 = media_catalog(data.pop('init_medium'))
            surfaces = data.pop('surfaces')
            out.surfaces = OrderedDict()
            for name, sdata in surfaces.items():
                m1 = media_catalog(sdata.pop('medium'))
                sdict = dict(
                    name=name,
                    outer=sdata['outer'],
                    inner=sdata['inner'],
                    type=sdata['surftype'],
                    m0=m0,
                    m1=m1)
                sagtype = sdata['sagtype']
                if sagtype == 'plane':
                    sdict['surface'] = jtrace.Plane(
                        sdata['zvertex'],
                        Rin=sdata['inner'],
                        Rout=sdata['outer'])
                elif sagtype == 'sphere':
                    sdict['surface'] = jtrace.Sphere(
                        sdata['R'],
                        sdata['zvertex'],
                        Rin=sdata['inner'],
                        Rout=sdata['outer'])
                elif sagtype == 'quadric':
                    sdict['surface'] = jtrace.Quadric(
                        sdata['R'],
                        sdata['conic'],
                        sdata['zvertex'],
                        Rin=sdata['inner'],
                        Rout=sdata['outer'])
                elif sagtype == 'asphere':
                    sdict['surface']=jtrace.Asphere(
                        sdata['R'],
                        sdata['conic'],
                        sdata['coef'],
                        sdata['zvertex'],
                        Rin=sdata['inner'],
                        Rout=sdata['outer'])
                else:
                    raise RuntimeError("Unknown surface type {}".format(sagtype))
                out.surfaces[name] = sdict
                m0 = m1
            # Then update any other params in file
            out.__dict__.update(data)
        return out

    def trace(self, r):
        if isinstance(r, jtrace.Ray):
            ray = r
            for name, surface in self.surfaces.items():
                isec = surface['surface'].intersect(ray)
                if surface['type'] == 'mirror':
                    ray = isec.reflectedRay(ray)
                elif surface['type'] in ['lens', 'filter']:
                    ray = isec.refractedRay(ray, surface['m0'], surface['m1'])
                elif surface['type'] == 'det':
                    pass
                else:
                    raise ValueError("Unknown optic type: {}".format(surface['type']))
            return ray, isec
        elif isinstance(r, jtrace.RayVector):
            rays = r
            for name, surface in self.surfaces.items():
                isecs = surface['surface'].intersect(rays)
                if surface['type'] == 'mirror':
                    rays = jtrace._jtrace.reflectMany(isecs, rays)
                elif surface['type'] in ['lens', 'filter']:
                    rays = jtrace._jtrace.refractMany(isecs, rays, surface['m0'], surface['m1'])
                elif surface['type'] == 'det':
                    pass
                else:
                    raise ValueError("Unknown optic type: {}".format(surface['type']))
            return rays, isecs

    def traceFull(self, r):
        out = []
        if isinstance(r, jtrace.Ray):
            ray = r
            for name, surface in self.surfaces.items():
                isec = surface['surface'].intersect(ray)
                data = {'name':name, 'isec': isec, 'inray':ray}
                if surface['type'] == 'mirror':
                    ray = isec.reflectedRay(ray)
                elif surface['type'] in ['lens', 'filter']:
                    ray = isec.refractedRay(ray, surface['m0'], surface['m1'])
                elif surface['type'] == 'det':
                    pass
                else:
                    raise ValueError("Unknown optic type: {}".format(surface['type']))
                data['outray'] = ray
                out.append(data)
            return out
        else:
            rays = r
            for name, surface in self.surfaces.items():
                isecs = surface['surface'].intersect(rays)
                data = {'name':name, 'isecs': isecs, 'inrays':rays}
                if surface['type'] == 'mirror':
                    rays = jtrace._jtrace.reflectMany(isecs, rays)
                elif surface['type'] in ['lens', 'filter']:
                    rays = jtrace._jtrace.refractMany(isecs, rays, surface['m0'], surface['m1'])
                elif surface['type'] == 'det':
                    pass
                else:
                    raise ValueError("Unknown optic type: {}".format(surface['type']))
                data['outrays'] = rays
                out.append(data)
            return out

    def clone(self):
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.surfaces = self.surfaces.copy()
        return out

    def withShift(self, surfaceId, dx, dy, dz):
        out = self.clone()
        if isinstance(surfaceId, numbers.Integral):
            container = out.surfaces.values()
        else:
            container = out.surfaces
        sdict = container[surfaceId].copy()
        surf = sdict['surface']
        sdict['surface'] = surf.shift(dx, dy, dz)
        container[surfaceId] = sdict
        return out

    def withRotX(self, surfaceId, theta):
        out = self.clone()
        if isinstance(surfaceId, numbers.Integral):
            container = out.surfaces.values()
        else:
            container = out.surfaces
        sorig = container[surfaceId]
        snew = sorig.rotX(theta)
        container[surfaceId] = snew
        return out

    def withRotY(self, surfaceId, theta):
        out = self.clone()
        if isinstance(surfaceId, numbers.Integral):
            container = out.surfaces.values()
        else:
            container = out.surfaces
        sorig = container[surfaceId]
        snew = sorig.rotY(theta)
        container[surfaceId] = snew
        return out

    def withRotZ(self, surfaceId, theta):
        out = self.clone()
        if isinstance(surfaceId, numbers.Integral):
            container = out.surfaces.values()
        else:
            container = out.surfaces
        sorig = container[surfaceId]
        snew = sorig.rotZ(theta)
        container[surfaceId] = snew
        return out
