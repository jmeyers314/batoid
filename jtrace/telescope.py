from collections import OrderedDict
import numbers
import jtrace
from .yaml import ordered_load


silica = jtrace.SellmeierMedium(
    0.6961663, 0.4079426, 0.8974794,
    0.0684043**2, 0.1162414**2, 9.896161**2)


class Telescope(object):
    @classmethod
    def makeFromYAML(cls, infn):
        out = cls.__new__(cls)
        with open(infn, 'r') as infile:
            data = ordered_load(infile)
            # First, parse the optics
            m0 = data.pop('init_medium', "air")
            if m0 == 'air':
                m0 = jtrace.Air()
            surfaces = data.pop('surfaces')
            out.surfaces = OrderedDict()
            for name, sdata in surfaces.items():
                m1 = sdata.pop('medium', 'air')
                if m1 == 'air':
                    m1 = jtrace.Air()
                elif m1 == 'silica':
                    m1 = silica
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
                    raise RuntimeError("Unknown surface type {}".format(scls))
                out.surfaces[name] = sdict
                m0 = m1
            # Then update any other params in file
            out.__dict__.update(data)
        return out

    def trace(self, r, ax=None, **kwargs):
        if isinstance(r, jtrace.Ray):
            ray = r
            for name, surface in self.surfaces.items():
                isec = surface['surface'].intersect(ray)
                if ax is not None:
                    ax.plot([ray.x0, isec.x0], [ray.z0, isec.z0], **kwargs)
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
