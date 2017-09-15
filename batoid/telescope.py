import numpy as np
import batoid
from .utils import ordered_load, ListDict
from .parse import parse_obscuration


def media_catalog(media_str):
    # This works for LSST, together with batoid.Air()
    silica = batoid.SellmeierMedium(
        0.6961663, 0.4079426, 0.8974794,
        0.0684043**2, 0.1162414**2, 9.896161**2)

    # For HSC, we interpolate between values in pdf description
    w = [0.4, 0.6, 0.75, 0.9, 1.1]
    w = np.array(w)*1e-6
    silica_n = [1.47009272, 1.45801158, 1.45421013, 1.45172729, 1.44917721]
    bsl7y_n = [1.53123287, 1.51671428, 1.51225242, 1.50939738, 1.50653251]
    pbl1y_n = [1.57046066, 1.54784671, 1.54157395, 1.53789058, 1.53457169]

    hsc_silica = batoid.TableMedium(
        batoid.Table(w, silica_n, batoid.Table.Interpolant.linear))
    hsc_bsl7y = batoid.TableMedium(
        batoid.Table(w, bsl7y_n, batoid.Table.Interpolant.linear))
    hsc_pbl1y = batoid.TableMedium(
        batoid.Table(w, pbl1y_n, batoid.Table.Interpolant.linear))

    if media_str == 'air':
        return batoid.Air()
    elif media_str == 'silica':
        return silica
    elif media_str == 'hsc_air':
        return batoid.ConstMedium(1.0)
    elif media_str == 'hsc_silica':
        return hsc_silica
    elif media_str == 'hsc_bsl7y':
        return hsc_bsl7y
    elif media_str == 'hsc_pbl1y':
        return hsc_pbl1y
    else:
        raise RuntimeError("Unknown medium {}".format(media_str))


class Telescope(object):
    def __init__(self, surfaceList, **kwargs):
        self.surfaces = ListDict()
        for surface in surfaceList:
            self.surfaces[surface['name']] = surface
        self.__dict__.update(kwargs)

    @classmethod
    def makeFromYAML(cls, infn):
        out = cls.__new__(cls)
        with open(infn, 'r') as infile:
            data = ordered_load(infile)
            # First, parse the optics
            m0 = media_catalog(data.pop('init_medium'))
            surfaces = data.pop('surfaces')
            out.surfaces = ListDict()
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
                    sdict['surface'] = batoid.Plane(
                        sdata['zvertex'],
                        Rin=sdata['inner'],
                        Rout=sdata['outer'])
                elif sagtype == 'sphere':
                    sdict['surface'] = batoid.Sphere(
                        sdata['R'],
                        sdata['zvertex'],
                        Rin=sdata['inner'],
                        Rout=sdata['outer'])
                elif sagtype == 'paraboloid':
                    sdict['surface']=batoid.Paraboloid(
                        sdata['R'],
                        sdata['zvertex'],
                        Rin=sdata['inner'],
                        Rout=sdata['outer'])
                elif sagtype == 'quadric':
                    sdict['surface'] = batoid.Quadric(
                        sdata['R'],
                        sdata['conic'],
                        sdata['zvertex'],
                        Rin=sdata['inner'],
                        Rout=sdata['outer'])
                elif sagtype == 'asphere':
                    sdict['surface']=batoid.Asphere(
                        sdata['R'],
                        sdata['conic'],
                        sdata['coef'],
                        sdata['zvertex'],
                        Rin=sdata['inner'],
                        Rout=sdata['outer'])
                else:
                    raise RuntimeError("Unknown surface type {}".format(sagtype))
                if 'obscuration' in sdata:
                    sdict['obscuration'] = parse_obscuration(sdata['obscuration'])
                out.surfaces[name] = sdict
                m0 = m1
            # Then update any other params in file
            out.__dict__.update(data)
        return out

    def trace(self, r):
        if isinstance(r, batoid.Ray):
            ray = r
            for name, surface in self.surfaces.items():
                isec = surface['surface'].intersect(ray)
                if surface['type'] == 'mirror':
                    ray = isec.reflectedRay(ray)
                elif surface['type'] in ['lens', 'filter']:
                    ray = isec.refractedRay(ray, surface['m0'], surface['m1'])
                elif surface['type'] == 'det':
                    ray = ray.propagatedToTime(isec.t)
                else:
                    raise ValueError("Unknown optic type: {}".format(surface['type']))
            return ray
        elif isinstance(r, batoid.RayVector):
            rays = r
            for name, surface in self.surfaces.items():
                isecs = surface['surface'].intersect(rays)
                if surface['type'] == 'mirror':
                    rays = batoid._batoid.reflectMany(isecs, rays)
                elif surface['type'] in ['lens', 'filter']:
                    rays = batoid._batoid.refractMany(isecs, rays, surface['m0'], surface['m1'])
                elif surface['type'] == 'det':
                    rays = batoid._batoid.propagatedToTimesMany(rays, isecs.t)
                else:
                    raise ValueError("Unknown optic type: {}".format(surface['type']))
            return rays

    def traceFull(self, r):
        out = []
        if isinstance(r, batoid.Ray):
            ray = r
            for name, surface in self.surfaces.items():
                isec = surface['surface'].intersect(ray)
                data = {'name':name, 'inray':ray}
                if surface['type'] == 'mirror':
                    ray = isec.reflectedRay(ray)
                elif surface['type'] in ['lens', 'filter']:
                    ray = isec.refractedRay(ray, surface['m0'], surface['m1'])
                elif surface['type'] == 'det':
                    ray = ray.propagatedToTime(isec.t)
                else:
                    raise ValueError("Unknown optic type: {}".format(surface['type']))
                data['outray'] = ray
                out.append(data)
            return out
        else:
            rays = r
            for name, surface in self.surfaces.items():
                isecs = surface['surface'].intersect(rays)
                data = {'name':name, 'inrays':rays}
                if surface['type'] == 'mirror':
                    rays = batoid._batoid.reflectMany(isecs, rays)
                elif surface['type'] in ['lens', 'filter']:
                    rays = batoid._batoid.refractMany(isecs, rays, surface['m0'], surface['m1'])
                elif surface['type'] == 'det':
                    rays = batoid._batoid.propagatedToTimesMany(rays, isecs.t)
                else:
                    raise ValueError("Unknown optic type: {}".format(surface['type']))
                data['outrays'] = rays
                out.append(data)
            return out

    def huygensPSF(self, xs, ys, zs=None, rays=None, wavelength=None, theta_x=0, theta_y=0, nradii=5, naz=50):
        if rays is None:
            # Generate some rays based on the first optic.
            s0 = self.surfaces[0]
            rays = batoid.parallelRays(
                z=10, outer=s0['outer'], inner=s0['inner'],
                theta_x=theta_x, theta_y=theta_y,
                nradii=nradii, naz=naz,
                wavelength=wavelength, medium=s0['m0']
            )
        rays = self.trace(rays)
        rays = batoid.RayVector([r for r in rays if not r.isVignetted])
        if zs is None:
            zs = np.empty(xs.shape, dtype=np.float64)
            zs.fill(self.surfaces[-1]['surface'].B)
        points = np.concatenate([aux[..., None] for aux in (xs, ys, zs)], axis=-1)
        time = rays[0].t0  # Doesn't actually matter, but use something close to intercept time
        amplitudes = np.empty(xs.shape, dtype=np.complex128)
        for (i, j) in np.ndindex(xs.shape):
            amplitudes[i, j] = np.sum(batoid._batoid.amplitudeMany(
                rays,
                batoid.Vec3(*points[i, j]),
                time
            )
        )
        return np.abs(amplitudes)**2

    def exit_pupil_z(self, wavelength, theta=10./206265):
        # Trace a parabasal ray, i.e., a ray that goes through the center of the entrance pupil at a
        # small angle, and see where it intersects the optic axis again.  We're assuming here both
        # that the entrance pupil is coincident with the first surface (which is reasonable for most
        # telescopes), and that the optics are centered.
        point = batoid.Vec3(0, 0, 0)
        v = batoid.Vec3(0.0, np.sin(theta), -np.cos(theta))
        m0 = self.surfaces[0]['m0']
        v /= m0.getN(wavelength)
        r = batoid.Ray(point, v, t=0, w=wavelength)
        # rewind a bit so we can find an intersection
        r = r.propagatedToTime(-1)
        r = self.trace(r)
        t = -r.y0/r.vy + r.t0
        XP = r.positionAtTime(t).z
        return XP

    def _reference_sphere(self, point, wavelength, theta=10./206265):
        XP = self.exit_pupil_z(wavelength, theta)
        ref_sphere_radius = XP - point.z
        return (batoid.Sphere(-ref_sphere_radius, point.z+ref_sphere_radius)
                .shift(point.x, point.y, 0.0))

    def wavefront(self, theta_x, theta_y, wavelength, rays=None, nx=32):
        if rays is None:
            EP_size = self.surfaces[0]['outer']
            m0 = self.surfaces[0]['m0']
            rays = batoid.rayGrid(
                    10, 2*EP_size,
                    theta_x=theta_x, theta_y=theta_y,
                    wavelength=wavelength, medium=m0, nx=nx)
        outrays = self.trace(rays)
        w = np.logical_not(outrays.isVignetted)
        point = batoid.Vec3(np.mean(outrays.x[w]), np.mean(outrays.y[w]), np.mean(outrays.z[w]))
        ref_sphere = self._reference_sphere(point, wavelength)
        isecs = ref_sphere.intersect(outrays)
        wf = (isecs.t-np.mean(isecs.t[w]))/wavelength
        wf = np.ma.masked_array(wf, mask=outrays.isVignetted)
        return wf

    def fftPSF(self, theta_x, theta_y, wavelength, nx=32, pad_factor=2):
        L = self.surfaces[0]['outer']*2*pad_factor
        im_dtheta = wavelength / L
        wf = self.wavefront(theta_x, theta_y, wavelength, nx=nx).reshape(nx, nx)
        pad_size = nx*pad_factor
        expwf = np.zeros((pad_size, pad_size), dtype=np.complex128)
        start = pad_size//2-nx//2
        stop = pad_size//2+nx//2
        expwf[start:stop, start:stop][~wf.mask] = np.exp(2j*np.pi*wf[~wf.mask])
        psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(expwf))))**2
        return im_dtheta, psf

    def clone(self):
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.surfaces = self.surfaces.copy()
        return out

    def withShift(self, surfaceId, dx, dy, dz):
        out = self.clone()
        sdict = out.surfaces[surfaceId].copy()
        surf = sdict['surface']
        sdict['surface'] = surf.shift(dx, dy, dz)
        out.surfaces[surfaceId] = sdict
        return out

    def withRotX(self, surfaceId, theta):
        out = self.clone()
        sorig = out.surfaces[surfaceId]
        snew = sorig.rotX(theta)
        out.surfaces[surfaceId] = snew
        return out

    def withRotY(self, surfaceId, theta):
        out = self.clone()
        sorig = out.surfaces[surfaceId]
        snew = sorig.rotY(theta)
        out.surfaces[surfaceId] = snew
        return out

    def withRotZ(self, surfaceId, theta):
        out = self.clone()
        sorig = out.surfaces[surfaceId]
        snew = sorig.rotZ(theta)
        out.surfaces[surfaceId] = snew
        return out
