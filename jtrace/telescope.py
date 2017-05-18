import jtrace
import re

class Telescope(object):
    def __init__(self, filename):
        # Start defining telescope at z=0, and assume initial refractive indices are that of air.
        z = 0.0
        n1 = 1.000277
        self.optics = []
        for line in open(filename, 'r'):
            re.sub(r"\s+", " ", line)
            if len(line) == 0: continue
            if line[0] == '#': continue
            (name, typ, R, dz, outer, inner, kappa,
             al3, al4, al5, al6, al7, al8, al9, al10,
             coating, medium) = line.split()
            assert float(al3) == float(al5) == float(al7) == float(al9) == 0.0
            z += float(dz)/1000
            if typ == 'none': continue
            # previous post-surface refractive index becomes current pre-surface refractive index.
            n0 = n1
            # Hard-code some values until dispersion module is developed
            if medium == 'air':
                n1 = 1.000277
            else:
                n1 = 1.4542367
            if float(R) == 0:
                surface = jtrace.Plane(z)
            else:
                surface = jtrace.Asphere(float(R)/1000,
                                         float(kappa),
                                         [-float(al4)*10**(4*3),
                                          -float(al6)*10**(6*3),
                                          -float(al8)*10**(8*3),
                                          -float(al10)*10**(10*3)],
                                         z)
            self.optics.append(dict(name=name, surface=surface,
                                    outer=float(outer)/1000, inner=float(inner)/1000,
                                    n0=n0, n1=n1, typ=typ))
    def trace(self, ray):
        r = ray
        for optic in self.optics:
            isec = optic['surface'].intersect(r)
            if optic['typ'] == 'mirror':
                r = isec.reflectedRay(r)
            elif optic['typ'] in ['lens', 'filter']:
                r = isec.refractedRay(r, optic['n0'], optic['n1'])
            elif optic['typ'] == 'det':
                pass
            else:
                raise ValueError("Unknown optic type: {}".format(optic['typ']))
        return isec

    def traceMany(self, rays):
        rs = rays
        for optic in self.optics:
            isecs = optic['surface'].intersect(rs)
            if optic['typ'] == 'mirror':
                rs = jtrace._jtrace.reflectMany(isecs, rs)
            elif optic['typ'] in ['lens', 'filter']:
                rs = jtrace._jtrace.refractMany(isecs, rs, optic['n0'], optic['n1'])
            elif optic['typ'] == 'det':
                pass
            else:
                raise ValueError("Unknown optic type: {}".format(optic['typ']))
        return isecs
