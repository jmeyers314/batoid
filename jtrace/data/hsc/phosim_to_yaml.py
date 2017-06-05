import re
from collections import OrderedDict
import numpy as np
from jtrace.yaml import ordered_dump


def translate(infn, outfn):
    out = OrderedDict(name="HSC")
    out['medium'] = "air"
    surfaces = []
    z = 0.0
    with open(infn, 'r') as infile:
        thickness = 0.0
        for line in infile:
            re.sub("\s+", " ", line)
            if len(line) == 0 : continue
            if line[0] == '#': continue
            (name, typ, R, dz, outer, inner, kappa,
             a3, a4, a5, a6, a7, a8, a9, a10, a11,
             a12, a13, a14, a15, a16,
             coating, medium) = line.split()
            z += float(dz)/1000
            thickness += float(dz)/1000
            data = OrderedDict(name=name)
            data['type'] = typ
            data['zvertex'] = z
            data['thickness'] = thickness
            data['inner'] = float(inner)/1000
            data['outer'] = float(outer)/1000
            data['medium'] = medium
            if float(R) == 0.0:
                surface = OrderedDict(plane=data)
            else:
                data['R'] = float(R)/1000
                data['conic'] = float(kappa)
                coef = [-float(np.double(a4)*10**((4-1)*3)),
                        -float(np.double(a6)*10**((6-1)*3)),
                        -float(np.double(a8)*10**((8-1)*3)),
                        -float(np.double(a10)*10**((10-1)*3)),
                        -float(np.double(a12)*10**((12-1)*3)),
                        -float(np.double(a14)*10**((14-1)*3)),
                        -float(np.double(a16)*10**((16-1)*3))]
                if all(c == 0.0 for c in coef):
                    surface = OrderedDict(quadric=data)
                else:
                    data['coef'] = coef
                    surface = OrderedDict(asphere=data)
            surfaces.append(surface)
            thickness = 0.0
    out['surfaces'] = surfaces
    with open(outfn, 'w') as outfile:
        ordered_dump(out, outfile)


if __name__ == '__main__':
    translate("optics.txt", "HSC.yaml")
