import re
from collections import OrderedDict
import numpy as np
from jtrace.utils import ordered_dump


def translate(infn, outfn, f):
    out = OrderedDict(name="LSST")
    out['config'] = f
    out['init_medium'] = "air"
    surfaces = OrderedDict()
    z = 0.0
    with open(infn, 'r') as infile:
        thickness = 0.0
        for line in infile:
            re.sub("\s+", " ", line)
            if len(line) == 0 : continue
            if line[0] == '#': continue
            name, typ, R, dz, outer, inner, kappa, a3, a4, a5, a6, a7, a8, a9, a10, coating, medium = line.split()
            if medium == "silica_dispersion.txt":
                medium = "silica"
            z += float(dz)/1000
            thickness += float(dz)/1000
            if typ == "none":
                continue
            data = OrderedDict()
            data['sagtype'] = '' # Want it first, but will overwrite with actual value in a sec
            data['surftype'] = typ
            data['zvertex'] = z
            data['thickness'] = thickness
            data['inner'] = float(inner)/1000
            data['outer'] = float(outer)/1000
            data['medium'] = medium
            if float(R) == 0.0:
                data['sagtype'] = "plane"
                surfaces[name] = data
            else:
                data['R'] = float(R)/1000
                coef = [-float(a4)*10**(4*3),
                        -float(a6)*10**(6*3),
                        -float(a8)*10**(8*3),
                        -float(a10)*10**(10*3)]
                coef = np.trim_zeros(coef, 'b')
                if len(coef) == 0:
                    if float(kappa) == 0.0:
                        data['sagtype'] = "sphere"
                        surfaces[name] = data
                    else:
                        data['sagtype'] = "quadric"
                        data['conic'] = float(kappa)
                        surfaces[name] = data
                else:
                    data['sagtype'] = "asphere"
                    data['conic'] = float(kappa)
                    data['coef'] = coef
                    surfaces[name] = data
            thickness = 0.0
    out['surfaces'] = surfaces
    with open(outfn, 'w') as outfile:
        ordered_dump(out, outfile)


if __name__ == '__main__':
    for i, f in zip(range(6), 'ugrizy'):
        translate("optics_{}.txt".format(i), "LSST_{}.yaml".format(f), f)
