import numpy as np
from urllib.request import urlopen
from astroquery.simbad import Simbad
from astropy.io import fits


url = "https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/western/constellationship.fab"


def ten(s):
    ss = s.split()
    h = float(ss[0])
    sign = +1
    if h < 0:
        sign = -1
        h *= -1
    m = float(ss[1])
    s = float(ss[2])
    return sign * (h + m/60 + s/3600)


with urlopen(url) as f:
    lines = f.readlines()

HIPset = set()  # Use set; no need to query same star twice
for line in lines:
    HIPset.update([int(s) for s in line.split()[2:]])

HIPlist = list(HIPset)
Simbad.add_votable_fields('typed_id')
table = Simbad.query_objects(
    [f"HIP {s}" for s in HIPlist]
)
table['HIPID'] = HIPlist

xs = []
ys = []
zs = []
for line in lines:
    # Separate line segment paths with nans so paths are separated when plotting
    xs.append(np.nan)
    ys.append(np.nan)
    zs.append(np.nan)
    prev_second = -1
    endpoints = iter(line.split()[2:])
    for first in endpoints:
        second = next(endpoints)  # parse 2 HIP ids at a time
        first = int(first)
        second = int(second)

        secondrow = table[np.nonzero(table['HIPID'] == int(second))]
        ra1 = np.deg2rad(15*ten(secondrow['RA'][0]))
        dec1 = np.deg2rad(ten(secondrow['DEC'][0]))
        x1 = np.cos(ra1)*np.cos(dec1)
        y1 = np.sin(ra1)*np.cos(dec1)
        z1 = np.sin(dec1)

        if first == prev_second:  # continuing path
            # just append new second
            xs.append(x1)
            ys.append(y1)
            zs.append(z1)
        else:  # Start a new line segment path
            firstrow = table[np.nonzero(table['HIPID'] == int(first))]
            ra0 = np.deg2rad(15*ten(firstrow['RA'][0]))
            dec0 = np.deg2rad(ten(firstrow['DEC'][0]))
            x0 = np.cos(ra0)*np.cos(dec0)
            y0 = np.sin(ra0)*np.cos(dec0)
            z0 = np.sin(dec0)
            xs.extend([np.nan, x0, x1])
            ys.extend([np.nan, y0, y1])
            zs.extend([np.nan, z0, z1])
        prev_second = second
xyz = np.array([xs, ys, zs])
fits.writeto("constellations.fits", xyz)
