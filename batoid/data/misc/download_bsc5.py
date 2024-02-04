import gzip
from urllib.request import urlopen

import numpy as np
from astropy.table import Table


url = "http://tdc-www.harvard.edu/catalogs/bsc5.dat.gz"

response = urlopen(url)
content = gzip.decompress(response.read())
decomp_req = content.splitlines()
ra_list = []
dec_list = []
mag_list = []
for line in decomp_req:
    try:
        line = line.decode('utf-8')
        rah = float(line[75:77])
        ram = float(line[77:79])
        ras = float(line[79:83])
        decsign = float(line[83]+"1")
        decd = float(line[84:86])
        decm = float(line[86:88])
        decs = float(line[88:90])
    except:
        continue
    ra_list.append(np.deg2rad(15*(rah+ram/60+ras/3600)))
    dec_list.append(np.deg2rad(decsign*(decd+decm/60+decs/3600)))
    mag_list.append(float(line[102:107]))

table = Table()
table['ra'] = ra_list
table['dec'] = dec_list
table['mag'] = mag_list
table.write("BSC5.fits")
