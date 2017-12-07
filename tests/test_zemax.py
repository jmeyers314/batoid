# Compare ray by ray tracing to Zemax
import os
import numpy as np
import batoid
from test_helpers import timer
import yaml

directory = os.path.dirname(__file__)

@timer
def test_HSC_trace():
    fn = os.path.join(batoid.datadir, "hsc", "HSC3.yaml")
    config = yaml.load(open(fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])

    # Zemax has a number of virtual surfaces that we don't trace in batoid.  Also, the HSC3.yaml
    # above includes Baffle surfaces not in Zemax.  The following lists select out the surfaces in
    # common to both models.
    HSC_surfaces = [3, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 24, 25, 28, 29, 31]
    surface_names = ['PM', 'G1_entrance', 'G1_exit', 'G2_entrance', 'G2_exit',
                     'ADC1_entrance', 'ADC1_exit', 'ADC2_entrance', 'ADC2_exit',
                     'G3_entrance', 'G3_exit', 'G4_entrance', 'G4_exit',
                     'G5_entrance', 'G5_exit', 'F_entrance', 'F_exit',
                     'W_entrance', 'W_exit', 'D']

    for fn in ["rt1.txt", "rt2.txt", "rt3.txt"]:
        filename = os.path.join(directory, "testdata", fn)
        with open(filename) as f:
            arr = np.loadtxt(f, skiprows=22, usecols=list(range(0, 12)))
        arr0 = arr[0]
        ray = batoid.Ray(arr0[1]/1000, arr0[2]/1000, 16.0, arr0[4], arr0[5], -arr0[6], t=0, w=750e-9)
        tf = telescope.traceFull(ray)

        i = 0
        for surface in tf:
            if surface['name'] != surface_names[i]:
                continue

            s = surface['out']
            v = s.v.UnitVec3()

            transform = batoid.CoordTransform(surface['outCoordSys'], batoid.CoordSys())
            s = transform.applyForward(s)
            jt_isec = np.array([s.x0, s.y0, s.z0-16.0])
            zx_isec = arr[HSC_surfaces[i]-1][1:4]/1000
            np.testing.assert_allclose(jt_isec, zx_isec, rtol=0, atol=1e-9) # nanometer agreement

            jt_angle = np.array([v.x, v.y, v.z])
            zx_angle = arr[HSC_surfaces[i]-1][4:7]
            # direction cosines agree to 1e-9
            np.testing.assert_allclose(jt_angle, zx_angle, rtol=0, atol=1e-9)

            i += 1

if __name__ == '__main__':
    test_HSC_trace()
