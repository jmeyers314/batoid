# Compare ray by ray tracing to Zemax
import os
import numpy as np
import jtrace

directory = os.path.dirname(__file__)

def test_HSC_trace():
    telescope = jtrace.Telescope.makeFromYAML(os.path.join(jtrace.datadir, "hsc", "HSC.yaml"))
    # Zemax has a number of virtual surfaces that we don't trace in jtrace.  This list picks out
    # the "real" surfaces.
    HSC_surfaces = [3, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 24, 25, 28, 29, 31]

    for fn in ["rt1.txt", "rt2.txt", "rt3.txt"]:
        filename = os.path.join(directory, "testdata", fn)
        with open(filename) as f:
            arr = np.loadtxt(f, skiprows=22, usecols=list(range(0, 12)))
        arr0 = arr[0]
        ray = jtrace.Ray(arr0[1]/1000, arr0[2]/1000, 16.0, arr0[4], arr0[5], -arr0[6], t=0, w=750e-9)
        tf = telescope.traceFull(ray)
        for i in range(len(tf)):
            s = tf[i]['outray']
            v = s.v.UnitVec3()

            jt_isec = np.array([s.x0, s.y0, s.z0-16.0])
            zx_isec = arr[HSC_surfaces[i]-1][1:4]/1000
            np.testing.assert_allclose(jt_isec, zx_isec, rtol=0, atol=1e-9) # nanometer agreement

            jt_angle = np.array([v.x, v.y, v.z])
            zx_angle = arr[HSC_surfaces[i]-1][4:7]
            # direction cosines agree to 1e-9
            np.testing.assert_allclose(jt_angle, zx_angle, rtol=0, atol=1e-9)

            # Would be nice to check optical path too...

if __name__ == '__main__':
    test_HSC_trace()
