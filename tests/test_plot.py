import os

import numpy as np
from matplotlib.figure import Figure

import batoid
from test_helpers import timer


@timer
def test_zernikePyramid():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")

    thxs = np.linspace(-1.7, 1.7, 9)
    thys = np.linspace(-1.7, 1.7, 9)

    vmin = -0.3
    vmax = 0.3
    zs = []
    thxplot = []
    thyplot = []
    for ix, thx in enumerate(thxs):
        for iy, thy in enumerate(thys):
            if np.hypot(thx, thy) > 1.75:
                continue
            zs.append(batoid.zernike(
                telescope, np.deg2rad(thx), np.deg2rad(thy), 500e-9,
                jmax=15, eps=0.61, nx=16
            ))
            thxplot.append(thx)
            thyplot.append(thy)
    zs = np.array(zs).T
    thxplot = np.array(thxplot)
    thyplot = np.array(thyplot)

    zranges = [slice(4, 16), slice(4, 11)]

    for zrange, vdim in zip(zranges, [True, False]):
        fig = Figure(figsize=(13, 8))
        batoid.plotUtils.zernikePyramid(
            thxplot, thyplot, zs[zrange],
            vmin=vmin, vmax=vmax, s=100,
            fig=fig
        )
        fig.savefig("pyramid.png")

        # cleanup
        try:
            os.remove("pyramid.png")
        except OSError:
            pass


if __name__ == '__main__':
    test_zernikePyramid()