import batoid
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


telescope = batoid.Optic.fromYaml("LSST_g.yaml")
factor = 0.2/1e-5/3600

# telescope = batoid.Optic.fromYaml("HSC.yaml")
# factor = 0.168/1.5e-5/3600

# telescope = batoid.Optic.fromYaml("DECam.yaml")
# factor = 0.27/1.5e-5/3600

# Make refractive interfaces partially reflective
for surface in telescope.itemDict.values():
    if isinstance(surface, batoid.RefractiveInterface):
        surface.forwardCoating = batoid.SimpleCoating(0.02, 0.98)
        surface.reverseCoating = batoid.SimpleCoating(0.02, 0.98)
    if isinstance(surface, batoid.Detector):
        surface.forwardCoating = batoid.SimpleCoating(0.02, 0.98)

angles = np.linspace(0.0, 1.0, 21, endpoint=True)
angles = [0.3]
for angle in angles:
    dirCos = batoid.utils.gnomonicToDirCos(0.0, np.deg2rad(angle))

    rays = batoid.circularGrid(
        telescope.backDist,
        telescope.pupilSize/2,
        telescope.pupilSize*telescope.pupilObscuration/2,
        dirCos[0], dirCos[1], dirCos[2],
        300, 900, 500e-9, 1.0, telescope.inMedium
    )

    rForward, rReverse = telescope.traceSplit(rays, minFlux=1e-4, _verbose=True)

    print("# input rays          = {}".format(len(rays)))
    print("# forward output rays = {}".format(len(rForward)))
    print("# reverse output rays = {}".format(len(rReverse)))
    print("input flux          = {}".format(np.sum(rays.flux)))
    print("forward output flux = {}".format(np.sum(rForward.flux)))
    print("reverse output flux = {}".format(np.sum(rReverse.flux)))
    print("destroyed flux      = {}".format(
        np.sum(rays.flux) - np.sum(rForward.flux) - np.sum(rReverse.flux))
    )


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.hexbin(
        rForward.x*factor, rForward.y*factor, rForward.flux,
        reduce_C_function=np.sum, vmax=4e-2, gridsize=1000, bins='log'
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_position([0,0,1,1])
    ax.set_facecolor('black')
    plt.show()
