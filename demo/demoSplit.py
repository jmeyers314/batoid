import batoid
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


# fn = os.path.join(batoid.datadir, "LSST", "LSST_r.yaml")
# factor = 0.2/1e-5/3600

fn = os.path.join(batoid.datadir, "HSC", "HSC.yaml")
factor = 0.168/1.5e-5/3600

# fn = os.path.join(batoid.datadir, "DECam", "DECam.yaml")
# factor = 0.27/1.5e-5/3600

config = yaml.load(open(fn))
telescope = batoid.parse.parse_optic(config['opticalSystem'])

angles = np.linspace(0.0, 1.0, 21, endpoint=True)
angles = [0.3]
for angle in angles:
    dirCos = batoid.utils.gnomicToDirCos(0.0, np.deg2rad(angle))

    rays = batoid.circularGrid(
        telescope.dist,
        telescope.pupilSize/2,
        telescope.pupilSize*telescope.pupilObscuration/2,
        dirCos[0], dirCos[1], -dirCos[2],
        300, 900, 500e-9, 1.0, telescope.inMedium)

    rForward, rReverse, _, _ = telescope.traceSplit(rays, minFlux=1e-4, verbose=True)

    print(len(rays))
    print(np.sum(rays.flux))
    print(len(rForward))
    print(len(rReverse))
    print(np.sum(rForward.flux))
    print(np.sum(rReverse.flux))


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
