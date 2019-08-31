import batoid
import time
import numpy as np
import pytest
from test_helpers import timer

# Use matplotlib with a non-interactive backend.
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@timer
def initialize(ngrid=25, theta_x=1.):
    telescope = batoid.Optic.fromYaml("DESI.yaml")
    dirCos = batoid.utils.gnomonicToDirCos(np.deg2rad(theta_x), 0.)
    rays = batoid.rayGrid(
        telescope.dist, telescope.pupilSize, dirCos[0], dirCos[1], -dirCos[2],
        ngrid, 500e-9, 1.0, telescope.inMedium
    )
    return telescope, telescope.traceFull(rays)

@timer
def draw2dtelescope(ax, telescope):
    telescope.draw2d(ax, c='k')

@timer
def draw2drays(ax, rays, start=None, stop=None):
    batoid.drawTrace2d(ax, rays, start, stop, c='b', lw=1)

def test_draw2d(ngrid=25):
    telescope, rays = initialize(ngrid)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    draw2dtelescope(ax, telescope)
    draw2drays(ax, rays)

def test_draw2d_only():
    telescope, rays = initialize(3)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    telescope.draw2d(ax, only=batoid.optic.Lens, fc='c', alpha=0.2)
    telescope.draw2d(ax, only=batoid.optic.Detector, c='b', lw=1)

@timer
def draw3dtelescope(ax, telescope):
    telescope.draw3d(ax, c='k')

@timer
def draw3drays(ax, rays, start=None, stop=None):
    batoid.drawTrace3d(ax, rays, start, stop, c='b', lw=1)

def test_draw3d(ngrid=25):
    telescope, rays = initialize(ngrid)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    draw3dtelescope(ax, telescope)
    draw3drays(ax, rays)

if __name__ == '__main__':
    test_draw3d()
    plt.savefig('draw3d.png')
    test_draw2d()
    plt.savefig('draw2d.png')
    test_draw2d_only()
