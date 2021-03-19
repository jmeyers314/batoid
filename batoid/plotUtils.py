from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


def zernikePyramid(xs, ys, zs, figsize=(13, 8), vmin=-1, vmax=1, vdim=True,
                   s=5, title=None, filename=None, fig=None, **kwargs):
    """Make a multi-zernike plot in a pyramid shape.

    Subplots show individual Zernikes over a range of x and y (presumably a
    field of view).

    Parameters
    ----------
    xs, ys: array of float
        Field angles (or other spatial coordinate over which to plot Zernikes)
    zs: array of float, shape (jmax, xymax)
        Zernike values.  First index labels the particular Zernike coefficient,
        second index labels spatial coordinate.  First index implicitly starts
        at j=4 defocus.
    """

    import warnings
    import galsim
    jmax = zs.shape[0]+3
    nmax, _ = galsim.zernike.noll_to_zern(jmax)

    nrow = nmax - 1
    ncol = nrow + 2
    gridspec = GridSpec(nrow, ncol)

    def shift(pos, amt):
        return [pos.x0+amt, pos.y0, pos.width, pos.height]

    def shiftAxes(axes, amt):
        for ax in axes:
            ax.set_position(shift(ax.get_position(), amt))

    if fig is None:
        fig = Figure(figsize=figsize, **kwargs)
    axes = {}
    shiftLeft = []
    shiftRight = []
    for j in range(4, jmax+1):
        n, m = galsim.zernike.noll_to_zern(j)
        if n%2 == 0:
            row, col = n-2, m//2 + ncol//2
        else:
            row, col = n-2, (m-1)//2 + ncol//2
        subplotspec = gridspec.new_subplotspec((row, col))
        axes[j] = fig.add_subplot(subplotspec)
        axes[j].set_aspect('equal')
        if nrow%2==0 and n%2==0:
            shiftLeft.append(axes[j])
        if nrow%2==1 and n%2==1:
            shiftRight.append(axes[j])

    cbar = {}
    for j, ax in axes.items():
        n, _ = galsim.zernike.noll_to_zern(j)
        ax.set_title("Z{}".format(j))
        if vdim:
            _vmin = vmin/n
            _vmax = vmax/n
        else:
            _vmin = vmin
            _vmax = vmax
        scat = ax.scatter(
            xs, ys, c=zs[j-4], s=s, linewidths=0.5, cmap='Spectral_r',
            rasterized=True, vmin=_vmin, vmax=_vmax
        )
        cbar[j] = fig.colorbar(scat, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, x=0.1)

    # Mistakenly raises MatplotlibDeprecationWarning.
    # See https://github.com/matplotlib/matplotlib/issues/19486
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
    amt = 0.5*(axes[4].get_position().x0 - axes[5].get_position().x0)
    shiftAxes(shiftLeft, -amt)
    shiftAxes(shiftRight, amt)

    shiftAxes([cbar[j].ax for j in cbar.keys() if axes[j] in shiftLeft], -amt)
    shiftAxes([cbar[j].ax for j in cbar.keys() if axes[j] in shiftRight], amt)

    if filename:
        fig.savefig(filename)

    return fig
