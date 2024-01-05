import galsim
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


# From https://joseph-long.com/writing/colorbars/
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def zernikePyramid(
    xs, ys, zs, jmin=4, figsize=(13, 8), vmin=-1, vmax=1, vdim=True,
    s=5, title=None, callback=None, filename=None, fig=None, cmap='seismic',
    **kwargs
):
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
        at j=jmin (defocus by default).
    jmin: int, optional
        Minimum Zernike to plot.  Default 4 (defocus).
    figsize: tuple of float, optional
        Figure size in inches.  Default (13, 8).
    vmin, vmax: float, optional
        Color scale limits.  Default (-1, 1).
    vdim: bool, optional
        If True, scale vmin and vmax by the Zernike radial order.  Default True.
    s: float, optional
        Marker size.  Default 5.
    callback: callable, optional
        A callable to execute just before adjusting axis locations.  Useful for
        setting suptitle, for example.  Takes two keyword arguments, fig for
        the Figure, and axes for a zernike-indexed dict of plot Axes.
        Default: None
    filename: str, optional
        If provided, save figure to this filename.  Default None.
    fig: matplotlib Figure, optional
        If provided, use this figure.  Default None.
    cmap: str, optional
        Colormap name.  Default 'seismic'.
    **kwargs:
        Additional keyword arguments passed to matplotlib Figure constructor.

    Returns
    -------
    fig: matplotlib Figure
        The figure.
    """
    jmax = zs.shape[0] + jmin - 1
    nmax, _ = galsim.zernike.noll_to_zern(jmax)
    nmin, _ = galsim.zernike.noll_to_zern(jmin)

    nrow = nmax - nmin + 1
    ncol = nmax + 1
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
    for j in range(jmin, jmax+1):
        n, m = galsim.zernike.noll_to_zern(j)
        if n%2 == 0:
            row, col = n-nmin, m//2 + ncol//2
        else:
            row, col = n-nmin, (m-1)//2 + ncol//2
        subplotspec = gridspec.new_subplotspec((row, col))
        axes[j] = fig.add_subplot(subplotspec)
        axes[j].set_aspect('equal')
        if nmax%2==0 and (nmax-n)%2==1:
            shiftRight.append(axes[j])
        if nmax%2==1 and (nmax-n)%2==1:
            shiftLeft.append(axes[j])

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
            xs, ys, c=zs[j-jmin], s=s, linewidths=0.5, cmap=cmap,
            rasterized=True, vmin=_vmin, vmax=_vmax
        )
        cbar[j] = colorbar(scat)
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        raise DeprecationWarning(
            "title argument is deprecated.  Use a callback."
        )
        fig.suptitle(title, x=0.1)

    if callback is not None:
        callback(fig=fig, axes=axes)

    fig.tight_layout()
    # Assume we always have Z4 and Z5?
    amt = 0.5*(axes[4].get_position().x0 - axes[5].get_position().x0)
    shiftAxes(shiftLeft, -amt)
    shiftAxes(shiftRight, amt)

    if filename:
        fig.savefig(filename)

    return fig
