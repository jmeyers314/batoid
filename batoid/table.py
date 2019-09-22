from . import _batoid

class Table:
    """A one-dimensional interpolating lookup table.

    Parameters
    ----------
    args : ndarray
        abscissae of table function.
    vals : ndarray
        ordinates of table function.
    interp : {'linear', 'ceil', 'floor', 'nearest'}
        Type of interpolant to use.
    """
    def __init__(self, args, vals, interp='linear'):
        self.args = args
        self.vals = vals
        self.interp = interp
        if self.interp == 'linear':
            self._interp = _batoid.CPPTable.Interpolant.linear
        elif self.interp == 'floor':
            self._interp = _batoid.CPPTable.Interpolant.floor
        elif self.interp == 'ceil':
            self._interp = _batoid.CPPTable.Interpolant.ceil
        elif self.interp == 'nearest':
            self._interp = _batoid.CPPTable.Interpolant.nearest
        else:
            raise ValueError("Unknown interpolant {}".format(interp))
        self._table = _batoid.CPPTable(args, vals, self._interp)

    def __call__(self, x):
        """Interpolate a value from lookup table.

        Parameters
        ----------
        x : array_like, shape (n,)
            Abscissae to interpolate

        Returns
        -------
        array_like, shape(n,)
            The interpolated ordinates.
        """
        return self._table(x)

    def __eq__(self, rhs):
        if not isinstance(rhs, Table): return False
        return self._table == rhs._table

    def __ne__(self, rhs):
        return not (self == rhs)

    def __hash__(self):
        return hash(("Table", self._table))

    def __repr__(self):
        return "Table({!r}, {!r}, {!r})".format(
            self.args, self.vals, self.interp
        )
