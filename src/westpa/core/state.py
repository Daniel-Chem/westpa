import numpy as np


class State:
    """Describes a specific configuration (i.e., a microstate) of the model being simulated.

    The propagator determines the required shape for `coord` or format for `file`.

    Parameters
    ----------
    coord : 1-D array_like, optional
        Coordinates of the state.
    file : str, optional
        Identifier (e.g., an absolute path or URI) of a file containing
        coordinate data for the state.
    label : str, optional
        Descriptive label for the state.

    Attributes
    ----------
    coord : numpy.ndarray
    file : str
    label : str

    Examples
    --------

    >>> import westpa
    >>> westpa.State(coord=[0.0, 0.0])
    State(coord=[0.0, 0.0])

    If coordinates are stored on disk, use the `file` parameter:

    >>> westpa.State(file='/path/to/file.xyz')
    State(file='/path/to/file.xyz')

    """

    def __init__(self, *, coord=None, file=None, label=None):
        if coord is None and file is None:
            raise ValueError("either 'coord' or 'file' must be provided")

        if coord is not None:
            coord = np.asarray(coord)
            if not np.issubdtype(coord.dtype, np.number):
                raise TypeError("scalar type of 'coord' must be numeric")
            if coord.ndim != 1:
                raise ValueError("'coord' must be a 1-D array")

        self._coord = coord
        self._file = str(file) if file is not None else None
        self._label = label or ''

    @property
    def coord(self):
        """Coordinate tuple."""
        return self._coord.copy() if self._coord is not None else None

    @property
    def file(self):
        """File containing coordinate data."""
        return self._file

    @property
    def label(self):
        """State label."""
        return self._label

    def __repr__(self):
        kwargs = {}
        if self._coord is not None:
            kwargs['coord'] = self._coord.tolist()
        if self._file is not None:
            kwargs['file'] = self._file
        if self._label:
            kwargs['label'] = self._label
        params = ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
        return type(self).__name__ + '(' + params + ')'
