import h5py
import numpy as np


class State:
    """Describes a specific configuration (i.e., a microstate) of the model being simulated.

    At least one of `coord` or `file` must be provided.

    Parameters
    ----------
    coord : 1-D array_like, optional
        Coordinates of the state.
    file : str, optional
        Identifier (e.g., an absolute path or URI) of a file containing
        coordinate data for the state.

    Attributes
    ----------
    coord : numpy.ndarray
    file : str

    Examples
    --------

    >>> import westpa
    >>> import numpy as np

    Coordinates stored in memory:

    >>> westpa.State(coord=[0., 0.])
    State(coord=array([0., 0.]))
    >>> westpa.State(coord=np.array([0., 0.], dtype=np.float32))
    State(coord=array([0., 0.], dtype=float32))

    Coordinates stored in an external file:

    >>> westpa.State(file='/path/to/file.xyz'))
    State(file='/path/to/file.xyz')

    """

    def __init__(self, *, coord=None, file=None):
        if coord is None and file is None:
            raise ValueError("at least one of 'coord' or 'file' must be provided")

        if coord is not None:
            coord = np.asarray(coord)
            if not np.issubdtype(coord.dtype, np.number):
                raise TypeError("scalar type of 'coord' must be numeric")
            if coord.ndim != 1:
                raise ValueError("'coord' must be a 1-D array")

        self._coord = coord
        self._file = str(file) if file is not None else None

    @property
    def coord(self):
        """Coordinate vector."""
        return self._coord.copy() if self._coord is not None else None

    @property
    def file(self):
        """File containing coordinate data."""
        return self._file

    def __repr__(self):
        kwargs = {}
        if self._coord is not None:
            kwargs['coord'] = self._coord
        if self._file is not None:
            kwargs['file'] = self._file
        params = ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
        return type(self).__name__ + '(' + params + ')'

    def to_numpy(self):
        fields = []
        values = []
        if (coord := self.coord) is not None:
            fields.append(('coord', coord.dtype, coord.shape))
            values.append(coord)
        if (file := self.file) is not None:
            fields.append(('file', h5py.special_dtype(vlen=str)))
            values.append(file)
        return np.array(values, dtype=np.dtype(fields))

    @classmethod
    def from_numpy(cls, array):
        return cls(
            coord=array['coord'] if 'coord' in array.dtype.names else None,
            file=array['file'].decode('utf-8') if 'file' in array.dtype.names else None,
        )
