import numpy as np


class State:
    """Describes a specific configuration (i.e., a microstate) of the system being simulated.

    States may be created by providing a coordinate tuple, a reference to a
    file containing coordinate data, or a unique identifier. At least one
    parameter must be provided.

    Parameters
    ----------
    coord : 1-D array-like, optional
        Coordinate tuple.
    file : str, optional
        Reference (e.g., an absolute path or URI) to a file containing coordinate data.
    id : str or int, optional
        Unique identifier.

    Attributes
    ----------
    coord : numpy.ndarray or None
    file : str or None
    id : str, int, or None

    Examples
    --------

    2-D system:

    >>> import westpa
    >>> westpa.State(coord=[0.0, 0.0])
    State(coord=[0.0, 0.0])

    Coordinates stored in a local file:

    >>> westpa.State(file='/path/to/file.xyz')
    State(file='/path/to/file.xyz')

    Discrete system with integer state IDs:

    >>> westpa.State(id=0)
    State(id=0)

    """

    def __init__(self, *, coord=None, file=None, id=None):
        if coord is None and file is None and id is None:
            raise ValueError("'coord', 'file', or 'id' must be provided")

        if coord is not None:
            coord = np.asarray(coord)
            if not coord.ndim == 1:
                raise ValueError("'coord' must be a 1-D array")

        if id is not None and not isinstance(id, (str, int)):
            raise TypeError("'id' must be a string or integer")

        self._coord = coord
        self._file = str(file) if file is not None else None
        self._id = id

    @property
    def id(self):
        """State ID."""
        return self._id

    @property
    def coord(self):
        """Coordinate tuple."""
        return self._coord.copy() if self._coord is not None else None

    @property
    def file(self):
        """File containing coordinate data."""
        return self._file

    def __repr__(self):
        args = ', '.join(f'{k}={v!r}' for k, v in self.to_dict().items())
        return type(self).__name__ + '(' + args + ')'

    def to_dict(self):
        d = {}
        for name in ('coord', 'file', 'id'):
            if (value := getattr(self, name)) is not None:
                if name == 'coord':
                    value = value.tolist()
                d[name] = value
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
